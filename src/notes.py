"""Step 3a: turn each free-text note into structured facts we can check against a flag.

Two independent readers look at every note:
  * a rule reader (regex and keywords, fully deterministic)
  * an LLM reader (Groq, temperature 0, JSON output, cached on disk)
A note counts as a cost-increase reason only when BOTH readers say so, and its
active window is the overlap of the two readings. Disagreements are written to
output/note_audit.csv so a human can see them. This makes the LLM a second
opinion that can only make the system more cautious, never less.
"""
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

import config

ALL_ROUTES = "All Routes"

# Phrases that mean "this note is NOT a reason for a cost rise". Checked first: negation wins.
NO_EFFECT_PATTERNS = [
    r"not (significantly )?affected", r"were not", r"without a rate change", r"absorbed",
    r"not part of this dataset", r"no (major|significant) disruptions?", r"remained (normal|stable)",
    r"returned to normal",
]
DECREASE_PATTERNS = [r"\bimproved\b", r"\bcheaper\b", r"\bfell\b", r"\bdecreas"]
INCREASE_PATTERNS = [r"\bhigher\b", r"surcharge", r"\brose\b", r"pushing up", r"\bincreas", r"detours?"]

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


@dataclass
class NoteFacts:
    note_id: str
    note_date: date
    applies_to: str
    text: str
    effect: str                     # "increase" | "no_effect" | "decrease"
    window_start: date
    window_end: Optional[date]      # None = open-ended ("starting this week")
    magnitude_max: Optional[float]  # e.g. 0.07 for "roughly 5-7%"
    rule_effect: str = ""
    llm_effect: str = ""
    audit: str = ""

    def applies_to_route(self, route: str) -> bool:
        return self.applies_to == ALL_ROUTES or self.applies_to == route


# ---------- rule reader ----------

def _week_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _parse_month_day(token: str, ref: date) -> date:
    mon, day = token.split()
    month = MONTHS[mon[:3].lower()]
    year = ref.year + (1 if month < ref.month - 6 else 0)  # handle Dec -> Jan wrap
    return date(year, month, int(day))


def rule_read(note_id, note_date: date, applies_to, text) -> dict:
    low = text.lower()
    if any(re.search(p, low) for p in NO_EFFECT_PATTERNS):
        effect = "no_effect"
    elif any(re.search(p, low) for p in DECREASE_PATTERNS):
        effect = "decrease"
    elif any(re.search(p, low) for p in INCREASE_PATTERNS):
        effect = "increase"
    else:
        effect = "no_effect"

    rng = re.search(r"from (\w{3,9} \d{1,2}) to (\w{3,9} \d{1,2})", text, flags=re.I)
    if rng:
        start = _parse_month_day(rng.group(1), note_date)
        end = _parse_month_day(rng.group(2), note_date)
    elif re.search(r"\bstarting\b", low):
        start, end = note_date, None
    else:  # a dated note with no explicit range describes that week
        start = _week_of(note_date)
        end = start + timedelta(days=6)

    mag = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*%", text)
    if m:
        mag = float(m.group(2)) / 100
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m:
            mag = float(m.group(1)) / 100

    return {"effect": effect, "window_start": start.isoformat(),
            "window_end": end.isoformat() if end else None, "magnitude_max": mag}


# ---------- LLM reader ----------

LLM_SYSTEM = """You read short logistics notes and extract facts. Answer ONLY with JSON:
{"effect": "increase" | "no_effect" | "decrease",
 "window_start": "YYYY-MM-DD",
 "window_end": "YYYY-MM-DD" or null,
 "magnitude_max_pct": number or null,
 "evidence": "<exact quote from the note>"}
Rules:
- effect is "increase" ONLY if the note says freight/transport costs on the stated routes went UP.
  If it says costs were not affected, absorbed, unchanged, routes not in the dataset, normal/stable
  conditions, or conditions improved, use "no_effect" (or "decrease" if costs/conditions got better).
- window_start/window_end: the dates the cost effect applied. Use the note date's year.
  If the note says something started ("starting this week", "from now on") and gives no end date,
  window_end MUST be null: the effect is ongoing, not limited to one week.
  If the note describes a single week with no range, use that week (Monday to Sunday).
- magnitude_max_pct: the upper bound of any percentage stated (e.g. "5-7%" -> 7), else null.
- Do not use outside knowledge."""


def _cache_path():
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", config.LLM_MODEL)
    return config.CACHE_DIR / f"note_readings_{safe}_{config.PROMPT_VERSION}.json"


def _note_key(note_id, note_date, applies_to, text) -> str:
    return hashlib.sha256(f"{note_id}|{note_date}|{applies_to}|{text}".encode()).hexdigest()[:16]


def llm_read_all(notes: pd.DataFrame, llm) -> dict:
    """Return {note_id: reading or None}. Readings are cached so later runs reuse
    exactly the same reading (reproducible) and spend no tokens."""
    path = _cache_path()
    cache = json.loads(path.read_text()) if path.exists() else {}
    out, dirty = {}, False
    for n in notes.itertuples():
        key = _note_key(n.note_id, n.date, n.applies_to, n.note)
        if key not in cache and llm.available:
            user = f"note_id: {n.note_id}\nnote_date: {n.date}\napplies_to: {n.applies_to}\nnote: {n.note}"
            reading = llm.chat(LLM_SYSTEM, user, purpose="read_note", json_mode=True)
            if reading is not None:
                cache[key] = reading
                dirty = True
        out[n.note_id] = cache.get(key)
    if dirty:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return out


# ---------- reconcile ----------

def _d(s):
    return date.fromisoformat(s) if s else None


def reconcile(note_id, note_date, applies_to, text, rule: dict, llm_reading: Optional[dict]) -> NoteFacts:
    effect = rule["effect"]
    start, end = _d(rule["window_start"]), _d(rule["window_end"])
    mag = rule["magnitude_max"]
    audit = []
    llm_effect = ""

    if llm_reading:
        llm_effect = str(llm_reading.get("effect", "")).strip().lower()
        if llm_effect != effect:
            audit.append(f"effect disagreement: rules={effect}, llm={llm_effect}")
            effect = "no_effect" if "increase" in (effect, llm_effect) else effect
        try:
            ls, le = _d(llm_reading.get("window_start")), _d(llm_reading.get("window_end"))
        except (TypeError, ValueError):
            ls, le = None, None
            audit.append("llm window unparseable; using rule window")
        if ls and ls > start:
            audit.append(f"window start narrowed {start} -> {ls}")
            start = ls
        if le and (end is None or le < end):
            audit.append(f"window end narrowed {end} -> {le}")
            end = le
        lm = llm_reading.get("magnitude_max_pct")
        if isinstance(lm, (int, float)):
            lm = lm / 100
            mag = lm if mag is None else min(mag, lm)
        if end is not None and end < start:
            audit.append("readings' windows do not overlap; treating note as no_effect")
            effect = "no_effect"
    else:
        audit.append("no llm reading (no API key or call failed); rules only")

    return NoteFacts(note_id, note_date, applies_to, text, effect, start, end, mag,
                     rule_effect=rule["effect"], llm_effect=llm_effect, audit="; ".join(audit))


def load_note_facts(llm, path=config.NOTES_CSV) -> list[NoteFacts]:
    notes = pd.read_csv(path, dtype=str).fillna("")
    notes["applies_to"] = notes["applies_to"].str.strip()
    llm_readings = llm_read_all(notes, llm)
    facts = []
    for n in notes.itertuples():
        nd = date.fromisoformat(n.date)
        rule = rule_read(n.note_id, nd, n.applies_to, n.note)
        facts.append(reconcile(n.note_id, nd, n.applies_to, n.note, rule, llm_readings.get(n.note_id)))
    return facts


def facts_table(facts: list[NoteFacts]) -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in facts])
