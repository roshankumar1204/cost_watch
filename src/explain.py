"""Step 4: write the plain-English `reason` for each row.

The LLM only gets to WORD a verdict that has already been decided. It sees the
numbers, the verdict, the matched note (if any) and why other notes were rejected.
Its output is then checked: a justified reason must name the matched note id and
must not mention any other note id. An unexplained reason must not name a note as
the cause. If a check fails, we fall back to a deterministic template.
"""
import re

import config

# LLMs like typographic hyphens/quotes (e.g. "Ahmedabad‑Mumbai"); normalise them so
# route names in the reason text match the route column exactly.
ASCII_PUNCT = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
                             "‘": "'", "’": "'", "“": '"', "”": '"', " ": " "})

SYSTEM = """You write one or two short, plain-English sentences explaining a freight cost flag.
Use ONLY the facts given. Do not invent causes, dates, or numbers.
- If verdict is JUSTIFIED: start with "Matches note <id> dated <date>:", summarise what the note says,
  include any date range the note gives (e.g. "Feb 24 to Mar 8"), and say the cost rise has a clear explanation.
- If verdict is UNEXPLAINED: say no note explains the rise. If rejected notes are given, you may mention
  the closest one and why it does not apply. End by saying it is flagged for human review.
Return only the sentences."""


def _numbers(row) -> str:
    own = "n/a" if row.vs_own_pct != row.vs_own_pct else f"{row.vs_own_pct:+.1%}"
    return (f"route {row.route}, week of {row.week_of:%Y-%m-%d}: cost per tonne-km "
            f"{row.cost_per_tonne_km:.2f} INR, {own} vs its own past average, "
            f"{row.vs_peer_pct:+.1%} vs similar-length routes this week")


def template_reason(row, decision) -> str:
    if decision.verdict == "justified":
        n = decision.note
        return (f"Matches note {n.note_id} dated {n.note_date}: {n.text} "
                f"The cost rise has a clear explanation.")
    if decision.rejected:
        nid, why = decision.rejected[0]
        return (f"No note explains this rise. The closest candidate was considered and rejected: {why}. "
                f"Cost rise looks unexplained; flagged for human review.")
    return ("No matching note found for this route or date range. "
            "Cost rise looks unexplained and worth a human review.")


def _passes_guard(text: str, decision) -> bool:
    if not text or len(text) > 600:
        return False
    ids = set(re.findall(r"\bN\d{3}\b", text))
    if decision.verdict == "justified":
        return decision.note.note_id in ids and ids == {decision.note.note_id}
    allowed = {nid for nid, _ in decision.rejected}
    return ids <= allowed and not re.search(r"\bmatches note\b", text, flags=re.I)


def write_reason(row, decision, llm) -> tuple[str, str]:
    """Return (reason, source) where source is 'llm' or 'template'."""
    if not llm.available:
        return template_reason(row, decision), "template"

    parts = [f"Facts: {_numbers(row)}.", f"Verdict: {decision.verdict.upper()}."]
    if decision.verdict == "justified":
        n = decision.note
        parts.append(f"Matched note {n.note_id} (dated {n.note_date}, applies to {n.applies_to}): {n.text}")
    for nid, why in decision.rejected[: config.RETRIEVAL_TOP_K]:
        parts.append(f"Rejected note: {why}.")
    text = llm.chat(SYSTEM, "\n".join(parts), purpose="write_reason")

    if text:
        text = " ".join(text.translate(ASCII_PUNCT).split())  # plain ASCII hyphens/quotes in the CSV
    if text and _passes_guard(text, decision):
        return text, "llm"
    return template_reason(row, decision), "template"
