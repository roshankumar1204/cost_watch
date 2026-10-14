"""Wires the steps together and writes the outputs."""
import json
from pathlib import Path

import pandas as pd

import config
from src.explain import write_reason
from src.flags import apply_flag_rule
from src.justify import decide
from src.llm import LLM
from src.metrics import build_weekly_table
from src.notes import facts_table, load_note_facts
from src.retrieval import NoteIndex


def _pct_text(value, suffix: str, prior_weeks=None) -> str:
    if pd.isna(value):
        return f"n/a (no prior weeks) {suffix}"
    return f"{value * 100:+.1f}% {suffix}"


def run(out_dir: Path = config.OUTPUT_DIR, use_llm: bool = True, verbose: bool = True) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(enabled=use_llm)
    log = print if verbose else (lambda *a, **k: None)
    log(f"LLM: {'Groq ' + config.LLM_MODEL if llm.available else 'off (template explanations)'}")

    weekly = apply_flag_rule(build_weekly_table())
    facts = load_note_facts(llm)
    index = NoteIndex(facts)
    log(f"{len(weekly)} route-weeks, {int(weekly.is_anomaly.sum())} flagged by the rule, {len(facts)} notes")

    rows, details = [], []
    for r in weekly[weekly.is_anomaly].itertuples():
        d = decide(r, index)
        reason, source = write_reason(r, d, llm)
        rows.append({
            "route": r.route,
            "week_of": r.week_of.strftime("%Y-%m-%d"),
            "cost_per_tonne_km": f"{r.cost_per_tonne_km:.2f}",  # keep "4.10", not "4.1"
            "vs_own_history": _pct_text(r.vs_own_pct, "vs this route's past average"),
            "vs_similar_routes": _pct_text(r.vs_peer_pct, "vs similar-length routes this week"),
            "flagged": "No (justified)" if d.verdict == "justified" else "Yes",
            "matched_note_id": d.note.note_id if d.note else "",
            "reason": reason,
        })
        details.append({
            "route": r.route, "week_of": r.week_of.strftime("%Y-%m-%d"),
            "verdict": d.verdict, "matched_note_id": d.note.note_id if d.note else "",
            "own_trigger": bool(r.own_trigger), "peer_trigger": bool(r.peer_trigger),
            "prior_weeks_used": int(r.prior_weeks_used), "shipments": int(r.shipments),
            "rejected_notes": " | ".join(f"{nid}: {why}" for nid, why in d.rejected),
            "reason_source": source,
        })

    result = pd.DataFrame(rows, columns=config.OUTPUT_COLUMNS)
    result.to_csv(out_dir / "flagged_routes.csv", index=False, encoding="utf-8")
    pd.DataFrame(details).to_csv(out_dir / "flag_details.csv", index=False, encoding="utf-8")

    metrics = weekly.assign(week_of=weekly.week_of.dt.strftime("%Y-%m-%d"))
    metrics.round(6).to_csv(out_dir / "weekly_metrics.csv", index=False, encoding="utf-8")
    facts_table(facts).to_csv(out_dir / "note_audit.csv", index=False, encoding="utf-8")

    summary = llm.log.summary()
    (out_dir / "token_log.json").write_text(json.dumps(summary, indent=2))
    n_just = int((result.flagged == "No (justified)").sum())
    log(f"wrote {len(result)} rows ({len(result) - n_just} flagged, {n_just} justified) to {out_dir}")
    log(f"LLM calls: {summary['llm_calls']}, tokens in/out: {summary['input_tokens']}/"
        f"{summary['output_tokens']}, est. cost ${summary['estimated_cost_usd_at_published_rate']}")
    return result
