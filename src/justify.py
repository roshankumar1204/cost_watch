"""Step 3c: for each flagged week, retrieve candidate notes and verify them.

The verdict is decided here by explicit checks, not by the LLM. That is what keeps
verdicts and cited notes identical across runs and stops a fluent but wrong
explanation from becoming a "justified" row.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import config
from src.notes import ALL_ROUTES, NoteFacts
from src.retrieval import NoteIndex, build_query

CANDIDATE_RADIUS = timedelta(weeks=config.HISTORY_WEEKS)


@dataclass
class Decision:
    verdict: str                          # "justified" | "unexplained"
    note: Optional[NoteFacts] = None
    rejected: list = field(default_factory=list)   # [(note_id, why_not)]


def effective_window(f: NoteFacts) -> tuple[date, date]:
    end = f.window_end
    if end is None:  # open-ended: only until the 8-week baseline has absorbed the step
        end = f.window_start + timedelta(weeks=config.OPEN_ENDED_WINDOW_WEEKS) - timedelta(days=1)
    return f.window_start, end


def _overlaps(f: NoteFacts, ws: date, we: date) -> bool:
    s, e = effective_window(f)
    return s <= we and e >= ws


def _is_candidate(f: NoteFacts, route: str, ws: date, we: date) -> bool:
    """Metadata filter: right route (or network-wide) and close enough in time to matter."""
    if not f.applies_to_route(route):
        return False
    s, e = effective_window(f)
    return s - CANDIDATE_RADIUS <= we and e + CANDIDATE_RADIUS >= ws


def _why_not(f: NoteFacts, row, ws: date, we: date) -> Optional[str]:
    """None if the note genuinely explains this week's rise, else the reason it does not.

    Checks run from most to least fundamental: what the note says, who it covers,
    how big an effect it claims, and only then when it applies. The first reason
    found is reported, so the stated reason doesn't depend on window edge cases."""
    if f.effect != "increase":
        return (f"{f.note_id} ({f.note_date}) is not a cost-increase reason; "
                f"it says: \"{f.text.split('. ')[0].rstrip('.')}\"")
    if f.applies_to == ALL_ROUTES and row.peer_trigger:
        return (f"{f.note_id} ({f.note_date}) is network-wide, so it cannot explain this route costing "
                f"{row.vs_peer_pct:+.1%} more than similar routes in the same week")
    if f.magnitude_max is not None and row.vs_own_pct > f.magnitude_max + config.MAGNITUDE_TOLERANCE:
        return (f"{f.note_id} ({f.note_date}) accounts for up to about {f.magnitude_max:.0%}, "
                f"but the rise is {row.vs_own_pct:+.1%}")
    if not _overlaps(f, ws, we):
        s, e = effective_window(f)
        return f"{f.note_id} applies {s} to {e}, which does not cover the week of {ws}"
    return None


def decide(row, index: NoteIndex) -> Decision:
    ws = row.week_of.date()
    we = ws + timedelta(days=6)
    query = build_query(row.route, ws)
    # Verify every candidate that passes the filter (there are only a few), in similarity order,
    # so a relevant note can never be missed just because it ranked 4th.
    candidates = index.search(query, keep=lambda f: _is_candidate(f, row.route, ws, we),
                              k=len(index.facts))

    rejected = []
    for f, _score in candidates:
        why = _why_not(f, row, ws, we)
        if why is None:
            return Decision("justified", note=f, rejected=rejected)
        rejected.append((f.note_id, why))
    return Decision("unexplained", rejected=rejected)
