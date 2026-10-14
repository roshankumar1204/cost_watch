"""Score an output folder against hand labels and independent checks.

    python eval/evaluate.py                  # checks output/
    python eval/evaluate.py output/run_2

Checks:
  1. Format: columns and order match sample_output_format_v2.csv exactly.
  2. Sample rows: our numbers, flag and note for the 3 sample rows equal the sample.
  3. Numbers: every output row is recomputed by a separate plain-Python implementation
     (no pandas) and must match to 2 decimals / 0.1 percentage points.
  4. Labeled set: 14 hand-labelled route-weeks (justified / flagged / should-not-flag).
  5. Grounding: every justified row is re-checked against the RAW notes and hand-labelled
     note windows: right route, week inside the window, note really says costs rose.
  6. Note reading: each reader's effect classification vs hand labels.
Exit code is non-zero if any check fails, so it can gate a submission.
"""
import csv
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

failures = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pct(text: str):
    t = text.split("%")[0]
    return None if t.startswith("n/a") else float(t) / 100


# ---------- independent recomputation (plain Python) ----------

def recompute():
    weeks = defaultdict(lambda: [0.0, 0.0])
    rtype = {}
    for r in read_csv(config.SHIPMENTS_CSV):
        d = datetime.strptime(r["shipment_date"], "%Y-%m-%d").date()
        monday = d - timedelta(days=d.weekday())
        route = f"{r['origin'].strip()}-{r['destination'].strip()}"
        rtype[route] = r["route_type"]
        weeks[(route, monday)][0] += float(r["freight_cost_inr"])
        weeks[(route, monday)][1] += float(r["quantity_tonnes"]) * float(r["distance_km"])
    cpt = {k: c / t for k, (c, t) in weeks.items()}
    by_route = defaultdict(list)
    for (route, wk) in sorted(cpt):
        by_route[route].append(wk)
    res = {}
    for route, wks in by_route.items():
        for i, wk in enumerate(wks):
            prior = [cpt[(route, w)] for w in wks[max(0, i - config.HISTORY_WEEKS):i]]
            own = sum(prior) / len(prior) if prior else None
            peers = [cpt[(r2, wk)] for r2 in by_route if r2 != route
                     and rtype[r2] == rtype[route] and (r2, wk) in cpt]
            peer = sum(peers) / len(peers) if peers else None
            c = cpt[(route, wk)]
            res[(route, wk.isoformat())] = (c, c / own - 1 if own else None, c / peer - 1 if peer else None)
    return res


def main(out_dir: Path) -> int:
    out = read_csv(out_dir / "flagged_routes.csv")
    sample = read_csv(config.SAMPLE_OUTPUT_CSV)
    rows = {(r["route"], r["week_of"]): r for r in out}

    print("1. Output format")
    with open(out_dir / "flagged_routes.csv", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    with open(config.SAMPLE_OUTPUT_CSV, newline="", encoding="utf-8") as fh:
        sample_header = next(csv.reader(fh))
    check(header == sample_header, f"columns match sample exactly: {header}")
    check(all(r["flagged"] in ("Yes", "No (justified)") for r in out), "flagged values are 'Yes' / 'No (justified)'")
    check(all((r["flagged"] == "Yes") == (r["matched_note_id"] == "") for r in out),
          "matched_note_id is blank exactly when flagged = Yes")
    check(all(r["reason"].isascii() and r["reason"].strip() for r in out),
          "every reason is non-empty plain ASCII (no typographic hyphens from the LLM)")

    print("2. Sample rows reproduced")
    for s in sample:
        o = rows.get((s["route"], s["week_of"]))
        same = o is not None and all(o[c] == s[c] for c in
                                     ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
                                      "flagged", "matched_note_id"])
        check(same, f"{s['route']} {s['week_of']}: {s['cost_per_tonne_km']}, {s['flagged']} {s['matched_note_id']}")

    print("3. Numbers recomputed independently")
    ref = recompute()
    bad = []
    for r in out:
        c, own, peer = ref[(r["route"], r["week_of"])]
        ok = (f"{c:.2f}" == r["cost_per_tonne_km"]
              and (own is None and pct(r["vs_own_history"]) is None or abs(own - pct(r["vs_own_history"])) < 0.0006)
              and abs(peer - pct(r["vs_similar_routes"])) < 0.0006)
        if not ok:
            bad.append(f"{r['route']} {r['week_of']}")
    check(not bad, f"all {len(out)} rows match plain-Python recomputation" + (f" (mismatch: {bad})" if bad else ""))

    print("4. Hand-labelled route-weeks")
    labels = read_csv(ROOT / "eval" / "labels.csv")
    correct = 0
    for lab in labels:
        o = rows.get((lab["route"], lab["week_of"]))
        got = o["flagged"] if o else "not_flagged"
        got_note = o["matched_note_id"] if o else ""
        ok = got == lab["expected"] and got_note == lab["expected_note_id"]
        correct += ok
        check(ok, f"{lab['route']} {lab['week_of']}: expected {lab['expected']} {lab['expected_note_id']}"
                  f" -> got {got} {got_note}   ({lab['why']})")
    print(f"  score: {correct}/{len(labels)}")

    print("5. Grounding audit of every justified row (against raw notes)")
    notes = {n["note_id"]: n for n in read_csv(config.NOTES_CSV)}
    nlab = {n["note_id"]: n for n in read_csv(ROOT / "eval" / "note_labels.csv")}
    justified = [r for r in out if r["flagged"] == "No (justified)"]
    for r in justified:
        nid = r["matched_note_id"]
        n, lab = notes.get(nid), nlab.get(nid)
        ws = date.fromisoformat(r["week_of"])
        we = ws + timedelta(days=6)
        route_ok = n is not None and n["applies_to"] in (r["route"], "All Routes")
        effect_ok = lab is not None and lab["expected_effect"] == "increase"
        win_ok = False
        if lab and lab["window_start"]:
            s = date.fromisoformat(lab["window_start"])
            e = date.fromisoformat(lab["window_end"]) if lab["window_end"] else s + timedelta(weeks=8)
            win_ok = s <= we and e >= ws
        cited_in_text = nid in r["reason"]
        check(route_ok and effect_ok and win_ok and cited_in_text,
              f"{r['route']} {r['week_of']} -> {nid}: route {route_ok}, raises cost {effect_ok}, "
              f"in window {win_ok}, cited in reason {cited_in_text}")
    if not justified:
        print("  (no justified rows)")

    print("6. Note reading vs hand labels")
    audit_path = out_dir / "note_audit.csv"
    if audit_path.exists():
        audit = read_csv(audit_path)
        for col, name in [("effect", "final (both readers)"), ("rule_effect", "rule reader"), ("llm_effect", "LLM reader")]:
            vals = [(a[col], nlab[a["note_id"]]["expected_effect"]) for a in audit if a[col]]
            if not vals:
                print(f"  ----  {name}: not run")
                continue
            acc = sum(g == e for g, e in vals)
            inc_ok = all((g == "increase") == (e == "increase") for g, e in vals)
            msg = f"{name}: {acc}/{len(vals)} exact, increase-vs-not {'all correct' if inc_ok else 'HAS ERRORS'}"
            if col == "llm_effect":  # informational: the LLM can only veto, never add, a justification
                print("  INFO  " + msg)
            else:
                check(inc_ok, msg)

    print()
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else config.OUTPUT_DIR))
