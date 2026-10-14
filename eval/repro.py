"""Reproducibility check: run the whole pipeline 3 times from scratch and diff.

    python eval/repro.py            # with Groq if GROQ_API_KEY is set
    python eval/repro.py --no-llm

Each run starts with an empty note-reading cache, so the LLM re-reads the notes
every time (a truly independent run). Verdicts, cited notes and numbers must be
identical; explanation wording is allowed to differ and is reported separately.
Writes output/repro/repro_report.md.
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

STRICT = ["route", "week_of", "cost_per_tonne_km", "vs_own_history",
          "vs_similar_routes", "flagged", "matched_note_id"]


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    extra = ["--no-llm"] if "--no-llm" in sys.argv else []
    base = config.OUTPUT_DIR / "repro"
    runs = []
    for i in (1, 2, 3):
        if config.CACHE_DIR.exists():
            shutil.rmtree(config.CACHE_DIR)
        out = base / f"run_{i}"
        print(f"--- run {i} ---")
        subprocess.run([sys.executable, str(ROOT / "run.py"), "--out", str(out), *extra], check=True)
        runs.append(load(out / "flagged_routes.csv"))

    strict = [[tuple(r[c] for c in STRICT) for r in run] for run in runs]
    strict_same = strict[0] == strict[1] == strict[2]
    reasons_same = [r["reason"] for r in runs[0]] == [r["reason"] for r in runs[1]] == [r["reason"] for r in runs[2]]
    diffs = [(a, b, c) for a, b, c in zip(*strict) if not (a == b == c)]

    lines = [
        "# Reproducibility report", "",
        f"Mode: {'templates only (--no-llm)' if extra else 'Groq ' + config.LLM_MODEL}, "
        f"temperature={config.LLM_TEMPERATURE}, seed={config.LLM_SEED}; note cache cleared before every run.", "",
        f"- Rows per run: {[len(r) for r in runs]}",
        f"- Numbers, flags and matched_note_id identical across 3 runs: **{'YES' if strict_same else 'NO'}**",
        f"- Explanation wording identical: {'yes' if reasons_same else 'no (allowed by the brief; verdicts unaffected)'}",
    ]
    if diffs:
        lines += ["", "## Differences", *[f"- {d}" for d in diffs]]
    report = "\n".join(lines) + "\n"
    (base / "repro_report.md").write_text(report, encoding="utf-8")
    print(report)
    sys.exit(0 if strict_same else 1)


if __name__ == "__main__":
    main()
