"""Ask the assistant about its own findings in plain English (stretch goal).

    python ask.py "why did Chennai-Bangalore get pricier in March 2025?"
    python ask.py            # interactive

Answers come ONLY from the files the last run wrote (output/). The question is
used to pick the relevant rows (route + month), those rows are given to the
LLM as the only evidence, and it is told to say so when the results don't cover it.
Without a GROQ_API_KEY it just prints the relevant rows.
"""
import re
import sys

import pandas as pd
from dotenv import load_dotenv

import config
from src.llm import LLM

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

SYSTEM = """You answer questions about a freight-cost monitoring run.
Use ONLY the result rows provided. Quote route, week_of, the percentages and note ids exactly.
If a week was justified, name the note. If it was flagged, say no supporting note was found.
If the rows do not answer the question, say "The results don't cover that." Keep it under 120 words."""


def load_results():
    out = config.OUTPUT_DIR
    flags = pd.read_csv(out / "flagged_routes.csv", keep_default_na=False)
    details = pd.read_csv(out / "flag_details.csv", keep_default_na=False)
    weekly = pd.read_csv(out / "weekly_metrics.csv")
    return flags.merge(details[["route", "week_of", "rejected_notes"]], on=["route", "week_of"], how="left"), weekly


def select_rows(question: str, flags: pd.DataFrame, weekly: pd.DataFrame):
    q = question.lower()
    routes = [r for r in weekly["route"].unique()
              if r.lower() in q or all(city.lower() in q for city in r.split("-"))]
    if not routes:  # a single city name is enough if it is unambiguous
        routes = [r for r in weekly["route"].unique() if any(city.lower() in q for city in r.split("-"))]
    months = [i + 1 for i, m in enumerate(MONTHS) if m in q or re.search(rf"\b{m[:3]}\b", q)]
    years = re.findall(r"\b(20\d\d)\b", q)

    def keep(df):
        dt = pd.to_datetime(df["week_of"])
        m = pd.Series(True, index=df.index)
        if routes:
            m &= df["route"].isin(routes)
        if months:
            m &= dt.dt.month.isin(months)
        if years:
            m &= dt.dt.year.isin([int(y) for y in years])
        return df[m]

    f = keep(flags)
    w = keep(weekly)[["route", "week_of", "cost_per_tonne_km", "vs_own_pct", "vs_peer_pct", "is_anomaly"]]
    return f, w.head(20)


def answer(question: str, llm: LLM) -> str:
    flags, weekly = load_results()
    f, w = select_rows(question, flags, weekly)
    if f.empty and w.empty:
        return "The results don't cover that (no matching route or month)."
    evidence = ("FLAGGED/JUSTIFIED ROWS:\n" + (f.to_csv(index=False) if not f.empty else "(none)\n")
                + "\nWEEKLY NUMBERS (vs_own_pct / vs_peer_pct are fractions):\n" + w.round(4).to_csv(index=False))
    if not llm.available:
        return evidence
    reply = llm.chat(SYSTEM, f"Question: {question}\n\n{evidence}", purpose="ask")
    return reply or evidence


if __name__ == "__main__":
    load_dotenv(config.ROOT / ".env")
    llm = LLM()
    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:]), llm))
    else:
        print("Ask about the results (blank line to quit).")
        while (q := input("> ").strip()):
            print(answer(q, llm), "\n")
