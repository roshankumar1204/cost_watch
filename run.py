"""Run the full analysis.

    python run.py                 # uses Groq if GROQ_API_KEY is set (in .env or the environment)
    python run.py --no-llm        # deterministic templates only, no API calls
    python run.py --out output/run_2
"""
import argparse

from dotenv import load_dotenv

import config
from src.pipeline import run

if __name__ == "__main__":
    load_dotenv(config.ROOT / ".env")
    ap = argparse.ArgumentParser(description="Flag unjustified freight cost rises.")
    ap.add_argument("--out", default=str(config.OUTPUT_DIR), help="output folder")
    ap.add_argument("--no-llm", action="store_true", help="skip Groq; use template explanations")
    args = ap.parse_args()
    run(out_dir=args.out, use_llm=not args.no_llm)
