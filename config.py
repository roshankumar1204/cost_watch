"""All tunable settings live here so every choice is visible in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"

SHIPMENTS_CSV = DATA_DIR / "shipment_records.csv"
NOTES_CSV = DATA_DIR / "context_notes.csv"
SAMPLE_OUTPUT_CSV = DATA_DIR / "sample_output_format_v2.csv"

# Output contract: column names and order, copied from sample_output_format_v2.csv.
OUTPUT_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history",
    "vs_similar_routes", "flagged", "matched_note_id", "reason",
]

# ---- Baselines (fixed by the brief) ----
HISTORY_WEEKS = 8  # trailing window, strictly before the current week

# ---- Flag rule ----
# A week is flagged when cost is RISING (above its own trailing average) AND it is
# out of the ordinary on at least one comparison. See README "Choosing thresholds".
OWN_HISTORY_THRESHOLD = 0.15   # +15% vs own 8-week average  (~3.5 std devs of week-to-week noise)
PEER_THRESHOLD = 0.20          # +20% vs same-type peers (above the largest calm-period gap, 16.6%)

# ---- Note matching ----
# A one-off price step is absorbed into the 8-week baseline after 8 weeks, so an
# open-ended note ("starting this week") can only explain rises in that window.
OPEN_ENDED_WINDOW_WEEKS = HISTORY_WEEKS
# A note that quantifies its effect (e.g. "roughly 5-7%") may explain a rise up to
# that amount plus this tolerance, never more.
MAGNITUDE_TOLERANCE = 0.03
RETRIEVAL_TOP_K = 3

# ---- LLM (Groq free tier) ----
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
LLM_REASONING_EFFORT = "low"   # gpt-oss is a reasoning model; low keeps tokens small
LLM_MAX_TOKENS = 600
# Groq's published on-demand price for openai/gpt-oss-120b (USD per 1M tokens).
# On the free tier the actual charge is $0; this is the "if you paid" figure.
PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60
PROMPT_VERSION = "v2"  # bump to invalidate cached note readings
