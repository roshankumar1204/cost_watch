# Freight Cost Watch: a smart assistant for unjustified freight cost rises

It finds route-weeks where cost per tonne-km jumps compared with the route's own history or with similar routes. It then checks the context notes for a genuine reason and explains each verdict in plain English. **It never invents a reason.**

```
shipments ─► weekly cost/t-km ─► baselines ─► flag rule ─► retrieve notes ─► verify ─► explain ─► CSV
             (pandas)            own 8 wk +    (thresholds)  (vector index +   (code    (Groq LLM,
                                 same-type peers               route/date filter) decides) wording only)
```

## Quick start (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          &:: then put your free Groq key in .env
python run.py                   &:: writes output\flagged_routes.csv
python eval\evaluate.py         &:: correctness checks (exit code 0 = all pass)
python eval\repro.py            &:: 3 clean runs + diff -> output\repro\repro_report.md
python ask.py "why did Chennai-Bangalore get pricier in March 2025?"
```

`python run.py --no-llm` runs fully offline. Verdicts and numbers are identical in both modes; only the wording of `reason` changes.

Python 3.10+. Get a free key at https://console.groq.com.

## What's in the box

| Path | What it does |
|---|---|
| `src/metrics.py` | Assigns shipments to Mon-Sun weeks, computes weekly cost per tonne-km, the 8-week own-history baseline and the same-type peer baseline |
| `src/flags.py` | Flag rule (thresholds in `config.py`) |
| `src/notes.py` | Turns each note into facts (effect, active window, magnitude) using **two independent readers**: rules and the LLM |
| `src/retrieval.py` | Local vector index over the notes, with metadata-filtered search |
| `src/justify.py` | Verifies retrieved notes against the flagged week; **decides the verdict** |
| `src/explain.py` | LLM writes the reason; an output guard checks it, with a template fallback |
| `src/llm.py` | Groq client, temperature 0, fixed seed, retries, token log |
| `ask.py` | Plain-English Q&A over the run's own results (stretch goal) |
| `eval/` | Hand labels, evaluation harness, reproducibility check |
| `output/flagged_routes.csv` | **The submission file** (exact sample format) |
| `output/flag_details.csv`, `note_audit.csv`, `weekly_metrics.csv`, `token_log.json` | Audit trail |

## How it works

### 1. Weekly cost and baselines (exactly as the brief defines them)
- `week_of` = the Monday of the shipment's Mon-Sun week.
- `cost_per_tonne_km` = Σ freight cost ÷ Σ (tonnes × km) for that route-week. This is a weighted ratio, not an average of per-shipment ratios.
- **vs own history:** the mean of the route's previous ≤ 8 weekly values. It uses `shift(1)` so there is no look-ahead.
- **vs similar routes:** the mean of the other routes of the same `route_type` in the same week, with the route itself excluded.

**Fewer than 8 prior weeks:** the data starts 2024-01-01, so weeks from 2024-01-08 to 2024-02-19 use 1 to 7 prior weeks (all that exist). The week of 2024-01-01 has no history, so it can't be flagged on that comparison. `flag_details.csv` records `prior_weeks_used` for every flagged row. Nothing is padded or extrapolated.

**Peer groups are small.** Short and Long have only 2 routes each, so their "peer average" is the single other route. Medium has 3 routes.

### 2. Flag rule
A week is flagged when cost is **rising** (above its own 8-week average) **and** at least one of these holds:
- it is **≥ +15% vs its own history**, or
- it is **≥ +20% vs similar routes**.

**Choosing thresholds.** I set these from the data's own noise, not by guessing:
- Week-to-week change vs own history has a std dev of 4.2%. +15% is about 3.5σ. The largest own-history move in calm 2024 weeks was 8.6%.
- The peer gap is noisier (std dev 9.0%) and partly structural: Mumbai-Pune normally runs about 11% above Delhi-Jaipur. The largest peer gap in 2024 with no event behind it was 16.6%, so the threshold sits above that at 20%.
- The peer comparison is what catches **slow creep**. Mumbai-Pune drifts up through late 2025, and its own 8-week baseline drifts up with it (only +3 to 9%). Against Delhi-Jaipur it reaches +39%. That's the "quietly creeping" case the brief describes.

Output rows are the weeks that trip the rule: `Yes` if no note explains them, `No (justified)` if one does. Weeks that never tripped the rule are left out, as in the sample. All weeks with their numbers are in `output/weekly_metrics.csv`.

### 3. Finding a reason (RAG, with guardrails)
**Reading the notes.** Every note is read twice:
1. a deterministic rule reader. Negations win: "not significantly affected", "without a rate change", "not part of this dataset", "returned to normal", "improved", and so on.
2. the LLM, at temperature 0 with JSON output, cached in `cache/`.

A note counts as a cost-increase reason only if **both** readers agree. Its active window is the **overlap** of both readings, and any disagreement is logged in `note_audit.csv`. So the LLM can make the system more cautious, but it can never make it more generous.

**Retrieval.** The notes go into a TF-IDF vector index. Groq has no embedding API, and for 10 short notes a dense model adds downloads without improving recall. Swapping in embeddings means changing one method. For a flagged week, search keeps only notes that could apply (that route or "All Routes", within 8 weeks of the week), ranked by similarity.

**Verification (code, not the LLM).** A note justifies a flag only if:
- it says costs went **up** (N004-N010 don't: not in dataset, not affected, stable, improved, back to normal, absorbed),
- its active window **covers the flagged week** (Mon-Sun overlap). An open-ended note like "starting this week" is active for 8 weeks, until the step change is absorbed into the 8-week baseline.
- an **"All Routes" note cannot explain a gap vs peers**, because the peers got the same increase,
- if the note states a size ("roughly 5-7%"), the rise is no bigger than that plus 3 points.

If no note passes, the verdict is **unexplained**. There is no guessing.

**Explanation.** The LLM is given the numbers, the verdict, the matched note and the notes that were rejected (with reasons), and it writes 1-2 sentences. An **output guard** checks the result:
- a justified reason must cite the matched note id and no other;
- an unexplained reason must not present any note as the cause.

If the check fails, a deterministic template is used instead.

## Results

19 flagged weeks: **3 justified, 16 flagged for review.**

| Route | Weeks | Verdict | Why |
|---|---|---|---|
| Ahmedabad-Mumbai | 2025-01-20 | Justified, N002 | festival surcharge week |
| Chennai-Bangalore | 2025-02-24, 03-03 | Justified, N001 | flood Feb 24 - Mar 8 |
| Chennai-Bangalore | 2025-03-10, 03-17 | Flagged | flood over by Mar 9, but costs stayed high. N009 ("returned to normal") is not a reason |
| Delhi-Jaipur | 2024-11-11, 11-18 | Flagged | no note covers it |
| Mumbai-Pune | 13 weeks, Jun-Dec 2025 | Flagged | steady creep vs peers. N003 (diesel) is network-wide, and N006/N010 are not cost increases |

The traps it correctly does **not** fall for: N004 (toll, not on these routes), N005 (costs not affected), N006/N008 (stable), N007 (roads improved), N009 (flood ended), N010 (absorbed, no rate change), and N003 used to explain a route-specific gap.

## How I convinced myself it's right (`eval/evaluate.py`)
1. **Format:** columns and order match `sample_output_format_v2.csv` exactly. `matched_note_id` is blank exactly when the row is flagged.
2. **The 3 sample rows** reproduce the sample exactly (numbers, flag, note).
3. **Independent recomputation:** every output number is recomputed by a separate plain-Python implementation with no pandas. This check caught a real formatting bug ("4.1" instead of "4.10") before submission.
4. **14 hand-labelled route-weeks**, built by reading every note against the data: justified weeks, trap weeks and should-not-flag weeks. Score: **14/14**.
5. **Grounding audit:** every justified row is re-checked against the raw notes and hand-labelled windows (right route, week inside the window, note really says costs rose, id cited in the text).
6. **Note reading:** each reader's classification vs hand labels. Rules: 10/10. LLM: 9-10/10 exact across runs, and 10/10 on the question that matters, "is this a cost increase or not". Final combined reading: 10/10.
7. **Clean text:** every reason is non-empty plain ASCII.

On the real Groq run: **all checks pass, 14/14 labels.**

## Reproducibility (`eval/repro.py`)
Three runs from scratch, with the note cache deleted before each one so the LLM re-reads every note. The script diffs `route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id` across the runs.

**Randomness:** the LLM runs at temperature 0, seed 42 and `reasoning_effort=low`. Even if the LLM varied, it can't move a verdict: verdicts come from code, the LLM's note reading can only veto, and its wording is guarded. See `output/repro/repro_report.md`.

**Result with Groq (`openai/gpt-oss-120b`), 3 runs with the cache cleared each time:**
- 19 rows in every run.
- Numbers, flags and `matched_note_id` were **identical** across the 3 runs.
- Explanation wording differed slightly between runs, which the brief allows.

**What the real LLM runs taught me** (this is why the design is "LLM proposes, code disposes"):
- **N007 ("road conditions improved"):** the LLM reader called it `decrease` in one run and `no_effect` in two. Both are "not a reason for a rise", so verdicts were unaffected, and the disagreement was logged in `note_audit.csv`.
- **N003 ("diesel prices rose ... starting this week"):** in prompt v1 the LLM sometimes read this as covering only one week, so its active window changed between runs. Verdicts were again unaffected, because N003 is network-wide and is rejected for that reason first. The fixes:
  - the prompt now says that "starting ..." with no end date is open-ended (prompt v2);
  - rejection checks now run in a fixed order: what the note says, who it covers, how big an effect it claims, and only then when it applies. So the stated reason no longer depends on a window edge case.
- **Typographic hyphens:** the LLM wrote "Ahmedabad‑Mumbai" with a non-breaking hyphen. Reasons are now normalised to ASCII, and `evaluate.py` checks for this. The new check caught the issue in the v1 output.
- **All 19 LLM explanations passed the output guard.** The guard's fallback path was tested separately, by feeding it a reason that blamed the wrong note.

## Cost of running it
Model: **`openai/gpt-oss-120b` on Groq's free tier**, the strongest open-weight model Groq hosts. It's free to use, and its published rate is $0.15 / 1M input tokens and $0.60 / 1M output tokens.

**One full run over the entire shipment file** (`output/token_log.json`, measured with the cache empty):

| | LLM calls | Input tokens | Output tokens |
|---|---|---|---|
| Read the 10 notes (then cached) | 10 | 3,896 | 1,422 |
| Write 19 explanations | 19 | 6,301 | 1,751 |
| **Total** | **29** | **10,197** | **3,173** |

The estimated cost at the published rate is 10,197 × $0.15/1M + 3,173 × $0.60/1M ≈ **$0.0034 (about ₹0.29)**. The actual cost on the free tier was **$0**.

The numbers and verdicts use no LLM calls at all. LLM calls grow with the number of flagged weeks, not with shipments: 2,940 shipments produce only 29 calls. A rerun with a warm cache makes 19 calls.

## Limitations and trade-offs
- The thresholds are calibrated on this dataset. On new data I would re-derive them from the noise, or use a robust z-score per route.
- Peer comparison with a single peer (Short/Long) is fragile: one noisy peer week can trigger a flag. The `rising` condition guards against some of this.
- The rule reader's phrase lists are tuned to this note style. That's why the LLM reader exists and why disagreements are logged instead of silently resolved.
- Mumbai-Pune 2025-11-10 is mid-creep but not flagged, because it dipped just below its own (already inflated) average. That's an honest consequence of the "rising" rule.
