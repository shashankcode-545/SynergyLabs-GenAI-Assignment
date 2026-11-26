# Progress report

## Update 2026-11-26: P2 bias + validation complete; A/B blocked by Gemma 503 (uncommitted, pending approval)

- **Judge** `gemini-3.1-flash-lite`, key `GEMINI_API_KEY_P2`. Every run is complete and verified:

  | Run | Calls |
  |---|---|
  | P | 96 |
  | Q | 96 |
  | S | 32 |
  | R | 32 |

  256 calls, $0.1405 (`results/usage.md`); 0 parse failures; 0 repairs.
- **Validation:** `tests/test_results.py` passes 3/3. An offline replay reproduces `bias_validation.md` and `report.md` byte-for-byte.
- **Results** (`results/bias_validation.md`):
  - no position bias, sycophancy or self-preference detected, in v0 or v1 (ceiling on clear-cut probes);
  - v0 failed 8/12 padded-but-correct answers, v1 0/12 (padded-minus-terse −0.55 → −0.11);
  - score entropy 0.63 → 0.76;
  - gold agreement 100% → 100% (κ 1.00);
  - test-retest flips 0%.
- **Errors during the run (all progress kept):**
  - core stage: 1× `504 DEADLINE_EXCEEDED` (not in the retry list at the time; 504 has since been added to the bounded-retry codes), then a successful resume;
  - Gemma: 503 "high demand" on 3 attempts. The last bounded loop made no progress and stopped.

  Logs: `evidence/p2_run_*.log`.
- **A/B: not completed.** Generations `gen_v1` 10/12, `gen_v2` 1/12 (13 missing); 48 A/B judge calls not run; **no winner declared**.
- **Deliverables updated:**
  - P2 README (results, Discussion, limitations);
  - `reports/submission.docx`: bias, validation, rubric and design cells filled from results; A/B cells PENDING; candidate cells untouched; headings verified.
- **Spend:** P2 total across all cache files $0.1805 of $3.50.
- **Remaining:** 13 Gemma generations ($0) and 48 judge calls (≈ $0.05 est.).
- **Resume:** `--stage gen --cache gen_calls.jsonl`, then `experiments`.

## Update 2026-11-25: offline recovery work while Google returns 503 (no live calls made)

- **Interim report** (offline, cached data only): `problem2-llm-judge/results/interim_gemini-3.5-flash-lite.md`.
  - One judge model only, clearly marked incomplete.
  - Coverage: P v0 48/48, Q v0 24/24, P v1 19/48, Q v1 1/24; S, R and AB 0.
  - Finding: a ceiling effect. The v0 judge scores 100% gold agreement, a 0% flip rate and AUROC 1.00 on the probes.
- **`--stage bias --offline`:** not run. The current judge (`gemini-3.1-flash-lite`) has only 6 cached calls, so it would produce no meaningful metrics.
- **Submission draft:** `reports/submission.docx`, from `reports/fill_submission.py`.
  - 43 headings are identical to the template.
  - Problem 1 cells come from real results.
  - Problem 2 static cells (diagram, commands, rubric, mitigations, design decisions) are filled.
  - Problem 2 result cells read **PENDING**.
  - Candidate-only cells are untouched.
- **Availability probe:** `python -m judge probe`. It makes 1 uncached P v1 judge call and 1 uncached A/B generation, with 2 attempts each. Both are real suite calls, so a success is cached and reused. Exit code 0 means both models are available. Unit-tested offline; **not run live**.
- **Scheduling capability:**
  - Desktop scheduled tasks (persistent, cron or one-time) and session-only `CronCreate` are both available.
  - **Nothing was scheduled.**
  - Proposed, pending approval: run `probe` about hourly; on exit 0, resume `--stage core`, `--stage gen`, then `experiments`, validate, and report.
  - Per probe: at most 2 successful calls (≈ $0.0012); failed 503 attempts return no tokens.
- **Remaining:** 298 judge calls (≈ $0.22) and 21 A/B generations ($0). P2 spend so far: $0.0434 of $3.50.

## Update 2026-11-25: smoke tests with GEMINI_API_KEY_P2 failed on 503; suite not launched

- The P2 client is verified to use `GEMINI_API_KEY_P2`: the authenticated catalog call succeeded.
- Live smoke tests (uncached inputs that belong to the real runs):
  - `gemini-3.1-flash-lite` pointwise (P v1 `f1-T`): `503 UNAVAILABLE "This model is currently experiencing high demand"` after bounded retries.
  - Pairwise test: not reached, because it runs after the failed pointwise call.
  - `gemma-4-26b-a4b-it` generation (A/B `gen_v1` `f3`): same 503.
- No new cache entries (judge 6, gen 27 lines unchanged). Per the instruction, the full suite was **not** launched.
- The new key changed nothing about the 503s, so the cause is model capacity, not the key or quota.
- Counts, remaining work and resume commands are unchanged from the 19:41 update below.

## Update 2026-11-25 19:41: full suite launched with the approved models; both stopped on 503 (external capacity)

- **Approved configuration:**
  - judge `gemini-3.1-flash-lite` (v0 and v1);
  - A/B generator `gemma-4-26b-a4b-it` (`configs/gen_v1/v2.yaml`);
  - the self-preference set stays pinned to `gemma-4-31b-it` and is reused from cache (8 pairs, 0 calls).
- **A/B generation** (`evidence/p2_final_gen.log`): 2 new outputs (`ab:gen_v1:f1`, `f2`), then `503 UNAVAILABLE "high demand"` from `gemma-4-26b-a4b-it` after bounded retries (about 19:39).
  - Arms so far: 3 of 24 (v1 f1, f2; v2 c1 from the earlier controlled test).
- **Judge core stage** (`evidence/p2_final_core.log`): `503 UNAVAILABLE "high demand"` from `gemini-3.1-flash-lite` on its first call (about 19:40); 0 new calls.
  - Judge cache `cache/judge_gemini-3.1-flash-lite.jsonl`: 6 entries (P v0/v1 `m1-VW`; Q v0/v1 `m1-T-vs-VW`, both orders).
- **Neither stage was restarted** (repeated-503 rule).
- **Completion vs expected judge calls** (new judge):

  | Run | Done / expected |
  |---|---|
  | P | 2/96 |
  | Q | 4/96 |
  | R | 0/32 |
  | S | 0/32 |
  | AB | 0/48 |

  **No experiment is complete, and no result files exist.**
- **P2 spend:** $0.0434 of the $3.50 cap, summed over all P2 cache files.
- **Added:** a `--stage bias` option (P, Q, S, R + bias/validation reports with no A/B dependency) and `tests/test_results.py`. The test validates counts, ids, parse rate, single judge model and metric recomputation once results exist; until then it skips.
- **Remaining:** 298 judge calls (≈ $0.22) and 21 A/B generations ($0). Then reports, README results, `reports/fill_submission.py` (written, not run) and the submission docx.
- **Resume:**
  ```powershell
  cd problem2-llm-judge
  ..\.venv\Scripts\python -u -m judge experiments --stage core
  $env:LLM_TIMEOUT_S=240; ..\.venv\Scripts\python -u -m judge experiments --stage gen --cache gen_calls.jsonl
  ..\.venv\Scripts\python -u -m judge experiments             # or --stage bias if Gemma stays down
  ```

## Update 2026-11-25: judge switched to gemini-3.1-flash-lite (approved); generator test pending approval

- **Judge:** `configs/judge_v0.yaml` and `judge_v1.yaml` now use `gemini-3.1-flash-lite`.
  - Responses go to `cache/judge_gemini-3.1-flash-lite.jsonl` (6 calls so far: the v1 and v0 sanity checks).
  - Production config check: v1 replayed entirely from cache (0 new calls); v0 made 3 new calls; all replies parsed and were grounded; no flips.
- **Old judge responses:** the 117 `gemini-3.5-flash-lite` responses stay untouched in `cache/calls.jsonl` for audit. Cache keys include the model id, so none of them can enter the new metrics.
- **Generator test (not adopted):** one call to `gemma-4-26b-a4b-it` on A/B input `c1` with the `gen_v2` prompt.
  - Output `"Yellow"`: one word, correct.
  - 21.9 s, 45 prompt / 1 output / 135 thinking tokens, $0 (Gemma is free tier only).
  - Saved in `cache/gen_test_gemma-4-26b-a4b-it.jsonl`.
  - `gemma-4-31b-it` still returned 503 at the last check.
- **Remaining (estimate):**
  - 298 judge calls ≈ $0.22, from measured per-call costs.
  - If the 26b generator is approved: 24 A/B regenerations for both arms ($0). The self-preference set is unchanged.

## Update 2026-11-25 19:06: P2 live runs partially done, stopped on a third 503 (not restarted)

- **Judge:** `gemini-3.5-flash-lite`. JSON-schema sanity checks passed; both pointwise and pairwise replies parsed and grounded.
- **Done and cached** (`cache/calls.jsonl`):

  | Run | Judge calls |
  |---|---|
  | P v0 | 48/48 |
  | Q v0 | 48/48 |
  | P v1 | 19/48 |
  | Q v1 | 2/48 (sanity pair) |
  | R | 0/32 |
  | S | 0/32 |
  | AB | 0/48 |

  There were 0 repair retries and every reply parsed. The offline replay confirms that cached calls are skipped on resume.
- **Blocker:** the judge stage failed twice with `503 UNAVAILABLE "This model is currently experiencing high demand"` after the SDK's bounded retries ran out: about 18:42 on `P:pw:f4-T:v1`, and 18:43 on `P:pw:f4-VW:v1` after a resume. Logs: `evidence/p2_core_run_attempt1_503.log` and `evidence/p2_core_run.log`. A single resume after a 10-minute pause (with the new timeout) cached 4 more calls, then failed at 19:06 with the same 503 on `P:pw:m1-CW:v1` (`evidence/p2_core_run_resume2.log`). Per instruction it was not restarted again.
  - Earlier, one call took 361 s inside the SDK before succeeding.
  - No 429 or quota errors.
  - Retry setup: 6 attempts, exponential back-off from 2 s capped at 60 s, on 429/500/503. A 120 s per-attempt HTTP timeout (`LLM_TIMEOUT_S`) has now been added; it does not change cache keys.
- **Gemma generation** (`cache/gen_calls.jsonl`): stopped by `503 "high demand"` from `gemma-4-31b-it` after bounded retries (`evidence/p2_gen_run.log`). The self-preference set is complete (4 items, originals + paraphrases, no empty outputs, in `suites/selfpref.jsonl`). A/B `gen_v1` is 12/12; A/B `gen_v2` is 3/12 and failed at `f4`, leaving 9 Gemma calls.
- **Remaining:** 187 judge calls (P v1 29, Q v1 46, R 32, S 32, AB 48) and 9 Gemma generations (A/B `gen_v2` f4-c4) plus the rest of the Gemma calls. Then validation, reports, README updates and `reports/fill_submission.py` (written, not yet run).
- **P2 spend so far:** ≈ $0.031.
- **Resume command:** `cd problem2-llm-judge; ..\.venv\Scripts\python -u -m judge experiments --stage core`, then `--stage gen --cache gen_calls.jsonl` if generation is unfinished, then `python -m judge experiments`.

---
## Earlier status (2026-11-25, before the P2 live runs)

**Offline tests:** 55/55 pass (repo `tests/` 6, `problem1-rag` 35, `problem2-llm-judge` 14). No API calls are needed to run them.

**Recorded API spend** (from cached usage):

| Item | Spend |
|---|---|
| P1 generation + judge | $0.0281 |
| P1 embeddings | $0.0647 (chars/4 estimate) |
| P2 | $0.0022 |
| **Total** | **≈ $0.095** |

Caps: P1 $1.5 + P2 $3.5 = $5 project cap. Billing was never enabled.

## Completed

| Component | State | Commit |
|---|---|---|
| Phase 0: scaffold, pinned models, smoke test | done | `cd907dc` |
| P1 corpus (8 docs, 2 PDF / 4 MD / 2 HTML) + 18 eval / 5 dev questions + validator | done (questions AI-drafted; candidate review recommended) | `7e17448` |
| P1 RAG service: ingest (idempotent), LanceDB, query path, abstention, FastAPI, CLI, logging, cache, budget cap | done | `14628ac` |
| P1 evaluation: retrieval (2 chunk configs), answers, latency (real + 100K synthetic), cost table, README with Discussion; offline replay is byte-identical | done | `14628ac` |
| P2 pipeline code: schema/suite loader (JSON/JSONL/YAML), rubric, prompts v0/v1, parse fallback + repair retry, grounding checks, AB/BA order swap, family guard, aggregation/stats, A/B decision rule, generation, probe builder (48 outputs + 24 pairs), experiments P/Q/S/R/AB + report writers, CLI (`run`, `compare`, `experiments --stage`, `show`) | done; **not yet run live end to end** | this commit |
| Evidence: `evidence/screenshot_checklist.md`, Phase 0 summary, run logs | done | this commit |

## Missing (mandatory)

1. **P2 live results.** Not produced yet:
   - bias before → after (all 5 biases)
   - judge validation (gold agreement + κ, test-retest, adversarial)
   - A/B winner
   - suite report (`results/*.md`)

   The P2 README results section is marked pending.
2. **P2 README Discussion**: how biased before vs after; would you let it gate a release (evidence only — the verdict is the candidate's).
3. **Submission template**: fill a copy of `reference/submission_template.docx` (non-reflection cells only) and check that the headings are unchanged.
4. **Screenshots**: taken by the candidate using `evidence/screenshot_checklist.md`.
5. **Reflection answers and AI-usage disclosure**: written by the candidate.

## Blockers

| Blocker | Detail | Resets |
|---|---|---|
| `gemini-3.6-flash` (judge) free tier: **20 requests/day** | Used up by P1 judging and P2 smoke tests. P2 needs about 340 judge calls, so the free tier cannot finish P2 even after the reset. | ≈ 2026-11-26 13:30 local (429 said "retry in 22h22m" at 15:28) |
| `gemma-4-31b-it` (second-family generator) | 503 "high demand"; the generation stage failed after 6 retries (only 2 of 6 self-preference answers cached). Gemma is free-tier only. | unknown (capacity, not quota) |
| `gemini-embedding-001` free tier: 1,000 texts/day | Used up. P1 switched to `gemini-embedding-2` (user-approved); P1 is complete. | ≈ 22 h |

**Decided:** the P2 judge is now `gemini-3.5-flash-lite` (configs updated; its free daily limit is still unknown). **Still open:**
- **Generator:** either keep retrying Gemma 31b with a smaller set (12 A/B inputs + 4 self-preference items), or switch to `gemma-4-26b-a4b-it`.

## Cache notes (do not delete)

- `problem2-llm-judge/cache/calls.jsonl` (14 entries): 6 Gemini and 2 Gemma self-preference generations, 1 pairwise v0 call, and 5 pointwise v0 calls.
  - The pointwise v0 calls for `f1-T` and `f1-P` (2 calls + 2 repairs) used the old permissive `{"type": "object"}` schema and returned `{ }`. That schema is fixed in `judge.judge.schema_for`; those keys will never be reused.
  - The pointwise `f1-VW` call (valid JSON) and the pairwise `Q:pair:f1-T-vs-VW:AB:v0` call used the new schema. The **BA order of that pair is not cached** (the 429 hit there).
  - The P2 client reads every `cache/*.jsonl` and appends only to its own file.
- `problem1-rag/cache/llm.jsonl` and `emb_gemini-embedding-2_768.npz`: all P1 results replay offline.

## Exact next steps (after the quota reset or billing)

```powershell
cd problem2-llm-judge
# 0. if switching judge: edit configs/judge_v0.yaml + judge_v1.yaml (model) and tests/test_phase0.py JUDGE; re-run pytest
# 1. generation (Gemma; resumable, own log file):
..\.venv\Scripts\python -u -m judge experiments --stage gen --cache gen_calls.jsonl
# 2. judge core runs P, Q, R (~226 judge calls):
..\.venv\Scripts\python -u -m judge experiments --stage core
# 3. remaining S + AB runs and all reports (results/bias_validation.md, ab_report.md, report.md, experiments_raw.json):
..\.venv\Scripts\python -u -m judge experiments
# 4. demo suite (mostly cache hits) and offline replay check:
..\.venv\Scripts\python -m judge run --suite suites/demo.yaml
..\.venv\Scripts\python -m judge experiments --offline
```

5. Fill the P2 README results and Discussion from `results/*.md`.
6. Fill the submission docx. `python-docx` 1.2.0 is installed in `.venv` (not yet in `requirements.lock.txt`). Template table indices:
   - Problem 1: 0 = candidate details; 5–9 = results; 10–15 = design decisions.
   - Problem 2: 25–28 = results; 29–33 = design decisions (33 = "gate a release" → evidence bullets only).
   - Leave 16–20 and 34–38 (screenshots and reflections) for the candidate.
7. Run all tests, then commit.
