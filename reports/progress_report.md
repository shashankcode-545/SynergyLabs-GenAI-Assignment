# Progress report (2026-11-25, paused at the user's request until the free-tier quotas reset)

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
