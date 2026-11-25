# Problem 2 — LLM-as-Judge Evaluation Pipeline

A test suite (JSON / JSONL / YAML) goes in; structured verdicts and a suite report come out. The pipeline handles malformed JSON, logs every prompt and raw response, compares two configs and declares a winner, and **measures five judge biases, before vs after mitigation**.

Tags: **[MEASURED]** = from our runs; every number in `results/` is regenerated from `cache/calls.jsonl` with `--offline`.

## Architecture

```mermaid
flowchart TD
  GC["generator config (configs/gen_v1|v2.yaml): model, family, system prompt"] -.-> GEN["pipeline.generate -> model_output (frozen into a suite)"]
  GEN --> S["Suite JSON / JSONL / YAML (schema.load_suite, pydantic)"]
  S --> GUARD{"judge.family == generator.family?"}
  GUARD -- yes --> STOP["refuse (unless --allow-same-family)"]
  GUARD -- no --> MODE{"mode"}
  MODE -- pointwise --> PP["prompt: rubric anchors + few-shot + reference block"]
  MODE -- pairwise --> SW["ORDER SWAP: AB and BA prompts"]
  PP --> CALL["judge call (configs/judge_v0|v1.yaml): JSON mode, temp 0, seed<br/>cache = audit log, budget cap, RPM throttle, retries"]
  SW --> CALL
  CALL --> PARSE["PARSE FALLBACK: json -> strip fences -> outer braces -> json_repair -> pydantic"]
  PARSE -- invalid --> RETRY["1 repair retry with the error"] --> PF{"still invalid?"}
  PF -- yes --> FAIL["parse_failed: kept in log, excluded, counted"]
  PARSE -- ok --> GR["GROUNDING: correctness / decisive quote must be verbatim in the response"]
  PF -- no --> GR
  GR --> AGG["pass rule in code; pairwise AB/BA must agree, else tie"]
  AGG --> REP["suite report: pass rate, mean scores, win rate, flip rate, CIs, calls/tokens/$"]
  REP --> CMP["compare A vs B: pre-registered decision rule -> winner"]
```

## Setup and run (Windows PowerShell, Python 3.11)

```powershell
cd problem2-llm-judge
..\.venv\Scripts\python -m pip install -r requirements.txt
# repo-root .env: GEMINI_API_KEY_P2 (Problem 2 uses only this key; all variable names: .env.example)
..\.venv\Scripts\python -m judge run --suite suites/demo.yaml                 # suite in -> results/run_demo_judge_v1.{md,json}
..\.venv\Scripts\python -m judge compare --a suites/ab_gen_v1.jsonl --b suites/ab_gen_v2.jsonl
..\.venv\Scripts\python -m judge experiments --offline                      # rebuild every bias/validation/A-B table from the log
..\.venv\Scripts\python -m judge show --pair f1-T-vs-VW                      # raw judge prompt + raw response, both orders
..\.venv\Scripts\python -m pytest -q                                         # offline tests
```

- **Independent configs:** judge and generator are configured separately, `configs/judge_v0.yaml` / `judge_v1.yaml` vs `configs/gen_v1.yaml` / `gen_v2.yaml`. Each has a `model` and a `family`.
- **Models:**
  - judge: `gemini-3.5-flash-lite` (family gemini). It replaced `gemini-3.6-flash`, whose free tier allows only 20 requests/day;
  - generator: `gemma-4-31b-it` (family gemma).
- **Spend cap:** a hard budget cap (`BUDGET_USD=3.5`) is checked before every call.
- **Audit log:** every call is appended to `cache/calls.jsonl` with tag, model version, system prompt, full prompt, raw text, tokens, cost and latency. It is committed, so all results can be audited and replayed.

## Judging design

**Modes:** pointwise and pairwise.

| Mode | When it fits | Used here for |
|---|---|---|
| Pointwise, reference-based | a gold answer exists (facts, math, extraction) | probe suite, A/B pass rates |
| Pointwise, reference-free | open-ended tasks; gating on an absolute bar | supported (e.g. `free-1` in the demo suite) |
| Pairwise A-vs-B | choosing between two configs; small differences; robust to score clustering | position/verbosity probes, A/B win rate |

**Rubric** (`judge/rubric.yaml`): 1–5 with anchors at 1, 3 and 5.

| Criterion | Weight |
|---|---|
| correctness | 0.35 |
| faithfulness | 0.20 |
| completeness | 0.15 |
| instruction-following (content-free length counts against it) | 0.15 |
| tone / safety (incl. endorsing a false premise) | 0.15 |

- **Pass rule, computed in code:** weighted score ≥ 3.5, correctness ≥ 3, tone/safety ≥ 4, **and** the correctness evidence is a verbatim quote from the response.
- **Before vs after:** the "before" judge **v0** uses one 1–10 score, no anchors and a single order. The "after" judge **v1** uses the rubric, few-shot anchors, verified evidence and both orders. Both run on the same model and the same data.

**Malformed JSON:** the fallback chain is `json.loads` → strip code fences → extract the outermost `{...}` → `json_repair` → pydantic validation. If that fails, one repair retry is sent with the error. If the retry also fails, the case is `parse_failed`: it is kept in the log, excluded from aggregates and counted in reports. `tests/test_judge.py` covers 8 malformed fixtures.

## Results

See [`results/bias_validation.md`](results/bias_validation.md) (bias before → after, judge validation), [`results/ab_report.md`](results/ab_report.md) (A/B winner) and [`results/report.md`](results/report.md) (suite report).

**Final results: pending.** The full suite with the current judge (`gemini-3.1-flash-lite`) and A/B generator (`gemma-4-26b-a4b-it`) has not run. Google returned `503 UNAVAILABLE "high demand"` for every model we tried, even after bounded retries (log in `../reports/progress_report.md`). No final number is reported until `results/bias_validation.md` and `results/ab_report.md` exist; `tests/test_results.py` then validates them.

**Interim, incomplete [MEASURED]:** [`results/interim_gemini-3.5-flash-lite.md`](results/interim_gemini-3.5-flash-lite.md).
- Source: the cached calls of the replaced judge `gemini-3.5-flash-lite`, one judge model only, built with no API calls.
- Coverage: v0 complete (48 pointwise, 24 pairs); v1 partial (19 pointwise, 1 pair).
- Finding: the naive v0 judge already scores 100% gold agreement, a 0% AB/BA flip rate and AUROC 1.00. These clear-cut probes hit a **ceiling** for this judge, so they may not show mitigation gains. Harder probes would be needed to show bias reduction.

### How to finish (resumable; all cached calls are skipped)

```powershell
..\.venv\Scripts\python -m judge probe          # 1 judge + 1 generator call (2 attempts each); exit 0 = both available
..\.venv\Scripts\python -u -m judge experiments --stage core
$env:LLM_TIMEOUT_S=240; ..\.venv\Scripts\python -u -m judge experiments --stage gen --cache gen_calls.jsonl
..\.venv\Scripts\python -u -m judge experiments                  # S + AB + all reports (or --stage bias without A/B)
..\.venv\Scripts\python -m pytest -q                              # includes results validation
```

Remaining: 298 judge calls (≈ $0.22 estimated from measured per-call costs) and 21 A/B generations ($0).
