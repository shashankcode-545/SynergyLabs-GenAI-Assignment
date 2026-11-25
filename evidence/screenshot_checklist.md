# Screenshot checklist (template sections 1.5 and 2.5)

The candidate takes these themselves. Each command prints real output; none of these images are generated for you.
Run from PowerShell in the repo root after `Copy-Item .env.example .env` and adding the key.

## Problem 1 (`cd problem1-rag`)

1. **End-to-end query** (per-query latency, chunk count, token usage):
   `..\.venv\Scripts\python -m rag query "What are the typical root causes of Excessive Agency in LLM-based systems?"`
   The JSON line shows `latency_ms` (embed / retrieve / generate / total), `chunk_count`, `tokens` and `cost_usd`.
   If the response is cached, generate shows ~0 ms. Ask a new question to show live timing.
2. **Idempotent re-ingest**: run `..\.venv\Scripts\python -m rag ingest` twice. Every document is shown as
   `skipped (unchanged)` and the output reads `vectors before: 356  after: 356`.

## Problem 2 (`cd problem2-llm-judge`)

3. **Auditable prompt + raw response**:
   `..\.venv\Scripts\python -m judge show --pair f1-T-vs-VW`, or `python -m judge show <key-prefix>` for any key in
   `cache/calls.jsonl`. It prints the system prompt, the full judge prompt and the raw model response.
4. **Position-bias result**: `..\.venv\Scripts\python -m judge show --pair <id>` prints the AB and BA verdicts for the
   same pair plus `flipped`. Pair it with the flip-rate row of `results/bias_validation.md`.
5. **Adversarial probe outcome**: `..\.venv\Scripts\python -m judge show --pair m1-T-vs-VW` (verbose-but-wrong vs
   terse-but-correct). The per-output pointwise verdicts are in `results/experiments_raw.json`
   (P → v1 → ids `m1-VW`, `m1-T`).

Reflection text boxes ("walk through one query", "something that broke", "explain your position-bias check",
"your read on it", "AI usage disclosure") must be written by the candidate.
