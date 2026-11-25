# Phase 0 — Gemini smoke test summary (2026-11-25)

All values are **[MEASURED]**. They come from the raw JSON files named below.

## Final pinned model configuration

| Role | Model | Pinned in |
|---|---|---|
| P1 embeddings | `gemini-embedding-001` @ 768-d (L2-normalized by us) | `problem1-rag/.env.example` |
| P1 generator | `gemini-3.1-flash-lite` | `problem1-rag/.env.example` |
| **Judge (P1 answer eval + P2 v0/v1), the only judge** | **`gemini-3.6-flash`** | `problem1-rag/.env.example`, `problem2-llm-judge/configs/judge_v{0,1}.yaml` |
| P2 generator, second family | `gemma-4-31b-it` (family `gemma`; same vendor as the judge, which is a disclosed limitation) | `problem2-llm-judge/configs/gen_v{1,2}.yaml` |

## Final smoke run: `phase0_smoke.json`, 8/8 checks OK

| Check | Result |
|---|---|
| Authentication | key loads from root `.env` (checked as True/False only); no auth errors |
| Model availability | all 4 pinned IDs listed (`gemma-4-31b-it` supports `generateContent`, `countTokens`) |
| Embedding | 3072-d and 768-d returned. 3072-d norm = 1.0; **768-d raw norm ≈ 0.58–0.59, so it must be normalized**. Truncated-and-renormalized 3072 → 768 vs API 768: cosine 1.00000 (3/3) |
| JSON-schema output | valid JSON from `gemini-3.1-flash-lite` (MINIMAL thinking, 1.8 s) and `gemini-3.6-flash` (LOW thinking, 0.9 s) |
| Usage fields | `prompt_token_count`, `candidates_token_count`, `thoughts_token_count` present (thoughts = null for both Gemini models at these levels) |
| Gemma generator | `system_instruction` supported; non-empty answer with max_output_tokens = 1024; 38 thought tokens; **21.3 s latency** |
| Quota | **no 429 errors** in any Phase 0 call. Free-tier RPM/RPD are **not measured**: the API does not expose them |

## Superseded runs (judge `gemini-3.8-flash`), kept for the record in `phase0_history/`

These are **not** results and are never mixed with final results. Every cache key includes the model ID.
- `smoke_attempt1_judge-3.8-flash.json`, `smoke_attempt2_judge-3.8-flash.json`: `gemini-3.8-flash` returned **503 UNAVAILABLE "high demand"** on every call (about 13 attempts with backoff across 3 runs). Attempt 2's `gemma:plain` was recorded as "ok" with **empty text**, because of a script bug (OK did not require non-empty text). The bug is fixed; OK now requires text.
- `phase0_fallback_probe.{py,json}`: one JSON call each. `gemini-3.8-flash` 503 · `gemini-3.6-flash` OK 1.6 s · `gemini-4-hydrogen` OK 4.3 s · `gemini-3.5-flash-lite` OK 1.5 s · `gemma-4-26b-a4b-it` 503 · `gemma-4-31b-it` OK, 73.6 s.

Decision: the judge was switched to `gemini-3.6-flash` (same price, $0.75 / $3.75 per 1M tokens) with the user's approval.

## Open risks after Phase 0
- **Gemma latency is variable** (21 s and 74 s on two single calls). About 54 generations could take 20–70 min, so they run in the background, resumable and cached.
- **Demand-driven 503s** can hit any model. The client retries with backoff, and runs are resumable from cache.
- **Free-tier daily limits are unknown** until a run hits them. Runs pause and keep their cache; billing is never enabled by us.
