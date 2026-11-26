# P2 A/B generator fallback: investigation (no changes made)

**Date:** 2026-11-26. This is a decision aid only. No configuration, cache, prompt or model ID was changed, and no generation call was made. Model metadata came from `models.get` (catalog calls, no quota). Token figures are **[MEASURED]** from the 11 existing `gemma-4-26b-a4b-it` A/B generations in `cache/gen_calls.jsonl`. Costs are **[ESTIMATE]**: list price × those token counts.

## What the generator call needs

Sent by `pipeline.generate`, from `configs/gen_v*.yaml`:
- `system_instruction`: the config prompt plus the item constraint;
- `temperature=0`, `seed=42`, `max_output_tokens=1024`;
- no JSON schema and no thinking configuration.

Measured per call (mean / max):

| Token type | Mean | Max |
|---|---|---|
| prompt | 25 | 45 |
| output | 35 | 116 |
| thinking | 143 | 307 |

Median latency was 15 s. The 1024 limit is not binding.

## Candidates (both compatible with the existing `google-genai` client and `GEMINI_API_KEY_P2`)

| | **A. `gemma-4-31b-it`** | **B. `gemini-2.5-flash-lite`** |
|---|---|---|
| Family vs judge (`gemini-3.1-flash-lite`, Gemini) | **Gemma: different family**, so the generator ≠ judge mitigation holds | **Gemini: same family** as the judge. `guard_family` refuses it unless `--allow-same-family` is set, and that flag is recorded in the report |
| `generateContent`, `system_instruction`, `temperature`, `seed`, 1024 tokens | supported. `system_instruction`, `seed` and `temperature=0` were verified live in Phase 0. Output limit 32,768 | supported (catalog: `generateContent`, max temperature 2.0, output limit 65,536). **Not yet verified live** |
| Thinking | yes (Gemma 4 thinks; the limit is ample) | yes (thinking model; thinking billed as output) |
| Price (pricing page read 2026-11-25) | free tier only ($0; not offered on paid tier) | $0.10 in / $0.40 out per 1M tokens; "not deprecated, served until further notice" |
| **Cost for 24 generations (both arms)** | **$0** | **≈ $0.002 mean, ≤ $0.005 worst case** |
| Availability evidence | 503 "high demand" on every attempt on 2026-11-25. Phase 0 single-call latency 21–161 s. Shares Gemma capacity with 26b-a4b (currently 503) | never called; different capacity pool from Gemma; **unknown** |
| Consistency with the rest of P2 | the originally approved generator; the self-preference set's Gemma side was generated with 31b | no prior use in P2 |
| Methodological risk | **lowest**: every documented mitigation is unchanged | **higher**: breaks the assignment's "judge from a different model family than the generator" mitigation for the A/B (both arms are Gemini, judged by Gemini) |

Rejected:
- `gemini-3.1-flash-lite`: it is the judge itself, so the generator would be the same model.
- `gemini-3.5-flash`: pricing not verified.
- `gemini-3.5-flash-lite`: 503s yesterday.
- `gemini-4-*`: no free tier, or 5× cost.
- `gemini-2.5-flash` / `-pro`: deprecated and rerouted on 2026-12-01.
- Non-Google models: not reachable through the existing client and key.

## Assessment

- **Safest methodological fallback: A (`gemma-4-31b-it`).** It changes nothing in the method, keeps the family separation and costs $0. Its weakness is availability, since it shares Gemma capacity that has been returning 503.
- **Pragmatic fallback if Gemma stays down: B (`gemini-2.5-flash-lite`).** Likely available and costs well under a cent. But the A/B result would then rest on a same-family judge.
  - Mitigating evidence [MEASURED]: the self-preference probe found δ = 0.00 (all 32 verdicts were ties).
  - Both A/B arms would come from the same generator, so any family preference shifts both arms. The remaining risk is an interaction with prompt style.
  - Report it with `--allow-same-family` recorded and the limitation stated.

## If either is approved (the same procedure for both)

1. Change `model` (and `family` for B) in `configs/gen_v1.yaml` **and** `gen_v2.yaml`. Both arms must use one generator, so the A/B is **regenerated in full: 24 calls**.
   - The 11 existing 26b-a4b generations stay in the cache for audit; keys contain the model id, so nothing mixes.
   - The completed self-preference set is pinned separately and is unaffected.
2. Update the `test_phase0.py` generator assertion, then run one controlled generation first (as with the probe).
3. Run `--stage gen`, then `experiments`: 48 A/B judge calls, ≈ $0.05 [ESTIMATE]. Bias and validation results are reused from cache unchanged.
4. Validate (`tests/test_results.py`, offline replay) and declare a winner only under the fixed decision rule.

**Option 0, no switch:** keep `gemma-4-26b-a4b-it` and retry later. Only 13 generations are missing and nothing is regenerated, so this has the least change. Its availability is the same as A's.

## Outcome of the automatic fallback attempt (2026-11-26)

Each model got exactly one controlled call with a single attempt and no retries. Overrides were in memory only.

1. `gemma-4-31b-it`:
   - The first call (`gen_v1:f1`) came from **cache** and made no API call. 31b already has A/B outputs from 2026-11-25: `gen_v1` 12/12 and `gen_v2` 3/12 (`f1`–`f3`).
   - The live call on the first uncached item, `gen_v2:f4`, returned `503 UNAVAILABLE "This model is currently experiencing high demand"` after 2.6 s.
2. `gemini-2.5-flash-lite` (`gen_v1:f1`): `404 NOT_FOUND "This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemini-3.5-flash-lite"`. The catalog lists it, but the P2 key's project cannot call it.

**Result:** both fallbacks are unavailable. **No generator was selected.** Nothing was generated, no config was changed, and the caches are unchanged. The A/B comparison stays **pending, with no winner**.

**Note for the next attempt:** with `gemma-4-31b-it`, only **9** generations (`gen_v2` `f4`–`c4`) are needed for a complete single-generator A/B set, against 13 for `gemma-4-26b-a4b-it`. Both are Gemma family and $0. Either way, the 48 judge calls follow (≈ $0.05).

## Decision (2026-11-26)

- **Preferred generator: `gemma-4-31b-it`.** Its 15 cached A/B answers are preserved. The configuration stays unchanged until a probe succeeds.
- **Next checkpoint:** 2026-11-26 14:00 local. One `judge probe --gen-model gemma-4-31b-it`; no repeated retries while it returns 503.
- **`gemini-3.5-flash-lite`** only with explicit user approval. All 24 answers would then be regenerated with it, and the same-family limitation documented.

## Completion (2026-11-26)

- The user's probe succeeded at 10:25 and cached `gen_v2:f4`.
- Both A/B configs were then switched to `gemma-4-31b-it`.
- Only the 8 missing answers were generated (cache 36 → 44, no duplicates).
- The 48 A/B judge calls followed: 42 live, 6 identical prompts served from cache.
- Result: **no significant difference, incumbent v1 kept** under the fixed decision rule (`problem2-llm-judge/results/ab_report.md`).
