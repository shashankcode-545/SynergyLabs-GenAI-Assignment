# Screenshot checklist (template sections 1.5 and 2.5)

**Take these yourself.** The template uses them to confirm the work is yours, so no image here was generated for you.

Every command below was run on 2026-11-26 against the committed data. Its real output is saved as text in `evidence/screenshot_outputs/` so you can compare. All commands run **offline from the committed caches**: no API key and no spend.

PowerShell with the venv activated (`.venv\Scripts\Activate.ps1`), from the repo root.

## Problem 1: `cd problem1-rag`

**1. End-to-end query** (template 1.5, first box: answer, citation, latency, chunk count, tokens)
```powershell
python -m rag query "What are the typical root causes of Excessive Agency in LLM-based systems?"
```
Expected output (`screenshot_outputs/1_p1_query.txt`):
- the answer with an `[S1]` citation to *LLM06:2025 Excessive Agency (Description)*, score 0.855;
- then a JSON line with `chunk_count: 5`, `latency_ms` per stage and `tokens`.

The response is served from cache, so `generate` shows ~0 ms, and `cost_usd` is the original call's recorded cost, not new spend. Say so in the caption.

**2. Idempotent re-ingest** (template 1.5, second box)
```powershell
python -m rag ingest
python -m rag ingest
```
Expected output (`2_p1_ingest_twice.txt`): both runs list all 8 documents as `skipped (unchanged)` and end with `vectors before: 356  after: 356`. Capture both runs in one screenshot.

## Problem 2: `cd problem2-llm-judge`

**3. Auditable raw prompt + response** (template 2.5, first box)
```powershell
python -m judge show --pair f1-T-vs-VW | Select-Object -First 50
```
The first 50 lines hold the verdict line, the system prompt, the full judge prompt and the raw JSON response for the AB order. The full output (both orders) is in `3_p2_raw_prompt_response_f1.txt`.

**4. Position-bias result** (template 2.5, second box)
```powershell
python -m judge show --pair f1-T-vs-VW | Select-Object -First 1
Select-String -Path results\bias_validation.md -Pattern "Position \(A/B order\)"
```
Expected output (`4_p2_position_bias.txt`):
- `AB -> candidate 1, BA -> candidate 1, flipped=False, final=1`;
- then the table row: flip rate 0% -> 0%, slot-1 win rate 50% -> 50%.

**5. Adversarial probe outcome** (template 2.5, "paste verdict + your read on it"; *your read* is yours to write)
```powershell
python -m judge show --pair m1-T-vs-VW | Select-Object -First 50
python -m judge show 80d101f95757 | Select-Object -First 45
```
- The first command shows the terse-correct `408.` and the verbose-wrong `418` walk-through judged in both orders. The correct answer wins both times, so the judge was not fooled (`5_p2_adversarial_m1.txt`).
- The second command shows the pointwise v1 verdict on the verbose-but-wrong answer alone: correctness 1, *"17 x 4 is 68, not 78"* (`5b_p2_adversarial_pointwise_m1-VW.txt`).

## Still yours to write

The template's reflection boxes:
- "Walk through one query" and "Something that broke first time" (1.5);
- "Explain your position-bias check" and "your read on it" for the adversarial probe (2.5);
- both "AI usage disclosure" boxes.

Plus the candidate details table and the two architecture diagrams (1.1, 2.1) as images. The Mermaid sources are in each problem's README.
