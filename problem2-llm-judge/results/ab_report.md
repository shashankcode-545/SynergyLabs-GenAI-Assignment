# A/B comparison [MEASURED]: generator prompt gen_v1 (A) vs gen_v2 (B)

Generator `gemma-4-31b-it` (family gemma) for both arms; judge v1 (`gemini-3.1-flash-lite`, family gemini). 12 inputs.
Decision rule (fixed before the run): B wins if the 95% CI of B's win rate is above 0.5, A if below, otherwise keep A.

| Config | Pass rate | Mean score (0-1) | Win rate |
|---|---|---|---|
| Config A - gen_v1 ("Answer the user's request.") | 0.83 [0.58, 1.00] | 1.00 [1.00, 1.00] | 0.42 |
| Config B - gen_v2 (constraint-following + fact-check prompt) | 0.92 [0.75, 1.00] | 1.00 [1.00, 1.00] | 0.58 [0.50, 0.71] |

Pairwise (both orders, agreement rule): B wins 2, A wins 0, ties 10; order flip rate 0%; sign test p = 0.500.
**Declared winner: A (no significant difference; incumbent kept).**
Judge cost: pairwise 24 calls $0.0110; pointwise A 12 calls $0.0102, B 12 calls $0.0096.
