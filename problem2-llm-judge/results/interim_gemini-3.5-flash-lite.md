# INTERIM report [MEASURED, INCOMPLETE]: judge `gemini-3.5-flash-lite`

**Not the final result.** This judge was replaced after repeated 503 errors. Its run stopped part-way, so this report uses
only the calls that were cached before it stopped, and only this one model (no cross-model mixing). No API calls were made to produce it.

| Run | Coverage |
|---|---|
| P_v0 | 48/48 |
| P_v1 | 19/48 |
| Q_v0 | 24/24 |
| Q_v1 | 1/24 |
| S | 0 (not run) |
| R | 0 (not run) |
| AB | 0 (not run) |

## v0 (naive) baseline: complete for P and Q

| Metric | v0 |
|---|---|
| Gold agreement (pointwise, T/VW/CW) | 100% (kappa 1.00, n = 36) |
| Verbose-wrong passed / confident-wrong passed / terse-correct failed | 0% / 0% / 0% |
| Padded-minus-terse score | -0.59 |
| Score clustering: normalized entropy, levels used, AUROC correct vs wrong | 0.60, 7/10, 1.00 |
| Pairwise flip rate (AB vs BA) / slot-1 win rate | 0% / 50% (n = 24 pairs) |
| Pairwise accuracy vs gold: single order / swap-aggregated | 100% / 100% |
| Padded preferred over terse / verbose-wrong preferred (single order) | 0% / 0% |

## Pointwise v0 vs v1 on the 19 cases cached for both (items f1, f2, f3, f4, m1)

| Metric | v0 | v1 |
|---|---|---|
| Gold agreement | 100% | 100% |
| Cohen's kappa vs gold | 1.00 | 1.00 |
| Verbose-wrong passed | 0% | 0% |
| Confident-wrong passed | 0% | 0% |
| AUROC correct vs wrong | 1.00 | 1.00 |
| Normalized entropy | 0.59 | 0.62 |

Not available from cached data: pairwise v1 (position bias after mitigation: 1/24 pairs), self-preference (S),
test-retest (R), and the A/B comparison. The final results need the live runs with the current judge.
