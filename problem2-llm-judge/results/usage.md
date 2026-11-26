# Judge usage [MEASURED]: `gemini-3.1-flash-lite` (from `cache/judge_gemini-3.1-flash-lite.jsonl`)

| Run | Version | Calls | Prompt tokens | Output tokens | Thinking tokens | Latency p50 (ms) | Latency p95 (ms) | Cost ($) |
|---|---|---|---|---|---|---|---|---|
| P | v0 | 48 | 7817 | 2280 | 5708 | 1822 | 18871 | 0.0139 |
| P | v1 | 48 | 37817 | 18008 | 5582 | 2722 | 43168 | 0.0448 |
| Q | v0 | 48 | 9836 | 2307 | 4973 | 2124 | 20258 | 0.0134 |
| Q | v1 | 48 | 29660 | 5910 | 5366 | 2120 | 17991 | 0.0243 |
| R | v1 | 32 | 25162 | 12244 | 3864 | 2247 | 13345 | 0.0305 |
| S | v0 | 16 | 4836 | 1088 | 1618 | 1444 | 6213 | 0.0053 |
| S | v1 | 16 | 11444 | 2000 | 1634 | 1670 | 6200 | 0.0083 |

Total: 256 calls, $0.1405. Cost = API usage metadata x list price ($0.25 / $1.50 per 1M input / output tokens, thinking billed as output).
