# Retrieval latency [MEASURED]

Hardware: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 16 logical CPUs, Windows-10-10.0.26300-SP0, Python 3.11.9, LanceDB 0.41.0 (embedded, local SSD).
Timing = vector search only (query embedding is a network call to Gemini and is excluded); k = 5; 20 warm-up queries.

| Setting | Rows | Index | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|---|---|
| Real corpus | 356 | exact flat scan | 6.99 | 8.28 | 9.41 | 360 |
| Synthetic | 100,000 | exact flat scan | 210.48 | 225.74 | 258.52 | 200 |
| Synthetic | 100,000 | IVF-PQ (316 partitions, 96 sub-vectors, nprobes 20, refine 5) | 9.33 | 11.48 | 14.34 | 200 |

- IVF-PQ build: 39.5 s; recall@10 vs exact scan: 0.134 (seeded random unit vectors (seed 42); no cluster structure, so pessimistic for ANN).
- On-disk size: real table 3930 B/row (768-d vector + chunk text + metadata); synthetic vectors 3074 B/vector, IVF-PQ index +119 B/vector.
- End-to-end (query embed + retrieval + generation, live run, answered questions only): p50 6040 ms, p95 53748 ms, n = 12 (`latency_e2e.json`).
- 10M vectors were **not measured** (≈ 31 GB of raw vectors exceeds this machine's free disk); the cost table projects it.
