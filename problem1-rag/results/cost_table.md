# Cost comparison across scale [PROJECTED]

Monthly cost of the store only (embedding/LLM spend is the same either way). Measured inputs from `latency.json`:
3930 B per stored row (768-d float32 vector + ~1.2 KB chunk text + metadata) and 119 B/vector of IVF-PQ index.

| Vectors | Self-hosted LanceDB ($/mo) | Pinecone Standard serverless ($/mo) | Managed minus self-hosted |
|---|---|---|---|
| 100,000 | 26.14 (t4g.medium (4 GiB), 0.40 GB) | 50.00 (usage 0.52; min $50) | +23.86 $/mo vs self-hosted |
| 1,000,000 | 26.22 (t4g.medium (4 GiB), 4.05 GB) | 50.00 (usage 5.23; min $50) | +23.78 $/mo vs self-hosted |
| 10,000,000 | 83.99 (r7g.large (16 GiB), 40.49 GB) | 52.27 (usage 52.27; min $50) | -31.73 $/mo vs self-hosted |

Baseline traffic: 50,000 queries/month (a large, lightly-queried index). Sensitivity, Pinecone $/mo by monthly query volume:

| Vectors | 50,000 q/mo | 1,000,000 q/mo | 10,000,000 q/mo | Self-hosted (flat in queries until the VM saturates) |
|---|---|---|---|---|
| 100,000 | 50.00 | 50.00 | 63.09 | 26.14 |
| 1,000,000 | 50.00 | 64.96 | 630.87 | 26.22 |
| 10,000,000 | 52.27 | 649.61 | 6308.67 | 83.99 |

**Pricing assumptions.**
- Pinecone Standard (verified on pinecone.io/pricing and the docs' cost page, 2026-11-25): $50/mo minimum, $0.33/GB-month storage, $16/M read units, $4/M write units (lowest listed regional rates). A query costs 1 RU per GB of namespace, minimum 0.25. Writes are 1 WU per KB upserted, with 5% of vectors re-upserted per month.
- Self-hosted on AWS us-east-1, 1 replica:
  - t4g.medium at $0.0336/h (≤1M vectors) or r7g.large at $0.1071/h (10M);
  - gp3 EBS at $0.08/GB-month, sized 1.5× the data with a 20 GB minimum;
  - one S3 Standard backup copy at $0.023/GB-month.
- The AWS prices are **assumptions**: the official pages did not expose the tables as text on 2026-11-25. Re-check them before relying on these figures.
- One-time embedding of the corpus with gemini-embedding-2 at 300 tokens per chunk: 100,000 → $6.00, 1,000,000 → $60.00, 10,000,000 → $600.00 (identical for both stores).
- Not costed: engineering and on-call time, HA (a second replica roughly doubles the VM + EBS lines), egress.
- 10M latency was not measured.
