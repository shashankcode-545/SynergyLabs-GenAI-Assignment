"""Monthly cost at 100K / 1M / 10M vectors: self-hosted LanceDB vs Pinecone Standard serverless [PROJECTED].

Inputs measured by eval/latency_bench.py (bytes per row/vector, index bytes); prices from eval/pricing.yaml.
Run: python -m eval.cost_model   -> results/cost_table.{json,md}
"""
import json, math

import yaml

from rag.config import ROOT

RESULTS = ROOT / "results"
SCALES = [100_000, 1_000_000, 10_000_000]
QUERY_VOLUMES = [50_000, 1_000_000, 10_000_000]
CHURN = 0.05
TOKENS_PER_CHUNK = 300


def self_hosted(n, bytes_row, idx_bytes, p):
    aws, h = p["aws_us_east_1"], p["hours_per_month"]
    data_gb = n * (bytes_row + idx_bytes) / 1e9
    big = n > 2_000_000
    vm = (aws["r7g_large_usd_per_hour"] if big else aws["t4g_medium_usd_per_hour"]) * h
    ebs = math.ceil(max(20, data_gb * 1.5)) * aws["ebs_gp3_usd_per_gb_month"]
    s3 = data_gb * aws["s3_standard_usd_per_gb_month"]
    return {"instance": "r7g.large (16 GiB)" if big else "t4g.medium (4 GiB)", "data_gb": data_gb,
            "vm": vm, "ebs": ebs, "s3_backup": s3, "total": vm + ebs + s3}


def pinecone(n, record_bytes, queries, p):
    pc, u = p["pinecone_standard"], p["pinecone_units"]
    gb = n * record_bytes / 1e9
    storage = gb * pc["storage_usd_per_gb_month"]
    reads = queries * max(u["min_ru_per_query"], gb * u["ru_per_gb_namespace_per_query"]) * pc["read_usd_per_million_ru"] / 1e6
    writes = n * CHURN * record_bytes / 1000 * u["wu_per_kb_upserted"] * pc["write_usd_per_million_wu"] / 1e6
    usage = storage + reads + writes
    return {"namespace_gb": gb, "storage": storage, "reads": reads, "writes": writes, "usage": usage,
            "total": max(pc["monthly_minimum_usd"], usage)}


def main():
    p = yaml.safe_load((ROOT / "eval/pricing.yaml").read_text())
    lat = json.loads((RESULTS / "latency.json").read_text())
    bytes_row = lat["real_corpus"]["bytes_per_row"]
    idx_bytes = lat["synthetic"]["index_bytes_per_vector"]
    rows = []
    for n in SCALES:
        sh = self_hosted(n, bytes_row, idx_bytes, p)
        pcs = {q: pinecone(n, bytes_row, q, p) for q in QUERY_VOLUMES}
        rows.append({"vectors": n, "self_hosted": sh, "pinecone": pcs,
                     "embed_once_usd": n * TOKENS_PER_CHUNK * p["gemini_embedding_2_usd_per_million_tokens"] / 1e6})
    out = {"kind": "projected", "inputs_measured": {"bytes_per_row": bytes_row, "index_bytes_per_vector": idx_bytes},
           "assumptions": {"queries_per_month_baseline": QUERY_VOLUMES[0], "monthly_churn": CHURN,
                           "tokens_per_chunk": TOKENS_PER_CHUNK, "region": "AWS us-east-1 / Pinecone lowest listed rate",
                           "replicas": 1}, "prices": p, "rows": rows}
    (RESULTS / "cost_table.json").write_text(json.dumps(out, indent=1))
    b = QUERY_VOLUMES[0]
    lines = [f"| {r['vectors']:,} | {r['self_hosted']['total']:.2f} ({r['self_hosted']['instance']}, {r['self_hosted']['data_gb']:.2f} GB) | "
             f"{r['pinecone'][b]['total']:.2f} (usage {r['pinecone'][b]['usage']:.2f}; min $50) | "
             f"{r['pinecone'][b]['total'] - r['self_hosted']['total']:+.2f} $/mo vs self-hosted |" for r in rows]
    sens = [f"| {r['vectors']:,} | " + " | ".join(f"{r['pinecone'][q]['total']:.2f}" for q in QUERY_VOLUMES) +
            f" | {r['self_hosted']['total']:.2f} |" for r in rows]
    md = f"""# Cost comparison across scale [PROJECTED]

Monthly cost of the store only (embedding/LLM spend is the same either way). Measured inputs from `latency.json`:
{bytes_row:.0f} B per stored row (768-d float32 vector + ~1.2 KB chunk text + metadata) and {idx_bytes:.0f} B/vector of IVF-PQ index.

| Vectors | Self-hosted LanceDB ($/mo) | Pinecone Standard serverless ($/mo) | Managed minus self-hosted |
|---|---|---|---|
""" + "\n".join(lines) + f"""

Baseline traffic: {b:,} queries/month (a large, lightly-queried index). Sensitivity, Pinecone $/mo by monthly query volume:

| Vectors | {' | '.join(f'{q:,} q/mo' for q in QUERY_VOLUMES)} | Self-hosted (flat in queries until the VM saturates) |
|---|---|---|---|---|
""" + "\n".join(sens) + f"""

**Pricing assumptions.**
- Pinecone Standard (verified on pinecone.io/pricing and the docs' cost page, 2026-11-25): $50/mo minimum, $0.33/GB-month storage, $16/M read units, $4/M write units (lowest listed regional rates). A query costs 1 RU per GB of namespace, minimum 0.25. Writes are 1 WU per KB upserted, with {CHURN:.0%} of vectors re-upserted per month.
- Self-hosted on AWS us-east-1, 1 replica:
  - t4g.medium at $0.0336/h (≤1M vectors) or r7g.large at $0.1071/h (10M);
  - gp3 EBS at $0.08/GB-month, sized 1.5× the data with a 20 GB minimum;
  - one S3 Standard backup copy at $0.023/GB-month.
- The AWS prices are **assumptions**: the official pages did not expose the tables as text on 2026-11-25. Re-check them before relying on these figures.
- One-time embedding of the corpus with gemini-embedding-2 at {TOKENS_PER_CHUNK} tokens per chunk: """ + ", ".join(
        f"{r['vectors']:,} → ${r['embed_once_usd']:.2f}" for r in rows) + """ (identical for both stores).
- Not costed: engineering and on-call time, HA (a second replica roughly doubles the VM + EBS lines), egress.
- 10M latency was not measured.
"""
    (RESULTS / "cost_table.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
