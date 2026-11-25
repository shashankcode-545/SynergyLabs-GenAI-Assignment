"""Retrieval latency [MEASURED] + on-disk bytes/vector (input to the cost model).

  python -m eval.latency_bench            # real corpus table + 100K synthetic (flat vs IVF-PQ)
  python -m eval.latency_bench --n 1000000  # optional larger synthetic run
No API calls: real-corpus query vectors come from the committed embedding cache.
"""
import argparse, json, os, platform, shutil, time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from rag.config import ROOT, Settings
from rag.llm import Gemini
from rag.store import open_table, search
from .validate_questions import EVAL, load

RESULTS, BENCH = ROOT / "results", ROOT / "bench_data"


def dir_bytes(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def pct(ms):
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)), "n": len(ms)}


def timed(fn, queries, reps, warmup=20):
    for q in queries[:warmup]:
        fn(q)
    ms = []
    for _ in range(reps):
        for q in queries:
            t = time.perf_counter()
            fn(q)
            ms.append((time.perf_counter() - t) * 1000)
    return ms


def real_corpus(s, k):
    llm = Gemini(s.cache_dir, s.budget_usd)
    qv = llm.embed([q["question"] for q in load(EVAL)], s.embed_model, s.embed_dim, "RETRIEVAL_QUERY", offline=True)
    table = open_table(s.db_path, s.table_name(), s.embed_dim)
    ms = timed(lambda v: search(table, v, k), list(qv), reps=20)
    path = s.db_path / f"{s.table_name()}.lance"
    return {"table": s.table_name(), "rows": table.count_rows(), "index": "none (exact flat scan)", "k": k,
            **pct(ms), "disk_bytes": dir_bytes(path), "bytes_per_row": dir_bytes(path) / table.count_rows()}


def synthetic(n, dim, k, seed=42):
    rng = np.random.default_rng(seed)
    if BENCH.exists():
        shutil.rmtree(BENCH)
    db = lancedb.connect(BENCH)
    sch = pa.schema([pa.field("id", pa.int64()), pa.field("vector", pa.list_(pa.float32(), dim))])
    t = db.create_table("synthetic", schema=sch)
    for i in range(0, n, 50_000):
        v = rng.standard_normal((min(50_000, n - i), dim)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        t.add(pa.table({"id": np.arange(i, i + len(v)), "vector": pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), dim)}))
    qs = rng.standard_normal((100, dim)).astype(np.float32)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    run = lambda q: t.search(q.tolist()).distance_type("cosine").limit(k).select(["id"]).to_list()
    flat_ms = timed(run, list(qs), reps=2)
    truth = [{r["id"] for r in t.search(q.tolist()).distance_type("cosine").limit(10).select(["id"]).to_list()} for q in qs]
    raw_bytes = dir_bytes(BENCH)
    t0 = time.perf_counter()
    t.create_index(metric="cosine", num_partitions=max(16, int(np.sqrt(n))), num_sub_vectors=dim // 8, index_type="IVF_PQ")
    build_s = time.perf_counter() - t0
    ann = lambda q: t.search(q.tolist()).distance_type("cosine").limit(k).nprobes(20).refine_factor(5).select(["id"]).to_list()
    ann_ms = timed(ann, list(qs), reps=2)
    recall = np.mean([len(truth[i] & {r["id"] for r in t.search(q.tolist()).distance_type("cosine").limit(10).nprobes(20)
                                       .refine_factor(5).select(["id"]).to_list()}) / 10 for i, q in enumerate(qs)])
    return {"n": n, "dim": dim, "k": k, "data": f"seeded random unit vectors (seed {seed}); no cluster structure, so pessimistic for ANN",
            "flat": pct(flat_ms), "ivf_pq": {**pct(ann_ms), "num_partitions": max(16, int(np.sqrt(n))), "num_sub_vectors": dim // 8,
                                            "nprobes": 20, "refine_factor": 5, "build_s": build_s, "recall@10_vs_flat": float(recall)},
            "disk_bytes_vectors_only": raw_bytes, "disk_bytes_with_index": dir_bytes(BENCH),
            "bytes_per_vector": raw_bytes / n, "index_bytes_per_vector": (dir_bytes(BENCH) - raw_bytes) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    a = ap.parse_args()
    s = Settings()
    hw = {"os": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "python": platform.python_version(),
          "lancedb": lancedb.__version__}
    res = {"kind": "measured", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "hardware": hw,
           "real_corpus": real_corpus(s, s.top_k), "synthetic": synthetic(a.n, s.embed_dim, s.top_k)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latency.json").write_text(json.dumps(res, indent=1))
    r, y = res["real_corpus"], res["synthetic"]
    e2e = json.loads((RESULTS / "latency_e2e.json").read_text()) if (RESULTS / "latency_e2e.json").exists() else None
    md = f"""# Retrieval latency [MEASURED]

Hardware: {hw['cpu']}, {hw['logical_cpus']} logical CPUs, {hw['os']}, Python {hw['python']}, LanceDB {hw['lancedb']} (embedded, local SSD).
Timing = vector search only (query embedding is a network call to Gemini and is excluded); k = {r['k']}; 20 warm-up queries.

| Setting | Rows | Index | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|---|---|
| Real corpus | {r['rows']} | exact flat scan | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['n']} |
| Synthetic | {y['n']:,} | exact flat scan | {y['flat']['p50_ms']:.2f} | {y['flat']['p95_ms']:.2f} | {y['flat']['p99_ms']:.2f} | {y['flat']['n']} |
| Synthetic | {y['n']:,} | IVF-PQ ({y['ivf_pq']['num_partitions']} partitions, {y['ivf_pq']['num_sub_vectors']} sub-vectors, nprobes 20, refine 5) | {y['ivf_pq']['p50_ms']:.2f} | {y['ivf_pq']['p95_ms']:.2f} | {y['ivf_pq']['p99_ms']:.2f} | {y['ivf_pq']['n']} |

- IVF-PQ build: {y['ivf_pq']['build_s']:.1f} s; recall@10 vs exact scan: {y['ivf_pq']['recall@10_vs_flat']:.3f} ({y['data']}).
- On-disk size: real table {r['bytes_per_row']:.0f} B/row (768-d vector + chunk text + metadata); synthetic vectors {y['bytes_per_vector']:.0f} B/vector, IVF-PQ index +{y['index_bytes_per_vector']:.0f} B/vector.
- End-to-end (query embed + retrieval + generation, live run, answered questions only): """ + (
        f"p50 {e2e['p50_ms']:.0f} ms, p95 {e2e['p95_ms']:.0f} ms, n = {e2e['n']} (`latency_e2e.json`)." if e2e else "not recorded.") + """
- 10M vectors were **not measured** (≈ 31 GB of raw vectors exceeds this machine's free disk); the cost table projects it.
"""
    (RESULTS / "latency.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
