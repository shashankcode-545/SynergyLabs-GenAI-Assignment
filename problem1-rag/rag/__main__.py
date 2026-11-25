"""CLI: python -m rag {ingest,query,stats,serve}  (run from problem1-rag/)"""
import argparse, json

from .config import ROOT, Settings
from .llm import Gemini


def main():
    ap = argparse.ArgumentParser(prog="python -m rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("corpus", nargs="?", default=str(ROOT / "data/corpus"))
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--overlap", type=int)
    p = sub.add_parser("query")
    p.add_argument("question")
    p.add_argument("-k", type=int)
    p.add_argument("--filter", action="append", default=[], help="key=value (source_type, doc_id)")
    sub.add_parser("stats")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    s = Settings()
    llm = Gemini(s.cache_dir, s.budget_usd, s.rpm)

    if a.cmd == "ingest":
        from .ingest import ingest
        st = ingest(a.corpus, s, llm, a.chunk_size, a.overlap)
        print(f"table {st['table']}  embed={st['embed_model']}@{st['embed_dim']}  chunk={st['chunk_size']}/{st['chunk_overlap']}")
        for d in st["ingested"]:
            print("  ingested", d)
        for d in st["skipped"]:
            print("  skipped (unchanged)", d)
        print(f"vectors before: {st['vectors_before']}  after: {st['vectors_after']}")
    elif a.cmd == "query":
        from .answer import answer
        from .store import open_table
        r = answer(a.question, s, llm, open_table(s.db_path, s.table_name(), s.embed_dim), k=a.k,
                   filters=dict(f.split("=", 1) for f in a.filter))
        print(f"Q: {r['question']}\nA: {r['answer']}")
        for c in r["citations"]:
            print(f"   [{c['sid']}] {c['title']} ({c['location']}) score={c['score']:.3f}")
        print(json.dumps({k: r[k] for k in ("abstained", "reason", "chunk_count", "latency_ms", "tokens", "cost_usd")}))
    elif a.cmd == "stats":
        from .store import open_table
        t = open_table(s.db_path, s.table_name(), s.embed_dim)
        print(json.dumps({"table": s.table_name(), "vectors": t.count_rows(), "spent_usd": round(llm.spent(), 4)}))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("rag.api:app", host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
