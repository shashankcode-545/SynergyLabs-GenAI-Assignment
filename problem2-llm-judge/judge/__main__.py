"""CLI (run from problem2-llm-judge/):
  python -m judge run --suite suites/demo.yaml [--judge judge_v1]     suite in -> results/<name>.{json,md}
  python -m judge compare --a suites/ab_gen_v1.jsonl --b suites/ab_gen_v2.jsonl [--judge judge_v1]
  python -m judge experiments [--offline]                             bias + validation + A/B (all cached)
  python -m judge show <key-prefix> | --pair <pair-id>               audit: raw judge prompt + raw response
"""
import argparse, json
from pathlib import Path

from .pipeline import RESULTS, compare, load_cfg, make_llm, run_suite, write
from .schema import load_suite


def main():
    ap = argparse.ArgumentParser(prog="python -m judge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--suite", required=True)
    p.add_argument("--judge", default="judge_v1")
    p.add_argument("--allow-same-family", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--judge", default="judge_v1")
    p = sub.add_parser("experiments")
    p.add_argument("--stage", choices=["all", "gen", "core", "bias"], default="all")
    p.add_argument("--cache", help="log file to append to (default: judge_<judge model>.jsonl; all cache/*.jsonl are read)")
    p = sub.add_parser("probe", help="API availability check: 1 judge + 1 generator call (both reused by the suite)")
    p = sub.add_parser("interim", help="offline report from cached calls of one judge model")
    p.add_argument("--model", required=True)
    p = sub.add_parser("show")
    p.add_argument("key", nargs="?")
    p.add_argument("--pair")
    for s in sub.choices.values():
        s.add_argument("--offline", action="store_true", help="replay from cache/calls.jsonl only; no API calls")
    a = ap.parse_args()
    # judge responses go to a per-judge-model file; keys also contain the model id, so models never mix
    llm = make_llm(a.offline, getattr(a, "cache", None) or f"judge_{load_cfg('judge_v1')['model']}.jsonl")

    if a.cmd == "run":
        cfg = load_cfg(a.judge)
        cases = load_suite(a.suite)
        for c in cases:
            c.meta["allow_same_family"] = a.allow_same_family
        recs, summary = run_suite(cases, cfg, llm, tag="run:")
        name = f"run_{Path(a.suite).stem}_{cfg['name']}"
        from .experiments import suite_md
        write(name, {"suite": a.suite, "judge": cfg, "summary": summary, "cases": recs}, suite_md(a.suite, recs))
        print((RESULTS / f"{name}.md").read_text(encoding="utf-8"))
    elif a.cmd == "compare":
        from .experiments import ab_md
        ab = compare(load_suite(a.a), load_suite(a.b), load_cfg(a.judge), llm)
        write(f"compare_{Path(a.a).stem}_vs_{Path(a.b).stem}", ab, ab_md(ab))
        print(ab_md(ab))
    elif a.cmd == "experiments":
        from .experiments import run_all
        run_all(llm, a.stage)
        if a.stage in ("all", "bias"):
            print((RESULTS / "bias_validation.md").read_text(encoding="utf-8"))
            if a.stage == "all":
                print((RESULTS / "ab_report.md").read_text(encoding="utf-8"))
    elif a.cmd == "probe":
        import os, sys
        os.environ.setdefault("LLM_RETRY_ATTEMPTS", "2")  # probe gently while the API is overloaded
        os.environ.setdefault("LLM_TIMEOUT_S", "240")
        from .experiments import availability_probe
        # judge responses -> judge_<model>.jsonl (llm), generator responses -> gen_calls.jsonl
        res = availability_probe(llm, make_llm(cache_name="gen_calls.jsonl"), make_llm(offline=True))
        print(json.dumps(res))
        sys.exit(0 if res["judge"] and res["generator"] else 1)
    elif a.cmd == "interim":
        from .experiments import interim_report
        interim_report(a.model)
        print((RESULTS / f"interim_{a.model}.md").read_text(encoding="utf-8"))
    elif a.cmd == "show":
        keys = []
        if a.pair:
            raw = json.loads((RESULTS / "experiments_raw.json").read_text(encoding="utf-8"))
            rec = next(r for grp in ("Q", "S") for v in raw[grp].values() for r in v if r["id"] == a.pair and r["version"] == "v1")
            print(f"pair {rec['id']}: AB -> candidate {rec['orders']['AB']['candidate']}, BA -> candidate {rec['orders']['BA']['candidate']}, "
                  f"flipped={rec['flipped']}, final={rec['final']} (1 = first output, 2 = second, 0 = tie)\n")
            keys = [o["keys"][0] for o in rec["orders"].values()]
        else:
            keys = [k for k in llm.entries if k.startswith(a.key)][:1]
        for k in keys:
            e = llm.entries[k]
            print(f"=== {e['tag']}  model={e['model_version']}  tokens={e['usage']}  cost=${e['cost_usd']:.5f}\n--- SYSTEM\n{e['params']['system']}\n"
                  f"--- PROMPT\n{e['prompt']}\n--- RAW RESPONSE\n{e['text']}\n")
    print(f"[api calls this run: {llm.calls}, cache hits: {llm.cached_hits}, total spend logged: ${llm.spent():.4f}]")


if __name__ == "__main__":
    main()
