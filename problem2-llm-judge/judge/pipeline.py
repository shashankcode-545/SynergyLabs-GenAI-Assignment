"""Suite in -> report out; A/B compare with a pre-registered decision rule; generation; self-enhancement family guard."""
import json, os, subprocess, time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .judge import judge_pair, judge_pointwise
from .llm import Gemini
from .schema import CRITERIA, TestCase
from .stats import bootstrap_ci, sign_test_p

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT.parent / ".env", encoding="utf-8-sig")
RESULTS = ROOT / "results"


def load_cfg(name_or_path):
    p = Path(name_or_path)
    p = p if p.suffix else ROOT / "configs" / f"{name_or_path}.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def make_llm(offline=False, cache_name="calls.jsonl"):
    return Gemini(ROOT / "cache" / cache_name, float(os.getenv("BUDGET_USD") or 3.5),
                  int(os.getenv("LLM_RPM") or 15), offline=offline)


def guard_family(judge_cfg, gen_family, allow_same=False):
    if gen_family and judge_cfg["family"] == gen_family and not allow_same:
        raise ValueError(f"self-enhancement guard: judge family '{judge_cfg['family']}' == generator family '{gen_family}'"
                         " (pass --allow-same-family to override; the report records it)")


def provenance(judge_cfg, llm):
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {"kind": "measured", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": sha, "judge": judge_cfg,
            "api_calls_made": llm.calls, "cache_hits": llm.cached_hits, "spent_total_usd": round(llm.spent(), 4)}


def summarize_pointwise(recs):
    ok = [r for r in recs if "passed" in r]
    out = {"n": len(recs), "parse_failed": len(recs) - len(ok), "pass_rate": bootstrap_ci([float(r["passed"]) for r in ok]),
           "mean_score_norm": bootstrap_ci([r["score_norm"] for r in ok]),
           "calls": sum(len(r["keys"]) for r in recs), "cost_usd": sum(r["cost_usd"] for r in recs),
           "tokens": {k: sum(r["tokens"][k] for r in recs) for k in ("prompt", "output", "thoughts")}}
    if ok and "scores" in ok[0]:
        out["mean_by_criterion"] = {c: sum(r["scores"][c] for r in ok if c in r["scores"]) / max(1, sum(c in r["scores"] for r in ok))
                                    for c in CRITERIA}
    return out


def run_suite(cases, judge_cfg, llm, tag=""):
    for c in cases:
        guard_family(judge_cfg, c.meta.get("family"), c.meta.get("allow_same_family", False))
    recs = [judge_pointwise(c, judge_cfg, llm, tag=tag) for c in cases]
    return recs, summarize_pointwise(recs)


def compare(cases_a, cases_b, judge_cfg, llm, tag="ab:"):
    """A/B: pairwise (both orders, agreement rule) + pointwise on both arms. Cases joined on id."""
    b_by_id = {c.id: c for c in cases_b}
    pairs = [judge_pair(a.id, a.input, a.system_prompt, a.expected_output, a.model_output, b_by_id[a.id].model_output,
                        judge_cfg, llm, tag=tag) for a in cases_a]
    pw_a, sum_a = run_suite(cases_a, judge_cfg, llm, tag=tag + "A:")
    pw_b, sum_b = run_suite(cases_b, judge_cfg, llm, tag=tag + "B:")
    decided = [p for p in pairs if p["final"] is not None]
    win_b = [1.0 if p["final"] == 2 else 0.0 if p["final"] == 1 else 0.5 for p in decided]
    wins_b, wins_a = sum(p["final"] == 2 for p in decided), sum(p["final"] == 1 for p in decided)
    wr = bootstrap_ci(win_b)
    winner = "B" if wr[1] > 0.5 else "A" if wr[2] < 0.5 else "A (no significant difference; incumbent kept)"
    return {"pairs": pairs, "pointwise_a": pw_a, "pointwise_b": pw_b, "summary_a": sum_a, "summary_b": sum_b,
            "win_rate_b": wr, "wins_b": wins_b, "wins_a": wins_a, "ties": len(decided) - wins_a - wins_b,
            "flip_rate": sum(p["flipped"] for p in decided) / max(1, len(decided)),
            "sign_test_p": sign_test_p(wins_b, wins_a), "winner": winner,
            "pair_calls": sum(len(o["keys"]) for p in pairs for o in p["orders"].values()),
            "pair_cost_usd": sum(o["cost_usd"] for p in pairs for o in p["orders"].values())}


def generate(gen_cfg, items, llm, tag="gen:"):
    """items: dicts with id, input, system_prompt?, expected_output?. Generator system prompt = config prompt + item constraint."""
    out = []
    for it in items:
        system = "\n".join(x for x in (gen_cfg.get("system_prompt"), it.get("system_prompt")) if x)
        r = llm.generate(gen_cfg["model"], it["input"], system=system, max_output_tokens=gen_cfg["max_output_tokens"],
                         temperature=gen_cfg["temperature"], seed=gen_cfg["seed"], tag=f"{tag}{gen_cfg['name']}:{it['id']}")
        out.append(TestCase(id=it["id"], input=it["input"], system_prompt=system, model_output=r["text"].strip(),
                            expected_output=it.get("expected_output"),
                            meta={"generator": gen_cfg["name"], "model": gen_cfg["model"], "family": gen_cfg["family"]}))
    return out


def write(name, obj, md=None):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    if md:
        (RESULTS / f"{name}.md").write_text(md, encoding="utf-8")
