"""Compact bias + validation + A/B experiments: one probe suite, judge v0 (naive) vs v1 (mitigated), same model.

Runs (all cached in cache/calls.jsonl, resumable, replayable with --offline):
  P  pointwise on 48 probe outputs (12 items x T/P/VW/CW) x {v0, v1}
  Q  pairwise on 24 pairs (T-vs-VW, T-vs-P) x both orders x {v0, v1}
  S  self-preference: 6 open questions, Gemini vs Gemma originals and cross-paraphrases, both orders x {v0, v1}
  R  test-retest: v1 pointwise on 16 outputs x 2 extra reruns (temperature 0)
  AB generator prompt gen_v1 vs gen_v2 (gemma-4-26b-a4b-it) on the 12 base inputs, judged by v1 (pairwise both orders + pointwise)
"""
import json

import yaml

from .judge import judge_pair, judge_pointwise
from .pipeline import ROOT, compare, generate, load_cfg, provenance, write
from .schema import TestCase, save_suite
from .stats import auroc, bootstrap_ci, cohen_kappa, norm_entropy, spearman

SUITES = ROOT / "suites"
PAD = ("Thank you for your question. {t} To elaborate, this is the answer to the question exactly as it was asked, and it "
       "reflects the information that was requested. It is worth keeping in mind that context can matter in many "
       "situations, so it is always a good idea to consider the details carefully. In summary: {t}")
AB_EXTRA = [
    {"id": "x1", "system_prompt": "Answer in exactly two sentences.", "input": "Explain what a vector database is."},
    {"id": "x2", "system_prompt": "Use at most 3 bullet points.", "input": "Give tips for writing a strong password."},
    {"id": "x3", "system_prompt": "Reply in under 20 words.", "input": "Why is the sky blue?",
     "expected_output": "Sunlight is scattered by air molecules (Rayleigh scattering), and blue light is scattered most."}]
SELFPREF_INPUTS = ["Explain what prompt injection is and give one mitigation.",
                   "What are the trade-offs of a managed vector database versus a self-hosted one?",
                   "Explain the difference between precision and recall.",
                   "Why do large language models sometimes hallucinate?",
                   "Give three tips for writing clear technical documentation.",
                   "What is retrieval-augmented generation and why is it used?"]
SELFPREF_N = 4
SP_SYSTEM = "Answer concisely in 3-5 sentences."
PARAPHRASE = ("Rewrite the following answer in your own words. Keep every fact and claim exactly the same; do not add, "
              "remove or change any information. Output only the rewritten answer.\n\n<answer>\n{t}\n</answer>")
RETEST_ITEMS = {"f1", "m1", "c1", "f2"}


def base_items():
    return yaml.safe_load((SUITES / "base_items.yaml").read_text(encoding="utf-8"))["items"]


def build_probes(items):
    cases, pairs = [], []
    for it in items:
        sp, ref, i = it.get("system_prompt", ""), it["expected_output"], it["id"]
        mk = lambda v, out, gold, inp=None, **m: TestCase(id=f"{i}-{v}", input=inp or it["input"], system_prompt=sp, model_output=out,
                                                          expected_output=ref, meta={"item": i, "variant": v, "gold": gold, "category": it["category"], **m})
        t, p = mk("T", it["terse_correct"], "pass"), mk("P", PAD.format(t=it["terse_correct"]), "n/a")
        vw = mk("VW", it["verbose_wrong"], "fail")
        cw = mk("CW", it["confident_wrong"], "fail", it.get("cw_input"), false_premise="cw_input" in it)
        cases += [t, p, vw, cw]
        pairs += [{"id": f"{i}-T-vs-VW", "kind": "T_vs_VW", "input": it["input"], "system_prompt": sp, "reference": ref,
                   "out1": t.model_output, "out2": vw.model_output, "gold": "1"},
                  {"id": f"{i}-T-vs-P", "kind": "T_vs_P", "input": it["input"], "system_prompt": sp, "reference": ref,
                   "out1": t.model_output, "out2": p.model_output, "gold": "1_or_tie"}]
    save_suite(cases, SUITES / "probes.jsonl")
    (SUITES / "pairs.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n", encoding="utf-8")
    return cases, pairs


def selfpref_pairs(llm):
    gem = {"name": "gen_gemini", "model": "gemini-3.1-flash-lite", "family": "gemini", "temperature": 0, "seed": 42,
           "max_output_tokens": 1024, "system_prompt": SP_SYSTEM}
    gma = {**load_cfg("gen_v1"), "name": "gen_gemma", "model": "gemma-4-31b-it", "system_prompt": SP_SYSTEM}
    items = [{"id": f"s{k + 1}", "input": q} for k, q in enumerate(SELFPREF_INPUTS[:SELFPREF_N])]
    g_orig, m_orig = generate(gem, items, llm, "sp:"), generate(gma, items, llm, "sp:")
    para = lambda cfg, case: generate({**cfg, "system_prompt": None}, [{"id": case.id, "input": PARAPHRASE.format(t=case.model_output)}],
                                      llm, "sp:para:")[0].model_output
    pairs = []
    for g, m in zip(g_orig, m_orig):
        pairs.append({"id": f"{g.id}-X", "set": "X", "input": g.input, "out1": g.model_output, "out2": para(gma, g)})
        pairs.append({"id": f"{g.id}-Y", "set": "Y", "input": g.input, "out1": m.model_output, "out2": para(gem, m)})
    (SUITES / "selfpref.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n", encoding="utf-8")
    return pairs


def ab_suites(llm, items):
    inputs = [{"id": it["id"], "input": it["input"], "system_prompt": it.get("system_prompt", ""),
               "expected_output": it["expected_output"]} for it in items]
    a, b = generate(load_cfg("gen_v1"), inputs, llm, "ab:"), generate(load_cfg("gen_v2"), inputs, llm, "ab:")
    save_suite(a, SUITES / "ab_gen_v1.jsonl")
    save_suite(b, SUITES / "ab_gen_v2.jsonl")
    return a, b


def _orders(recs):
    return [o for r in recs for o in r["orders"].values() if o["candidate"] is not None]


def pointwise_metrics(recs):
    ok = [r for r in recs if "passed" in r]
    by = lambda v: [r for r in ok if r["variant"] == v]
    gold = [r for r in ok if r["gold"] in ("pass", "fail")]
    raw = [r["raw_score"] if "raw_score" in r else round(r["weighted"]) for r in ok]
    levels = 10 if "raw_score" in ok[0] else 5
    tp = {r["item"]: r for r in by("T")}
    return {
        "n": len(recs), "parse_failed": len(recs) - len(ok),
        "gold_accuracy": sum(r["passed"] == (r["gold"] == "pass") for r in gold) / len(gold),
        "gold_kappa": cohen_kappa([r["passed"] for r in gold], [r["gold"] == "pass" for r in gold]),
        "n_gold": len(gold),
        "T_failed_rate": sum(not r["passed"] for r in by("T")) / max(1, len(by("T"))),
        "VW_passed_rate": sum(r["passed"] for r in by("VW")) / max(1, len(by("VW"))),
        "CW_passed_rate": sum(r["passed"] for r in by("CW")) / max(1, len(by("CW"))),
        "CW_false_premise_passed": f"{sum(r['passed'] for r in by('CW') if r.get('false_premise'))}/{sum(1 for r in by('CW') if r.get('false_premise'))}",
        "pad_minus_terse": bootstrap_ci([p["score_norm"] - tp[p["item"]]["score_norm"] for p in by("P") if p["item"] in tp]),
        "length_spearman_correct_only": spearman([r["len_chars"] for r in by("T") + by("P")], [r["score_norm"] for r in by("T") + by("P")]),
        "norm_entropy": norm_entropy(raw, levels), "distinct_levels": len(set(raw)), "scale_levels": levels,
        "auroc_correct_vs_wrong": auroc([r["score_norm"] for r in by("T")], [r["score_norm"] for r in by("VW") + by("CW")]),
        "ungrounded_correctness": sum(1 for r in ok if r.get("grounded_correctness") is False)}


def pairwise_metrics(recs):
    good = [r for r in recs if r["final"] is not None]
    od = _orders(recs)
    corr = lambda r, v: (v == 1) if r["gold"] == "1" else (v in (1, 0))
    k = lambda kind: [r for r in good if r["kind"] == kind]
    return {"n": len(recs), "parse_failed": len(recs) - len(good),
            "flip_rate": sum(r["flipped"] for r in good) / len(good),
            "slot1_win_rate": sum(o["slot"] == "A" for o in od) / max(1, sum(o["slot"] in ("A", "B") for o in od)),
            "single_order_accuracy": sum(corr(r, r["single_order"]) for r in good) / len(good),
            "swap_final_accuracy": sum(corr(r, r["final"]) for r in good) / len(good),
            "ties_from_disagreement": sum(r["flipped"] for r in good) / len(good),
            "padded_preferred_single_order": sum(r["single_order"] == 2 for r in k("T_vs_P")) / max(1, len(k("T_vs_P"))),
            "padded_preferred_final": sum(r["final"] == 2 for r in k("T_vs_P")) / max(1, len(k("T_vs_P"))),
            "verbose_wrong_preferred_single_order": sum(r["single_order"] == 2 for r in k("T_vs_VW")) / max(1, len(k("T_vs_VW"))),
            "verbose_wrong_preferred_final": sum(r["final"] == 2 for r in k("T_vs_VW")) / max(1, len(k("T_vs_VW"))),
            "ungrounded_winner_orders": sum(o["ungrounded"] for o in od)}


def selfpref_delta(recs):
    def p_orig(s):
        od = [o["candidate"] for r in recs if r["set"] == s for o in r["orders"].values() if o["candidate"] is not None]
        return sum({1: 1.0, 0: 0.5, 2: 0.0}[c] for c in od) / max(1, len(od))
    px, py = p_orig("X"), p_orig("Y")
    return {"P_X_gemini_original_wins": px, "P_Y_gemma_original_wins": py, "delta_self_preference": (px - py) / 2}


def retest(recs0, reruns):
    by = {}
    for r in recs0 + reruns:
        if "passed" in r and r["item"] in RETEST_ITEMS:
            by.setdefault(r["id"], []).append(r)
    flips = sum(len({x["passed"] for x in v}) > 1 for v in by.values())
    spread = [max(x["weighted"] for x in v) - min(x["weighted"] for x in v) for v in by.values()]
    return {"outputs": len(by), "runs_each": 3, "temperature": 0, "verdict_flip_rate": flips / max(1, len(by)),
            "mean_weighted_score_range": sum(spread) / max(1, len(spread))}


def run_all(llm, stage="all"):
    """stage: 'gen' = generations only; 'core' = P, Q, R judge runs only; 'bias' = P, Q, S, R + bias/validation reports
    (no A/B, so it does not need the A/B generator); 'all' = everything + all reports."""
    v0, v1 = load_cfg("judge_v0"), load_cfg("judge_v1")
    items = base_items()
    cases, pairs = build_probes(items)
    if stage == "gen":
        selfpref_pairs(llm), ab_suites(llm, items)
        return None, None
    if stage == "core":
        for c in (v0, v1):
            [judge_pointwise(x, c, llm, tag="P:") for x in cases]
            [judge_pair(p["id"], p["input"], p["system_prompt"], p["reference"], p["out1"], p["out2"], c, llm, tag="Q:",
                        meta={"kind": p["kind"], "gold": p["gold"]}) for p in pairs]
        [judge_pointwise(x, v1, llm, rerun_idx=i, tag="R:") for i in (1, 2) for x in cases if x.meta["item"] in RETEST_ITEMS]
        return None, None
    sp_pairs = selfpref_pairs(llm)
    ab_a, ab_b = ab_suites(llm, items) if stage == "all" else (None, None)
    P = {c["prompt_version"]: [judge_pointwise(x, c, llm, tag="P:") for x in cases] for c in (v0, v1)}
    Q = {c["prompt_version"]: [judge_pair(p["id"], p["input"], p["system_prompt"], p["reference"], p["out1"], p["out2"], c, llm,
                                          tag="Q:", meta={"kind": p["kind"], "gold": p["gold"]}) for p in pairs] for c in (v0, v1)}
    S = {c["prompt_version"]: [judge_pair(p["id"], p["input"], "", None, p["out1"], p["out2"], c, llm, tag="S:", meta={"set": p["set"]})
                               for p in sp_pairs] for c in (v0, v1)}
    R = [judge_pointwise(x, v1, llm, rerun_idx=i, tag="R:") for i in (1, 2) for x in cases if x.meta["item"] in RETEST_ITEMS]
    AB = compare(ab_a, ab_b, v1, llm) if stage == "all" else None
    m = {"pointwise": {v: pointwise_metrics(P[v]) for v in P}, "pairwise": {v: pairwise_metrics(Q[v]) for v in Q},
         "self_preference": {v: selfpref_delta(S[v]) for v in S}, "test_retest_v1": retest(P["v1"], R)}
    prov = provenance(v1, llm)
    write("experiments_raw", {"provenance": prov, "P": P, "Q": Q, "S": S, "R": R})
    write("bias_validation", {"provenance": prov, "metrics": m}, bias_md(m) + "\n" + validation_md(m))
    if AB:
        write("ab_report", {"provenance": prov, **{k: v for k, v in AB.items() if k not in ("pairs", "pointwise_a", "pointwise_b")},
                            "pairs": AB["pairs"], "pointwise_a": AB["pointwise_a"], "pointwise_b": AB["pointwise_b"]}, ab_md(AB))
    write("report", {"provenance": prov, "suite": "suites/probes.jsonl", "judge": "judge_v1",
                     "summary": __import__("judge.pipeline", fromlist=["summarize_pointwise"]).summarize_pointwise(P["v1"])},
          suite_md("suites/probes.jsonl (48 probe outputs)", P["v1"]))
    return m, AB


pct = lambda x: "n/a" if x is None else f"{100 * x:.0f}%"
num = lambda x: "n/a" if x is None else f"{x:.2f}"


def bias_md(m):
    a, b = m["pointwise"]["v0"], m["pointwise"]["v1"]
    qa, qb = m["pairwise"]["v0"], m["pairwise"]["v1"]
    sa, sb = m["self_preference"]["v0"], m["self_preference"]["v1"]
    return f"""# Bias handling [MEASURED]: judge v0 (naive) -> v1 (mitigated)

Same judge model (`{load_cfg('judge_v1')['model']}`), same data; only the prompt and the code-side mitigations change.
Probe suite: 12 items x {{T terse-correct, P padded-correct, VW verbose-wrong, CW confident-wrong}}; 24 pairs judged in both orders.

| Bias | Mitigation implemented (in code) | Metric & result (before v0 -> after v1) |
|---|---|---|
| Position (A/B order) | every pair judged AB **and** BA; final verdict only if both orders agree, else tie (`judge.judge_pair`) | flip rate {pct(qa['flip_rate'])} -> {pct(qb['flip_rate'])}; slot-1 win rate {pct(qa['slot1_win_rate'])} -> {pct(qb['slot1_win_rate'])} (ideal 50%); accuracy vs gold: v0 single-order {pct(qa['single_order_accuracy'])} -> v1 swap-aggregated {pct(qb['swap_final_accuracy'])} |
| Verbosity / length | rubric penalizes content-free length under instruction_following; padded-answer probe; "do not reward length" rule | padded-minus-terse score {num(a['pad_minus_terse'][0])} -> {num(b['pad_minus_terse'][0])}; padded preferred over terse (pairwise) {pct(qa['padded_preferred_single_order'])} -> {pct(qb['padded_preferred_final'])}; verbose-wrong preferred over terse-correct {pct(qa['verbose_wrong_preferred_single_order'])} -> {pct(qb['verbose_wrong_preferred_final'])} |
| Self-enhancement | `family` on every config; `guard_family` refuses judge.family == generator.family; A/B generator is Gemma (judge is Gemini) | self-preference delta {num(sa['delta_self_preference'])} -> {num(sb['delta_self_preference'])} (0 = no preference for Gemini-written text; {SELFPREF_N} items x 2 sets x 2 orders) |
| Sycophancy / style | per-criterion verbatim evidence, correctness quote verified in code (unverified -> fail); reference block; pairwise decisive quote verified (else tie) | confident-wrong passed {pct(a['CW_passed_rate'])} -> {pct(b['CW_passed_rate'])} (false-premise subset {a['CW_false_premise_passed']} -> {b['CW_false_premise_passed']}); verbose-wrong passed {pct(a['VW_passed_rate'])} -> {pct(b['VW_passed_rate'])} |
| Score clustering | 1-5 anchors per criterion + 3 few-shot calibration examples; pairwise for comparisons | normalized entropy {num(a['norm_entropy'])} -> {num(b['norm_entropy'])} ({a['distinct_levels']}/{a['scale_levels']} -> {b['distinct_levels']}/{b['scale_levels']} levels used); AUROC correct vs wrong {num(a['auroc_correct_vs_wrong'])} -> {num(b['auroc_correct_vs_wrong'])} |

Parse failures: pointwise {a['parse_failed']}/{a['n']} -> {b['parse_failed']}/{b['n']}, pairwise {qa['parse_failed']}/{qa['n']} -> {qb['parse_failed']}/{qb['n']}.
Self-preference raw: P(Gemini original beats Gemma paraphrase) {num(sa['P_X_gemini_original_wins'])} -> {num(sb['P_X_gemini_original_wins'])}; P(Gemma original beats Gemini paraphrase) {num(sa['P_Y_gemma_original_wins'])} -> {num(sb['P_Y_gemma_original_wins'])}.
Limitation: Gemma and Gemini share a vendor (Google), so the "different family" mitigation is partial; n is small, so differences are indicative.
"""


def validation_md(m):
    a, b, qa, qb, r = m["pointwise"]["v0"], m["pointwise"]["v1"], m["pairwise"]["v0"], m["pairwise"]["v1"], m["test_retest_v1"]
    return f"""# Judge validation [MEASURED]

| Method | Result (v0 -> v1) | Notes |
|---|---|---|
| Agreement with gold labels (pointwise pass/fail) | {pct(a['gold_accuracy'])} -> {pct(b['gold_accuracy'])} | n = {b['n_gold']} outputs with constructed gold (T pass, VW/CW fail) |
| Cohen's kappa vs gold | {num(a['gold_kappa'])} -> {num(b['gold_kappa'])} | same outputs |
| Pairwise agreement with gold winner | {pct(qa['single_order_accuracy'])} (single order) -> {pct(qb['swap_final_accuracy'])} (swap-aggregated) | 24 pairs; T-vs-P counts tie or terse as correct |
| Test-retest flip rate (v1) | {pct(r['verdict_flip_rate'])} pass/fail flips | {r['outputs']} outputs x {r['runs_each']} runs at temperature {r['temperature']}; mean weighted-score range {num(r['mean_weighted_score_range'])} |
| Adversarial: verbose-but-wrong passed? | {pct(a['VW_passed_rate'])} -> {pct(b['VW_passed_rate'])} | fooled = passed |
| Adversarial: terse-but-correct failed? | {pct(a['T_failed_rate'])} -> {pct(b['T_failed_rate'])} | fooled = failed |
| Adversarial: confident-wrong passed? | {pct(a['CW_passed_rate'])} -> {pct(b['CW_passed_rate'])} | includes 4 false-premise (sycophancy) items |
"""


def ab_md(ab):
    sa, sb, wr = ab["summary_a"], ab["summary_b"], ab["win_rate_b"]
    f = lambda t: "n/a" if t[0] is None else f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    return f"""# A/B comparison [MEASURED]: generator prompt gen_v1 (A) vs gen_v2 (B)

Generator `{load_cfg('gen_v1')['model']}` (family gemma) for both arms; judge v1 (`{load_cfg('judge_v1')['model']}`, family gemini). {len(ab['pairs'])} inputs.
Decision rule (fixed before the run): B wins if the 95% CI of B's win rate is above 0.5, A if below, otherwise keep A.

| Config | Pass rate | Mean score (0-1) | Win rate |
|---|---|---|---|
| Config A - gen_v1 ("Answer the user's request.") | {f(sa['pass_rate'])} | {f(sa['mean_score_norm'])} | {1 - wr[0]:.2f} |
| Config B - gen_v2 (constraint-following + fact-check prompt) | {f(sb['pass_rate'])} | {f(sb['mean_score_norm'])} | {f(wr)} |

Pairwise (both orders, agreement rule): B wins {ab['wins_b']}, A wins {ab['wins_a']}, ties {ab['ties']}; order flip rate {100 * ab['flip_rate']:.0f}%; sign test p = {ab['sign_test_p']:.3f}.
**Declared winner: {ab['winner']}.**
Judge cost: pairwise {ab['pair_calls']} calls ${ab['pair_cost_usd']:.4f}; pointwise A {sa['calls']} calls ${sa['cost_usd']:.4f}, B {sb['calls']} calls ${sb['cost_usd']:.4f}.
"""


def suite_md(name, recs):
    s = __import__("judge.pipeline", fromlist=["summarize_pointwise"]).summarize_pointwise(recs)
    crit = "\n".join(f"| {c} | {v:.2f} |" for c, v in s.get("mean_by_criterion", {}).items())
    return f"""# Suite report [MEASURED]: {name}, judge v1

| Metric | Value |
|---|---|
| Cases | {s['n']} |
| Parse failures | {s['parse_failed']} |
| Pass rate | {s['pass_rate'][0]:.2f} [{s['pass_rate'][1]:.2f}, {s['pass_rate'][2]:.2f}] |
| Mean score (0-1) | {s['mean_score_norm'][0]:.2f} [{s['mean_score_norm'][1]:.2f}, {s['mean_score_norm'][2]:.2f}] |
| Judge calls / cost | {s['calls']} / ${s['cost_usd']:.4f} |
| Tokens (prompt / output / thoughts) | {s['tokens']['prompt']} / {s['tokens']['output']} / {s['tokens']['thoughts']} |

| Criterion | Mean score (1-5) |
|---|---|
{crit}
"""


def interim_report(model):
    """Offline-only report from whatever is cached for ONE judge model (no API calls, no cross-model mixing).
    Pointwise v0 vs v1 is compared only on case ids present for both versions."""
    from .pipeline import make_llm
    llm = make_llm(offline=True)
    cases, pairs = build_probes(base_items())
    cfg = {v: {**load_cfg(f"judge_{v}"), "model": model} for v in ("v0", "v1")}

    def cached(fn):
        try:
            return fn()
        except KeyError:
            return None
    P = {v: [r for x in cases if (r := cached(lambda: judge_pointwise(x, cfg[v], llm, tag="P:")))] for v in cfg}
    Q = {v: [r for p in pairs if (r := cached(lambda: judge_pair(p["id"], p["input"], p["system_prompt"], p["reference"], p["out1"],
             p["out2"], cfg[v], llm, tag="Q:", meta={"kind": p["kind"], "gold": p["gold"]})))] for v in cfg}
    common = {r["id"] for r in P["v0"]} & {r["id"] for r in P["v1"]}
    sub = {v: [r for r in P[v] if r["id"] in common] for v in P}
    cov = {"P_v0": f"{len(P['v0'])}/{len(cases)}", "P_v1": f"{len(P['v1'])}/{len(cases)}", "Q_v0": f"{len(Q['v0'])}/{len(pairs)}",
           "Q_v1": f"{len(Q['v1'])}/{len(pairs)}", "S": "0 (not run)", "R": "0 (not run)", "AB": "0 (not run)"}
    m = {"coverage": cov, "common_pointwise_ids": sorted(common),
         "pointwise_v0_full": pointwise_metrics(P["v0"]) if P["v0"] else None,
         "pairwise_v0_full": pairwise_metrics(Q["v0"]) if Q["v0"] else None,
         "pointwise_common": {v: pointwise_metrics(sub[v]) for v in sub} if common else None,
         "pairwise_v1": pairwise_metrics(Q["v1"]) if Q["v1"] else None}
    assert llm.calls == 0
    a, q = m["pointwise_v0_full"], m["pairwise_v0_full"]
    c0, c1 = (m["pointwise_common"] or {}).get("v0"), (m["pointwise_common"] or {}).get("v1")
    row = lambda name, k, f: f"| {name} | {f(c0[k])} | {f(c1[k])} |" if c0 and c1 else ""
    md = f"""# INTERIM report [MEASURED, INCOMPLETE]: judge `{model}`

**Not the final result.** This judge was replaced after repeated 503 errors. Its run stopped part-way, so this report uses
only the calls that were cached before it stopped, and only this one model (no cross-model mixing). No API calls were made to produce it.

| Run | Coverage |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in cov.items()) + f"""

## v0 (naive) baseline: complete for P and Q

| Metric | v0 |
|---|---|
| Gold agreement (pointwise, T/VW/CW) | {pct(a['gold_accuracy'])} (kappa {num(a['gold_kappa'])}, n = {a['n_gold']}) |
| Verbose-wrong passed / confident-wrong passed / terse-correct failed | {pct(a['VW_passed_rate'])} / {pct(a['CW_passed_rate'])} / {pct(a['T_failed_rate'])} |
| Padded-minus-terse score | {num(a['pad_minus_terse'][0])} |
| Score clustering: normalized entropy, levels used, AUROC correct vs wrong | {num(a['norm_entropy'])}, {a['distinct_levels']}/{a['scale_levels']}, {num(a['auroc_correct_vs_wrong'])} |
| Pairwise flip rate (AB vs BA) / slot-1 win rate | {pct(q['flip_rate'])} / {pct(q['slot1_win_rate'])} (n = {q['n']} pairs) |
| Pairwise accuracy vs gold: single order / swap-aggregated | {pct(q['single_order_accuracy'])} / {pct(q['swap_final_accuracy'])} |
| Padded preferred over terse / verbose-wrong preferred (single order) | {pct(q['padded_preferred_single_order'])} / {pct(q['verbose_wrong_preferred_single_order'])} |

## Pointwise v0 vs v1 on the {len(common)} cases cached for both (items {", ".join(sorted({i.split('-')[0] for i in common}))})

| Metric | v0 | v1 |
|---|---|---|
{row("Gold agreement", "gold_accuracy", pct)}
{row("Cohen's kappa vs gold", "gold_kappa", num)}
{row("Verbose-wrong passed", "VW_passed_rate", pct)}
{row("Confident-wrong passed", "CW_passed_rate", pct)}
{row("AUROC correct vs wrong", "auroc_correct_vs_wrong", num)}
{row("Normalized entropy", "norm_entropy", num)}

Not available from cached data: pairwise v1 (position bias after mitigation: {cov['Q_v1']} pairs), self-preference (S),
test-retest (R), and the A/B comparison. The final results need the live runs with the current judge.
"""
    write(f"interim_{model}", {"kind": "measured_incomplete", "judge_model": model, "metrics": m}, md)
    return m


def _missing(fn):
    try:
        fn()
        return False
    except KeyError:
        return True


def availability_probe(online, gen_online, offline):
    """One uncached P v1 judge call + one uncached A/B generation. Both are real suite calls, so a success is cached and
    reused (no wasted quota). Returns {"judge": bool, "generator": bool, ...errors}."""
    v1, out = load_cfg("judge_v1"), {}
    cases, _ = build_probes(base_items())
    case = next((x for x in cases if _missing(lambda: judge_pointwise(x, v1, offline, tag="P:"))), None)
    try:
        out["judge"] = case is None or "passed" in judge_pointwise(case, v1, online, tag="P:")
        out["judge_case"] = case.id if case else "all P v1 cached"
    except Exception as e:
        out.update(judge=False, judge_error=f"{type(e).__name__}: {str(e)[:160]}")
    items = [{"id": it["id"], "input": it["input"], "system_prompt": it.get("system_prompt", ""),
              "expected_output": it["expected_output"]} for it in base_items()]
    todo = next(((cfg, it) for cfg in (load_cfg("gen_v1"), load_cfg("gen_v2")) for it in items
                 if _missing(lambda: generate(cfg, [it], offline, "ab:"))), None)
    try:
        out["generator"] = todo is None or bool(generate(todo[0], [todo[1]], gen_online, "ab:")[0].model_output.strip())
        out["generator_case"] = f"{todo[0]['name']}:{todo[1]['id']}" if todo else "all A/B cached"
    except Exception as e:
        out.update(generator=False, generator_error=f"{type(e).__name__}: {str(e)[:160]}")
    return out


def usage_report():
    """Judge usage per run/version from the judge cache (offline): calls, tokens, latency, cost."""
    import numpy as np
    from .pipeline import ROOT
    model = load_cfg("judge_v1")["model"]
    rows = [json.loads(l) for l in (ROOT / "cache" / f"judge_{model}.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = {}
    for e in rows:
        run, ver = e["tag"].split(":")[0], ("repair" if e["tag"].endswith("repair") else e["tag"].split(":")[-1])
        groups.setdefault((run, ver), []).append(e)
    lines = []
    for (run, ver), es in sorted(groups.items()):
        lat = [e["latency_ms"] for e in es]
        lines.append(f"| {run} | {ver} | {len(es)} | {sum(e['usage']['prompt'] for e in es)} | {sum(e['usage']['output'] for e in es)} | "
                     f"{sum(e['usage']['thoughts'] for e in es)} | {np.percentile(lat, 50):.0f} | {np.percentile(lat, 95):.0f} | "
                     f"{sum(e['cost_usd'] for e in es):.4f} |")
    total = sum(e["cost_usd"] for e in rows)
    md = (f"# Judge usage [MEASURED]: `{model}` (from `cache/judge_{model}.jsonl`)\n\n"
          "| Run | Version | Calls | Prompt tokens | Output tokens | Thinking tokens | Latency p50 (ms) | Latency p95 (ms) | Cost ($) |\n"
          "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(lines) +
          f"\n\nTotal: {len(rows)} calls, ${total:.4f}. Cost = API usage metadata x list price "
          "($0.25 / $1.50 per 1M input / output tokens, thinking billed as output).\n")
    write("usage", {"judge_model": model, "calls": len(rows), "cost_usd": total}, md)
    return md
