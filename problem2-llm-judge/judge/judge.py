"""Pointwise and pairwise judging with parse fallback + one repair retry, grounding checks and order swap."""
from . import prompts as P
from .parse import parse_verdict
from .schema import CRITERIA, RUBRIC, PairwiseV0, PairwiseV1, PointwiseV0, PointwiseV1

S, E = {"type": "string"}, {"type": "string", "enum": ["A", "B", "tie"]}


def _obj(props, order=None):
    return {"type": "object", "properties": props, "required": list(props), "propertyOrdering": order or list(props)}


def schema_for(model_cls, crit):
    """Explicit response schemas (a bare {"type": "object"} made the model return an empty object).
    Rationale/evidence come before scores so the judge reasons first. Pydantic still validates every reply."""
    if model_cls is PointwiseV0:
        return _obj({"rationale": S, "score": {"type": "integer"}})
    if model_cls is PointwiseV1:
        c = _obj({"evidence": S, "rationale": S, "score": {"type": "integer"}})
        return _obj({"criteria": _obj({k: c for k in crit}), "overall_rationale": S})
    if model_cls is PairwiseV0:
        return _obj({"rationale": S, "winner": E})
    return _obj({"per_criterion": _obj({k: E for k in crit}), "rationale": S, "decisive_quote": S, "winner": E})


def _norm(s):
    return " ".join((s or "").replace("’", "'").replace("“", '"').replace("”", '"').lower().split())


def grounded(quote, *texts):
    q = _norm(quote)
    return bool(q) and any(q in _norm(t) for t in texts)


def call(llm, cfg, prompt, model_cls, tag, rerun_idx=0, check=None, crit=CRITERIA):
    """-> (verdict | None, meta). `check(verdict)` returns an error string to treat a parsed verdict as invalid."""
    kw = dict(system=P.SYSTEM, schema=schema_for(model_cls, crit), thinking_level=cfg.get("thinking_level"),
              max_output_tokens=cfg["max_output_tokens"], temperature=cfg["temperature"], seed=cfg["seed"],
              rerun_idx=rerun_idx)
    r = llm.generate(cfg["model"], prompt, tag=tag, **kw)
    meta = {"keys": [r["key"]], "cost_usd": r["cost_usd"], "tokens": dict(r["usage"]), "cached": r["cached"]}
    v, step, err = parse_verdict(r["text"], model_cls)
    err = err or (check(v) if check else None)
    if err:  # one repair retry, logged as its own call
        r2 = llm.generate(cfg["model"], prompt + f"\n\nYour previous reply was invalid ({err}). Reply again with ONLY the JSON object.",
                          tag=tag + ":repair", **kw)
        meta["keys"].append(r2["key"])
        meta["cost_usd"] += r2["cost_usd"]
        v, step2, err = parse_verdict(r2["text"], model_cls)
        err = err or (check(v) if check else None)
        step = "parse_failed" if err else f"retry:{step2}"
    meta.update(parse=step, error=err)
    return (None if err else v), meta


def judge_pointwise(case, cfg, llm, rerun_idx=0, tag=""):
    version = cfg["prompt_version"]
    rec = {"id": case.id, "version": version, "rerun_idx": rerun_idx, "len_chars": len(case.model_output), **case.meta}
    if version == "v0":
        v, meta = call(llm, cfg, P.pointwise_v0(case), PointwiseV0, f"{tag}pw:{case.id}:v0", rerun_idx)
        if v:  # pass bar at the same normalized level as v1's 3.5/5
            rec.update(raw_score=v.score, score_norm=(v.score - 1) / 9, passed=v.score >= 7, rationale=v.rationale)
        return {**rec, **meta}
    crit = case.criteria or CRITERIA
    missing = lambda v: (f"missing criteria {sorted(set(crit) - set(v.criteria))}" if set(crit) - set(v.criteria) else None)
    v, meta = call(llm, cfg, P.pointwise_v1(case, crit), PointwiseV1, f"{tag}pw:{case.id}:v1", rerun_idx, check=missing,
                   crit=crit)
    if v:
        scores = {c: v.criteria[c].score for c in crit}
        w = {c: RUBRIC["criteria"][c]["weight"] for c in crit}
        weighted = sum(w[c] * scores[c] for c in crit) / sum(w.values())
        rule = RUBRIC["pass_rule"]
        g = grounded(v.criteria["correctness"].evidence, case.model_output) if "correctness" in crit else True
        passed = (weighted >= rule["min_weighted"] and scores.get("correctness", 5) >= rule["min_correctness"]
                  and scores.get("tone_safety", 5) >= rule["min_tone_safety"] and (g or not rule["require_grounded_correctness"]))
        rec.update(scores=scores, weighted=weighted, score_norm=(weighted - 1) / 4, grounded_correctness=g, passed=passed,
                   evidence={c: v.criteria[c].evidence for c in crit}, rationale=v.overall_rationale)
    return {**rec, **meta}


def judge_pair(pid, inp, system_prompt, reference, out1, out2, cfg, llm, rerun_idx=0, tag="", meta=None):
    """Both orders. Candidate ids: 1 = out1, 2 = out2, 0 = tie. Final verdict requires AB/BA agreement, else tie."""
    version, crit = cfg["prompt_version"], CRITERIA
    rec = {"id": pid, "version": version, "rerun_idx": rerun_idx, **(meta or {}), "orders": {}}
    for order in ("AB", "BA"):
        a, b = (out1, out2) if order == "AB" else (out2, out1)
        if version == "v0":
            v, m = call(llm, cfg, P.pairwise_v0(inp, system_prompt, reference, a, b), PairwiseV0, f"{tag}pair:{pid}:{order}:v0", rerun_idx)
        else:
            v, m = call(llm, cfg, P.pairwise_v1(inp, system_prompt, reference, a, b, crit), PairwiseV1, f"{tag}pair:{pid}:{order}:v1", rerun_idx)
        slot = v.winner if v else None
        cand = {"A": 1 if order == "AB" else 2, "B": 2 if order == "AB" else 1, "tie": 0}.get(slot)
        ungrounded = bool(version == "v1" and v and cand and not grounded(v.decisive_quote, out1 if cand == 1 else out2))
        if ungrounded:  # winner not backed by a verbatim quote from the winning response -> no decision
            cand = 0
        rec["orders"][order] = {"slot": slot, "candidate": cand, "ungrounded": ungrounded, **m}
    ab, ba = rec["orders"]["AB"]["candidate"], rec["orders"]["BA"]["candidate"]
    rec.update(single_order=ab, flipped=(ab is not None and ba is not None and ab != ba),
               final=(ab if ab == ba else 0) if None not in (ab, ba) else None)
    return rec
