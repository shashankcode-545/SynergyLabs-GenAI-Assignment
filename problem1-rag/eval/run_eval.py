"""P1 evaluation harness. Run from problem1-rag/:

  python -m eval.run_eval calibrate     # tau from the 5 dev questions only -> eval/calibration.json
  python -m eval.run_eval retrieval     # IR metrics @k=1,3,5,10 for 3 chunk configs -> results/retrieval.{json,md}
  python -m eval.run_eval answers       # answer quality on the 18 eval questions -> results/answers.{json,md}
  python -m eval.run_eval all [--offline]   # --offline: replay from committed caches, no API key needed
"""
import argparse, json, subprocess, time
from dataclasses import replace

import numpy as np

from rag.answer import answer
from rag.config import ROOT, Settings
from rag.ingest import ingest
from rag.llm import Gemini
from rag.loaders import find_span, load_corpus, normalize
from rag.store import open_table, search
from .metrics import bootstrap_ci, contains_all, exact_match, ir_metrics, is_relevant, token_f1
from .validate_questions import DEV, EVAL, load

CORPUS, RESULTS = ROOT / "data/corpus", ROOT / "results"
CONFIGS = [(1200, 200), (2400, 400)]
KS = (1, 3, 5, 10)
JUDGE_SYSTEM = "You are a strict evaluator of answers produced by a retrieval-augmented QA system. Judge only against the given context."
JUDGE_SCHEMA = {"type": "object", "required": ["claims", "relevance", "relevance_rationale"], "properties": {
    "claims": {"type": "array", "items": {"type": "object", "required": ["claim", "supported", "evidence"], "properties": {
        "claim": {"type": "string"}, "supported": {"type": "boolean"}, "evidence": {"type": "string"}}}},
    "relevance": {"type": "integer", "minimum": 1, "maximum": 5}, "relevance_rationale": {"type": "string"}}}
JUDGE_PROMPT = """Context (the only allowed evidence):
{context}

Question: {question}
Answer: {answer}

1. Split the answer into atomic factual claims (ignore citation markers like [S1]). For each claim set supported=true only
   if the context states it, and copy the shortest verbatim quote from the context that supports it into evidence
   (empty string if unsupported).
2. relevance (1-5): how directly the answer addresses the question. 5 = fully answers it; 3 = partially or with
   unrequested material; 1 = does not address it."""


def provenance(s):
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {"kind": "measured", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": sha, "embed": f"{s.embed_model}@{s.embed_dim}",
            "gen_model": s.gen_model, "judge_model": s.judge_model, "min_similarity": s.min_similarity,
            "questions_sha256": __import__("hashlib").sha256(EVAL.read_bytes()).hexdigest()[:16]}


def gold_spans(q, docs):
    return [(e["doc_id"], *find_span(docs[e["doc_id"]].text, e["quote"])) for e in q["evidence"]]


def calibrate(s, llm, offline):
    table = open_table(s.db_path, s.table_name(), s.embed_dim)
    sims = [(q["id"], q["answerable"], search(table, llm.embed([q["question"]], s.embed_model, s.embed_dim, "RETRIEVAL_QUERY",
                                                               offline=offline)[0], 1)[0]["score"]) for q in load(DEV)]
    tau = min(x for _, ans, x in sims if ans) - 0.02
    acc = sum((x >= tau) == ans for _, ans, x in sims)
    out = {"min_similarity": round(tau, 4), "dev_gating_accuracy": f"{acc}/{len(sims)}",
           "dev_top1_similarity": [{"id": i, "answerable": a, "top1": round(x, 4)} for i, a, x in sims],
           "rule": "lowest answerable dev top-1 similarity minus 0.02 (recall-preserving pre-filter). Dev questions only."}
    (ROOT / "eval/calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def retrieval(s, llm, offline):
    docs = {d.doc_id: d for d in load_corpus(CORPUS)}
    qs = [q for q in load(EVAL) if q["answerable"]]
    res = {"provenance": provenance(s), "relevance_rule": "chunk covers >= 50% of a gold evidence span (same doc)",
           "configs": []}
    for size, overlap in CONFIGS:
        ingest(CORPUS, s, llm, size, overlap, offline=offline)
        table = open_table(s.db_path, s.table_name(size, overlap), s.embed_dim)
        rows = table.search().select(["doc_id", "char_start", "char_end"]).limit(100_000).to_list()
        per_q = []
        for q in qs:
            spans = gold_spans(q, docs)
            qv = llm.embed([q["question"]], s.embed_model, s.embed_dim, "RETRIEVAL_QUERY", offline=offline)[0]
            hits = search(table, qv, max(KS), q.get("filters"))
            ranked = [(h["doc_id"], h["char_start"], h["char_end"]) for h in hits]
            n_rel = sum(is_relevant((r["doc_id"], r["char_start"], r["char_end"]), spans) for r in rows)
            per_q.append({"id": q["id"], "type": q["type"], "n_relevant_in_index": n_rel,
                          "first_relevant_rank": next((i + 1 for i, c in enumerate(ranked) if is_relevant(c, spans)), None),
                          **{f"@{k}": ir_metrics(ranked, spans, n_rel, k) for k in KS},
                          "context_tokens@5": sum(h["char_end"] - h["char_start"] for h in hits[:5]) / 4})
        agg = {f"@{k}": {m: bootstrap_ci([p[f"@{k}"][m] for p in per_q]) for m in per_q[0]["@1"]} for k in KS}
        res["configs"].append({"chunk_size": size, "chunk_overlap": overlap, "vectors": table.count_rows(),
                               "mean_context_tokens@5": float(np.mean([p["context_tokens@5"] for p in per_q])),
                               "metrics": agg, "per_question": per_q})
    (RESULTS / "retrieval.json").write_text(json.dumps(res, indent=1))
    f = lambda t: f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"
    md = [f"# Retrieval evaluation [MEASURED]\n\n{len(qs)} answerable eval questions (filter-scoped ones use their filter). "
          f"Relevant = chunk covers >= 50% of a gold evidence span. Mean [95% bootstrap CI]. Provenance: `retrieval.json`.\n"]
    for c in res["configs"]:
        md.append(f"\n## chunk {c['chunk_size']}/{c['chunk_overlap']} chars - {c['vectors']} vectors - "
                  f"mean context @5 = {c['mean_context_tokens@5']:.0f} tokens (chars/4)\n\n| k | Hit/Recall | Recall (evidence) | MRR | nDCG | Context precision (P@k) | AP@k |\n|---|---|---|---|---|---|---|")
        md += [f"| {k} | {f(c['metrics'][f'@{k}']['hit'])} | {f(c['metrics'][f'@{k}']['recall'])} | {f(c['metrics'][f'@{k}']['mrr'])} | "
               f"{f(c['metrics'][f'@{k}']['ndcg'])} | {f(c['metrics'][f'@{k}']['precision'])} | {f(c['metrics'][f'@{k}']['ap'])} |" for k in KS]
    (RESULTS / "retrieval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def judge(s, llm, q, r, offline):
    ctx = "\n\n".join(f"[S{i}] {t}" for i, t in enumerate(r["context"], 1))
    j = llm.generate(s.judge_model, JUDGE_PROMPT.format(context=ctx, question=q["question"], answer=r["answer"]),
                     system=JUDGE_SYSTEM, schema=JUDGE_SCHEMA, thinking_level="LOW", max_output_tokens=2048, offline=offline)
    v = json.loads(j["text"])
    claims = v["claims"]
    ok = [c for c in claims if c["supported"] and c["evidence"].strip() and find_span(normalize(ctx), c["evidence"])]
    return {"faithfulness": len(ok) / len(claims) if claims else None, "relevance": (v["relevance"] - 1) / 4,
            "claims": claims, "verified_supported": len(ok), "judge_cost_usd": j["cost_usd"]}


def answers(s, llm, offline):
    docs = {d.doc_id: d for d in load_corpus(CORPUS)}
    table = open_table(s.db_path, s.table_name(), s.embed_dim)
    per_q, gen_live = [], []
    for q in load(EVAL):
        r = answer(q["question"], s, llm, table, k=s.top_k, filters=q.get("filters"), offline=offline, log=False)
        rec = {"id": q["id"], "type": q["type"], "answerable": q["answerable"], "abstained": r["abstained"],
               "reason": r["reason"], "answer": r["answer"], "gold": q["gold_answer"], "tokens": r["tokens"],
               "gen_cost_usd": r["cost_usd"], "latency_ms": r["latency_ms"]}
        if not r["abstained"] and not r.get("cached", True):
            gen_live.append(r["latency_ms"])
        if q["answerable"]:
            spans = gold_spans(q, docs)
            pred = "" if r["abstained"] else r["answer"]
            rec.update(em=exact_match(pred, q["gold_answer"]), f1=token_f1(pred, q["gold_answer"]),
                       contains_gold=contains_all(pred, q.get("gold_answer_parts") or [q["gold_answer"]]))
            if r["citations"]:
                rec["citation_precision"] = sum(is_relevant((c["doc_id"], c["char_start"], c["char_end"]), spans)
                                                for c in r["citations"]) / len(r["citations"])
        if not r["abstained"]:
            rec.update(judge(s, llm, q, r, offline))
        per_q.append(rec)
    ans = [p for p in per_q if p["answerable"]]
    una = [p for p in per_q if not p["answerable"]]
    summary = {
        "faithfulness": bootstrap_ci([p.get("faithfulness") for p in per_q if "faithfulness" in p]),
        "answer_relevance": bootstrap_ci([p.get("relevance") for p in per_q if "relevance" in p and p["answerable"]]),
        "em": bootstrap_ci([p["em"] for p in ans]), "f1": bootstrap_ci([p["f1"] for p in ans]),
        "contains_gold": bootstrap_ci([p["contains_gold"] for p in ans]),
        "citation_precision": bootstrap_ci([p["citation_precision"] for p in ans if "citation_precision" in p]),
        "correct_abstention": f"{sum(p['abstained'] for p in una)}/{len(una)}",
        "false_abstention": f"{sum(p['abstained'] for p in ans)}/{len(ans)}",
        "answered_unanswerable": [p["id"] for p in una if not p["abstained"]],
        "gen_cost_usd": sum(p["gen_cost_usd"] for p in per_q), "judge_cost_usd": sum(p.get("judge_cost_usd", 0) for p in per_q)}
    res = {"provenance": provenance(s), "k": s.top_k, "chunk": f"{s.chunk_size}/{s.chunk_overlap}", "summary": summary,
           "per_question": per_q}
    (RESULTS / "answers.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if len(gen_live) == len([p for p in per_q if not p["abstained"]]) and gen_live:
        tot = [l["total"] for l in gen_live]
        (RESULTS / "latency_e2e.json").write_text(json.dumps({"kind": "measured", "n": len(tot), "p50_ms": float(np.percentile(tot, 50)),
            "p95_ms": float(np.percentile(tot, 95)), "note": "live run: query embed + retrieval + generation, answered questions only",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    f = lambda t: "n/a" if t[0] is None else f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"
    md = f"""# Answer evaluation [MEASURED]

18 eval questions, k={s.top_k}, chunk {s.chunk_size}/{s.chunk_overlap}, generator `{s.gen_model}`, judge `{s.judge_model}`.
Mean [95% bootstrap CI]. Per-question detail and judge claims: `answers.json`.

| Metric | Score | Method |
|---|---|---|
| Faithfulness | {f(summary['faithfulness'])} | LLM judge splits the answer into claims; a claim counts only if its quote is verified in the context by code |
| Answer relevance | {f(summary['answer_relevance'])} | LLM judge 1-5 -> (s-1)/4, answerable questions that were answered |
| EM | {f(summary['em'])} | SQuAD-normalized exact match vs short gold answer (abstain = 0) |
| F1 | {f(summary['f1'])} | SQuAD token F1 vs gold (abstain = 0) |
| Contains gold | {f(summary['contains_gold'])} | all gold answer parts appear in the answer |
| Citation precision | {f(summary['citation_precision'])} | cited chunks that are relevant to the gold evidence (no LLM) |
| Correct abstention (unanswerable) | {summary['correct_abstention']} | |
| False abstention (answerable) | {summary['false_abstention']} | |

Generation spend ${summary['gen_cost_usd']:.4f}, judge spend ${summary['judge_cost_usd']:.4f} (from API usage metadata; $0 when replayed from cache).
"""
    (RESULTS / "answers.md").write_text(md, encoding="utf-8")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibrate", "retrieval", "answers", "all"])
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    s = Settings()
    llm = Gemini(s.cache_dir, s.budget_usd, s.rpm)
    if a.step in ("calibrate", "all"):
        ingest(CORPUS, s, llm, offline=a.offline)
        calibrate(s, llm, a.offline)
        s = replace(s, min_similarity=json.loads((ROOT / "eval/calibration.json").read_text())["min_similarity"])
    if a.step in ("retrieval", "all"):
        retrieval(s, llm, a.offline)
    if a.step in ("answers", "all"):
        answers(s, llm, a.offline)
    print(f"total API spend recorded in caches: ${llm.spent():.4f} (budget ${s.budget_usd})")


if __name__ == "__main__":
    main()
