"""Validate the frozen question sets against the corpus text that ingestion will see.

Run from problem1-rag/:  ..\\.venv\\Scripts\\python -m eval.validate_questions
Exit code 1 if any check fails. Checks:
  * schema + type mix: eval = 18 (6 single_fact, 3 paraphrased, 3 multi_span, 2 filter_scoped, 4 unanswerable);
    dev = 5 (3 answerable, 2 unanswerable)
  * every evidence quote is found verbatim (whitespace-insensitive) in its document's extracted text
  * the gold answer (or each gold_answer_part) appears inside the evidence quotes, so answers are extractive
  * multi_span: >= 2 non-overlapping evidence spans
  * paraphrased: <= PARAPHRASE_MAX_OVERLAP of the question's content words appear in its evidence
  * filter_scoped: evidence docs match the filter, and `topic_term` also occurs in a doc the filter excludes
  * unanswerable: no evidence; none of `absent_terms` occurs anywhere in the corpus
  * dev and eval are disjoint (ids, question text, and no overlapping evidence spans)
"""
import json, re, sys
from pathlib import Path

from rag.loaders import find_span, load_corpus, normalize

ROOT = Path(__file__).resolve().parents[1]
CORPUS, EVAL, DEV = ROOT / "data/corpus", ROOT / "eval/questions.jsonl", ROOT / "eval/dev_questions.jsonl"
TYPES = {"single_fact", "paraphrased", "multi_span", "filter_scoped", "unanswerable"}
EVAL_MIX = {"single_fact": 6, "paraphrased": 3, "multi_span": 3, "filter_scoped": 2, "unanswerable": 4}
FILTER_KEYS = {"source_type", "doc_id"}
PARAPHRASE_MAX_OVERLAP = 0.25
STOP = set("a an the of to in on for and or by with from as at is are was were be been it its this that these those "
           "what which who whom how why when where does do did can could should would may might must not no than "
           "their there they them about into over under between according say says said describe described".split())


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def content_words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


def overlap(q):
    qw = content_words(q["question"])
    ew = content_words(" ".join(e["quote"] for e in q["evidence"]))
    return len(qw & ew) / len(qw) if qw else 0.0


def contains(haystack, needle):
    return normalize(needle).lower() in normalize(haystack).lower()


def check_question(q, docs):
    errs, qid = [], q.get("id", "?")
    for f in ("id", "question", "type", "answerable", "gold_answer", "evidence"):
        if f not in q:
            errs.append(f"{qid}: missing field {f}")
    if errs:
        return errs, []
    if q["type"] not in TYPES:
        errs.append(f"{qid}: unknown type {q['type']}")
    if q["answerable"] != (q["type"] != "unanswerable"):
        errs.append(f"{qid}: answerable flag inconsistent with type")
    spans = []
    if not q["answerable"]:
        if q["evidence"] or q["gold_answer"] is not None:
            errs.append(f"{qid}: unanswerable must have no evidence and gold_answer null")
        if not q.get("absent_terms"):
            errs.append(f"{qid}: unanswerable needs absent_terms")
        for t in q.get("absent_terms", []):
            found = [d.doc_id for d in docs.values() if t.lower() in d.text.lower()]
            if found:
                errs.append(f"{qid}: absent term {t!r} found in {found}")
        return errs, spans
    if not q["evidence"] or not q["gold_answer"]:
        return errs + [f"{qid}: answerable needs evidence and gold_answer"], spans
    for e in q["evidence"]:
        doc = docs.get(e["doc_id"])
        span = find_span(doc.text, e["quote"]) if doc else None
        if span is None:
            errs.append(f"{qid}: quote not found in {e['doc_id']}: {e['quote'][:60]!r}")
        else:
            spans.append((e["doc_id"], *span))
    quotes = " ".join(e["quote"] for e in q["evidence"])
    for part in q.get("gold_answer_parts") or [q["gold_answer"]]:
        if not contains(quotes, part):
            errs.append(f"{qid}: gold answer part not in evidence: {part!r}")
    if q["type"] == "multi_span":
        distinct = {s for s in spans if not any(o != s and o[0] == s[0] and o[1] < s[2] and s[1] < o[2] for o in spans)}
        if len(distinct) < 2:
            errs.append(f"{qid}: multi_span needs >= 2 non-overlapping evidence spans")
    if q["type"] == "paraphrased" and overlap(q) > PARAPHRASE_MAX_OVERLAP:
        errs.append(f"{qid}: paraphrased but word overlap {overlap(q):.2f} > {PARAPHRASE_MAX_OVERLAP}")
    if q["type"] == "filter_scoped":
        flt = q.get("filters") or {}
        if not flt or set(flt) - FILTER_KEYS:
            errs.append(f"{qid}: filter_scoped needs filters with keys in {FILTER_KEYS}")
        match = lambda d: all(getattr(d, "source_type" if k == "source_type" else "doc_id") == v for k, v in flt.items())
        if any(not match(docs[e["doc_id"]]) for e in q["evidence"] if e["doc_id"] in docs):
            errs.append(f"{qid}: evidence doc outside its own filter")
        term = q.get("topic_term", "")
        if not term or not any(term.lower() in d.text.lower() for d in docs.values() if not match(d)):
            errs.append(f"{qid}: topic_term must also occur in a document the filter excludes (else the filter is moot)")
    elif q.get("filters"):
        errs.append(f"{qid}: only filter_scoped questions may set filters")
    return errs, spans


def validate(eval_qs, dev_qs, docs):
    errs, eval_spans = [], []
    for qs in (eval_qs, dev_qs):
        ids = [q.get("id") for q in qs]
        errs += [f"duplicate id {i}" for i in set(ids) if ids.count(i) > 1]
    for q in eval_qs:
        e, s = check_question(q, docs)
        errs += e
        eval_spans += s
    for q in dev_qs:
        e, s = check_question(q, docs)
        errs += e
        for d, a, b in s:
            if any(d == d2 and a < b2 and a2 < b for d2, a2, b2 in eval_spans):
                errs.append(f"{q['id']}: dev evidence overlaps an eval evidence span (tuning leak)")
    mix = {t: sum(q.get("type") == t for q in eval_qs) for t in TYPES}
    if mix != EVAL_MIX:
        errs.append(f"eval type mix {mix} != {EVAL_MIX}")
    if (len(dev_qs), sum(q.get("answerable", False) for q in dev_qs)) != (5, 3):
        errs.append("dev set must have 5 questions, 3 answerable")
    shared = {q.get("id") for q in eval_qs} & {q.get("id") for q in dev_qs}
    shared |= {normalize(q["question"]).lower() for q in eval_qs} & {normalize(q["question"]).lower() for q in dev_qs}
    errs += [f"eval/dev share {x!r}" for x in shared]
    return errs


def main():
    docs = {d.doc_id: d for d in load_corpus(CORPUS)}
    eval_qs, dev_qs = load(EVAL), load(DEV)
    errs = validate(eval_qs, dev_qs, docs)
    for q in eval_qs + dev_qs:
        ov = f"{overlap(q):.2f}" if q.get("evidence") else "  - "
        print(f"{q['id']:4s} {q['type']:13s} overlap={ov} evidence={len(q.get('evidence', []))} {q['question'][:70]}")
    print(f"\n{len(eval_qs)} eval + {len(dev_qs)} dev questions, {len(docs)} docs: "
          + ("ALL CHECKS PASS" if not errs else f"{len(errs)} ERROR(S)"))
    for e in errs:
        print("  -", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
