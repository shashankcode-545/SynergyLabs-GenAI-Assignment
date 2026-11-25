"""Fill a COPY of reference/submission_template.docx -> reports/submission.docx.

Only non-reflection placeholders are filled, with numbers read from the committed results files (nothing typed by hand).
Candidate-only cells (personal details, screenshots, reflections, AI-usage disclosure, the release-gate verdict) are left
as they are. Asserts the heading list is identical to the template's.
Run from repo root: .venv\\Scripts\\python reports\\fill_submission.py
"""
import json
from pathlib import Path

import yaml
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE, OUT = ROOT / "reference/submission_template.docx", ROOT / "reports/submission.docx"
P1R, P2R = ROOT / "problem1-rag/results", ROOT / "problem2-llm-judge/results"
CANDIDATE_TABLES = {16, 17, 18, 19, 20, 34, 35, 36, 37, 38}  # screenshots + reflections: never touched
j = lambda p: json.loads(p.read_text(encoding="utf-8"))


def set_cell(cell, text):
    if hasattr(cell, "rows"):  # single-cell box tables are passed as Table
        cell = cell.cell(0, 0)
    paras = cell.paragraphs
    for p in paras[1:]:
        p._element.getparent().remove(p._element)
    p = paras[0]
    for r in p.runs[1:]:
        r._element.getparent().remove(r._element)
    (p.runs[0] if p.runs else p.add_run()).text = text


def set_placeholder(paragraph, text):
    runs = [r for r in paragraph.runs if "[" in r.text or "]" in r.text or "..." in r.text]
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def headings(doc):
    return [(p.style.name, p.text) for p in doc.paragraphs if p.style is not None and p.style.name.startswith("Heading")]


def ci(t):
    return f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"


def p1_values():
    ret, ans, lat, cost = j(P1R / "retrieval.json"), j(P1R / "answers.json"), j(P1R / "latency.json"), j(P1R / "cost_table.json")
    e2e = j(P1R / "latency_e2e.json")
    m = ret["configs"][0]["metrics"]["@5"]
    s = ans["summary"]
    return ret, ans, lat, cost, e2e, m, s


def fill(doc):
    T, paras = doc.tables, doc.paragraphs
    ret, ans, lat, cost, e2e, m, s = p1_values()
    bv = j(P2R / "bias_validation.json") if (P2R / "bias_validation.json").exists() else None  # None -> P2 results pending
    ab = j(P2R / "ab_report.json") if (P2R / "ab_report.json").exists() else None
    rub = yaml.safe_load((ROOT / "problem2-llm-judge/judge/rubric.yaml").read_text(encoding="utf-8"))
    judge_model = yaml.safe_load((ROOT / "problem2-llm-judge/configs/judge_v1.yaml").read_text())["model"]

    # candidate details: only the two technical rows
    for row in T[0].rows:
        if row.cells[0].text.startswith("Problem 1"):
            set_cell(row.cells[1], "LanceDB (embedded, local disk)")
        if row.cells[0].text.startswith("Problem 2"):
            set_cell(row.cells[1], f"judge: {judge_model} (Gemini family) · generator: gemma-4-31b-it (Gemma family)")

    # ---------------- Problem 1
    set_cell(T[1], "Mermaid source: problem1-rag/README.md (Architecture). PDF/HTML/MD -> loaders (normalized text + page/"
             "section map) -> doc-hash check (unchanged = skip) -> recursive chunker 1200/200 chars with offsets -> "
             "gemini-embedding-2 @768-d (cached) -> LanceDB (vector + doc_id, source_type, title, location, text) -> "
             "query: embed -> top-k cosine + metadata filter -> similarity gate tau [NO-CONTEXT branch: fixed abstention, no "
             "LLM call] -> gemini-3.1-flash-lite JSON {answerable, answer, citations} -> citation check [invalid or "
             "answerable=false -> NO-CONTEXT branch] -> cited answer; every query logged (latency, chunk count, tokens).")
    set_cell(T[2], "git clone <repo> && cd SynergyLabs-GenAI-Assignment\npy -3.11 -m venv .venv\n"
             ".venv\\Scripts\\python -m pip install -r requirements.lock.txt\nCopy-Item .env.example .env   # add GEMINI_API_KEY")
    set_cell(T[3], "cd problem1-rag\n..\\.venv\\Scripts\\python -m rag ingest   # default chunk 1200 chars, overlap 200; "
             "re-run = 0 new vectors")
    set_cell(T[4], '..\\.venv\\Scripts\\python -m rag query "What are the typical root causes of Excessive Agency?" '
             "--filter source_type=md\n..\\.venv\\Scripts\\python -m rag serve   # POST http://127.0.0.1:8000/query "
             '{"question": "...", "k": 5, "filters": {"source_type": "md"}}')
    rows = {r.cells[0].text.strip(): r for r in T[5].rows}
    for key, metric, how in [
        ("Recall@k / Hit Rate", None, "Hit@5 = any relevant chunk in top 5; Recall@5 = share of gold evidence spans covered"),
        ("MRR", "mrr", "1/rank of first relevant chunk (0 if none in top 5)"),
        ("nDCG@k", "ndcg", "binary gain; IDCG over all relevant chunks in the index"),
        ("Context precision", "precision", "relevant chunks in top 5 / 5. Relevant = chunk covers >= 50% of a gold evidence "
                                           "span (verbatim quote, char offsets)")]:
        val = f"Hit {ci(m['hit'])}; Recall {ci(m['recall'])}" if metric is None else ci(m[metric])
        set_cell(rows[key].cells[1], val)
        set_cell(rows[key].cells[2], how + f"; 14 answerable questions, chunk 1200/200, mean [95% bootstrap CI]")
    for p in paras:
        if p.text.startswith("k used"):
            set_placeholder(p, "k = 5 (also reported at k = 1, 3, 10 in problem1-rag/results/retrieval.md)")
    rows = {r.cells[0].text.strip(): r for r in T[6].rows}
    for key, val, meth, note in [
        ("Faithfulness / groundedness", f"{s['faithfulness'][0]:.3f}", "LLM-judge (gemini-3.6-flash)",
         "claims split by judge; a claim counts only if its quote is verified in the context by code; 12 answered questions"),
        ("Answer relevance", f"{s['answer_relevance'][0]:.3f}", "LLM-judge", "1-5 anchored -> (s-1)/4; same-family judge, see caveat in README"),
        ("EM", f"{s['em'][0]:.3f}", "gold", "SQuAD-normalized vs short gold; answers are full sentences so EM ~ 0"),
        ("F1", ci(s["f1"]), "gold", f"SQuAD token F1; contains-gold {ci(s['contains_gold'])}; citation precision "
                                    f"{ci(s['citation_precision'])}; abstention {s['correct_abstention']} correct, "
                                    f"{s['false_abstention']} false")]:
        r = rows[key]
        set_cell(r.cells[1], val), set_cell(r.cells[2], meth), set_cell(r.cells[3], note)
    for r, row in zip(T[7].rows[1:], cost["rows"]):
        sh, pc = row["self_hosted"], row["pinecone"]["50000"]
        set_cell(r.cells[1], f"{sh['total']:.2f}")
        set_cell(r.cells[2], f"{pc['total']:.2f}")
        set_cell(r.cells[3], f"{pc['total'] - sh['total']:+.2f}/mo managed minus self-hosted; at 1M queries/mo Pinecone "
                             f"{row['pinecone']['1000000']['total']:.2f}")
    set_cell(T[8], "[PROJECTED] Store cost only, 50K queries/month. Pinecone Standard (verified 2026-11-25): $50/mo min, "
             "$0.33/GB-mo, $16/M RU, $4/M WU; 1 RU per GB namespace per query (min 0.25); 1 WU per KB upserted; 5% monthly "
             f"churn. Self-hosted (AWS us-east-1, 1 replica; prices are unverified assumptions): t4g.medium $0.0336/h "
             f"(<=1M) or r7g.large $0.1071/h (10M), gp3 $0.08/GB-mo x1.5 headroom, S3 backup $0.023/GB-mo. Measured "
             f"{cost['inputs_measured']['bytes_per_row']:.0f} B/row (768-d float32 + chunk text + metadata) and "
             f"{cost['inputs_measured']['index_bytes_per_vector']:.0f} B/vector IVF-PQ. Excludes ops time, HA, egress. "
             "Details: problem1-rag/results/cost_table.md")
    r_, y = lat["real_corpus"], lat["synthetic"]
    rows = {r.cells[0].text.strip(): r for r in T[9].rows}
    set_cell(rows["Retrieval p50"].cells[1], f"{r_['p50_ms']:.2f}")
    set_cell(rows["Retrieval p50"].cells[2], f"k=5, {r_['rows']} vectors, exact scan, i5-13450HX laptop, local SSD; "
                                              f"100K synthetic: flat {y['flat']['p50_ms']:.1f}, IVF-PQ {y['ivf_pq']['p50_ms']:.1f}")
    set_cell(rows["Retrieval p95"].cells[1], f"{r_['p95_ms']:.2f}")
    set_cell(rows["Retrieval p95"].cells[2], f"same; 100K synthetic: flat {y['flat']['p95_ms']:.1f}, IVF-PQ "
                                              f"{y['ivf_pq']['p95_ms']:.1f} (recall@10 {y['ivf_pq']['recall@10_vs_flat']:.3f} on random vectors)")
    k95 = next(k for k in rows if k.startswith("End-to-end"))
    set_cell(rows[k95].cells[1], f"{e2e['p95_ms']:.0f}")
    set_cell(rows[k95].cells[2], f"live run, n={e2e['n']}, query embed + retrieval + generation (p50 {e2e['p50_ms']:.0f}); "
                                 "free-tier API retries dominate the tail")
    d1 = [
        "LanceDB: embedded library (pip install, no server; Docker absent here), SQL-like metadata filters, merge_insert "
        "upsert, data in files on local disk or object storage so cost follows storage rather than always-on pods. "
        "Rejected: Qdrant / pgvector (need a server), Chroma (in-memory HNSW is RAM-heavy at 10M), FAISS (no metadata or "
        "persistence), sqlite-vec (brute-force scan).",
        f"1200 chars / 200 overlap (~300/50 tokens), recursive paragraph > line > sentence > word with exact offsets. "
        f"2400/400 retrieved better (Hit@5 {ret['configs'][1]['metrics']['@5']['hit'][0]:.2f} vs {m['hit'][0]:.2f}, MRR "
        f"{ret['configs'][1]['metrics']['@5']['mrr'][0]:.2f} vs {m['mrr'][0]:.2f}) but sends "
        f"{ret['configs'][1]['mean_context_tokens@5']:.0f} vs {ret['configs'][0]['mean_context_tokens@5']:.0f} context "
        "tokens/query, so generation cost roughly doubles; we kept 1200. A planned 600/100 run was not measured because the "
        "free-tier embedding quota ran out.",
        "gemini-embedding-2 at 768-d (MRL): a quarter of 3072-d storage. Switched from gemini-embedding-001 after its free "
        "daily quota (1,000 texts) was exhausted. Each text sent as its own Content (a plain list is merged into one vector).",
        f"Three layers: similarity gate tau = {ans['provenance']['min_similarity']} (lowest answerable dev similarity - 0.02, "
        f"dev questions only); model must return answerable=false; answers without a valid [S#] citation become a fixed "
        f"abstention. Result {s['correct_abstention']} unanswerable abstained, but the gate also blocked {s['false_abstention']} "
        "answerable questions.",
        "doc_id = stable slug; chunk_id = sha256(doc | config | offset | text). A per-table manifest of document text hashes "
        "skips unchanged docs; a changed doc is deleted then re-inserted via merge_insert on chunk_id; the manifest is written "
        "only after the store write. Re-ingest: 356 -> 356 vectors; tested offline.",
        "Accepted: single-writer embedded store, no built-in HA (S3 snapshots), ANN tuning left to us. Switch back to managed "
        "for a very large, lightly-queried index (10M vectors at 50K q/mo projects $52 managed vs $84 self-hosted), or when "
        "HA, multi-tenant isolation, many writers or no ops capacity matter. Self-hosted wins as traffic grows.",
    ]
    for t, text in zip(T[10:16], d1):
        set_cell(t, text)

    # ---------------- Problem 2
    PENDING = ("PENDING - not yet measured: live judge runs blocked by Gemini 503 'high demand' (reports/progress_report.md). "
               "Interim, incomplete data: problem2-llm-judge/results/interim_gemini-3.5-flash-lite.md")
    gen_model = yaml.safe_load((ROOT / "problem2-llm-judge/configs/gen_v1.yaml").read_text())["model"]
    if bv:
        mt = bv["metrics"]
        pa, pb, qa, qb = mt["pointwise"]["v0"], mt["pointwise"]["v1"], mt["pairwise"]["v0"], mt["pairwise"]["v1"]
        sa, sb, rr = mt["self_preference"]["v0"], mt["self_preference"]["v1"], mt["test_retest_v1"]
    pct = lambda x: f"{100 * x:.0f}%"
    set_cell(T[21], "Mermaid source: problem2-llm-judge/README.md. Suite JSON/JSONL/YAML -> pydantic validation -> family "
             "guard -> prompt (v1: rubric anchors + few-shot + reference) -> [pairwise: ORDER SWAP, AB and BA prompts] -> judge "
             "call (JSON schema, cache = audit log, budget cap) -> PARSE FALLBACK json -> strip fences -> outer braces -> "
             "json_repair -> pydantic -> 1 repair retry -> parse_failed -> grounding check (verbatim quotes) -> pass rule / "
             "AB-BA agreement (else tie) -> suite report -> A/B decision rule.")
    set_cell(T[22], "cd problem2-llm-judge\n..\\.venv\\Scripts\\python -m pip install -r requirements.txt")
    set_cell(T[23], "..\\.venv\\Scripts\\python -m judge run --suite suites/demo.yaml   # -> results/run_demo_judge_v1.md")
    set_cell(T[24], "..\\.venv\\Scripts\\python -m judge compare --a suites/ab_gen_v1.jsonl --b suites/ab_gen_v2.jsonl")
    pr = iter([p for p in paras if p.text.startswith("Prerequisites")])
    for p, txt in zip(pr, ["Python 3.11 (Windows 11 tested), Gemini API key; no services or GPU; LanceDB is embedded",
                           "Python 3.11 (Windows 11 tested), Gemini API key; no services or GPU"]):
        set_placeholder(p, txt)
    ev = iter([p for p in paras if p.text.startswith("Environment variables")])
    for p, txt in zip(ev, ["GEMINI_API_KEY, RAG_EMBED_MODEL, RAG_EMBED_DIM, RAG_GEN_MODEL, RAG_JUDGE_MODEL, RAG_CHUNK_SIZE, "
                           "RAG_CHUNK_OVERLAP, RAG_TOP_K, RAG_MIN_SIMILARITY, RAG_DB_PATH, RAG_LOG_PATH, BUDGET_USD, LLM_RPM",
                           "GEMINI_API_KEY_P2, BUDGET_USD, LLM_RPM, LLM_TIMEOUT_S (models in configs/*.yaml)"]):
        set_placeholder(p, txt)
    for p in paras:
        if p.text.startswith("Vector store chosen"):
            set_placeholder(p, "LanceDB - embedded, disk/object-storage backed, no always-on server")
        elif p.text.startswith("Judging mode used"):
            set_placeholder(p, "pointwise (reference-based when expected_output exists, else reference-free) + pairwise A-vs-B with order swap")
        elif p.text.startswith("Judge & generator configured independently"):
            set_placeholder(p, "yes - separate configs/judge_v*.yaml and configs/gen_v*.yaml, each with model + family; "
                               "guard refuses same-family judge/generator")
        elif p.text.startswith("Declared winner"):
            set_placeholder(p, f"{ab['winner']} - win rate of B {ab['win_rate_b'][0]:.2f} [{ab['win_rate_b'][1]:.2f}, "
                               f"{ab['win_rate_b'][2]:.2f}], sign test p = {ab['sign_test_p']:.3f} (rule fixed before the run)"
                            if ab else PENDING)
    names = {"Correctness": "correctness", "Faithfulness": "faithfulness", "Completeness": "completeness",
             "Instruction-following": "instruction_following", "Tone / safety": "tone_safety"}
    for r in T[25].rows[1:]:
        c = rub["criteria"][names[r.cells[0].text.strip()]]
        a = c["anchors"]
        set_cell(r.cells[1], f"{c['definition']} 1 = {a[1]}; 3 = {a[3]}; 5 = {a[5]}.")
        set_cell(r.cells[2], f"{c['weight']:.2f}")
    mit = {"Position (A/B order)": "Both orders for every pair; final verdict only if AB and BA agree, else tie (judge.judge_pair)",
           "Verbosity / length": "Rubric penalizes content-free length (instruction_following); padded-answer probe; 'do not reward length' rule",
           "Self-enhancement": f"Judge (Gemini) and A/B generator {gen_model} (Gemma) from different families; guard_family refuses same family; shared vendor disclosed",
           "Sycophancy / style": "Per-criterion verbatim evidence; correctness quote verified in code (else fail); reference block; pairwise decisive quote verified",
           "Score clustering": "1-5 anchors per criterion + 3 few-shot calibration examples; pairwise for comparisons"}
    res = {k: PENDING for k in mit}
    if bv:
        res = {"Position (A/B order)": f"flip rate {pct(qa['flip_rate'])} -> {pct(qb['flip_rate'])}; slot-1 win rate "
                                       f"{pct(qa['slot1_win_rate'])} -> {pct(qb['slot1_win_rate'])}; accuracy vs gold "
                                       f"{pct(qa['single_order_accuracy'])} (v0 single order) -> {pct(qb['swap_final_accuracy'])} (v1 swap-aggregated)",
               "Verbosity / length": f"padded-minus-terse score {pa['pad_minus_terse'][0]:+.2f} -> {pb['pad_minus_terse'][0]:+.2f}; padded "
                                     f"preferred {pct(qa['padded_preferred_single_order'])} -> {pct(qb['padded_preferred_final'])}; "
                                     f"verbose-wrong preferred {pct(qa['verbose_wrong_preferred_single_order'])} -> {pct(qb['verbose_wrong_preferred_final'])}",
               "Self-enhancement": f"self-preference delta {sa['delta_self_preference']:+.2f} -> {sb['delta_self_preference']:+.2f} (paraphrase-swap probe)",
               "Sycophancy / style": f"confident-wrong passed {pct(pa['CW_passed_rate'])} -> {pct(pb['CW_passed_rate'])} (false premise "
                                     f"{pa['CW_false_premise_passed']} -> {pb['CW_false_premise_passed']}); verbose-wrong passed "
                                     f"{pct(pa['VW_passed_rate'])} -> {pct(pb['VW_passed_rate'])}",
               "Score clustering": f"normalized entropy {pa['norm_entropy']:.2f} -> {pb['norm_entropy']:.2f}; AUROC correct vs wrong "
                                   f"{pa['auroc_correct_vs_wrong']:.2f} -> {pb['auroc_correct_vs_wrong']:.2f}"}
    for r in T[26].rows[1:]:
        k = r.cells[0].text.strip()
        set_cell(r.cells[1], mit[k]), set_cell(r.cells[2], res[k])
    val = [(PENDING, "")] * 4
    if bv:
        val = [(f"{pct(pa['gold_accuracy'])} -> {pct(pb['gold_accuracy'])}", f"n = {pb['n_gold']} probe outputs with constructed gold labels (v0 -> v1)"),
               (f"kappa {pa['gold_kappa']:.2f} -> {pb['gold_kappa']:.2f}", "Cohen's kappa, pass/fail vs gold"),
               (pct(rr["verdict_flip_rate"]), f"{rr['outputs']} outputs x {rr['runs_each']} runs at temp = {rr['temperature']}; mean score range {rr['mean_weighted_score_range']:.2f}"),
               ("Y" if pb["VW_passed_rate"] > 0 or pb["T_failed_rate"] > 0 else "N",
                f"v1: verbose-but-wrong passed {pct(pb['VW_passed_rate'])}, terse-but-correct failed {pct(pb['T_failed_rate'])} "
                f"(v0: {pct(pa['VW_passed_rate'])}, {pct(pa['T_failed_rate'])})")]
    for r, (a_, b_) in zip(T[27].rows[1:], val):
        set_cell(r.cells[1], a_)
        if b_:
            set_cell(r.cells[2], b_)
    for r, name, arm in zip(T[28].rows[1:], ["gen_v1 (plain prompt)", "gen_v2 (constraint + fact-check prompt)"], ["summary_a", "summary_b"]):
        set_cell(r.cells[0], r.cells[0].text.replace("[ name ]", name))
        if ab:
            sm, wr = ab[arm], ab["win_rate_b"]
            w = f"{1 - wr[0]:.2f}" if arm == "summary_a" else f"{wr[0]:.2f} [{wr[1]:.2f}, {wr[2]:.2f}]"
            set_cell(r.cells[1], f"{sm['pass_rate'][0]:.2f}"), set_cell(r.cells[2], f"{sm['mean_score_norm'][0]:.2f}"), set_cell(r.cells[3], w)
        else:
            set_cell(r.cells[1], "PENDING"), set_cell(r.cells[2], "PENDING"), set_cell(r.cells[3], "PENDING")
    d2 = [
        "Pointwise (reference-based when a gold answer exists, reference-free otherwise) gives pass rates and per-criterion "
        "scores for gating; pairwise A-vs-B gives win rates for config comparison, is less prone to score clustering, and "
        "is required for the position-bias check (both orders).",
        "Gemini JSON mode with an explicit response schema (a bare object schema made the model return an empty object); then "
        "json.loads -> strip code fences -> extract outer braces -> json_repair -> pydantic validation; one repair retry with "
        "the error; otherwise parse_failed (kept in the log, excluded, counted). Overall score and pass/fail are computed in "
        "code. 8 malformed fixtures are unit-tested.",
        f"Judge {judge_model} is Gemini-family; the A/B generator {gen_model} is Gemma-family, and a config guard refuses "
        "same-family judging. Both are Google models, so the separation is partial; the paraphrase-swap probe measures the "
        "residual self-preference: " + (f"delta {sa['delta_self_preference']:+.2f} -> {sb['delta_self_preference']:+.2f}." if bv else PENDING),
        (f"[Draft from measured results - candidate to confirm] Position bias: v0 flip rate {pct(qa['flip_rate'])}, v1 "
         f"{pct(qb['flip_rate'])}; the swap-and-agree rule removes order effects from final verdicts by construction, at the "
         f"cost of {pct(qb['ties_from_disagreement'])} of pairs becoming ties. Confidence is limited by n = 24 pairs.") if bv else
        ("[Candidate to answer once results exist] " + PENDING + ". Interim note (replaced judge gemini-3.5-flash-lite, v0 only): "
         "100% gold agreement and 0% flip rate on the probe suite, i.e. a ceiling effect - these probes may be too easy to show gains."),
        (f"[Evidence for the candidate's own verdict] v1 gold agreement {pct(pb['gold_accuracy'])} (kappa {pb['gold_kappa']:.2f}) "
         f"on constructed probes; test-retest flips {pct(rr['verdict_flip_rate'])}; confident-wrong passed {pct(pb['CW_passed_rate'])}; "
         "probes are clear-cut, so real-task reliability is likely lower. Guardrails to consider: pairwise both orders, "
         "verified quotes, human review of failures and ties.") if bv else "[Candidate's own verdict] " + PENDING,
    ]
    for t, text in zip(T[29:34], d2):
        set_cell(t, text)


def main():
    tpl, doc = Document(TEMPLATE), Document(TEMPLATE)
    before = [doc.tables[i]._element.xml for i in CANDIDATE_TABLES]
    fill(doc)
    assert headings(doc) == headings(tpl), "template headings changed"
    assert [doc.tables[i]._element.xml for i in CANDIDATE_TABLES] == before, "candidate-only cells were modified"
    doc.save(OUT)
    print(f"wrote {OUT}; {len(headings(doc))} headings identical to the template; candidate cells untouched")


if __name__ == "__main__":
    main()
