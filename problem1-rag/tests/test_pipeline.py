"""Offline pipeline tests with a fake Gemini: chunking, idempotent ingest, answer/abstain logic, cache + budget cap."""
import hashlib, json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rag.answer import ABSTAIN, answer
from rag.chunking import chunk_spans
from rag.config import Settings
from rag.ingest import ingest
from rag.llm import BudgetExceeded, Gemini
from rag.store import open_table, where_clause


class FakeLLM:
    def __init__(self, reply=None):
        self.reply, self.gen_calls, self.embedded = reply, 0, 0

    def embed(self, texts, model, dim, task_type, offline=False):
        self.embedded += len(texts)
        out = []
        for t in texts:  # bag-of-words hashing: shared words -> similar vectors
            v = np.zeros(dim, np.float32)
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return np.stack(out)

    def generate(self, *a, **k):
        self.gen_calls += 1
        return {"text": json.dumps(self.reply), "usage": {"prompt": 10, "output": 5, "thoughts": 0}, "cost_usd": 0.0, "cached": False}


def make_corpus(tmp_path, docs):
    (tmp_path / "md").mkdir(parents=True, exist_ok=True)
    meta = []
    for doc_id, text in docs.items():
        (tmp_path / "md" / f"{doc_id}.md").write_text(text, encoding="utf-8")
        meta.append({"doc_id": doc_id, "file": f"md/{doc_id}.md", "source_type": "md", "title": doc_id.upper()})
    (tmp_path / "sources.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


DOCS = {"alpha": "# Alpha\n" + "Vector stores keep embeddings on disk. " * 60,
        "beta": "# Beta\n" + "Prompt injection alters model behavior. " * 60}


@pytest.fixture
def env(tmp_path):
    corpus = make_corpus(tmp_path / "corpus", DOCS)
    s = replace(Settings(), db_path=tmp_path / "db", log_path=tmp_path / "q.jsonl", chunk_size=300, chunk_overlap=50)
    return corpus, s


def test_chunk_spans_cover_text_with_bounded_size_and_overlap():
    text = ("Sentence number one is here. " * 40 + "\n\n") * 3
    spans = chunk_spans(text, 300, 50)
    assert all(e - s <= 300 for s, e in spans)
    assert spans[0][0] == 0 and spans[-1][1] == len(text.rstrip())
    assert all(b[0] < a[1] for a, b in zip(spans, spans[1:]))  # consecutive chunks overlap
    assert all(b[0] > a[0] for a, b in zip(spans, spans[1:]))  # always progresses


def test_ingest_is_idempotent_and_only_changed_doc_is_reembedded(env):
    corpus, s = env
    llm = FakeLLM()
    first = ingest(corpus, s, llm)
    n, embedded = first["vectors_after"], llm.embedded
    assert n > 0 and first["vectors_before"] == 0
    again = ingest(corpus, s, llm)
    assert again["vectors_after"] == n and sorted(again["skipped"]) == ["alpha", "beta"] and llm.embedded == embedded
    (corpus / "md" / "beta.md").write_text("# Beta\n" + "Changed text about agents. " * 30, encoding="utf-8")
    third = ingest(corpus, s, llm)
    assert third["skipped"] == ["alpha"] and len(third["ingested"]) == 1
    t = open_table(s.db_path, s.table_name(), s.embed_dim)
    assert t.count_rows("doc_id = 'beta'") == int(third["ingested"][0].split("(")[1].split()[0])
    ids = [r["chunk_id"] for r in t.search().limit(10_000).to_list()]
    assert len(ids) == len(set(ids))  # no duplicate vectors


def test_answer_drops_invalid_citations_and_abstains_without_valid_ones(env):
    corpus, s = env
    ingest(corpus, s, FakeLLM())
    table = open_table(s.db_path, s.table_name(), s.embed_dim)
    s = replace(s, min_similarity=0.0)
    ok = answer("prompt injection alters model behavior", s, FakeLLM({"answerable": True, "answer": "It alters behavior [S1] [S9].", "citations": ["S1", "S9"]}), table, k=3)
    assert not ok["abstained"] and [c["sid"] for c in ok["citations"]] == ["S1"]
    assert ok["citations"][0]["doc_id"] == "beta"
    bad = answer("prompt injection", s, FakeLLM({"answerable": True, "answer": "No cites.", "citations": ["S7"]}), table, k=3)
    assert bad["abstained"] and bad["answer"] == ABSTAIN
    no = answer("prompt injection", s, FakeLLM({"answerable": False, "answer": "", "citations": []}), table, k=3)
    assert no["abstained"] and no["reason"] == "model reported unanswerable"
    assert len(s.log_path.read_text().splitlines()) == 3


def test_low_similarity_abstains_without_calling_the_llm(env):
    corpus, s = env
    ingest(corpus, s, FakeLLM())
    llm = FakeLLM({"answerable": True, "answer": "x [S1]", "citations": ["S1"]})
    r = answer("completely unrelated quantum gardening", replace(s, min_similarity=0.99), llm,
               open_table(s.db_path, s.table_name(), s.embed_dim))
    assert r["abstained"] and llm.gen_calls == 0 and "similarity" in r["reason"]


def test_metadata_filter_restricts_results_and_rejects_injection(env):
    corpus, s = env
    ingest(corpus, s, FakeLLM())
    r = answer("prompt injection", replace(s, min_similarity=0.0), FakeLLM({"answerable": False, "answer": "", "citations": []}),
               open_table(s.db_path, s.table_name(), s.embed_dim), filters={"doc_id": "alpha"})
    assert {c["doc_id"] for c in r["retrieved"]} == {"alpha"}
    with pytest.raises(ValueError):
        where_clause({"doc_id": "x' OR '1'='1"})
    with pytest.raises(ValueError):
        where_clause({"text": "anything"})


def _fake_client(calls):
    def gen(**kw):
        calls.append(kw)
        return SimpleNamespace(text='{"a": 1}', model_version="m",
                               usage_metadata=SimpleNamespace(prompt_token_count=1000, candidates_token_count=1000, thoughts_token_count=0))
    return SimpleNamespace(models=SimpleNamespace(generate_content=gen))


def test_cache_hit_makes_no_api_call_and_budget_cap_blocks_spend(tmp_path):
    calls = []
    g = Gemini(tmp_path, budget_usd=1.0, rpm=6000, client=_fake_client(calls))
    a = g.generate("gemini-3.6-flash", "hello")
    b = g.generate("gemini-3.6-flash", "hello")
    assert len(calls) == 1 and not a["cached"] and b["cached"] and b["text"] == a["text"]
    assert Gemini(tmp_path, 1.0, client=_fake_client(calls)).generate("gemini-3.6-flash", "hello")["cached"]  # survives restart
    tight = Gemini(tmp_path, budget_usd=g.spent() + 1e-6, rpm=6000, client=_fake_client(calls))
    with pytest.raises(BudgetExceeded):
        tight.generate("gemini-3.6-flash", "a new prompt")
    assert len(calls) == 1
    with pytest.raises(KeyError):
        g.generate("gemini-3.6-flash", "never seen", offline=True)
