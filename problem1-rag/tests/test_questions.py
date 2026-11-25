"""Question-set validity: the frozen eval/dev sets pass every validator check, and the validator itself catches bad data."""
import copy

import pytest

from eval.validate_questions import CORPUS, DEV, EVAL, check_question, load, validate
from rag.loaders import load_corpus

DOCS = {d.doc_id: d for d in load_corpus(CORPUS)}
EVAL_QS, DEV_QS = load(EVAL), load(DEV)


def test_frozen_sets_pass_all_checks():
    assert validate(EVAL_QS, DEV_QS, DOCS) == []


def test_sizes_match_plan():
    assert len(EVAL_QS) == 18 and sum(q["answerable"] for q in EVAL_QS) == 14
    assert len(DEV_QS) == 5 and sum(q["answerable"] for q in DEV_QS) == 3


def _q(qid):
    return copy.deepcopy(next(q for q in EVAL_QS if q["id"] == qid))


@pytest.mark.parametrize("mutate, expected", [
    (lambda q: q["evidence"][0].update(quote=q["evidence"][0]["quote"] + " INVENTED"), "quote not found"),
    (lambda q: q.update(gold_answer="something else entirely"), "gold answer part not in evidence"),
    (lambda q: q["evidence"][0].update(doc_id="no-such-doc"), "quote not found"),
])
def test_validator_rejects_fabricated_evidence(mutate, expected):
    q = _q("q01")
    mutate(q)
    errs, _ = check_question(q, DOCS)
    assert any(expected in e for e in errs), errs


def test_validator_rejects_unanswerable_whose_answer_is_in_corpus():
    q = _q("q15")
    q["absent_terms"] = ["GOVERN"]
    assert any("absent term" in e for e in check_question(q, DOCS)[0])


def test_validator_rejects_paraphrase_with_high_overlap():
    q = _q("q07")
    q["question"] = "What is the production of confidently stated but erroneous or false content called?"
    assert any("overlap" in e for e in check_question(q, DOCS)[0])


def test_validator_rejects_moot_filter():
    q = _q("q13")
    q["filters"] = {"source_type": "md"}
    q["topic_term"] = "Proof Pudding"  # occurs only in an md doc, so the md filter excludes nothing on-topic
    assert any("topic_term" in e for e in check_question(q, DOCS)[0])


def test_dev_leak_is_detected():
    dev = copy.deepcopy(DEV_QS)
    dev[0]["evidence"] = copy.deepcopy(_q("q01")["evidence"])
    dev[0]["gold_answer"] = _q("q01")["gold_answer"]
    assert any("tuning leak" in e for e in validate(EVAL_QS, dev, DOCS))
