"""Offline tests: parser fallback chain, stats, pass rule + grounding, order swap, repair retry, family guard, cache/budget."""
import json, math
from types import SimpleNamespace

import pytest

from judge.judge import judge_pair, judge_pointwise
from judge.llm import BudgetExceeded, Gemini
from judge.parse import parse_verdict
from judge.pipeline import guard_family, load_cfg
from judge.schema import CRITERIA, PairwiseV0, PointwiseV1, TestCase
from judge.stats import auroc, cohen_kappa, norm_entropy, sign_test_p, spearman

V1 = load_cfg("judge_v1")
GOOD = {"criteria": {c: {"score": 5, "evidence": "Au", "rationale": "ok"} for c in CRITERIA}, "overall_rationale": "fine"}

MALFORMED = [  # (raw judge text, expected fallback step)
    ('```json\n{"winner": "A", "rationale": "x"}\n```', "fences"),
    ('Sure! Here is my verdict: {"winner": "B", "rationale": "x"} Hope that helps.', "braces"),
    ('{"winner": "A", "rationale": "x",}', "repair"),                 # trailing comma
    ("{'winner': 'tie', 'rationale': 'x'}", "repair"),                # single quotes
    ('{"winner": "A", "rationale": "cut off mid', "repair"),          # truncated
    ('{"winner": "C", "rationale": "x"}', "parse_failed"),            # out-of-schema value
    ('{"rationale": "no winner"}', "parse_failed"),                   # missing field
    ("", "parse_failed"),
]


@pytest.mark.parametrize("raw, step", MALFORMED)
def test_parser_fallback_chain(raw, step):
    v, got, err = parse_verdict(raw, PairwiseV0)
    assert got == step and (v is None) == (step == "parse_failed")


def test_stats_hand_computed():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert math.isclose(cohen_kappa([1, 1, 1, 0], [1, 0, 1, 0]), 0.5)
    assert auroc([3, 4], [1, 4]) == 0.625 and norm_entropy([1, 1, 1], 5) == 0
    assert math.isclose(sign_test_p(9, 1), 0.021484375) and sign_test_p(0, 0) == 1.0
    assert math.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, model, prompt, **kw):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        return {"key": str(len(self.prompts)), "text": r if isinstance(r, str) else json.dumps(r), "cost_usd": 0.0,
                "usage": {"prompt": 1, "output": 1, "thoughts": 0}, "cached": False}


CASE = TestCase(id="c", input="Symbol for gold?", model_output="Au.", expected_output="Au")


def test_pointwise_pass_rule_and_grounding():
    ok = judge_pointwise(CASE, V1, FakeLLM([GOOD]))
    assert ok["passed"] and ok["grounded_correctness"] and ok["weighted"] == 5
    fake_quote = json.loads(json.dumps(GOOD))
    fake_quote["criteria"]["correctness"]["evidence"] = "Gold is Au, element 79"  # not in the response
    bad = judge_pointwise(CASE, V1, FakeLLM([fake_quote]))
    assert not bad["grounded_correctness"] and not bad["passed"]  # high scores without verifiable evidence do not pass


def test_repair_retry_then_parse_failed():
    partial = {"criteria": {"correctness": {"score": 5, "evidence": "Au", "rationale": "x"}}, "overall_rationale": "x"}
    fixed = judge_pointwise(CASE, V1, FakeLLM([partial, GOOD]))
    assert fixed["parse"] == "retry:json" and fixed["passed"] and len(fixed["keys"]) == 2
    failed = judge_pointwise(CASE, V1, FakeLLM(["not json at all", "still not json"]))
    assert failed["parse"] == "parse_failed" and "passed" not in failed


def _pair(slot, quote="Au."):
    return {"per_criterion": {c: slot for c in CRITERIA}, "winner": slot, "decisive_quote": quote, "rationale": "x"}


def test_pair_order_swap_mapping_and_agreement():
    # consistent: output1 wins in both orders (slot A in AB, slot B in BA)
    r = judge_pair("p", "q", "", "Au", "Au.", "Ag.", V1, FakeLLM([_pair("A"), _pair("B")]))
    assert r["final"] == 1 and not r["flipped"] and r["single_order"] == 1
    # position-biased: judge always picks slot A -> flip -> tie
    r = judge_pair("p", "q", "", "Au", "Au.", "Ag.", V1, FakeLLM([_pair("A"), _pair("A", "Ag.")]))
    assert r["flipped"] and r["final"] == 0
    # ungrounded winner (quote not in the winning response) -> treated as tie
    r = judge_pair("p", "q", "", "Au", "Au.", "Ag.", V1, FakeLLM([_pair("A", "invented"), _pair("B", "invented")]))
    assert r["final"] == 0 and r["orders"]["AB"]["ungrounded"]


def test_family_guard():
    guard_family(V1, "gemma")
    with pytest.raises(ValueError):
        guard_family(V1, "gemini")
    guard_family(V1, "gemini", allow_same=True)


def test_cache_and_budget_cap(tmp_path):
    calls = []
    def gen(**kw):
        calls.append(kw)
        return SimpleNamespace(text="{}", model_version="m", usage_metadata=SimpleNamespace(
            prompt_token_count=1000, candidates_token_count=1000, thoughts_token_count=0))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
    g = Gemini(tmp_path / "c.jsonl", budget_usd=1.0, rpm=6000, client=client)
    assert not g.generate("gemini-3.6-flash", "x")["cached"] and g.generate("gemini-3.6-flash", "x")["cached"]
    assert len(calls) == 1
    with pytest.raises(BudgetExceeded):
        Gemini(tmp_path / "c.jsonl", budget_usd=g.spent(), rpm=6000, client=client).generate("gemini-3.6-flash", "new")
    with pytest.raises(KeyError):
        Gemini(tmp_path / "c.jsonl", 1.0, client=client, offline=True).generate("gemini-3.6-flash", "never")


def test_client_requires_the_p2_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY_P2", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "not-used-by-p2")  # another key present must not be picked up
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY_P2"):
        Gemini(tmp_path / "c.jsonl", 1.0).client
    monkeypatch.setenv("GEMINI_API_KEY_P2", "test-key-p2")
    assert Gemini(tmp_path / "c.jsonl", 1.0).client._api_client.api_key == "test-key-p2"
