"""Validate committed experiment results (skipped until results/experiments_raw.json exists): expected counts, unique ids,
parse rate, single judge model, and saved metrics == metrics recomputed from the raw per-call records."""
import json

import pytest

from judge.experiments import RETEST_ITEMS, SELFPREF_N, base_items, pairwise_metrics, pointwise_metrics, retest, selfpref_delta
from judge.pipeline import RESULTS, load_cfg

RAW = RESULTS / "experiments_raw.json"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="experiments not run yet")


@pytest.fixture(scope="module")
def raw():
    return json.loads(RAW.read_text(encoding="utf-8"))


def test_counts_and_unique_ids(raw):
    n = len(base_items())
    for v in ("v0", "v1"):
        assert len(raw["P"][v]) == 4 * n and len({r["id"] for r in raw["P"][v]}) == 4 * n
        assert len(raw["Q"][v]) == 2 * n and len({r["id"] for r in raw["Q"][v]}) == 2 * n
        assert len(raw["S"][v]) == 2 * SELFPREF_N and len({r["id"] for r in raw["S"][v]}) == 2 * SELFPREF_N
    assert len(raw["R"]) == 2 * 4 * len(RETEST_ITEMS) and len({(r["id"], r["rerun_idx"]) for r in raw["R"]}) == len(raw["R"])
    if (RESULTS / "ab_report.json").exists():  # A/B is produced only by the full 'all' stage
        ab = json.loads((RESULTS / "ab_report.json").read_text(encoding="utf-8"))
        assert len(ab["pairs"]) == n and len(ab["pointwise_a"]) == n and len(ab["pointwise_b"]) == n


def test_parse_rate_and_single_judge(raw):
    recs = raw["P"]["v0"] + raw["P"]["v1"] + raw["R"]
    orders = [o for grp in ("Q", "S") for v in raw[grp].values() for r in v for o in r["orders"].values()]
    assert sum("passed" in r for r in recs) / len(recs) >= 0.95
    assert sum(o["candidate"] is not None for o in orders) / len(orders) >= 0.95
    assert raw["provenance"]["judge"]["model"] == load_cfg("judge_v0")["model"] == load_cfg("judge_v1")["model"]


def test_saved_metrics_match_recomputation(raw):
    saved = json.loads((RESULTS / "bias_validation.json").read_text(encoding="utf-8"))["metrics"]
    fresh = {"pointwise": {v: pointwise_metrics(raw["P"][v]) for v in raw["P"]},
             "pairwise": {v: pairwise_metrics(raw["Q"][v]) for v in raw["Q"]},
             "self_preference": {v: selfpref_delta(raw["S"][v]) for v in raw["S"]},
             "test_retest_v1": retest(raw["P"]["v1"], raw["R"])}
    assert json.loads(json.dumps(fresh)) == saved
