"""IR / answer metrics against hand-computed values."""
import math

from eval.metrics import bootstrap_ci, contains_all, exact_match, ir_metrics, token_f1

SPAN = ("d", 100, 200)
REL = ("d", 90, 180)      # covers 80/100 of the span -> relevant
PART = ("d", 160, 300)    # covers 40/100 -> not relevant
OTHER = ("x", 100, 200)   # other doc


def test_ir_metrics_hand_computed():
    m = ir_metrics([OTHER, REL, PART], [SPAN], n_relevant_in_index=2, k=3)
    assert m["hit"] == 1 and m["recall"] == 1 and m["mrr"] == 0.5
    assert math.isclose(m["ndcg"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))  # IDCG uses 2 relevant chunks in the index
    assert math.isclose(m["precision"], 1 / 3) and math.isclose(m["ap"], 0.5)


def test_ir_metrics_miss_and_multi_span_recall():
    assert ir_metrics([OTHER, PART], [SPAN], 1, 2) == {"hit": 0, "recall": 0, "mrr": 0, "ndcg": 0, "precision": 0, "ap": 0}
    m = ir_metrics([REL], [SPAN, ("e", 0, 50)], 2, 1)
    assert m["recall"] == 0.5 and m["ndcg"] == 1.0


def test_text_metrics():
    assert exact_match("The Confabulation.", "confabulation") == 1
    assert math.isclose(token_f1("govern map measure", "GOVERN, MAP, MEASURE, and MANAGE"), 2 * 1 * 0.6 / 1.6)  # P=3/3, R=3/5
    assert token_f1("", "x") == 0 and contains_all("Released January 2023; by 2028", ["january 2023", "2028"]) == 1


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    a = bootstrap_ci([0, 1, 1, 1, 0, 1])
    assert a == bootstrap_ci([0, 1, 1, 1, 0, 1]) and a[1] <= a[0] <= a[2]
