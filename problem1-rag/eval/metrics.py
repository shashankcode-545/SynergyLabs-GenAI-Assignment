"""IR + answer metrics (pure functions, hand-tested in tests/test_metrics.py)."""
import math, re, string
from collections import Counter

import numpy as np

REL_MIN_COVER = 0.5


def covers(chunk, span):
    """chunk/span: (doc_id, start, end). Fraction of the span covered by the chunk."""
    if chunk[0] != span[0]:
        return 0.0
    return max(0, min(chunk[2], span[2]) - max(chunk[1], span[1])) / max(1, span[2] - span[1])


def is_relevant(chunk, spans):
    return any(covers(chunk, s) >= REL_MIN_COVER for s in spans)


def ir_metrics(ranked, spans, n_relevant_in_index, k):
    """ranked: [(doc_id, start, end)] in rank order; spans: gold evidence spans."""
    top = ranked[:k]
    rel = [is_relevant(c, spans) for c in top]
    first = next((i for i, r in enumerate(rel) if r), None)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, n_relevant_in_index)))
    prec_at = [sum(rel[:i + 1]) / (i + 1) for i in range(len(rel))]
    return {
        "hit": float(any(rel)),
        "recall": sum(any(covers(c, s) >= REL_MIN_COVER for c in top) for s in spans) / len(spans),
        "mrr": 0.0 if first is None else 1 / (first + 1),
        "ndcg": dcg / idcg if idcg else 0.0,
        "precision": sum(rel) / k,
        "ap": sum(p for p, r in zip(prec_at, rel) if r) / sum(rel) if any(rel) else 0.0,
    }


def _norm(s):
    s = s.lower().replace("’", "'")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(_norm(pred) == _norm(gold))


def token_f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def contains_all(pred, parts):
    return float(all(_norm(x) in _norm(pred) for x in parts))


def bootstrap_ci(values, n=10_000, seed=42):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None, None
    means = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
