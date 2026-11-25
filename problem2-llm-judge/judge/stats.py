"""Small statistics helpers (tested by hand-computed values in tests/test_stats.py)."""
import math
from collections import Counter

import numpy as np


def bootstrap_ci(values, n=10_000, seed=42):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None, None
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sign_test_p(wins, losses):
    """Exact two-sided binomial sign test (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def auroc(pos, neg):
    """P(score of a positive > score of a negative), ties count 1/2."""
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def norm_entropy(values, levels):
    """Shannon entropy of the score distribution / log(levels): 0 = all one score, 1 = uniform over the scale."""
    c = Counter(values)
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values()) / math.log(levels) if n else None


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    def avg_rank(v):  # average ranks for ties
        v = np.asarray(v, float)
        r = rank(v)
        for val in np.unique(v):
            idx = v == val
            r[idx] = r[idx].mean()
        return r
    rx, ry = avg_rank(x), avg_rank(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and rx.std() and ry.std() else None
