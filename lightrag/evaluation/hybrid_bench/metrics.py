"""Retrieval metrics and fixed-seed bootstrap confidence intervals.

Conventions, pinned by hand-computed tests:
- ``recall_at_k``: distinct retrieved document ids in the first k ranks,
  divided by the number of qrels-positive documents.
- ``ndcg_at_k``: linear gain equal to the qrels score, discount
  ``1 / log2(rank + 1)``, ranks 1-based. Chunk ranks map to documents, so a
  repeated document counts ONCE at its first occurrence, keeping the score
  in [0, 1]; IDCG ranks all positive scores descending, truncated at k.
- ``hit_at_k``: 1.0 when any of the first k retrieved ids is in the truth set.
- Empty ground truth is a caller error, not a zero: raise ``ValueError`` so a
  query without derivable truth is reported separately, never averaged in.
- Bootstrap: per-query resampling with replacement from ``random.Random(seed)``;
  the same seed and inputs give byte-identical intervals. Paired statistics
  resample ONE index list shared by both arms. Interval endpoints are
  nearest-rank percentiles of the sorted bootstrap statistics.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence


def _require_nonempty(name: str, values: Sequence) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")


def recall_at_k(
    retrieved_docs: Sequence[str], positives: Mapping[str, int], k: int
) -> float:
    """Fraction of positive documents present in the first k retrieved ranks."""
    _require_nonempty("positives", positives)
    if k <= 0:
        raise ValueError("k must be positive")
    hits = set(retrieved_docs[:k]) & set(positives)
    return len(hits) / len(positives)


def ndcg_at_k(
    retrieved_docs: Sequence[str], positives: Mapping[str, int], k: int
) -> float:
    """nDCG with linear qrels-score gain; see the module docstring rules."""
    _require_nonempty("positives", positives)
    if k <= 0:
        raise ValueError("k must be positive")

    def dcg(gains: Sequence[float]) -> float:
        return sum(
            gain / math.log2(rank + 1) for rank, gain in enumerate(gains[:k], start=1)
        )

    ideal = sorted(positives.values(), reverse=True)
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    seen: set[str] = set()
    unique_docs = []
    for doc in retrieved_docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
    gains = [positives.get(doc, 0) for doc in unique_docs]
    return dcg(gains) / idcg


def hit_at_k(retrieved: Sequence[str], truth: set[str], k: int) -> float:
    """1.0 when the first k retrieved ids intersect the truth set, else 0.0."""
    _require_nonempty("truth", truth)
    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 if set(retrieved[:k]) & truth else 0.0


def mean(values: Sequence[float]) -> float:
    _require_nonempty("values", values)
    return sum(values) / len(values)


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: ``ceil(p/100 * n)``-th smallest, 1-based."""
    _require_nonempty("values", values)
    if not 0.0 < p <= 100.0:
        raise ValueError("p must be in (0, 100]")
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[index]


def relative_gain(candidate: float, baseline: float) -> float | None:
    """``(candidate - baseline) / baseline``; None when the baseline is zero."""
    if baseline == 0.0:
        return None
    return (candidate - baseline) / baseline


def p95_overhead(dense_p95: float, hybrid_p95: float) -> float | None:
    """Relative p95 overhead of hybrid versus dense; None on a zero base."""
    return relative_gain(hybrid_p95, dense_p95)


def _resample_indices(rng: random.Random, n: int, resamples: int) -> list[list[int]]:
    return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]


def _ci_from_statistics(statistics: list[float]) -> tuple[float, float]:
    ordered = sorted(statistics)
    count = len(ordered)
    lo = ordered[max(0, math.ceil(0.025 * count) - 1)]
    hi = ordered[min(count - 1, math.ceil(0.975 * count) - 1)]
    return lo, hi


def bootstrap_mean_ci(
    values: Sequence[float], seed: int, resamples: int = 1000
) -> tuple[float, float]:
    """95% CI of the mean via per-query resampling; deterministic per seed."""
    _require_nonempty("values", values)
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    n = len(values)
    statistics = [
        sum(values[i] for i in indices) / n
        for indices in _resample_indices(rng, n, resamples)
    ]
    return _ci_from_statistics(statistics)


def paired_diff_ci(
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    seed: int,
    resamples: int = 1000,
    absolute: bool = False,
) -> tuple[float, float]:
    """95% CI of ``mean(arm_a) - mean(arm_b)`` under PAIRED resampling: one
    index list per resample feeds both arms. ``absolute=True`` bootstraps
    ``|mean(arm_a) - mean(arm_b)|`` instead, for non-inferiority gates."""
    _require_nonempty("arm_a", arm_a)
    if len(arm_a) != len(arm_b):
        raise ValueError("paired arms must have equal length")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    n = len(arm_a)
    statistics = []
    for indices in _resample_indices(rng, n, resamples):
        diff = sum(arm_a[i] - arm_b[i] for i in indices) / n
        statistics.append(abs(diff) if absolute else diff)
    return _ci_from_statistics(statistics)


def bootstrap_p95_overhead_ci(
    dense_per_query: Sequence[float],
    hybrid_per_query: Sequence[float],
    seed: int,
    resamples: int = 1000,
) -> tuple[float, float]:
    """95% CI of the p95 overhead ``(p95_hybrid - p95_dense) / p95_dense``
    under paired per-query resampling. A zero dense p95 in any resample is a
    caller error: real latencies are positive."""
    _require_nonempty("dense_per_query", dense_per_query)
    if len(dense_per_query) != len(hybrid_per_query):
        raise ValueError("paired arms must have equal length")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    n = len(dense_per_query)
    statistics = []
    for indices in _resample_indices(rng, n, resamples):
        dense_p95 = percentile([dense_per_query[i] for i in indices], 95)
        hybrid_p95 = percentile([hybrid_per_query[i] for i in indices], 95)
        if dense_p95 <= 0.0:
            raise ValueError("dense p95 must be positive for an overhead CI")
        statistics.append((hybrid_p95 - dense_p95) / dense_p95)
    return _ci_from_statistics(statistics)
