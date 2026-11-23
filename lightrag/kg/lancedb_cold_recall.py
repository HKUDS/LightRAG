"""Recall comparison between DuckDB cold ids and a hot vector baseline.

Rules:

- Inputs are ranked id lists produced from the SAME stored vectors; the harness
  never embeds, reads a storage backend or writes anything.
- ``recall_at_k`` divides by the de-duplicated hot top-k, so a thresholded hot
  baseline that returns fewer than k rows does not penalise the cold ranking.
- ``paired_bootstrap_recall`` resamples QUERIES, not hits, and is deterministic
  for a given seed. One query yields a zero-width range.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RecallReport:
    """Per-query recall@k and its paired-bootstrap 95% range."""

    k: int
    per_query: tuple[float, ...]
    mean: float
    ci_low: float
    ci_high: float
    samples: int


def recall_at_k(cold_ids: Sequence[str], hot_ids: Sequence[str], k: int) -> float:
    """Fraction of the hot top-k found in the cold top-k; 0.0 when hot is empty."""
    if k < 1:
        raise ValueError("k must be >= 1")
    hot = list(dict.fromkeys(str(hit_id) for hit_id in hot_ids))[:k]
    if not hot:
        return 0.0
    cold = {str(hit_id) for hit_id in cold_ids[:k]}
    return sum(1 for hit_id in hot if hit_id in cold) / len(hot)


def paired_bootstrap_recall(
    cold_runs: Sequence[Sequence[str]],
    hot_runs: Sequence[Sequence[str]],
    k: int,
    *,
    samples: int = 1000,
    seed: int = 0,
) -> RecallReport:
    """Mean paired recall@k with a 95% percentile bootstrap range."""
    cold = list(cold_runs)
    hot = list(hot_runs)
    if len(cold) != len(hot):
        raise ValueError("cold_runs and hot_runs must have the same length")
    if not cold:
        raise ValueError("at least one paired query is required")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    per_query = tuple(recall_at_k(c, h, k) for c, h in zip(cold, hot))
    rng = random.Random(seed)
    count = len(per_query)
    means = sorted(
        sum(per_query[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return RecallReport(
        k=k,
        per_query=per_query,
        mean=sum(per_query) / count,
        ci_low=means[int(0.025 * (samples - 1))],
        ci_high=means[int(0.975 * (samples - 1))],
        samples=samples,
    )
