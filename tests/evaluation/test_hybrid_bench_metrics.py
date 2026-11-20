"""Hand-computed metric fixtures and bootstrap determinism pins.

Expected values are worked out by hand in the comments, never recomputed
with the code under test.
"""

from __future__ import annotations

import pytest

from lightrag.evaluation.hybrid_bench.metrics import (
    bootstrap_mean_ci,
    bootstrap_p95_overhead_ci,
    hit_at_k,
    mean,
    ndcg_at_k,
    p95_overhead,
    paired_diff_ci,
    percentile,
    recall_at_k,
    relative_gain,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("retrieved", "k", "expected"),
    [
        # positives {d1, d2}: ranks 1..3 hold d1 and d2 -> 2/2.
        (["d1", "d3", "d2"], 3, 1.0),
        # only d1 in the first rank -> 1/2.
        (["d1", "d3", "d2"], 1, 0.5),
        # nothing relevant retrieved -> 0.
        (["d3", "d4"], 2, 0.0),
        # empty retrieval is a real zero, not an error.
        ([], 10, 0.0),
    ],
    ids=["full", "half", "miss", "empty"],
)
def test_recall_at_k_hand_computed(retrieved, k, expected):
    assert recall_at_k(retrieved, {"d1": 1, "d2": 1}, k) == expected


def test_ndcg_at_k_hand_computed():
    # positives a:2, b:1; retrieved [b, x, a, b] dedups to [b, x, a].
    # DCG  = 1/log2(2) + 0/log2(3) + 2/log2(4) = 1 + 0 + 1 = 2.0
    # IDCG = 2/log2(2) + 1/log2(3) = 2 + 0.6309297535714575
    # nDCG = 2.0 / 2.6309297535714575 = 0.7601875334318685
    positives = {"a": 2, "b": 1}
    assert ndcg_at_k(["b", "x", "a", "b"], positives, 4) == pytest.approx(
        0.7601875334318685, abs=1e-12
    )


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        # ideal order [a, b]: DCG == IDCG -> 1.0.
        (["a", "b"], 1.0),
        # a repeated relevant doc counts once: DCG = 1 = IDCG.
        (["a", "a"], 1.0),
        # nothing retrieved -> 0.0.
        ([], 0.0),
    ],
    ids=["ideal", "repeat-dedup", "empty"],
)
def test_ndcg_at_k_bounded_cases(retrieved, expected):
    assert ndcg_at_k(retrieved, {"a": 1}, 10) == expected


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_empty_ground_truth_raises(metric):
    with pytest.raises(ValueError, match="positives"):
        metric(["d1"], {}, 10)


@pytest.mark.parametrize(
    ("retrieved", "k", "expected"),
    [
        (["e3", "e1"], 2, 1.0),
        (["e3", "e1"], 1, 0.0),
        ([], 10, 0.0),
    ],
    ids=["hit-second-rank", "miss-at-one", "empty"],
)
def test_hit_at_k_hand_computed(retrieved, k, expected):
    assert hit_at_k(retrieved, {"e1", "e2"}, k) == expected


def test_hit_at_k_empty_truth_raises():
    with pytest.raises(ValueError, match="truth"):
        hit_at_k(["e1"], set(), 10)


@pytest.mark.parametrize(
    ("values", "p", "expected"),
    [
        # nearest rank: ceil(0.5*10)=5 -> 5th smallest = 5.
        (list(range(1, 11)), 50, 5),
        # ceil(0.95*10)=10 -> 10.
        (list(range(1, 11)), 95, 10),
        # ceil(0.5*4)=2 -> 2.
        ([1, 2, 3, 4], 50, 2),
    ],
    ids=["p50-ten", "p95-ten", "p50-four"],
)
def test_percentile_hand_computed(values, p, expected):
    assert percentile(values, p) == expected


def test_relative_gain_and_overhead():
    assert relative_gain(1.15, 1.0) == pytest.approx(0.15)
    assert relative_gain(0.5, 0.0) is None
    assert p95_overhead(2.0, 2.5) == pytest.approx(0.25)
    assert p95_overhead(0.0, 1.0) is None


def test_bootstrap_ci_fixed_seed_is_deterministic():
    # Many distinct values make the resample-mean percentiles seed-sensitive;
    # a tiny discrete fixture can collide across seeds and hide an unseeded RNG.
    values = [i / 97 for i in range(40)]
    first = bootstrap_mean_ci(values, seed=7, resamples=1000)
    second = bootstrap_mean_ci(values, seed=7, resamples=1000)
    assert first == second
    lo, hi = first
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_degenerate_constant_values():
    # every resample mean equals the constant -> CI collapses to it.
    # 0.5 is binary-exact, so the resample sums carry no float drift.
    assert bootstrap_mean_ci([0.5] * 6, seed=1, resamples=1000) == (0.5, 0.5)


def test_paired_diff_ci_degenerate_arms():
    # constant gap of 1.0 in every resample.
    assert paired_diff_ci([1.0] * 5, [0.0] * 5, seed=3, resamples=1000) == (
        1.0,
        1.0,
    )
    # identical arms -> zero gap, absolute variant too.
    assert paired_diff_ci(
        [0.2, 0.9], [0.2, 0.9], seed=3, resamples=1000, absolute=True
    ) == (0.0, 0.0)


def test_paired_diff_ci_seed_determinism():
    a = [i / 53 for i in range(30)]
    b = [(i % 7) / 11 for i in range(30)]
    assert paired_diff_ci(a, b, seed=11, resamples=1000) == paired_diff_ci(
        a, b, seed=11, resamples=1000
    )


def test_p95_overhead_ci_constant_latency():
    # dense 1.0, hybrid 1.25 everywhere -> overhead 0.25 in every resample.
    lo, hi = bootstrap_p95_overhead_ci([1.0] * 4, [1.25] * 4, seed=5, resamples=1000)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: bootstrap_mean_ci([], seed=1), "values"),
        (lambda: bootstrap_mean_ci([0.5], seed=1, resamples=0), "resamples"),
        (lambda: paired_diff_ci([0.5], [0.1, 0.2], seed=1), "equal length"),
        (lambda: percentile([], 50), "values"),
        (lambda: percentile([1.0], 0), "p must be"),
        (lambda: mean([]), "values"),
    ],
    ids=[
        "empty-bootstrap",
        "zero-resamples",
        "unequal-arms",
        "empty-percentile",
        "bad-percentile",
        "empty-mean",
    ],
)
def test_metric_fault_paths_fail_loud(call, message):
    with pytest.raises(ValueError, match=message):
        call()
