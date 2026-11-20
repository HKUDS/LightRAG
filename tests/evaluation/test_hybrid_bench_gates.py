"""Gate verdict boundary pins: thresholds are proposals, verdicts never tune.

Every boundary case sits exactly on the proposal threshold or one epsilon
past it, with the expected status derived from the documented interpretation.
"""

from __future__ import annotations

import pytest

from lightrag.evaluation.hybrid_bench.gates import (
    GateInputs,
    GateThresholds,
    MetricComparison,
    evaluate_gates,
    gate_b2a_g,
    gate_b2a_j,
    gate_b2b_g,
    gate_b2b_j,
    gate_b3c,
    gate_b3d,
    gate_b4,
    gate_b5,
    gate_p95_overhead,
)

pytestmark = pytest.mark.offline

T = GateThresholds()


def test_proposal_threshold_values_are_pinned():
    assert T.b2a_j_relative_gain == 0.15
    assert T.b2a_g_relative_floor == -0.01
    assert T.b2b_j_relative_gain == 0.10
    assert T.b2b_g_absolute_delta == 0.005
    assert T.b3a_p95_overhead == 0.25
    assert T.b3b_p95_overhead == 0.15
    assert T.b3c_upsert_p95_increase == 0.10
    assert T.b3c_gin_vector_ratio == 1.5
    assert T.b3d_chunks_per_min == 5000.0
    assert T.b4_survival == 0.50


@pytest.mark.parametrize(
    ("hybrid", "ci_lower", "status"),
    [
        # exactly +15% (15/100 is float-exact) with CI lower > 0 -> pass.
        (115.0, 0.001, "pass"),
        # below +15% -> fail.
        (114.99, 0.001, "fail"),
        # +15% but CI lower touches zero -> fail (strict > 0).
        (115.0, 0.0, "fail"),
    ],
    ids=["at-threshold", "below-threshold", "ci-at-zero"],
)
def test_b2a_j_boundaries(hybrid, ci_lower, status):
    verdict = gate_b2a_j("terminology", 100.0, hybrid, ci_lower, T, "entity_hit@10")
    assert verdict.status == status
    assert verdict.gate_id == "entity-hit/terminology/entity_hit@10"


@pytest.mark.parametrize(
    ("hybrid", "ci_upper", "status"),
    [
        # exactly -1% (float-exact at dense=100); noise band = 1.0,
        # CI upper strictly inside -> pass.
        (99.0, 0.999, "pass"),
        # CI upper exactly at the band -> fail (strict <).
        (99.0, 1.0, "fail"),
        # below the -1% floor -> fail even with a tight CI.
        (98.9, 0.1, "fail"),
    ],
    ids=["at-floor", "ci-at-band", "below-floor"],
)
def test_b2a_g_boundaries(hybrid, ci_upper, status):
    verdict = gate_b2a_g("general", 100.0, hybrid, ci_upper, T, "entity_hit@10")
    assert verdict.status == status
    assert verdict.gate_id == "entity-hit/general/entity_hit@10"


def test_b2a_zero_dense_is_not_run():
    assert gate_b2a_j("J", 0.0, 0.5, 0.1, T, "m").status == "not-run"
    assert gate_b2a_g("G", 0.0, 0.5, 0.0, T, "m").status == "not-run"


@pytest.mark.parametrize(
    ("hybrid", "ci_lower", "status"),
    [
        (1.10, 0.001, "pass"),
        (1.0999, 0.001, "fail"),
        (1.10, -0.001, "fail"),
    ],
    ids=["at-threshold", "below-threshold", "negative-ci"],
)
def test_b2b_j_boundaries(hybrid, ci_lower, status):
    verdict = gate_b2b_j("J", "recall@10", 1.0, hybrid, ci_lower, T)
    assert verdict.status == status


@pytest.mark.parametrize(
    ("hybrid", "ci_upper", "status"),
    [
        # |delta| exactly 0.005 (dense 0 keeps the subtraction float-exact)
        # with CI upper inside -> pass.
        (0.005, 0.004, "pass"),
        # |delta| past 0.005 -> fail.
        (0.0051, 0.004, "fail"),
        # CI upper exactly 0.005 -> fail (strict <).
        (0.005, 0.005, "fail"),
    ],
    ids=["at-limit", "past-limit", "ci-at-limit"],
)
def test_b2b_g_boundaries(hybrid, ci_upper, status):
    verdict = gate_b2b_g("general", 0.0, hybrid, ci_upper, T)
    assert verdict.status == status


@pytest.mark.parametrize(
    ("overhead", "limit", "status"),
    [
        (0.25, 0.25, "pass"),
        (0.2501, 0.25, "fail"),
        (0.15, 0.15, "pass"),
        (0.1501, 0.15, "fail"),
        (None, 0.25, "not-run"),
    ],
    ids=["retrieval-at", "retrieval-past", "answer-at", "answer-past", "absent"],
)
def test_p95_overhead_boundaries(overhead, limit, status):
    assert gate_p95_overhead("B3x", overhead, limit).status == status


@pytest.mark.parametrize(
    ("increase", "ratio", "statuses"),
    [
        (0.10, 1.5, ["pass", "pass"]),
        (0.1001, 1.5, ["fail", "pass"]),
        (0.10, 1.5001, ["pass", "fail"]),
        (None, None, ["not-run", "not-run"]),
    ],
    ids=["both-at", "upsert-past", "gin-past", "absent"],
)
def test_b3c_boundaries(increase, ratio, statuses):
    verdicts = gate_b3c(increase, ratio, T)
    assert [v.status for v in verdicts] == statuses


def test_b3d_is_report_only():
    assert gate_b3d(100.0, T).status == "report"
    assert gate_b3d(99999.0, T).status == "report"
    assert gate_b3d(None, T).status == "not-run"


@pytest.mark.parametrize(
    ("survival", "status"),
    [
        (0.50, "pass"),
        (0.4999, "residue"),
        (0.0, "residue"),
        (None, "not-run"),
    ],
    ids=["at-floor", "below-floor", "zero", "absent"],
)
def test_b4_survival_is_residue_not_fail(survival, status):
    assert gate_b4(survival, T).status == status


@pytest.mark.parametrize(
    ("both", "entity_only", "chunk_only", "status"),
    [
        (0.6, 0.6, 0.6, "pass"),
        (0.7, 0.6, 0.5, "pass"),
        (0.59, 0.6, 0.5, "fail"),
        (0.59, 0.5, 0.6, "fail"),
    ],
    ids=["all-equal", "both-best", "below-entity", "below-chunk"],
)
def test_b5_both_ge_each_single(both, entity_only, chunk_only, status):
    assert gate_b5("recall@10", both, entity_only, chunk_only).status == status


def _comparison(dense, hybrid, diff_ci, reverse_ci, abs_ci):
    return MetricComparison(dense, hybrid, diff_ci, reverse_ci, abs_ci)


def test_evaluate_gates_all_absent_reports_not_run():
    verdicts = evaluate_gates(GateInputs({}, {}, {}))
    assert verdicts
    assert all(v.status == "not-run" for v in verdicts)
    ids = {v.gate_id for v in verdicts}
    assert "entity-hit/terminology/entity_hit@10" in ids
    assert "entity-hit/general/entity_hit@10" in ids
    assert "chunk-quality/terminology/recall@10" in ids
    assert "chunk-quality/terminology/ndcg@10" in ids
    assert "chunk-quality/general/ndcg@10" in ids
    # Terminology gets recall and nDCG; general gets nDCG only.
    assert "chunk-quality/general/recall@10" not in ids
    assert (
        sum(v.gate_id == "entity-hit/terminology/entity_hit@10" for v in verdicts) == 1
    )
    assert "retrieval-latency/retrieval-p95" in ids
    assert "answer-latency/answer-p95" in ids
    assert "write-amp/upsert-p95" in ids
    assert "write-amp/gin-size" in ids
    assert "backfill/throughput" in ids
    assert "rerank-survival" in ids
    assert "ablation/all" in ids


def test_evaluate_gates_mixed_verdicts():
    inputs = GateInputs(
        entity_hit={
            # terminology: +25% relative, CI lower 0.05 > 0 -> pass.
            "terminology": _comparison(
                0.4, 0.5, (0.05, 0.15), (-0.15, -0.05), (0.05, 0.15)
            ),
            # general: -1% relative (float-exact at dense=100), reverse CI
            # upper 0.999 < band 1.0 -> pass.
            "general": _comparison(100.0, 99.0, (-1.0, 0.1), (-0.1, 0.999), (0.0, 1.0)),
        },
        chunk_recall={
            # terminology: +10% relative (float-exact at dense=100),
            # CI lower 0.1 > 0 -> pass.
            "terminology": _comparison(
                100.0, 110.0, (0.1, 15.0), (-15.0, -0.1), (0.1, 15.0)
            )
        },
        chunk_ndcg={
            "terminology": _comparison(
                100.0, 110.0, (0.1, 15.0), (-15.0, -0.1), (0.1, 15.0)
            ),
            # general: |delta| 0.002 <= 0.005, abs CI upper 0.004 < 0.005
            # -> pass.
            "general": _comparison(
                0.8, 0.802, (-0.001, 0.004), (-0.004, 0.001), (0.0, 0.004)
            ),
        },
        retrieval_p95_overhead=0.20,
        answer_p95_overhead=0.30,
        upsert_increase=0.05,
        gin_vector_ratio=1.2,
        backfill_chunks_per_min=8000.0,
        rerank_survival=0.4,
        ablation={"recall@10": (0.7, 0.6, 0.6)},
    )
    verdicts = {v.gate_id: v.status for v in evaluate_gates(inputs)}
    assert verdicts["entity-hit/terminology/entity_hit@10"] == "pass"
    assert verdicts["entity-hit/general/entity_hit@10"] == "pass"
    assert verdicts["chunk-quality/terminology/recall@10"] == "pass"
    assert verdicts["chunk-quality/terminology/ndcg@10"] == "pass"
    assert verdicts["chunk-quality/general/ndcg@10"] == "pass"
    assert verdicts["retrieval-latency/retrieval-p95"] == "pass"
    assert verdicts["answer-latency/answer-p95"] == "fail"
    assert verdicts["write-amp/upsert-p95"] == "pass"
    assert verdicts["write-amp/gin-size"] == "pass"
    assert verdicts["backfill/throughput"] == "report"
    assert verdicts["rerank-survival"] == "residue"
    assert verdicts["ablation/recall@10"] == "pass"
