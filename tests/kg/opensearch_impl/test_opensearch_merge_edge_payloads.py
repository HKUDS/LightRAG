"""_merge_edge_payloads folds duplicate edge-doc fragments together when
consolidating legacy reciprocal edges onto the canonical edge id. A fragment
can contribute new source_ids while carrying a missing, non-numeric, or
non-finite legacy weight; the plain sum used for the result's weight can
then fall below the merged evidence count, or overflow to +inf, either of
which violates the relation weight contract documented in AGENTS.md
(weight >= len(distinct real source IDs)).
"""

import pytest

pytest.importorskip(
    "opensearchpy", reason="opensearchpy is required for OpenSearch storage tests"
)

from lightrag.kg.opensearch_impl import _coerce_weight, _merge_edge_payloads

pytestmark = pytest.mark.offline


def test_coerce_weight_rejects_non_finite_values():
    assert _coerce_weight("NaN") is None
    assert _coerce_weight(float("inf")) is None
    assert _coerce_weight("-Infinity") is None
    assert _coerce_weight("2.5") == 2.5
    assert _coerce_weight(None) is None
    assert _coerce_weight("abc") is None


def test_fragment_with_missing_weight_does_not_undercut_evidence_floor():
    """Base carries weight=1.0 for source_ids=["chunk1"]. The duplicate
    contributes a brand-new source_id but has no weight field at all. A
    plain sum would keep the result at 1.0 even though the merged evidence
    count is now 2."""
    docs = [
        {"source_ids": ["chunk1"], "weight": 1.0},
        {"source_ids": ["chunk2"]},
    ]

    merged = _merge_edge_payloads(docs)

    assert merged["source_ids"] == ["chunk1", "chunk2"]
    assert merged["weight"] >= 2


def test_all_fragments_missing_weight_still_gets_the_evidence_floor():
    """If no fragment has a coercible weight at all, the merge must still
    set weight to the evidence count instead of leaving the field unset."""
    docs = [
        {"source_ids": ["chunk1"]},
        {"source_ids": ["chunk2"]},
    ]

    merged = _merge_edge_payloads(docs)

    assert merged["weight"] == 2


def test_non_finite_legacy_weight_is_skipped_not_raised():
    """A NaN/inf legacy weight parses fine through float() but is rejected
    by apply_relation_weight_floor's storability check. The merge must skip
    it like any other unusable legacy weight, not raise."""
    docs = [
        {"source_ids": ["chunk1"], "weight": float("nan")},
        {"source_ids": ["chunk2"]},
    ]

    merged = _merge_edge_payloads(docs)

    assert merged["weight"] == 2


def test_weight_sum_overflow_falls_back_to_the_evidence_floor():
    """Each weight is individually finite (1e308 passes _coerce_weight's
    isfinite check), but summing two of them overflows to +inf.
    apply_relation_weight_floor rejects a non-finite aggregate outright --
    the merge must fall back to the evidence-count floor instead of
    raising."""
    docs = [
        {"source_ids": ["chunk1"], "weight": 1e308},
        {"source_ids": ["chunk2"], "weight": 1e308},
    ]

    merged = _merge_edge_payloads(docs)

    assert merged["weight"] == 2


def test_source_less_fragments_keep_the_plain_sum():
    """A relation with no real source IDs is exempt from the evidence floor
    (any non-negative weight is valid), so this path is unaffected."""
    docs = [
        {"weight": 1.5},
        {"weight": 2.5},
    ]

    merged = _merge_edge_payloads(docs)

    assert "source_ids" not in merged
    assert merged["weight"] == 4.0
