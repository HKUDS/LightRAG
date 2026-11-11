"""``_merge_edge_payloads`` folds duplicate/reciprocal legacy edge docs into one
canonical doc while OpenSearch-backed deployments migrate onto the canonical
edge id. A fragment can contribute new source_ids while carrying a missing or
non-numeric legacy weight; a plain sum then leaves the merged weight below the
merged evidence count, violating the relation weight contract documented in
AGENTS.md (weight >= len(distinct real source IDs)). Mirrors the Mongo dedupe
coverage in tests/kg/mongo_impl/test_mongo_dedupe_legacy_edges.py.
"""

import sys

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from lightrag.kg.opensearch_impl import _merge_edge_payloads

pytestmark = pytest.mark.offline


def test_fragment_with_missing_weight_does_not_undercut_evidence_floor():
    """The base carries weight=1.0 for one chunk; the folded fragment brings a
    brand-new source_id but no weight field at all (a pre-contract legacy row).
    A plain sum would keep the result at 1.0 with an evidence count of 2."""
    merged = _merge_edge_payloads(
        [
            {"source_ids": ["chunk1"], "weight": 1.0},
            {"source_ids": ["chunk2"]},
        ]
    )

    assert merged["source_ids"] == ["chunk1", "chunk2"]
    assert merged["weight"] >= 2


def test_all_fragments_missing_weight_still_get_the_evidence_floor():
    """With no coercible weight anywhere, the merge must still emit a weight:
    omitting it would layer the merged source_ids onto a doc whose stored
    weight was never raised to match them."""
    merged = _merge_edge_payloads(
        [{"source_ids": ["chunk1"]}, {"source_ids": ["chunk2"]}]
    )

    assert merged["weight"] == 2


def test_weight_sum_overflow_is_clamped_to_the_largest_finite_weight():
    """Individually finite weights can still sum to +inf, which no graph
    backend can store. Clamp to the largest representable weight rather than
    writing +inf or collapsing the magnitude down to the evidence count."""
    merged = _merge_edge_payloads(
        [
            {"source_ids": ["chunk1"], "weight": 1e308},
            {"source_ids": ["chunk2"], "weight": 1e308},
        ]
    )

    assert merged["weight"] == sys.float_info.max


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), "NaN", "-Infinity"])
def test_non_finite_legacy_weight_is_skipped_not_propagated(bad_weight):
    """A NaN/inf legacy weight parses fine through float() but no backend can
    store one, and a NaN would poison the sum/max it reaches. Skip it like any
    other unusable legacy weight and fall back to the evidence floor."""
    merged = _merge_edge_payloads(
        [
            {"source_ids": ["chunk1"], "weight": bad_weight},
            {"source_ids": ["chunk2"]},
        ]
    )

    assert merged["weight"] == 2


def test_unstorable_legacy_source_id_does_not_raise():
    """A legacy source_id can hold a character no graph backend can store.
    Routing the floor through `apply_relation_weight_floor` -- which validates
    the whole relation the way a caller ingress does -- would raise on that row
    and abort the one-time migration it is supposed to carry through."""
    merged = _merge_edge_payloads(
        [
            {"source_ids": ["chunk-\x0bbad"], "weight": 1.0},
            {"source_ids": ["chunk2"]},
        ]
    )

    assert merged["weight"] >= 2


def test_placeholder_only_evidence_leaves_a_missing_weight_alone():
    """`manual_creation`/`UNKNOWN` are not evidence, so the floor is zero. With
    no weight on any fragment the merge must omit "weight" entirely: the merged
    payload is layered over the surviving doc, so emitting 0.0 would overwrite
    whatever that doc held (or the 1.0 a missing weight reads as)."""
    merged = _merge_edge_payloads(
        [{"source_ids": ["UNKNOWN"]}, {"source_ids": ["manual_creation"]}]
    )

    assert "weight" not in merged
    assert merged["source_ids"] == ["UNKNOWN", "manual_creation"]


def test_placeholder_only_evidence_keeps_a_real_weight():
    """The guard above must skip only the write it has nothing to say about: a
    source-less relation may carry any non-negative fractional weight, and the
    base's own 0.5 has to survive the merge."""
    merged = _merge_edge_payloads(
        [
            {"source_ids": ["UNKNOWN"], "weight": 0.5},
            {"source_ids": ["manual_creation"]},
        ]
    )

    assert merged["weight"] == 0.5


def test_negative_weight_sum_overflow_takes_the_evidence_floor():
    """The overflow clamp must not pull a hugely NEGATIVE sum up to the largest
    positive weight. Summing two finite negatives underflows to -inf, which the
    evidence floor -- not the clamp -- is what absorbs."""
    merged = _merge_edge_payloads(
        [
            {"source_ids": ["chunk1"], "weight": -1e308},
            {"source_ids": ["chunk2"], "weight": -1e308},
        ]
    )

    assert merged["weight"] == 2
