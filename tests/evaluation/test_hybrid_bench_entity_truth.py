"""Derived entity-truth mapping on a tiny fake KG, worked out by hand."""

from __future__ import annotations

import pytest

from lightrag.evaluation.hybrid_bench.entity_truth import derive_entity_truth

pytestmark = pytest.mark.offline


class FakeGraph:
    """Three entities: E1 spans two chunks, E2 one, E3 has no source."""

    def __init__(self):
        self.calls: list[str] = []

    async def get_all_labels(self):
        self.calls.append("labels")
        return ["E1", "E2", "E3"]

    async def get_nodes_batch(self, node_ids):
        self.calls.append("nodes")
        return {
            "E1": {"source_id": "c1<SEP>c2"},
            "E2": {"source_id": "c3"},
            "E3": {"source_id": ""},
        }


class FakeChunksKV:
    """c1->d1, c2->d2, c3->d3; missing ids yield None in request order."""

    def __init__(self, missing: set[str] | None = None):
        self.missing = missing or set()
        self.requested: list[list[str]] = []

    async def get_by_ids(self, ids):
        self.requested.append(list(ids))
        owners = {"c1": "d1", "c2": "d2", "c3": "d3"}
        return [
            None if cid in self.missing else {"full_doc_id": owners[cid]} for cid in ids
        ]


@pytest.mark.asyncio
async def test_derived_truth_hand_computed():
    graph, kv = FakeGraph(), FakeChunksKV()
    positives = {
        "q1": {"d1": 1},
        "q2": {"d3": 2},
        "q3": {"d9": 1},
    }
    truth = await derive_entity_truth(graph, kv, positives)
    # E1 owns {d1,d2} -> relevant to q1 only; E2 owns {d3} -> q2;
    # q3's positive d9 matches nothing -> zero-truth, reported separately.
    assert truth.per_query == {"q1": {"E1"}, "q2": {"E2"}}
    assert truth.zero_truth_query_ids == ["q3"]
    assert truth.entity_documents == {"E1": {"d1", "d2"}, "E2": {"d3"}}
    assert kv.requested == [["c1", "c2", "c3"]]
    assert graph.calls == ["labels", "nodes"]


@pytest.mark.asyncio
async def test_missing_chunk_row_drops_only_that_document():
    graph, kv = FakeGraph(), FakeChunksKV(missing={"c2"})
    truth = await derive_entity_truth(graph, kv, {"q1": {"d2": 1}})
    # E1's remaining chunk c1 owns d1, not d2 -> no derivable truth.
    assert truth.per_query == {}
    assert truth.zero_truth_query_ids == ["q1"]
    assert truth.entity_documents == {"E1": {"d1"}, "E2": {"d3"}}


class FakeTrackingKV:
    """entity_chunks rows keyed by entity name; absent ids yield None in
    request order, the BaseKVStorage.get_by_ids contract."""

    def __init__(self, rows: dict[str, dict]):
        self.rows = rows
        self.requested: list[list[str]] = []

    async def get_by_ids(self, ids):
        self.requested.append(list(ids))
        return [self.rows.get(label) for label in ids]


@pytest.mark.asyncio
async def test_tracking_row_extends_truth_beyond_the_graph_cap():
    # E1's graph source_id is capped at c1,c2; the tracking row carries the
    # FULL list including c3. A query positive only on d3 (owned by c3) must
    # still derive E1 as truth — the capped graph field alone would call it
    # zero-truth and drop it from hit@k.
    graph = FakeGraph()
    kv = FakeChunksKV()
    tracking = FakeTrackingKV({"E1": {"chunk_ids": ["c1", "c2", "c3"], "count": 3}})
    truth = await derive_entity_truth(
        graph, kv, {"q1": {"d3": 1}}, entity_chunks=tracking
    )
    # E2 legitimately owns d3 through its graph fallback; the pin is that
    # E1 is ALSO truth via the beyond-cap tracking chunk c3.
    assert truth.per_query == {"q1": {"E1", "E2"}}
    assert truth.zero_truth_query_ids == []
    assert truth.entity_documents["E1"] == {"d1", "d2", "d3"}
    assert tracking.requested == [["E1", "E2", "E3"]]
    assert kv.requested == [["c1", "c2", "c3"]]


@pytest.mark.asyncio
async def test_absent_tracking_row_falls_back_to_graph_source_id():
    # Only E1 has a tracking row; E2/E3 fall back to the graph field.
    graph = FakeGraph()
    kv = FakeChunksKV()
    tracking = FakeTrackingKV({"E1": {"chunk_ids": ["c1"], "count": 1}})
    truth = await derive_entity_truth(
        graph, kv, {"q1": {"d1": 1}, "q2": {"d3": 1}}, entity_chunks=tracking
    )
    assert truth.per_query == {"q1": {"E1"}, "q2": {"E2"}}
    assert truth.entity_documents == {"E1": {"d1"}, "E2": {"d3"}}


@pytest.mark.asyncio
async def test_present_empty_tracking_row_is_authoritative():
    # A present row with an empty chunk list means no source chunks; the
    # capped graph field must NOT resurrect documents for it.
    graph = FakeGraph()
    kv = FakeChunksKV()
    tracking = FakeTrackingKV({"E1": {"chunk_ids": [], "count": 0}})
    truth = await derive_entity_truth(
        graph, kv, {"q1": {"d1": 1}}, entity_chunks=tracking
    )
    assert truth.per_query == {}
    assert truth.zero_truth_query_ids == ["q1"]
    assert "E1" not in truth.entity_documents


@pytest.mark.asyncio
async def test_empty_graph_yields_all_zero_truth():
    class EmptyGraph:
        async def get_all_labels(self):
            return []

    class NeverCalledKV:
        def __init__(self):
            self.calls = []

        async def get_by_ids(self, ids):
            self.calls.append(ids)
            raise AssertionError("no chunk ids means no KV lookup")

    kv = NeverCalledKV()
    truth = await derive_entity_truth(EmptyGraph(), kv, {"q1": {"d1": 1}})
    assert truth.per_query == {}
    assert truth.zero_truth_query_ids == ["q1"]
    assert kv.calls == []
