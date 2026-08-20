"""Regression test: amerge_entities must not lose relation evidence-count when
the merged entities each already hold a separate edge to the same third node.

`_merge_entities_impl` (lightrag/utils_graph.py) merges a relation's
`source_id` with `join_unique` (a lossless union of every contributing
chunk) but, before this fix, merged its `weight` with `max`. Elsewhere
(`operate.py::_merge_edges_then_upsert`, resolving issue #3367) `weight` is
documented and maintained as the count of distinct contributing sources, and
`_find_most_related_edges_from_entities` sorts retrieval context by
`(rank, weight)`. So a `max`-merged edge can silently under-report its true
evidence count and rank lower than it should.
"""

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag import utils_graph


class _MergeGraphStorage:
    """Two source entities (A, B) each hold a separate edge to a shared
    third node C, mirroring the case the "max" strategy mishandled."""

    def __init__(self):
        self.nodes = {
            "A": {
                "entity_id": "A",
                "description": "Entity A",
                "entity_type": "ORG",
                "source_id": "chunk-a-node",
                "file_path": "docA.txt",
            },
            "B": {
                "entity_id": "B",
                "description": "Entity B",
                "entity_type": "ORG",
                "source_id": "chunk-b-node",
                "file_path": "docB.txt",
            },
            "C": {
                "entity_id": "C",
                "description": "Entity C",
                "entity_type": "ORG",
                "source_id": "chunk-c-node",
                "file_path": "docC.txt",
            },
        }
        self.edges = {
            ("A", "C"): {
                "description": "A relates to C",
                "keywords": "k1",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-a1", "chunk-a2", "chunk-a3"]),
                "weight": 3.0,
                "file_path": "docA.txt",
            },
            ("B", "C"): {
                "description": "B relates to C",
                "keywords": "k2",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-b1", "chunk-b2"]),
                "weight": 2.0,
                "file_path": "docB.txt",
            },
        }

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        return self.nodes[node_id]

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def get_node_edges(self, node_id):
        return [
            (src, tgt) for (src, tgt) in self.edges if src == node_id or tgt == node_id
        ]

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = {
            (src, tgt): data
            for (src, tgt), data in self.edges.items()
            if src != node_id and tgt != node_id
        }

    async def index_done_callback(self):
        return True


class _DummyVectorStorage:
    def __init__(self):
        self.global_config = {"workspace": "test"}
        self.upserts = []
        self.deletes = []

    async def upsert(self, data):
        self.upserts.append(data)
        return None

    async def delete(self, ids):
        self.deletes.append(ids)
        return None

    async def get_by_id(self, id_):
        return None

    async def index_done_callback(self):
        return True


@pytest.mark.asyncio
async def test_merge_entities_relation_weight_matches_distinct_source_count():
    graph = _MergeGraphStorage()
    entities_vdb = _DummyVectorStorage()
    relationships_vdb = _DummyVectorStorage()

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["A", "B"],
        target_entity="AB",
    )

    merged_edge = graph.edges[("AB", "C")]
    source_chunks = [s for s in merged_edge["source_id"].split(GRAPH_FIELD_SEP) if s]

    # source_id is joined losslessly: all 5 distinct chunks survive.
    assert set(source_chunks) == {
        "chunk-a1",
        "chunk-a2",
        "chunk-a3",
        "chunk-b1",
        "chunk-b2",
    }
    # weight must track that same distinct-source count, not max(3.0, 2.0).
    assert merged_edge["weight"] == len(source_chunks) == 5.0

    # The vector-store payload the API actually returns must agree.
    vdb_payload = relationships_vdb.upserts[-1]
    persisted = next(iter(vdb_payload.values()))
    assert persisted["weight"] == 5.0


class _ExplicitWeightGraphStorage(_MergeGraphStorage):
    """Both edges carry the legacy no-source marker and explicit boosts.

    ``manual_creation`` was historically synthesized when callers omitted a
    source. It contributes no evidence floor, while the strongest explicit
    weight must still survive the merge.
    """

    def __init__(self):
        super().__init__()
        self.edges = {
            ("A", "C"): {
                "description": "A relates to C",
                "keywords": "k1",
                "source_id": "manual_creation",
                "weight": 10.0,
                "file_path": "docA.txt",
            },
            ("B", "C"): {
                "description": "B relates to C",
                "keywords": "k2",
                "source_id": "manual_creation",
                "weight": 8.0,
                "file_path": "docB.txt",
            },
        }


class _SourcedLowWeightGraphStorage(_MergeGraphStorage):
    """Legacy/permissive input below the evidence-count floor.

    Each distinct non-empty source contributes a baseline weight of 1. These
    explicit values are therefore invalid under the unified relation-weight
    contract, and entity merge must normalize the resulting edge upward.
    """

    def __init__(self):
        super().__init__()
        self.edges = {
            ("A", "C"): {
                "description": "A relates to C",
                "keywords": "k1",
                "source_id": "manual-a",
                "weight": 0.25,
                "file_path": "docA.txt",
            },
            ("B", "C"): {
                "description": "B relates to C",
                "keywords": "k2",
                "source_id": "manual-b",
                "weight": 0.5,
                "file_path": "docB.txt",
            },
        }


class _LegacySourceLessLowWeightGraphStorage(_ExplicitWeightGraphStorage):
    def __init__(self):
        super().__init__()
        self.edges[("A", "C")]["weight"] = 0.25
        self.edges[("B", "C")]["weight"] = 0.5


class _SingleSourcedLowWeightGraphStorage(_MergeGraphStorage):
    def __init__(self):
        super().__init__()
        self.edges = {
            ("A", "C"): {
                "description": "A relates to C",
                "keywords": "k1",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-1", "chunk-2"]),
                "weight": 0.25,
                "file_path": "docA.txt",
            }
        }


@pytest.mark.asyncio
async def test_merge_entities_relation_weight_never_regresses_below_max_input():
    graph = _ExplicitWeightGraphStorage()
    entities_vdb = _DummyVectorStorage()
    relationships_vdb = _DummyVectorStorage()

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["A", "B"],
        target_entity="AB",
    )

    merged_edge = graph.edges[("AB", "C")]

    # The legacy marker contributes no evidence, while the explicit boost is
    # monotonic and therefore stays at max(10.0, 8.0) == 10.0.
    assert merged_edge["weight"] == 10.0

    vdb_payload = relationships_vdb.upserts[-1]
    persisted = next(iter(vdb_payload.values()))
    assert persisted["weight"] == 10.0


@pytest.mark.asyncio
async def test_merge_entities_preserves_legacy_source_less_fractional_weight():
    graph = _LegacySourceLessLowWeightGraphStorage()
    entities_vdb = _DummyVectorStorage()
    relationships_vdb = _DummyVectorStorage()

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["A", "B"],
        target_entity="AB",
    )

    merged_edge = graph.edges[("AB", "C")]
    assert merged_edge["source_id"] == "manual_creation"
    assert merged_edge["weight"] == 0.5


@pytest.mark.asyncio
async def test_merge_entities_lifts_sourced_low_weight_to_evidence_floor():
    graph = _SourcedLowWeightGraphStorage()
    entities_vdb = _DummyVectorStorage()
    relationships_vdb = _DummyVectorStorage()

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["A", "B"],
        target_entity="AB",
    )

    merged_edge = graph.edges[("AB", "C")]
    assert set(merged_edge["source_id"].split(GRAPH_FIELD_SEP)) == {
        "manual-a",
        "manual-b",
    }
    # Both inputs are below the contract's one-per-source baseline. The merge
    # intentionally normalizes the edge to its two-source evidence floor.
    assert merged_edge["weight"] == 2.0

    vdb_payload = relationships_vdb.upserts[-1]
    persisted = next(iter(vdb_payload.values()))
    assert persisted["weight"] == 2.0


@pytest.mark.asyncio
async def test_merge_entities_lifts_first_redirected_relation_to_evidence_floor():
    graph = _SingleSourcedLowWeightGraphStorage()
    entities_vdb = _DummyVectorStorage()
    relationships_vdb = _DummyVectorStorage()

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["A"],
        target_entity="AB",
    )

    redirected_edge = graph.edges[("AB", "C")]
    assert redirected_edge["weight"] == 2.0

    vdb_payload = relationships_vdb.upserts[-1]
    persisted = next(iter(vdb_payload.values()))
    assert persisted["weight"] == 2.0
