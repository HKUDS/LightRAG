"""Regression coverage for self-loop relations during manual entity rename."""

from copy import deepcopy

import networkx as nx
import pytest

from lightrag import utils_graph
from lightrag.utils import compute_mdhash_id, make_relation_chunk_key


pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Graph:
    """NetworkX-backed double with real incident-edge deletion semantics."""

    def __init__(self):
        self.graph = nx.Graph()

    async def has_node(self, entity_name):
        return self.graph.has_node(entity_name)

    async def get_node(self, entity_name):
        data = self.graph.nodes.get(entity_name)
        return deepcopy(data) if data is not None else None

    async def upsert_node(self, entity_name, node_data):
        self.graph.add_node(entity_name, **deepcopy(node_data))

    async def get_node_edges(self, entity_name):
        return list(self.graph.edges(entity_name))

    async def get_edge(self, source_entity, target_entity):
        data = self.graph.get_edge_data(source_entity, target_entity)
        return deepcopy(data) if data is not None else None

    async def upsert_edge(self, source_entity, target_entity, edge_data):
        self.graph.add_edge(source_entity, target_entity, **deepcopy(edge_data))

    async def delete_node(self, entity_name):
        self.graph.remove_node(entity_name)

    async def index_done_callback(self):
        return None


class _VectorStorage:
    def __init__(self):
        self.global_config = {"workspace": ""}
        self.records = {}

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, record_ids):
        for record_id in record_ids:
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


class _ChunkTrackingStorage:
    def __init__(self, records):
        self.records = deepcopy(records)

    async def get_by_id(self, record_id):
        value = self.records.get(record_id)
        return deepcopy(value) if value is not None else None

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, record_ids):
        for record_id in record_ids:
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


async def test_entity_rename_preserves_self_loop_across_all_storages(monkeypatch):
    """Renaming A to B must map both endpoints of (A, A) to (B, B).

    Keeping an ordinary neighbour edge in the same rename pins the existing
    one-endpoint cascade while exercising the self-loop-specific boundary.
    """
    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: _NoopLock(),
    )

    graph = _Graph()
    await graph.upsert_node(
        "A", {"entity_id": "A", "description": "source", "source_id": "chunk-a"}
    )
    await graph.upsert_node(
        "C", {"entity_id": "C", "description": "neighbour", "source_id": "chunk-c"}
    )
    self_loop = {
        "description": "A refers to itself",
        "keywords": "self",
        "source_id": "chunk-self",
        "weight": 2.0,
    }
    neighbour = {
        "description": "A relates to C",
        "keywords": "neighbour",
        "source_id": "chunk-neighbour",
        "weight": 1.0,
    }
    await graph.upsert_edge("A", "A", self_loop)
    await graph.upsert_edge("A", "C", neighbour)

    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()
    old_self_id = compute_mdhash_id("AA", prefix="rel-")
    old_neighbour_id = compute_mdhash_id("AC", prefix="rel-")
    relationships_vdb.records = {
        old_self_id: {"src_id": "A", "tgt_id": "A"},
        old_neighbour_id: {"src_id": "A", "tgt_id": "C"},
    }

    old_self_key = make_relation_chunk_key("A", "A")
    old_neighbour_key = make_relation_chunk_key("A", "C")
    relation_chunks = _ChunkTrackingStorage(
        {
            old_self_key: {"chunk_ids": ["chunk-self"], "count": 1},
            old_neighbour_key: {"chunk_ids": ["chunk-neighbour"], "count": 1},
        }
    )

    result = await utils_graph.aedit_entity(
        graph,
        entities_vdb,
        relationships_vdb,
        "A",
        {"entity_name": "B"},
        relation_chunks_storage=relation_chunks,
    )

    assert result["operation_summary"]["operation_status"] == "success"
    assert set(graph.graph.nodes) == {"B", "C"}
    assert {frozenset(edge) for edge in graph.graph.edges} == {
        frozenset(("B", "B")),
        frozenset(("B", "C")),
    }
    assert graph.graph.get_edge_data("B", "B") == self_loop
    assert graph.graph.get_edge_data("B", "C") == neighbour

    new_self_id = compute_mdhash_id("BB", prefix="rel-")
    new_neighbour_id = compute_mdhash_id("BC", prefix="rel-")
    assert set(relationships_vdb.records) == {new_self_id, new_neighbour_id}
    assert relationships_vdb.records[new_self_id]["src_id"] == "B"
    assert relationships_vdb.records[new_self_id]["tgt_id"] == "B"

    assert set(relation_chunks.records) == {
        make_relation_chunk_key("B", "B"),
        make_relation_chunk_key("B", "C"),
    }
    assert relation_chunks.records[make_relation_chunk_key("B", "B")]["chunk_ids"] == [
        "chunk-self"
    ]
