"""Regression tests (#3609 follow-up): the rename key migrations must upsert
the new key BEFORE deleting the old one.

On RPC-backed KV storages (Redis/PG/Mongo) each call commits independently, so
delete-then-upsert opened a crash window in which the tracking row existed
under neither key. A row lost that way is ABSENT, and an absent row is exactly
what re-arms the stale ``source_id`` reseed PR #3660 closed — while an
orphaned old-key row left behind by the reverse failure is only dead
bookkeeping. Pinned for both migrations ``_edit_entity_impl`` performs on
rename: the entity's own row and every migrated relation row.
"""

from copy import deepcopy

import pytest

from lightrag import utils_graph
from lightrag.utils import make_relation_chunk_key

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    async def has_node(self, entity_name):
        return entity_name in self.nodes

    async def get_node(self, entity_name):
        data = self.nodes.get(entity_name)
        return deepcopy(data) if data is not None else None

    async def upsert_node(self, entity_name, node_data):
        self.nodes[entity_name] = deepcopy(node_data)

    async def get_node_edges(self, entity_name):
        return [
            (src, tgt)
            for (src, tgt) in self.edges
            if src == entity_name or tgt == entity_name
        ]

    async def get_edge(self, source_entity, target_entity):
        data = self.edges.get((source_entity, target_entity)) or self.edges.get(
            (target_entity, source_entity)
        )
        return deepcopy(data) if data is not None else None

    async def upsert_edge(self, source_entity, target_entity, edge_data):
        self.edges[(source_entity, target_entity)] = deepcopy(edge_data)

    async def delete_node(self, entity_name):
        self.nodes.pop(entity_name, None)
        self.edges = {
            (src, tgt): data
            for (src, tgt), data in self.edges.items()
            if src != entity_name and tgt != entity_name
        }

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


class _RecordingKV:
    """KV double that records the order of upsert/delete calls."""

    def __init__(self, data: dict | None = None):
        self.data = deepcopy(data or {})
        self.ops: list[tuple[str, tuple[str, ...]]] = []

    async def get_by_id(self, key):
        value = self.data.get(key)
        return deepcopy(value) if value is not None else None

    async def upsert(self, data):
        self.ops.append(("upsert", tuple(data.keys())))
        self.data.update(deepcopy(data))

    async def delete(self, keys):
        self.ops.append(("delete", tuple(keys)))
        for key in keys:
            self.data.pop(key, None)

    async def index_done_callback(self):
        return None


def _op_index(ops, wanted_op, key):
    return next(
        i for i, (op, keys) in enumerate(ops) if op == wanted_op and key in keys
    )


async def _rename_a_to_b(entity_chunks, relation_chunks, monkeypatch):
    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: _NoopLock(),
    )

    graph = _Graph()
    await graph.upsert_node(
        "A",
        {
            "entity_id": "A",
            "description": "source",
            "entity_type": "ORG",
            "source_id": "chunk-a",
            "file_path": "docA.txt",
        },
    )
    await graph.upsert_node(
        "C",
        {
            "entity_id": "C",
            "description": "neighbour",
            "entity_type": "ORG",
            "source_id": "chunk-c",
            "file_path": "docC.txt",
        },
    )
    await graph.upsert_edge(
        "A",
        "C",
        {
            "description": "A relates to C",
            "keywords": "k",
            "source_id": "chunk-rel",
            "weight": 1.0,
            "file_path": "docA.txt",
        },
    )

    result = await utils_graph.aedit_entity(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        "A",
        {"entity_name": "B"},
        entity_chunks_storage=entity_chunks,
        relation_chunks_storage=relation_chunks,
    )
    assert result["operation_summary"]["operation_status"] == "success"


async def test_rename_upserts_new_entity_key_before_deleting_old(monkeypatch):
    entity_chunks = _RecordingKV({"A": {"chunk_ids": ["chunk-a"], "count": 1}})
    relation_chunks = _RecordingKV(
        {make_relation_chunk_key("A", "C"): {"chunk_ids": ["chunk-rel"], "count": 1}}
    )

    await _rename_a_to_b(entity_chunks, relation_chunks, monkeypatch)

    assert entity_chunks.data["B"]["chunk_ids"] == ["chunk-a"]
    assert "A" not in entity_chunks.data
    assert _op_index(entity_chunks.ops, "upsert", "B") < _op_index(
        entity_chunks.ops, "delete", "A"
    )


async def test_rename_upserts_new_relation_key_before_deleting_old(monkeypatch):
    old_key = make_relation_chunk_key("A", "C")
    new_key = make_relation_chunk_key("B", "C")
    entity_chunks = _RecordingKV({"A": {"chunk_ids": ["chunk-a"], "count": 1}})
    relation_chunks = _RecordingKV({old_key: {"chunk_ids": ["chunk-rel"], "count": 1}})

    await _rename_a_to_b(entity_chunks, relation_chunks, monkeypatch)

    assert relation_chunks.data[new_key]["chunk_ids"] == ["chunk-rel"]
    assert old_key not in relation_chunks.data
    assert _op_index(relation_chunks.ops, "upsert", new_key) < _op_index(
        relation_chunks.ops, "delete", old_key
    )


async def test_rename_migrates_curated_empty_relation_row_upsert_first(monkeypatch):
    """The PR's empty-row migration composed with the ordering fix: the
    curated empty row is written under the new key before the old key dies."""
    old_key = make_relation_chunk_key("A", "C")
    new_key = make_relation_chunk_key("B", "C")
    entity_chunks = _RecordingKV({"A": {"chunk_ids": [], "count": 0}})
    relation_chunks = _RecordingKV({old_key: {"chunk_ids": [], "count": 0}})

    await _rename_a_to_b(entity_chunks, relation_chunks, monkeypatch)

    assert relation_chunks.data[new_key] == {"chunk_ids": [], "count": 0}
    assert old_key not in relation_chunks.data
    assert _op_index(relation_chunks.ops, "upsert", new_key) < _op_index(
        relation_chunks.ops, "delete", old_key
    )
