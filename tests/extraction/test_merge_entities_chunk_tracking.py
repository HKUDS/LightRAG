"""Regression tests (#3609 follow-up): entity merge must keep chunk-tracking
row PRESENCE intact and never open a window where merged attribution is lost.

Two defects are pinned here, both in ``_merge_entities_impl`` step 9
(``lightrag/utils_graph.py``):

1. **Curated-empty rows survive the merge.** Row presence is decided by
   schema (``has_chunk_tracking_row``), never list truthiness. Merging
   entities whose rows are all present-but-empty must leave the target with a
   present empty row — deleting the sources while skipping the target upsert
   made the target's row ABSENT, so the next edit reseeded it from the merged
   node's possibly-stale graph ``source_id`` (``join_unique`` keeps every
   source's IDs), resurrecting purged attribution through the HTTP-reachable
   merge endpoint.

2. **Target upsert commits before source deletes.** On RPC-backed KV
   storages each call commits independently; deleting the source rows first
   opened a crash window in which the merged attribution existed under no key
   at all.
"""

import pytest

from lightrag import utils_graph
from lightrag.constants import GRAPH_FIELD_SEP

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _MergeGraphStorage:
    """Two source entities (A, B), each with an edge to a shared node C."""

    def __init__(self):
        self.nodes = {
            name: {
                "entity_id": name,
                "description": f"Entity {name}",
                "entity_type": "ORG",
                "source_id": f"chunk-{name.lower()}-node",
                "file_path": f"doc{name}.txt",
            }
            for name in ("A", "B", "C")
        }
        self.edges = {
            ("A", "C"): {
                "description": "A relates to C",
                "keywords": "k1",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-a1"]),
                "weight": 1.0,
                "file_path": "docA.txt",
            },
            ("B", "C"): {
                "description": "B relates to C",
                "keywords": "k2",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-b1"]),
                "weight": 1.0,
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

    async def upsert(self, data):
        return None

    async def delete(self, ids):
        return None

    async def get_by_id(self, id_):
        return None

    async def index_done_callback(self):
        return True


class _RecordingKV:
    """KV double that records the order of upsert/delete calls."""

    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})
        self.ops: list[tuple[str, tuple[str, ...]]] = []

    async def get_by_id(self, key):
        return self.data.get(key)

    async def upsert(self, data):
        self.ops.append(("upsert", tuple(data.keys())))
        self.data.update(data)

    async def delete(self, keys):
        self.ops.append(("delete", tuple(keys)))
        for key in keys:
            self.data.pop(key, None)

    async def index_done_callback(self):
        return True


async def _run_merge(ecs: _RecordingKV) -> None:
    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=_MergeGraphStorage(),
        entities_vdb=_DummyVectorStorage(),
        relationships_vdb=_DummyVectorStorage(),
        source_entities=["A", "B"],
        target_entity="AB",
        entity_chunks_storage=ecs,
    )


async def test_merge_preserves_curated_empty_entity_rows():
    """All-sources-empty: the target must end with a PRESENT empty row, not an
    absent one — presence is what blocks the next edit from reseeding tracking
    from the merged node's stale graph source_id."""
    ecs = _RecordingKV(
        {
            "A": {"chunk_ids": [], "count": 0},
            "B": {"chunk_ids": [], "count": 0},
        }
    )

    await _run_merge(ecs)

    assert ecs.data["AB"] == {"chunk_ids": [], "count": 0}
    assert "A" not in ecs.data
    assert "B" not in ecs.data


async def test_merge_leaves_target_row_absent_when_no_source_has_one():
    """Unknown stays unknown: with no source row present there is nothing to
    assert authority over, so the target row is not fabricated."""
    ecs = _RecordingKV()

    await _run_merge(ecs)

    assert "AB" not in ecs.data


async def test_merge_upserts_target_row_before_deleting_source_rows():
    """Crash-ordering pin: the merged row must be durable under the target key
    before the source keys are deleted."""
    ecs = _RecordingKV(
        {
            "A": {"chunk_ids": ["c-a"], "count": 1},
            "B": {"chunk_ids": ["c-b"], "count": 1},
        }
    )

    await _run_merge(ecs)

    assert ecs.data["AB"]["chunk_ids"] == ["c-a", "c-b"]
    upsert_idx = ecs.ops.index(("upsert", ("AB",)))
    delete_idx = next(
        i for i, (op, keys) in enumerate(ecs.ops) if op == "delete" and "A" in keys
    )
    assert upsert_idx < delete_idx
