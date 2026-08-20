"""Regression tests (#3609 follow-up): entity merge must keep chunk-tracking
row PRESENCE intact, decide it PER INPUT ENTITY, and never open a window
where merged attribution is lost.

Pinned defects, all in ``_merge_entities_impl`` (``lightrag/utils_graph.py``):

1. **Row presence is per input entity.** One entity's curated-empty row
   asserts only that THAT entity tracks no chunks — never that another
   input's unknown (absent/malformed) row is empty too. An input without a
   usable row falls back to its own pre-merge graph ``source_id``, exactly as
   a single-entity edit would. Folding it into an authoritative empty target
   row would silently discard its attribution with no recovery path: the
   empty row is authoritative from then on, so later edits never reseed it.

2. **Curated-empty rows survive the merge.** Merging entities whose rows are
   all present-but-empty must leave the target with a present empty row —
   an absent target row would let the next edit reseed tracking from the
   merged node's possibly-stale graph ``source_id`` (``join_unique`` keeps
   every input's IDs) through the HTTP-reachable merge endpoint.

3. **Unknown stays unknown.** With no input row present at all, the target
   row stays absent; fabricating one would promote UNKNOWN to authoritative.

4. **Upserts commit before deletes**, for the entity row (target before
   source-row deletes) and the relation rows (migrated rows before old-key
   deletes). On RPC-backed KV storages each call commits independently, so
   the reverse order opened crash windows in which attribution existed under
   no key. A relation key can be both old and new (relations already attached
   to an existing target keep their key), so only keys outside the new key
   set may be deleted.
"""

import pytest

from lightrag import utils_graph
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import make_relation_chunk_key

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _MergeGraphStorage:
    """Entities A and B, each with an edge to a shared node C."""

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


async def _run_merge(
    ecs: _RecordingKV,
    *,
    source_entities: list[str] | None = None,
    target_entity: str = "AB",
    rcs: _RecordingKV | None = None,
) -> None:
    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=_MergeGraphStorage(),
        entities_vdb=_DummyVectorStorage(),
        relationships_vdb=_DummyVectorStorage(),
        source_entities=source_entities if source_entities is not None else ["A", "B"],
        target_entity=target_entity,
        entity_chunks_storage=ecs,
        relation_chunks_storage=rcs,
    )


async def test_merge_preserves_curated_empty_entity_rows():
    """All inputs curated-empty: the target must end with a PRESENT empty row
    — presence is what blocks the next edit from reseeding tracking from the
    merged node's stale graph source_id. The graph nodes DO carry source_ids
    (chunk-a-node / chunk-b-node), so this also pins that present rows are
    never overridden by the per-entity graph fallback."""
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
    """Unknown stays unknown: with no input row present there is nothing to
    assert authority over, so the target row is not fabricated — even though
    the graph fallback could name chunks."""
    ecs = _RecordingKV()

    await _run_merge(ecs)

    assert "AB" not in ecs.data


async def test_merge_mixed_empty_row_and_absent_row_falls_back_per_entity():
    """A's authoritative empty row proves only that A owns nothing; B's row is
    absent, so B's attribution must be recovered from B's own pre-merge graph
    source_id. Promoting 'partially known' to 'all empty' would discard
    chunk-b-node forever (the written empty row is authoritative)."""
    ecs = _RecordingKV({"A": {"chunk_ids": [], "count": 0}})

    await _run_merge(ecs)

    assert ecs.data["AB"] == {"chunk_ids": ["chunk-b-node"], "count": 1}


async def test_merge_mixed_valid_row_and_malformed_row_falls_back_per_entity():
    """A has a valid tracking row (authoritative); B's row is malformed (no
    usable chunk_ids list), so only B falls back to its graph source_id. The
    target gets the ordered union of both contributions."""
    ecs = _RecordingKV(
        {
            "A": {"chunk_ids": ["c-a"], "count": 1},
            "B": {"chunk_ids": None},
        }
    )

    await _run_merge(ecs)

    assert ecs.data["AB"] == {"chunk_ids": ["c-a", "chunk-b-node"], "count": 2}


async def test_merge_into_existing_target_with_absent_row_uses_its_pre_merge_source_id():
    """Existing-target mixed merge: the target node exists in the graph but
    has no tracking row. Its contribution comes from its own PRE-MERGE graph
    source_id — not from the merged node's source_id, which by step 5 already
    unions every input's chunks and would double-attribute them."""
    ecs = _RecordingKV({"A": {"chunk_ids": ["c-a"], "count": 1}})

    await _run_merge(ecs, source_entities=["A"], target_entity="B")

    assert ecs.data["B"] == {"chunk_ids": ["c-a", "chunk-b-node"], "count": 2}
    assert "A" not in ecs.data


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


async def test_merge_upserts_relation_rows_before_deleting_only_stale_keys():
    """Relation-row migration ordering: merging A into existing B maps A's
    relation (A,C) onto B's existing key (B,C). The migrated union must be
    upserted BEFORE any delete, and (B,C) — old key and new key at once —
    must never be deleted, or the just-written union would be dropped. Only
    the genuinely stale key (A,C) dies."""
    old_key = make_relation_chunk_key("A", "C")
    shared_key = make_relation_chunk_key("B", "C")
    ecs = _RecordingKV({"A": {"chunk_ids": ["c-a"], "count": 1}})
    rcs = _RecordingKV(
        {
            old_key: {"chunk_ids": ["c-ac"], "count": 1},
            shared_key: {"chunk_ids": ["c-bc"], "count": 1},
        }
    )

    await _run_merge(ecs, source_entities=["A"], target_entity="B", rcs=rcs)

    assert rcs.data[shared_key]["chunk_ids"] == ["c-ac", "c-bc"]
    assert old_key not in rcs.data
    upsert_idx = next(
        i
        for i, (op, keys) in enumerate(rcs.ops)
        if op == "upsert" and shared_key in keys
    )
    delete_idx = next(
        i for i, (op, keys) in enumerate(rcs.ops) if op == "delete" and old_key in keys
    )
    assert upsert_idx < delete_idx
    assert all(shared_key not in keys for op, keys in rcs.ops if op == "delete")
