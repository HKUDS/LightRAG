"""Tests for the offline VDB rebuild tool and the fail-loud merge semantics.

Covers:
- rebuild: payload fidelity against the authoritative write points,
  bidirectional edge dedup, AGE quote stripping, dirty-data skipping,
  drop-before-upsert ordering, batching with periodic flushes, and
  error collection on persistently failing batches.
- check: consistent stores, missing-record detection, legacy reverse
  relation ids not being misreported, and batching.
- merge: _merge_entities_impl raising VectorStorageConsistencyError on
  persistent VDB failure without deleting source entities.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

import lightrag.tools.rebuild_vdb as rebuild_vdb
from lightrag.tools.rebuild_vdb import (
    _strip_agtype_quotes,
    check_vdb_consistency,
    rebuild_chunks_vdb,
    rebuild_entities_vdb,
    rebuild_relationships_vdb,
)
from lightrag.utils import (
    VectorStorageConsistencyError,
    compute_mdhash_id,
    make_relation_vdb_ids,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockVDB:
    """Vector storage mock: records upserts, aligned-None get_by_ids."""

    def __init__(self):
        self.global_config = {}  # no tokenizer -> _truncate_vdb_content is a no-op
        self.records = {}
        self.call_order = []

        async def _drop():
            self.call_order.append("drop")
            self.records.clear()
            return {"status": "success", "message": "data dropped"}

        async def _upsert(payload):
            self.call_order.append("upsert")
            self.records.update(payload)

        async def _get_by_ids(ids):
            return [self.records.get(i) for i in ids]

        self.drop = AsyncMock(side_effect=_drop)
        self.upsert = AsyncMock(side_effect=_upsert)
        self.get_by_ids = AsyncMock(side_effect=_get_by_ids)
        self.delete = AsyncMock()
        self.index_done_callback = AsyncMock()


def make_graph(nodes=None, edges=None):
    graph = MagicMock()
    graph.get_all_nodes = AsyncMock(return_value=nodes or [])
    graph.get_all_edges = AsyncMock(return_value=edges or [])
    return graph


class _FakeLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class JsonKVStorage:
    """Minimal stand-in; the class NAME drives enumerate_kv_keys dispatch."""

    def __init__(self, data):
        self._data = data
        self._storage_lock = _FakeLock()

    async def get_by_ids(self, ids):
        return [self._data.get(i) for i in ids]


def node(name, **overrides):
    data = {
        "entity_id": name,
        "description": f"description of {name}",
        "entity_type": "person",
        "source_id": "chunk-abc",
        "file_path": "doc.txt",
    }
    data.update(overrides)
    return data


def edge(src, tgt, **overrides):
    data = {
        "source": src,
        "target": tgt,
        "description": f"{src} knows {tgt}",
        "keywords": "knows",
        "source_id": "chunk-abc",
        "weight": 2.0,
        "file_path": "doc.txt",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Rebuild: entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebuild_entities_payload_matches_authoritative_format():
    graph = make_graph(nodes=[node("Alice")])
    vdb = MockVDB()

    stats = await rebuild_entities_vdb(graph, vdb, {})

    entity_id = compute_mdhash_id("Alice", prefix="ent-")
    assert stats["rebuilt"] == 1
    assert stats["skipped"] == 0
    assert vdb.records == {
        entity_id: {
            "entity_name": "Alice",
            "entity_type": "person",
            "content": "Alice\ndescription of Alice",
            "source_id": "chunk-abc",
            "description": "description of Alice",
            "file_path": "doc.txt",
        }
    }


@pytest.mark.asyncio
async def test_rebuild_entities_handles_missing_optional_fields():
    graph = make_graph(
        nodes=[{"id": "Bob"}]  # only "id", no entity_id, no optional fields
    )
    vdb = MockVDB()

    await rebuild_entities_vdb(graph, vdb, {})

    record = vdb.records[compute_mdhash_id("Bob", prefix="ent-")]
    assert record["entity_name"] == "Bob"
    assert record["content"] == "Bob\n"
    assert record["entity_type"] == ""
    assert record["source_id"] == ""
    assert record["file_path"] == ""


@pytest.mark.asyncio
async def test_rebuild_entities_skips_dirty_nodes():
    graph = make_graph(nodes=[node("Alice"), {"description": "no id"}, node("  ")])
    vdb = MockVDB()

    stats = await rebuild_entities_vdb(graph, vdb, {})

    assert stats["source_total"] == 3
    assert stats["rebuilt"] == 1
    assert stats["skipped"] == 2
    assert list(vdb.records) == [compute_mdhash_id("Alice", prefix="ent-")]


@pytest.mark.asyncio
async def test_rebuild_drops_once_before_any_upsert():
    graph = make_graph(nodes=[node("Alice"), node("Bob")])
    vdb = MockVDB()

    await rebuild_entities_vdb(graph, vdb, {}, batch_size=1)

    assert vdb.drop.await_count == 1
    assert vdb.call_order[0] == "drop"
    assert vdb.call_order.count("upsert") == 2


@pytest.mark.asyncio
async def test_rebuild_batches_and_periodic_flush():
    graph = make_graph(nodes=[node(f"E{i:03d}") for i in range(25)])
    vdb = MockVDB()

    stats = await rebuild_entities_vdb(graph, vdb, {}, batch_size=1)

    assert stats["batches"] == 25
    assert vdb.upsert.await_count == 25
    # periodic flush at batches 10 and 20, plus the final flush
    assert vdb.index_done_callback.await_count == 3


@pytest.mark.asyncio
async def test_rebuild_collects_batch_errors_and_continues(monkeypatch):
    # Single-attempt wrapper: no retry delays in tests
    async def _single_attempt(operation, **kwargs):
        await operation()

    monkeypatch.setattr(
        rebuild_vdb, "safe_vdb_operation_with_exception", _single_attempt
    )

    graph = make_graph(nodes=[node("Alice"), node("Bob"), node("Carol")])
    vdb = MockVDB()
    poisoned_id = compute_mdhash_id("Bob", prefix="ent-")

    async def _failing_upsert(payload):
        if poisoned_id in payload:
            raise RuntimeError("embedder down")
        vdb.records.update(payload)

    vdb.upsert = AsyncMock(side_effect=_failing_upsert)

    stats = await rebuild_entities_vdb(graph, vdb, {}, batch_size=1)

    assert stats["rebuilt"] == 2
    assert stats["failed_batches"] == 1
    assert len(stats["errors"]) == 1
    assert stats["errors"][0]["error_msg"] == "embedder down"
    assert poisoned_id not in vdb.records
    assert len(vdb.records) == 2


# ---------------------------------------------------------------------------
# Rebuild: relationships
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebuild_relationships_payload_normalized():
    # Endpoints arrive in reverse lexicographic order; payload must be sorted
    graph = make_graph(edges=[edge("Bob", "Alice")])
    vdb = MockVDB()

    stats = await rebuild_relationships_vdb(graph, vdb, {})

    rel_id = compute_mdhash_id("Alice" + "Bob", prefix="rel-")
    assert stats["rebuilt"] == 1
    assert vdb.records == {
        rel_id: {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "source_id": "chunk-abc",
            "content": "knows\tAlice\nBob\nBob knows Alice",
            "keywords": "knows",
            "description": "Bob knows Alice",
            "weight": 2.0,
            "file_path": "doc.txt",
        }
    }


@pytest.mark.asyncio
async def test_rebuild_relationships_weight_fallback():
    graph = make_graph(edges=[edge("A", "B", weight="3"), edge("C", "D", weight=None)])
    vdb = MockVDB()

    await rebuild_relationships_vdb(graph, vdb, {})

    assert vdb.records[compute_mdhash_id("AB", prefix="rel-")]["weight"] == 3.0
    assert vdb.records[compute_mdhash_id("CD", prefix="rel-")]["weight"] == 1.0


@pytest.mark.asyncio
async def test_rebuild_relationships_dedupes_bidirectional_edges():
    # Neo4j/Memgraph return each undirected edge once per direction
    graph = make_graph(edges=[edge("Alice", "Bob"), edge("Bob", "Alice")])
    vdb = MockVDB()

    stats = await rebuild_relationships_vdb(graph, vdb, {})

    assert stats["source_total"] == 2
    assert stats["rebuilt"] == 1
    assert stats["duplicates"] == 1


@pytest.mark.asyncio
async def test_rebuild_relationships_strips_age_quotes():
    # PG/AGE agtype::text casts wrap endpoint strings in double quotes
    graph = make_graph(edges=[edge('"Alice"', '"Bob"')])
    vdb = MockVDB()

    await rebuild_relationships_vdb(graph, vdb, {})

    rel_id = compute_mdhash_id("AliceBob", prefix="rel-")
    assert vdb.records[rel_id]["src_id"] == "Alice"
    assert vdb.records[rel_id]["tgt_id"] == "Bob"


@pytest.mark.asyncio
async def test_rebuild_relationships_skips_edges_without_endpoints():
    graph = make_graph(edges=[edge("Alice", "Bob"), {"description": "no endpoints"}])
    vdb = MockVDB()

    stats = await rebuild_relationships_vdb(graph, vdb, {})

    assert stats["rebuilt"] == 1
    assert stats["skipped"] == 1


def test_strip_agtype_quotes():
    assert _strip_agtype_quotes('"Alice"') == "Alice"
    assert _strip_agtype_quotes("Alice") == "Alice"
    assert _strip_agtype_quotes('"') == '"'
    assert _strip_agtype_quotes(None) is None
    assert _strip_agtype_quotes(3) == 3


# ---------------------------------------------------------------------------
# Rebuild: chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebuild_chunks_payload_from_kv_records():
    kv = JsonKVStorage(
        {
            "chunk-1": {
                "content": "chunk text",
                "full_doc_id": "doc-1",
                "file_path": "doc.txt",
                "tokens": 3,
                "chunk_order_index": 0,
            },
        }
    )
    vdb = MockVDB()

    stats = await rebuild_chunks_vdb(kv, vdb)

    assert stats["source_total"] == 1
    assert stats["rebuilt"] == 1
    record = vdb.records["chunk-1"]
    assert record["content"] == "chunk text"
    assert record["full_doc_id"] == "doc-1"
    assert record["file_path"] == "doc.txt"
    # extra pipeline fields pass through, as in the authoritative upsert
    assert record["tokens"] == 3
    assert record["chunk_order_index"] == 0
    assert vdb.drop.await_count == 1


@pytest.mark.asyncio
async def test_rebuild_chunks_skips_records_without_content():
    kv = JsonKVStorage(
        {
            "chunk-1": {"content": "ok", "full_doc_id": "doc-1"},
            "chunk-2": {"full_doc_id": "doc-1"},  # no content
        }
    )
    vdb = MockVDB()

    stats = await rebuild_chunks_vdb(kv, vdb)

    assert stats["rebuilt"] == 1
    assert stats["skipped"] == 1
    assert "chunk-2" not in vdb.records


@pytest.mark.asyncio
async def test_rebuild_chunks_covers_all_id_schemes():
    # Chunks live under several id schemes that no single prefix matches:
    # custom KG ("chunk-<hash>"), the text pipeline ("{doc_id}-chunk-{order}",
    # which does NOT start with "chunk-"), and multimodal
    # ("{doc_id}-mm-<modality>-{order}"). A prefix-only filter dropped the
    # pipeline/multimodal chunks and rebuilt to zero. All schemes must rebuild.
    kv = JsonKVStorage(
        {
            "chunk-deadbeef": {"content": "custom kg", "full_doc_id": "doc-xyz"},
            "doc-abc123-chunk-000": {"content": "first", "full_doc_id": "doc-abc123"},
            "doc-abc123-chunk-001": {"content": "second", "full_doc_id": "doc-abc123"},
            "doc-abc123-mm-drawing-000": {
                "content": "image caption",
                "full_doc_id": "doc-abc123",
            },
        }
    )
    vdb = MockVDB()

    stats = await rebuild_chunks_vdb(kv, vdb)

    assert stats["source_total"] == 4
    assert stats["rebuilt"] == 4
    assert set(vdb.records) == {
        "chunk-deadbeef",
        "doc-abc123-chunk-000",
        "doc-abc123-chunk-001",
        "doc-abc123-mm-drawing-000",
    }


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------


def seeded_vdbs(nodes, edges):
    """Build entity/relation VDB mocks already containing all graph records."""
    entities_vdb = MockVDB()
    for n in nodes:
        entities_vdb.records[compute_mdhash_id(n["entity_id"], prefix="ent-")] = {
            "entity_name": n["entity_id"]
        }
    relationships_vdb = MockVDB()
    for e in edges:
        src, tgt = sorted([e["source"], e["target"]])
        relationships_vdb.records[compute_mdhash_id(src + tgt, prefix="rel-")] = {
            "src_id": src,
            "tgt_id": tgt,
        }
    return entities_vdb, relationships_vdb


@pytest.mark.asyncio
async def test_check_all_consistent():
    nodes = [node("Alice"), node("Bob")]
    edges = [edge("Alice", "Bob")]
    graph = make_graph(nodes=nodes, edges=edges)
    entities_vdb, relationships_vdb = seeded_vdbs(nodes, edges)

    report = await check_vdb_consistency(graph, entities_vdb, relationships_vdb)

    assert report["consistent"] is True
    assert report["graph_entities"] == 2
    assert report["graph_relations"] == 1
    assert report["missing_entities"] == 0
    assert report["missing_relations"] == 0


@pytest.mark.asyncio
async def test_check_detects_missing_records():
    nodes = [node("Alice"), node("Bob")]
    edges = [edge("Alice", "Bob"), edge("Bob", "Carol")]
    graph = make_graph(nodes=nodes, edges=edges)
    entities_vdb, relationships_vdb = seeded_vdbs(nodes[:1], edges[:1])

    report = await check_vdb_consistency(graph, entities_vdb, relationships_vdb)

    assert report["consistent"] is False
    assert report["missing_entities"] == 1
    assert report["missing_entity_names"] == ["Bob"]
    assert report["missing_relations"] == 1
    assert report["missing_relation_pairs"] == ["Bob ~ Carol"]


@pytest.mark.asyncio
async def test_check_accepts_legacy_reverse_relation_id():
    # Legacy custom-KG imports hashed the relation in original endpoint
    # order; the VDB holds only the reverse-order id. Not an inconsistency.
    graph = make_graph(edges=[edge("Bob", "Alice")])
    entities_vdb = MockVDB()
    relationships_vdb = MockVDB()
    reverse_id = make_relation_vdb_ids("Bob", "Alice")[1]
    relationships_vdb.records[reverse_id] = {"src_id": "Bob", "tgt_id": "Alice"}

    report = await check_vdb_consistency(graph, entities_vdb, relationships_vdb)

    assert report["missing_relations"] == 0
    assert report["consistent"] is True


@pytest.mark.asyncio
async def test_check_batches_requests():
    nodes = [node(f"E{i:03d}") for i in range(5)]
    graph = make_graph(nodes=nodes)
    entities_vdb, relationships_vdb = seeded_vdbs(nodes, [])

    report = await check_vdb_consistency(
        graph, entities_vdb, relationships_vdb, batch_size=2
    )

    assert report["consistent"] is True
    assert entities_vdb.get_by_ids.await_count == 3  # ceil(5 / 2)
    assert all(
        len(call.args[0]) <= 2 for call in entities_vdb.get_by_ids.await_args_list
    )


# ---------------------------------------------------------------------------
# Merge: fail-loud semantics in _merge_entities_impl
# ---------------------------------------------------------------------------


def make_merge_graph():
    """Graph mock for merging source 'Bob' into new target 'Alice'.

    Bob has one edge to Carol; Alice does not exist yet.
    """
    graph = MagicMock()
    graph.has_node = AsyncMock(side_effect=lambda name: name == "Bob")
    graph.get_node = AsyncMock(
        return_value={
            "entity_id": "Bob",
            "description": "desc",
            "entity_type": "person",
            "source_id": "chunk-abc",
            "file_path": "doc.txt",
        }
    )
    graph.get_node_edges = AsyncMock(return_value=[("Bob", "Carol")])
    graph.get_edge = AsyncMock(
        return_value={
            "description": "knows",
            "keywords": "kw",
            "source_id": "chunk-abc",
            "weight": 1.0,
            "file_path": "doc.txt",
        }
    )
    graph.upsert_node = AsyncMock()
    graph.upsert_edge = AsyncMock()
    graph.delete_node = AsyncMock()
    graph.index_done_callback = AsyncMock()
    return graph


@pytest.fixture
def single_attempt_vdb_ops(monkeypatch):
    """Replace the retry wrapper with a single attempt to keep tests fast."""
    import lightrag.utils_graph as utils_graph

    async def _single_attempt(operation, **kwargs):
        await operation()

    monkeypatch.setattr(
        utils_graph, "safe_vdb_operation_with_exception", _single_attempt
    )


@pytest.mark.asyncio
async def test_merge_relation_vdb_failure_raises_consistency_error(
    single_attempt_vdb_ops,
):
    from lightrag.utils_graph import _merge_entities_impl

    graph = make_merge_graph()
    entities_vdb = MockVDB()
    relationships_vdb = MockVDB()
    relationships_vdb.upsert = AsyncMock(side_effect=RuntimeError("embedder down"))

    with pytest.raises(VectorStorageConsistencyError) as excinfo:
        await _merge_entities_impl(
            graph, entities_vdb, relationships_vdb, ["Bob"], "Alice"
        )

    # Fail-loud guidance in the message
    assert "lightrag-rebuild-vdb" in str(excinfo.value)
    # Graph state was written and is kept (no rollback)
    graph.upsert_edge.assert_awaited()
    # Source entities were NOT deleted (step 10 never reached)
    graph.delete_node.assert_not_awaited()
    entities_vdb.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_entity_vdb_failure_raises_consistency_error(
    single_attempt_vdb_ops,
):
    from lightrag.utils_graph import _merge_entities_impl

    graph = make_merge_graph()
    entities_vdb = MockVDB()
    relationships_vdb = MockVDB()
    entities_vdb.upsert = AsyncMock(side_effect=RuntimeError("embedder down"))

    with pytest.raises(VectorStorageConsistencyError) as excinfo:
        await _merge_entities_impl(
            graph, entities_vdb, relationships_vdb, ["Bob"], "Alice"
        )

    assert "lightrag-rebuild-vdb" in str(excinfo.value)
    # Relation VDB write succeeded before the entity failure
    relationships_vdb.upsert.assert_awaited()
    graph.delete_node.assert_not_awaited()
    entities_vdb.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_success_path_unaffected(single_attempt_vdb_ops):
    import lightrag.utils_graph as utils_graph
    from lightrag.utils_graph import _merge_entities_impl

    graph = make_merge_graph()
    entities_vdb = MockVDB()
    relationships_vdb = MockVDB()

    async def _noop_persist(**kwargs):
        return None

    async def _entity_info(*args, **kwargs):
        return {"entity_name": "Alice"}

    # Avoid index_done_callback/inspection helpers needing full storages
    orig_persist = utils_graph._persist_graph_updates
    orig_info = utils_graph.get_entity_info
    utils_graph._persist_graph_updates = _noop_persist
    utils_graph.get_entity_info = _entity_info
    try:
        result = await _merge_entities_impl(
            graph, entities_vdb, relationships_vdb, ["Bob"], "Alice"
        )
    finally:
        utils_graph._persist_graph_updates = orig_persist
        utils_graph.get_entity_info = orig_info

    assert result == {"entity_name": "Alice"}
    # Both VDB writes happened and the source entity was deleted
    relationships_vdb.upsert.assert_awaited()
    entities_vdb.upsert.assert_awaited()
    graph.delete_node.assert_awaited_with("Bob")


@pytest.mark.asyncio
async def test_merge_deferred_flush_failure_raises_before_delete(
    single_attempt_vdb_ops,
):
    # Deferred-embedding backends (nano/faiss) buffer in upsert() and only embed
    # in index_done_callback, so an embedder outage surfaces at flush time. The
    # fail-loud guarantee must still hold: raise before deleting source entities.
    from lightrag.utils_graph import _merge_entities_impl

    graph = make_merge_graph()
    entities_vdb = MockVDB()
    relationships_vdb = MockVDB()
    # upsert succeeds (buffers); the embedding failure happens at flush.
    relationships_vdb.index_done_callback = AsyncMock(
        side_effect=RuntimeError("embedder down")
    )

    with pytest.raises(VectorStorageConsistencyError) as excinfo:
        await _merge_entities_impl(
            graph, entities_vdb, relationships_vdb, ["Bob"], "Alice"
        )

    assert "lightrag-rebuild-vdb" in str(excinfo.value)
    # upsert succeeded; the failure is the deferred flush
    relationships_vdb.upsert.assert_awaited()
    relationships_vdb.index_done_callback.assert_awaited()
    # Source entities NOT deleted (step 10 never reached), so the message holds
    graph.delete_node.assert_not_awaited()
    entities_vdb.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_entity_deferred_flush_failure_raises_before_delete(
    single_attempt_vdb_ops,
):
    from lightrag.utils_graph import _merge_entities_impl

    graph = make_merge_graph()
    entities_vdb = MockVDB()
    relationships_vdb = MockVDB()
    entities_vdb.index_done_callback = AsyncMock(
        side_effect=RuntimeError("embedder down")
    )

    with pytest.raises(VectorStorageConsistencyError) as excinfo:
        await _merge_entities_impl(
            graph, entities_vdb, relationships_vdb, ["Bob"], "Alice"
        )

    assert "lightrag-rebuild-vdb" in str(excinfo.value)
    # Relation flush succeeded before the entity flush failed
    relationships_vdb.index_done_callback.assert_awaited()
    entities_vdb.index_done_callback.assert_awaited()
    graph.delete_node.assert_not_awaited()
    entities_vdb.delete.assert_not_awaited()
