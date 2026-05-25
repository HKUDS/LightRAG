"""
Pin the business-layer keyed-lock contracts on the entity-mutation paths.

`get_storage_keyed_lock(keys, namespace=...)` acquires one mutex per key in
the given namespace, so identical key strings share the same mutex across
callers. Locking `[entity_name]` is therefore already enough to mutually
exclude any concurrent edge write that names the same entity in
`sorted([src, tgt])` — no need to enumerate incident edges here.

These tests pin:
- `aedit_entity` locks {old, new} on rename, {entity_name} otherwise.
- `adelete_by_entity` locks {entity_name}.
- `ainsert_custom_kg` locks every entity name plus every relationship
  endpoint that the batch will write, sharing the doc-ingest namespace.
- An empty `ainsert_custom_kg` batch skips the lock entirely.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keyed_lock_spy():
    """Return (spy_callable, captured_calls_list).

    Spy yields a no-op async context manager and records every invocation's
    `keys` / `namespace` arguments.
    """
    captured: list[dict] = []

    @asynccontextmanager
    async def _noop_lock():
        yield

    def spy(keys, namespace="default", enable_logging=False):
        captured.append({"keys": list(keys), "namespace": namespace})
        return _noop_lock()

    return spy, captured


def _make_graph_mock(
    edges_for_entity: list[tuple[str, str]] | None = None,
    *,
    existing_entity: str = "X",
):
    """Minimal `chunk_entity_relation_graph` mock.

    `has_node` returns True only for `existing_entity` so a rename target
    (e.g. "Y") is treated as not-yet-existing — otherwise aedit_entity would
    short-circuit with "Entity name 'Y' already exists".
    """
    graph = MagicMock()
    graph.get_node_edges = AsyncMock(return_value=edges_for_entity or [])
    graph.has_node = AsyncMock(side_effect=lambda name: name == existing_entity)
    graph.get_node = AsyncMock(
        return_value={
            "entity_id": existing_entity,
            "description": "old description",
            "entity_type": "PERSON",
            "source_id": "chunk-1",
            "file_path": "test.txt",
        }
    )
    graph.upsert_node = AsyncMock(return_value=None)
    graph.upsert_edge = AsyncMock(return_value=None)
    graph.upsert_nodes_batch = AsyncMock(return_value=None)
    graph.upsert_edges_batch = AsyncMock(return_value=None)
    graph.has_nodes_batch = AsyncMock(return_value=set())
    graph.delete_node = AsyncMock(return_value=None)
    graph.get_edge = AsyncMock(
        return_value={
            "weight": 1.0,
            "description": "rel",
            "keywords": "k",
            "source_id": "chunk-1",
            "file_path": "test.txt",
            "created_at": 0,
        }
    )
    graph.index_done_callback = AsyncMock(return_value=None)
    return graph


def _make_vdb_mock(workspace: str = ""):
    vdb = MagicMock()
    vdb.global_config = {"workspace": workspace}
    vdb.upsert = AsyncMock(return_value=None)
    vdb.delete = AsyncMock(return_value=None)
    vdb.delete_entity = AsyncMock(return_value=None)
    vdb.delete_entity_relation = AsyncMock(return_value=None)
    vdb.index_done_callback = AsyncMock(return_value=None)
    vdb.client_storage = MagicMock()
    return vdb


# ---------------------------------------------------------------------------
# aedit_entity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aedit_entity_rename_locks_old_and_new_names():
    """Renaming X -> Y locks only {X, Y}. The doc-ingest pipeline uses the
    same namespace and acquires per-key mutexes, so locking the entity name
    already excludes any sorted([X, *]) or sorted([Y, *]) edge lock — no
    need to enumerate incident edges here."""
    from lightrag import utils_graph

    spy, captured = _make_keyed_lock_spy()
    graph = _make_graph_mock()
    entities_vdb = _make_vdb_mock(workspace="ws1")
    relationships_vdb = _make_vdb_mock(workspace="ws1")

    # Short-circuit before the rename actually runs — we only care about the
    # lock arguments.
    graph.upsert_node.side_effect = RuntimeError("stop after lock acquisition")

    with patch.object(utils_graph, "get_storage_keyed_lock", spy):
        with pytest.raises(RuntimeError, match="stop after lock acquisition"):
            await utils_graph.aedit_entity(
                chunk_entity_relation_graph=graph,
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                entity_name="X",
                updated_data={"entity_name": "Y", "description": "renamed"},
                allow_rename=True,
            )

    assert len(captured) == 1
    assert captured[0]["keys"] == ["X", "Y"]
    assert captured[0]["namespace"] == "ws1:GraphDB"

    # No pre-fetch of incident edges — that would only add I/O.
    graph.get_node_edges.assert_not_called()


@pytest.mark.asyncio
async def test_aedit_entity_non_rename_locks_single_entity_name():
    """Non-rename edits lock just the entity name."""
    from lightrag import utils_graph

    spy, captured = _make_keyed_lock_spy()
    graph = _make_graph_mock()
    entities_vdb = _make_vdb_mock(workspace="")
    relationships_vdb = _make_vdb_mock(workspace="")

    graph.upsert_node.side_effect = RuntimeError("stop after lock acquisition")

    with patch.object(utils_graph, "get_storage_keyed_lock", spy):
        with pytest.raises(RuntimeError, match="stop after lock acquisition"):
            await utils_graph.aedit_entity(
                chunk_entity_relation_graph=graph,
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                entity_name="X",
                updated_data={"description": "updated"},
                allow_rename=False,
            )

    assert len(captured) == 1
    assert captured[0]["keys"] == ["X"]
    # Empty workspace falls back to the bare "GraphDB" namespace.
    assert captured[0]["namespace"] == "GraphDB"
    graph.get_node_edges.assert_not_called()


# ---------------------------------------------------------------------------
# adelete_by_entity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adelete_by_entity_locks_single_entity_name():
    """Entity delete locks just the entity name."""
    from lightrag import utils_graph

    spy, captured = _make_keyed_lock_spy()
    graph = _make_graph_mock(edges_for_entity=[("X", "Y"), ("Z", "X")])
    entities_vdb = _make_vdb_mock(workspace="ws1")
    relationships_vdb = _make_vdb_mock(workspace="ws1")

    with patch.object(utils_graph, "get_storage_keyed_lock", spy):
        result = await utils_graph.adelete_by_entity(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            entity_name="X",
        )

    assert result.status == "success"
    assert len(captured) == 1
    assert captured[0]["keys"] == ["X"]
    assert captured[0]["namespace"] == "ws1:GraphDB"
    # get_node_edges runs exactly once, inside the lock, to drive cleanup —
    # not as a pre-fetch for lock-set extension.
    assert graph.get_node_edges.await_count == 1


# ---------------------------------------------------------------------------
# ainsert_custom_kg
# ---------------------------------------------------------------------------


class _AbortOnEnterLock:
    """Async context manager that captures lock args and aborts on __aenter__.

    Lets the test inspect the lock_keys argument without having to mock every
    downstream storage operation that would run inside the with-block.
    """

    def __init__(self):
        self.captured: list[dict] = []

    def __call__(self, keys, namespace="default", enable_logging=False):
        self.captured.append({"keys": list(keys), "namespace": namespace})
        return self

    async def __aenter__(self):
        raise _LockCaptured("captured")

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _LockCaptured(RuntimeError):
    """Sentinel raised from the captured lock context to short-circuit the
    enclosing async with block."""


@pytest.mark.asyncio
async def test_ainsert_custom_kg_locks_every_entity_and_endpoint():
    """ainsert_custom_kg must hold a single coarse-grained keyed lock whose
    key set covers every entity name plus every relationship endpoint in the
    batch — sharing the doc-ingest namespace so concurrent callers on
    overlapping entities serialise instead of racing.
    """
    from lightrag import lightrag as lightrag_module
    from lightrag.lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "ws1"
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunks_vdb = _make_vdb_mock(workspace="ws1")
    rag.text_chunks = _make_vdb_mock(workspace="ws1")
    rag.chunk_entity_relation_graph = _make_graph_mock()
    rag.entities_vdb = _make_vdb_mock(workspace="ws1")
    rag.relationships_vdb = _make_vdb_mock(workspace="ws1")
    rag._insert_done = AsyncMock(return_value=None)

    lock_spy = _AbortOnEnterLock()

    custom_kg = {
        "chunks": [],
        "entities": [
            {
                "entity_name": "Alice",
                "entity_type": "PERSON",
                "description": "x",
                "source_id": "chunk-1",
                "file_path": "f",
            },
            {
                "entity_name": "Bob",
                "entity_type": "PERSON",
                "description": "y",
                "source_id": "chunk-1",
                "file_path": "f",
            },
        ],
        "relationships": [
            {
                "src_id": "Alice",
                "tgt_id": "Bob",
                "description": "knows",
                "keywords": "k",
                "weight": 1.0,
                "source_id": "chunk-1",
                "file_path": "f",
            },
            {
                "src_id": "Bob",
                "tgt_id": "Carol",
                "description": "knows",
                "keywords": "k",
                "weight": 1.0,
                "source_id": "chunk-1",
                "file_path": "f",
            },
        ],
    }

    with patch.object(lightrag_module, "get_storage_keyed_lock", lock_spy):
        with pytest.raises(_LockCaptured):
            await rag.ainsert_custom_kg(custom_kg)

    assert len(lock_spy.captured) == 1
    call = lock_spy.captured[0]

    # Namespace matches the doc-ingest pipeline so the same key strings
    # mutually exclude across paths.
    assert call["namespace"] == "ws1:GraphDB"

    # Union of entity names ({Alice, Bob}) and every relationship endpoint
    # ({Alice, Bob, Carol}), sorted.
    assert call["keys"] == ["Alice", "Bob", "Carol"]


@pytest.mark.asyncio
async def test_ainsert_custom_kg_empty_batch_skips_keyed_lock():
    """A custom_kg with no entities or relationships has nothing for the
    business-layer keyed lock to serialise on — no lock is acquired and the
    chunk-only path still completes."""
    from lightrag import lightrag as lightrag_module
    from lightrag.lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunks_vdb = _make_vdb_mock(workspace="")
    rag.text_chunks = _make_vdb_mock(workspace="")
    rag.chunk_entity_relation_graph = _make_graph_mock()
    rag.entities_vdb = _make_vdb_mock(workspace="")
    rag.relationships_vdb = _make_vdb_mock(workspace="")
    rag._insert_done = AsyncMock(return_value=None)

    lock_spy = _AbortOnEnterLock()

    with patch.object(lightrag_module, "get_storage_keyed_lock", lock_spy):
        await rag.ainsert_custom_kg({"chunks": [], "entities": [], "relationships": []})

    assert lock_spy.captured == []
