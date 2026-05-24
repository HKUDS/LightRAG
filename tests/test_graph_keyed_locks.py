"""
Regression tests for the keyed-lock extension on entity-mutation paths.

`aedit_entity` (rename) and `adelete_by_entity` both rewrite or detach every
incident edge of the target entity. The doc-ingest pipeline
(`operate.py:_locked_process_edges`) locks edges under `sorted([src, tgt])`
keys, so a lock set of only {entity_name} (or {entity_name, new_entity_name})
lives in a disjoint partition from those per-edge locks and would race against
concurrent edge writes.

The fix pre-fetches each entity's edges outside the lock, extends the lock
set to cover every other endpoint, and acquires all keys atomically. These
tests pin the extension by spying on `get_storage_keyed_lock` and asserting
the `keys` argument it receives.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keyed_lock_spy():
    """Return (spy_callable, captured_calls_list).

    The spy returns an async context manager so `async with get_storage_keyed_lock(...)`
    just yields. Each invocation appends {"keys": ..., "namespace": ...} to the
    captured list.
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
    edges_for_entity: list[tuple[str, str]],
    *,
    existing_entity: str = "X",
):
    """Minimal `chunk_entity_relation_graph` mock for aedit / adelete.

    `has_node` returns True only for `existing_entity` so a rename target
    (e.g. "Y") is treated as not-yet-existing — otherwise aedit_entity would
    short-circuit with "Entity name 'Y' already exists" before reaching the
    upsert path.
    """
    graph = MagicMock()
    graph.get_node_edges = AsyncMock(return_value=edges_for_entity)
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
async def test_aedit_entity_rename_locks_extend_to_edge_endpoints():
    """When renaming X -> Y, the lock set must include every other endpoint
    of X's edges (A, B in this fixture), not just {X, Y}.

    Without this, the doc-ingest pipeline can hold sorted([X, A]) or
    sorted([Y, A]) edge locks at the same time aedit_entity is rewriting
    those edges, since {X, Y} and sorted([X, A]) live in disjoint key
    partitions within the same namespace.
    """
    from lightrag import utils_graph

    spy, captured = _make_keyed_lock_spy()
    graph = _make_graph_mock(edges_for_entity=[("X", "A"), ("B", "X")])
    entities_vdb = _make_vdb_mock(workspace="ws1")
    relationships_vdb = _make_vdb_mock(workspace="ws1")

    # Stop the actual edit work from running too far — we only care about the
    # lock arguments. Patching get_node also blocks _edit_entity_impl from
    # walking past the lock with a sentinel exception that we catch below.
    sentinel = RuntimeError("stop after lock acquisition")
    graph.upsert_node.side_effect = sentinel

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

    # Exactly one lock acquisition happened.
    assert len(captured) == 1
    call = captured[0]

    # Namespace pinned to workspace-aware form.
    assert call["namespace"] == "ws1:GraphDB"

    # Keys: {X, Y} ∪ {A, B} from get_node_edges, sorted.
    assert call["keys"] == ["A", "B", "X", "Y"]


@pytest.mark.asyncio
async def test_aedit_entity_non_rename_keeps_single_entity_lock():
    """Non-rename edits don't touch edges, so the lock set stays at just the
    entity name — no need to extend or pre-fetch edges."""
    from lightrag import utils_graph

    spy, captured = _make_keyed_lock_spy()
    graph = _make_graph_mock(edges_for_entity=[])
    entities_vdb = _make_vdb_mock(workspace="")
    relationships_vdb = _make_vdb_mock(workspace="")

    sentinel = RuntimeError("stop after lock acquisition")
    graph.upsert_node.side_effect = sentinel

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

    # No need to pre-fetch edges when not renaming — the implementation only
    # walks get_node_edges inside the rename branch.
    graph.get_node_edges.assert_not_called()


@pytest.mark.asyncio
async def test_aedit_entity_rename_tolerates_edge_fetch_failure():
    """If pre-fetching edges fails (storage hiccup), the rename still
    proceeds with the narrower {entity, new_entity} lock set, logging a
    warning. The PG advisory lock is the final safety net here."""
    from lightrag import utils_graph

    spy, captured = _make_keyed_lock_spy()
    graph = _make_graph_mock(edges_for_entity=[])
    graph.get_node_edges.side_effect = RuntimeError("simulated storage error")
    entities_vdb = _make_vdb_mock(workspace="")
    relationships_vdb = _make_vdb_mock(workspace="")

    sentinel = RuntimeError("stop after lock acquisition")
    graph.upsert_node.side_effect = sentinel

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
    # Falls back to the narrow set when pre-fetch fails.
    assert captured[0]["keys"] == ["X", "Y"]


# ---------------------------------------------------------------------------
# adelete_by_entity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adelete_by_entity_locks_extend_to_edge_endpoints():
    """`delete_node` detaches every incident edge, so the lock set must
    include every other endpoint."""
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

    # Deletion ran to completion since all mocks return defaults.
    assert result.status == "success"

    assert len(captured) == 1
    call = captured[0]
    assert call["namespace"] == "ws1:GraphDB"
    # {X} ∪ {Y, Z} from get_node_edges (called pre-lock for set extension), sorted.
    assert call["keys"] == ["X", "Y", "Z"]

    # get_node_edges is called twice: once pre-lock (for set extension), once
    # inside the lock (the authoritative read used by the cleanup code).
    assert graph.get_node_edges.await_count == 2


@pytest.mark.asyncio
async def test_adelete_by_entity_tolerates_edge_fetch_failure():
    """Pre-fetch failure falls back to the narrow {entity_name} lock; the
    delete still proceeds via the second (inside-lock) get_node_edges."""
    from lightrag import utils_graph

    spy, captured = _make_keyed_lock_spy()
    graph = _make_graph_mock(edges_for_entity=[("X", "Y")])

    call_outcomes = [RuntimeError("simulated storage error"), [("X", "Y")]]

    async def flaky_get_node_edges(_name):
        outcome = call_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    graph.get_node_edges = AsyncMock(side_effect=flaky_get_node_edges)
    entities_vdb = _make_vdb_mock(workspace="")
    relationships_vdb = _make_vdb_mock(workspace="")

    with patch.object(utils_graph, "get_storage_keyed_lock", spy):
        result = await utils_graph.adelete_by_entity(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            entity_name="X",
        )

    assert result.status == "success"
    assert len(captured) == 1
    # Narrow fallback when pre-fetch fails.
    assert captured[0]["keys"] == ["X"]


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
async def test_ainsert_custom_kg_locks_extend_to_every_endpoint():
    """ainsert_custom_kg must hold a single coarse-grained keyed lock whose
    key set covers every entity name plus every relationship endpoint in the
    batch — sharing the namespace with the doc-ingest pipeline so a
    concurrent insert_custom_kg call on overlapping entities serialises
    properly instead of racing.
    """
    from lightrag import lightrag as lightrag_module
    from lightrag.lightrag import LightRAG

    # Build a bare LightRAG shell without running __init__ — we only need a
    # handful of attributes for ainsert_custom_kg to walk far enough to
    # acquire the keyed lock.
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = "ws1"
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunks_vdb = _make_vdb_mock(workspace="ws1")
    rag.text_chunks = _make_vdb_mock(workspace="ws1")
    rag.chunk_entity_relation_graph = _make_graph_mock(edges_for_entity=[])
    rag.entities_vdb = _make_vdb_mock(workspace="ws1")
    rag.relationships_vdb = _make_vdb_mock(workspace="ws1")
    # The finally-block calls _insert_done() which walks several storages we
    # haven't stubbed; replace it wholesale since we only care about the lock
    # arguments.
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

    # Exactly one keyed lock acquisition attempted.
    assert len(lock_spy.captured) == 1
    call = lock_spy.captured[0]

    # Namespace pinned to workspace-aware form, matching the doc-ingest pipeline.
    assert call["namespace"] == "ws1:GraphDB"

    # Keys: union of entity names ({Alice, Bob}) and every relationship
    # endpoint ({Alice, Bob, Carol}), sorted.
    assert call["keys"] == ["Alice", "Bob", "Carol"]


@pytest.mark.asyncio
async def test_ainsert_custom_kg_empty_batch_skips_keyed_lock():
    """A custom_kg with no entities or relationships has nothing for the
    business-layer keyed lock to serialise, so it must not acquire one — the
    chunk-only path still works."""
    from lightrag import lightrag as lightrag_module
    from lightrag.lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunks_vdb = _make_vdb_mock(workspace="")
    rag.text_chunks = _make_vdb_mock(workspace="")
    rag.chunk_entity_relation_graph = _make_graph_mock(edges_for_entity=[])
    rag.entities_vdb = _make_vdb_mock(workspace="")
    rag.relationships_vdb = _make_vdb_mock(workspace="")
    rag._insert_done = AsyncMock(return_value=None)

    lock_spy = _AbortOnEnterLock()

    with patch.object(lightrag_module, "get_storage_keyed_lock", lock_spy):
        # No exception expected — the with-block is skipped entirely.
        await rag.ainsert_custom_kg({"chunks": [], "entities": [], "relationships": []})

    assert lock_spy.captured == []
