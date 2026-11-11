"""
Pin the business-layer keyed-lock contracts on the entity-mutation paths.

`get_storage_keyed_lock(keys, namespace=...)` acquires one mutex per key in
the given namespace, so identical key strings share the same mutex across
callers. Locking `[entity_name]` is therefore already enough to mutually
exclude any concurrent edge write that names the same entity in
`sorted([src, tgt])` — no need to enumerate incident edges here.

These tests pin:
- `aedit_entity` locks exact and canonical source/target candidates.
- `amerge_entities` locks exact and canonical source/target candidates.
- `adelete_by_entity` locks {entity_name}.
- `ainsert_custom_kg` locks every normalized entity name plus every normalized
  relationship endpoint that the batch will write, sharing the doc-ingest
  namespace.
- An empty `ainsert_custom_kg` batch skips the lock entirely.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def single_process_shared_data():
    """Initialize the single-process shared_storage singleton.

    ``LightRAG.ainsert_custom_kg`` calls ``_raise_if_recovery_required``,
    which reads the ``pipeline_status`` namespace via ``get_namespace_data``.
    That call raises ``ValueError`` if ``initialize_share_data()`` was never
    called at all (as it never is running this file standalone), but is a
    documented no-op — via a caught ``PipelineNotInitializedError`` — when
    shared data is initialized yet ``pipeline_status`` itself was not. So we
    only need ``initialize_share_data()`` here, not
    ``initialize_pipeline_status()``. Mirrors the fixture in
    ``tests/kg/test_shared_storage_rpc_counts.py``.
    """
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


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


@pytest.mark.asyncio
async def test_aedit_entity_locks_exact_and_normalized_name_candidates():
    """Normalization resolution stays inside the complete candidate lock set."""
    from lightrag import utils_graph

    spy, captured = _make_keyed_lock_spy()
    graph = _make_graph_mock(existing_entity="A公司")
    entities_vdb = _make_vdb_mock(workspace="ws1")
    relationships_vdb = _make_vdb_mock(workspace="ws1")

    graph.upsert_node.side_effect = RuntimeError("stop after lock acquisition")

    with patch.object(utils_graph, "get_storage_keyed_lock", spy):
        with pytest.raises(RuntimeError, match="stop after lock acquisition"):
            await utils_graph.aedit_entity(
                chunk_entity_relation_graph=graph,
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                entity_name="“Ａ 公 司”",
                updated_data={"entity_name": "“Ｂ 公 司”", "description": "renamed"},
                allow_rename=True,
            )

    assert len(captured) == 1
    assert captured[0]["keys"] == ["A公司", "B公司", "“Ａ 公 司”", "“Ｂ 公 司”"]
    assert captured[0]["namespace"] == "ws1:GraphDB"


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
async def test_amerge_entities_locks_exact_and_normalized_name_candidates():
    """Merge resolution stays inside the complete exact/canonical lock set."""
    from lightrag import utils_graph

    graph = _make_graph_mock(existing_entity="A公司")
    entities_vdb = _make_vdb_mock(workspace="ws1")
    relationships_vdb = _make_vdb_mock(workspace="ws1")
    lock_spy = _AbortOnEnterLock()

    with patch.object(utils_graph, "get_storage_keyed_lock", lock_spy):
        with pytest.raises(_LockCaptured):
            await utils_graph.amerge_entities(
                chunk_entity_relation_graph=graph,
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                source_entities=["“Ａ 公 司”"],
                target_entity="“Ｔ 目 标”",
            )

    assert lock_spy.captured == [
        {
            "keys": ["A公司", "T目标", "“Ａ 公 司”", "“Ｔ 目 标”"],
            "namespace": "ws1:GraphDB",
        }
    ]


@pytest.mark.asyncio
async def test_ainsert_custom_kg_locks_every_entity_and_endpoint(
    single_process_shared_data,
):
    """ainsert_custom_kg must hold a single coarse-grained keyed lock whose
    key set covers every normalized entity name plus every normalized
    relationship endpoint in the batch — sharing the doc-ingest namespace so
    concurrent callers on overlapping entities serialise instead of racing.
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
                "entity_name": "Ａｌｉｃｅ",
                "entity_type": "PERSON",
                "description": "x",
                "source_id": "chunk-1",
                "file_path": "f",
            },
            {
                "entity_name": "“Ｂｏｂ”",
                "entity_type": "PERSON",
                "description": "y",
                "source_id": "chunk-1",
                "file_path": "f",
            },
        ],
        "relationships": [
            {
                "src_id": "Ａｌｉｃｅ",
                "tgt_id": "“Ｂｏｂ”",
                "description": "knows",
                "keywords": "k",
                "weight": 1.0,
                "source_id": "chunk-1",
                "file_path": "f",
            },
            {
                "src_id": "Ｂｏｂ",
                "tgt_id": "Ｃａｒｏｌ",
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

    # The raw full-width/quoted spellings are normalized before the union is
    # locked, so keys collide with the extraction pipeline's canonical names.
    assert call["keys"] == ["Alice", "Bob", "Carol"]


@pytest.mark.asyncio
async def test_ainsert_custom_kg_empty_batch_skips_keyed_lock(
    single_process_shared_data,
):
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


# ---------------------------------------------------------------------------
# Admin-write gate (issue #3899 R1)
# ---------------------------------------------------------------------------
#
# On a graph storage that declares ``requires_single_writer`` the public
# ``LightRAG`` admin writers wrap their body in ``_admin_write_gate``, which
# acquires, in this fixed order: the workspace admin lock (``{ws}:GraphAdmin`` /
# ``"admin"``, WAITED for), then the pipeline ``busy`` reservation (REFUSES),
# and only then does the ``utils_graph`` body take its per-entity keys in
# ``{ws}:GraphDB``. These tests pin that order, that no public writer re-enters
# the admin lock, the acquire-timeout refusal, the two distinguishable 409
# bodies, the scoping rule, and the ``ClassVar`` trap on the declaration.


class _SingleWriterGraphMock(MagicMock):
    """A graph mock whose CLASS declares the single-writer requirement, the way
    ``NetworkXStorage`` does (``type(graph).requires_single_writer``)."""

    requires_single_writer = True


class _ServerBackedGraphMock(MagicMock):
    """A graph mock standing in for a server-backed store: no declaration, so
    the ``BaseGraphStorage`` default (``False``) applies."""


def _admin_graph_mock(cls=_SingleWriterGraphMock, **kwargs):
    graph = _make_graph_mock(**kwargs)
    typed = cls()
    for name in (
        "get_node_edges",
        "has_node",
        "get_node",
        "upsert_node",
        "upsert_edge",
        "upsert_nodes_batch",
        "upsert_edges_batch",
        "has_nodes_batch",
        "delete_node",
        "get_edge",
        "index_done_callback",
    ):
        setattr(typed, name, getattr(graph, name))
    typed.remove_nodes = AsyncMock(return_value=None)
    typed.remove_edges = AsyncMock(return_value=None)
    typed.has_edge = AsyncMock(return_value=True)
    return typed


def _make_gated_rag(workspace: str = "ws1", *, graph=None):
    """A bare ``LightRAG`` with the storages an admin writer touches mocked."""
    from lightrag.lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = workspace
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunk_entity_relation_graph = (
        graph if graph is not None else _admin_graph_mock()
    )
    rag.entities_vdb = _make_vdb_mock(workspace=workspace)
    rag.relationships_vdb = _make_vdb_mock(workspace=workspace)
    rag.chunks_vdb = _make_vdb_mock(workspace=workspace)
    rag.text_chunks = _make_vdb_mock(workspace=workspace)
    rag.entity_chunks = None
    rag.relation_chunks = None
    rag._insert_done = AsyncMock(return_value=None)
    rag._migrate_chunk_tracking_before_creation = AsyncMock(return_value=None)
    return rag


class _OrderRecorder:
    """Spy for BOTH ``get_storage_keyed_lock`` bindings and the reservation.

    Records ``("lock", namespace, keys)`` for every keyed-lock acquisition and
    ``("reservation",)`` when the gate takes the pipeline reservation. Lock
    contexts are no-ops unless ``abort_namespace_suffix`` matches, in which case
    entering raises ``_LockCaptured`` so a writer stops right after its first
    per-key acquisition (the point where the order is already decided).
    """

    def __init__(self, abort_namespace_suffix: str | None = None):
        self.events: list[tuple] = []
        self._abort_suffix = abort_namespace_suffix

    def keyed_lock(self, keys, namespace="default", enable_logging=False):
        self.events.append(("lock", namespace, sorted(keys)))
        recorder = self

        class _Ctx:
            async def __aenter__(self_inner):
                if recorder._abort_suffix and namespace.endswith(
                    recorder._abort_suffix
                ):
                    raise _LockCaptured("captured")
                return self_inner

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()

    def wrap_reservation(self, real):
        async def _acquire(*args, **kwargs):
            self.events.append(("reservation",))
            return await real(*args, **kwargs)

        return _acquire

    def admin_lock_events(self):
        return [
            e for e in self.events if e[0] == "lock" and e[1].endswith(":GraphAdmin")
        ]

    def key_lock_events(self):
        return [e for e in self.events if e[0] == "lock" and e[1].endswith(":GraphDB")]


@pytest.fixture
def ws1_pipeline_status(single_process_shared_data):
    """Shared data for the gate tests; each test bootstraps ``pipeline_status``
    for workspace ws1 through ``_bootstrap_status`` so the gate's reservation
    half is exercised, not skipped."""
    yield "ws1"


async def _bootstrap_status(workspace: str):
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_namespace_lock,
        initialize_pipeline_status,
    )

    await initialize_pipeline_status(workspace=workspace)
    return (
        await get_namespace_data("pipeline_status", workspace=workspace),
        get_namespace_lock("pipeline_status", workspace=workspace),
    )


@pytest.mark.asyncio
async def test_amerge_entities_takes_admin_lock_then_reservation_then_entity_keys(
    ws1_pipeline_status,
):
    """The fixed acquisition order (R1.3): admin lock -> reservation -> keys."""
    from lightrag import lightrag as lightrag_module
    from lightrag import utils_graph

    status, _lock = await _bootstrap_status("ws1")
    rag = _make_gated_rag("ws1")
    recorder = _OrderRecorder(abort_namespace_suffix=":GraphDB")

    with (
        patch.object(lightrag_module, "get_storage_keyed_lock", recorder.keyed_lock),
        patch.object(utils_graph, "get_storage_keyed_lock", recorder.keyed_lock),
        patch.object(
            lightrag_module,
            "acquire_reservation",
            recorder.wrap_reservation(lightrag_module.acquire_reservation),
        ),
    ):
        with pytest.raises(_LockCaptured):
            await rag.amerge_entities(["A", "B"], "T")

    assert recorder.events[0] == ("lock", "ws1:GraphAdmin", ["admin"])
    assert recorder.events[1] == ("reservation",)
    assert recorder.events[2][0] == "lock" and recorder.events[2][1] == "ws1:GraphDB"
    assert {"A", "B", "T"} <= set(recorder.events[2][2])
    assert len(recorder.events) == 3
    # The reservation is released on the way out even though the body aborted.
    assert status["busy"] is False and status["busy_owner"] is None


_PUBLIC_ADMIN_WRITERS = [
    pytest.param(
        lambda rag: rag.adelete_by_entity("X"),
        id="adelete_by_entity",
    ),
    pytest.param(
        lambda rag: rag.adelete_by_relation("X", "Y"),
        id="adelete_by_relation",
    ),
    pytest.param(
        lambda rag: rag.aedit_entity("X", {"description": "d"}, allow_rename=False),
        id="aedit_entity",
    ),
    pytest.param(
        # Rename onto an existing name with allow_merge: the branch that
        # delegates to _edit_entity_impl AND _merge_entities_impl -- the one
        # place a re-entry through the public merge API could hide.
        lambda rag: rag.aedit_entity(
            "X", {"entity_name": "Y"}, allow_rename=True, allow_merge=True
        ),
        id="aedit_entity_rename_merge",
    ),
    pytest.param(
        lambda rag: rag.aedit_relation("X", "Y", {"description": "d"}),
        id="aedit_relation",
    ),
    pytest.param(
        lambda rag: rag.acreate_entity("New", {"description": "d"}),
        id="acreate_entity",
    ),
    pytest.param(
        lambda rag: rag.acreate_relation("X", "Y", {"description": "d"}),
        id="acreate_relation",
    ),
    pytest.param(
        lambda rag: rag.amerge_entities(["X"], "Y"),
        id="amerge_entities",
    ),
    pytest.param(
        lambda rag: rag.ainsert_custom_kg(
            {
                "chunks": [],
                "entities": [
                    {
                        "entity_name": "X",
                        "entity_type": "PERSON",
                        "description": "d",
                        "source_id": "chunk-1",
                        "file_path": "f",
                    }
                ],
                "relationships": [],
            }
        ),
        id="ainsert_custom_kg",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("invoke", _PUBLIC_ADMIN_WRITERS)
async def test_no_public_admin_writer_re_enters_the_admin_lock(
    ws1_pipeline_status, invoke
):
    """Each writer acquires the admin lock exactly ONCE, first, and never
    re-enters it from inside its body (the R1.3 property the order rests on:
    ``aedit_entity``'s rename branch and ``amerge_entities`` reach the
    lock-free ``_edit_entity_impl`` / ``_merge_entities_impl``, never a public
    admin API). Both graph names exist here so every branch runs."""
    from lightrag import lightrag as lightrag_module
    from lightrag import utils_graph

    status, _lock = await _bootstrap_status("ws1")
    graph = _admin_graph_mock()
    graph.has_node = AsyncMock(side_effect=lambda name: name in {"X", "Y"})
    rag = _make_gated_rag("ws1", graph=graph)
    recorder = _OrderRecorder()

    with (
        patch.object(lightrag_module, "get_storage_keyed_lock", recorder.keyed_lock),
        patch.object(utils_graph, "get_storage_keyed_lock", recorder.keyed_lock),
    ):
        try:
            await invoke(rag)
        except Exception:
            # A mock-induced failure deep in the body is fine: the property
            # under test is decided by the acquisitions recorded before it.
            pass

    assert recorder.events[0] == ("lock", "ws1:GraphAdmin", ["admin"])
    assert len(recorder.admin_lock_events()) == 1
    assert recorder.key_lock_events(), "the body never reached its per-key lock"
    assert status["busy"] is False and status["busy_owner"] is None


@pytest.mark.asyncio
async def test_admin_lock_acquire_timeout_is_refused_with_its_own_phrase(
    ws1_pipeline_status, monkeypatch
):
    """R1.5 / R1.6: a peer holding the admin lock past the acquire timeout turns
    into ``AdminWriteGateRefusedError`` (the graph routes map it to 409) whose
    text starts with the admin-lock phrase, not the pipeline-busy one."""
    from lightrag import lightrag as lightrag_module
    from lightrag.exceptions import (
        ADMIN_WRITE_LOCK_BUSY_PREFIX,
        AdminWriteGateRefusedError,
    )
    from lightrag.kg.shared_storage import get_storage_keyed_lock

    status, _lock = await _bootstrap_status("ws1")
    rag = _make_gated_rag("ws1")
    monkeypatch.setattr(lightrag_module, "ADMIN_WRITE_LOCK_ACQUIRE_TIMEOUT", 0.05)

    # A "peer admin write" holding the real admin lock.
    async with get_storage_keyed_lock(["admin"], namespace="ws1:GraphAdmin"):
        with pytest.raises(AdminWriteGateRefusedError) as excinfo:
            await rag.acreate_entity("New", {"description": "d"})

    assert str(excinfo.value).startswith(ADMIN_WRITE_LOCK_BUSY_PREFIX)
    assert excinfo.value.fence == "admin_lock"
    assert excinfo.value.recovery_required is False
    # Refused BEFORE the reservation: the pipeline slot was never taken.
    assert status["busy"] is False and status["busy_owner"] is None
    rag.chunk_entity_relation_graph.upsert_node.assert_not_called()


@pytest.mark.asyncio
async def test_the_two_409_bodies_are_distinguishable_by_leading_phrase(
    ws1_pipeline_status, monkeypatch
):
    """R1.6: the admin-lock refusal and the pipeline-busy refusal must be
    tellable apart from the ``detail`` text, because the client retry semantics
    differ (one clears when a peer edit finishes, the other when ingestion
    does)."""
    from lightrag import lightrag as lightrag_module
    from lightrag.exceptions import (
        ADMIN_WRITE_LOCK_BUSY_PREFIX,
        ADMIN_WRITE_PIPELINE_BUSY_PREFIX,
        AdminWriteGateRefusedError,
    )
    from lightrag.kg.shared_storage import (
        PipelineReservationConflict,
        get_storage_keyed_lock,
    )

    status, lock = await _bootstrap_status("ws1")
    rag = _make_gated_rag("ws1")
    monkeypatch.setattr(lightrag_module, "ADMIN_WRITE_LOCK_ACQUIRE_TIMEOUT", 0.05)

    async with get_storage_keyed_lock(["admin"], namespace="ws1:GraphAdmin"):
        with pytest.raises(AdminWriteGateRefusedError) as lock_refusal:
            await rag.acreate_entity("New", {"description": "d"})

    async with lock:
        status.update(
            {"busy": True, "busy_owner": {"token": "peer", "kind": "processing"}}
        )
    try:
        with pytest.raises(AdminWriteGateRefusedError) as busy_refusal:
            await rag.acreate_entity("New", {"description": "d"})
    finally:
        async with lock:
            status.update({"busy": False, "busy_owner": None})

    lock_text, busy_text = str(lock_refusal.value), str(busy_refusal.value)
    assert lock_text.startswith(ADMIN_WRITE_LOCK_BUSY_PREFIX)
    assert busy_text.startswith(ADMIN_WRITE_PIPELINE_BUSY_PREFIX)
    assert not lock_text.startswith(ADMIN_WRITE_PIPELINE_BUSY_PREFIX)
    assert not busy_text.startswith(ADMIN_WRITE_LOCK_BUSY_PREFIX)
    assert busy_refusal.value.conflict is PipelineReservationConflict.BUSY
    assert busy_refusal.value.fence == "busy"


@pytest.mark.asyncio
async def test_gate_is_scoped_by_the_graph_storage_declaration(ws1_pipeline_status):
    """R1.7: a ``requires_single_writer`` graph store takes the admin lock; a
    server-backed graph store -- even combined with JSON chunk tracking and
    Nano vectors, the file-backed stores the mechanism cannot reach -- takes
    neither the lock nor the reservation and runs exactly as before."""
    from lightrag import lightrag as lightrag_module
    from lightrag import utils_graph
    from lightrag.kg.json_kv_impl import JsonKVStorage
    from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage

    status, _lock = await _bootstrap_status("ws1")

    gated = _make_gated_rag("ws1")
    assert gated._admin_write_gate_required() is True

    ungated = _make_gated_rag("ws1", graph=_ServerBackedGraphMock())
    ungated.chunk_entity_relation_graph.has_node = AsyncMock(return_value=False)
    ungated.chunk_entity_relation_graph.upsert_node = AsyncMock(return_value=None)
    ungated.chunk_entity_relation_graph.index_done_callback = AsyncMock(
        return_value=None
    )
    # File-backed KV / vector stores beside the server-backed graph: they do
    # not make the gate apply -- only the graph storage's class decides.
    ungated.entity_chunks = MagicMock(spec=JsonKVStorage)
    ungated.entity_chunks.get_by_id = AsyncMock(return_value=None)
    ungated.entity_chunks.upsert = AsyncMock(return_value=None)
    ungated.entity_chunks.index_done_callback = AsyncMock(return_value=None)
    ungated.entities_vdb = MagicMock(spec=NanoVectorDBStorage)
    ungated.entities_vdb.global_config = {"workspace": "ws1"}
    ungated.entities_vdb.upsert = AsyncMock(return_value=None)
    ungated.entities_vdb.index_done_callback = AsyncMock(return_value=None)
    assert ungated._admin_write_gate_required() is False

    recorder = _OrderRecorder()
    with (
        patch.object(lightrag_module, "get_storage_keyed_lock", recorder.keyed_lock),
        patch.object(utils_graph, "get_storage_keyed_lock", recorder.keyed_lock),
        patch.object(
            lightrag_module,
            "acquire_reservation",
            recorder.wrap_reservation(lightrag_module.acquire_reservation),
        ),
    ):
        try:
            await ungated.acreate_entity("New", {"description": "d"})
        except Exception:
            pass

    assert recorder.admin_lock_events() == []
    assert ("reservation",) not in recorder.events
    assert recorder.key_lock_events(), "the per-key lock is still taken"
    assert status["busy"] is False


def test_requires_single_writer_is_a_class_declaration_not_a_dataclass_field():
    """R1.1: the declaration lives on ``BaseGraphStorage`` as a ``ClassVar``
    (True on ``NetworkXStorage`` alone), so it neither appears in any graph
    storage's dataclass fields nor changes a constructor signature."""
    import dataclasses
    import importlib

    from lightrag.base import BaseGraphStorage
    from lightrag.kg import STORAGES
    from lightrag.kg.networkx_impl import NetworkXStorage

    assert BaseGraphStorage.requires_single_writer is False
    assert NetworkXStorage.requires_single_writer is True
    # ``dataclasses.fields`` lists constructor fields only (a ClassVar is
    # recorded as a pseudo-field and excluded), which is exactly the trap.
    assert "requires_single_writer" not in {
        f.name for f in dataclasses.fields(NetworkXStorage)
    }
    assert "requires_single_writer" not in {
        f.name for f in dataclasses.fields(BaseGraphStorage)
    }

    declared_true = set()
    for name, module_path in STORAGES.items():
        if not name.endswith("GraphStorage") and name != "NetworkXStorage":
            continue
        try:
            module = importlib.import_module(module_path, package="lightrag")
        except Exception:
            continue  # optional backend dependency not installed here
        cls = getattr(module, name)
        assert "requires_single_writer" not in {
            f.name for f in dataclasses.fields(cls)
        }, name
        if cls.requires_single_writer:
            declared_true.add(name)
    assert declared_true == {"NetworkXStorage"}
