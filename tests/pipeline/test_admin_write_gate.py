"""Admin graph writes are mutually exclusive with the pipeline (issue #3899 R2/R3).

``NetworkXStorage`` reloads the whole graph when a peer commit lands, and a
reload discards this process's uncommitted in-memory mutations. An admin write
that lands inside a running pipeline batch discards the batch's merge results;
the batch then commits and marks its documents PROCESSED with entities missing,
which nothing heals. ``check_pipeline_busy_or_raise`` only covers one direction
(a running pipeline refuses an admin write) and only at request entry.

``LightRAG._admin_write_gate`` closes the other direction in the core: on a
graph storage that declares ``requires_single_writer`` an admin write holds the
pipeline ``busy`` reservation (``kind="admin"``) for its whole body, so a
pipeline START arriving meanwhile is deferred into the ingress mailbox -- and,
because an admin holder runs no quiescence decision, the gate must drive the
queue once when it releases, or the deferred document sits PENDING forever.

These tests drive a real ``LightRAG`` on the default JSON / Nano / NetworkX
storages, offline, and pin:

- an admin reservation defers a pipeline start and the release-time drive runs
  (the document does NOT stay PENDING);
- the drive is skipped, and the mailbox flag left armed, when nothing was
  deferred and when the release path is cancelled;
- a running or scanning pipeline still refuses the admin write;
- a confirmed-dead admin owner is reclaimed WITHOUT ``recovery_required``, and
  the manual-freeze fields the reclaim clears were already false;
- the hold ceiling releases both gates and fails loud;
- the core-level gate fires for a direct SDK call with no router involved;
- ``ainsert_custom_kg`` takes both gates (R3).
"""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag import LightRAG
from lightrag import lightrag as lightrag_module
from lightrag.base import DocStatus
from lightrag.exceptions import (
    ADMIN_WRITE_PIPELINE_BUSY_PREFIX,
    AdminWriteGateRefusedError,
    AdminWriteHoldExceededError,
)
from lightrag.kg.shared_storage import (
    PipelineReservationConflict,
    acquire_processing_reservation,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
    get_storage_keyed_lock,
    initialize_share_data,
    reconcile_dead_pipeline_reservations,
)
from lightrag.utils import EmbeddingFunc, Tokenizer

# ``check_pipeline_busy_or_raise`` is the preflight every /graph/* route runs
# before the core method, so the REST-path tests at the bottom need it. Import
# it under a clean argv: the router package parses ``sys.argv`` with argparse at
# import time and would choke on pytest's flags. Same idiom, and same reason, as
# tests/kg/test_reservation_dead_process_recovery.py; done via importlib (an
# assignment, not an ``import`` statement) so it is not flagged as a late
# module-level import (E402).
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv
check_pipeline_busy_or_raise = _document_routes.check_pipeline_busy_or_raise

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture
async def rag(tmp_path):
    instance = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"admin-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await instance.initialize_storages()
    # The default graph storage is the one storage the gate applies to.
    assert instance._admin_write_gate_required() is True
    yield instance
    # Let a release-time drive finish before the storages go away.
    await asyncio.gather(
        *lightrag_module._ADMIN_RELEASE_DRIVE_TASKS, return_exceptions=True
    )
    await instance.finalize_storages()


async def _status_handles(rag):
    return (
        await get_namespace_data("pipeline_status", workspace=rag.workspace),
        get_namespace_lock("pipeline_status", workspace=rag.workspace),
    )


def _status_value(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


class _HeldAdminWrite:
    """Start an admin write and park it INSIDE the gate, at the embedding
    round-trip (``entities_vdb.upsert``), until ``release()`` is called."""

    def __init__(self, rag, coro_factory):
        self._rag = rag
        self._factory = coro_factory
        self.entered = asyncio.Event()
        self._release = asyncio.Event()
        self.task: asyncio.Task | None = None
        self._original = rag.entities_vdb.upsert

    async def _slow_upsert(self, data):
        self.entered.set()
        await self._release.wait()
        return await self._original(data)

    async def __aenter__(self):
        self._rag.entities_vdb.upsert = self._slow_upsert
        self.task = asyncio.create_task(self._factory())
        await asyncio.wait(
            {self.task, asyncio.create_task(self.entered.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self.task.done():
            self.task.result()  # surface an early failure
        return self

    def release(self) -> None:
        self._release.set()

    async def __aexit__(self, exc_type, exc, tb):
        self._rag.entities_vdb.upsert = self._original
        if not self.task.done():
            self.release()
            try:
                await self.task
            except BaseException:
                pass
        return False


def _create_alice(rag):
    return rag.acreate_entity(
        "Alice", {"description": "a person", "entity_type": "PERSON"}
    )


async def _wait_for_status(rag, doc_id: str, wanted: str, timeout: float = 10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        doc = await rag.doc_status.get_by_id(doc_id)
        if doc is not None and _status_value(doc["status"]) == wanted:
            return doc
        if asyncio.get_running_loop().time() > deadline:
            return doc
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# R2.1 / R2.4 / R2.5 -- a deferred pipeline start is driven on release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_reservation_defers_pipeline_start_and_release_drives_it(rag):
    """The regression that would otherwise ship: an admin holder of ``busy``
    turns a pipeline start into a mailbox flag that no quiescence decision
    will ever consume. The gate's release must drive the queue, so the
    document does NOT stay PENDING."""
    status, _lock = await _status_handles(rag)
    ingress = await get_pipeline_ingress(rag.workspace)
    await rag.apipeline_enqueue_documents(["hello world document"], ids=["doc-1"])

    async with _HeldAdminWrite(rag, lambda: _create_alice(rag)) as held:
        # The admin write owns the busy slot for its whole body (R2.1) ...
        assert status["busy"] is True
        assert status["busy_owner"]["kind"] == "admin"
        assert status.get("destructive_busy", False) is False  # enqueue allowed
        # ... so a pipeline start is deferred, not run, and not lost.
        await rag.apipeline_process_enqueue_documents()
        assert ingress.counts()["auto_rescan_pending"] is True
        doc = await rag.doc_status.get_by_id("doc-1")
        assert _status_value(doc["status"]) == "pending"
        held.release()
        await held.task

    # Release drove the queue once, in the background.
    doc = await _wait_for_status(rag, "doc-1", "processed")
    assert _status_value(doc["status"]) == "processed"
    assert await rag.chunk_entity_relation_graph.has_node("Alice") is True
    await asyncio.gather(*lightrag_module._ADMIN_RELEASE_DRIVE_TASKS)
    assert status["busy"] is False and status["busy_owner"] is None
    # The drive's own acquire consumed the flag.
    assert ingress.counts()["auto_rescan_pending"] is False


@pytest.mark.asyncio
async def test_release_drive_is_skipped_when_nothing_was_deferred(rag, monkeypatch):
    """No pipeline start during the hold -> no drive, and nothing else changes."""
    scheduled: list[str] = []
    original = LightRAG._schedule_deferred_pipeline_drive

    def _spy(self, workspace):
        scheduled.append(workspace)
        return original(self, workspace)

    monkeypatch.setattr(LightRAG, "_schedule_deferred_pipeline_drive", _spy)
    ingress = await get_pipeline_ingress(rag.workspace)

    await _create_alice(rag)

    assert scheduled == []
    assert ingress.counts()["auto_rescan_pending"] is False


@pytest.mark.asyncio
async def test_release_drive_is_skipped_on_cancellation_and_flag_stays_armed(
    rag, monkeypatch
):
    """A cancelled admin write must not start a processing run on its way
    out (shutdown may be waiting for it); the mailbox flag stays armed so the
    next scan or upload honours the deferred start."""
    scheduled: list[str] = []
    monkeypatch.setattr(
        LightRAG,
        "_schedule_deferred_pipeline_drive",
        lambda self, workspace: scheduled.append(workspace),
    )
    status, _lock = await _status_handles(rag)
    ingress = await get_pipeline_ingress(rag.workspace)
    await rag.apipeline_enqueue_documents(["hello world document"], ids=["doc-1"])

    async with _HeldAdminWrite(rag, lambda: _create_alice(rag)) as held:
        await rag.apipeline_process_enqueue_documents()
        assert ingress.counts()["auto_rescan_pending"] is True
        held.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await held.task

    assert scheduled == []
    assert ingress.counts()["auto_rescan_pending"] is True
    # Both gates released regardless.
    assert status["busy"] is False and status["busy_owner"] is None
    async with asyncio.timeout(2):
        async with get_storage_keyed_lock(
            ["admin"], namespace=f"{rag.workspace}:GraphAdmin"
        ):
            pass


# ---------------------------------------------------------------------------
# R2.1 -- a running or scanning pipeline still refuses the admin write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flags, conflict",
    [
        pytest.param(
            {"busy": True, "busy_owner": {"token": "p", "kind": "processing"}},
            PipelineReservationConflict.BUSY,
            id="busy",
        ),
        pytest.param(
            {"scanning": True, "scanning_owner": {"token": "s", "kind": "scan"}},
            PipelineReservationConflict.SCANNING,
            id="scanning",
        ),
    ],
)
async def test_busy_or_scanning_pipeline_refuses_the_admin_write(rag, flags, conflict):
    status, lock = await _status_handles(rag)
    async with lock:
        status.update(flags)
    try:
        with pytest.raises(AdminWriteGateRefusedError) as excinfo:
            await _create_alice(rag)
    finally:
        async with lock:
            status.update(
                {key: (None if key.endswith("_owner") else False) for key in flags}
            )

    assert str(excinfo.value).startswith(ADMIN_WRITE_PIPELINE_BUSY_PREFIX)
    assert excinfo.value.conflict is conflict
    assert excinfo.value.recovery_required is False
    assert await rag.chunk_entity_relation_graph.has_node("Alice") is False
    # The admin lock was released on the refusal path.
    async with asyncio.timeout(2):
        async with get_storage_keyed_lock(
            ["admin"], namespace=f"{rag.workspace}:GraphAdmin"
        ):
            pass


# ---------------------------------------------------------------------------
# R2.2 -- a dead admin owner is re-runnable, not a recovery fence
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = proc.pid
    proc.kill()
    proc.wait()
    return pid


def test_dead_admin_owner_is_reclaimed_without_recovery_required(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)
    status = {
        "busy": True,
        "busy_owner": {
            "token": "t",
            "pid": _dead_pid(),
            "process_start_id": "gone",
            "kind": "admin",
        },
        # What an admin holder can coexist with: the manual-freeze fields are
        # already idle (see the test below), so the reclaim's clearing of them
        # changes nothing.
        "manual_freeze_requested": False,
        "manual_resetting": False,
        "manual_phase": shared_storage.MANUAL_PHASE_IDLE,
        "manual_owner": None,
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["busy"] is False and status["busy_owner"] is None
    assert "recovery_required" not in status
    assert status["manual_freeze_requested"] is False
    assert status["manual_resetting"] is False
    assert status["manual_phase"] == shared_storage.MANUAL_PHASE_IDLE
    assert status["manual_owner"] is None


@pytest.mark.asyncio
async def test_manual_freeze_cannot_be_raised_while_an_admin_write_holds_busy(rag):
    """Pins the premise R2.2 rests on: the manual freeze is only ever set by
    the processing run that owns ``busy``, and an admin holder of ``busy``
    keeps any processing run from starting -- so while an admin write holds
    the slot the manual-freeze fields are, and stay, false."""
    status, lock = await _status_handles(rag)
    ingress = await get_pipeline_ingress(rag.workspace)

    async with _HeldAdminWrite(rag, lambda: _create_alice(rag)) as held:
        admin_token = status["busy_owner"]["token"]
        # A processing run cannot take the slot (its start is deferred) ...
        reservation = await acquire_processing_reservation(
            status,
            lock,
            token=uuid4().hex,
            already_held=False,
            pipeline_ingress=ingress,
            flags={},
        )
        assert reservation.acquired is False
        assert reservation.conflict is PipelineReservationConflict.BUSY
        # ... and the owner-checked freeze entry refuses a non-owner token.
        assert (
            await rag._begin_manual_drain(uuid4().hex, uuid4().hex, status, lock)
            is False
        )
        assert status.get("manual_freeze_requested", False) is False
        assert status.get("manual_owner") is None
        assert status["busy_owner"]["token"] == admin_token
        held.release()
        await held.task

    await asyncio.gather(
        *lightrag_module._ADMIN_RELEASE_DRIVE_TASKS, return_exceptions=True
    )


# ---------------------------------------------------------------------------
# R2.3 -- the hold ceiling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hold_ceiling_releases_both_gates_and_fails_loud(rag, monkeypatch):
    """A hung embedding endpoint must not fence ingestion indefinitely: past
    ``ADMIN_WRITE_MAX_HOLD_SECONDS`` the admin write fails (500 through the
    routes), and its ``finally`` releases the reservation AND the admin lock."""
    monkeypatch.setattr(lightrag_module, "ADMIN_WRITE_MAX_HOLD_SECONDS", 0.2)
    status, _lock = await _status_handles(rag)

    never = asyncio.Event()

    async def _hung_upsert(data):
        await never.wait()

    original = rag.entities_vdb.upsert
    rag.entities_vdb.upsert = _hung_upsert
    try:
        with pytest.raises(AdminWriteHoldExceededError) as excinfo:
            await _create_alice(rag)
    finally:
        rag.entities_vdb.upsert = original

    assert "LIGHTRAG_ADMIN_WRITE_MAX_HOLD_SECONDS" in str(excinfo.value)
    # Stopped at a suspension point: no commit was in flight, and the message
    # says that rather than the mid-commit wording (see the two tests below).
    assert "stopped at a suspension point" in str(excinfo.value)
    assert "IS durable" not in str(excinfo.value)
    assert status["busy"] is False and status["busy_owner"] is None
    # The admin lock is free again: the next admin write goes straight through.
    async with asyncio.timeout(5):
        result = await rag.acreate_entity(
            "Bob", {"description": "another person", "entity_type": "PERSON"}
        )
    assert result["entity_name"] == "Bob"
    await asyncio.gather(
        *lightrag_module._ADMIN_RELEASE_DRIVE_TASKS, return_exceptions=True
    )


# ---------------------------------------------------------------------------
# R2.6 -- the gate is in the core, not the router
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_gate_fires_for_a_direct_sdk_call(rag):
    """No router, no ``check_pipeline_busy_or_raise``: a direct
    ``LightRAG.aedit_entity`` holds the admin reservation for its body."""
    await _create_alice(rag)
    await asyncio.gather(
        *lightrag_module._ADMIN_RELEASE_DRIVE_TASKS, return_exceptions=True
    )
    status, _lock = await _status_handles(rag)
    assert status["busy"] is False

    async with _HeldAdminWrite(
        rag, lambda: rag.aedit_entity("Alice", {"description": "edited"})
    ) as held:
        assert status["busy"] is True
        assert status["busy_owner"]["kind"] == "admin"
        # And the admin lock is held: a peer cannot take it meanwhile.
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.2):
                async with get_storage_keyed_lock(
                    ["admin"], namespace=f"{rag.workspace}:GraphAdmin"
                ):
                    pass
        held.release()
        result = await held.task

    assert result is not None
    node = await rag.chunk_entity_relation_graph.get_node("Alice")
    assert node["description"] == "edited"
    assert status["busy"] is False and status["busy_owner"] is None


# ---------------------------------------------------------------------------
# R3 -- ainsert_custom_kg is the eighth gated writer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ainsert_custom_kg_takes_both_gates(rag):
    status, _lock = await _status_handles(rag)
    custom_kg = {
        "chunks": [
            {"content": "Alice knows Bob.", "source_id": "chunk-1", "file_path": "f"}
        ],
        "entities": [
            {
                "entity_name": "Alice",
                "entity_type": "PERSON",
                "description": "a person",
                "source_id": "chunk-1",
                "file_path": "f",
            }
        ],
        "relationships": [],
    }

    async with _HeldAdminWrite(rag, lambda: rag.ainsert_custom_kg(custom_kg)) as held:
        assert status["busy"] is True
        assert status["busy_owner"]["kind"] == "admin"
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.2):
                async with get_storage_keyed_lock(
                    ["admin"], namespace=f"{rag.workspace}:GraphAdmin"
                ):
                    pass
        held.release()
        await held.task

    assert status["busy"] is False and status["busy_owner"] is None
    assert await rag.chunk_entity_relation_graph.has_node("Alice") is True


# ---------------------------------------------------------------------------
# R2.3 -- the ceiling must not report a durable write as one that did not happen
# ---------------------------------------------------------------------------
#
# The ceiling stops an admin write by cancelling it, but the admin flows
# deliberately WITHHOLD a cancellation while a storage commit is in flight:
# ``commit_in_storage_io`` finishes the GraphML write and its publication hook
# before re-raising, and the edit/delete/merge paths wrap their commit in
# ``_finish_deferring_cancellation``. So a ceiling that fires mid-commit lets
# that commit land and only then gets its cancellation back. Reporting that as
# "aborted" would send the caller to retry a write that already happened -- into
# "entity already exists" for a create, or a re-applied edit -- and would break
# AGENTS.md *Consistency without transactions*: a durable write must never be
# reported as one that did not happen.


@pytest.mark.asyncio
async def test_ceiling_firing_mid_commit_reports_the_commit_as_durable(
    rag, monkeypatch
):
    """The entity IS on disk afterwards, and the error says so.

    Verified red before the fix: the ceiling used to rewrite every expiry
    cancellation into the same "was aborted" message.
    """
    from lightrag.kg.networkx_impl import NetworkXStorage

    monkeypatch.setattr(lightrag_module, "ADMIN_WRITE_MAX_HOLD_SECONDS", 0.3)
    status, _lock = await _status_handles(rag)
    graphml_file = rag.chunk_entity_relation_graph._graphml_xml_file
    original_write = NetworkXStorage.write_nx_graph

    def _slow_write(graph, file_name, workspace="_"):
        # Runs in the storage-IO pool, so the ceiling's timer still fires: the
        # cancellation lands while commit_in_storage_io is deferring it.
        time.sleep(1.0)
        return original_write(graph, file_name, workspace)

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(_slow_write))

    with pytest.raises(AdminWriteHoldExceededError) as excinfo:
        await _create_alice(rag)

    message = str(excinfo.value)
    assert "IS durable" in message
    assert "Re-read the entity or relation before retrying" in message
    # Not a claim that the write did not happen ...
    assert "aborted" not in message
    # ... because it did: the commit was allowed to finish.
    on_disk = NetworkXStorage.load_nx_graph(graphml_file)
    assert on_disk is not None and on_disk.has_node("Alice")

    # Both gates still released, which is what the ceiling exists for.
    assert status["busy"] is False and status["busy_owner"] is None
    async with asyncio.timeout(5):
        async with get_storage_keyed_lock(
            ["admin"], namespace=f"{rag.workspace}:GraphAdmin"
        ):
            pass


@pytest.mark.asyncio
async def test_the_two_ceiling_messages_are_distinguishable(rag, monkeypatch):
    """Codex review of PR #3901: the timeout must distinguish a cancellation
    taken before the commit point from one re-raised after a successful commit,
    instead of reporting both as aborted. Neither message may claim the
    operation wrote nothing -- a multi-step flow (``_merge_entities_impl``
    commits the merged node before it removes the sources) can have committed at
    an earlier step even when the ceiling catches it at a clean await."""
    from lightrag.kg.networkx_impl import NetworkXStorage

    monkeypatch.setattr(lightrag_module, "ADMIN_WRITE_MAX_HOLD_SECONDS", 0.3)

    # (a) stopped at an ordinary suspension point: the embedding round-trip.
    never = asyncio.Event()
    original_upsert = rag.entities_vdb.upsert
    rag.entities_vdb.upsert = lambda data: never.wait()
    try:
        with pytest.raises(AdminWriteHoldExceededError) as clean:
            await _create_alice(rag)
    finally:
        rag.entities_vdb.upsert = original_upsert

    # (b) stopped inside the commit region.
    original_write = NetworkXStorage.write_nx_graph

    def _slow_write(graph, file_name, workspace="_"):
        time.sleep(1.0)
        return original_write(graph, file_name, workspace)

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(_slow_write))
    with pytest.raises(AdminWriteHoldExceededError) as deferred:
        await rag.acreate_entity(
            "Bob", {"description": "another person", "entity_type": "PERSON"}
        )

    clean_text, deferred_text = str(clean.value), str(deferred.value)
    assert clean_text != deferred_text
    assert "stopped at a suspension point" in clean_text
    assert "IS durable" in deferred_text
    # Neither one tells the caller the operation is undone.
    for text in (clean_text, deferred_text):
        assert "Re-read the entity or relation before retrying" in text
        assert "still durable" in text or "IS durable" in text


# ---------------------------------------------------------------------------
# R1.5 over REST -- the router preflight must not swallow the admin queue
# ---------------------------------------------------------------------------
#
# Every /graph/* route calls check_pipeline_busy_or_raise before the core
# method. It refused on the raw ``busy`` flag, which an admin write now sets
# itself -- so a second concurrent REST admin write was refused with the
# pipeline-busy 409 and never reached the workspace admin lock. The bounded
# queueing R1.5 promises, and the distinct admin-lock refusal, existed only for
# direct SDK callers, and the message blamed document ingestion for what was
# actually another UI edit. Found by the Codex review of PR #3901 on d9ba12b.


async def _preflight(rag):
    """The exact call every /graph/* route makes before the core method."""
    await check_pipeline_busy_or_raise(rag)


@pytest.mark.asyncio
async def test_router_preflight_lets_a_second_admin_write_through_to_the_lock(rag):
    """An admin-owned ``busy`` is not a pipeline-busy refusal."""
    async with _HeldAdminWrite(rag, lambda: _create_alice(rag)) as held:
        status, _lock = await _status_handles(rag)
        assert status["busy"] is True
        assert status["busy_owner"]["kind"] == "admin"

        await _preflight(rag)  # must NOT raise

        held.release()
        await held.task
    await asyncio.gather(
        *lightrag_module._ADMIN_RELEASE_DRIVE_TASKS, return_exceptions=True
    )


@pytest.mark.asyncio
async def test_router_preflight_still_refuses_a_pipeline_owned_busy(rag):
    """The exemption is for ``kind == "admin"`` only. A processing or
    destructive holder, and an unidentifiable one, still refuse: those DO write
    the same graph storages these endpoints mutate."""
    from fastapi import HTTPException

    status, lock = await _status_handles(rag)
    for owner in (
        {"token": "p", "kind": "processing"},
        {"token": "d", "kind": "delete"},
        {"token": "c", "kind": "clear"},
        {"token": "legacy"},  # no kind at all
        "bare-token",  # not a record
        None,  # busy with no owner
    ):
        async with lock:
            status.update({"busy": True, "busy_owner": owner})
        try:
            with pytest.raises(HTTPException) as excinfo:
                await _preflight(rag)
            assert excinfo.value.status_code == 409, owner
            assert "Pipeline is busy" in excinfo.value.detail
        finally:
            async with lock:
                status.update({"busy": False, "busy_owner": None})


@pytest.mark.asyncio
async def test_a_second_rest_admin_write_queues_instead_of_being_refused(rag):
    """End to end on the REST path: preflight, then the core gate. The second
    write WAITS for the first and then succeeds -- it is not refused, and it
    does not run concurrently.

    Verified red before the fix: the preflight raised the pipeline-busy 409.
    """
    order: list[str] = []

    async def _second_write():
        await _preflight(rag)
        order.append("second-entered-gate")
        result = await rag.acreate_entity(
            "Bob", {"description": "another person", "entity_type": "PERSON"}
        )
        order.append("second-done")
        return result

    async with _HeldAdminWrite(rag, lambda: _create_alice(rag)) as held:
        second = asyncio.create_task(_second_write())
        # Give it room to pass the preflight and park on the admin lock.
        await asyncio.sleep(0.2)
        assert order == ["second-entered-gate"]  # past the preflight ...
        assert second.done() is False  # ... and queued, not refused
        assert await rag.chunk_entity_relation_graph.has_node("Bob") is False

        order.append("first-releasing")
        held.release()
        await held.task

    result = await asyncio.wait_for(second, timeout=10)
    assert result["entity_name"] == "Bob"
    assert order == ["second-entered-gate", "first-releasing", "second-done"]
    assert await rag.chunk_entity_relation_graph.has_node("Alice") is True
    assert await rag.chunk_entity_relation_graph.has_node("Bob") is True
    await asyncio.gather(
        *lightrag_module._ADMIN_RELEASE_DRIVE_TASKS, return_exceptions=True
    )


# ---------------------------------------------------------------------------
# R2.4/R2.5 under the SYNCHRONOUS wrappers -- the drive must not outlive the loop
# ---------------------------------------------------------------------------
#
# ``_run_sync`` drives one coroutine with ``run_until_complete``, which stops the
# loop the moment that coroutine returns. A release-time drive created as a
# background task in the gate's ``finally`` is therefore left parked at its first
# await, resuming only if some later synchronous call happens to run the same
# loop. Measured, that is worse than never scheduling it: the drive gets far
# enough to CONSUME the mailbox's auto-rescan flag and take the ``busy``
# reservation, then parks forever -- so the workspace is held busy by a LIVE pid
# that dead-owner reclaim cannot reclaim, and the document has lost the sticky
# signal that would have recovered it. Found by the Codex review of PR #3901 on
# d667799.


def _sync_rag(tmp_path):
    """A LightRAG whose storages are initialized on a loop the SYNC wrappers
    will reuse -- the shape a plain script has."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    instance = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"sync-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    loop.run_until_complete(instance.initialize_storages())
    return instance, loop


def test_sync_wrapper_completes_the_deferred_drive_before_returning(tmp_path):
    """A synchronous admin write must leave the workspace usable.

    Verified red against the background-task version, which returned with the
    document PENDING, ``auto_rescan_pending`` consumed, ``busy`` held by this
    live process, and the drive task parked.
    """
    instance, loop = _sync_rag(tmp_path)
    try:
        loop.run_until_complete(
            instance.apipeline_enqueue_documents(
                ["hello world document"], ids=["doc-1"]
            )
        )

        # A peer's pipeline start, refused while the admin write holds ``busy``:
        # this is what arms the mailbox flag the release must then honour.
        original = instance.entities_vdb.upsert

        async def _upsert_with_a_deferred_start(data):
            await instance.apipeline_process_enqueue_documents()
            return await original(data)

        instance.entities_vdb.upsert = _upsert_with_a_deferred_start

        # THE SYNCHRONOUS WRAPPER -- what a script calls.
        instance.create_entity(
            "Alice", {"description": "a person", "entity_type": "PERSON"}
        )

        async def _inspect():
            status = await get_namespace_data(
                "pipeline_status", workspace=instance.workspace
            )
            ingress = await get_pipeline_ingress(instance.workspace)
            doc = await instance.doc_status.get_by_id("doc-1")
            return status, ingress.counts(), doc

        status, counts, doc = loop.run_until_complete(_inspect())

        # The drive actually ran, inside the call.
        assert _status_value(doc["status"]) == "processed"
        # And released everything it took.
        assert status["busy"] is False
        assert status["busy_owner"] is None
        assert counts["auto_rescan_pending"] is False
        # Nothing was left parked on a loop that has stopped.
        assert [
            t for t in lightrag_module._ADMIN_RELEASE_DRIVE_TASKS if not t.done()
        ] == []
        assert loop.run_until_complete(
            instance.chunk_entity_relation_graph.has_node("Alice")
        )
    finally:
        instance.entities_vdb.upsert = original
        loop.run_until_complete(instance.finalize_storages())
        loop.close()
        asyncio.set_event_loop(None)


def test_sync_wrapper_without_a_deferred_start_drives_nothing(tmp_path):
    """The inline path is still conditional: no deferred start, no drive, so an
    ordinary synchronous edit does not block on the queue."""
    instance, loop = _sync_rag(tmp_path)
    scheduled: list[str] = []
    original_inline = LightRAG._deferred_pipeline_drive

    async def _spy(self, workspace):
        scheduled.append(workspace)
        return await original_inline(self, workspace)

    LightRAG._deferred_pipeline_drive = _spy
    try:
        loop.run_until_complete(
            instance.apipeline_enqueue_documents(
                ["hello world document"], ids=["doc-1"]
            )
        )
        instance.create_entity(
            "Alice", {"description": "a person", "entity_type": "PERSON"}
        )
        assert scheduled == []

        async def _doc():
            return await instance.doc_status.get_by_id("doc-1")

        assert _status_value(loop.run_until_complete(_doc())["status"]) == "pending"
    finally:
        LightRAG._deferred_pipeline_drive = original_inline
        loop.run_until_complete(instance.finalize_storages())
        loop.close()
        asyncio.set_event_loop(None)
