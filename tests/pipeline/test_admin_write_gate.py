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
import subprocess
import sys
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
