"""Layer-3 dead-process recovery for pipeline reservations.

When a worker holding a pipeline reservation (busy / destructive / scanning /
pending-enqueue) is SIGKILLed, ``reconcile_dead_pipeline_reservations`` reclaims
the slot at the next acquire IFF the owner process is confirmed dead:

- processing / scan owners → cleared for a safe re-run,
- custom_chunks / delete / clear owners → cleared but ``recovery_required`` is
  raised to fence the workspace (they may have half-committed),
- pending-enqueue tokens → dead tokens dropped, count recalibrated.

The reclaim itself is Linux-multiworker gated; tests force the gate on to drive
the logic off-Linux. The ``recovery_required`` *guard* (refusing mutations while
fenced) is NOT gated and is tested directly.
"""

import asyncio
import importlib
import os
import subprocess
import sys

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    _INTERNAL_PIPELINE_STATUS_FIELDS,
    finalize_share_data,
    get_namespace_data,
    initialize_pipeline_status,
    initialize_share_data,
    make_owner_record,
    pipeline_recovery_blocked_message,
    reconcile_dead_pipeline_reservations,
)

# Import document_routes under a clean argv — its module chain parses CLI args at
# import time and would otherwise choke on pytest's argv. Kept below the regular
# imports and done via importlib (an assignment, not an ``import`` statement) so
# it is not flagged as a late module-level import (E402); shared_storage is a
# lower-level module that does not parse argv, so it imports normally above.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = proc.pid
    proc.kill()
    proc.wait()
    return pid


@pytest.fixture
def recovery_enabled(monkeypatch):
    """Force the Linux-multiworker gate on so reconcile logic runs anywhere."""
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)


# ---------------------------------------------------------------------------
# reconcile_dead_pipeline_reservations — pure logic on a plain status dict
# ---------------------------------------------------------------------------


def _dead_owner(kind: str) -> dict:
    return {
        "token": "t",
        "pid": _dead_pid(),
        "process_start_id": "gone",
        "kind": kind,
    }


def test_reconcile_reclaims_dead_processing_owner(recovery_enabled):
    status = {"busy": True, "busy_owner": _dead_owner("processing")}
    reconcile_dead_pipeline_reservations(status)
    assert status["busy"] is False
    assert status["busy_owner"] is None
    assert "recovery_required" not in status  # re-runnable, not fenced


def test_reconcile_reclaims_dead_scan_owner(recovery_enabled):
    status = {
        "scanning": True,
        "scanning_exclusive": True,
        "scanning_owner": _dead_owner("scan"),
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["scanning"] is False
    assert status["scanning_exclusive"] is False
    assert status["scanning_owner"] is None
    assert "recovery_required" not in status


@pytest.mark.parametrize("kind", ["custom_chunks", "delete", "clear"])
def test_reconcile_fences_dead_non_rerunnable_owner(recovery_enabled, kind):
    status = {
        "busy": True,
        "destructive_busy": kind in ("delete", "clear"),
        "busy_owner": _dead_owner(kind),
        "operation_record": {"kind": kind, "doc_id": "doc-1"},
    }
    reconcile_dead_pipeline_reservations(status)
    # Flags + owner cleared, but the workspace is fenced.
    assert status["busy"] is False
    assert status["destructive_busy"] is False
    assert status["busy_owner"] is None
    rec = status["recovery_required"]
    assert rec["kind"] == kind
    assert rec["owner_key"] == "busy_owner"
    assert rec["operation_record"] == {"kind": kind, "doc_id": "doc-1"}


def test_reconcile_keeps_live_owner(recovery_enabled):
    status = {
        "busy": True,
        "busy_owner": make_owner_record("t", "processing"),  # our own live pid
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["busy"] is True  # live owner never reclaimed
    assert status["busy_owner"] is not None


def test_reconcile_drops_dead_enqueue_tokens(recovery_enabled):
    status = {
        "pending_enqueues": 2,
        "pending_enqueue_tokens": {
            "live": {"pid": os.getpid(), "process_start_id": None},
            "dead": {"pid": _dead_pid(), "process_start_id": "gone"},
        },
    }
    reconcile_dead_pipeline_reservations(status)
    assert set(status["pending_enqueue_tokens"]) == {"live"}
    assert status["pending_enqueues"] == 1


def test_reconcile_recalibrates_enqueue_count(recovery_enabled):
    # count drifted (e.g. crash between "dropped token" and "updated count").
    status = {
        "pending_enqueues": 5,
        "pending_enqueue_tokens": {"live": {"pid": os.getpid()}},
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["pending_enqueues"] == 1


def test_reconcile_is_noop_when_disabled():
    """Without the Linux-multiworker gate, a dead owner is NOT reclaimed."""
    status = {"busy": True, "busy_owner": _dead_owner("processing")}
    reconcile_dead_pipeline_reservations(status)  # gate off (default off-Linux)
    assert status["busy"] is True
    assert status["busy_owner"] is not None


def test_pipeline_recovery_blocked_message():
    status = {
        "recovery_required": {
            "kind": "delete",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "delete", "doc_id": "doc-9"},
        }
    }
    msg = pipeline_recovery_blocked_message(status)
    assert "delete" in msg and "doc-9" in msg
    assert (
        pipeline_recovery_blocked_message({}) == "Pipeline is not fenced for recovery."
    )


def test_internal_fields_constant_covers_recovery_state():
    for field in (
        "busy_owner",
        "scanning_owner",
        "pending_enqueue_tokens",
        "operation_record",
        "recovery_required",
    ):
        assert field in _INTERNAL_PIPELINE_STATUS_FIELDS


# ---------------------------------------------------------------------------
# recovery_required guard — NOT gated; refuses mutations while fenced
# ---------------------------------------------------------------------------


class _Rag:
    workspace = "recovery-guard-ws"


async def _seed_recovery_required(rag):
    await initialize_pipeline_status(workspace=rag.workspace)
    ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    ps["recovery_required"] = {
        "kind": "delete",
        "owner_key": "busy_owner",
        "operation_record": {"kind": "delete", "doc_id": "doc-1"},
    }
    return ps


@pytest.mark.offline
async def test_acquire_destructive_refuses_when_recovery_required():
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await _seed_recovery_required(rag)
        acquired, reason = await dr._acquire_destructive_busy(
            rag,
            "tok",
            kind="clear",
            operation_record={"kind": "clear", "scope": "all"},
        )
        assert acquired is False
        assert "fenced" in reason.lower()
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_reserve_enqueue_slot_raises_when_recovery_required():
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await _seed_recovery_required(rag)
        with pytest.raises(dr.HTTPException) as excinfo:
            await dr._reserve_enqueue_slot(rag, "enq-tok")
        assert excinfo.value.status_code == 503
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_apipeline_enqueue_documents_refuses_when_recovery_required():
    """The core write path — public ``ainsert`` / direct callers bypass the REST
    ``_reserve_enqueue_slot`` guard — must refuse on a fenced workspace, or it
    would write full_docs/doc_status onto a partially-committed store."""
    from lightrag import LightRAG

    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = LightRAG.__new__(LightRAG)
        rag.workspace = "recovery-enqueue-ws"
        await initialize_pipeline_status(workspace=rag.workspace)
        ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        ps["recovery_required"] = {
            "kind": "clear",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "clear", "scope": "all"},
        }
        with pytest.raises(RuntimeError) as excinfo:
            await rag.apipeline_enqueue_documents(["hello"])
        assert "fenced" in str(excinfo.value).lower()
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_processing_loop_bails_when_recovery_required():
    """The processing loop must NOT acquire + process on a fenced workspace,
    even though reconcile cleared ``busy`` when it raised the fence."""
    from lightrag import LightRAG

    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = LightRAG.__new__(LightRAG)
        rag.workspace = "recovery-process-ws"
        await initialize_pipeline_status(workspace=rag.workspace)
        ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        ps["busy"] = False  # reconcile cleared it when fencing
        ps["recovery_required"] = {
            "kind": "delete",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "delete", "doc_id": "doc-1"},
        }
        # Returns early without acquiring busy or touching storage.
        await rag.apipeline_process_enqueue_documents()
        assert ps.get("busy") is False  # never acquired
        assert ps.get("recovery_required") is not None  # fence untouched
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_graph_mutation_sdk_refuses_when_recovery_required():
    """Direct SDK graph edits (bypassing the REST check_pipeline_busy_or_raise
    fence) must refuse on a fenced workspace — the ``_raise_if_recovery_required``
    guard fires before any graph storage is touched."""
    from lightrag import LightRAG

    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = LightRAG.__new__(LightRAG)
        rag.workspace = "recovery-graph-ws"
        await initialize_pipeline_status(workspace=rag.workspace)
        ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        ps["recovery_required"] = {
            "kind": "delete",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "delete", "doc_id": "doc-1"},
        }
        with pytest.raises(RuntimeError, match="fenced"):
            await rag.acreate_entity("E", {"description": "x"})
        with pytest.raises(RuntimeError, match="fenced"):
            await rag.adelete_by_entity("E")
        with pytest.raises(RuntimeError, match="fenced"):
            await rag.amerge_entities(["A"], "B")
        with pytest.raises(RuntimeError, match="fenced"):
            await rag.ainsert_custom_kg({"chunks": []})
    finally:
        finalize_share_data()


def _endpoint(router, name):
    return [r.endpoint for r in router.routes if getattr(r, "name", "") == name][-1]


@pytest.mark.offline
async def test_reprocess_endpoint_refuses_when_recovery_required(tmp_path):
    """/reprocess_failed must return 503 on a fenced workspace instead of
    reporting a false 'reprocessing_started' (the processing loop would bail, so
    nothing would actually be queued)."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await _seed_recovery_required(rag)
        router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
        reprocess = _endpoint(router, "reprocess_failed_documents")
        with pytest.raises(dr.HTTPException) as excinfo:
            await reprocess(set())
        assert excinfo.value.status_code == 503
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_force_reset_recovery_endpoint(tmp_path):
    """The recovery entry requires explicit confirm and only then drops the
    fence (it does not repair — that is #3400)."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        ps = await _seed_recovery_required(rag)
        router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
        force_reset = _endpoint(router, "force_reset_recovery")

        # confirm=False → refused, fence untouched.
        with pytest.raises(dr.HTTPException) as excinfo:
            await force_reset(dr.ForceResetRecoveryRequest(confirm=False))
        assert excinfo.value.status_code == 400
        assert ps.get("recovery_required") is not None

        # confirm=True → fence + lingering reservation state cleared.
        resp = await force_reset(dr.ForceResetRecoveryRequest(confirm=True))
        assert resp.status == "reset"
        assert ps.get("recovery_required") is None
        assert ps.get("busy") is False
        assert ps.get("busy_owner") is None

        # No fence left → idempotent no-op.
        resp2 = await force_reset(dr.ForceResetRecoveryRequest(confirm=True))
        assert resp2.status == "no_recovery_required"
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_pipeline_status_filters_internal_fields(tmp_path):
    """Internal reservation-ownership / recovery bookkeeping must never appear on
    the /pipeline_status response (raw tokens, PIDs, per-token sets)."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await initialize_pipeline_status(workspace=rag.workspace)
        ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        ps.update(
            {
                "busy_owner": {"token": "secret", "pid": 1, "kind": "processing"},
                "scanning_owner": {"token": "s", "pid": 2, "kind": "scan"},
                "pending_enqueue_tokens": {"e": {"pid": 3}},
                "operation_record": {"kind": "delete", "doc_id": "d"},
                "recovery_required": {"kind": "delete"},
            }
        )
        router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
        get_status = _endpoint(router, "get_pipeline_status")
        resp = await get_status()
        dumped = resp.model_dump()
        for field in _INTERNAL_PIPELINE_STATUS_FIELDS:
            assert field not in dumped, f"{field} leaked to /pipeline_status"
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_pipeline_status_reads_one_proxy_snapshot(monkeypatch, tmp_path):
    class SnapshotOnlyStatus(dict):
        copy_calls = 0

        def copy(self):
            self.copy_calls += 1
            return dict.copy(self)

        def get(self, *_args, **_kwargs):
            raise AssertionError("pipeline status must be read from one snapshot")

    status = SnapshotOnlyStatus(
        {
            "busy": False,
            "history_messages": [],
            "job_start": None,
        }
    )

    async def fake_get_namespace_data(*_args, **_kwargs):
        return status

    async def fake_update_flags(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_storage, "get_namespace_lock", lambda *_a, **_k: asyncio.Lock()
    )
    monkeypatch.setattr(
        shared_storage, "get_all_update_flags_status", fake_update_flags
    )

    rag = _Rag()
    router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
    response = await _endpoint(router, "get_pipeline_status")()

    assert response.busy is False
    assert status.copy_calls == 1


class _DeferRag:
    """Minimal rag double for run_scanning_process's deferred-drive path."""

    def __init__(self, workspace, process_calls):
        self.workspace = workspace
        self._process_calls = process_calls

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return None

    async def apipeline_process_enqueue_documents(self):
        self._process_calls.append(1)


class _BoomDocManager:
    """Raises during classification so no scan branch drives the queue."""

    def scan_directory_for_new_files(self):
        raise RuntimeError("classification boom")


@pytest.mark.offline
async def test_scan_drives_deferred_processing_on_error_path():
    """A scan whose ``scanning_exclusive`` fence turned away a processing request
    must still drive the queue once after releasing — otherwise the SDK-inserted
    PENDING doc (no scan-visible file) is stranded. Here classification raises, so
    only the finally's deferred-processing drive can trigger it."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "scan-defer-ws"
        await initialize_pipeline_status(workspace=ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        # The scanning_exclusive fence deferred a processing request earlier.
        ps["scan_deferred_processing"] = True

        process_calls: list[int] = []
        await dr.run_scanning_process(
            _DeferRag(ws, process_calls), _BoomDocManager(), "track-defer"
        )

        # The deferred request was honoured despite the classification error.
        assert process_calls == [1]
        # The check is read-only: the real clear happens in
        # acquire_processing_reservation when a run takes the slot, which this
        # mock drive skips — so the flag stays set (a live acquire would clear it).
        ps_after = await get_namespace_data("pipeline_status", workspace=ws)
        assert ps_after.get("scan_deferred_processing") is True
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_scan_without_deferred_flag_does_not_extra_drive():
    """The finally's deferred drive is gated on the flag: a scan that fenced no
    processing request must not add a spurious drive (preserves the
    process_calls==0 contract of the all-already-processed / error paths)."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "scan-nodefer-ws"
        await initialize_pipeline_status(workspace=ws)
        # scan_deferred_processing never set.

        process_calls: list[int] = []
        await dr.run_scanning_process(
            _DeferRag(ws, process_calls), _BoomDocManager(), "track-nodefer"
        )

        assert process_calls == []
    finally:
        finalize_share_data()


class _CancelDocManager:
    """Raises CancelledError to simulate a scan cancelled by server shutdown."""

    def scan_directory_for_new_files(self):
        raise asyncio.CancelledError()


@pytest.mark.offline
async def test_cancelled_scan_skips_deferred_drive():
    """A cancelled scan (server shutdown) must NOT start a processing run in its
    finally — shutdown is waiting for this task to exit, and one cancel injects
    CancelledError only once, so an unguarded post-release drive would run to
    completion and stall the shutdown. The deferred flag stays set for the next
    scan / trigger instead of being driven or lost."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "scan-cancel-ws"
        await initialize_pipeline_status(workspace=ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        ps["scan_deferred_processing"] = True

        process_calls: list[int] = []
        with pytest.raises(asyncio.CancelledError):
            await dr.run_scanning_process(
                _DeferRag(ws, process_calls), _CancelDocManager(), "track-cancel"
            )

        # No processing kicked off during shutdown...
        assert process_calls == []
        # ...and the deferred request is preserved for the next scan / trigger.
        ps_after = await get_namespace_data("pipeline_status", workspace=ws)
        assert ps_after.get("scan_deferred_processing") is True
    finally:
        finalize_share_data()
