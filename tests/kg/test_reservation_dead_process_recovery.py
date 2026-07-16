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
