"""PR-1 组件一: unit tests for the cancellation-safe reservation primitives.

Covers acquire_reservation / acquire_enqueue_reservation (atomic single-update
take + reject-with-reason), with_reservation_lock / with_token_set_reservation_lock
(owner-checked, idempotent finalize), and run_to_completion (defers a caller
cancellation until the work finishes; restarts on work-task cancellation).
"""

import asyncio

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    acquire_enqueue_reservation,
    acquire_processing_reservation,
    acquire_reservation,
    check_pipeline_status_mutation,
    has_scan_deferred_processing,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_pipeline_status,
    initialize_share_data,
    release_owned_reservation,
    release_token_set_reservation,
    run_to_completion,
    with_reservation_lock,
    with_token_set_reservation_lock,
)


class _CountingStatus(dict):
    """DictProxy-shaped fake that counts remote-style method calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copy_calls = 0
        self.get_calls = 0
        self.update_calls = 0

    def copy(self):
        self.copy_calls += 1
        return dict.copy(self)

    def get(self, key, default=None):
        self.get_calls += 1
        return super().get(key, default)

    def update(self, *args, **kwargs):
        self.update_calls += 1
        return super().update(*args, **kwargs)


# ---------------------------------------------------------------------------
# acquire_reservation — plain dict + asyncio.Lock (no shared-data needed)
# ---------------------------------------------------------------------------


@pytest.mark.offline
async def test_acquire_reservation_takes_atomically_and_rejects():
    ps = {"busy": False, "scanning": False, "busy_owner": None}
    lock = asyncio.Lock()

    result = await acquire_reservation(
        ps,
        lock,
        owner_key="busy_owner",
        owner="tok1",
        flags={"busy": True},
        reject_when=[("busy", "pipeline busy"), ("scanning", "scanning")],
    )
    assert result.acquired is True and result.message is None
    # flag and owner land together (single atomic update).
    assert ps["busy"] is True and ps["busy_owner"] == "tok1"

    result2 = await acquire_reservation(
        ps,
        lock,
        owner_key="busy_owner",
        owner="tok2",
        flags={"busy": True},
        reject_when=[("busy", "pipeline busy")],
    )
    assert result2.acquired is False and result2.message == "pipeline busy"
    assert ps["busy_owner"] == "tok1"  # untouched


@pytest.mark.offline
async def test_mutation_check_uses_one_snapshot_and_no_update_when_idle(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: False)
    ps = _CountingStatus(
        {
            "busy": False,
            "busy_owner": None,
            "scanning_owner": None,
            "pending_enqueue_tokens": {},
            "pending_enqueues": 0,
        }
    )

    result = await check_pipeline_status_mutation(ps, asyncio.Lock())

    assert result.acquired is True
    assert ps.copy_calls == 1
    assert ps.get_calls == 0
    assert ps.update_calls == 0


@pytest.mark.offline
async def test_enqueue_acquire_combines_recovery_and_reservation_update(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)
    monkeypatch.setattr(shared_storage, "_process_alive", lambda *_: False)
    ps = _CountingStatus(
        {
            "busy": False,
            "busy_owner": None,
            "scanning": False,
            "scanning_exclusive": False,
            "scanning_owner": None,
            "destructive_busy": False,
            "pending_enqueue_tokens": {"dead": {"pid": 999999}},
            "pending_enqueues": 1,
        }
    )

    result = await acquire_enqueue_reservation(
        ps,
        asyncio.Lock(),
        token="new",
        reject_when=(("destructive_busy", "destructive"),),
    )

    assert result.acquired is True
    assert set(ps["pending_enqueue_tokens"]) == {"new"}
    assert ps["pending_enqueues"] == 1
    assert ps.copy_calls == 1
    assert ps.get_calls == 0
    assert ps.update_calls == 1


@pytest.mark.offline
async def test_single_owner_acquire_always_honors_recovery_fence(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)
    monkeypatch.setattr(shared_storage, "_process_alive", lambda *_: False)
    ps = _CountingStatus(
        {
            "busy": True,
            "destructive_busy": True,
            "busy_owner": {
                "token": "dead",
                "pid": 999999,
                "kind": "clear",
            },
            "scanning_owner": None,
            "pending_enqueue_tokens": {},
            "pending_enqueues": 0,
        }
    )

    result = await acquire_reservation(
        ps,
        asyncio.Lock(),
        owner_key="busy_owner",
        owner="new",
        owner_kind="processing",
        flags={"busy": True},
        reject_when=(),
    )

    assert result.acquired is False
    assert (
        result.conflict is shared_storage.PipelineReservationConflict.RECOVERY_REQUIRED
    )
    assert ps["busy"] is False
    assert ps["busy_owner"] is None
    assert ps["recovery_required"]["kind"] == "clear"
    assert ps.copy_calls == 1
    assert ps.get_calls == 0
    assert ps.update_calls == 1


@pytest.mark.offline
async def test_processing_reservation_fences_busy_and_scanning(monkeypatch):
    """acquire_processing_reservation must NOT take the slot while a destructive
    op holds ``busy`` or a scan holds ``scanning_exclusive``: reading/processing
    doc_status then would race their storage rewrites. A handed-off run
    (``already_held``) already owns the slot and is exempt from both fences.
    """
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: False)
    lock = asyncio.Lock()
    flags = {"job_name": "Default Job"}

    # A clear/delete holds ``busy`` → processing is nudged (request_pending); the
    # destructive slot is left untouched.
    busy_ps = {
        "busy": True,
        "scanning_exclusive": False,
        "busy_owner": {"token": "destructive", "pid": 1, "kind": "clear"},
        "history_messages": [],
    }
    busy_res = await acquire_processing_reservation(
        busy_ps, lock, token="proc", already_held=False, flags=flags
    )
    assert busy_res.acquired is False
    assert busy_res.conflict is shared_storage.PipelineReservationConflict.BUSY
    assert busy_ps["busy"] is True
    assert busy_ps["busy_owner"]["token"] == "destructive"
    assert busy_ps["request_pending"] is True

    # A scan classification phase holds ``scanning_exclusive`` → processing is
    # refused without flipping ``busy`` mid-classification.
    scan_ps = {
        "busy": False,
        "scanning_exclusive": True,
        "scanning_owner": {"token": "scan", "pid": 1, "kind": "scan"},
        "history_messages": [],
    }
    scan_res = await acquire_processing_reservation(
        scan_ps, lock, token="proc", already_held=False, flags=flags
    )
    assert scan_res.acquired is False
    assert scan_res.conflict is shared_storage.PipelineReservationConflict.SCANNING
    assert scan_ps["busy"] is False
    assert scan_ps.get("busy_owner") is None
    # The turned-away request is recorded so the scan drives the queue on release.
    assert scan_ps["scan_deferred_processing"] is True

    # A handed-off run already owns the slot: exempt from the scanning fence and
    # takes it over (owner stamped, history cleared).
    handoff_ps = {
        "busy": True,
        "scanning_exclusive": True,
        "busy_owner": {"token": "proc", "pid": 1, "kind": "processing"},
        "history_messages": ["stale"],
    }
    handoff_res = await acquire_processing_reservation(
        handoff_ps, lock, token="proc", already_held=True, flags=flags
    )
    assert handoff_res.acquired is True
    assert handoff_ps["busy"] is True
    assert handoff_ps["busy_owner"]["token"] == "proc"
    assert list(handoff_ps["history_messages"]) == []
    # Taking the slot clears any deferred-processing flag: this run drains it.
    assert handoff_ps["scan_deferred_processing"] is False


@pytest.mark.offline
async def test_has_scan_deferred_processing_is_read_only():
    """The deferred-processing check reports the flag WITHOUT clearing it — the
    clear is owned by acquire_processing_reservation when a run takes the slot, so
    a cancelled or failed post-scan drive keeps the flag for the next scan."""
    lock = asyncio.Lock()
    ps = {"scan_deferred_processing": True}
    assert await has_scan_deferred_processing(ps, lock) is True
    assert ps["scan_deferred_processing"] is True  # NOT cleared by the check
    # Missing / false → False.
    assert await has_scan_deferred_processing({}, lock) is False
    assert (
        await has_scan_deferred_processing({"scan_deferred_processing": False}, lock)
        is False
    )


# ---------------------------------------------------------------------------
# run_to_completion
# ---------------------------------------------------------------------------


@pytest.mark.offline
async def test_run_to_completion_returns_result():
    async def work():
        return 42

    assert await run_to_completion(work) == 42


@pytest.mark.offline
async def test_run_to_completion_defers_caller_cancel_until_work_done():
    gate = asyncio.Event()
    completed = False

    async def work():
        nonlocal completed
        await gate.wait()
        completed = True
        return "done"

    task = asyncio.ensure_future(run_to_completion(work))
    await asyncio.sleep(0)  # let the work task start and block on the gate

    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)

    # Deferred: the work is still running, so run_to_completion has NOT returned.
    assert not completed
    assert not task.done()

    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed  # work ran to completion despite the caller being cancelled


@pytest.mark.offline
async def test_run_to_completion_restarts_when_work_task_cancelled():
    calls = 0

    async def work():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError()  # first attempt "cancelled"
        return "recovered"

    # A work-task cancellation is retried (idempotent release) and does NOT
    # surface as a spurious caller cancellation.
    assert await run_to_completion(work) == "recovered"
    assert calls == 2


# ---------------------------------------------------------------------------
# with_reservation_lock / with_token_set_reservation_lock (real namespace)
# ---------------------------------------------------------------------------


def _release_action(status):
    status.update({"busy": False, "busy_owner": None})
    return "released"


@pytest.mark.offline
async def test_with_reservation_lock_is_owner_checked():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "resv_ws"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)

        result = await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner="tok",
            flags={"busy": True},
            reject_when=[("busy", "busy")],
        )
        assert result.acquired

        # Correct owner → action runs and releases.
        res = await with_reservation_lock(
            ps, lock, owner_key="busy_owner", token="tok", action=_release_action
        )
        assert res == "released"
        assert ps["busy"] is False and ps["busy_owner"] is None

        # New owner takes the slot; a stale task with the WRONG token must no-op.
        await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner="tok2",
            flags={"busy": True},
            reject_when=[("busy", "busy")],
        )
        ran = []

        def bad(status):
            ran.append(1)
            status.update({"busy": False, "busy_owner": None})
            return "oops"

        res2 = await with_reservation_lock(
            ps, lock, owner_key="busy_owner", token="WRONG", action=bad
        )
        assert res2 is None
        assert ran == []  # action never ran
        assert ps["busy"] is True and ps["busy_owner"] == "tok2"  # not clobbered
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_with_reservation_lock_matches_owner_record_token():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "resv_ws2"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)

        owner = {
            "token": "tokX",
            "pid": 1,
            "process_start_id": None,
            "kind": "processing",
        }
        await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner=owner,
            flags={"busy": True},
            reject_when=[("busy", "busy")],
        )
        assert ps["busy_owner"] == owner

        res = await with_reservation_lock(
            ps, lock, owner_key="busy_owner", token="tokX", action=_release_action
        )
        assert res == "released"
        assert ps["busy"] is False and ps["busy_owner"] is None
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_enqueue_token_set_acquire_and_idempotent_release():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "enq_ws"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)

        res1 = await acquire_enqueue_reservation(
            ps, lock, token="e1", reject_when=[("destructive_busy", "d")]
        )
        res2 = await acquire_enqueue_reservation(
            ps, lock, token="e2", reject_when=[("destructive_busy", "d")]
        )
        assert res1.acquired and res2.acquired
        assert ps["pending_enqueues"] == 2
        assert set(ps["pending_enqueue_tokens"]) == {"e1", "e2"}

        # Release e1: count mirrors, e2 survives (concurrent enqueue contract).
        await with_token_set_reservation_lock(
            ps, lock, tokens_key="pending_enqueue_tokens", token="e1"
        )
        assert ps["pending_enqueues"] == 1
        assert set(ps["pending_enqueue_tokens"]) == {"e2"}

        # Releasing e1 again is a no-op (does not double-decrement e2's slot).
        await with_token_set_reservation_lock(
            ps, lock, tokens_key="pending_enqueue_tokens", token="e1"
        )
        assert ps["pending_enqueues"] == 1
        assert set(ps["pending_enqueue_tokens"]) == {"e2"}
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_release_owned_reservation_survives_fetch_cancellation(monkeypatch):
    """release_owned_reservation fetches pipeline_status INSIDE
    run_to_completion, so a caller cancelled while the fetch is in flight still
    releases the slot (the deferred cancel is re-raised only afterwards)."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "rel_owned_cancel_ws"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)
        await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner="tok",
            flags={"busy": True, "destructive_busy": True},
            reject_when=[("busy", "busy")],
        )

        real_get = shared_storage.get_namespace_data
        entered = asyncio.Event()

        async def _slow_get(name, workspace=None):
            entered.set()
            await asyncio.sleep(0.05)
            return await real_get(name, workspace=workspace)

        monkeypatch.setattr(shared_storage, "get_namespace_data", _slow_get)

        def _release(status):
            status.update(
                {"busy": False, "destructive_busy": False, "busy_owner": None}
            )

        task = asyncio.ensure_future(
            release_owned_reservation(
                ws, owner_key="busy_owner", token="tok", action=_release
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert ps["busy"] is False
        assert ps["destructive_busy"] is False
        assert ps["busy_owner"] is None
    finally:
        finalize_share_data()


@pytest.mark.offline
async def test_release_token_set_reservation_survives_fetch_cancellation(monkeypatch):
    """release_token_set_reservation also fetches inside run_to_completion, so a
    cancellation during the fetch still removes the token from the set."""
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "rel_tokset_cancel_ws"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)
        await acquire_enqueue_reservation(
            ps, lock, token="e1", reject_when=[("destructive_busy", "d")]
        )
        assert ps["pending_enqueues"] == 1

        real_get = shared_storage.get_namespace_data
        entered = asyncio.Event()

        async def _slow_get(name, workspace=None):
            entered.set()
            await asyncio.sleep(0.05)
            return await real_get(name, workspace=workspace)

        monkeypatch.setattr(shared_storage, "get_namespace_data", _slow_get)

        task = asyncio.ensure_future(
            release_token_set_reservation(
                ws, tokens_key="pending_enqueue_tokens", token="e1"
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert ps["pending_enqueues"] == 0
        assert dict(ps["pending_enqueue_tokens"]) == {}
    finally:
        finalize_share_data()
