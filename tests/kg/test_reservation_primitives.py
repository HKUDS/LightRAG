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
    acquire_reservation,
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


# ---------------------------------------------------------------------------
# acquire_reservation — plain dict + asyncio.Lock (no shared-data needed)
# ---------------------------------------------------------------------------


@pytest.mark.offline
async def test_acquire_reservation_takes_atomically_and_rejects():
    ps = {"busy": False, "scanning": False, "busy_owner": None}
    lock = asyncio.Lock()

    ok, reason = await acquire_reservation(
        ps,
        lock,
        owner_key="busy_owner",
        owner="tok1",
        flags={"busy": True},
        reject_when=[("busy", "pipeline busy"), ("scanning", "scanning")],
    )
    assert ok is True and reason is None
    # flag and owner land together (single atomic update).
    assert ps["busy"] is True and ps["busy_owner"] == "tok1"

    ok2, reason2 = await acquire_reservation(
        ps,
        lock,
        owner_key="busy_owner",
        owner="tok2",
        flags={"busy": True},
        reject_when=[("busy", "pipeline busy")],
    )
    assert ok2 is False and reason2 == "pipeline busy"
    assert ps["busy_owner"] == "tok1"  # untouched


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

        ok, _ = await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner="tok",
            flags={"busy": True},
            reject_when=[("busy", "busy")],
        )
        assert ok

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

        ok1, _ = await acquire_enqueue_reservation(
            ps, lock, token="e1", reject_when=[("destructive_busy", "d")]
        )
        ok2, _ = await acquire_enqueue_reservation(
            ps, lock, token="e2", reject_when=[("destructive_busy", "d")]
        )
        assert ok1 and ok2
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
