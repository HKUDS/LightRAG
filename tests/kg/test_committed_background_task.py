"""Two-state commit/ownership startup protocol (Phase 0).

``start_reserved_background_task`` cancels its child on ANY caller
cancellation — even after takeover — which would orphan a just-published
sticky manual request.  ``start_committed_background_task`` keys its cancel
behavior on the commit state instead: before the commit a cancellation tears
everything down with zero side effects; after it the child is never
cancelled and owns both its reservation and the published intent.
"""

import asyncio

import pytest

from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import (
    ManualIntentRefused,
    commit_manual_retry_request,
    finalize_share_data,
    get_pipeline_ingress,
    initialize_share_data,
    start_committed_background_task,
)

pytestmark = pytest.mark.offline


class _Probe:
    def __init__(self):
        self.work_started = asyncio.Event()
        self.work_release = asyncio.Event()
        self.work_completed = False
        self.backstop_calls = 0
        self.published = False

    async def backstop(self):
        self.backstop_calls += 1

    async def work(self):
        self.work_started.set()
        await self.work_release.wait()
        self.work_completed = True


def _commit_ok(probe):
    async def commit(state):
        probe.published = True
        state["committed"] = True
        return None

    return commit


async def test_success_path_runs_work_and_returns_task():
    probe = _Probe()
    tasks: set = set()
    task = await start_committed_background_task(
        tasks,
        commit=_commit_ok(probe),
        work=probe.work,
        backstop_release=probe.backstop,
    )
    await probe.work_started.wait()
    probe.work_release.set()
    await task
    assert probe.work_completed
    assert probe.backstop_calls == 0


async def test_refusal_raises_and_backstops_without_side_effects():
    probe = _Probe()

    async def commit(state):
        return "fenced: destructive job running"

    with pytest.raises(ManualIntentRefused, match="fenced"):
        await start_committed_background_task(
            set(), commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    assert probe.backstop_calls == 1
    assert not probe.work_started.is_set()


async def test_commit_crash_backstops_and_raises():
    probe = _Probe()

    async def commit(state):
        raise RuntimeError("commit boom")

    with pytest.raises(RuntimeError, match="failed to start"):
        await start_committed_background_task(
            set(), commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    assert probe.backstop_calls == 1


async def test_caller_cancel_before_commit_cancels_child_and_backstops():
    probe = _Probe()
    commit_entered = asyncio.Event()
    commit_release = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def commit(state):
        commit_entered.set()
        try:
            await commit_release.wait()  # hold NOT_COMMITTED
        except asyncio.CancelledError:
            child_cancelled.set()
            raise
        state["committed"] = True
        return None

    caller = asyncio.create_task(
        start_committed_background_task(
            set(), commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    )
    await commit_entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert child_cancelled.is_set()  # NOT_COMMITTED → child torn down
    assert probe.backstop_calls == 1
    assert not probe.work_started.is_set()


async def test_caller_cancel_after_commit_leaves_child_running():
    probe = _Probe()
    committed = asyncio.Event()
    commit_release = asyncio.Event()

    async def commit(state):
        # Publish, mark committed synchronously, then park at an await —
        # simulating a cancellation landing in the lock's __aexit__.
        probe.published = True
        state["committed"] = True
        committed.set()
        await commit_release.wait()
        return None

    tasks: set = set()
    caller = asyncio.create_task(
        start_committed_background_task(
            tasks, commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    )
    await committed.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # COMMITTED_AND_OWNED: the child was NOT cancelled — release it and it
    # completes its work; no backstop ran (ownership transferred).
    commit_release.set()
    probe.work_release.set()
    child = next(iter(tasks))
    await child
    assert probe.work_completed
    assert probe.backstop_calls == 0


async def test_commit_crash_after_commit_still_backstops_reservation():
    """Committed but never reached work (crash in the lock release): the
    reservation must be reclaimed via the backstop, while the published
    sticky intent stays for the next run — exactly the sticky contract."""
    probe = _Probe()

    async def commit(state):
        probe.published = True
        state["committed"] = True
        raise RuntimeError("lock release boom")

    with pytest.raises(RuntimeError, match="failed to start"):
        await start_committed_background_task(
            set(), commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    assert probe.backstop_calls == 1
    assert probe.published  # intent survives; only the reservation was reclaimed


# ---------------------------------------------------------------------------
# commit_manual_retry_request — fence + publish in one critical section
# ---------------------------------------------------------------------------


@pytest.fixture
def share_data():
    finalize_share_data()
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


def _base_status() -> dict:
    return {
        "busy": False,
        "busy_owner": None,
        "scanning": False,
        "scanning_owner": None,
        "scanning_exclusive": False,
        "destructive_busy": False,
        "pending_enqueues": 0,
        "pending_enqueue_tokens": {},
    }


async def test_commit_manual_retry_publishes_and_marks_committed(share_data):
    ingress = await get_pipeline_ingress("wsC")
    state = {"committed": False}
    refusal = await commit_manual_retry_request(
        _base_status(), asyncio.Lock(), ingress, "req-1", state
    )
    assert refusal is None
    assert state["committed"] is True
    assert ingress.peek_next_manual_retry().request_id == "req-1"


async def test_commit_manual_retry_refuses_on_destructive_without_publish(
    share_data,
):
    ingress = await get_pipeline_ingress("wsC")
    status = _base_status()
    status.update({"busy": True, "destructive_busy": True})
    state = {"committed": False}
    refusal = await commit_manual_retry_request(
        status, asyncio.Lock(), ingress, "req-1", state
    )
    assert refusal is not None and "clear/delete" in refusal
    assert state["committed"] is False
    assert not ingress.has_work()  # refused strictly BEFORE the publish

    # A plain busy pipeline does NOT refuse — the sticky request waits for
    # the running loop's quiescence point.
    status = _base_status()
    status["busy"] = True
    status["busy_owner"] = {"token": "t", "kind": "processing", "pid": 1}
    refusal = await commit_manual_retry_request(
        status, asyncio.Lock(), ingress, "req-2", state
    )
    assert refusal is None
    assert ingress.peek_next_manual_retry().request_id == "req-2"


async def test_commit_manual_retry_refuses_terminal_id_without_commit(share_data):
    """A terminal (already ACKed/cleared) request id must be a refusal, not a
    phantom commit — nothing was published and nothing is owned."""
    ingress = await get_pipeline_ingress("wsC")
    ingress.request_manual_retry(
        "req-done",
        PipelineIngressMessage(kind="rescan", retry_failed=True, request_id="req-done"),
    )
    ingress.ack_manual_retry("req-done")

    state = {"committed": False}
    refusal = await commit_manual_retry_request(
        _base_status(), asyncio.Lock(), ingress, "req-done", state
    )
    assert refusal is not None and "already" in refusal
    assert state["committed"] is False
    assert not ingress.has_work()


async def test_commit_manual_retry_is_idempotent_for_pending_id(share_data):
    ingress = await get_pipeline_ingress("wsC")
    state = {"committed": False}
    msg = PipelineIngressMessage(kind="rescan", retry_failed=True, request_id="req-1")
    ingress.request_manual_retry("req-1", msg)
    refusal = await commit_manual_retry_request(
        _base_status(), asyncio.Lock(), ingress, "req-1", state
    )
    assert refusal is None
    assert len(ingress.snapshot_manual_retries()) == 1
