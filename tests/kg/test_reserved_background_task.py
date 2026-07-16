"""PR-1 组件二: unit tests for managed reservation-holding background tasks.

Covers the start-barrier handoff (child takes over vs. teardown paths), the
"child never runs unreserved" guarantee, join-resistance, and the shutdown drain.
"""

import asyncio

import pytest

from lightrag.kg.shared_storage import (
    _join_resistant,
    drain_reserved_background_tasks,
    start_reserved_background_task,
)


async def _noop_backstop():
    return None


@pytest.mark.offline
async def test_start_hands_over_to_child():
    """Happy path: child signals takeover, helper returns the running task and
    the backstop release is NOT used."""
    bg: set = set()
    events = []
    gate = asyncio.Event()

    async def work(started):
        try:
            started.set()
            events.append("set")
            await gate.wait()
        finally:
            events.append("released")

    async def backstop():
        events.append("backstop")

    task = await start_reserved_background_task(
        bg, work=work, backstop_release=backstop
    )
    assert task in bg
    assert events == ["set"]  # took over; backstop not called

    gate.set()
    await task
    assert "released" in events and "backstop" not in events


@pytest.mark.offline
async def test_start_backstop_and_raises_when_child_fails_before_signalling():
    """Child ends before signalling takeover → backstop release + startup error."""
    bg: set = set()
    released = []

    async def work(started):
        raise RuntimeError("boom before started.set()")

    async def backstop():
        released.append("backstop")

    with pytest.raises(RuntimeError, match="failed to start"):
        await start_reserved_background_task(bg, work=work, backstop_release=backstop)
    assert released == ["backstop"]


@pytest.mark.offline
async def test_start_teardown_on_caller_cancel_before_takeover():
    """Caller cancelled before takeover → child is cancelled+joined, backstop
    releases, cancellation propagates, and the child never runs its work."""
    bg: set = set()
    events = []
    gate = asyncio.Event()

    async def work(started):
        # Delay the takeover signal so the pre-takeover cancel window is open.
        try:
            await gate.wait()
            started.set()
            events.append("ran_unreserved_work")
        finally:
            events.append("child_finally")

    async def backstop():
        events.append("backstop")

    helper = asyncio.ensure_future(
        start_reserved_background_task(bg, work=work, backstop_release=backstop)
    )
    await asyncio.sleep(0)  # helper reaches asyncio.wait; child blocked on gate

    helper.cancel()
    for _ in range(3):
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await helper

    assert "ran_unreserved_work" not in events  # never ran unreserved
    assert "backstop" in events  # reservation released via backstop


@pytest.mark.offline
async def test_join_resistant_returns_caller_cancel_without_propagating_task_error():
    # Normal completion → no observed cancel.
    async def ok():
        return 1

    t = asyncio.ensure_future(ok())
    assert await _join_resistant(t) is None

    # Task that raises → not propagated (retrieved), returns None.
    async def boom():
        raise ValueError("x")

    t2 = asyncio.ensure_future(boom())
    assert await _join_resistant(t2) is None


@pytest.mark.offline
async def test_drain_cancels_and_joins_all_tasks():
    bg: set = set()
    released = []

    async def work(started):
        try:
            started.set()
            await asyncio.Event().wait()  # block until cancelled
        finally:
            released.append(1)

    await start_reserved_background_task(bg, work=work, backstop_release=_noop_backstop)
    await start_reserved_background_task(bg, work=work, backstop_release=_noop_backstop)
    assert len(bg) == 2

    pending = await drain_reserved_background_tasks(bg)
    assert pending is None
    assert released == [1, 1]  # every child's finally ran

    await asyncio.sleep(0)  # let done-callbacks run
    assert len(bg) == 0
