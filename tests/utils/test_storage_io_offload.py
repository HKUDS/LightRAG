"""Offline tests for the storage-IO offload helpers.

The file backends used to rewrite their whole file synchronously on the event
loop. ``run_in_storage_io`` moves that write to a single-worker pool, which
introduces a window the synchronous code did not have: the caller holds its
namespace lock across the write, so a cancelled caller must NOT return while the
worker is still touching shared state.

Pinned here:

* once the work is submitted, the caller waits for it — through repeated
  cancellation, which a lone ``asyncio.shield`` does not survive;
* while merely waiting for a submission permit, the caller stays cancellable —
  nothing is in flight yet, and refusing to return there would keep the caller's
  lock held for the whole queue;
* splitting the body out of ``bounded_submit`` did not change its forwarding
  contract: it still declares no keyword arguments of its own.
"""

import asyncio
import itertools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from lightrag import utils as lr_utils
from lightrag.utils import (
    _bounded_submit_impl,
    bounded_submit,
    get_storage_io_executor,
    run_in_storage_io,
)

pytestmark = pytest.mark.offline


class _CountingExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that records how many submissions it accepted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submissions = 0

    def submit(self, *args, **kwargs):
        self.submissions += 1
        return super().submit(*args, **kwargs)


class _BlockingWork:
    """A callable that parks in the worker thread until released."""

    def __init__(self, result="value", exc=None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished_at = None
        self._result = result
        self._exc = exc

    def __call__(self):
        self.started.set()
        assert self.release.wait(timeout=5), "work was never released"
        self.finished_at = next(_ORDER)
        if self._exc is not None:
            raise self._exc
        return self._result


_ORDER = itertools.count()


async def _wait_for(predicate, *, timeout=2.0):
    """Await a thread-set condition without blocking the loop."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for condition"
        await asyncio.sleep(0.01)


async def test_cancelled_caller_waits_for_the_submitted_worker():
    """A cancelled caller must not return while its worker is still running.

    Fix-proof: with the pre-change semantics (``wait_for_completion=False``,
    i.e. a bare ``bounded_submit``) the task completes as soon as it is
    cancelled, so ``task.done()`` is True before the worker is released and the
    ordering assertion at the end sees the caller return first.
    """
    work = _BlockingWork()
    task = asyncio.create_task(run_in_storage_io(work))

    await _wait_for(work.started.is_set)
    task.cancel()
    # Give the cancellation every chance to land: if the caller were allowed to
    # return early it would already be done here.
    for _ in range(10):
        await asyncio.sleep(0)
    assert not task.done(), "caller returned while the worker was still running"

    work.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    returned_at = next(_ORDER)

    assert work.finished_at is not None
    assert work.finished_at < returned_at


async def test_repeated_cancellation_does_not_release_the_caller(monkeypatch):
    """Three cancellations, each proven to land in a distinct shield round.

    A lone ``await asyncio.shield(fut)`` returns immediately when the awaiting
    coroutine is cancelled again, so an implementation that catches
    ``CancelledError`` once and shields once would escape on the second cancel.
    Sleeping between cancels cannot prove they landed in different rounds — the
    scheduler is free to collapse them — so each cancel here waits for the spy
    to observe the NEXT entry into ``asyncio.shield``.
    """
    entries = itertools.count(1)
    seen = {}
    real_shield = asyncio.shield

    def spy_shield(awaitable, **kwargs):
        seen.setdefault(next(entries), asyncio.Event()).set()
        return real_shield(awaitable, **kwargs)

    def entered(n):
        return seen.setdefault(n, asyncio.Event()).is_set()

    monkeypatch.setattr(asyncio, "shield", spy_shield)

    work = _BlockingWork()
    task = asyncio.create_task(run_in_storage_io(work))

    await _wait_for(work.started.is_set)
    for round_number in range(1, 4):
        # Round N's shield is already active; cancel it and require the helper
        # to re-enter shield for round N+1 before cancelling again.
        await _wait_for(lambda n=round_number: entered(n))
        task.cancel()
        await _wait_for(lambda n=round_number: entered(n + 1))
        assert not task.done(), "caller escaped after a repeated cancellation"

    work.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert work.finished_at is not None


async def test_worker_exception_propagates_when_not_cancelled():
    boom = OSError("disk full")
    work = _BlockingWork(exc=boom)
    task = asyncio.create_task(run_in_storage_io(work))

    await _wait_for(work.started.is_set)
    work.release.set()

    with pytest.raises(OSError, match="disk full"):
        await task


async def test_cancellation_wins_over_a_worker_exception(caplog, monkeypatch):
    """Cancel + IO failure: CancelledError wins, the failure is still logged.

    The worker's exception can never reach the caller once the caller is being
    cancelled, so it must not vanish silently, and it must not resurface as an
    "exception was never retrieved" callback either.

    The cancellation must be observed by the helper BEFORE the worker fails,
    otherwise this races: a worker that finishes first ends the wait loop with
    no deferred cancellation and the caller legitimately sees the OSError. The
    shield spy below pins the ordering instead of hoping for it.
    """
    handled = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: handled.append(context)
    )

    entries = itertools.count(1)
    seen = {}
    real_shield = asyncio.shield

    def spy_shield(awaitable, **kwargs):
        seen.setdefault(next(entries), asyncio.Event()).set()
        return real_shield(awaitable, **kwargs)

    monkeypatch.setattr(asyncio, "shield", spy_shield)

    work = _BlockingWork(exc=OSError("disk full"))
    task = asyncio.create_task(run_in_storage_io(work))

    await _wait_for(work.started.is_set)
    await _wait_for(lambda: seen.setdefault(1, asyncio.Event()).is_set())
    task.cancel()
    # Re-entering shield proves the cancellation was caught and deferred.
    await _wait_for(lambda: seen.setdefault(2, asyncio.Event()).is_set())
    work.release.set()

    # lightrag's logger does not propagate, so caplog needs it turned on.
    lr_logger = logging.getLogger("lightrag")
    previous = lr_logger.propagate
    lr_logger.propagate = True
    try:
        with caplog.at_level(logging.ERROR, logger="lightrag"):
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        lr_logger.propagate = previous

    assert any("disk full" in record.getMessage() for record in caplog.records)

    # Force the finalizer that would report an unretrieved exception.
    del work, task
    for _ in range(3):
        await asyncio.sleep(0)
    assert not any(
        "never retrieved" in str(context.get("message", "")) for context in handled
    )


async def test_cancellation_while_waiting_for_a_permit_never_submits():
    """Waiting for a permit stays cancellable, and cancels before submitting.

    Nothing is in flight while a caller waits for a permit, so refusing to
    return there would keep its storage lock held for the whole queue ahead of
    it — and would still submit the work afterwards, producing a disk write
    after the cancellation.
    """
    executor = _CountingExecutor(max_workers=1)
    semaphore = asyncio.Semaphore(1)
    holder = _BlockingWork()

    try:
        held = asyncio.create_task(
            _bounded_submit_impl(
                executor, semaphore, holder, (), {}, wait_for_completion=True
            )
        )
        await _wait_for(holder.started.is_set)
        assert executor.submissions == 1

        queued = _BlockingWork()
        waiting = asyncio.create_task(
            _bounded_submit_impl(
                executor, semaphore, queued, (), {}, wait_for_completion=True
            )
        )
        await _wait_for(lambda: semaphore.locked())
        for _ in range(10):
            await asyncio.sleep(0)

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert executor.submissions == 1, "cancelled caller still submitted work"
        assert not queued.started.is_set()

        holder.release.set()
        await held
    finally:
        holder.release.set()
        executor.shutdown(wait=True)


async def test_bounded_submit_still_forwards_every_keyword_to_fn():
    """``bounded_submit`` declares no keyword arguments of its own.

    The control flag lives on the private ``_bounded_submit_impl`` precisely so
    that a target function may use the name ``wait_for_completion`` itself. Put
    the flag on ``bounded_submit`` after ``*args`` and this fails: the value is
    swallowed as the control flag instead of reaching ``fn``.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    semaphore = asyncio.Semaphore(1)

    def target(*, wait_for_completion):
        return wait_for_completion

    try:
        result = await bounded_submit(
            executor, semaphore, target, wait_for_completion="payload"
        )
    finally:
        executor.shutdown(wait=True)

    assert result == "payload"


def test_storage_io_executor_is_a_single_worker_named_pool():
    """Configuration pin: the pool's shape is load-bearing, not incidental.

    One worker keeps the commits as serialized as they are today; the name makes
    the thread identifiable in a stack dump taken during a long write.
    """
    executor = get_storage_io_executor()

    assert executor is get_storage_io_executor()
    assert executor._max_workers == 1
    assert executor._thread_name_prefix == "lightrag-storage-io"
    assert lr_utils._STORAGE_IO_EXECUTOR is executor
