"""Offline tests for bounded submission to the CPU thread pools.

``bounded_submit`` exists because moving CPU work off the event loop lets more
requests be in flight at once while a ``ThreadPoolExecutor``'s wait queue stays
unbounded. Two things are easy to get wrong and are pinned here:

* the permit must belong to the executor future, not to the awaiting coroutine —
  otherwise cancelling submissions hands back permits that are still consumed;
* the per-loop semaphore must not outlive its loop, which rules out keying a
  module-level container by the loop.
"""

import asyncio
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from lightrag import utils as lr_utils

pytestmark = pytest.mark.offline


class _CountingExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that records how many submissions it accepted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted = 0

    def submit(self, fn, /, *args, **kwargs):
        self.submitted += 1
        return super().submit(fn, *args, **kwargs)


def _blocking(release: threading.Event, marker: str) -> str:
    release.wait(5.0)
    return marker


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_result_is_returned_and_the_loop_keeps_running():
    async def _main():
        executor = _CountingExecutor(max_workers=1)
        try:
            beats = 0

            async def _heartbeat():
                nonlocal beats
                while True:
                    beats += 1
                    await asyncio.sleep(0.005)

            pulse = asyncio.create_task(_heartbeat())
            semaphore = lr_utils.get_loop_semaphore("test-basic", 4)
            result = await lr_utils.bounded_submit(
                executor, semaphore, lambda: (threading.Event().wait(0.1), "done")[1]
            )
            pulse.cancel()

            assert result == "done"
            # The loop was free for the whole 100 ms of thread work.
            assert beats > 1

        finally:
            executor.shutdown(wait=True)

    asyncio.run(_main())


def test_exception_propagates_to_the_caller():
    async def _main():
        executor = _CountingExecutor(max_workers=1)
        try:
            semaphore = lr_utils.get_loop_semaphore("test-error", 4)

            def _boom():
                raise ValueError("from the thread")

            with pytest.raises(ValueError, match="from the thread"):
                await lr_utils.bounded_submit(executor, semaphore, _boom)
        finally:
            executor.shutdown(wait=True)

    asyncio.run(_main())


def test_kwargs_are_forwarded():
    async def _main():
        executor = _CountingExecutor(max_workers=1)
        try:
            semaphore = lr_utils.get_loop_semaphore("test-kwargs", 4)
            result = await lr_utils.bounded_submit(
                executor, semaphore, lambda a, b=0: a + b, 1, b=41
            )
            assert result == 42
        finally:
            executor.shutdown(wait=True)

    asyncio.run(_main())


# ---------------------------------------------------------------------------
# Permit ownership — the load-bearing property
# ---------------------------------------------------------------------------


def test_cancelling_a_submission_does_not_return_the_permit_early():
    """``async with sem: await run_in_executor(...)`` would fail this.

    The thread pool cannot cancel a running task, so returning the permit when
    the awaiting coroutine is cancelled would let a caller submit-and-cancel in a
    loop, holding an unbounded number of live tasks with one permit's worth of
    accounting.
    """

    async def _main():
        executor = _CountingExecutor(max_workers=1)
        release = threading.Event()
        try:
            semaphore = lr_utils.get_loop_semaphore("test-cancel", 1)

            first = asyncio.create_task(
                lr_utils.bounded_submit(executor, semaphore, _blocking, release, "one")
            )
            # Let the submission actually happen before cancelling.
            while executor.submitted < 1:
                await asyncio.sleep(0.005)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            follow_ups = [
                asyncio.create_task(
                    lr_utils.bounded_submit(executor, semaphore, lambda: "later")
                )
                for _ in range(3)
            ]
            await asyncio.sleep(0.1)

            # The cancelled task's thread is still running and still owns the
            # only permit, so nothing new may have been submitted.
            assert executor.submitted == 1
            assert not any(task.done() for task in follow_ups)

            release.set()
            assert (
                await asyncio.wait_for(asyncio.gather(*follow_ups), timeout=5.0)
                == ["later"] * 3
            )
            assert executor.submitted == 4
        finally:
            release.set()
            executor.shutdown(wait=True)

    asyncio.run(_main())


def test_saturation_is_backpressure_not_refusal():
    async def _main():
        executor = _CountingExecutor(max_workers=1)
        release = threading.Event()
        try:
            semaphore = lr_utils.get_loop_semaphore("test-backpressure", 2)
            tasks = [
                asyncio.create_task(
                    lr_utils.bounded_submit(
                        executor, semaphore, _blocking, release, f"m{i}"
                    )
                )
                for i in range(5)
            ]
            await asyncio.sleep(0.05)
            assert executor.submitted == 2  # ceiling honoured

            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            assert results == [f"m{i}" for i in range(5)]  # nobody was dropped
        finally:
            release.set()
            executor.shutdown(wait=True)

    asyncio.run(_main())


# ---------------------------------------------------------------------------
# Per-loop semaphore lifetime
# ---------------------------------------------------------------------------


def test_same_helper_works_across_successive_event_loops():
    """A module-level singleton semaphore raises 'bound to a different loop'."""

    async def _main():
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            semaphore = lr_utils.get_loop_semaphore("test-crossloop", 2)
            return await lr_utils.bounded_submit(executor, semaphore, lambda: "ok")
        finally:
            executor.shutdown(wait=True)

    assert asyncio.run(_main()) == "ok"
    assert asyncio.run(_main()) == "ok"


def test_a_contended_semaphore_does_not_retain_its_closed_loop():
    """``WeakKeyDictionary[loop] -> Semaphore`` leaks here.

    Contention is mandatory: ``Semaphore.acquire()`` only records ``_loop`` when
    it has to wait, and it is that back-reference from the value to the key that
    makes the weak key immortal. Without contention the leak hides.
    """
    captured: dict[str, object] = {}

    async def _main():
        loop = asyncio.get_running_loop()
        captured["ref"] = weakref.ref(loop)
        executor = ThreadPoolExecutor(max_workers=1)
        release = threading.Event()
        try:
            semaphore = lr_utils.get_loop_semaphore("test-gc", 1)
            tasks = [
                asyncio.create_task(
                    lr_utils.bounded_submit(
                        executor, semaphore, _blocking, release, str(i)
                    )
                )
                for i in range(2)
            ]
            await asyncio.sleep(0.05)
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            assert semaphore._loop is loop  # contention really happened
        finally:
            release.set()
            executor.shutdown(wait=True)

    asyncio.run(_main())
    gc.collect()
    gc.collect()

    assert captured["ref"]() is None


def test_repeated_lookups_in_one_loop_return_the_same_semaphore():
    async def _main():
        first = lr_utils.get_loop_semaphore("test-identity", 3)
        second = lr_utils.get_loop_semaphore("test-identity", 99)
        assert first is second
        # Capacity is only honoured on creation, by design.
        assert second._value == 3

    asyncio.run(_main())


# ---------------------------------------------------------------------------
# Fallback table (loops that reject attribute assignment, e.g. C implementations)
# ---------------------------------------------------------------------------


class _FakeLoop:
    """Minimal stand-in exposing only what the fallback table needs."""

    def __init__(self):
        self._closed = False

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True


@pytest.fixture(autouse=True)
def _clear_fallback_table():
    lr_utils._LOOP_SEMAPHORE_FALLBACK.clear()
    yield
    lr_utils._LOOP_SEMAPHORE_FALLBACK.clear()


def test_fallback_table_reuses_one_semaphore_per_loop_and_name():
    loop = _FakeLoop()
    first = lr_utils._fallback_semaphore(loop, "a", 2)
    assert lr_utils._fallback_semaphore(loop, "a", 2) is first
    assert lr_utils._fallback_semaphore(loop, "b", 2) is not first


def test_fallback_table_sweeps_closed_loops_without_needing_contention():
    """The entry holds the loop itself, so sweeping never depends on ``_loop``.

    The earlier design planned to reach the loop through the semaphore, which
    only works after contention — an uncontended semaphore never learns its loop,
    leaving the entry unsweepable and able to collide with a reused ``id()``.
    """
    loop = _FakeLoop()
    lr_utils._fallback_semaphore(loop, "a", 2)  # never contended
    assert len(lr_utils._LOOP_SEMAPHORE_FALLBACK) == 1

    loop.close()
    survivor = _FakeLoop()
    lr_utils._fallback_semaphore(survivor, "a", 2)

    assert list(lr_utils._LOOP_SEMAPHORE_FALLBACK) == [id(survivor)]


def test_fallback_table_rejects_an_entry_whose_loop_identity_changed():
    """Guards against ``id()`` reuse handing a new loop an old semaphore."""
    loop = _FakeLoop()
    stale = lr_utils._fallback_semaphore(loop, "a", 2)

    impostor = _FakeLoop()
    lr_utils._LOOP_SEMAPHORE_FALLBACK[id(impostor)] = (loop, {"a": stale})

    assert lr_utils._fallback_semaphore(impostor, "a", 2) is not stale
