"""Submissions to the chunking executor stay bounded (GHSA-26pm-px5v-8c4w).

The pipeline's own ``max_parallel_insert`` is not a ceiling for this pool: the
public SDK entry points that also submit here — ``ainsert_custom_kg`` and
``ainsert_custom_chunks`` — are gated by neither it nor the pipeline busy flag.
So the pool carries its own semaphore, and the permit belongs to the executor
future rather than to the awaiting coroutine: the thread pool cannot cancel a
running task, so releasing on cancellation would let submit-and-cancel rebuild an
unbounded queue.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from lightrag import utils as lr_utils
from lightrag.constants import CHUNKING_SUBMIT_LIMIT

pytestmark = pytest.mark.offline


class _CountingProxy:
    """Wraps the real executor and records accepted submissions."""

    def __init__(self, real):
        self._real = real
        self.submitted = 0

    def submit(self, fn, /, *args, **kwargs):
        self.submitted += 1
        return self._real.submit(fn, *args, **kwargs)


@pytest.fixture
def counting_executor(monkeypatch):
    proxy = _CountingProxy(lr_utils.get_chunking_executor())
    monkeypatch.setattr(lr_utils, "get_chunking_executor", lambda: proxy)
    return proxy


def _blocking(release: threading.Event, marker: str) -> str:
    release.wait(5.0)
    return marker


def test_the_pool_has_exactly_one_worker():
    """Preserves the concurrency the event loop already imposed on chunking."""
    assert lr_utils.get_chunking_executor()._max_workers == 1


def test_cancelling_a_submission_does_not_free_its_slot_early(counting_executor):
    """The thread keeps running, so the permit must keep being held.

    ``async with sem: await run_in_executor(...)`` fails this: the permit comes
    back on cancellation while the work does not stop, so a caller can hold an
    unbounded number of live tasks against one permit's worth of accounting.
    """

    async def _main():
        release = threading.Event()
        try:
            tasks = [
                asyncio.create_task(
                    lr_utils.run_in_chunking_executor(_blocking, release, f"m{i}")
                )
                for i in range(CHUNKING_SUBMIT_LIMIT)
            ]
            await asyncio.sleep(0.05)
            assert counting_executor.submitted == CHUNKING_SUBMIT_LIMIT

            for task in tasks:
                task.cancel()
            for task in tasks:
                with pytest.raises(asyncio.CancelledError):
                    await task

            follow_up = asyncio.create_task(
                lr_utils.run_in_chunking_executor(lambda: "later")
            )
            await asyncio.sleep(0.1)

            # Every permit is still consumed by a thread that is still running.
            assert counting_executor.submitted == CHUNKING_SUBMIT_LIMIT
            assert not follow_up.done()

            release.set()
            assert await asyncio.wait_for(follow_up, timeout=5.0) == "later"
            assert counting_executor.submitted == CHUNKING_SUBMIT_LIMIT + 1
        finally:
            release.set()

    asyncio.run(_main())


def test_saturation_waits_rather_than_refusing(counting_executor):
    async def _main():
        release = threading.Event()
        try:
            tasks = [
                asyncio.create_task(
                    lr_utils.run_in_chunking_executor(_blocking, release, f"m{i}")
                )
                for i in range(CHUNKING_SUBMIT_LIMIT + 3)
            ]
            await asyncio.sleep(0.05)
            assert counting_executor.submitted == CHUNKING_SUBMIT_LIMIT

            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)
            # Nobody was dropped; the excess simply waited at the submit point.
            assert len(results) == CHUNKING_SUBMIT_LIMIT + 3
        finally:
            release.set()

    asyncio.run(_main())


def test_kwargs_reach_the_callable(counting_executor):
    async def _main():
        result = await lr_utils.run_in_chunking_executor(lambda a, b=0: a + b, 1, b=41)
        assert result == 42

    asyncio.run(_main())


def test_an_exception_propagates_to_the_caller(counting_executor):
    """Chunker failures must keep reaching the pipeline's error handling."""

    def _boom():
        raise RuntimeError("chunker exploded")

    async def _main():
        with pytest.raises(RuntimeError, match="chunker exploded"):
            await lr_utils.run_in_chunking_executor(_boom)

    asyncio.run(_main())
