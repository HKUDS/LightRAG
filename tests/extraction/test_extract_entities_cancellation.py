"""``extract_entities`` must not leave chunk tasks behind when it exits early.

It fans the chunks out as tasks and waits with ``asyncio.wait``, which never
cancels what it waits on. An external cancellation of ``extract_entities``
(pipeline cancel, server shutdown) therefore used to leave the chunk tasks
running: they kept calling the LLM for a document already abandoned, and when
one failed later nobody was left to retrieve its exception, so asyncio
reported "Task exception was never retrieved" -- one traceback per chunk on
every shutdown that interrupted an extraction.
"""

from __future__ import annotations

import asyncio
import gc

import pytest

from lightrag.operate import extract_entities

from .test_truncation_reporting import (
    _EXTRACTION_RESULT,
    _high_token_limit,  # noqa: F401  (fixture)
    _make_chunks,
    _make_global_config,
)

pytestmark = pytest.mark.offline


@pytest.fixture
async def unretrieved_task_exceptions():
    """Collect what asyncio reports through the loop exception handler."""
    loop = asyncio.get_running_loop()
    reported: list[str] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: reported.append(context.get("message", ""))
    )
    yield reported
    loop.set_exception_handler(previous)


async def _never_retrieved(reported: list[str]) -> list[str]:
    # The report is emitted when the task object is collected, and the loop
    # still references a finished task until its pending callbacks have run.
    for _ in range(3):
        await asyncio.sleep(0)
        gc.collect()
    return [m for m in reported if "never retrieved" in m]


async def test_external_cancellation_cancels_the_running_chunk_tasks(
    _high_token_limit,  # noqa: F811
    unretrieved_task_exceptions,
):
    config = _make_global_config()
    config["llm_model_max_async"] = 4
    entered = 0
    all_entered = asyncio.Event()
    cancelled_calls = 0
    release = asyncio.Event()

    async def _llm(*args, **kwargs):
        nonlocal entered, cancelled_calls
        entered += 1
        if entered == 3:
            all_entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_calls += 1
            raise
        # Reached only by a task that was left running: this is the late
        # failure nobody retrieves (a rejected call during shutdown).
        raise RuntimeError("Queue is shutting down")

    config["role_llm_funcs"]["extract"].side_effect = _llm

    extract_task = asyncio.create_task(
        extract_entities(chunks=_make_chunks(3), global_config=config)
    )
    await asyncio.wait_for(all_entered.wait(), timeout=2.0)
    extract_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await extract_task

    # Every in-flight LLM call was cancelled before extract_entities returned.
    assert cancelled_calls == 3

    # Whatever might still be running would fail now; nothing is.
    release.set()
    await asyncio.sleep(0.05)
    del extract_task
    assert await _never_retrieved(unretrieved_task_exceptions) == []


async def test_sibling_failures_during_fail_fast_are_retrieved(
    _high_token_limit,  # noqa: F811
    unretrieved_task_exceptions,
):
    """One chunk fails, its siblings are cancelled, and a sibling answers the
    cancellation with an exception of its own. That exception must be
    retrieved too, while the first failure is still the one raised."""
    config = _make_global_config()
    config["llm_model_max_async"] = 4
    entered = 0
    siblings_parked = asyncio.Event()
    release_first = asyncio.Event()

    async def _llm(*args, **kwargs):
        nonlocal entered
        entered += 1
        if entered == 1:
            await release_first.wait()
            raise ValueError("first failure")
        if entered == 3:
            siblings_parked.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise RuntimeError("sibling failed while being cancelled") from None
        return _EXTRACTION_RESULT

    config["role_llm_funcs"]["extract"].side_effect = _llm

    extract_task = asyncio.create_task(
        extract_entities(chunks=_make_chunks(3), global_config=config)
    )
    await asyncio.wait_for(siblings_parked.wait(), timeout=2.0)
    release_first.set()
    with pytest.raises(ValueError, match="first failure"):
        await extract_task

    del extract_task
    assert await _never_retrieved(unretrieved_task_exceptions) == []
