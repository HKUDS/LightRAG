"""Async token-counting helpers (GHSA-r8jh-295g-vv42).

Tokenizing is CPU-bound and linear in the input the caller chose, so doing it
inline in an ``async def`` stops the process answering anything for the duration.
These helpers move it to a thread; what has to hold is that the loop stays free,
that the answers are identical to the synchronous path, and that a request
submits O(1) tasks no matter how much work it hands over.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from lightrag import utils as lr_utils

pytestmark = pytest.mark.offline


class _SlowTokenizer:
    def __init__(self, delay: float = 0.0):
        self._delay = delay
        self.calls = 0

    def encode(self, content: str):
        self.calls += 1
        if self._delay:
            threading.Event().wait(self._delay)
        return list(range(len(content)))

    def decode(self, tokens):
        return "x" * len(tokens)


def _wrap(underlying) -> lr_utils.Tokenizer:
    return lr_utils.Tokenizer("test-model", underlying)


async def _heartbeat(stop: asyncio.Event) -> int:
    beats = 0
    while not stop.is_set():
        beats += 1
        await asyncio.sleep(0)
    return beats


async def _count_beats_during(coro):
    stop = asyncio.Event()
    pulse = asyncio.create_task(_heartbeat(stop))
    try:
        result = await coro
    finally:
        stop.set()
    return result, await pulse


# --------------------------------------------------------------------------- #
# The loop stays free
# --------------------------------------------------------------------------- #


async def test_acount_tokens_does_not_block_the_loop():
    tokenizer = _wrap(_SlowTokenizer(delay=0.05))
    count, beats = await _count_beats_during(lr_utils.acount_tokens(tokenizer, "abcd"))
    assert count == 4
    assert beats > 1


async def test_atruncate_does_not_block_the_loop():
    tokenizer = _wrap(_SlowTokenizer(delay=0.01))
    data = [{"text": "abcde"} for _ in range(10)]
    _, beats = await _count_beats_during(
        lr_utils.atruncate_list_by_token_size(
            data,
            key=lambda x: x["text"],
            separator="",
            max_token_size=12,
            tokenizer=tokenizer,
        )
    )
    assert beats > 1


# --------------------------------------------------------------------------- #
# Same answers as the synchronous path
# --------------------------------------------------------------------------- #


async def test_acount_tokens_matches_the_sync_result():
    underlying = _SlowTokenizer()
    tokenizer = _wrap(underlying)
    assert await lr_utils.acount_tokens(tokenizer, "hello") == len(
        tokenizer.encode("hello")
    )


async def test_aencode_matches_the_sync_result():
    tokenizer = _wrap(_SlowTokenizer())
    assert await lr_utils.aencode(tokenizer, "hello") == tokenizer.encode("hello")


@pytest.mark.parametrize("budget", [0, 1, 5, 11, 12, 1000])
async def test_atruncate_matches_the_sync_result(budget):
    tokenizer = _wrap(_SlowTokenizer())
    data = [{"text": "abcde"} for _ in range(6)]

    expected = lr_utils.truncate_list_by_token_size(
        data,
        key=lambda x: x["text"],
        separator="",
        max_token_size=budget,
        tokenizer=tokenizer,
    )
    actual = await lr_utils.atruncate_list_by_token_size(
        data,
        key=lambda x: x["text"],
        separator="",
        max_token_size=budget,
        tokenizer=tokenizer,
    )
    assert actual == expected


# --------------------------------------------------------------------------- #
# Submissions stay O(1) per call
# --------------------------------------------------------------------------- #


async def test_truncating_a_long_list_is_a_single_submission(monkeypatch):
    """The queue must scale with concurrency, not with the size of the work.

    ``atruncate_list_by_token_size`` runs over every retrieved chunk. Submitting
    per element would multiply the executor's queue depth by the list length and
    turn a bounded queue into an amplifier.
    """
    submissions = 0
    real_executor = lr_utils.get_tokenizer_executor()

    class _CountingProxy:
        def submit(self, fn, /, *args, **kwargs):
            nonlocal submissions
            submissions += 1
            return real_executor.submit(fn, *args, **kwargs)

    monkeypatch.setattr(lr_utils, "get_tokenizer_executor", lambda: _CountingProxy())

    tokenizer = _wrap(_SlowTokenizer())
    data = [{"text": "abcde"} for _ in range(1000)]
    await lr_utils.atruncate_list_by_token_size(
        data,
        key=lambda x: x["text"],
        separator="",
        max_token_size=10_000,
        tokenizer=tokenizer,
    )

    assert submissions == 1


async def test_the_executor_has_exactly_one_worker():
    """Preserves today's concurrency rather than introducing parallelism.

    Query-side tokenizing is serialized by the event loop as things stand; the
    move to a thread is meant to free the loop, not to change how much of it runs
    at once.
    """
    assert lr_utils.get_tokenizer_executor()._max_workers == 1


async def test_the_executor_is_a_process_wide_singleton():
    assert lr_utils.get_tokenizer_executor() is lr_utils.get_tokenizer_executor()
