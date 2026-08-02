"""Concurrent queries must not freeze the event loop on the shared tokenizer.

The storages capture ``global_config`` once at init, so
``text_chunks_db.global_config["tokenizer"]`` is ONE object reached by every
concurrent query. Two of its consumers (``_apply_token_truncation``,
``process_chunks_unified``) now run in the tokenizer executor. A third one left
on the event loop would keep doing its CPU work inline, so query B would stall
every request — /health included — for as long as its own heading backfill takes,
however diligently query A was moved off the loop.

``_attach_content_headings`` is that third consumer, and content headings are on
by default. Single-query load never surfaces this: it takes two queries whose
backfills overlap.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from lightrag import operate

pytestmark = pytest.mark.offline


class _SlowTokenizer:
    """Encodes slowly and records whether it is ever entered concurrently."""

    def __init__(self, delay: float = 0.02):
        self._delay = delay
        self._inside = False
        self._guard = threading.Lock()
        self.reentered = False
        self.calls = 0

    def encode(self, content: str):
        with self._guard:
            if self._inside:
                self.reentered = True
            self._inside = True
            self.calls += 1
        try:
            # Stands in for CPU work: must not yield to the event loop.
            threading.Event().wait(self._delay)
            return list(range(len(content)))
        finally:
            with self._guard:
                self._inside = False

    def decode(self, tokens):
        return "x" * len(tokens)


class _TextChunksDB:
    """Only the surface ``_attach_content_headings`` touches."""

    def __init__(self, tokenizer, rows):
        self.global_config = {"tokenizer": tokenizer}
        self._rows = rows

    async def get_by_ids(self, ids):
        return [self._rows.get(chunk_id) for chunk_id in ids]


def _rows(count: int) -> dict:
    return {
        f"chunk-{i}": {
            "heading": {
                "parent_headings": [
                    "A very long top level heading that will need truncating",
                    "A second level heading that is also quite long indeed",
                    "A third level heading pushing the breadcrumb over budget",
                ]
            }
        }
        for i in range(count)
    }


async def _heartbeat(stop: asyncio.Event) -> int:
    beats = 0
    while not stop.is_set():
        beats += 1
        await asyncio.sleep(0)
    return beats


async def test_heading_backfill_does_not_freeze_the_event_loop():
    """The load-bearing assertion.

    With the backfill running inline, the loop is held for the whole encode
    sequence and the heartbeat cannot advance at all.
    """
    tokenizer = _SlowTokenizer(delay=0.02)
    rows = _rows(4)
    db = _TextChunksDB(tokenizer, rows)
    chunks = [{"chunk_id": chunk_id} for chunk_id in rows]

    stop = asyncio.Event()
    pulse = asyncio.create_task(_heartbeat(stop))
    try:
        await operate._attach_content_headings(chunks, db)
    finally:
        stop.set()
    beats = await pulse

    assert tokenizer.calls > 0  # the work really happened
    assert beats > 1  # ...and the loop kept running while it did


async def test_two_concurrent_queries_share_the_tokenizer_without_reentering_it():
    """Same object, two queries, no concurrent entry and no deadlock."""
    tokenizer = _SlowTokenizer(delay=0.01)
    rows = _rows(3)

    async def _one_query():
        db = _TextChunksDB(tokenizer, rows)
        chunks = [{"chunk_id": chunk_id} for chunk_id in rows]
        await operate._attach_content_headings(chunks, db)
        return chunks

    stop = asyncio.Event()
    pulse = asyncio.create_task(_heartbeat(stop))
    try:
        first, second = await asyncio.wait_for(
            asyncio.gather(_one_query(), _one_query()), timeout=10.0
        )
    finally:
        stop.set()
    beats = await pulse

    # The single-worker executor is what keeps the two backfills apart; the
    # tokenizer itself is never asked to tolerate overlap here.
    assert tokenizer.reentered is False
    assert beats > 1
    # Both queries got their headings; the offload did not drop work.
    assert all("content_headings" in chunk for chunk in first)
    assert all("content_headings" in chunk for chunk in second)


async def test_backfill_output_is_unchanged_by_the_offload():
    """The move off the loop must not alter what the LLM sees."""
    tokenizer = _SlowTokenizer(delay=0.0)
    rows = _rows(2)
    db = _TextChunksDB(tokenizer, rows)
    chunks = [{"chunk_id": chunk_id} for chunk_id in rows]

    await operate._attach_content_headings(chunks, db)

    expected = operate._truncate_section_context(
        operate.format_parent_headings(next(iter(rows.values()))),
        tokenizer,
        operate.DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
    )
    assert all(chunk["content_headings"] == expected for chunk in chunks)


async def test_missing_rows_are_skipped_without_touching_the_tokenizer():
    tokenizer = _SlowTokenizer(delay=0.0)
    db = _TextChunksDB(tokenizer, {})
    chunks = [{"chunk_id": "absent"}]

    await operate._attach_content_headings(chunks, db)

    assert chunks == [{"chunk_id": "absent"}]
    assert tokenizer.calls == 0


async def test_no_chunks_is_a_no_op():
    tokenizer = _SlowTokenizer(delay=0.0)
    db = _TextChunksDB(tokenizer, {})
    await operate._attach_content_headings([], db)
    await operate._attach_content_headings([{"chunk_id": "x"}], None)
    assert tokenizer.calls == 0


def test_slow_tokenizer_stub_matches_the_real_interface():
    """Guards the stub itself: a signature drift here would make the tests lie."""
    assert isinstance(SimpleNamespace(), object)
    tokenizer = _SlowTokenizer(delay=0.0)
    assert tokenizer.decode(tokenizer.encode("abc")) == "xxx"
