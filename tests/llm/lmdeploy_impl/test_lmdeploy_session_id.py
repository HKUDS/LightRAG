"""Concurrent lmdeploy_model_if_cache calls must not share an lmdeploy session.

LightRAG runs LLM calls concurrently (``llm_model_max_async``). lmdeploy's
``Pipeline.generate()`` refuses a request whose ``session_id`` already has one
in flight with ``RequestError(REQUEST_CONFLICT)`` (HTTP 409, not retried), so
a fixed session id fails every call that overlaps another.
"""

import asyncio

import pytest

from lightrag.llm.lmdeploy import lmdeploy_model_if_cache

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


async def test_concurrent_calls_use_distinct_sessions(fake_lmdeploy):
    active_sessions: set[int] = set()
    seen_sessions: list[int] = []
    both_in_flight = asyncio.Event()

    async def fake_generate(*_args, session_id, **_kwargs):
        # Mirrors lmdeploy's preprocess(): a session with an active request
        # is refused with a 409 REQUEST_CONFLICT.
        if session_id in active_sessions:
            raise fake_lmdeploy.RequestError(409)
        active_sessions.add(session_id)
        seen_sessions.append(session_id)
        if len(active_sessions) == 2:
            both_in_flight.set()
        try:
            await asyncio.wait_for(both_in_flight.wait(), timeout=5)
            yield type("Out", (), {"text": f"answer-{session_id}"})()
        finally:
            active_sessions.discard(session_id)

    fake_lmdeploy.generate = fake_generate

    results = await asyncio.gather(
        lmdeploy_model_if_cache("fake-model", "first"),
        lmdeploy_model_if_cache("fake-model", "second"),
    )

    assert len(set(seen_sessions)) == 2
    assert sorted(results) == sorted(f"answer-{sid}" for sid in seen_sessions)
