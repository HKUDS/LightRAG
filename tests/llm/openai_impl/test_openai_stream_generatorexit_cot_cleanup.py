"""Regression test: closing the streaming generator mid-COT must not abort
cleanup.

``inner()`` in ``openai_complete_if_cache``'s streaming branch has a
finally-block safety net that closes an still-open ``<think>`` tag with
``yield "</think>"``. When the consumer disconnects while COT is open
(``gen.aclose()``, e.g. an HTTP client going away mid reasoning-phase),
Python throws ``GeneratorExit`` at the generator's suspension point; yielding
again in response is illegal and makes ``aclose()`` raise "async generator
ignored GeneratorExit", aborting the finally block before the
``response.aclose()`` / ``openai_async_client.close()`` calls that follow it
run. Mirrors ``tests/llm/anthropic_impl/test_anthropic_client_cleanup.py``'s
``test_stream_closed_on_early_consumer_break``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.openai import openai_complete_if_cache


class _FakeOpenAIStream:
    """Async-iterable fake of an OpenAI streaming response.

    ``__aiter__``/``__anext__`` live on the class, matching the real
    ``openai.AsyncStream`` shape that ``async for`` requires.
    """

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.aclose = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _make_chunk(content=None, reasoning_content=""):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content, reasoning_content=reasoning_content
                )
            )
        ],
        usage=None,
    )


def _make_stream_client(stream: _FakeOpenAIStream) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=stream))
        ),
        close=AsyncMock(),
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stream_closed_on_early_close_while_cot_open():
    """Client disconnect mid-COT still closes the stream and the OpenAI client."""
    stream = _FakeOpenAIStream(
        [
            _make_chunk(reasoning_content="Let me think"),
            _make_chunk(reasoning_content=" about this."),
            _make_chunk(reasoning_content=" still reasoning..."),
        ]
    )
    fake_client = _make_stream_client(stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache.__wrapped__(
            model="deepseek-reasoner",
            prompt="hello",
            enable_cot=True,
            stream=True,
            api_key="test-key",
        )
        first = await gen.__anext__()  # opens the COT tag
        second = await gen.__anext__()
        await gen.aclose()  # consumer disconnects mid reasoning phase

    assert first == "<think>"
    assert second == "Let me think"
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stream_closed_after_full_consumption_with_cot():
    """Draining a COT stream to completion still closes stream and client."""
    stream = _FakeOpenAIStream(
        [
            _make_chunk(reasoning_content="thinking"),
            _make_chunk(content="answer"),
        ]
    )
    fake_client = _make_stream_client(stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache.__wrapped__(
            model="deepseek-reasoner",
            prompt="hello",
            enable_cot=True,
            stream=True,
            api_key="test-key",
        )
        chunks = [chunk async for chunk in gen]

    assert chunks == ["<think>", "thinking", "</think>", "answer"]
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()
