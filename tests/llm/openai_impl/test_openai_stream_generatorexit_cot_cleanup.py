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


class _FailingOpenAIStream(_FakeOpenAIStream):
    """Fake stream that raises after yielding its chunks, like an upstream 5xx."""

    def __init__(self, chunks, error):
        super().__init__(chunks)
        self._error = error

    async def __anext__(self):
        if not self._chunks:
            raise self._error
        return self._chunks.pop(0)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stream_closed_on_disconnect_during_error_path_cot_close():
    """A disconnect at the ERROR path's COT close must not abort cleanup.

    ``inner()`` closes an open ``<think>`` tag from three places. Only the two
    inside the outer ``try`` were guarded: a ``GeneratorExit`` raised there is
    caught by ``except GeneratorExit`` and tells the ``finally`` to skip its
    own yield. The third close lives in the ``except Exception`` handler, and
    ``GeneratorExit`` is a ``BaseException`` -- neither the handler's own
    ``except Exception as close_error`` nor the co-sibling ``except
    GeneratorExit`` can catch it. Left unguarded, the ``finally`` then yields
    while the generator is closing, ``aclose()`` raises "async generator
    ignored GeneratorExit", and the frame is abandoned before
    ``response.aclose()`` and ``openai_async_client.close()`` -- leaking both.

    Named as out of scope by #3635, which fixed the consumer-disconnect
    variant.
    """
    stream = _FailingOpenAIStream(
        [_make_chunk(reasoning_content="thinking...")],
        RuntimeError("upstream 500 mid-reasoning"),
    )
    fake_client = _make_stream_client(stream)

    yielded = []
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
        async for piece in gen:
            yielded.append(piece)
            if piece == "</think>":
                break  # consumer goes away right after the error path closes COT
        await gen.aclose()

    assert yielded == ["<think>", "thinking...", "</think>"]
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()
