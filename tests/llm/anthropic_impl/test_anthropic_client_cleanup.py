"""Regression tests for Anthropic client resource cleanup.

Verifies that the ``AsyncAnthropic`` HTTP client is properly closed in:
  * API error paths (all four except branches)
  * Non-streaming success path
  * Streaming path (stream + client closed in the generator's finally)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.llm.anthropic import anthropic_complete_if_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_response(content_text: str = "hello world") -> SimpleNamespace:
    """A minimal object that looks like an Anthropic Messages response."""

    class _Content:
        text = content_text

    class _Message:
        content = [_Content()]

    return SimpleNamespace(content=[_Content()])


def _make_error_client(error: Exception) -> SimpleNamespace:
    """Fake AsyncAnthropic whose ``messages.create`` raises *error*."""
    return SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=error)),
        close=AsyncMock(),
    )


def _make_success_client(
    content_text: str = "hello world",
) -> SimpleNamespace:
    """Fake AsyncAnthropic whose ``messages.create`` succeeds."""
    return SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=_make_fake_response(content_text))
        ),
        close=AsyncMock(),
    )


class _FakeAnthropicStream:
    """Async-iterable fake of ``anthropic.AsyncStream``.

    ``__aiter__`` must live on the *type* (not an instance attribute) — ``async
    for`` looks it up on the class, so a ``SimpleNamespace`` would raise
    ``TypeError: 'async for' requires an object with __aiter__ method``.
    """

    def __init__(self, events, error: Exception | None = None):
        self._events = list(events)
        self._error = error
        self.close = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._error is not None:
            raise self._error
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _make_text_event(text: str) -> SimpleNamespace:
    """An event shaped like a streaming ``content_block_delta`` with ``.text``."""
    return SimpleNamespace(delta=SimpleNamespace(text=text))


def _make_stream_client(stream: _FakeAnthropicStream) -> SimpleNamespace:
    """Fake AsyncAnthropic whose ``messages.create`` returns *stream*."""
    return SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=stream)),
        close=AsyncMock(),
    )


# ---------------------------------------------------------------------------
# Tests: API error paths — client must be closed before re-raise
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_client_closed_on_rate_limit_error():
    """RateLimitError from the API triggers client.close() before re-raise."""
    from anthropic import RateLimitError

    err = RateLimitError(
        message="rate limited",
        response=MagicMock(),
        body=None,
    )
    fake_client = _make_error_client(err)

    with (
        patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client),
        pytest.raises(RateLimitError),
    ):
        await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key"
        )

    fake_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_client_closed_on_api_connection_error():
    """APIConnectionError triggers client.close() before re-raise."""
    import httpx
    from anthropic import APIConnectionError

    err = APIConnectionError(
        message="connection failed",
        request=httpx.Request("POST", "https://api.anthropic.com"),
    )
    fake_client = _make_error_client(err)

    with (
        patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client),
        pytest.raises(APIConnectionError),
    ):
        await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key"
        )

    fake_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_client_closed_on_api_timeout_error():
    """APITimeoutError triggers client.close() before re-raise."""
    import httpx
    from anthropic import APITimeoutError

    err = APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com"))
    fake_client = _make_error_client(err)

    with (
        patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client),
        pytest.raises(APITimeoutError),
    ):
        await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key"
        )

    fake_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_client_closed_on_generic_exception():
    """Any other exception from the API triggers client.close() before re-raise."""
    err = RuntimeError("something unexpected")
    fake_client = _make_error_client(err)

    with (
        patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client),
        pytest.raises(RuntimeError),
    ):
        await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key"
        )

    fake_client.close.assert_awaited()


# ---------------------------------------------------------------------------
# Tests: non-streaming success — client closed after return
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_client_closed_after_non_streaming_success():
    """Non-streaming response: client.close() is called after returning content."""
    fake_client = _make_success_client("test response")

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client):
        result = await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key"
        )

    assert result == "test response"
    fake_client.close.assert_awaited()


# ---------------------------------------------------------------------------
# Tests: close() error does not swallow the original exception
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_close_error_does_not_swallow_original_exception():
    """If client.close() itself raises, the original API error still propagates."""
    from anthropic import RateLimitError

    err = RateLimitError(
        message="rate limited",
        response=MagicMock(),
        body=None,
    )
    fake_client = _make_error_client(err)
    fake_client.close = AsyncMock(side_effect=RuntimeError("close failed"))

    with (
        patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client),
        pytest.raises(RateLimitError),  # original error, not RuntimeError
    ):
        await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key"
        )


# ---------------------------------------------------------------------------
# Tests: streaming path — stream + client closed in the generator's finally
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stream_closed_after_full_consumption():
    """Draining the stream closes both the stream and the client."""
    stream = _FakeAnthropicStream(
        [_make_text_event("hello "), _make_text_event("world")]
    )
    fake_client = _make_stream_client(stream)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client):
        gen = await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key", stream=True
        )
        chunks = [chunk async for chunk in gen]

    assert chunks == ["hello ", "world"]
    stream.close.assert_awaited()
    fake_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stream_closed_on_early_consumer_break():
    """Closing the generator early (GeneratorExit) still runs the finally cleanup."""
    stream = _FakeAnthropicStream([_make_text_event("a"), _make_text_event("b")])
    fake_client = _make_stream_client(stream)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client):
        gen = await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key", stream=True
        )
        first = await gen.__anext__()  # start iteration, then bail out early
        await gen.aclose()

    assert first == "a"
    stream.close.assert_awaited()
    fake_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stream_closed_on_iteration_error():
    """An error mid-stream propagates but stream + client are still closed."""
    stream = _FakeAnthropicStream([], error=RuntimeError("stream boom"))
    fake_client = _make_stream_client(stream)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client):
        gen = await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key", stream=True
        )
        with pytest.raises(RuntimeError):
            async for _ in gen:
                pass

    stream.close.assert_awaited()
    fake_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stream_close_error_does_not_block_client_close():
    """If stream.close() raises, the client is still closed and data is intact."""
    stream = _FakeAnthropicStream([_make_text_event("x")])
    stream.close = AsyncMock(side_effect=RuntimeError("stream close failed"))
    fake_client = _make_stream_client(stream)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=fake_client):
        gen = await anthropic_complete_if_cache.__wrapped__(
            model="claude-3-opus", prompt="hello", api_key="test-key", stream=True
        )
        chunks = [chunk async for chunk in gen]

    assert chunks == ["x"]
    stream.close.assert_awaited()
    fake_client.close.assert_awaited()
