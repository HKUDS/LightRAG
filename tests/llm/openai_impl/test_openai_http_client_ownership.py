"""Caller-supplied httpx clients must outlive a LightRAG OpenAI call.

``openai_complete_if_cache`` and ``openai_embed`` used to call
``AsyncOpenAI.close()`` on every failure and success path. That method is
``await self._client.aclose()``, so a client passed through
``openai_client_configs["http_client"]`` (or embed ``client_configs``) was
closed on the first attempt. tenacity then rebuilt ``AsyncOpenAI`` around the
same closed transport; attempts 2 and 3 never reached the network.

``http_client=None`` is not ownership: the SDK still constructs its own
client. Stream cleanup stays independent of that flag, because openai 2.x
``AsyncStream`` exposes async ``close()`` rather than ``aclose()``.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest
from openai import APIConnectionError
from tenacity import wait_none

from lightrag.llm.openai import (
    _caller_supplied_http_client,
    openai_complete_if_cache,
    openai_embed,
)

pytestmark = pytest.mark.offline


def _chat_success_body(content: str = "OK") -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _embed_success_body() -> dict:
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    import base64

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": base64.b64encode(embedding.tobytes()).decode("ascii"),
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


def _make_error_client(error: Exception) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=error))
        ),
        close=AsyncMock(),
    )


@pytest.fixture
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())
    monkeypatch.setattr(openai_embed.func.retry, "wait", wait_none())


@pytest.mark.parametrize(
    "configs, expected",
    [
        (None, False),
        ({}, False),
        ({"http_client": None}, False),
        ({"http_async_client": object()}, False),
        ({"http_client": object()}, True),
    ],
)
def test_caller_supplied_http_client_requires_a_live_instance(configs, expected):
    assert _caller_supplied_http_client(configs) is expected


@pytest.mark.asyncio
async def test_complete_retries_on_injected_http_client(no_retry_wait):
    """A single ConnectError must not close the caller's client or skip retries."""
    calls = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        if calls["http"] == 1:
            raise httpx.ConnectError("transient blip", request=request)
        return httpx.Response(200, json=_chat_success_body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        out = await openai_complete_if_cache(
            "gpt-4o-mini",
            "hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs={"http_client": client, "max_retries": 0},
        )
        assert out == "OK"
        assert calls["http"] == 2
        assert client.is_closed is False

        out_again = await openai_complete_if_cache(
            "gpt-4o-mini",
            "hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs={"http_client": client, "max_retries": 0},
        )
        assert out_again == "OK"
        assert calls["http"] == 3
        assert client.is_closed is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_embed_retries_on_injected_http_client(no_retry_wait):
    calls = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        if calls["http"] == 1:
            raise httpx.ConnectError("transient blip", request=request)
        return httpx.Response(200, json=_embed_success_body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        first = await openai_embed.func(
            texts=["hi"],
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            client_configs={"http_client": client, "max_retries": 0},
        )
        np.testing.assert_allclose(first, [[0.1, 0.2, 0.3]], rtol=1e-6)
        assert calls["http"] == 2
        assert client.is_closed is False

        second = await openai_embed.func(
            texts=["hi"],
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            client_configs={"http_client": client, "max_retries": 0},
        )
        np.testing.assert_allclose(second, [[0.1, 0.2, 0.3]], rtol=1e-6)
        assert calls["http"] == 3
        assert client.is_closed is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_http_client_none_still_closes_sdk_owned_client():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = APIConnectionError(request=request, message="boom")
    fake = _make_error_client(error)

    with patch("lightrag.llm.openai.create_openai_async_client", return_value=fake):
        with pytest.raises(APIConnectionError):
            await openai_complete_if_cache.__wrapped__(
                model="gpt-4o-mini",
                prompt="hi",
                api_key="sk-test",
                openai_client_configs={"http_client": None},
            )

    fake.close.assert_awaited()


class _TrackingAsyncByteStream(httpx.AsyncByteStream):
    """Yield SSE events lazily so early response closure is observable."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.chunks_yielded = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.chunks_yielded += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_real_sdk_stream_early_stop_closes_response_and_reuses_client():
    """Closing LightRAG's generator must release a real SDK stream response."""
    events = [
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
        }
        for content in ("hello", " world")
    ]
    body = _TrackingAsyncByteStream(
        [f"data: {json.dumps(event)}\n\n".encode() for event in events]
        + [b"data: [DONE]\n\n"]
    )
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            assert json.loads(request.content)["stream"] is True
            return response
        assert json.loads(request.content).get("stream", False) is False
        return httpx.Response(200, json=_chat_success_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        configs = {"http_client": client, "max_retries": 0}
        gen = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            stream=True,
            openai_client_configs=configs,
        )
        try:
            assert await gen.__anext__() == "hello"
            assert body.chunks_yielded == 1
            assert body.closed is False
            assert response.is_closed is False
        finally:
            await gen.aclose()

        assert body.closed is True
        assert response.is_closed is True
        assert body.chunks_yielded == 1
        assert client.is_closed is False

        result = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="hi again",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs=configs,
        )
        assert result == "OK"
        assert len(requests) == 2
        assert client.is_closed is False


class _FailingAfterFirstChunkStream(httpx.AsyncByteStream):
    """Yield one SSE event, then fail the read like a dropped connection."""

    def __init__(self, first_chunk: bytes) -> None:
        self.first_chunk = first_chunk
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first_chunk
        raise httpx.ReadError("mid-stream failure")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_real_sdk_stream_error_after_chunk_propagates_and_keeps_client():
    """A stream that fails after yielding must surface the error to the
    consumer and leave the caller's shared client usable."""
    event = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "A"}, "finish_reason": None}],
    }
    body = _FailingAfterFirstChunkStream(f"data: {json.dumps(event)}\n\n".encode())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=body
            )
        return httpx.Response(200, json=_chat_success_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        configs = {"http_client": client, "max_retries": 0}
        gen = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            stream=True,
            openai_client_configs=configs,
        )
        received: list[str] = []
        with pytest.raises(httpx.ReadError, match="mid-stream failure"):
            async for piece in gen:
                received.append(piece)

        assert received == ["A"]
        assert body.closed is True
        assert client.is_closed is False

        result = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="hi again",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs=configs,
        )
        assert result == "OK"
        assert len(requests) == 2
        assert client.is_closed is False


@pytest.mark.asyncio
async def test_cancelled_request_keeps_shared_client_open_for_concurrent_calls():
    """Cancelling one in-flight request (query timeout, client disconnect)
    must not close a caller-supplied client that other calls share."""
    slow_request_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["messages"][-1]["content"] == "slow":
            slow_request_started.set()
            await asyncio.Event().wait()  # never answers; only cancellation ends it
        return httpx.Response(200, json=_chat_success_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        configs = {"http_client": client, "max_retries": 0}
        slow = asyncio.create_task(
            openai_complete_if_cache(
                model="gpt-4o-mini",
                prompt="slow",
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                openai_client_configs=configs,
            )
        )
        await asyncio.wait_for(slow_request_started.wait(), timeout=5)
        slow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow

        assert client.is_closed is False
        result = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="fast",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs=configs,
        )
        assert result == "OK"
        assert client.is_closed is False


@pytest.mark.asyncio
async def test_cancelled_request_still_closes_sdk_owned_client():
    started = asyncio.Event()

    async def hang(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=hang)),
        close=AsyncMock(),
    )
    with patch("lightrag.llm.openai.create_openai_async_client", return_value=fake):
        task = asyncio.create_task(
            openai_complete_if_cache.__wrapped__(
                model="gpt-4o-mini", prompt="hi", api_key="sk-test"
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    fake.close.assert_awaited()
