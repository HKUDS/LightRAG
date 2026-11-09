"""Regression: do not close a caller-supplied httpx client on retry paths.

When openai_client_configs={"http_client": ...} is passed, LightRAG must leave
that object open so tenacity retries can rebuild AsyncOpenAI around it.
See https://github.com/HKUDS/LightRAG/issues/4236.
"""

from __future__ import annotations

import httpx
import pytest
from tenacity import RetryError

from lightrag.llm.openai import openai_complete_if_cache, openai_embed


@pytest.mark.offline
@pytest.mark.asyncio
async def test_complete_does_not_close_caller_http_client_on_retry():
    calls = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        if calls["http"] == 1:
            raise httpx.ConnectError("transient blip", request=request)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

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
        assert client.is_closed is False
        # First attempt fails, second succeeds → at least 2 real requests.
        assert calls["http"] >= 2
    finally:
        if not client.is_closed:
            await client.aclose()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_embed_does_not_close_caller_http_client():
    calls = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        if calls["http"] == 1:
            raise httpx.ConnectError("transient blip", request=request)
        # Minimal embeddings response shape
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        # openai_embed may be wrapped; call the underlying func if present
        embed_fn = getattr(openai_embed, "func", openai_embed)
        # Bypass outer embedding attrs if any by going through the retry layer
        out = await embed_fn(
            ["hello"],
            model="text-embedding-3-small",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            client_configs={"http_client": client, "max_retries": 0},
        )
        assert out is not None
        assert client.is_closed is False
        assert calls["http"] >= 2
    except RetryError:
        # If embed path still closes (should not), surface it
        pytest.fail(
            f"RetryError with closed client; is_closed={client.is_closed}, requests={calls['http']}"
        )
    finally:
        if not client.is_closed:
            await client.aclose()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_stream_exception_propagates_with_caller_http_client():
    """An exception during stream iteration must reach the consumer even when
    the http_client is caller-owned (and therefore must not be closed).
    """
    calls = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        # Simulate a streaming response that yields one chunk then fails
        async def gen():
            yield b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
            raise httpx.ReadError("mid-stream failure", request=request)
        return httpx.Response(200, content=gen())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        stream = await openai_complete_if_cache(
            "gpt-4o-mini",
            "hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs={"http_client": client, "max_retries": 0},
            stream=True,
        )
        chunks = []
        with pytest.raises(Exception, match="Connection error|ReadError|mid-stream"):
            async for chunk in stream:
                chunks.append(chunk)
        assert client.is_closed is False
        assert calls["http"] >= 1
        # At least the first chunk should have been received
        assert any("Hello" in str(c) for c in chunks)
    finally:
        if not client.is_closed:
            await client.aclose()
