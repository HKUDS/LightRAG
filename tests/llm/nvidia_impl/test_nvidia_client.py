"""nvidia_openai_embed's per-call AsyncOpenAI client and its @retry wrapper.

The client is held in an ``async with`` so its httpx pool is released on
every exit path. The success path is pinned by
``test_nvidia_embedding.py`` (``client.is_closed()`` on a real SDK
client); the error path is pinned here, on every retry attempt.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai._base_client as openai_base_client
import pytest
from openai import (
    APIStatusError,
    AsyncOpenAI,
    ConflictError,
    InternalServerError,
    RateLimitError,
)
from tenacity import wait_none

from lightrag.llm import nvidia_openai
from lightrag.llm.nvidia_openai import nvidia_openai_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _FakeAsyncOpenAI:
    """Mimics ``AsyncOpenAI`` as an async context manager that closes on exit."""

    def __init__(self, *, create):
        self.embeddings = SimpleNamespace(create=create)
        self.close = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


async def test_exhausted_retries_reraise_the_sdk_error(monkeypatch):
    """After the last attempt the caller gets the SDK's own RateLimitError,
    not tenacity.RetryError, and every attempt has closed its client."""
    monkeypatch.setattr(nvidia_openai_embed.func.retry, "wait", wait_none())
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/embeddings")
    error = RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )
    fake = _FakeAsyncOpenAI(create=AsyncMock(side_effect=error))

    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        with pytest.raises(RateLimitError) as caught:
            await nvidia_openai_embed(texts=["hello"], api_key="k")

    assert caught.value is error
    assert fake.embeddings.create.await_count == 3
    assert fake.close.await_count == 3


@pytest.mark.parametrize(
    "status, error_type",
    [
        (429, RateLimitError),
        (503, InternalServerError),
        (408, APIStatusError),
        (409, ConflictError),
    ],
)
async def test_each_retry_attempt_sends_one_request(monkeypatch, status, error_type):
    """@retry owns retries: the SDK must not retry inside an attempt, and the
    predicate must still retry every status the SDK's own loop retried."""
    monkeypatch.setattr(nvidia_openai_embed.func.retry, "wait", wait_none())
    monkeypatch.setattr(openai_base_client, "INITIAL_RETRY_DELAY", 0.001)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "unavailable"}})

    def create_client(**kwargs):
        transport = httpx.MockTransport(respond)
        return AsyncOpenAI(http_client=httpx.AsyncClient(transport=transport), **kwargs)

    monkeypatch.setattr(nvidia_openai, "AsyncOpenAI", create_client)

    with pytest.raises(error_type):
        await nvidia_openai_embed(texts=["hello"], api_key="k")

    assert len(requests) == 3


async def test_explicit_api_key_is_request_local(monkeypatch):
    """An explicit NVIDIA key must not replace the process-wide OpenAI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "shared-openai-key")
    fake = _FakeAsyncOpenAI(
        create=AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1])])
        )
    )

    with patch(
        "lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake
    ) as client_cls:
        await nvidia_openai_embed.func.__wrapped__(
            texts=["hello"], api_key="request-nvidia-key"
        )

    client_cls.assert_called_once_with(
        max_retries=0,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="request-nvidia-key",
    )
    assert os.environ["OPENAI_API_KEY"] == "shared-openai-key"
