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
import pytest
from openai import RateLimitError
from tenacity import wait_none

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
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="request-nvidia-key",
    )
    assert os.environ["OPENAI_API_KEY"] == "shared-openai-key"
