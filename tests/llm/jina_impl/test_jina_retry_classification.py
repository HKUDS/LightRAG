"""Regression test: permanent Jina API errors must not be retried."""

from unittest.mock import patch

import aiohttp
import pytest
from tenacity import RetryError, wait_none

from lightrag.llm.jina import jina_embed


def _client_response_error(status: int) -> aiohttp.ClientResponseError:
    request_info = aiohttp.RequestInfo(
        url="https://api.jina.ai/v1/embeddings",
        method="POST",
        headers={},
        real_url="https://api.jina.ai/v1/embeddings",
    )
    return aiohttp.ClientResponseError(
        request_info=request_info,
        history=(),
        status=status,
        message=f"Jina API error: HTTP {status}",
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_permanent_401_is_not_retried(monkeypatch):
    """A 401 (bad API key) must fail on the first attempt, not be retried 3x."""
    monkeypatch.setenv("JINA_API_KEY", "irrelevant")
    # Keep the test fast regardless of outcome; only call count is asserted.
    jina_embed.func.retry.wait = wait_none()

    call_count = 0

    async def fake_fetch_data(url, headers, data):
        nonlocal call_count
        call_count += 1
        raise _client_response_error(401)

    with patch("lightrag.llm.jina.fetch_data", fake_fetch_data):
        # A predicate that stops retrying re-raises the original exception
        # directly; only an exhausted-but-still-retryable loop wraps it in
        # tenacity.RetryError, so accepting either keeps the assertion
        # focused on the call count, which is what the bug actually breaks.
        with pytest.raises((aiohttp.ClientResponseError, RetryError)):
            await jina_embed.func(texts=["hello"], api_key="bad-key", embedding_dim=3)

    assert call_count == 1, "a permanent 401 must not be retried"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_transient_503_is_retried(monkeypatch):
    """A 503 (transient overload) should still be retried up to 3 attempts."""
    monkeypatch.setenv("JINA_API_KEY", "irrelevant")
    jina_embed.func.retry.wait = wait_none()

    call_count = 0

    async def fake_fetch_data(url, headers, data):
        nonlocal call_count
        call_count += 1
        raise _client_response_error(503)

    with patch("lightrag.llm.jina.fetch_data", fake_fetch_data):
        with pytest.raises(RetryError):
            await jina_embed.func(texts=["hello"], api_key="bad-key", embedding_dim=3)

    assert call_count == 3, "a transient 503 should be retried to the stop limit"
