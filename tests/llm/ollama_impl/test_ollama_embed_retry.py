"""Regression tests for retrying a transient Ollama embedding failure.

Ollama's server can itself crash mid-request -- its llama-server runner
subprocess dying under memory pressure from a large embedding batch -- and
reports that as an HTTP 500 whose body is literally the runner's own error
text, e.g. "do embedding request: Post http://127.0.0.1:50830/embedding:
EOF". ``ollama.AsyncClient`` turns that into
``ollama.ResponseError(body, status_code=500)``. Before this fix,
``ollama_embed`` carried no ``@retry`` decorator at all, so that one 500
failed the whole batch immediately with zero retries, unlike the equivalent
completion path.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import ollama
import pytest
from tenacity import RetryError, wait_none

from lightrag.llm.ollama import ollama_embed

pytestmark = pytest.mark.offline

# Matches ollama.ResponseError.__str__ exactly for the status-500 case
# reported against a large bge-m3 batch.
_EOF_ERROR_TEXT = "do embedding request: Post http://127.0.0.1:50830/embedding: EOF"


def _make_flaky_client(*, error: Exception, failures: int):
    """A fake AsyncClient whose embed() raises `error` `failures` times then
    succeeds, mirroring Ollama's runner subprocess crashing on a large batch
    and coming back up before the caller gives up."""
    calls = {"count": 0}

    async def embed(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= failures:
            raise error
        return {"embeddings": [[0.1, 0.2, 0.3]]}

    return SimpleNamespace(
        embed=embed, _client=SimpleNamespace(aclose=AsyncMock())
    ), calls


@pytest.mark.asyncio
async def test_transient_500_response_error_is_retried():
    """RED before the fix: ollama_embed had no @retry at all, so this single
    transient 500 would propagate on the first call. GREEN after: the retry
    predicate matches ollama.ResponseError with status_code >= 500 and the
    second attempt succeeds."""
    ollama_embed.func.retry.wait = wait_none()

    error = ollama.ResponseError(_EOF_ERROR_TEXT, 500)
    fake_client, calls = _make_flaky_client(error=error, failures=1)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))
    assert calls["count"] == 2, "a transient 500 should be retried"


@pytest.mark.asyncio
async def test_persistent_500_response_error_gives_up_after_stop_after_attempt():
    ollama_embed.func.retry.wait = wait_none()

    error = ollama.ResponseError(_EOF_ERROR_TEXT, 500)
    fake_client, calls = _make_flaky_client(error=error, failures=10)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises((ollama.ResponseError, RetryError)):
            await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert calls["count"] == 3, "a persistent 500 stops at attempt 3"


@pytest.mark.asyncio
async def test_connection_error_is_retried():
    """ConnectionError is what ollama.AsyncClient re-raises internally from
    httpx.ConnectError -- a plain connect-time failure, not a 500."""
    ollama_embed.func.retry.wait = wait_none()

    error = ConnectionError(
        "Failed to connect to Ollama. Please check that Ollama is downloaded, "
        "running and accessible. https://ollama.com/download"
    )
    fake_client, calls = _make_flaky_client(error=error, failures=1)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))
    assert calls["count"] == 2, "a transient connect failure should be retried"


@pytest.mark.asyncio
async def test_genuine_4xx_response_error_is_not_retried():
    """A 404 (unknown model) is a caller error, not a transient one, and
    must fail on the first attempt."""
    ollama_embed.func.retry.wait = wait_none()

    error = ollama.ResponseError("model 'bge-m3:latest' not found", 404)
    fake_client, calls = _make_flaky_client(error=error, failures=10)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(ollama.ResponseError):
            await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert calls["count"] == 1, "a genuine 404 must not be retried"
