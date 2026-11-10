"""Regression tests for retrying a transient Ollama embedding failure.

Ollama's server can itself crash mid-request -- its llama-server runner
subprocess dying under memory pressure from a large embedding batch -- and
reports that as an HTTP 500 whose body is literally the runner's own error
text, e.g. "do embedding request: Post http://127.0.0.1:50830/embedding:
EOF". ``ollama.AsyncClient`` turns that into
``ollama.ResponseError(body, status_code=500)``. Before this fix,
``ollama_embed`` carried no ``@retry`` decorator at all, so that one 500
failed the whole batch immediately with zero retries.

Fix-proof tests fail behaviorally (wrong call count / wrong exception) on the
pre-fix code: the ``no_retry_wait`` fixture tolerates a function without a
retry controller instead of raising AttributeError on it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import ollama
import pytest
from tenacity import wait_none

from lightrag.llm.ollama import ollama_embed

pytestmark = pytest.mark.offline

# Matches ollama.ResponseError.__str__ exactly for the status-500 case
# reported against a large bge-m3 batch.
_EOF_ERROR_TEXT = "do embedding request: Post http://127.0.0.1:50830/embedding: EOF"


@pytest.fixture
def no_retry_wait(monkeypatch):
    """Drop the 4-10s backoff for the duration of one test.

    ``monkeypatch`` restores the decorator's real wait strategy afterwards, so
    the module-global retry controller does not leak into other tests.
    """
    retrying = getattr(ollama_embed.func, "retry", None)
    if retrying is not None:
        monkeypatch.setattr(retrying, "wait", wait_none())


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
async def test_transient_500_response_error_is_retried(no_retry_wait):
    """Fix-proof: before the fix the single 500 propagated on the first call."""
    error = ollama.ResponseError(_EOF_ERROR_TEXT, 500)
    fake_client, calls = _make_flaky_client(error=error, failures=1)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))
    assert calls["count"] == 2, "a transient 500 should be retried"


@pytest.mark.asyncio
async def test_persistent_500_reraises_original_error_after_three_attempts(
    no_retry_wait,
):
    """Fix-proof: three attempts, then the ORIGINAL ResponseError -- not a
    tenacity.RetryError, which would hide the runner's message from the
    document's error_msg."""
    error = ollama.ResponseError(_EOF_ERROR_TEXT, 500)
    fake_client, calls = _make_flaky_client(error=error, failures=10)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(ollama.ResponseError) as exc_info:
            await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert type(exc_info.value) is ollama.ResponseError
    assert "EOF (status code: 500)" in str(exc_info.value)
    assert calls["count"] == 3, "a persistent 500 stops at attempt 3"


@pytest.mark.asyncio
async def test_connection_error_is_retried(no_retry_wait):
    """Fix-proof: ConnectionError is what ollama.AsyncClient re-raises from
    httpx.ConnectError -- a plain connect-time failure, not a 500."""
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
async def test_genuine_4xx_response_error_is_not_retried(no_retry_wait):
    """Stability (passes before and after the fix): a 404 (unknown model) is a
    caller error, not a transient one, and must fail on the first attempt."""
    error = ollama.ResponseError("model 'bge-m3:latest' not found", 404)
    fake_client, calls = _make_flaky_client(error=error, failures=10)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(ollama.ResponseError):
            await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert calls["count"] == 1, "a genuine 404 must not be retried"


_REQUEST = httpx.Request("POST", "http://localhost:11434/api/embed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        # The `ollama serve` process dying mid-request: the connection closes
        # without a complete response. ollama.AsyncClient does not wrap this.
        httpx.RemoteProtocolError(
            "Server disconnected without sending a response.", request=_REQUEST
        ),
        httpx.ReadError("Connection reset by peer", request=_REQUEST),
    ],
    ids=["remote-protocol-error", "read-error"],
)
async def test_mid_request_disconnect_is_retried(no_retry_wait, error):
    """Fix-proof: an established connection dropping mid-request is retried,
    as the OpenAI binding retries it through APIConnectionError."""
    fake_client, calls = _make_flaky_client(error=error, failures=1)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))
    assert calls["count"] == 2, "a mid-request disconnect should be retried"


@pytest.mark.asyncio
async def test_rate_limit_429_is_retried(no_retry_wait):
    """Fix-proof: 429 is Ollama's documented rate-limit status
    (https://docs.ollama.com/api/errors)."""
    error = ollama.ResponseError('{"error": "too many requests"}', 429)
    fake_client, calls = _make_flaky_client(error=error, failures=1)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))
    assert calls["count"] == 2, "a 429 should be retried"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 408, 409])
async def test_other_4xx_is_not_retried(no_retry_wait, status):
    """Stability: only 429 is exempted from fail-fast. 408 / 409 are not
    documented Ollama statuses, so they fail on the first attempt."""
    error = ollama.ResponseError("bad request", status)
    fake_client, calls = _make_flaky_client(error=error, failures=10)

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(ollama.ResponseError):
            await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    assert calls["count"] == 1, f"a {status} must not be retried"
