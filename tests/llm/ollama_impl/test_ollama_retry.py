"""Offline tests for retrying transient Ollama chat and embedding failures.

The installed ``ollama`` package catches ``httpx.ConnectError`` internally and
re-raises it as the builtin ``ConnectionError`` (see
``ollama._client.CONNECTION_ERROR_MESSAGE``), lets other httpx transport errors
through unwrapped, and reports server-side failures as ``ollama.ResponseError``
-- including the HTTP 500 Ollama returns when its llama-server runner dies
under memory pressure from a large embedding batch. Both decorators must match
those real types, not only ``lightrag.exceptions`` types the ``ollama`` package
never raises.

Both decorators read the same transport tuple and status predicate, so every
member of them is pinned once, on the chat side. The embed side pins that its
decorator carries the tuple, the predicate and ``reraise=True``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import ollama
import pytest
from tenacity import wait_none

from lightrag.llm.ollama import _ollama_model_if_cache, ollama_embed

pytestmark = pytest.mark.offline

_CONNECTION_ERROR_MESSAGE = (
    "Failed to connect to Ollama. Please check that Ollama is downloaded, "
    "running and accessible. https://ollama.com/download"
)
# ollama.ResponseError.__str__ of the runner crash reported on a bge-m3 batch.
_EOF_ERROR_TEXT = "do embedding request: Post http://127.0.0.1:50830/embedding: EOF"


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    monkeypatch.setattr(_ollama_model_if_cache.retry, "wait", wait_none())
    monkeypatch.setattr(ollama_embed.func.retry, "wait", wait_none())


def _make_flaky_client(error: BaseException, *, failures: int):
    """A fake AsyncClient whose chat() and embed() raise ``error`` ``failures``
    times in total and then succeed, like a server that recovers."""
    calls = {"count": 0}

    async def _attempt(result):
        calls["count"] += 1
        if calls["count"] <= failures:
            raise error
        return result

    async def chat(*args, **kwargs):
        return await _attempt({"message": {"content": "ok"}, "done_reason": "stop"})

    async def embed(*args, **kwargs):
        return await _attempt({"embeddings": [[0.1, 0.2, 0.3]]})

    client = SimpleNamespace(
        chat=chat, embed=embed, _client=SimpleNamespace(aclose=AsyncMock())
    )
    return client, calls


async def _run(entry, fake_client):
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        if entry == "chat":
            return await _ollama_model_if_cache(model="test-model", prompt="hello")
        return await ollama_embed.func(["hello"], embed_model="bge-m3:latest")


def _assert_recovered(entry, result):
    if entry == "chat":
        assert result == "ok"
    else:
        assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        pytest.param(
            "chat", ConnectionError(_CONNECTION_ERROR_MESSAGE), id="chat-connect"
        ),
        pytest.param("chat", httpx.ReadTimeout("timed out"), id="chat-read-timeout"),
        pytest.param("chat", httpx.ReadError("connection reset"), id="chat-network"),
        # The `ollama serve` process dying mid-request: the connection closes
        # without a complete response. ollama.AsyncClient does not wrap this.
        pytest.param(
            "chat",
            httpx.RemoteProtocolError("server disconnected"),
            id="chat-remote-protocol",
        ),
        pytest.param(
            "chat", ollama.ResponseError("runner died", status_code=500), id="chat-5xx"
        ),
        # 429 is Ollama's documented rate-limit status
        # (https://docs.ollama.com/api/errors).
        pytest.param(
            "chat", ollama.ResponseError("rate limited", status_code=429), id="chat-429"
        ),
        pytest.param(
            "embed", ConnectionError(_CONNECTION_ERROR_MESSAGE), id="embed-connect"
        ),
        pytest.param(
            "embed", ollama.ResponseError(_EOF_ERROR_TEXT, 500), id="embed-5xx"
        ),
    ],
)
async def test_transient_error_is_retried(entry, error):
    fake_client, calls = _make_flaky_client(error, failures=1)

    result = await _run(entry, fake_client)

    _assert_recovered(entry, result)
    assert calls["count"] == 2


@pytest.mark.parametrize(
    ("entry", "error", "message"),
    [
        pytest.param(
            "chat",
            ConnectionError(_CONNECTION_ERROR_MESSAGE),
            "Failed to connect to Ollama",
            id="chat-connect",
        ),
        pytest.param(
            "embed",
            ollama.ResponseError(_EOF_ERROR_TEXT, 500),
            "EOF (status code: 500)",
            id="embed-5xx",
        ),
    ],
)
async def test_persistent_error_reraises_original_after_three_attempts(
    entry, error, message
):
    """The original error, not a tenacity.RetryError, which would hide the
    provider's message from the document's error_msg."""
    fake_client, calls = _make_flaky_client(error, failures=10)

    with pytest.raises(type(error)) as excinfo:
        await _run(entry, fake_client)

    assert type(excinfo.value) is type(error)
    assert message in str(excinfo.value)
    assert calls["count"] == 3


@pytest.mark.parametrize(
    ("entry", "status_code"),
    [
        ("chat", 400),
        ("chat", 404),
        # Only 429 is exempted from fail-fast: 408 / 409 are not documented
        # Ollama statuses.
        ("chat", 408),
        ("chat", 409),
        ("embed", 404),
    ],
)
async def test_client_error_response_is_not_retried(entry, status_code):
    fake_client, calls = _make_flaky_client(
        ollama.ResponseError("bad request", status_code=status_code), failures=10
    )
    with pytest.raises(ollama.ResponseError) as excinfo:
        await _run(entry, fake_client)

    assert excinfo.value.status_code == status_code
    assert calls["count"] == 1
