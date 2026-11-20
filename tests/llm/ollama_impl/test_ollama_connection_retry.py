"""Offline tests for retrying transient Ollama chat failures.

The installed ``ollama`` package catches ``httpx.ConnectError`` internally and
re-raises it as the builtin ``ConnectionError`` (see
``ollama._client.CONNECTION_ERROR_MESSAGE``), lets other httpx transport errors
through unwrapped, and reports server-side failures as ``ollama.ResponseError``.
The retry predicate on ``_ollama_model_if_cache`` must match those real types,
not only ``lightrag.exceptions`` types the ``ollama`` package never raises, and
must use the same transient set as ``ollama_embed``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import ollama
import pytest
from tenacity import wait_none

from lightrag.llm.ollama import _ollama_model_if_cache

pytestmark = pytest.mark.offline

_CONNECTION_ERROR_MESSAGE = (
    "Failed to connect to Ollama. Please check that Ollama is downloaded, "
    "running and accessible. https://ollama.com/download"
)


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    monkeypatch.setattr(_ollama_model_if_cache.retry, "wait", wait_none())


def _make_flaky_client(error: BaseException, *, failures: int):
    """A fake AsyncClient whose chat() raises ``error`` ``failures`` times and
    then succeeds, like ollama.AsyncClient against a server that recovers."""
    calls = {"count": 0}

    async def chat(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= failures:
            raise error
        return {"message": {"content": "ok"}, "done_reason": "stop"}

    client = SimpleNamespace(chat=chat, _client=SimpleNamespace(aclose=AsyncMock()))
    return client, calls


_TRANSIENT_ERRORS = [
    pytest.param(ConnectionError(_CONNECTION_ERROR_MESSAGE), id="connect"),
    pytest.param(httpx.ReadTimeout("timed out"), id="read-timeout"),
    pytest.param(httpx.ReadError("connection reset"), id="network"),
    pytest.param(
        httpx.RemoteProtocolError("server disconnected"), id="remote-protocol"
    ),
    pytest.param(ollama.ResponseError("runner died", status_code=500), id="5xx"),
    pytest.param(ollama.ResponseError("rate limited", status_code=429), id="429"),
]


@pytest.mark.parametrize("error", _TRANSIENT_ERRORS)
async def test_transient_error_is_retried(error):
    fake_client, calls = _make_flaky_client(error, failures=1)
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(model="test-model", prompt="hello")

    assert result == "ok"
    assert calls["count"] == 2


async def test_persistent_connection_error_reraises_after_three_attempts():
    fake_client, calls = _make_flaky_client(
        ConnectionError(_CONNECTION_ERROR_MESSAGE), failures=10
    )
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(ConnectionError, match="Failed to connect to Ollama"):
            await _ollama_model_if_cache(model="test-model", prompt="hello")

    assert calls["count"] == 3


@pytest.mark.parametrize("status_code", [400, 404])
async def test_client_error_response_is_not_retried(status_code):
    fake_client, calls = _make_flaky_client(
        ollama.ResponseError("bad request", status_code=status_code), failures=10
    )
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(ollama.ResponseError) as excinfo:
            await _ollama_model_if_cache(model="test-model", prompt="hello")

    assert excinfo.value.status_code == status_code
    assert calls["count"] == 1
