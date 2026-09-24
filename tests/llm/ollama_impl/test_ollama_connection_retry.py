"""Offline tests for retrying a transient Ollama connection failure.

The installed ``ollama`` package catches ``httpx.ConnectError`` internally and
re-raises it as the builtin ``ConnectionError`` (see
``ollama._client.CONNECTION_ERROR_MESSAGE``, "Failed to connect to Ollama...").
The retry predicate on ``_ollama_model_if_cache`` must match that real
exception type, not only ``lightrag.exceptions`` types the ``ollama`` package
never raises.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from tenacity import RetryError, wait_none

from lightrag.llm.ollama import _ollama_model_if_cache

pytestmark = pytest.mark.offline

_CONNECTION_ERROR_MESSAGE = (
    "Failed to connect to Ollama. Please check that Ollama is downloaded, "
    "running and accessible. https://ollama.com/download"
)


def _make_flaky_client(*, failures: int):
    """A fake AsyncClient whose chat() raises ConnectionError `failures` times
    then succeeds, mirroring ollama.AsyncClient on a startup-race connect
    failure followed by a server that comes up before the caller gives up."""
    calls = {"count": 0}

    async def chat(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= failures:
            raise ConnectionError(_CONNECTION_ERROR_MESSAGE)
        return {"message": {"content": "ok"}, "done_reason": "stop"}

    return SimpleNamespace(
        chat=chat, _client=SimpleNamespace(aclose=AsyncMock())
    ), calls


async def test_transient_connection_error_is_retried():
    # Keep the test fast regardless of outcome; only call count is asserted.
    _ollama_model_if_cache.retry.wait = wait_none()

    fake_client, calls = _make_flaky_client(failures=1)
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(model="test-model", prompt="hello")

    assert result == "ok"
    assert calls["count"] == 2, "a transient connect failure should be retried"


async def test_connection_error_gives_up_after_stop_after_attempt():
    _ollama_model_if_cache.retry.wait = wait_none()

    fake_client, calls = _make_flaky_client(failures=10)
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises((ConnectionError, RetryError)):
            await _ollama_model_if_cache(model="test-model", prompt="hello")

    assert calls["count"] == 3, "a persistent connect failure stops at attempt 3"
