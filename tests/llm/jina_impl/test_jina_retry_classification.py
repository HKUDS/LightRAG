"""Regression test: permanent Jina API errors must not be retried."""

from unittest.mock import patch

import aiohttp
import pytest
from tenacity import wait_none

from lightrag.llm.jina import jina_embed

pytestmark = pytest.mark.offline

_URL = "https://api.jina.ai/v1/embeddings"


def _client_response_error(status: int) -> aiohttp.ClientResponseError:
    request_info = aiohttp.RequestInfo(
        url=_URL, method="POST", headers={}, real_url=_URL
    )
    return aiohttp.ClientResponseError(
        request_info=request_info,
        history=(),
        status=status,
        message=f"Jina API error: HTTP {status}",
    )


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Skip the real backoff; only call counts and exception types matter.

    monkeypatch.setattr reverts this automatically, so later tests still see
    the real exponential backoff instead of inheriting wait_none().
    """
    monkeypatch.setenv("JINA_API_KEY", "irrelevant")
    monkeypatch.setattr(jina_embed.func.retry, "wait", wait_none())


async def _call_count_for(exc: BaseException) -> tuple[int, BaseException]:
    """Run jina_embed with fetch_data always raising ``exc``."""
    call_count = 0

    async def fake_fetch_data(url, headers, data):
        nonlocal call_count
        call_count += 1
        raise exc

    with patch("lightrag.llm.jina.fetch_data", fake_fetch_data):
        # reraise=True: the caller sees the provider's own exception, never
        # tenacity.RetryError, whether or not the error was retried.
        with pytest.raises(type(exc)) as excinfo:
            await jina_embed.func(texts=["hello"], api_key="key", embedding_dim=3)
    return call_count, excinfo.value


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
async def test_permanent_status_is_not_retried(status):
    """Bad key, exhausted balance or a malformed request fails on attempt one."""
    calls, raised = await _call_count_for(_client_response_error(status))

    assert calls == 1, f"a permanent {status} must not be retried"
    assert raised.status == status


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
async def test_transient_status_is_retried(status):
    """Timeouts, conflicts, rate limits and 5xx are retried to the stop limit."""
    calls, raised = await _call_count_for(_client_response_error(status))

    assert calls == 3, f"a transient {status} should be retried to the stop limit"
    assert raised.status == status


async def test_invalid_url_is_not_retried():
    """A malformed base_url raises InvalidURL, which no retry can fix."""
    calls, _ = await _call_count_for(aiohttp.InvalidURL("not-a-valid-url"))

    assert calls == 1, "InvalidURL is permanent and must not be retried"


@pytest.mark.parametrize(
    "exc",
    [
        aiohttp.ClientConnectionError("connection reset"),
        aiohttp.ServerDisconnectedError(),
        aiohttp.ClientPayloadError("response body truncated"),
    ],
    ids=["connection-error", "server-disconnected", "payload-error"],
)
async def test_connection_and_payload_errors_are_retried(exc):
    """Dropped connections and truncated bodies are retried to the limit."""
    calls, _ = await _call_count_for(exc)

    assert calls == 3, "a transient transport error should be retried"
