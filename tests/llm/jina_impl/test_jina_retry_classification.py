"""Regression test: jina_embed retries only what the shared policy calls transient."""

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

    async def fake_fetch_data(url, headers, data, timeout=None):
        nonlocal call_count
        call_count += 1
        raise exc

    with patch("lightrag.llm.jina.fetch_data", fake_fetch_data):
        # reraise=True: the caller sees the provider's own exception, never
        # tenacity.RetryError, whether or not the error was retried.
        with pytest.raises(type(exc)) as excinfo:
            await jina_embed.func(texts=["hello"], api_key="key", embedding_dim=3)
    return call_count, excinfo.value


@pytest.mark.parametrize(
    ("exc", "attempts"),
    [
        pytest.param(_client_response_error(401), 1, id="permanent-status"),
        pytest.param(_client_response_error(503), 3, id="transient-status"),
        pytest.param(aiohttp.InvalidURL("not-a-valid-url"), 1, id="invalid-url"),
        pytest.param(aiohttp.ClientConnectionError("reset"), 3, id="connection"),
    ],
)
async def test_retry_follows_the_shared_classification(exc, attempts):
    """One case per branch of ``is_transient_aiohttp_error``.

    The full table is pinned in ``tests/llm/test_aiohttp_retry.py``; this pins
    that jina_embed uses it, stops at three attempts and re-raises the
    provider's own exception.
    """
    calls, raised = await _call_count_for(exc)

    assert calls == attempts
    assert raised is exc
