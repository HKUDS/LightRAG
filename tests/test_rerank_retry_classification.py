"""Regression test: generic_rerank_api must not retry permanent errors.

Every rerank query with a bad API key used to pay three requests plus the
exponential backoff before ``apply_rerank_if_enabled`` fell back to the
original chunk order, and the fallback log showed ``tenacity.RetryError``
instead of the provider's message.
"""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from tenacity import wait_none

from lightrag.rerank import generic_rerank_api

pytestmark = pytest.mark.offline

_URL = "http://rerank.test/v1/rerank"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Skip the real backoff; monkeypatch restores it after each test."""
    monkeypatch.setattr(generic_rerank_api.retry, "wait", wait_none())


def _response(status: int, json_body=None) -> Mock:
    response = Mock()
    response.status = status
    response.headers = {"content-type": "application/json"}
    response.request_info = aiohttp.RequestInfo(
        url=_URL, method="POST", headers={}, real_url=_URL
    )
    response.history = ()
    response.text = AsyncMock(return_value=f'{{"detail": "HTTP {status}"}}')
    response.json = AsyncMock(return_value=json_body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


def _session(*outcomes) -> Mock:
    """A ClientSession whose successive posts yield ``outcomes``.

    An exception outcome is raised when the request is entered, the way a
    transport failure surfaces from ``session.post``; anything else is the
    response object.
    """
    posts = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            ctx = Mock()
            ctx.__aenter__ = AsyncMock(side_effect=outcome)
            ctx.__aexit__ = AsyncMock(return_value=None)
            posts.append(ctx)
        else:
            posts.append(outcome)
    session = Mock()
    session.post = Mock(side_effect=posts)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


async def _rerank(session: Mock):
    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        return await generic_rerank_api(
            query="q",
            documents=["a", "b"],
            model="m",
            base_url=_URL,
            api_key="key",
        )


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
async def test_permanent_status_is_not_retried(status):
    """Bad key, exhausted balance or a malformed request fails on attempt one."""
    session = _session(*[_response(status) for _ in range(3)])

    # reraise=True: the provider's own exception, never tenacity.RetryError.
    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        await _rerank(session)

    assert session.post.call_count == 1, f"a permanent {status} must not be retried"
    assert excinfo.value.status == status


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
async def test_transient_status_is_retried(status):
    """Timeouts, conflicts, rate limits and 5xx are retried to the stop limit."""
    session = _session(*[_response(status) for _ in range(3)])

    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        await _rerank(session)

    assert session.post.call_count == 3
    assert excinfo.value.status == status


async def test_transient_status_then_success_returns_results():
    """A 503 followed by a 200 yields the second attempt's results."""
    ok = _response(200, {"results": [{"index": 1, "relevance_score": 0.9}]})
    session = _session(_response(503), ok)

    result = await _rerank(session)

    assert session.post.call_count == 2
    assert result == [{"index": 1, "relevance_score": 0.9}]


async def test_invalid_url_is_not_retried():
    """A malformed base_url raises InvalidURL, which no retry can fix."""
    session = _session(*[aiohttp.InvalidURL("not-a-url") for _ in range(3)])

    with pytest.raises(aiohttp.InvalidURL):
        await _rerank(session)

    assert session.post.call_count == 1


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: aiohttp.ClientConnectionError("connection reset"),
        lambda: aiohttp.ServerDisconnectedError(),
        lambda: aiohttp.ClientPayloadError("response body truncated"),
    ],
    ids=["connection-error", "server-disconnected", "payload-error"],
)
async def test_connection_and_payload_errors_are_retried(make_exc):
    """Dropped connections and truncated bodies are retried to the limit."""
    first = make_exc()
    session = _session(*[make_exc() for _ in range(3)])

    with pytest.raises(type(first)):
        await _rerank(session)

    assert session.post.call_count == 3
