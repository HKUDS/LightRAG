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


@pytest.mark.parametrize(
    ("status", "attempts"),
    [pytest.param(401, 1, id="permanent"), pytest.param(503, 3, id="transient")],
)
async def test_status_retry_follows_the_shared_classification(status, attempts):
    """One status per branch; the full table is in tests/llm/test_aiohttp_retry.py."""
    session = _session(*[_response(status) for _ in range(3)])

    # reraise=True: the provider's own exception, never tenacity.RetryError.
    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        await _rerank(session)

    assert session.post.call_count == attempts
    assert excinfo.value.status == status


async def test_transient_status_then_success_returns_results():
    """A 503 followed by a 200 yields the second attempt's results."""
    ok = _response(200, {"results": [{"index": 1, "relevance_score": 0.9}]})
    session = _session(_response(503), ok)

    result = await _rerank(session)

    assert session.post.call_count == 2
    assert result == [{"index": 1, "relevance_score": 0.9}]


@pytest.mark.parametrize(
    ("make_exc", "attempts"),
    [
        pytest.param(lambda: aiohttp.InvalidURL("not-a-url"), 1, id="invalid-url"),
        pytest.param(
            lambda: aiohttp.ClientConnectionError("connection reset"),
            3,
            id="connection",
        ),
    ],
)
async def test_transport_retry_follows_the_shared_classification(make_exc, attempts):
    """A transport failure raised by ``session.post`` reaches the same policy."""
    session = _session(*[make_exc() for _ in range(3)])

    with pytest.raises(type(make_exc())):
        await _rerank(session)

    assert session.post.call_count == attempts
