"""The shared retry classification for aiohttp-based bindings.

``is_transient_aiohttp_error`` is the single place that decides what jina,
lollms and the generic rerank API retry; this file pins its whole table. The
per-binding tests only pin the wiring (one case per branch, run through the
real tenacity decorator), so a status or exception class added here does not
need a copy in every binding's suite.
"""

import asyncio

import aiohttp
import pytest

from lightrag.llm._aiohttp_retry import is_transient_aiohttp_error

pytestmark = pytest.mark.offline

_URL = "http://provider.test/v1"


def _status_error(status: int) -> aiohttp.ClientResponseError:
    request_info = aiohttp.RequestInfo(
        url=_URL, method="POST", headers={}, real_url=_URL
    )
    return aiohttp.ClientResponseError(request_info, (), status=status)


@pytest.mark.parametrize(
    ("status", "transient"),
    [
        # The listed transient statuses, each one: dropping any is a change.
        (408, True),
        (409, True),
        (429, True),
        # Every 5xx: the first code pins the threshold.
        (500, True),
        # Every other status is a property of the request or the account;
        # 499 pins the other side of the 5xx threshold.
        (401, False),
        (499, False),
    ],
)
def test_status_classification(status, transient):
    assert is_transient_aiohttp_error(_status_error(status)) is transient


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        pytest.param(aiohttp.ClientConnectionError("reset"), True, id="connection"),
        pytest.param(aiohttp.ClientPayloadError("truncated"), True, id="payload"),
        # An expired ClientTimeout(total=...) is a bare TimeoutError.
        pytest.param(asyncio.TimeoutError(), True, id="timeout"),
        # A ClientError no retry can fix: a malformed base_url.
        pytest.param(aiohttp.InvalidURL("not-a-url"), False, id="invalid-url"),
        # The LightRAG queue's own cap cancels the call; never retry that.
        pytest.param(asyncio.CancelledError(), False, id="cancelled"),
        pytest.param(ValueError("bug"), False, id="non-aiohttp"),
    ],
)
def test_exception_classification(exc, transient):
    assert is_transient_aiohttp_error(exc) is transient
