"""The rerank call declares three attempts with exponential backoff, so a rerank
service that is briefly unavailable -- a restart, a 503 from the gateway in
front of it -- must be retried rather than failing the query on the first try.
A failure that is a property of the request itself (bad API key, wrong base
URL, malformed payload) must still fail fast: retrying it only re-buys the same
answer, after the caller has waited out the backoff.

``generic_rerank_api`` raises ``ClientResponseError`` itself for every non-200
status, so the status half of the matrix runs through the real raise site and
one local server covers the HTTP path end to end.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
from aiohttp import ClientResponseError, RequestInfo
from tenacity import wait_none
from yarl import URL

from lightrag.rerank import _is_transient_rerank_error, generic_rerank_api

pytestmark = pytest.mark.offline

# An upstream that is not ready yet: a throttle, or the 5xx family.
_TRANSIENT_STATUSES = [429, 500, 502, 503, 504]
# A property of the request or the configuration, not of the moment.
_PERMANENT_STATUSES = [400, 401, 403, 404, 422]

_URL = URL("http://127.0.0.1:9600/v1/rerank")
_BASE_URL = "http://127.0.0.1:9600/v1/rerank"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """The 4s/60s backoff is not what these tests measure."""
    monkeypatch.setattr(generic_rerank_api.retry, "wait", wait_none())
    yield


def _http_status_error(status):
    """A real ``ClientResponseError``, shaped like the one the non-200 branch raises."""
    request_info = RequestInfo(_URL, "POST", {}, _URL)
    return ClientResponseError(request_info, (), status=status, message="unavailable")


# --- classification -------------------------------------------------------


@pytest.mark.parametrize("status", _TRANSIENT_STATUSES)
def test_transient_status_is_classified_transient(status):
    assert _is_transient_rerank_error(_http_status_error(status)) is True


@pytest.mark.parametrize("status", _PERMANENT_STATUSES)
def test_permanent_status_is_not_classified_transient(status):
    assert _is_transient_rerank_error(_http_status_error(status)) is False


def test_connection_failure_is_classified_transient():
    assert _is_transient_rerank_error(aiohttp.ClientConnectionError("boom")) is True


def test_unrelated_error_is_not_classified_transient():
    """A bug is not a transient failure; re-running it re-buys the same failure."""
    assert _is_transient_rerank_error(ValueError("boom")) is False


# --- attempt counts through the real decorator ----------------------------


class _RaisingResponse:
    def __init__(self, counter, error):
        self._counter, self._error = counter, error

    async def __aenter__(self):
        self._counter["n"] += 1
        raise self._error

    async def __aexit__(self, *exc):
        return False


class _RaisingSession:
    def __init__(self, counter, error):
        self._counter, self._error = counter, error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        return _RaisingResponse(self._counter, self._error)


async def _attempts_against(monkeypatch, error):
    counter = {"n": 0}
    monkeypatch.setattr(
        "lightrag.rerank.aiohttp.ClientSession",
        lambda *args, **kwargs: _RaisingSession(counter, error),
    )
    with pytest.raises(type(error)):
        await generic_rerank_api(
            "query", ["doc"], "model", base_url=_BASE_URL, api_key=None
        )
    return counter["n"]


@pytest.mark.parametrize("status", _TRANSIENT_STATUSES)
async def test_transient_status_is_retried(monkeypatch, status):
    assert await _attempts_against(monkeypatch, _http_status_error(status)) == 3


@pytest.mark.parametrize("status", _PERMANENT_STATUSES)
async def test_permanent_status_fails_on_the_first_attempt(monkeypatch, status):
    """A bad key or a wrong base URL is not worth three requests."""
    assert await _attempts_against(monkeypatch, _http_status_error(status)) == 1


async def test_connection_failure_is_retried(monkeypatch):
    error = aiohttp.ClientConnectionError("Cannot connect to host")
    assert await _attempts_against(monkeypatch, error) == 3


async def test_unrelated_error_is_not_retried(monkeypatch):
    assert await _attempts_against(monkeypatch, ValueError("boom")) == 1


async def test_the_service_error_survives_the_exhausted_loop(monkeypatch):
    """The caller must see the status, not tenacity's opaque ``RetryError``."""
    error = _http_status_error(503)
    monkeypatch.setattr(
        "lightrag.rerank.aiohttp.ClientSession",
        lambda *args, **kwargs: _RaisingSession({"n": 0}, error),
    )
    with pytest.raises(ClientResponseError) as exc_info:
        await generic_rerank_api(
            "query", ["doc"], "model", base_url=_BASE_URL, api_key=None
        )
    assert exc_info.value.status == 503
    assert "unavailable" in str(exc_info.value)


# --- end to end against a local server ------------------------------------


class _RerankServer:
    """Answers POSTs with a scripted sequence of statuses; the last one repeats."""

    def __init__(self, statuses, payload=None):
        self.statuses = statuses
        self.payload = {"results": []} if payload is None else payload
        self.requests = 0
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                server.requests += 1
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                index = min(server.requests - 1, len(server.statuses) - 1)
                body = json.dumps(server.payload).encode()
                self.send_response(server.statuses[index])
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._srv.server_address[1]}/v1/rerank"

    def shutdown(self):
        self._srv.shutdown()


async def test_a_bad_api_key_fails_after_one_request():
    """The case this fix is about: 401 must not cost three requests plus backoff."""
    server = _RerankServer([401])
    try:
        with pytest.raises(ClientResponseError) as exc_info:
            await generic_rerank_api(
                "query", ["doc"], "model", base_url=server.base_url, api_key="wrong"
            )
    finally:
        server.shutdown()
    assert exc_info.value.status == 401
    assert server.requests == 1


async def test_a_restarting_service_is_retried_until_it_answers():
    server = _RerankServer([503, 503, 200])
    try:
        result = await generic_rerank_api(
            "query", ["doc"], "model", base_url=server.base_url, api_key=None
        )
    finally:
        server.shutdown()
    assert server.requests == 3
    assert result == []
