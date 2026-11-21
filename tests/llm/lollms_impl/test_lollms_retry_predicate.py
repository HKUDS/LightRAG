"""Offline tests for the LoLLMs binding's transient-failure retry policy.

The binding declares three attempts with exponential backoff, so a server that
is briefly unavailable -- a restart, a model still loading -- must be retried
rather than failing the extraction on the first try. A failure that is a
property of the call itself (bad credentials, a malformed request) must still
fail fast: retrying it only re-buys the same failure.

The binding uses the shared ``is_transient_aiohttp_error`` classification,
whose whole table is pinned in ``tests/llm/test_aiohttp_retry.py``. These tests
pin the wiring: one case per branch runs through the real decorator with the
session stubbed, and two local-server tests cover the HTTP path end to end (a
dropped connection, and an expired ``ClientTimeout``).
"""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
from aiohttp import ClientResponseError, RequestInfo
from tenacity import wait_none
from yarl import URL

from lightrag.llm.lollms import lollms_model_if_cache

pytestmark = pytest.mark.offline

_URL = URL("http://127.0.0.1:9600/lollms_generate")


def _http_status_error(status):
    """A real ``ClientResponseError`` -- what a session raises for a status."""
    request_info = RequestInfo(_URL, "POST", {}, _URL)
    return ClientResponseError(request_info, (), status=status, message="unavailable")


class _RaisingResponse:
    """Stands in for the async context manager ``session.post(...)`` returns."""

    def __init__(self, counter, error):
        self._counter = counter
        self._error = error

    async def __aenter__(self):
        self._counter["n"] += 1
        raise self._error

    async def __aexit__(self, *exc_info):
        return False


class _RaisingSession:
    def __init__(self, counter, error):
        self._counter = counter
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def post(self, url, json):
        return _RaisingResponse(self._counter, self._error)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Drop the 4s/8s sleeps; the attempt count is what these tests assert."""
    monkeypatch.setattr(lollms_model_if_cache.retry, "wait", wait_none())


async def _attempts_against(monkeypatch, error):
    counter = {"n": 0}
    monkeypatch.setattr(
        "lightrag.llm.lollms.aiohttp.ClientSession",
        lambda *args, **kwargs: _RaisingSession(counter, error),
    )

    # reraise=True: the caller sees the transport error itself, not an opaque
    # tenacity.RetryError. The pipeline's FAILED summary renders the exception
    # message, so wrapping it would bury the only actionable half.
    with pytest.raises(type(error)) as exc_info:
        await lollms_model_if_cache("test-model", "hello")

    assert exc_info.value is error
    return counter["n"]


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        # A refused connection: the server is restarting, not gone.
        pytest.param(
            aiohttp.ClientConnectionError("Cannot connect to host"),
            3,
            id="connection",
        ),
        # A server still loading its model.
        pytest.param(_http_status_error(503), 3, id="transient-status"),
        # A bad key is not worth three requests.
        pytest.param(_http_status_error(401), 1, id="permanent-status"),
        # A malformed base_url: no retry can fix the configuration.
        pytest.param(aiohttp.InvalidURL("htp//bad"), 1, id="invalid-url"),
    ],
)
async def test_retry_follows_the_shared_classification(monkeypatch, error, attempts):
    assert await _attempts_against(monkeypatch, error) == attempts


class _FlakyServer:
    """Drops the first ``drops`` connections, then answers normally."""

    def __init__(self, drops, body=b"model answer"):
        self.drops = drops
        self.body = body
        self.requests = 0
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's API
                server.requests += 1
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                if server.requests <= server.drops:
                    self.close_connection = True
                    return  # hang up without a response
                self.send_response(200)
                self.send_header("Content-Length", str(len(server.body)))
                self.end_headers()
                self.wfile.write(server.body)

            def log_message(self, *args):  # keep pytest output clean
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        # shutdown() waits for serve_forever's next poll; the 0.5 s default
        # would be most of this test's runtime.
        self._thread = threading.Thread(
            target=self._server.serve_forever, args=(0.01,), daemon=True
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}"


async def test_a_server_that_comes_back_is_recovered_not_failed():
    """The case the retry exists for: unavailable now, available moments later."""
    with _FlakyServer(drops=2) as server:
        result = await lollms_model_if_cache(
            "test-model", "hello", base_url=server.base_url
        )

        assert result == "model answer"
        assert server.requests == 3


async def test_an_expired_total_timeout_is_retried_end_to_end():
    """The binding's own ``ClientTimeout`` fires against a server that accepts
    the connection but never answers; every attempt must reach the server."""
    connections = 0

    async def _never_answer(reader, writer):
        nonlocal connections
        connections += 1
        try:
            await reader.read()  # hold the connection until the client leaves
        finally:
            writer.close()

    server = await asyncio.start_server(_never_answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(asyncio.TimeoutError):
            await lollms_model_if_cache(
                "test-model",
                "hello",
                base_url=f"http://127.0.0.1:{port}",
                timeout=0.1,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert connections == 3
