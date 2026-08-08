"""Redirect / opener hardening for the native markdown image downloader.

GHSA-25c3-j78v-83qx: the redirect path was the one channel that escaped every
resource control in the fetch. Two distinct defects, both covered here:

* ``HTTPRedirectHandler.http_error_302`` drains the redirect body with an
  unbounded ``fp.read()`` before following the jump, so ``302`` + a huge body
  is an attacker-controlled allocation that lands before the image itself ever
  reaches the size-capped read.
* the stdlib allows redirecting to ``ftp://`` and ``build_opener()`` installs
  an ``FTPHandler``, which never goes through ``_GuardedHTTPConnection``.
"""

from __future__ import annotations

import email.message
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lightrag.parser.markdown import parser as md_parser

from tests.parser.markdown.conftest import PNG_BYTES as _PNG_BYTES

# How long the redirect handler holds its body open. Long enough that a client
# which drains it cannot finish inside the assertion window.
_BODY_HOLD_SECONDS = 10.0
_FAST_ENOUGH_SECONDS = 3.0


# --------------------------------------------------------------------------
# Handler unit tests: deterministic, no sockets.
# --------------------------------------------------------------------------


class _ExplodingBody:
    """A response ``fp`` that fails the test if anyone reads it."""

    def __init__(self) -> None:
        self.closes = 0

    def read(self, *args, **kwargs):
        raise AssertionError("redirect body must never be read")

    def readline(self, *args, **kwargs):
        raise AssertionError("redirect body must never be read")

    def close(self) -> None:
        self.closes += 1


class _RecordingParent:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def open(self, req, timeout=None):
        self.opened.append(req.full_url)
        return "FOLLOWED"


def _headers(location: str) -> email.message.Message:
    msg = email.message.Message()
    msg["location"] = location
    return msg


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirect_handler_never_reads_the_body(monkeypatch, code):
    # The base class assigns http_error_301/303/307/308 as independent
    # attributes pointing at ITS OWN http_error_302, so a subclass that
    # overrides only http_error_302 leaves the other four resolving to the
    # original draining implementation. Every status code is checked.
    monkeypatch.setattr(md_parser, "_host_is_public", lambda host: True)
    handler = md_parser._GuardedRedirectHandler()
    parent = _RecordingParent()
    handler.parent = parent
    fp = _ExplodingBody()
    req = urllib.request.Request("http://origin.example/a.png")
    req.timeout = 30  # normally stamped by OpenerDirector.open()

    result = getattr(handler, f"http_error_{code}")(
        req, fp, code, "Found", _headers("http://origin.example/b.png")
    )

    assert result == "FOLLOWED"
    assert parent.opened == ["http://origin.example/b.png"]
    assert fp.closes == 1


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirect_to_ftp_is_refused_for_every_status_code(monkeypatch, code):
    monkeypatch.setattr(md_parser, "_host_is_public", lambda host: True)
    handler = md_parser._GuardedRedirectHandler()
    parent = _RecordingParent()
    handler.parent = parent
    fp = _ExplodingBody()
    req = urllib.request.Request("http://origin.example/a.png")
    req.timeout = 30  # normally stamped by OpenerDirector.open()

    with pytest.raises(urllib.error.HTTPError, match="scheme 'ftp'"):
        getattr(handler, f"http_error_{code}")(
            req, fp, code, "Found", _headers("ftp://elsewhere.example/b.png")
        )
    assert parent.opened == []
    assert fp.closes == 1


def test_redirect_closes_the_body_and_charges_before_resolving_the_target(
    monkeypatch,
):
    """Order of operations on a hop: close, charge, only then resolve.

    All three matter. Resolving first means an exhausted request budget still
    buys one more DNS lookup — the single phase of a fetch that can be neither
    deadline-bounded nor cancelled. It also meant a hop the SSRF guard went on
    to refuse was never counted as the attempt it was, and that the refusal
    path handed a still-open body to the HTTPError.
    """
    order: list[str] = []

    def _refusing_host_check(host):
        order.append("resolve")
        return False

    monkeypatch.setattr(md_parser, "_host_is_public", _refusing_host_check)

    budget = md_parser._ImageBudget(
        max_total_bytes=10**9, max_requests=10, total_timeout=300
    )

    class _TrackedBody(_ExplodingBody):
        def close(self):
            order.append("close")
            super().close()

    fp = _TrackedBody()
    handler = md_parser._GuardedRedirectHandler()
    handler.parent = _RecordingParent()
    req = urllib.request.Request("http://origin.example/a.png")
    req.timeout = 30

    with md_parser._download_context(deadline=None, budget=budget, poll_interval=60.0):
        with pytest.raises(urllib.error.HTTPError, match="non-public host"):
            handler.http_error_302(
                req, fp, 302, "Found", _headers("http://internal.example/b.png")
            )

    assert order == ["close", "resolve"]
    assert fp.closes == 1
    assert budget.requests_used == 1  # the refused hop was still an attempt


def test_redirect_budget_exhaustion_skips_the_target_resolution(monkeypatch):
    resolves = {"n": 0}

    def _counting_host_check(host):
        resolves["n"] += 1
        return True

    monkeypatch.setattr(md_parser, "_host_is_public", _counting_host_check)
    budget = md_parser._ImageBudget(
        max_total_bytes=10**9, max_requests=1, total_timeout=300
    )
    budget.charge_request()  # the initial request already spent it

    fp = _ExplodingBody()
    handler = md_parser._GuardedRedirectHandler()
    handler.parent = _RecordingParent()
    req = urllib.request.Request("http://origin.example/a.png")
    req.timeout = 30

    with md_parser._download_context(deadline=None, budget=budget, poll_interval=60.0):
        with pytest.raises(md_parser._ImageRequestBudgetExceeded):
            handler.http_error_302(
                req, fp, 302, "Found", _headers("http://elsewhere.example/b.png")
            )

    assert fp.closes == 1
    assert resolves["n"] == 0  # no DNS bought by an exhausted budget


def test_guarded_opener_carries_no_non_http_handlers():
    opener = md_parser._build_guarded_opener()
    names = {type(h).__name__ for h in opener.handlers}
    assert "FTPHandler" not in names
    assert "CacheFTPHandler" not in names
    assert "FileHandler" not in names
    assert "DataHandler" not in names
    # …and still speaks the two schemes we need.
    assert "_GuardedHTTPHandler" in names
    assert "_GuardedHTTPSHandler" in names


def test_guarded_opener_rejects_a_non_http_url():
    opener = md_parser._build_guarded_opener()
    # "unknown url type" is UnknownHandler talking, i.e. no handler claimed the
    # scheme. With an FTPHandler installed this would instead fail late, with a
    # connection error, after actually dialling the target.
    with pytest.raises(urllib.error.URLError, match="unknown url type"):
        opener.open("ftp://127.0.0.1:1/x.png")


# --------------------------------------------------------------------------
# Loopback listener tests: prove the client does not wait for a redirect body.
# --------------------------------------------------------------------------


class _RedirectHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: A003 - silence the stdlib access log
        pass

    def do_GET(self):  # noqa: N802 - stdlib naming
        if self.path.startswith("/redir"):
            body_len = 1024 * 1024 * 1024  # 1 GiB claimed, never delivered
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{self.server.server_port}/img.png"
            )
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(body_len))
            self.end_headers()
            # Trickle: a client that drains this body blocks for the hold.
            deadline = time.monotonic() + _BODY_HOLD_SECONDS
            try:
                while time.monotonic() < deadline:
                    self.wfile.write(b"\x00")
                    self.wfile.flush()
                    time.sleep(0.2)
            except OSError:
                pass  # client hung up, which is exactly what we want
            return
        if self.path.startswith("/ftpredir"):
            self.send_response(302)
            self.send_header("Location", self.server.ftp_target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(_PNG_BYTES)))
        self.end_headers()
        self.wfile.write(_PNG_BYTES)


@pytest.fixture
def redirect_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    server.daemon_threads = True
    server.ftp_target = ""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def ftp_probe():
    """A bare TCP listener standing in for an FTP server; records connections."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(4)
    connections: list[str] = []
    stop = threading.Event()

    def _accept():
        sock.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, addr = sock.accept()
            except (TimeoutError, OSError):
                continue
            connections.append(str(addr))
            conn.close()

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    try:
        yield sock.getsockname()[1], connections
    finally:
        stop.set()
        thread.join(timeout=2)
        sock.close()


def _resolve_one(parser, url: str):
    _, warnings, meta = parser._extract_text(f"# H\n\n![x]({url})\n", bundle_root=None)
    (drawing,) = meta["md_drawings"].values()
    return drawing, warnings


def test_redirect_does_not_wait_for_the_body(monkeypatch, redirect_server):
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "127.0.0.0/8")
    port = redirect_server.server_port
    parser = md_parser.NativeMarkdownParser()

    started = time.monotonic()
    drawing, _ = _resolve_one(parser, f"http://127.0.0.1:{port}/redir")
    elapsed = time.monotonic() - started

    # The image behind the redirect is fetched normally…
    assert drawing["kind"] == "local"
    # …without ever waiting on the 1 GiB body the redirect advertised.
    assert elapsed < _FAST_ENOUGH_SECONDS, f"took {elapsed:.1f}s draining the redirect"


def test_ftp_redirect_is_refused_before_any_ftp_connection(
    monkeypatch, redirect_server, ftp_probe
):
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "127.0.0.0/8")
    ftp_port, connections = ftp_probe
    redirect_server.ftp_target = f"ftp://127.0.0.1:{ftp_port}/x.png"
    parser = md_parser.NativeMarkdownParser()

    drawing, warnings = _resolve_one(
        parser, f"http://127.0.0.1:{redirect_server.server_port}/ftpredir"
    )

    assert drawing["kind"] == "external"
    assert warnings.get("images_download_failed") == 1
    time.sleep(0.4)  # give the probe a chance to have recorded a stray connect
    assert connections == []
