"""Wall-clock bounding and cancellability of the native markdown image fetch.

GHSA-25c3-j78v-83qx defect 2: ``NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT`` was passed
to ``opener.open`` as urllib's per-socket-operation timeout and the body was
read in a single call, so a peer trickling one byte per interval held a parse
worker indefinitely — and nothing in the markdown engine ever consulted a
cancellation event, so ``/documents/cancel_pipeline`` could not reclaim it.

The tests are grouped by which phase of the fetch they pin:

* deterministic unit tests for the trip-reason logic, the DNS checkpoints and
  the connect loop (fake clock / fake socket, no real blocking);
* loopback listener tests for the phases that only misbehave over a real
  socket. Each listener TRICKLES rather than stalling outright: a peer that
  sends nothing is already handled by the socket timeout and proves nothing,
  whereas one byte per interval is exactly what resets a per-socket-operation
  timeout forever. The header and body cases hang indefinitely on the pre-fix
  tree; the TLS case is coverage only (noted on the test itself).
"""

from __future__ import annotations

import errno
import ipaddress
import socket
import ssl
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lightrag.parser.markdown import parser as md_parser
from lightrag.parser.llm_bridge import LLMBridgePipelineCancelled

from tests.parser.markdown.conftest import PNG_BYTES as _PNG_BYTES

# The listener emits a byte every _TRICKLE_INTERVAL, comfortably inside the
# socket timeout, so a per-socket-operation timeout never fires.
_TRICKLE_INTERVAL = 0.25
_DEADLINE_SECONDS = 2
_DEADLINE_PLUS_SLACK = 8.0


# --------------------------------------------------------------------------
# trip_reason(): classification must not depend on watchdog scheduling.
# --------------------------------------------------------------------------


def _state(clock, *, timeout=30.0, cancel_events=()):
    return md_parser._DownloadState(
        deadline=clock.now + timeout, cancel_events=cancel_events
    )


def test_trip_reason_is_none_while_the_fetch_is_healthy(clock):
    assert _state(clock).trip_reason() is None


def test_trip_reason_reports_cancellation_without_the_watchdog(clock):
    event = threading.Event()
    state = _state(clock, cancel_events=((event, LLMBridgePipelineCancelled),))
    event.set()
    # No watchdog has run: the reason is derived synchronously.
    assert isinstance(state.trip_reason(), LLMBridgePipelineCancelled)


def test_trip_reason_reports_the_request_deadline_without_the_watchdog(clock):
    state = _state(clock, timeout=5.0)
    clock.advance(6.0)
    assert isinstance(state.trip_reason(), TimeoutError)


def test_cancellation_outranks_an_expired_deadline(clock):
    event = threading.Event()
    state = _state(
        clock, timeout=5.0, cancel_events=((event, LLMBridgePipelineCancelled),)
    )
    event.set()
    clock.advance(6.0)
    # A cancelled document must be reported as cancelled, not as a timeout:
    # only the former is recorded as "cancelled" rather than "failed".
    assert isinstance(state.trip_reason(), LLMBridgePipelineCancelled)


def test_watchdog_shuts_down_a_registered_socket_and_records_the_reason(clock):
    state = _state(clock, timeout=5.0)

    class _Sock:
        def __init__(self):
            self.shutdowns = 0

        def shutdown(self, how):
            self.shutdowns += 1

    sock = _Sock()
    state.register_socket(sock)
    assert state.watchdog_tick() is False
    clock.advance(6.0)
    assert state.watchdog_tick() is True
    assert sock.shutdowns == 1


def test_socket_registered_after_the_watchdog_tripped_is_shut_down_at_once(clock):
    state = _state(clock, timeout=5.0)
    clock.advance(6.0)
    assert state.watchdog_tick() is True

    class _Sock:
        def __init__(self):
            self.shutdowns = 0

        def shutdown(self, how):
            self.shutdowns += 1

    sock = _Sock()
    state.register_socket(sock)
    # Closes the arm/register race: the socket must not survive because it
    # arrived a moment after the trip.
    assert sock.shutdowns == 1


# --------------------------------------------------------------------------
# _pin_socket(): connect is deadline-bounded and cancellable in-thread.
# --------------------------------------------------------------------------


class _NeverConnectingSocket:
    """A socket whose connect stays EINPROGRESS forever."""

    def __init__(self, *args, **kwargs):
        self.closed = 0
        self.timeout = None

    def bind(self, addr):
        pass

    def setblocking(self, flag):
        pass

    def settimeout(self, value):
        self.timeout = value

    def setsockopt(self, *args):
        pass

    def connect_ex(self, addr):
        return errno.EINPROGRESS

    def getsockopt(self, *args):
        return 0

    def fileno(self):
        return -1

    def close(self):
        self.closed += 1


class _NeverWritableSelector:
    """Never reports the socket writable; runs a hook on each poll.

    The hook is how a test makes the deadline expire (or a cancel land) *while*
    the connect loop is polling, which is the state the loop actually has to
    handle — pre-tripping before the call is caught earlier, by the checkpoint
    around DNS.
    """

    on_select = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def register(self, *args):
        pass

    def select(self, timeout=None):
        if type(self).on_select is not None:
            type(self).on_select(timeout)
        return []


@pytest.fixture
def stalled_connect(monkeypatch):
    created: list[_NeverConnectingSocket] = []

    def _factory(*args, **kwargs):
        sock = _NeverConnectingSocket()
        created.append(sock)
        return sock

    monkeypatch.setattr(md_parser.socket, "socket", _factory)
    monkeypatch.setattr(md_parser.selectors, "DefaultSelector", _NeverWritableSelector)
    monkeypatch.setattr(
        md_parser, "_validated_addresses", lambda host: ["93.184.216.34"]
    )
    monkeypatch.setattr(_NeverWritableSelector, "on_select", None)
    return created


def test_connect_gives_up_at_the_fetch_deadline(monkeypatch, clock, stalled_connect):
    monkeypatch.setattr(
        _NeverWritableSelector,
        "on_select",
        staticmethod(lambda timeout: clock.advance((timeout or 0.1) + 0.01)),
    )
    with md_parser._download_context(deadline=clock.now + 5.0, poll_interval=60.0):
        with pytest.raises(TimeoutError):
            md_parser._pin_socket("host.example", 80, 30, None)
    assert stalled_connect[0].closed == 1


def test_connect_aborts_promptly_on_cancellation(monkeypatch, clock, stalled_connect):
    event = threading.Event()
    polls = {"n": 0}

    def _on_select(timeout):
        polls["n"] += 1
        if polls["n"] == 2:
            event.set()  # cancelled mid-poll

    monkeypatch.setattr(_NeverWritableSelector, "on_select", staticmethod(_on_select))
    with md_parser._download_context(
        deadline=clock.now + 300.0,
        cancel_events=((event, LLMBridgePipelineCancelled),),
        poll_interval=60.0,  # the watchdog must not be what notices
    ):
        with pytest.raises(LLMBridgePipelineCancelled):
            md_parser._pin_socket("host.example", 80, 30, None)
    # Noticed within a couple of poll intervals, and aborted by THIS thread —
    # a cross-thread shutdown cannot abort a connect (on macOS it makes
    # connect return successfully instead).
    assert polls["n"] <= 3
    assert stalled_connect[0].closed == 1


def test_connect_without_an_active_context_still_honours_its_timeout(
    monkeypatch, clock, stalled_connect
):
    # No _download_context here: the timeout argument is the only bound left,
    # and dropping it would turn an EINPROGRESS peer into an infinite poll —
    # a regression against socket.create_connection(..., timeout).
    assert md_parser._active_download() is None
    monkeypatch.setattr(
        _NeverWritableSelector,
        "on_select",
        staticmethod(lambda timeout: clock.advance((timeout or 0.1) + 0.01)),
    )
    with pytest.raises(TimeoutError):
        md_parser._pin_socket("host.example", 80, 0.5, None)
    assert stalled_connect[0].closed == 1


# --------------------------------------------------------------------------
# DNS: the one uninterruptible phase, bracketed by checkpoints.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resolved",
    [
        pytest.param(["93.184.216.34"], id="public"),
        pytest.param([], id="empty"),
        pytest.param("gaierror", id="gaierror"),
    ],
)
def test_cancel_during_dns_is_not_swallowed(monkeypatch, clock, resolved):
    """A cancel raised while resolving must propagate, whatever DNS returned.

    The empty / gaierror branches are the ones that used to lose it: with no
    usable address ``_host_is_public`` returns False and the caller raises
    before the connect is ever attempted, so a checkpoint placed in
    ``_pin_socket`` would never run and the cancellation would be reported as
    an ordinary download failure.
    """
    event = threading.Event()
    calls = {"resolve": 0, "connect": 0}

    def _fake_validated(host):
        calls["resolve"] += 1
        event.set()  # cancelled *during* the resolve
        if resolved == "gaierror":
            return []
        return list(resolved)

    monkeypatch.setattr(md_parser, "_validated_addresses", _fake_validated)
    monkeypatch.setattr(
        md_parser.socket,
        "socket",
        lambda *a, **k: pytest.fail("connect attempted after cancellation"),
    )

    with md_parser._download_context(
        deadline=clock.now + 300.0,
        cancel_events=((event, LLMBridgePipelineCancelled),),
        poll_interval=60.0,
    ):
        with pytest.raises(LLMBridgePipelineCancelled):
            md_parser._host_is_public("host.example")
    assert calls["resolve"] == 1


def test_resolution_memo_keys_on_membership_not_truthiness(monkeypatch, clock):
    # [] is a RESULT (resolution failed, or an address was non-public), not a
    # cache miss. Keying on truthiness would re-resolve it every time and
    # reopen the very window the memo exists to shrink.
    calls = {"n": 0}

    def _fake_validated(host):
        calls["n"] += 1
        return []

    monkeypatch.setattr(md_parser, "_validated_addresses", _fake_validated)
    with md_parser._download_context(deadline=clock.now + 300.0, poll_interval=60.0):
        assert md_parser._host_is_public("host.example") is False
        assert md_parser._resolve_shared("host.example") == []
        assert md_parser._resolve_shared("host.example") == []
    assert calls["n"] == 1


def test_resolution_is_shared_between_validation_and_connect(monkeypatch, clock):
    calls = {"n": 0}

    def _fake_validated(host):
        calls["n"] += 1
        return ["93.184.216.34"]

    monkeypatch.setattr(md_parser, "_validated_addresses", _fake_validated)
    with md_parser._download_context(deadline=clock.now + 300.0, poll_interval=60.0):
        assert md_parser._host_is_public("host.example") is True
        assert md_parser._resolve_shared("host.example") == ["93.184.216.34"]
    # Validation and the connect that follows share one resolution, which is
    # what the _validated_addresses docstring has always claimed.
    assert calls["n"] == 1


def test_resolution_memo_folds_hostname_case(monkeypatch, clock):
    # _fetch seeds the memo with urlparse().hostname, which lowercases, while
    # the connect path looks it up with http.client's case-preserved req.host.
    # DNS is case-insensitive; a memo that splits on case re-runs the one
    # uninterruptible phase on every fetch of a non-lowercase host.
    calls = {"n": 0}

    def _fake_validated(host):
        calls["n"] += 1
        return ["93.184.216.34"]

    monkeypatch.setattr(md_parser, "_validated_addresses", _fake_validated)
    with md_parser._download_context(deadline=clock.now + 300.0, poll_interval=60.0):
        assert md_parser._host_is_public("host.example") is True
        assert md_parser._resolve_shared("HOST.example") == ["93.184.216.34"]
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# Loopback listeners: phases that only misbehave over a real socket.
# --------------------------------------------------------------------------


class _StalledListener:
    """Accepts connections and then does (almost) nothing with them."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._conns: list[socket.socket] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except (TimeoutError, OSError):
                continue
            self._conns.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        # Every mode below trickles rather than stalling outright: a peer that
        # sends NOTHING is already handled by the socket timeout, so it proves
        # nothing about the deadline. One byte per interval is what resets a
        # per-socket-operation timeout forever, which is the actual defect.
        try:
            if self.mode == "trickle_headers":
                conn.recv(4096)
                conn.sendall(b"HTTP/1.1 200 OK\r\n")
                # Never terminates the header block.
                while not self._stop.wait(_TRICKLE_INTERVAL):
                    conn.sendall(b"X")
            elif self.mode == "trickle_tls":
                # A handshake record announcing a 16383-byte payload, then one
                # payload byte per interval: the client keeps waiting for a
                # record it will never receive in full. The length must stay
                # within the 2**14 TLSPlaintext limit — announce more and the
                # client rejects the record immediately, which would make this
                # a "server speaks nonsense" test rather than a stall test.
                conn.recv(4096)
                conn.sendall(b"\x16\x03\x03\x3f\xff")
                while not self._stop.wait(_TRICKLE_INTERVAL):
                    conn.sendall(b"\x00")
            elif self.mode == "trickle_body":
                conn.recv(4096)
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n\r\n")
                conn.sendall(_PNG_BYTES[:8])
                while not self._stop.wait(_TRICKLE_INTERVAL):
                    conn.sendall(b"\x00")
        except OSError:
            pass

    def close(self) -> None:
        self._stop.set()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        self.sock.close()


@pytest.fixture
def stalled_listener(request):
    listener = _StalledListener(request.param)
    try:
        yield listener
    finally:
        listener.close()


def _fetch_one(url: str, warnings: dict | None = None):
    parser = md_parser.NativeMarkdownParser()
    _, warns, meta = parser._extract_text(f"# H\n\n![x]({url})\n", bundle_root=None)
    (drawing,) = meta["md_drawings"].values()
    return drawing, warns


def _assert_bounded_fetch(url: str, *, phase: str) -> None:
    started = time.monotonic()
    drawing, warns = _fetch_one(url)
    elapsed = time.monotonic() - started
    assert drawing["kind"] == "external"
    assert warns.get("images_download_failed") == 1
    assert elapsed < _DEADLINE_PLUS_SLACK, (
        f"{phase} trickle held the worker for {elapsed:.1f}s "
        f"against a {_DEADLINE_SECONDS}s deadline"
    )


@pytest.fixture
def download_env(monkeypatch):
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "127.0.0.0/8")
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT", str(_DEADLINE_SECONDS))


@pytest.mark.parametrize("stalled_listener", ["trickle_headers"], indirect=True)
def test_deadline_trips_on_trickled_response_headers(download_env, stalled_listener):
    # The status line arrives, then one header byte per interval and never a
    # blank line. http.client reads headers with repeated readline()s, each of
    # which resets the socket timeout, so pre-fix this blocked in opener.open()
    # indefinitely — before the body loop was ever reached.
    _assert_bounded_fetch(
        f"http://127.0.0.1:{stalled_listener.port}/x.png", phase="header"
    )


@pytest.mark.parametrize("stalled_listener", ["trickle_tls"], indirect=True)
def test_deadline_trips_during_a_trickled_tls_handshake(download_env, stalled_listener):
    """A stalled handshake stays inside the deadline.

    Coverage, not a pre/post discriminator: OpenSSL's own read behaviour
    already made the pre-fix code give up here within the slack, so this does
    not fail on the old tree. It is here because the handshake is the phase
    ``wrap_socket()`` hides — it DETACHES the socket it wraps, leaving the
    watchdog holding an fd of -1 — and the split-handshake wiring that fixes
    that needs a stalled peer to be exercised at all.
    """
    _assert_bounded_fetch(
        f"https://127.0.0.1:{stalled_listener.port}/x.png", phase="TLS handshake"
    )


def test_https_still_works_against_a_real_tls_server(monkeypatch, tmp_path):
    """The split handshake must not have broken ordinary HTTPS.

    ``do_handshake_on_connect=False`` plus an explicit ``do_handshake()`` is
    equivalent to what ``wrap_socket`` does internally, but "equivalent" is
    worth proving: nothing else in the suite completes a real TLS handshake
    through ``_GuardedHTTPSConnection``.
    """
    pytest.importorskip("cryptography")
    cert_file, ca_file = _self_signed_cert(tmp_path)

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert_file)
    client_ctx = ssl.create_default_context(cafile=str(ca_file))

    server = ThreadingHTTPServer(("127.0.0.1", 0), _PngHandler)
    server.socket = server_ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def _trusting_opener():
        opener = urllib.request.OpenerDirector()
        for handler in (
            urllib.request.ProxyHandler({}),
            md_parser._GuardedHTTPHandler(),
            md_parser._GuardedHTTPSHandler(context=client_ctx),
            md_parser._GuardedRedirectHandler(),
            urllib.request.HTTPErrorProcessor(),
            urllib.request.HTTPDefaultErrorHandler(),
            urllib.request.UnknownHandler(),
        ):
            opener.add_handler(handler)
        return opener

    monkeypatch.setattr(md_parser, "_build_guarded_opener", _trusting_opener)
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    # Dial the literal, not "localhost": that name resolves to ::1 as well,
    # and _pin_socket dials the first validated address, which would miss an
    # IPv4-only listener. The cert carries a 127.0.0.1 IP SAN so verification
    # still runs for real.
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "127.0.0.0/8")
    try:
        drawing, warns = _fetch_one(f"https://127.0.0.1:{server.server_port}/x.png")
    finally:
        server.shutdown()
        server.server_close()

    assert drawing["kind"] == "local", warns
    assert not warns.get("images_download_failed")


def _self_signed_cert(tmp_path):
    """Write a self-signed cert+key valid for localhost/127.0.0.1."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    # Fixed validity window anchored to "now" — the cert lives for the length
    # of one test run only.
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM)
    combined = tmp_path / "server.pem"
    combined.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        + pem
    )
    ca = tmp_path / "ca.pem"
    ca.write_bytes(pem)
    return str(combined), ca


# --------------------------------------------------------------------------
# End to end: /documents/cancel_pipeline reaches the image downloader.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stalled_listener", ["trickle_body"], indirect=True)
def test_pipeline_cancel_event_aborts_a_native_md_parse(
    monkeypatch, tmp_path, stalled_listener
):
    """Drive the production path: get_parser("native").parse(ParseContext(...)).

    Exercises the whole chain the fix touches — NativeParserBase building
    runtime.cancel_events from ctx.pipeline_cancel_event, the executor
    dispatch, and the image downloader polling those events mid-fetch. Before
    this, ``grep -rn cancel lightrag/parser/markdown/`` returned nothing: the
    pipeline event reached the LLM bridge only, so a document stuck in a
    trickled image download could not be reclaimed by anything short of a
    process restart.
    """
    import asyncio
    import threading as _threading

    from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.parser.registry import get_parser

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "127.0.0.0/8")
    # Long enough that the deadline is NOT what ends the parse — the cancel is.
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT", "300")
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_TOTAL_TIMEOUT", "300")

    source = input_dir / "doc.md"
    source.write_text(f"# H\n\n![x](http://127.0.0.1:{stalled_listener.port}/x.png)\n")

    cancel_event = _threading.Event()
    ctx = ParseContext(
        build_debug_rag(),
        "doc-cancel",
        str(source),
        {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        pipeline_cancel_event=cancel_event,
    )

    async def _drive():
        task = asyncio.create_task(get_parser("native").parse(ctx))
        await asyncio.sleep(1.0)  # let the fetch get into the trickled body
        cancel_event.set()
        return await asyncio.wait_for(task, timeout=10.0)

    started = time.monotonic()
    with pytest.raises(LLMBridgePipelineCancelled):
        asyncio.run(_drive())
    elapsed = time.monotonic() - started

    # Reclaimed on the cancel, not on the 300 s deadline.
    assert elapsed < 10.0, f"cancel took {elapsed:.1f}s to reach the download"
    # The type matters as much as the timing: pipeline.py catches exactly this
    # family to record the document as cancelled rather than failed.
    assert issubclass(LLMBridgePipelineCancelled, RuntimeError)
    # No parse thread left behind holding the pool slot.
    leftover = [
        t for t in _threading.enumerate() if t.name.startswith("native-md-download")
    ]
    assert leftover == []


def test_cancellation_is_not_swallowed_into_an_external_link(monkeypatch):
    """A cancel must abort the document, not degrade one image and carry on.

    LLMBridgeCancelled derives from RuntimeError, so without the explicit
    re-raise ahead of ``_resolve_remote``'s blanket ``except Exception`` this
    would quietly become an external-link fallback — and the parse would keep
    fetching the rest of the document after the user asked it to stop.
    """
    event = threading.Event()
    opens = {"n": 0}

    class _Opener:
        def open(self, req, timeout=None):
            opens["n"] += 1
            event.set()  # cancelled during the first fetch
            raise OSError("peer went away")

    monkeypatch.setattr(md_parser, "_host_is_public", lambda host: True)
    monkeypatch.setattr(md_parser, "_build_guarded_opener", lambda: _Opener())
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")

    md = "\n\n".join(f"![i{i}](http://host.example/x.png?u={i})" for i in range(3))
    parser = md_parser.NativeMarkdownParser()
    with pytest.raises(LLMBridgePipelineCancelled):
        parser._extract_text(
            md,
            bundle_root=None,
            cancel_events=((event, LLMBridgePipelineCancelled),),
        )
    # Stopped at the first image rather than working through all three.
    assert opens["n"] == 1


def test_cancellation_is_seen_on_repeated_references_to_one_url(monkeypatch):
    """The in-MEMORY memo must not outrun a cancel either.

    ``resolve()`` short-circuits on ``self._cache[src]``, so a document that
    references the same image thousands of times does I/O exactly once. A
    cancel check placed any deeper than ``resolve()`` never runs for the other
    references, and the document keeps going after the user stopped it. The
    disk-cache test below uses distinct URLs and does not cover this path.
    """
    event = threading.Event()
    resolved = {"n": 0}

    def _fake_download(self, src):
        resolved["n"] += 1
        # Cancelled DURING the one and only fetch, so every later reference is
        # served from the memo. Setting it before the document starts would be
        # caught by any check anywhere and would prove nothing.
        event.set()
        return (_PNG_BYTES, "png")

    monkeypatch.setattr(md_parser._MarkdownImageResolver, "_download", _fake_download)
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")

    # One URL, 50 references: exactly one fetch, then 49 memo hits.
    md = "\n\n".join("![x](http://host.example/same.png)" for _ in range(50))
    parser = md_parser.NativeMarkdownParser()
    with pytest.raises(LLMBridgePipelineCancelled):
        parser._extract_text(
            md,
            bundle_root=None,
            cancel_events=((event, LLMBridgePipelineCancelled),),
        )
    assert resolved["n"] == 1  # the memo hits stopped the document, not a refetch


def test_cancellation_is_seen_between_all_cache_hit_images(monkeypatch, tmp_path):
    """A document that does no blocking I/O must still observe a cancel.

    Every image being a cache hit means no socket is ever opened, so the
    in-fetch checkpoints never run; the poll at the top of _resolve_remote is
    what stops the document.
    """
    event = threading.Event()
    hits = {"n": 0}

    def _never_download(self, src):
        raise AssertionError("cache hit must not reach the network")

    class _Cache:
        def get(self, src):
            hits["n"] += 1
            if hits["n"] == 2:
                event.set()
            return (_PNG_BYTES, "png")

        def put(self, *a, **k):
            pass

    monkeypatch.setattr(md_parser._MarkdownImageResolver, "_download", _never_download)
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")

    md = "\n\n".join(f"![i{i}](http://host.example/x.png?u={i})" for i in range(5))
    parser = md_parser.NativeMarkdownParser()
    with pytest.raises(LLMBridgePipelineCancelled):
        parser._extract_text(
            md,
            bundle_root=None,
            raw_cache=_Cache(),
            cancel_events=((event, LLMBridgePipelineCancelled),),
        )
    assert hits["n"] == 2  # stopped on the image after the cancel landed


class _PngHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: A003
        pass

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(_PNG_BYTES)))
        self.end_headers()
        self.wfile.write(_PNG_BYTES)


@pytest.mark.parametrize("stalled_listener", ["trickle_body"], indirect=True)
def test_deadline_trips_on_a_trickled_body(download_env, stalled_listener):
    # The advisory's own proof of concept.
    _assert_bounded_fetch(
        f"http://127.0.0.1:{stalled_listener.port}/x.png", phase="body"
    )
