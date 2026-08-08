"""Native Markdown engine adapter (.md / .textpack).

Implements the :class:`NativeParserBase` hooks for markdown input. ``.textpack``
is a zipped TextBundle (a ``text.markdown`` plus an ``assets/`` resource dir;
Bear / Ulysses export format); a plain ``.md`` file is parsed directly.

Image handling (see the parser plan):

- base64 data-URL images are decoded and materialized into the sidecar assets.
- file-reference images are resolved ONLY inside a ``.textpack`` bundle, from
  the safely-extracted bundle directory (with a read-side path-traversal
  guard); a relative reference in a standalone ``.md`` is skipped + warned.
- external ``http(s)`` images are downloaded + embedded by default
  (``NATIVE_MD_IMAGE_DOWNLOAD_ENABLED`` defaults to ``true``), SSRF/size guarded;
  a drawing is always emitted — the fetched asset on success, or an external-link
  fallback on failure. Set the flag to ``false`` to instead DROP external images
  entirely (no drawing emitted, so a doc whose only images are external links
  produces no drawings.json).
- SVG images (base64 / textpack file / downloaded) are rasterized to PNG via
  cairosvg before entering the sidecar; if cairosvg is unavailable or rendering
  fails the image is skipped + warned.
"""

from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import http.client
import os
import re
import selectors
import socket
import string
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from ipaddress import ip_address, ip_network
from math import ceil
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote, urljoin, urlparse, urlunparse

from lightrag.constants import NATIVE_RAW_DIR_SUFFIX, PARSER_ENGINE_NATIVE
from lightrag.parser.external._common import raw_dir_for_parsed_dir
from lightrag.parser.llm_bridge import first_cancellation, normalize_cancel_events
from lightrag.parser.markdown.extract import ResolvedImage, extract_markdown
from lightrag.parser.markdown.raw_cache import (
    NativeImageRawCache,
    native_md_options_signature,
)
from lightrag.parser.native_base import NativeExtractRuntime, NativeParserBase
from lightrag.utils import logger

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc

# Zip-bomb guards for .textpack extraction.
_TEXTPACK_MAX_ENTRIES = 10_000
_TEXTPACK_MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB uncompressed

# Magic-byte signatures → file extension. Authoritative for download
# validation; ``None`` means "not a supported image".
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
_WEBP_RIFF = b"RIFF"
_WEBP_TAG = b"WEBP"

# Content-Type values that carry no usable type signal — fall back to magic.
_GENERIC_CONTENT_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


def _image_ext_from_magic(raw: bytes) -> str | None:
    """Return the file extension for ``raw`` raster image bytes, or ``None`` if
    the bytes are not a recognised raster image (unlike
    ``_vision_utils._detect_mime``, which always falls back to PNG). SVG is text,
    not a raster format, and is handled separately via :func:`_looks_like_svg`
    / :func:`_rasterize_svg`."""
    if raw.startswith(_PNG_SIGNATURE):
        return "png"
    if raw.startswith(_JPEG_SIGNATURE):
        return "jpg"
    if any(raw.startswith(sig) for sig in _GIF_SIGNATURES):
        return "gif"
    if len(raw) >= 12 and raw[0:4] == _WEBP_RIFF and raw[8:12] == _WEBP_TAG:
        return "webp"
    return None


def _looks_like_svg(raw: bytes) -> bool:
    """True iff ``raw`` sniffs as SVG markup (an ``<svg`` tag near the start,
    possibly after an XML declaration / DOCTYPE / comments)."""
    head = raw[:4096].lstrip(b"\xef\xbb\xbf").lstrip()  # drop UTF-8 BOM + space
    return b"<svg" in head[:2048].lower()


# CSS absolute-length units → pixels (96 px/in). Relative units (``%``, ``em``,
# ``ex``) depend on a viewport we do not have, so a dimension in those units is
# treated as unresolvable and falls back to the ``viewBox``.
_SVG_UNIT_TO_PX = {
    "": 1.0,
    "px": 1.0,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
    "in": 96.0,
    "cm": 96.0 / 2.54,
    "mm": 96.0 / 25.4,
    "q": 96.0 / 25.4 / 4.0,
}
_SVG_LENGTH_RE = re.compile(r"^([0-9]*\.?[0-9]+)\s*([a-z%]*)$")


def _svg_length_px(value: str) -> float | None:
    """Parse a CSS/SVG length to pixels, or ``None`` for an unresolvable unit
    (``%`` / ``em`` / ``ex``) or unparseable text."""
    match = _SVG_LENGTH_RE.match(value.strip().lower())
    if not match:
        return None
    factor = _SVG_UNIT_TO_PX.get(match.group(2))
    if factor is None:
        return None
    return float(match.group(1)) * factor


def _svg_pixel_dimensions(raw: bytes) -> tuple[int, int] | None:
    """Best-effort render dimensions (px) for an SVG, from ``width``/``height``
    or, failing that, the ``viewBox``. Returns ``None`` when the root is not an
    ``<svg>``, dimensions are missing/unresolvable, or the XML does not parse
    (parsed with defusedxml, which also blocks XML entity-expansion bombs)."""
    try:
        from defusedxml.ElementTree import fromstring

        root = fromstring(raw)
    except Exception as exc:  # noqa: BLE001 - malformed / hostile XML
        logger.debug("[native_md] SVG XML parse failed: %s", exc)
        return None
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        return None
    width = _svg_length_px(root.get("width") or "")
    height = _svg_length_px(root.get("height") or "")
    if width is None or height is None:
        view_box = (root.get("viewBox") or "").replace(",", " ").split()
        if len(view_box) != 4:
            return None
        try:
            width, height = float(view_box[2]), float(view_box[3])
        except ValueError:
            return None
    if width <= 0 or height <= 0:
        return None
    return ceil(width), ceil(height)


def _rasterize_svg(raw: bytes, *, max_pixels: int) -> bytes | None:
    """Render SVG bytes to PNG bytes via cairosvg. Returns ``None`` (caller
    skips the image) when cairosvg is unavailable, rendering fails, or the SVG's
    declared canvas exceeds ``max_pixels`` — the dimension check runs *before*
    rendering so a tiny SVG declaring a huge canvas cannot blow up memory/CPU
    inside cairosvg before the output-size cap would catch it. A single bad SVG
    must not abort the whole document."""
    dims = _svg_pixel_dimensions(raw)
    if dims is None:
        logger.warning("[native_md] SVG dimensions missing/unparseable; skipping")
        return None
    width, height = dims
    if width * height > max_pixels:
        logger.warning(
            "[native_md] SVG canvas %dx%d exceeds %d px budget; skipping",
            width,
            height,
            max_pixels,
        )
        return None
    try:
        import cairosvg
    except Exception as exc:  # noqa: BLE001 - optional native dep
        logger.warning(
            "[native_md] cairosvg unavailable, cannot rasterize SVG: %s", exc
        )
        return None
    try:
        return cairosvg.svg2png(bytestring=raw)
    except Exception as exc:  # noqa: BLE001 - malformed / hostile SVG
        logger.warning("[native_md] SVG rasterization failed: %s", exc)
        return None


_SVG_RASTERIZER_PROBE = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
)


def check_svg_rasterizer() -> str | None:
    """Startup probe for the cairosvg SVG->PNG rasterization path (advisory only).

    ``cairosvg`` is a cffi binding: ``pip install cairosvg`` always succeeds,
    but rendering only works if the native ``libcairo`` shared library is
    *also* present on the host — pip/uv cannot install system libraries, so a
    missing ``libcairo2`` (Debian/Ubuntu) / ``cairo`` (RHEL/Fedora/Homebrew)
    package would otherwise only surface as a per-document warning the first
    time a user uploads a file with an embedded SVG. Rendering a trivial probe
    SVG here surfaces the same gap once, at startup, instead.

    Returns ``None`` when rasterization works, otherwise a short human-readable
    diagnostic for the caller to surface (e.g. in the startup splash screen).
    The underlying exception is logged at debug level only — cairocffi's
    library-not-found error repeats the lookup across every platform's naming
    convention (``.so``, ``.dylib``, ``.dll``) and is too noisy for a one-line
    warning. Never raises — :func:`_rasterize_svg` already degrades gracefully
    per image.
    """
    try:
        import cairosvg
    except Exception as exc:  # noqa: BLE001 - optional native dep
        logger.debug("[native_md] cairosvg import failed: %s", exc)
        return "cairosvg is not installed; SVG images will be skipped."
    try:
        cairosvg.svg2png(bytestring=_SVG_RASTERIZER_PROBE)
    except Exception as exc:  # noqa: BLE001 - missing libcairo or similar
        logger.debug("[native_md] cairosvg rasterization probe failed: %s", exc)
        return (
            "cairosvg is installed but the native libcairo library was not "
            "found; SVG images will be skipped."
        )
    return None


def _image_bytes_and_ext(
    raw: bytes, *, max_bytes: int, max_svg_pixels: int
) -> tuple[bytes, str] | None:
    """Coerce ``raw`` image bytes into a sidecar-ready raster image.

    A recognised raster image passes through unchanged; an SVG is rasterized to
    PNG — bounded *before* rendering by ``max_svg_pixels`` (declared canvas) and
    *after* by ``max_bytes`` (output size), so a hostile SVG that explodes into a
    giant bitmap is rejected rather than embedded. Returns ``(bytes, ext)`` or
    ``None`` when the bytes are neither a raster image nor a convertible SVG.
    """
    ext = _image_ext_from_magic(raw)
    if ext is not None:
        return raw, ext
    if _looks_like_svg(raw):
        png = _rasterize_svg(raw, max_pixels=max_svg_pixels)
        if png is not None and len(png) <= max_bytes and _image_ext_from_magic(png):
            return png, "png"
    return None


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "t", "y")


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Per-fetch download context: wall-clock deadline, cancellation, DNS memo and
# the socket watchdog. One instance is active per thread for the duration of
# one _download() call (redirect chain included).
# --------------------------------------------------------------------------

# Indirected so tests can install a fake clock with
# ``monkeypatch.setattr(md_parser, "_monotonic", fake)``. Patching
# ``time.monotonic`` itself would be a process-wide change.
_monotonic = time.monotonic

_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_CONNECT_POLL_SECONDS = 0.2
_WATCHDOG_POLL_SECONDS = 0.5

_ACTIVE_DOWNLOAD = threading.local()


class _DownloadState:
    """Mutable per-fetch state shared between the fetch and its watchdog."""

    def __init__(
        self,
        *,
        deadline: float | None,
        cancel_events: tuple,
    ) -> None:
        self.deadline = deadline
        self.cancel_events = normalize_cancel_events(cancel_events)
        self.dns: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._sock = None
        self._watchdog_reason: BaseException | None = None

    # -- cancellation / deadline ------------------------------------------

    def cancellation(self, message: str) -> BaseException | None:
        return first_cancellation(self.cancel_events, message)

    def expired(self) -> bool:
        return self.deadline is not None and _monotonic() >= self.deadline

    def trip_reason(self) -> BaseException | None:
        """The exception this fetch should die with, or ``None`` to continue.

        Evaluated SYNCHRONOUSLY, in the calling thread, in a fixed priority
        order — the watchdog's own finding is only a fallback. The watchdog
        polls on an interval, so relying on it would misclassify every race it
        loses: a socket timeout or a peer EOF that arrives before the next poll
        would surface as an ordinary download failure, and a cancellation
        followed by an EOF would be read as a *successful* truncated image.
        """
        cancelled = self.cancellation("native markdown image download cancelled")
        if cancelled is not None:
            return cancelled
        if self.expired():
            return TimeoutError("image download exceeded its deadline")
        return self._watchdog_reason

    def raise_trip_reason(self) -> None:
        reason = self.trip_reason()
        if reason is not None:
            raise reason

    # -- socket registration / watchdog ------------------------------------

    def register_socket(self, sock) -> None:
        """Hand a CONNECTED socket to the watchdog.

        Never register a socket that is still connecting: aborting a connect
        is the calling thread's job (see :func:`_pin_socket`).
        """
        with self._lock:
            self._sock = sock
            already_tripped = self._watchdog_reason is not None
        if already_tripped:
            self._shutdown_socket()

    def _shutdown_socket(self) -> None:
        with self._lock:
            sock = self._sock
        if sock is None:
            return
        try:
            # shutdown() only — never close(). Closing an fd another thread is
            # still using invites fd reuse; a half-close is enough to wake a
            # blocked recv (it returns b'', which is why trip_reason() is also
            # consulted after a clean read loop).
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def watchdog_tick(self) -> bool:
        """One watchdog poll. Returns True once it has tripped."""
        if self._watchdog_reason is not None:
            return True
        reason = self.cancellation("native markdown image download cancelled")
        if reason is None and self.expired():
            reason = TimeoutError("image download exceeded its deadline")
        if reason is None:
            return False
        self._watchdog_reason = reason
        self._shutdown_socket()
        return True


def _active_download() -> _DownloadState | None:
    return getattr(_ACTIVE_DOWNLOAD, "state", None)


def _checkpoint() -> None:
    """Raise if the active fetch has been cancelled or has run out of time."""
    state = _active_download()
    if state is not None:
        state.raise_trip_reason()


def _deadline() -> float | None:
    state = _active_download()
    return None if state is None else state.deadline


def _effective_timeout(timeout: float | None) -> float | None:
    """Clamp a socket timeout to the remaining fetch budget."""
    deadline = _deadline()
    if deadline is None:
        return timeout
    remaining = max(0.001, deadline - _monotonic())
    return remaining if timeout is None else min(timeout, remaining)


def _register_download_socket(sock) -> None:
    state = _active_download()
    if state is not None:
        state.register_socket(sock)


@contextmanager
def _download_context(
    *,
    deadline: float | None,
    cancel_events: tuple = (),
    poll_interval: float = _WATCHDOG_POLL_SECONDS,
):
    """Activate a download context (and its watchdog) for one fetch.

    MUST wrap the whole of :meth:`_MarkdownImageResolver._download`, starting
    before the first ``_host_is_public`` call. Wrapping only ``opener.open()``
    would leave that first DNS resolve outside the context: no checkpoints, no
    shared memo, so the resolve happens twice and a cancellation during it is
    reported as a download failure.
    """
    state = _DownloadState(deadline=deadline, cancel_events=cancel_events)
    previous = getattr(_ACTIVE_DOWNLOAD, "state", None)
    _ACTIVE_DOWNLOAD.state = state
    stop = threading.Event()

    def _watch() -> None:
        while not stop.wait(poll_interval):
            if state.watchdog_tick():
                return

    thread = threading.Thread(
        target=_watch, name="native-md-download-watchdog", daemon=True
    )
    thread.start()
    try:
        yield state
    finally:
        stop.set()
        thread.join(timeout=poll_interval * 2)
        _ACTIVE_DOWNLOAD.state = previous


def _allowed_non_public_networks() -> list:
    """Parse ``NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS`` (comma-separated
    CIDRs / IPs) into networks. Invalid tokens are warned and dropped."""
    raw = os.getenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "")
    nets = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            nets.append(ip_network(token, strict=False))
        except ValueError:
            logger.warning("[native_md] ignoring invalid allowed CIDR: %s", token)
    return nets


# IPv6 blocks that older CPython (pre gh-113171) classified as globally routable
# but that are not, and whose embedded-IPv4 position we must NOT trust:
#   - ``64:ff9b:1::/48`` — RFC 8215 local-use NAT64. Technology-agnostic; the
#     embedded-IPv4 position is not guaranteed (RFC 8215 §5), so a fixed-offset
#     decode is itself bypassable. IANA marks the whole /48 not-globally-reachable.
#   - ``2002::/16`` — 6to4. IANA marks its global-reachability N/A and CPython
#     3.13+ / the gh-113171 backports treat the whole block as private, so a
#     decode of the embedded IPv4 must not re-permit it. (RFC 7526 deprecated
#     only the 6to4 *anycast relay* path, 192.88.99.0/24; basic unicast 6to4 and
#     the 2002::/16 prefix itself are not deprecated — but the block is still not
#     globally reachable.)
# Both are default-denied as whole blocks, independent of the interpreter, so we
# never decode past them. An operator who genuinely runs a local NAT64 / 6to4
# there can still opt in via ``NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS``.
_FORCE_NON_GLOBAL_V6 = (ip_network("64:ff9b:1::/48"), ip_network("2002::/16"))


def _unwrap_embedded_ipv4(ip):
    """Return the IPv4 embedded in an IPv6 transition wrapper so it is judged by
    IPv4 rules; otherwise return ``ip`` unchanged.

    Some IPv6 wrappers embed an internal IPv4 yet the *literal* is classified
    globally routable, so a bare ``is_global`` check on the literal passes them.
    On a host with NAT64/DNS64 routing the request is then delivered to the
    embedded internal target (loopback / RFC1918 / cloud metadata). We therefore
    decode the embedded IPv4 so the caller can *additionally* require it to be
    global (see :func:`_is_globally_routable`).

    Only wrappers whose embedded-IPv4 position is fixed by spec are decoded:
    IPv4-mapped (``::ffff:0:0/96``), IPv4-compatible (``::/96``, minus
    ``::``/``::1``), and the NAT64 well-known prefix (``64:ff9b::/96``, RFC 6052
    §2.2 — IPv4 in the low 32 bits). 6to4 (``2002::/16``) and the RFC 8215
    local-use NAT64 prefix (``64:ff9b:1::/48``) are deliberately NOT decoded
    here — they are default-denied as whole blocks via
    :data:`_FORCE_NON_GLOBAL_V6` (the /48's embedded-IPv4 position is not
    guaranteed, and both are non-globally-reachable per IANA).

    Boundary: a NAT64 deployment using a *custom*, globally-routable Network-
    Specific Prefix (RFC 6052 permits any /32../96) is not recognised here — the
    prefix is deployment-specific and indistinguishable from ordinary global
    IPv6. Such operators should pin the download egress or use
    ``NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS`` deliberately.
    """
    if ip.version != 6:
        return ip
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    b = ip.packed
    # IPv4-compatible ``::a.b.c.d`` (top 96 bits zero), excluding ``::`` and
    # ``::1`` which are not IPv4 wrappers (and are non-global anyway).
    if b[:12] == b"\x00" * 12 and b[12:] not in (
        b"\x00\x00\x00\x00",
        b"\x00\x00\x00\x01",
    ):
        return ip_address(b[12:])
    # NAT64 well-known prefix ``64:ff9b::/96`` — IPv4 in the low 32 bits.
    if b[:12] == b"\x00\x64\xff\x9b" + b"\x00" * 8:
        return ip_address(b[12:])
    return ip


def _is_globally_routable(ip) -> bool:
    """True iff ``ip`` is safe to fetch (a globally routable public address).

    Starts from the stdlib's ``is_global`` verdict on the literal address and
    only ever *tightens* it — the guard is never more permissive than
    ``ipaddress`` itself. On top of that: force-denied blocks
    (:data:`_FORCE_NON_GLOBAL_V6`) are always rejected, and a fixed-position
    IPv6 transition wrapper (:func:`_unwrap_embedded_ipv4`) whose embedded IPv4
    is not itself global is rejected — so a NAT64 / IPv4-compatible literal that
    wraps an internal IPv4 (e.g. ``64:ff9b::7f00:1``) is blocked even though the
    literal is globally routable, while a NAT64 wrapper of a public IPv4
    (``64:ff9b::808:808`` = 8.8.8.8) still passes.
    """
    if not ip.is_global:
        return False
    if ip.version == 6:
        if any(ip in net for net in _FORCE_NON_GLOBAL_V6):
            return False
        if not _unwrap_embedded_ipv4(ip).is_global:
            return False
    return True


def _validated_addresses(host: str) -> list[str]:
    """Resolve ``host`` and return its addresses iff ALL are safe to fetch.

    Default-deny: an address is accepted only when it is globally routable (so
    SSRF to loopback / private / link-local / reserved / CGNAT ``100.64.0.0/10``
    / TEST-NET and any other non-globally-routable range is blocked) or it
    matches the operator-configured ``NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS``
    escape hatch. Global-routability is decided by :func:`_is_globally_routable`,
    which judges the IPv4 embedded in any IPv6 transition wrapper so a NAT64 /
    6to4 / IPv4-compatible literal that wraps an internal IPv4 cannot smuggle
    past the gate. Returns ``[]`` when resolution fails or *any* resolved
    address is non-public — so a single poisoned A/AAAA record rejects the whole
    host.

    Call it through :func:`_resolve_shared`, never directly: that wrapper is
    what makes "validating and connecting share one resolution" actually true
    (it memoises per host for the duration of one fetch) and what surrounds the
    uninterruptible resolve with cancellation checkpoints.
    """
    allow = _allowed_non_public_networks()
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    addrs: list[str] = []
    for info in infos:
        addr = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ip_address(addr)
        except ValueError:
            return []
        if not (_is_globally_routable(ip) or any(ip in net for net in allow)):
            return []
        addrs.append(addr)
    return addrs


def _resolve_shared(host: str) -> list[str]:
    """:func:`_validated_addresses`, memoised per host for one fetch.

    Two jobs, both load-bearing:

    * **One resolve per host per fetch.** ``_host_is_public`` and
      :func:`_pin_socket` each used to resolve independently (and a redirect
      hop doubled that again), which doubled the one phase of a fetch that
      can be neither deadline-bounded nor cancelled.
    * **Checkpoints around the resolve.** This is the only place a fetch
      blocks in the system resolver, so a cancellation raised during it can
      only be observed here. Putting the checkpoints in ``_pin_socket``
      instead would miss the common case entirely: when the resolve yields
      nothing usable, ``_host_is_public`` returns False and the caller raises
      before ``_pin_socket`` is ever reached, so the cancellation would be
      swallowed and reported as an ordinary download failure.

    Membership, not truthiness, decides a memo hit: ``[]`` is a *result*
    (resolution failed, or an address was non-public), not a miss.
    """
    state = _active_download()
    if state is None:
        return _validated_addresses(host)
    _checkpoint()
    if host not in state.dns:
        state.dns[host] = _validated_addresses(host)
    _checkpoint()
    return state.dns[host]


def _host_is_public(host: str) -> bool:
    """True iff every resolved address for ``host`` is safe to fetch."""
    return bool(_resolve_shared(host))


def _timeout_seconds(timeout) -> float | None:
    """Normalise a socket-style timeout argument to seconds, or ``None``."""
    if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
        timeout = socket.getdefaulttimeout()
    if timeout is None:
        return None
    return float(timeout)


def _pin_socket(host: str, port: int, timeout, source_address):
    """Connect to a *validated* resolved address of ``host``, cancellably.

    The address comes from the same :func:`_resolve_shared` resolution that
    authorised the host, so urllib never independently re-resolves the name
    (which a DNS-rebinding attacker could flip to an internal IP between our
    check and the actual connect).

    The connect is non-blocking + selector-polled rather than a blocking
    ``create_connection``, so it observes both the fetch deadline and a
    cancellation within one poll interval. It must be done this way and not
    with the socket watchdog: a ``shutdown()`` on a socket that is still
    connecting does not abort the connect — on macOS it makes ``connect()``
    return *successfully* — so a mid-connect abort has to be driven by the
    calling thread itself.
    """
    addrs = _resolve_shared(host)
    if not addrs:
        raise OSError(f"refusing connection to non-public host: {host!r}")
    addr = addrs[0]
    family = socket.AF_INET6 if ":" in addr else socket.AF_INET
    timeout_s = _timeout_seconds(timeout)

    # A connect deadline of its own, not just the fetch deadline: without an
    # active download context there is no fetch deadline, and the timeout
    # argument would otherwise not be honoured until AFTER the connect
    # completes — an EINPROGRESS peer would poll forever, where
    # create_connection(timeout=...) used to give up.
    candidates = [
        d
        for d in (_deadline(), None if timeout_s is None else _monotonic() + timeout_s)
        if d is not None
    ]
    connect_deadline = min(candidates) if candidates else None

    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if source_address:
            sock.bind(source_address)
        sock.setblocking(False)
        err = sock.connect_ex((addr, port))
        if err not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK):
            raise OSError(err, os.strerror(err))
        if err:
            with selectors.DefaultSelector() as sel:
                sel.register(sock, selectors.EVENT_WRITE)
                while True:
                    _checkpoint()
                    wait = _CONNECT_POLL_SECONDS
                    if connect_deadline is not None:
                        remaining = connect_deadline - _monotonic()
                        if remaining <= 0:
                            raise TimeoutError("image download connect timed out")
                        wait = min(wait, remaining)
                    if sel.select(wait):
                        # Re-check before accepting: a cancellation that landed
                        # while we were blocked in select() must not be
                        # overtaken by the connection completing.
                        _checkpoint()
                        code = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                        if code:
                            raise OSError(code, os.strerror(code))
                        break
        _checkpoint()
        sock.setblocking(True)
        sock.settimeout(_effective_timeout(timeout_s))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except BaseException:
        # Aborts during connect are closed by THIS thread; the watchdog is
        # deliberately not involved (see the docstring).
        sock.close()
        raise
    _register_download_socket(sock)
    return sock


# No proxy/CONNECT-tunnel handling below: the opener disables proxies, so the
# connection always dials the origin directly (``_tunnel_host`` is never set).
class _GuardedHTTPConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = _pin_socket(self.host, self.port, self.timeout, self.source_address)


class _GuardedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        sock = _pin_socket(self.host, self.port, self.timeout, self.source_address)
        # Wrap with the original hostname so SNI / certificate validation still
        # runs against the domain, not the pinned IP.
        #
        # The handshake is driven explicitly rather than by wrap_socket,
        # because wrap_socket DETACHES the socket it wraps (the original
        # object's fd becomes -1). Handshaking inside it would therefore run
        # with only the detached object registered, and the watchdog would have
        # nothing it could shut down — a peer that stalls mid-handshake would
        # hold the worker indefinitely. Registering the SSLSocket first makes
        # the handshake interruptible like every later read.
        sslsock = self._context.wrap_socket(
            sock, server_hostname=self.host, do_handshake_on_connect=False
        )
        _register_download_socket(sslsock)
        sslsock.do_handshake()
        self.sock = sslsock


class _GuardedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_GuardedHTTPConnection, req)


class _GuardedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        # Mirror the stdlib handler's own arguments, which differ across
        # versions: Python <3.12 stores ``_check_hostname`` on the handler and
        # forwards it; 3.12+ folds it into ``_context`` and drops the attribute.
        # Forwarding a non-existent ``_check_hostname`` raised AttributeError on
        # 3.12, silently failing every download into the external-link fallback.
        kwargs = {"context": self._context}
        if hasattr(self, "_check_hostname"):
            kwargs["check_hostname"] = self._check_hostname
        return self.do_open(_GuardedHTTPSConnection, req, **kwargs)


# Redirect schemes we are willing to follow. The stdlib allows ``ftp`` here;
# an ftp:// redirect would be served by urllib's FTPHandler, which never goes
# through _GuardedHTTPConnection and so escapes the IP pin, the download
# deadline, cancellation and the request budget. Empty scheme is a relative
# redirect, resolved by urljoin against the (http/https) request URL.
_REDIRECT_ALLOWED_SCHEMES = ("http", "https", "")


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that re-validates the host and never drains the body.

    Two departures from :class:`urllib.request.HTTPRedirectHandler`:

    * ``redirect_request`` narrows the scheme allowlist to http(s) — see
      :data:`_REDIRECT_ALLOWED_SCHEMES`.
    * :meth:`http_error_302` closes the redirect response instead of calling
      ``fp.read()``. The stdlib reads the whole body before following the
      redirect (so it can hand a live ``fp`` to ``HTTPError``), which is an
      unbounded allocation an attacker controls: ``302`` + a huge body lands
      in memory before the image itself ever reaches the size-capped chunked
      read. We give up the readable ``fp`` on the error path — the loop /
      max-redirection errors below are raised with the body already closed —
      and keep the residency bounded instead.

    Host validation is defense in depth: the pinned connection re-validates at
    connect time too (and shares that resolution, see ``_validated_addresses``).
    """

    # Tighter than the stdlib default of 10: one image reference should not be
    # able to turn into a double-digit chain of outbound requests.
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        host = urlparse(newurl).hostname or ""
        if not _host_is_public(host):
            raise urllib.error.HTTPError(
                newurl, code, "redirect to non-public host blocked", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_302(self, req, fp, code, msg, headers):  # type: ignore[override]
        # Mirrors the stdlib implementation up to the point where it drains the
        # body. Kept deliberately close to it so future stdlib changes are easy
        # to diff against.
        if "location" in headers:
            newurl = headers["location"]
        elif "uri" in headers:
            newurl = headers["uri"]
        else:
            return None

        urlparts = urlparse(newurl)
        if urlparts.scheme not in _REDIRECT_ALLOWED_SCHEMES:
            fp.close()
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"{msg} - redirection to scheme {urlparts.scheme!r} is not allowed",
                headers,
                None,
            )
        if not urlparts.path and urlparts.netloc:
            parts = list(urlparts)
            parts[2] = "/"
            newurl = urlunparse(parts)
        else:
            newurl = urlunparse(urlparts)
        # http.client.parse_headers() decodes as ISO-8859-1. Recover the
        # original bytes and percent-encode non-ASCII bytes and specials.
        newurl = quote(newurl, encoding="iso-8859-1", safe=string.punctuation)
        newurl = urljoin(req.full_url, newurl)

        new = self.redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None

        # Loop detection (unchanged semantics).
        if hasattr(req, "redirect_dict"):
            visited = new.redirect_dict = req.redirect_dict
            if (
                visited.get(newurl, 0) >= self.max_repeats
                or len(visited) >= self.max_redirections
            ):
                fp.close()
                raise urllib.error.HTTPError(
                    req.full_url, code, self.inf_msg + msg, headers, None
                )
        else:
            visited = new.redirect_dict = req.redirect_dict = {}
        visited[newurl] = visited.get(newurl, 0) + 1

        # THE fix: discard the redirect body instead of ``fp.read()``-ing it.
        fp.close()
        return self.parent.open(new, timeout=req.timeout)

    # The base class assigns these as five independent attributes pointing at
    # its own http_error_302, so overriding only http_error_302 in a subclass
    # leaves 301/303/307/308 resolving to the ORIGINAL draining implementation.
    # They must be rebound here explicitly.
    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _build_guarded_opener() -> urllib.request.OpenerDirector:
    """Opener that speaks http(s) only, pins connections to a validated IP, and
    ignores any ambient ``HTTP(S)_PROXY`` (an env proxy would otherwise fetch
    the blocked URL on our behalf, bypassing the IP guard).

    Assembled explicitly rather than via ``build_opener()``, which additionally
    installs ``FTPHandler`` / ``FileHandler`` / ``DataHandler``. Those handlers
    bypass every control in this module, so the opener simply does not carry
    them; ``UnknownHandler`` turns any non-http(s) URL into a ``URLError``.
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler({}),
        _GuardedHTTPHandler(),
        _GuardedHTTPSHandler(),
        _GuardedRedirectHandler(),
        urllib.request.HTTPErrorProcessor(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.UnknownHandler(),
    ):
        opener.add_handler(handler)
    return opener


class _MarkdownImageResolver:
    """Concrete :class:`~lightrag.parser.markdown.extract.ImageResolver`.

    Caches by ``src`` so a repeated reference downloads / decodes once and
    every occurrence shares one on-disk asset.
    """

    def __init__(
        self,
        *,
        bundle_root: Path | None,
        warnings: dict[str, int],
        download_enabled: bool,
        download_required: bool,
        timeout: int,
        max_bytes: int,
        max_svg_pixels: int,
        raw_cache: NativeImageRawCache | None = None,
        cancel_events: tuple = (),
    ) -> None:
        self._bundle_root = bundle_root.resolve() if bundle_root else None
        self._warnings = warnings
        self._download_enabled = download_enabled
        self._download_required = download_required
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_svg_pixels = max_svg_pixels
        self._raw_cache = raw_cache
        self._cancel_events = cancel_events
        self._cache: dict[str, ResolvedImage] = {}

    def resolve(self, src: str) -> ResolvedImage:
        cached = self._cache.get(src)
        if cached is not None:
            return cached
        result = self._resolve_uncached(src)
        self._cache[src] = result
        return result

    def _bump(self, key: str) -> None:
        self._warnings[key] = self._warnings.get(key, 0) + 1

    def _skip(self, reason: str, src: str) -> ResolvedImage:
        logger.warning("[native_md] skipping image (%s): %s", reason, src[:120])
        self._bump("images_skipped")
        return ResolvedImage(kind="skip")

    def _local(self, data: bytes, ext: str, suggested_name: str) -> ResolvedImage:
        asset_ref = "sha256:" + hashlib.sha256(data).hexdigest()
        return ResolvedImage(
            kind="local",
            asset_ref=asset_ref,
            data=data,
            suggested_name=suggested_name,
            fmt=ext,
        )

    def _resolve_uncached(self, src: str) -> ResolvedImage:
        lower = src.lower()
        if lower.startswith("data:"):
            return self._resolve_data_url(src)
        if lower.startswith(("http://", "https://")):
            return self._resolve_remote(src)
        return self._resolve_relative(src)

    def _resolve_data_url(self, src: str) -> ResolvedImage:
        # ``data:[<mime>][;base64],<payload>`` — only base64 image payloads.
        if ";base64," not in src:
            return self._skip("data url not base64", src)
        _, _, payload = src.partition(";base64,")
        try:
            data = base64.b64decode("".join(payload.split()), validate=True)
        except (ValueError, binascii.Error):
            return self._skip("invalid base64", src)
        # Magic bytes are authoritative — the declared MIME type is not trusted
        # for validation (matching the remote-download path). SVG is rasterized
        # to PNG here.
        coerced = _image_bytes_and_ext(
            data, max_bytes=self._max_bytes, max_svg_pixels=self._max_svg_pixels
        )
        if coerced is None:
            return self._skip("unrecognised image", src)
        data, ext = coerced
        digest = hashlib.sha256(data).hexdigest()[:12]
        return self._local(data, ext, f"image-{digest}.{ext}")

    def _resolve_relative(self, src: str) -> ResolvedImage:
        if self._bundle_root is None:
            # Standalone .md: file references are unresolved by design.
            return self._skip("file reference outside .textpack", src)
        rel = unquote(src.split("#", 1)[0].split("?", 1)[0])
        if not rel or "\\" in rel or rel.startswith("/") or os.path.isabs(rel):
            return self._skip("unsafe image path", src)
        if any(part == ".." for part in Path(rel).parts):
            return self._skip("unsafe image path", src)
        candidate = (self._bundle_root / rel).resolve()
        if not candidate.is_relative_to(self._bundle_root):
            return self._skip("image path escapes bundle", src)
        if not candidate.is_file():
            return self._skip("image file missing", src)
        # Cap a single bundled asset at the same ceiling as a remote download,
        # so one oversized file inside the (zip-bomb-bounded) bundle cannot pull
        # hundreds of MB into memory.
        if candidate.stat().st_size > self._max_bytes:
            return self._skip("image exceeds size limit", src)
        data = candidate.read_bytes()
        # Magic bytes are authoritative; the filename suffix is not trusted for
        # validation. SVG is rasterized to PNG (so the on-disk name takes the
        # resolved extension, e.g. ``logo.svg`` -> ``logo.png``).
        coerced = _image_bytes_and_ext(
            data, max_bytes=self._max_bytes, max_svg_pixels=self._max_svg_pixels
        )
        if coerced is None:
            return self._skip("unrecognised image", src)
        data, ext = coerced
        return self._local(data, ext, f"{candidate.stem}.{ext}")

    def _resolve_remote(self, src: str) -> ResolvedImage:
        ext_hint = Path(urlparse(src).path).suffix.lower().lstrip(".")
        if not self._download_enabled:
            # Downloading is opt-in: with it disabled, external images are
            # dropped entirely (no drawing emitted), so a doc whose only images
            # are external links produces no drawings.json. This is expected
            # configuration, not a problem, so it is logged at debug level and
            # counted under a dedicated key rather than warned per image.
            logger.debug(
                "[native_md] dropping external image (download disabled): %s", src[:120]
            )
            self._bump("images_external_dropped")
            return ResolvedImage(kind="skip")
        if self._raw_cache is not None:
            hit = self._raw_cache.get(src)
            if hit is not None:
                # Reuse the cached bytes (already post-SVG-rasterization), so a
                # re-parse skips both the network fetch and the rasterization.
                data, ext = hit
                self._bump("images_cache_hit")
                digest = hashlib.sha256(data).hexdigest()[:12]
                return self._local(data, ext, f"image-{digest}.{ext}")
        try:
            data, ext = self._download(src)
        except Exception as exc:  # noqa: BLE001 - best-effort network fetch
            if self._download_required:
                raise
            logger.warning("[native_md] image download failed (%s): %s", exc, src[:120])
            self._bump("images_download_failed")
            return ResolvedImage(kind="external", url=src, fmt=ext_hint)
        if self._raw_cache is not None:
            self._raw_cache.put(src, data, ext)
        digest = hashlib.sha256(data).hexdigest()[:12]
        return self._local(data, ext, f"image-{digest}.{ext}")

    def _download(self, src: str) -> tuple[bytes, str]:
        """Fetch one image, bounded by a real wall-clock deadline.

        ``NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT`` is applied as a deadline for the
        whole request, not merely as urllib's per-socket-operation timeout: a
        peer that emits one byte per interval resets a socket timeout forever,
        which at the shipped defaults kept a parse worker for years.

        What each phase can and cannot do:

        ==================  =================  =====================
        phase               bounded by         interruptible by cancel
        ==================  =================  =====================
        DNS resolution      system resolver    no
        TCP connect         deadline           yes (poll interval)
        TLS handshake       deadline           yes (watchdog)
        response headers    deadline           yes (watchdog)
        response body       deadline           yes (watchdog)
        redirect hop        deadline           yes (watchdog)
        ==================  =================  =====================

        DNS is the sole exception on both counts: it happens before any socket
        exists, so neither the deadline nor the watchdog can reach it. The
        bound is therefore "deadline plus one DNS resolution", and a cancel is
        observed at worst one resolution plus one poll interval late.

        The signature is fixed by tests that monkeypatch it; everything the
        fetch needs comes off ``self``.
        """
        deadline = _monotonic() + self._timeout
        with _download_context(
            deadline=deadline, cancel_events=self._cancel_events
        ) as guard:
            data, content_type = self._fetch(src, guard)
        if len(data) > self._max_bytes:
            raise ValueError(f"image exceeds {self._max_bytes} bytes")
        # Magic bytes are authoritative; SVG is rasterized to PNG here. Reject
        # only an *explicit* non-image Content-Type — but ``image/svg+xml`` is a
        # valid image type, so the check stays correct for converted SVGs.
        if (
            content_type
            and content_type not in _GENERIC_CONTENT_TYPES
            and not content_type.startswith("image/")
        ):
            raise ValueError(f"non-image Content-Type: {content_type!r}")
        coerced = _image_bytes_and_ext(
            data, max_bytes=self._max_bytes, max_svg_pixels=self._max_svg_pixels
        )
        if coerced is None:
            raise ValueError("downloaded bytes are not a supported image")
        return coerced

    def _fetch(self, src: str, guard: _DownloadState) -> tuple[bytes, str]:
        """The network half of :meth:`_download`, inside an active context."""
        parsed = urlparse(src)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"unsupported scheme: {parsed.scheme!r}")
        # Inside the context, so this resolve is checkpointed and memoised for
        # the connect that follows.
        if not _host_is_public(parsed.hostname or ""):
            raise ValueError("non-public host blocked")
        opener = _build_guarded_opener()
        req = urllib.request.Request(src, headers={"User-Agent": "lightrag-native-md"})
        cap = self._max_bytes
        try:
            with opener.open(req, timeout=self._timeout) as resp:
                content_type = (resp.headers.get_content_type() or "").lower()
                chunks: list[bytes] = []
                total = 0
                # Read at most cap + 1 so the over-limit check below still sees
                # one byte past the ceiling, exactly as the single read did.
                while total <= cap:
                    _checkpoint()
                    want = min(_DOWNLOAD_CHUNK_BYTES, cap + 1 - total)
                    chunk = resp.read(want)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                data = b"".join(chunks)
        except BaseException:
            # A watchdog-induced teardown surfaces as a bare OSError, which the
            # caller's blanket handler would silently degrade to an external
            # link — including for a cancellation. Re-label it first.
            guard.raise_trip_reason()
            raise
        # Also on the clean path: a half-closed socket makes read() return b'',
        # which looks exactly like a complete body. Without this, an aborted
        # fetch would be accepted as a (truncated) image.
        guard.raise_trip_reason()
        return data, content_type


class NativeMarkdownParser(NativeParserBase):
    engine_name = PARSER_ENGINE_NATIVE
    empty_content_label = "Markdown"

    def validate_source(self, source: Path, file_path: str) -> None:
        if not (
            source.exists()
            and source.is_file()
            and source.suffix.lower() in (".md", ".textpack")
        ):
            raise ValueError(
                f"Native markdown parser does not support pending file: {file_path}"
            )

    def extract(
        self,
        source: Path,
        *,
        parsed_dir: Path,
        asset_dir: Path,
        base_name: str,
        runtime: NativeExtractRuntime | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        # Engine params (e.g. smart_heading) only apply to the docx path — md
        # lacks the font-size signals the algorithm needs. Warn-and-ignore so a
        # rule like ``LIGHTRAG_PARSER=native(smart_heading=true)`` doesn't hard
        # fail every markdown document it happens to route. Only TRUTHY params
        # count as "ignored": a blanket opt-out (smart_heading=false) turned
        # nothing on, so it should not warn on every md/textpack document.
        ignored_params = (
            sorted(k for k, v in runtime.engine_params.items() if v) if runtime else []
        )
        # Per-document downloaded-image cache. Lives in a ``<file>.native_raw/``
        # sibling of ``parsed_dir`` so it survives the ``rmtree(parsed_dir)`` the
        # base parser runs before each re-extraction; reused across re-parses
        # unless the source content or download config changed.
        raw_cache = NativeImageRawCache(
            raw_dir_for_parsed_dir(parsed_dir, suffix=NATIVE_RAW_DIR_SUFFIX),
            source_path=source,
            options_signature=native_md_options_signature(),
            force_reparse=_env_bool("LIGHTRAG_FORCE_REPARSE_NATIVE", False),
        )
        raw_cache.load()
        if source.suffix.lower() == ".textpack":
            tmp_dir = Path(tempfile.mkdtemp(prefix="textpack-"))
            try:
                md_text, bundle_root = self._open_textpack(source, tmp_dir)
                result = self._extract_text(
                    md_text, bundle_root=bundle_root, raw_cache=raw_cache
                )
            finally:
                rmtree(tmp_dir, ignore_errors=True)
        else:
            md_text = source.read_bytes().decode("utf-8-sig")
            result = self._extract_text(md_text, bundle_root=None, raw_cache=raw_cache)
        # Flush only on successful extraction so a transient failure cannot prune
        # a previously-valid bundle (an exception propagates before this line).
        raw_cache.flush()
        if ignored_params:
            logger.warning(
                "[native_md] engine params %s only apply to .docx; ignored for %s",
                ignored_params,
                source.name,
            )
            result[1]["engine_params_ignored"] = 1
        return result

    def _open_textpack(self, source: Path, tmp_dir: Path) -> tuple[str, Path]:
        from lightrag.parser.external._zip import safe_extract_zip

        safe_extract_zip(
            source.read_bytes(),
            tmp_dir,
            max_entries=_TEXTPACK_MAX_ENTRIES,
            max_total_bytes=_TEXTPACK_MAX_TOTAL_BYTES,
        )
        text_file = self._find_text_file(tmp_dir, source.name)
        return text_file.read_bytes().decode("utf-8-sig"), text_file.parent

    @staticmethod
    def _find_text_file(root: Path, source_name: str) -> Path:
        # The body is located by extension, not a fixed ``text.markdown`` name,
        # so any zip tool can produce a valid textpack. Layout rules:
        #   * If the archive holds a ``*.textbundle`` directory, exactly one is
        #     allowed and the body must live directly inside it.
        #   * Otherwise the body must live directly in the archive root.
        #   * The chosen directory must hold exactly one ``*.md``/``*.markdown``.
        # ``bundle_root`` (the returned file's parent) anchors asset resolution.
        bundles = sorted(
            p
            for p in root.iterdir()
            if p.is_dir() and p.suffix.lower() == ".textbundle"
        )
        if len(bundles) > 1:
            names = ", ".join(p.name for p in bundles)
            raise ValueError(
                f"multiple .textbundle directories in textpack: {source_name} ({names})"
            )
        search_dir = bundles[0] if bundles else root
        markdown = sorted(
            p
            for p in search_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".md", ".markdown")
        )
        if len(markdown) > 1:
            names = ", ".join(p.name for p in markdown)
            raise ValueError(
                f"multiple markdown files in textpack: {source_name} ({names})"
            )
        if not markdown:
            raise ValueError(f"no markdown text file found in textpack: {source_name}")
        return markdown[0]

    def _extract_text(
        self,
        md_text: str,
        *,
        bundle_root: Path | None,
        raw_cache: NativeImageRawCache | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        warnings: dict[str, int] = {}
        resolver = _MarkdownImageResolver(
            bundle_root=bundle_root,
            warnings=warnings,
            download_enabled=_env_bool("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", True),
            download_required=_env_bool("NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED", False),
            timeout=_env_int("NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT", 30),
            max_bytes=_env_int("NATIVE_MD_IMAGE_MAX_BYTES", 25 * 1024 * 1024),
            max_svg_pixels=_env_int("NATIVE_MD_IMAGE_MAX_SVG_PIXELS", 16_000_000),
            raw_cache=raw_cache,
        )
        extraction = extract_markdown(md_text, image_resolver=resolver)
        metadata: dict[str, Any] = {
            "md_tables": extraction.tables,
            "md_equations": extraction.equations,
            "md_drawings": extraction.drawings,
            "md_assets": extraction.assets,
        }
        return extraction.blocks, dict(warnings), metadata

    def build_ir(
        self,
        blocks: list[dict[str, Any]],
        *,
        document_name: str,
        asset_dir_name: str,
        metadata: dict[str, Any],
    ) -> "IRDoc":
        from lightrag.parser.markdown.ir_builder import NativeMarkdownIRBuilder

        return NativeMarkdownIRBuilder().normalize(
            blocks,
            document_name=document_name,
            asset_dir_name=asset_dir_name,
            parse_metadata=metadata,
        )

    def surface_warnings(
        self, warnings: dict[str, Any], source: Path
    ) -> dict[str, Any] | None:
        relevant = {k: v for k, v in warnings.items() if v}
        return relevant or None
