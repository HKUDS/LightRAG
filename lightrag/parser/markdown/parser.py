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
import hashlib
import http.client
import os
import re
import socket
import tempfile
import urllib.error
import urllib.request
from ipaddress import ip_address, ip_network
from math import ceil
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from lightrag.constants import NATIVE_RAW_DIR_SUFFIX, PARSER_ENGINE_NATIVE
from lightrag.parser.external._common import raw_dir_for_parsed_dir
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


def _validated_addresses(host: str) -> list[str]:
    """Resolve ``host`` and return its addresses iff ALL are safe to fetch.

    Default-deny: an address is accepted only when ``ip.is_global`` (so SSRF to
    loopback / private / link-local / reserved / CGNAT ``100.64.0.0/10`` /
    TEST-NET and any other non-globally-routable range is blocked) or it matches
    the operator-configured ``NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS`` escape
    hatch. Returns ``[]`` when resolution fails or *any* resolved address is
    non-public — so a single poisoned A/AAAA record rejects the whole host.

    The returned addresses are what the connection actually dials (see
    :class:`_GuardedHTTPConnection`): validating and connecting share one
    resolution, closing the DNS-rebinding TOCTOU window between check and
    connect.
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
        if not (ip.is_global or any(ip in net for net in allow)):
            return []
        addrs.append(addr)
    return addrs


def _host_is_public(host: str) -> bool:
    """True iff every resolved address for ``host`` is safe to fetch."""
    return bool(_validated_addresses(host))


def _pin_socket(host: str, port: int, timeout, source_address):
    """Open a TCP socket to a *validated* resolved address of ``host``.

    The address comes from the same :func:`_validated_addresses` resolution
    that authorised the host, so urllib never independently re-resolves the
    name (which a DNS-rebinding attacker could flip to an internal IP between
    our check and the actual connect)."""
    addrs = _validated_addresses(host)
    if not addrs:
        raise OSError(f"refusing connection to non-public host: {host!r}")
    sock = socket.create_connection((addrs[0], port), timeout, source_address)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


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


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate the host on every redirect so a redirect cannot bounce the
    request to an internal address. (Defense in depth: the pinned connection
    re-validates at connect time too.)"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        host = urlparse(newurl).hostname or ""
        if not _host_is_public(host):
            raise urllib.error.HTTPError(
                newurl, code, "redirect to non-public host blocked", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_guarded_opener() -> urllib.request.OpenerDirector:
    """Opener whose connections pin to a validated IP and that ignores any
    ambient ``HTTP(S)_PROXY`` (an env proxy would otherwise fetch the blocked
    URL on our behalf, bypassing the IP guard)."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _GuardedHTTPHandler(),
        _GuardedHTTPSHandler(),
        _GuardedRedirectHandler(),
    )


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
    ) -> None:
        self._bundle_root = bundle_root.resolve() if bundle_root else None
        self._warnings = warnings
        self._download_enabled = download_enabled
        self._download_required = download_required
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_svg_pixels = max_svg_pixels
        self._raw_cache = raw_cache
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
        parsed = urlparse(src)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"unsupported scheme: {parsed.scheme!r}")
        if not _host_is_public(parsed.hostname or ""):
            raise ValueError("non-public host blocked")
        opener = _build_guarded_opener()
        req = urllib.request.Request(src, headers={"User-Agent": "lightrag-native-md"})
        with opener.open(req, timeout=self._timeout) as resp:
            content_type = (resp.headers.get_content_type() or "").lower()
            # Read at most max_bytes + 1 so we can detect an over-limit body.
            data = resp.read(self._max_bytes + 1)
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
        # fail every markdown document it happens to route.
        ignored_params = sorted(runtime.engine_params) if runtime else []
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
