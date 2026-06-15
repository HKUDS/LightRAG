"""Native Markdown engine adapter (.md / .textpack).

Implements the :class:`NativeParserBase` hooks for markdown input. ``.textpack``
is a zipped TextBundle (a ``text.markdown`` plus an ``assets/`` resource dir;
Bear / Ulysses export format); a plain ``.md`` file is parsed directly.

Image handling (see the parser plan):

- base64 data-URL images are decoded and materialized into the sidecar assets.
- file-reference images are resolved ONLY inside a ``.textpack`` bundle, from
  the safely-extracted bundle directory (with a read-side path-traversal
  guard); a relative reference in a standalone ``.md`` is skipped + warned.
- external ``http(s)`` images are kept as external links by default; downloading
  is opt-in via ``NATIVE_MD_IMAGE_DOWNLOAD_ENABLED`` and SSRF/size guarded.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import socket
import tempfile
import urllib.error
import urllib.request
from ipaddress import ip_address, ip_network
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from lightrag.constants import PARSER_ENGINE_NATIVE
from lightrag.parser.markdown.extract import ResolvedImage, extract_markdown
from lightrag.parser.native_base import NativeParserBase
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
    """Return the file extension for ``raw`` image bytes, or ``None`` if the
    bytes are not a recognised image (unlike ``_vision_utils._detect_mime``,
    which always falls back to PNG)."""
    if raw.startswith(_PNG_SIGNATURE):
        return "png"
    if raw.startswith(_JPEG_SIGNATURE):
        return "jpg"
    if any(raw.startswith(sig) for sig in _GIF_SIGNATURES):
        return "gif"
    if len(raw) >= 12 and raw[0:4] == _WEBP_RIFF and raw[8:12] == _WEBP_TAG:
        return "webp"
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


def _host_is_public(host: str) -> bool:
    """True iff every resolved address for ``host`` is safe to fetch.

    Default-deny: an address is allowed only when ``ip.is_global`` (so SSRF to
    loopback / private / link-local / reserved / CGNAT ``100.64.0.0/10`` /
    TEST-NET and any other non-globally-routable range is blocked). A
    non-global address is allowed only when it matches the operator-configured
    ``NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS`` escape hatch. A resolution
    failure is treated as non-public.
    """
    allow = _allowed_non_public_networks()
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        addr = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ip_address(addr)
        except ValueError:
            return False
        if ip.is_global:
            continue
        if any(ip in net for net in allow):
            continue
        return False
    return True


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate the host on every redirect so a redirect cannot bounce the
    request to an internal address."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        host = urlparse(newurl).hostname or ""
        if not _host_is_public(host):
            raise urllib.error.HTTPError(
                newurl, code, "redirect to non-public host blocked", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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
    ) -> None:
        self._bundle_root = bundle_root.resolve() if bundle_root else None
        self._warnings = warnings
        self._download_enabled = download_enabled
        self._download_required = download_required
        self._timeout = timeout
        self._max_bytes = max_bytes
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
        # for validation (matching the remote-download path).
        ext = _image_ext_from_magic(data)
        if ext is None:
            return self._skip("unrecognised image", src)
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
        data = candidate.read_bytes()
        # Magic bytes are authoritative; the filename suffix is not trusted for
        # validation (it is still used as the suggested on-disk name).
        ext = _image_ext_from_magic(data)
        if ext is None:
            return self._skip("unrecognised image", src)
        return self._local(data, ext, candidate.name)

    def _resolve_remote(self, src: str) -> ResolvedImage:
        ext_hint = Path(urlparse(src).path).suffix.lower().lstrip(".")
        if not self._download_enabled:
            return ResolvedImage(kind="external", url=src, fmt=ext_hint)
        try:
            data, ext = self._download(src)
        except Exception as exc:  # noqa: BLE001 - best-effort network fetch
            if self._download_required:
                raise
            logger.warning("[native_md] image download failed (%s): %s", exc, src[:120])
            self._bump("images_download_failed")
            return ResolvedImage(kind="external", url=src, fmt=ext_hint)
        digest = hashlib.sha256(data).hexdigest()[:12]
        return self._local(data, ext, f"image-{digest}.{ext}")

    def _download(self, src: str) -> tuple[bytes, str]:
        parsed = urlparse(src)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"unsupported scheme: {parsed.scheme!r}")
        if not _host_is_public(parsed.hostname or ""):
            raise ValueError("non-public host blocked")
        opener = urllib.request.build_opener(_GuardedRedirectHandler())
        req = urllib.request.Request(src, headers={"User-Agent": "lightrag-native-md"})
        with opener.open(req, timeout=self._timeout) as resp:
            content_type = (resp.headers.get_content_type() or "").lower()
            # Read at most max_bytes + 1 so we can detect an over-limit body.
            data = resp.read(self._max_bytes + 1)
        if len(data) > self._max_bytes:
            raise ValueError(f"image exceeds {self._max_bytes} bytes")
        ext = _image_ext_from_magic(data)
        if ext is None:
            raise ValueError("downloaded bytes are not a supported image")
        # Magic bytes are authoritative; reject only an *explicit* non-image
        # Content-Type. Missing / generic types defer to the magic check above.
        if (
            content_type
            and content_type not in _GENERIC_CONTENT_TYPES
            and not content_type.startswith("image/")
        ):
            raise ValueError(f"non-image Content-Type: {content_type!r}")
        return data, ext


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
        self, source: Path, *, parsed_dir: Path, asset_dir: Path, base_name: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        if source.suffix.lower() == ".textpack":
            tmp_dir = Path(tempfile.mkdtemp(prefix="textpack-"))
            try:
                md_text, bundle_root = self._open_textpack(source, tmp_dir)
                return self._extract_text(md_text, bundle_root=bundle_root)
            finally:
                rmtree(tmp_dir, ignore_errors=True)
        md_text = source.read_bytes().decode("utf-8-sig")
        return self._extract_text(md_text, bundle_root=None)

    def _open_textpack(self, source: Path, tmp_dir: Path) -> tuple[str, Path]:
        from lightrag.parser.external._zip import safe_extract_zip

        safe_extract_zip(
            source.read_bytes(),
            tmp_dir,
            max_entries=_TEXTPACK_MAX_ENTRIES,
            max_total_bytes=_TEXTPACK_MAX_TOTAL_BYTES,
        )
        # TextBundle: the text lives in ``text.markdown`` / ``text.md``; fall
        # back to any single markdown file. ``bundle_root`` is its directory.
        text_file = self._find_text_file(tmp_dir)
        if text_file is None:
            raise ValueError(f"no markdown text file found in textpack: {source.name}")
        return text_file.read_bytes().decode("utf-8-sig"), text_file.parent

    @staticmethod
    def _find_text_file(root: Path) -> Path | None:
        for name in ("text.markdown", "text.md"):
            for candidate in (root / name, *root.glob(f"*/{name}")):
                if candidate.is_file():
                    return candidate
        markdown = [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in (".md", ".markdown")
        ]
        return markdown[0] if len(markdown) == 1 else None

    def _extract_text(
        self, md_text: str, *, bundle_root: Path | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        warnings: dict[str, int] = {}
        resolver = _MarkdownImageResolver(
            bundle_root=bundle_root,
            warnings=warnings,
            download_enabled=_env_bool("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", False),
            download_required=_env_bool("NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED", False),
            timeout=_env_int("NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT", 30),
            max_bytes=_env_int("NATIVE_MD_IMAGE_MAX_BYTES", 25 * 1024 * 1024),
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
