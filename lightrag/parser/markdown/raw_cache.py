"""Per-document download cache for native-markdown external images.

Mirrors the ``.mineru_raw`` / ``.docling_raw`` raw-bundle pattern, scoped to the
one expensive native-markdown step: downloading external ``http(s)`` images.
The bundle lives in a ``<file>.native_raw/`` directory that is a **sibling** of
``<file>.parsed/`` so it survives the ``rmtree(parsed_dir)`` that
:meth:`NativeParserBase.parse` performs before every re-extraction.

Bundle layout::

    <file>.native_raw/
        _manifest.json              # atomic success marker + cache key
        <sha256(url)[:16]>.png      # one file per cached image (final bytes)
        ...

The manifest is the cache key: a bundle is reused only when the source file
content hash AND the download-options signature both still match, mirroring the
external engines. On a hit the resolver reuses the stored bytes (already
post-SVG-rasterization), skipping both the network fetch and the rasterization.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from lightrag.parser.external._common import (
    clear_dir_contents,
    compute_size_and_hash,
)
from lightrag.utils import logger

_MANIFEST_FILENAME = "_manifest.json"
_MANIFEST_VERSION = "1"
_CACHE_ENGINE = "native_md"


def native_md_options_signature() -> str:
    """A ``sha256`` over the download knobs that change an image's bytes.

    Deliberately excludes ``NATIVE_MD_IMAGE_DOWNLOAD_ENABLED`` /
    ``..._TIMEOUT`` / ``..._REQUIRED`` — those gate *whether* a fetch happens,
    not the resulting bytes — and includes the size / SVG-pixel ceilings and the
    SSRF allowlist (which govern what bytes are accepted at all)."""
    payload = {
        "signature_version": 1,
        "max_bytes": os.getenv("NATIVE_MD_IMAGE_MAX_BYTES", ""),
        "max_svg_pixels": os.getenv("NATIVE_MD_IMAGE_MAX_SVG_PIXELS", ""),
        "allowed_non_public_cidrs": os.getenv(
            "NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", ""
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _url_filename(url: str, fmt: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    ext = fmt or "bin"
    return f"{digest}.{ext}"


class NativeImageRawCache:
    """Reuse already-downloaded external images across re-parses.

    Per-document and single-writer: each parse owns one ``raw_dir`` and the
    parse queue serializes work per document, so no locking is needed.
    """

    def __init__(
        self,
        raw_dir: Path,
        *,
        source_path: Path,
        options_signature: str,
        force_reparse: bool,
    ) -> None:
        self._raw_dir = raw_dir
        self._source_path = source_path
        self._options_signature = options_signature
        self._force_reparse = force_reparse
        self._source_hash = ""
        self._valid = False
        self._cleared = False
        # Index of reusable entries from a valid prior bundle (url -> entry).
        self._index: dict[str, dict] = {}
        # Entries referenced this run (reused or freshly put) -> manifest output.
        self._entries: dict[str, dict] = {}

    def load(self) -> None:
        """Compute the current source hash and decide whether the on-disk
        bundle is a cache hit (valid) or must be rebuilt."""
        try:
            _, self._source_hash = compute_size_and_hash(self._source_path)
        except OSError as exc:
            logger.debug("[native_md_cache] source hash failed: %s", exc)
            self._source_hash = ""
        if self._force_reparse:
            return
        manifest = self._read_manifest()
        if manifest is None:
            return
        if (
            manifest.get("source_content_hash") != self._source_hash
            or manifest.get("options_signature") != self._options_signature
        ):
            return
        images = manifest.get("images")
        if not isinstance(images, dict):
            return
        self._index = {k: v for k, v in images.items() if isinstance(v, dict)}
        self._valid = True

    def get(self, url: str) -> tuple[bytes, str] | None:
        """Return ``(bytes, fmt)`` for a cached image, or ``None`` on a miss /
        integrity failure (corrupt or tampered cache file)."""
        if not self._valid:
            return None
        entry = self._index.get(url)
        if not entry:
            return None
        file_name = str(entry.get("file") or "")
        if not file_name:
            return None
        path = self._raw_dir / file_name
        if not path.is_file():
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if "sha256:" + hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            logger.warning("[native_md_cache] cached file integrity mismatch: %s", url)
            return None
        fmt = str(entry.get("fmt") or "")
        self._entries[url] = entry
        return data, fmt

    def put(self, url: str, data: bytes, fmt: str) -> None:
        """Store freshly-downloaded image bytes and record them for the manifest."""
        self._ensure_writable_dir()
        file_name = _url_filename(url, fmt)
        try:
            (self._raw_dir / file_name).write_bytes(data)
        except OSError as exc:
            logger.warning("[native_md_cache] failed to write cache file: %s", exc)
            return
        self._entries[url] = {
            "file": file_name,
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "fmt": fmt,
        }

    def flush(self) -> None:
        """Write the manifest for the images referenced this run and prune any
        bundle files no longer referenced.

        No-op when nothing was downloaded or reused this run (an image-less doc
        or a download-disabled run), so a pre-existing valid bundle is left
        intact rather than emptied."""
        if not self._entries:
            return
        self._ensure_writable_dir()
        referenced = {e["file"] for e in self._entries.values()}
        for child in self._raw_dir.iterdir():
            if child.name == _MANIFEST_FILENAME:
                continue
            if child.is_file() and child.name not in referenced:
                try:
                    child.unlink()
                except OSError:
                    pass
        manifest = {
            "version": _MANIFEST_VERSION,
            "engine": _CACHE_ENGINE,
            "source_content_hash": self._source_hash,
            "options_signature": self._options_signature,
            "images": self._entries,
        }
        final = self._raw_dir / _MANIFEST_FILENAME
        tmp = final.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, final)
        except OSError as exc:
            logger.warning("[native_md_cache] failed to write manifest: %s", exc)

    def _ensure_writable_dir(self) -> None:
        """Create ``raw_dir`` and, on the first write of an invalidated bundle,
        drop the stale contents so reused entries never mingle with old ones."""
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        if not self._valid and not self._cleared:
            clear_dir_contents(self._raw_dir)
            self._cleared = True

    def _read_manifest(self) -> dict | None:
        path = self._raw_dir / _MANIFEST_FILENAME
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != _MANIFEST_VERSION:
            return None
        if payload.get("engine") != _CACHE_ENGINE:
            return None
        return payload


__all__ = ["NativeImageRawCache", "native_md_options_signature"]
