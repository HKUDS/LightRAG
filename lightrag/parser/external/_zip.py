"""Shared zip-bundle extraction for external parser engines.

Engines like docling return their full output as a zip archive. This helper
extracts it safely (refusing path traversal / absolute paths) into a target
directory. Engine-specific post-extraction normalization (e.g. mineru's
nested-subdir hoist) is *not* done here — each engine's client handles its
own quirks.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path


def safe_extract_zip(
    payload: bytes,
    dest_dir: Path,
    *,
    max_entries: int | None = None,
    max_total_bytes: int | None = None,
) -> list[str]:
    """Extract a zip archive into ``dest_dir``, refusing unsafe paths.

    Raises ``RuntimeError`` if any entry name is absolute or contains ``..``
    components after normalization. Returns the list of extracted member
    names (as stored in the zip, prior to OS-specific normalization), so
    callers can validate the bundle layout without re-walking the directory.

    Optional zip-bomb guards (both default ``None`` = unlimited, preserving the
    original behaviour for existing callers): ``max_entries`` caps the member
    count and ``max_total_bytes`` caps the summed *uncompressed* size declared
    in the archive's central directory. Both are checked from ``infolist()``
    metadata *before* extraction, so a malicious archive is rejected without
    being written to disk.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(payload)
    with zipfile.ZipFile(buf) as zf:
        infos = zf.infolist()
        if max_entries is not None and len(infos) > max_entries:
            raise RuntimeError(
                f"Refusing zip with {len(infos)} entries (max {max_entries})"
            )
        if max_total_bytes is not None:
            total = sum(info.file_size for info in infos)
            if total > max_total_bytes:
                raise RuntimeError(
                    f"Refusing zip: uncompressed size {total} bytes "
                    f"exceeds limit {max_total_bytes}"
                )
        names = zf.namelist()
        for name in names:
            norm = os.path.normpath(name)
            if (
                norm.startswith("..")
                or os.path.isabs(norm)
                or norm.startswith(("/", os.sep))
            ):
                raise RuntimeError(f"Refusing zip entry with unsafe path: {name!r}")
        zf.extractall(dest_dir)
    return names


__all__ = ["safe_extract_zip"]
