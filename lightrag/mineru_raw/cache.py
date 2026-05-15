"""Cache validation for ``*.mineru_raw/`` bundles.

Validation policy (settled in design discussion; see
``LightRAGSidecarFormat-zh.md`` related notes):

1. ``_manifest.json`` exists, parses, ``version=1.0`` ∧ ``engine=mineru``.
2. **Source size fast-path**: ``source_file.stat().st_size`` matches manifest;
   mismatch → miss without hashing.
3. **Source content_hash**: full sha256 of the current source file matches
   manifest. The size+hash pair is computed by a single-read helper so the
   stored manifest is internally self-consistent.
4. **Engine version**: if ``MINERU_ENGINE_VERSION`` is set and the manifest
   recorded a non-empty one, they must match.
5. **Endpoint signature**: if ``MINERU_ENDPOINT`` is set and the manifest
   recorded a non-empty one, they must match.
6. **Critical file**: ``content_list.json`` must exist with matching size
   **and** sha256 — sha256 here is the final tie-breaker against silent
   corruption affecting the file the adapter depends on.
7. **Other files**: size-only verification (cheap; covers most corruption
   modes for image / middle.json / layout.pdf).

Any failed step ⇒ cache miss; the caller wipes the directory contents
(preserving the directory itself) and re-runs the download.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from lightrag.mineru_raw.manifest import Manifest, load_manifest

MINERU_RAW_DIR_SUFFIX = ".mineru_raw"


def raw_dir_for_parsed_dir(parsed_dir: Path) -> Path:
    """Sibling raw dir for a given ``*.parsed`` dir.

    ``foo.parsed/`` → ``foo.mineru_raw/``. Used both at download time and at
    cache check time so the layout is canonical.
    """
    stem = parsed_dir.name
    if stem.endswith(".parsed"):
        stem = stem[: -len(".parsed")]
    return parsed_dir.parent / f"{stem}{MINERU_RAW_DIR_SUFFIX}"


def clear_dir_contents(directory: Path) -> None:
    """Delete everything inside ``directory`` but keep ``directory`` itself."""
    if not directory.exists():
        return
    for entry in directory.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink():
                _rmtree_safe(entry)
            else:
                entry.unlink()
        except OSError:
            # Best-effort cleanup; subsequent download will overwrite.
            continue


def _rmtree_safe(directory: Path) -> None:
    import shutil

    shutil.rmtree(directory, ignore_errors=True)


def compute_size_and_hash(path: Path) -> tuple[int, str]:
    """Single-read computation of ``(size_bytes, "sha256:<hex>")``.

    Manifest writes use this so the recorded size and hash are guaranteed to
    describe the same byte stream; using two ``open()`` calls would risk a
    TOCTOU mismatch if the file changed in between.
    """
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, f"sha256:{h.hexdigest()}"


def is_bundle_valid(raw_dir: Path, source_file: Path) -> bool:
    """Return True iff the bundle is intact and matches the current source.

    See module docstring for the full policy. Returns False on any of:
    missing manifest, malformed manifest, schema version mismatch, source
    size/hash mismatch, engine/endpoint env mismatch, critical file
    missing or corrupted, or any non-critical file size mismatch.
    """
    if not raw_dir.is_dir():
        return False

    manifest = load_manifest(raw_dir)
    if manifest is None:
        return False

    # 1. Source size fast-path
    try:
        cur_size = source_file.stat().st_size
    except OSError:
        return False
    if cur_size != int(manifest.source_size_bytes):
        return False

    # 2. Source content_hash
    _, cur_hash = compute_size_and_hash(source_file)
    if cur_hash != manifest.source_content_hash:
        return False

    # 3. Engine version (only when current env exposes one AND manifest had one)
    cur_engine_version = os.getenv("MINERU_ENGINE_VERSION", "").strip()
    if (
        cur_engine_version
        and manifest.engine_version
        and cur_engine_version != manifest.engine_version
    ):
        return False

    # 4. Endpoint signature
    cur_endpoint = os.getenv("MINERU_ENDPOINT", "").strip()
    if (
        cur_endpoint
        and manifest.endpoint_signature
        and cur_endpoint != manifest.endpoint_signature
    ):
        return False

    # 5. Critical file: size + sha256
    crit = manifest.critical_file
    crit_path = raw_dir / crit.path
    try:
        if crit_path.stat().st_size != int(crit.size):
            return False
    except OSError:
        return False
    if crit.sha256:
        _, crit_actual = compute_size_and_hash(crit_path)
        if crit_actual != crit.sha256:
            return False

    # 6. Other files: size only
    for entry in manifest.files:
        ep = raw_dir / entry.path
        try:
            if ep.stat().st_size != int(entry.size):
                return False
        except OSError:
            return False

    return True


__all__ = [
    "MINERU_RAW_DIR_SUFFIX",
    "clear_dir_contents",
    "compute_size_and_hash",
    "is_bundle_valid",
    "raw_dir_for_parsed_dir",
]
