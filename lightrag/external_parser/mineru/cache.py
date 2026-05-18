"""Cache validation for ``*.mineru_raw/`` bundles.

Validation policy (settled in design discussion; see
``LightRAGSidecarFormat-zh.md`` related notes):

1. ``_manifest.json`` exists, parses, ``version=1.0`` ∧ ``engine=mineru``.
2. **Source size fast-path**: ``source_file.stat().st_size`` matches manifest;
   mismatch → miss without hashing.
3. **Source content_hash**: full sha256 of the current source file matches
   manifest. The size+hash pair is computed by a single-read helper so the
   stored manifest is internally self-consistent.
4. **API mode**: if the manifest recorded ``api_mode`` and it differs from
   current ``MINERU_API_MODE``, miss.
5. **Engine version**: if ``MINERU_ENGINE_VERSION`` is set and the manifest
   recorded a non-empty one, they must match.
6. **Endpoint signature**: if the active MinerU endpoint is set and the
   manifest recorded a non-empty one, they must match.
7. **Critical file**: ``content_list.json`` must exist with matching size
   **and** sha256 — sha256 here is the final tie-breaker against silent
   corruption affecting the file the adapter depends on.
8. **Other files**: size-only verification (cheap; covers most corruption
   modes for image / middle.json / layout.pdf).

Any failed step ⇒ cache miss; the caller wipes the directory contents
(preserving the directory itself) and re-runs the download.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from lightrag.external_parser.mineru.manifest import load_manifest

MINERU_RAW_DIR_SUFFIX = ".mineru_raw"
DEFAULT_MINERU_API_MODE = "local"
DEFAULT_MINERU_OFFICIAL_ENDPOINT = "https://mineru.net"


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


def _current_api_mode() -> str:
    mode = os.getenv("MINERU_API_MODE", DEFAULT_MINERU_API_MODE).strip().lower()
    return mode if mode in {"official", "local"} else DEFAULT_MINERU_API_MODE


def _current_endpoint_signature() -> str:
    mode = _current_api_mode()
    if mode == "official":
        return (
            os.getenv("MINERU_OFFICIAL_ENDPOINT", DEFAULT_MINERU_OFFICIAL_ENDPOINT)
            .strip()
            .rstrip("/")
        )
    if mode == "local":
        return os.getenv("MINERU_LOCAL_ENDPOINT", "").strip().rstrip("/")
    return ""


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

    # 3. API mode (only when manifest had one; old manifests remain compatible)
    cur_api_mode = _current_api_mode()
    if manifest.api_mode and cur_api_mode != manifest.api_mode:
        return False

    # 4. Engine version (only when current env exposes one AND manifest had one)
    cur_engine_version = os.getenv("MINERU_ENGINE_VERSION", "").strip()
    if (
        cur_engine_version
        and manifest.engine_version
        and cur_engine_version != manifest.engine_version
    ):
        return False

    # 5. Endpoint signature
    cur_endpoint = _current_endpoint_signature()
    if (
        cur_endpoint
        and manifest.endpoint_signature
        and cur_endpoint != manifest.endpoint_signature
    ):
        return False

    # 6. Critical file: size + sha256
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

    # 7. Other files: size only
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
