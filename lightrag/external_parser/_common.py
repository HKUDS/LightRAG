"""Shared helpers for ``lightrag/external_parser/<engine>/`` packages.

Currently consumed by the docling subpackage; expected to be reused when
mineru is migrated under ``external_parser/mineru/``.

These are pure functions with no engine-specific knowledge. Engine-specific
logic (endpoint signature, options signature, cache validation policy) lives
in each engine's own ``cache.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from lightrag.utils import logger


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


def clear_dir_contents(directory: Path) -> None:
    """Delete everything inside ``directory`` but keep ``directory`` itself."""
    if not directory.exists():
        return
    for entry in directory.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError:
            continue


def raw_dir_for_parsed_dir(parsed_dir: Path, *, suffix: str) -> Path:
    """Sibling raw dir for a ``*.parsed`` dir.

    ``foo.parsed/`` with ``suffix=".docling_raw"`` → ``foo.docling_raw/``.
    ``suffix`` must start with ``.`` and be engine-specific (the caller
    binds it via ``functools.partial`` or a thin wrapper).
    """
    if not suffix.startswith("."):
        raise ValueError(f"raw dir suffix must start with '.', got {suffix!r}")
    stem = parsed_dir.name
    if stem.endswith(".parsed"):
        stem = stem[: -len(".parsed")]
    return parsed_dir.parent / f"{stem}{suffix}"


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "[external_parser] %s=%r is not an integer; using %s", name, raw, default
        )
        return default


def env_json(name: str, default: Any) -> Any:
    """Parse a JSON env var; on parse error log a warning and return default."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "[external_parser] %s=%r is not valid JSON; using default", name, raw
        )
        return default


__all__ = [
    "clear_dir_contents",
    "compute_size_and_hash",
    "env_bool",
    "env_int",
    "env_json",
    "raw_dir_for_parsed_dir",
]
