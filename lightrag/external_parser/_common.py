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

from lightrag.constants import PARSED_DIR_SUFFIX
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
    if stem.endswith(PARSED_DIR_SUFFIX):
        stem = stem[: -len(PARSED_DIR_SUFFIX)]
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


def response_error_detail(resp: Any, *, limit: int = 1000) -> str:
    """Return a compact response body snippet for HTTP error reporting."""
    try:
        payload = resp.json() if getattr(resp, "text", "") else None
    except Exception:
        payload = None

    if payload is not None:
        try:
            detail = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except TypeError:
            detail = repr(payload)
    else:
        detail = str(getattr(resp, "text", "") or "").strip()

    detail = " ".join(detail.split())
    if not detail:
        return "empty response body"
    if len(detail) > limit:
        return f"{detail[:limit]}...<truncated>"
    return detail


def raise_for_status_with_detail(resp: Any, operation: str) -> None:
    """Raise an HTTP error that preserves service-provided response details.

    Treats any non-2xx response as an error, matching httpx's
    ``raise_for_status`` status handling (which also raises on 1xx/3xx,
    not just 4xx/5xx) while attaching a compact response-body snippet to
    the message for faster diagnosis.
    """
    status_code = int(getattr(resp, "status_code", 0) or 0)
    if 200 <= status_code < 300:
        return
    detail = response_error_detail(resp)
    raise RuntimeError(f"{operation} failed: HTTP {status_code} {detail}")


__all__ = [
    "clear_dir_contents",
    "compute_size_and_hash",
    "env_bool",
    "env_int",
    "env_json",
    "raise_for_status_with_detail",
    "raw_dir_for_parsed_dir",
    "response_error_detail",
]
