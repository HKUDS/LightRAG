"""Shared helpers for ``lightrag/parser/external/<engine>/`` packages.

Currently consumed by the docling subpackage; expected to be reused when
mineru is migrated under ``parser/external/mineru/``.

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
from typing import TYPE_CHECKING, Any

import async_timeout

from lightrag.constants import (
    DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES,
    DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT,
    PARSED_DIR_SUFFIX,
)
from lightrag.utils import logger

if TYPE_CHECKING:
    import httpx


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


def download_deadline_seconds() -> float | None:
    """Live-read the overall result-bundle download wall-clock budget.

    Read at call time (like :func:`result_bundle_limits`) so an operator
    can raise or disable it without a code change. A non-positive value
    disables the deadline (``None``), matching the zip-bomb guards'
    "non-positive = unlimited" convention. Parsed as a float, not through
    ``env_int`` — this value feeds straight into :func:`download_timeout`,
    which takes fractional seconds natively, and mirrors the float style
    ``httpx.Timeout(120.0, connect=30.0)`` already uses for the per-read
    timeout it complements.
    """
    raw = os.getenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", "").strip()
    if not raw:
        seconds = float(DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT)
    else:
        try:
            seconds = float(raw)
        except ValueError:
            logger.warning(
                "[external_parser] PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT=%r "
                "is not a number; using %s",
                raw,
                DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT,
            )
            seconds = float(DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT)
    return seconds if seconds > 0 else None


def download_max_bytes() -> int | None:
    """Live-read the raw-wire streaming cap for a result-bundle download.

    Deliberately a separate env var from ``PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES``
    (see :func:`result_bundle_limits`): that one bounds the *uncompressed*
    size a zip's central directory declares, checked by ``safe_extract_zip``
    against archive metadata after download. This bounds the *compressed*
    bytes actually received over the wire, checked chunk-by-chunk by
    :func:`stream_capped_get` before the bytes ever reach that check.
    Reusing one value for both would force an operator who wants to tune
    one budget to also move the other. Same "non-positive disables the
    gate" convention as the rest of this module's env-driven limits.
    """
    raw = env_int(
        "PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES",
        DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES,
    )
    return raw if raw > 0 else None


def download_timeout(seconds: float | None) -> "async_timeout.Timeout":
    """Wall-clock timeout context manager for a download call.

    Delegates to the ``async_timeout`` backport rather than the stdlib
    ``asyncio.timeout()``, which is Python 3.11+ only while this package
    declares ``requires-python = ">=3.10"``. Raises the same builtin
    ``TimeoutError`` and preserves the same distinction between its own
    deadline firing and an unrelated external cancellation.
    """
    return async_timeout.timeout(seconds)


async def stream_capped_get(
    client: "httpx.AsyncClient", url: str, *, max_bytes: int | None, operation: str
) -> tuple["httpx.Response", bytes]:
    """GET ``url`` via streaming, rejecting once more than ``max_bytes`` of
    body have been received.

    A plain ``client.get(url)`` fully buffers the response into memory
    before returning, no matter its size — a compromised or misbehaving
    endpoint can hold arbitrarily large data in memory before any
    zip-level check (entry count, declared uncompressed size) ever runs.
    Checking the running total on every chunk means an oversized response
    is rejected mid-download instead of after being fully buffered.
    ``max_bytes=None`` disables the cap (reads the full body, same as
    ``client.get()``).

    Returns ``(response, body)`` rather than relying on
    ``response.content`` / ``response.text`` — accessing either on a
    streamed response before the body has been read raises httpx's
    ``ResponseNotRead``, so callers that need the body must use the
    returned ``body`` instead.

    ``operation`` labels the error message the same way
    ``raise_for_status_with_detail`` does, deliberately instead of
    including ``url`` itself: a MinerU official-mode ``full_zip_url`` is a
    cloud-storage download link that may carry a signed access token in
    its query string, and this error can end up persisted into
    ``doc_status.error_msg`` or logs — so the URL must never appear here.
    """
    async with client.stream("GET", url) as response:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise RuntimeError(
                    f"{operation}: response body exceeds {max_bytes} bytes; "
                    f"refusing to buffer further"
                )
            chunks.append(chunk)
        return response, b"".join(chunks)


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
    "download_deadline_seconds",
    "download_max_bytes",
    "download_timeout",
    "env_bool",
    "env_int",
    "env_json",
    "raise_for_status_with_detail",
    "raw_dir_for_parsed_dir",
    "response_error_detail",
    "stream_capped_get",
]
