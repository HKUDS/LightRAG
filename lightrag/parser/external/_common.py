"""Shared helpers for ``lightrag/parser/external/<engine>/`` packages.

Currently consumed by the docling subpackage; expected to be reused when
mineru is migrated under ``parser/external/mineru/``.

These are pure functions with no engine-specific knowledge. Engine-specific
logic (endpoint signature, options signature, cache validation policy) lives
in each engine's own ``cache.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.constants import (
    DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES,
    DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT,
    PARSED_DIR_SUFFIX,
)
from lightrag.utils import logger

if TYPE_CHECKING:
    import httpx

    import async_timeout


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


def response_error_detail(
    resp: Any, *, body: bytes | None = None, limit: int = 1000
) -> str:
    """Return a compact response body snippet for HTTP error reporting.

    ``body``, when provided, is used *instead of* ``resp.text`` /
    ``resp.json()`` / ``resp.content`` -- none of those are touched at
    all in that case. A response read via :func:`stream_capped_get`'s
    manual ``response.aiter_bytes()`` loop (rather than ``resp.aread()``)
    is never marked read in a way httpx accepts for those properties;
    accessing any of them raises ``httpx.ResponseNotRead``, which is
    itself a ``RuntimeError`` subclass (via ``httpx.StreamError``) and so
    would silently masquerade as this function's own intended error
    instead of surfacing the real HTTP status and detail. Callers that
    already collected the body via streaming must pass it here.
    """
    if body is not None:
        try:
            payload = json.loads(body) if body else None
        except Exception:
            payload = None
        if payload is not None:
            try:
                detail = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            except TypeError:
                detail = repr(payload)
        else:
            detail = body.decode("utf-8", errors="replace").strip()
    else:
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
    """Live-read the streaming cap on a result-bundle download's response body.

    Bounds the accepted response-body bytes received, checked chunk-by-chunk
    by :func:`stream_capped_get`. Separate from
    ``PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES`` (see :func:`result_bundle_limits`),
    which bounds the *uncompressed* size a zip's central directory declares.
    Non-positive disables the cap, same convention as this module's other
    env-driven limits.
    """
    raw = env_int(
        "PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES",
        DEFAULT_PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES,
    )
    return raw if raw > 0 else None


def download_timeout(
    seconds: float | None,
) -> "asyncio.Timeout | async_timeout.Timeout":
    """Wall-clock timeout context manager for a download call.

    Uses the stdlib ``asyncio.timeout()`` on Python 3.11+; that API does
    not exist on 3.10, which this package's ``requires-python = ">=3.10"``
    still allows, so 3.10 falls back to the ``async_timeout`` backport
    instead. The backport is imported lazily, inside this function, rather
    than at module level: this module is also imported by code with no
    HTTP/timeout involvement at all (e.g. the native markdown parser via
    ``raw_dir_for_parsed_dir``), and ``async-timeout`` is only declared as
    a dependency for Python <3.11 (see the ``api`` extra in
    ``pyproject.toml``) -- a top-level import would make merely importing
    this module fail on a base install running 3.11+ that never even
    reaches this function.

    Both raise ``asyncio.TimeoutError`` on their own deadline firing --
    on 3.11+ that is the exact same class as builtin ``TimeoutError``
    (aliased since 3.11), so callers should catch ``asyncio.TimeoutError``
    specifically rather than plain ``TimeoutError`` to also catch the
    3.10 backport's distinct exception class. Both implementations
    preserve the same distinction between their own deadline firing and
    an unrelated external cancellation.
    """
    if sys.version_info >= (3, 11):
        return asyncio.timeout(seconds)
    import async_timeout

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

    ``operation`` labels the error message instead of ``url`` itself: a
    MinerU official-mode ``full_zip_url`` is a cloud-storage link that may
    carry a signed access token in its query string, and this error can
    end up persisted into ``doc_status.error_msg`` or logs.

    Requests ``Accept-Encoding: identity`` and rejects any response whose
    ``Content-Encoding`` is not absent/``identity``, checked right after
    the response headers arrive and before any body byte is read.
    Content-decoding would hand back decoded bytes instead of what was
    received, letting a compressed body expand past ``max_bytes`` in
    memory before the count catches up; rejecting any encoding keeps the
    byte count accurate and avoids that amplification.
    """
    async with client.stream(
        "GET", url, headers={"Accept-Encoding": "identity"}
    ) as response:
        content_encoding = response.headers.get("content-encoding")
        if (
            content_encoding is not None
            and content_encoding.strip().lower() != "identity"
        ):
            raise RuntimeError(
                f"{operation}: refusing response with Content-Encoding "
                f"{content_encoding!r}; only an absent header or "
                f"'identity' is accepted so the download's byte cap "
                f"measures the accepted response-body bytes"
            )
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


def raise_for_status_with_detail(
    resp: Any, operation: str, *, body: bytes | None = None
) -> None:
    """Raise an HTTP error that preserves service-provided response details.

    Treats any non-2xx response as an error, matching httpx's
    ``raise_for_status`` status handling (which also raises on 1xx/3xx,
    not just 4xx/5xx) while attaching a compact response-body snippet to
    the message for faster diagnosis.

    ``body``: pass the bytes already collected via :func:`stream_capped_get`
    when ``resp`` came from a streamed request. See
    :func:`response_error_detail` for why this matters.
    """
    status_code = int(getattr(resp, "status_code", 0) or 0)
    if 200 <= status_code < 300:
        return
    detail = response_error_detail(resp, body=body)
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
