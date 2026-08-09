"""Decompression budget for the native DOCX engine (GHSA-2wpj-ffvv-2pq8).

A ``.docx`` is a ZIP archive, and every size control on the ingestion path
bounds the *compressed* form: ``MAX_UPLOAD_SIZE`` bounds the file on disk,
``MAX_REQUEST_BODY_BYTES`` bounds the request body. Neither bounds what the
parser actually pays for, which is the uncompressed size — ``python-docx``
materializes ``word/document.xml`` whole, and four more call sites reopen the
same archive.

The budget is read from the ZIP central directory, so it costs one
``infolist()`` and decompresses nothing. See
:data:`lightrag.constants.DEFAULT_DOCX_MAX_UNCOMPRESSED_BYTES` for why the
declared sizes are trustworthy enough to gate on.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

from lightrag.constants import (
    DEFAULT_DOCX_MAX_COMPRESSION_RATIO,
    DEFAULT_DOCX_MAX_ENTRIES,
    DEFAULT_DOCX_MAX_UNCOMPRESSED_BYTES,
    DEFAULT_DOCX_RATIO_FLOOR_BYTES,
)
from lightrag.utils import get_env_value


class DocxDecompressionBudgetError(ValueError):
    """A .docx declares more expansion than the parser will spend on it.

    Subclasses ``ValueError`` so it travels the same path as the existing
    ``validate_source`` rejection: the pipeline records the document FAILED
    with this message rather than crashing the worker.
    """


def _limits() -> tuple[int, int, int, int]:
    """Read the budget from the environment at call time.

    Live-read (not import-time) for the same reason the smart_heading and
    CHUNK_P_REFERENCES_* knobs are: tests and embedded callers set these
    around a single parse. A non-positive value disables that one gate.
    """
    return (
        get_env_value(
            "DOCX_MAX_UNCOMPRESSED_BYTES", DEFAULT_DOCX_MAX_UNCOMPRESSED_BYTES, int
        ),
        get_env_value(
            "DOCX_MAX_COMPRESSION_RATIO", DEFAULT_DOCX_MAX_COMPRESSION_RATIO, int
        ),
        get_env_value("DOCX_RATIO_FLOOR_BYTES", DEFAULT_DOCX_RATIO_FLOOR_BYTES, int),
        get_env_value("DOCX_MAX_ENTRIES", DEFAULT_DOCX_MAX_ENTRIES, int),
    )


_EOCD_SIG = b"PK\x05\x06"
# EOCD record (22 bytes) plus the largest possible trailing comment.
_EOCD_SEARCH_WINDOW = 22 + 0xFFFF
# The EOCD's entry count is 16-bit; 0xFFFF means "65535, or more and the real
# count lives in the ZIP64 record".
_EOCD_COUNT_SATURATED = 0xFFFF


def _declared_entry_count(source: Path | bytes) -> int | None:
    """Total-entries field from the End of Central Directory record.

    Reads the tail of the archive only — the point is to answer "how many
    members does this claim?" WITHOUT ``zipfile.ZipFile`` walking the central
    directory, which allocates one ``ZipInfo`` per member before any limit can
    be applied (measured: 24 s and 272 MB for a 500k-member archive).

    Returns ``None`` when the record cannot be found or read. That is
    deliberately fail-OPEN: this is a fast pre-filter in front of the
    authoritative ``infolist()`` count, so a parsing failure here must cost a
    malformed archive nothing more than the slow path it would have taken
    anyway. It must never be the reason a legitimate document is refused.
    """
    try:
        if isinstance(source, bytes):
            tail = source[-_EOCD_SEARCH_WINDOW:]
        else:
            size = source.stat().st_size
            with source.open("rb") as fh:
                if size > _EOCD_SEARCH_WINDOW:
                    fh.seek(-_EOCD_SEARCH_WINDOW, io.SEEK_END)
                tail = fh.read()
    except OSError:
        return None

    marker = tail.rfind(_EOCD_SIG)
    if marker < 0 or len(tail) - marker < 22:
        return None
    # EOCD layout: sig(4) disk(2) cd_disk(2) this_disk_entries(2)
    #              total_entries(2) ...  → total at offset 10.
    return int.from_bytes(tail[marker + 10 : marker + 12], "little")


def enforce_docx_decompression_budget(source: Path | bytes, file_path: str) -> None:
    """Refuse ``source`` if its declared expansion exceeds the budget.

    Args:
        source: The ``.docx`` as a path on disk (native engine) or as the
            bytes already read into memory (legacy engine, which hands
            ``python-docx`` a ``BytesIO``). Both engines parse ``.docx`` and
            the default routing (``*:native-teP,*:legacy-R``) falls back from
            one to the other, so a guard on only one of them is bypassed by a
            ``bomb.[legacy].docx`` filename hint.
        file_path: Caller-facing name, used in the error message only.

    Raises:
        DocxDecompressionBudgetError: The archive declares more members or
            more uncompressed bytes than the budget allows, or expands at a
            ratio above the cap once past the small-file floor.

    A file that is not a readable ZIP is passed through, NOT rejected here.
    This function decides one thing — how much expansion the parser will pay
    for — and an unreadable archive has no declared expansion to weigh. The
    "is this really a .docx" verdict stays with python-docx downstream, which
    already reports it and reports it more precisely; taking it over here
    would change the error an existing caller sees for an unrelated reason.
    """
    max_uncompressed, max_ratio, ratio_floor, max_entries = _limits()

    # Entry count first, from the EOCD record, because opening the archive is
    # what an entry-count bomb makes expensive: ZipFile() builds every ZipInfo
    # before infolist() can be counted, so checking there means paying the
    # whole cost to then refuse. Only a refusal is decided here; anything else
    # falls through to the authoritative count below.
    if max_entries > 0:
        declared = _declared_entry_count(source)
        if declared is not None:
            over = (
                max_entries < _EOCD_COUNT_SATURATED
                if declared >= _EOCD_COUNT_SATURATED
                else declared > max_entries
            )
            if over:
                raise DocxDecompressionBudgetError(
                    f"Refusing {file_path}: archive declares "
                    f"{'>=' if declared >= _EOCD_COUNT_SATURATED else ''}{declared} "
                    f"members (limit {max_entries}, env DOCX_MAX_ENTRIES)"
                )

    handle: Any = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        with zipfile.ZipFile(handle) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, OSError):
        return

    if max_entries > 0 and len(infos) > max_entries:
        raise DocxDecompressionBudgetError(
            f"Refusing {file_path}: archive declares {len(infos)} members "
            f"(limit {max_entries}, env DOCX_MAX_ENTRIES)"
        )

    total = sum(info.file_size for info in infos)
    if max_uncompressed > 0 and total > max_uncompressed:
        raise DocxDecompressionBudgetError(
            f"Refusing {file_path}: declares {total} uncompressed bytes "
            f"(limit {max_uncompressed}, env DOCX_MAX_UNCOMPRESSED_BYTES)"
        )

    # Ratio is measured against the whole archive, not against the summed
    # compress_size: the gap between them (local headers, the central
    # directory itself) is part of what the attacker had to send.
    compressed = len(source) if isinstance(source, bytes) else source.stat().st_size
    if (
        max_ratio > 0
        and compressed > 0
        and total > ratio_floor
        and total / compressed > max_ratio
    ):
        raise DocxDecompressionBudgetError(
            f"Refusing {file_path}: declares {total} uncompressed bytes from "
            f"{compressed} on disk ({total / compressed:.0f}x, limit {max_ratio}x, "
            f"env DOCX_MAX_COMPRESSION_RATIO)"
        )


__all__ = [
    "DocxDecompressionBudgetError",
    "enforce_docx_decompression_budget",
]
