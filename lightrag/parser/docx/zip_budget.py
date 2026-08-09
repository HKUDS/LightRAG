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
    DEFAULT_DOCX_MAX_CENTRAL_DIR_BYTES,
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


def _limits() -> tuple[int, int, int, int, int]:
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
        get_env_value(
            "DOCX_MAX_CENTRAL_DIR_BYTES", DEFAULT_DOCX_MAX_CENTRAL_DIR_BYTES, int
        ),
    )


_EOCD_SIG = b"PK\x05\x06"
_EOCD64_LOCATOR_SIG = b"PK\x06\x07"
_EOCD64_SIG = b"PK\x06\x06"
# EOCD record (22 bytes) plus the largest possible trailing comment.
_EOCD_SEARCH_WINDOW = 22 + 0xFFFF
_EOCD64_LOCATOR_SIZE = 20
# ``size_cd`` is the 32-bit field in the classic EOCD. This value there means
# "the real size is in the ZIP64 record", NOT "the size is >= 4 GiB": a writer
# may use ZIP64 proactively for a tiny archive, and CPython reads and accepts
# it. So the sentinel is followed into the ZIP64 record, never treated as huge.
_ZIP64_SENTINEL_32 = 0xFFFFFFFF


def _read_slice(source: Path | bytes, offset: int, length: int) -> bytes | None:
    """Read ``length`` bytes at absolute ``offset``, or None if unavailable."""
    if offset < 0:
        return None
    if isinstance(source, bytes):
        if offset + length > len(source):
            return None
        return source[offset : offset + length]
    try:
        with source.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(length)
    except OSError:
        return None
    return data if len(data) == length else None


def _declared_central_directory_bytes(source: Path | bytes) -> int | None:
    """Central-directory byte size from the End of Central Directory record.

    Reads the tail of the archive only, to answer "how much work is opening
    this going to be?" WITHOUT ``zipfile.ZipFile`` doing that work — it walks
    the directory as ``while total < size_cd``, allocating one ``ZipInfo`` per
    member during construction, so any check made afterwards has already been
    paid for (measured: 24 s and 272 MB for a 500k-member archive).

    This reads ``size_cd`` and deliberately NOT the neighbouring member count.
    The count is attacker-controlled and inert: setting both count fields to
    100 leaves the directory untouched and the archive valid, CPython ignores
    them, and all 500,002 members are still walked. ``size_cd`` is the field
    CPython actually obeys, which is what makes it trustworthy — understating
    it makes ZipFile walk fewer bytes and see fewer members, i.e. the attack
    failing rather than succeeding.

    When the classic field holds the ZIP64 sentinel, the real ``size_cd`` is
    followed into the ZIP64 EOCD record via its locator, because that value
    can legitimately be small — a sentinel is NOT proof of a 4 GiB directory.

    Returns ``None`` when the record cannot be found or read. That is
    deliberately fail-OPEN: this is a fast pre-filter in front of the
    authoritative checks below, so a parsing failure here must cost a
    malformed archive nothing more than the slow path it would have taken
    anyway. It must never be the reason a legitimate document is refused.
    """
    try:
        if isinstance(source, bytes):
            total_size = len(source)
            tail = source[-_EOCD_SEARCH_WINDOW:]
            tail_start = max(0, total_size - len(tail))
        else:
            total_size = source.stat().st_size
            with source.open("rb") as fh:
                if total_size > _EOCD_SEARCH_WINDOW:
                    fh.seek(-_EOCD_SEARCH_WINDOW, io.SEEK_END)
                tail = fh.read()
            tail_start = max(0, total_size - len(tail))
    except OSError:
        return None

    marker = tail.rfind(_EOCD_SIG)
    if marker < 0 or len(tail) - marker < 22:
        return None
    # EOCD layout: sig(4) disk(2) cd_disk(2) this_disk_entries(2)
    #              total_entries(2) size_cd(4) ...  → size_cd at offset 12.
    size_cd = int.from_bytes(tail[marker + 12 : marker + 16], "little")
    if size_cd != _ZIP64_SENTINEL_32:
        return size_cd

    # ZIP64: the locator sits immediately before the classic EOCD and points
    # at the ZIP64 EOCD record (which carries the real 8-byte size_cd). A
    # signature mismatch anywhere here — including the concatenated-archive
    # case, where the recorded offset is shifted — falls through to fail-open.
    loc = marker - _EOCD64_LOCATOR_SIZE
    if loc < 0 or tail[loc : loc + 4] != _EOCD64_LOCATOR_SIG:
        return None
    # locator: sig(4) disk(4) z64_eocd_offset(8) total_disks(4)
    z64_offset = int.from_bytes(tail[loc + 8 : loc + 16], "little")
    # Prefer the copy already in the tail; only re-read for a far-back record.
    if z64_offset >= tail_start:
        rec = tail[z64_offset - tail_start : z64_offset - tail_start + 56]
    else:
        rec = _read_slice(source, z64_offset, 56)
    if rec is None or len(rec) < 48 or rec[:4] != _EOCD64_SIG:
        return None
    # ZIP64 EOCD record: sig(4) rec_size(8) vmade(2) vneed(2) disk(4)
    #   cd_disk(4) entries_disk(8) total_entries(8) size_cd(8) ...
    #   → size_cd at offset 40.
    return int.from_bytes(rec[40:48], "little")


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
    max_uncompressed, max_ratio, ratio_floor, max_entries, max_cd_bytes = _limits()

    # Cost gate first: opening the archive is itself the expensive step an
    # entry bomb exploits, so the member limit below cannot protect the work
    # that produces it. Bounding the central directory's byte size is what
    # bounds that walk. Only a refusal is decided here; everything else falls
    # through to the authoritative checks.
    if max_cd_bytes > 0:
        declared_cd = _declared_central_directory_bytes(source)
        if declared_cd is not None and declared_cd > max_cd_bytes:
            raise DocxDecompressionBudgetError(
                f"Refusing {file_path}: central directory is {declared_cd} bytes "
                f"(limit {max_cd_bytes}, env DOCX_MAX_CENTRAL_DIR_BYTES)"
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
