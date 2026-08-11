"""Decompression budget for the native DOCX engine (GHSA-2wpj-ffvv-2pq8).

A ``.docx`` is a ZIP archive, and every size control on the ingestion path
bounds the *compressed* form: ``MAX_UPLOAD_SIZE`` bounds the file on disk,
``MAX_REQUEST_BODY_BYTES`` bounds the request body. Neither bounds what the
parser actually pays for, which is the uncompressed size — ``python-docx``
materializes ``word/document.xml`` whole, and four more call sites reopen the
same archive.

The budget is read from the ZIP central directory via one ``infolist()`` and
decompresses nothing. See
:data:`lightrag.constants.DEFAULT_DOCX_MAX_UNCOMPRESSED_BYTES` for why the
declared sizes are trustworthy enough to gate on.

Scope note: ``DOCX_MAX_ENTRIES`` is checked AFTER ``zipfile.ZipFile`` has
built the ``ZipInfo`` list, i.e. after the central-directory walk it bounds
has already happened. An earlier revision tried to reject an entry-count bomb
before that walk by parsing the End-of-Central-Directory record from the file
tail; that hand-rolled ZIP/ZIP64 parsing proved error-prone and was removed.
The member walk runs inside the parser executor (``NativeParserBase.parse``
submits ``_extract_sync`` there), so a large central directory costs a parse
thread and its memory but does not stall the event loop. The compression-ratio
and absolute-size gates below — the actual GHSA-2wpj fix — are unaffected.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

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
    around a single parse.

    Returns ``(max_uncompressed, max_ratio, ratio_floor, max_entries)``. For
    the three GATES — ``max_uncompressed``, ``max_ratio``, ``max_entries`` — a
    non-positive value disables that gate. ``ratio_floor`` is NOT a gate: it is
    the small-content EXEMPTION threshold for the ratio gate. The member-ratio
    check sums the bytes by which over-ratio members exceed their individual
    ratio budgets. The fixed floor plus the archive bytes outside those
    members form the allowance for that excess. This lets real stored media
    support highly repetitive XML at one byte per byte without letting it buy
    another ``max_ratio``-fold expansion. The archive-wide ratio likewise
    applies only once the archive total exceeds the floor. A non-positive
    ``ratio_floor`` therefore does the OPPOSITE of "disable" — it removes the
    fixed exemption and makes both ratio views strictest.
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

    # Single dispatch on the source union: derive the ZIP handle and the
    # compressed (on-disk / in-memory) size together, so the two never drift.
    handle: Path | io.BytesIO
    if isinstance(source, bytes):
        handle, compressed = io.BytesIO(source), len(source)
    else:
        handle, compressed = source, source.stat().st_size
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

    total = 0
    over_ratio_excess_bytes = 0
    over_ratio_compressed_bytes = 0
    over_ratio_members = 0
    for info in infos:
        member_size = info.file_size
        total += member_size

        # The total ceiling necessarily bounds every member too, but keep the
        # member verdict explicit: it rejects as soon as one entry alone is
        # over budget and makes the protected quantity clear in the error.
        if max_uncompressed > 0 and member_size > max_uncompressed:
            raise DocxDecompressionBudgetError(
                f"Refusing {file_path}: one archive member declares "
                f"{member_size} uncompressed bytes (per-member limit "
                f"{max_uncompressed}, env DOCX_MAX_UNCOMPRESSED_BYTES)"
            )

        if max_ratio <= 0 or member_size <= 0:
            continue

        # Give each member its own max_ratio * compressed-size allowance.
        # Only the expansion beyond that allowance is suspicious. This is
        # additive across members, so splitting cannot recreate a per-member
        # floor. A positive member with no compressed bytes has no allowance.
        member_ratio_budget = max(0, info.compress_size) * max_ratio
        if member_size > member_ratio_budget:
            over_ratio_members += 1
            over_ratio_excess_bytes += member_size - member_ratio_budget
            over_ratio_compressed_bytes += max(0, info.compress_size)

    if max_uncompressed > 0 and total > max_uncompressed:
        raise DocxDecompressionBudgetError(
            f"Refusing {file_path}: declares {total} uncompressed bytes "
            f"(limit {max_uncompressed}, env DOCX_MAX_UNCOMPRESSED_BYTES)"
        )

    # The floor exempts a fixed amount of excess high-ratio expansion. Real
    # archive bytes outside those members add a 1:1 allowance: this preserves
    # mixed OOXML documents with repetitive XML plus incompressible media, but
    # padding cannot buy max_ratio-fold expansion and member splitting remains
    # additive. Clamp a non-positive floor to zero (strictest fixed exemption).
    supporting_archive_bytes = max(0, compressed - over_ratio_compressed_bytes)
    excess_allowance = max(0, ratio_floor) + supporting_archive_bytes
    if (
        max_ratio > 0
        and over_ratio_members > 0
        and over_ratio_excess_bytes > excess_allowance
    ):
        raise DocxDecompressionBudgetError(
            f"Refusing {file_path}: archive contains {over_ratio_members} "
            f"over-ratio members with {over_ratio_excess_bytes} excess "
            f"uncompressed bytes (allowance {excess_allowance}, limit "
            f"{max_ratio}x, env DOCX_MAX_COMPRESSION_RATIO)"
        )

    # Retain the archive-wide gate as a second view of amplification. The
    # member gate above closes stored-padding dilution; this one includes local
    # headers and the central directory in what the attacker had to send.
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
