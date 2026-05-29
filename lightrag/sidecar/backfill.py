"""Backfill chunk ``sidecar`` provenance for the F/R/V chunking strategies.

The ``P`` (paragraph_semantic) strategy records a ``sidecar`` field on each chunk
mapping it back to its source row(s) in the parse-time ``*.blocks.jsonl`` sidecar
file. The ``F`` / ``R`` / ``V`` strategies chunk the merged document text without
that provenance, but each chunk carries a private ``_source_span`` — the half-open
``[start, end)`` char offsets of its content within the merged text the chunker
received. When a document was parsed into the LightRAG sidecar format
(``blocks.jsonl`` exists), :func:`backfill_chunk_sidecars` reconstructs that merged
text, records each block's character span, and attaches a ``sidecar`` to each chunk
by mapping its ``_source_span`` onto the block(s) it overlaps.

Span contract
-------------
The merged text is exactly reproducible from ``blocks.jsonl``: both
:func:`lightrag.sidecar.writer.write_sidecar` and
:func:`lightrag.utils_pipeline.load_lightrag_document_content` build it as
``"\\n\\n".join(raw_content for content rows where content.strip())``. We rebuild
the same string here so a chunk's ``_source_span`` indexes directly into it.

F/R emit byte-verbatim spans (the span text equals the chunk content). V's
``SemanticChunker`` rejoins sentences with a single space, so a V chunk's content
may differ from its span text by whitespace alone; span validation therefore accepts
either a byte-exact match or a **whitespace-stripped** match (every whitespace char
removed, not merely collapsed to a single space). A span whose text matches the
chunk under neither test is treated as absent.

A chunk that reaches this stage without a usable ``_source_span`` is a hard error:
the document is marked FAILED via :class:`ChunkBlockMatchError`. The sole exception
is a chunk whose decoded content carries the Unicode replacement character
(:data:`_REPLACEMENT_CHAR`) — a multi-byte UTF-8 char split at a token-window
boundary corrupts both its span probe and its own content, making provenance
impossible; such a chunk degrades to no-sidecar rather than failing the document.

Multimodal placeholder tags (``<table …>…</table>``, ``<drawing …/>``,
``<equation …>…</equation>``) appear verbatim in both block content and chunk
content, so a span covering them maps to the right block(s) unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightrag.chunk_schema import normalize_chunk_sidecar
from lightrag.exceptions import ChunkBlockMatchError
from lightrag.utils import logger

# Separator used to join block content into the merged document text. Must match
# lightrag/sidecar/writer.py and lightrag/utils_pipeline.py.
_BLOCK_SEPARATOR = "\n\n"

# U+FFFD. The fixed-token chunker decodes arbitrary token windows; when a window
# boundary splits a multi-byte UTF-8 character (4-byte supplementary-plane chars:
# emoji, rare CJK extensions), the partial decode yields this replacement character
# in *both* the span probe and the chunk's own content. Such a chunk is then
# inherently unlocatable in the source — by span or by text — so provenance is
# impossible for it. We degrade to no-sidecar for that single chunk rather than
# failing the whole document.
_REPLACEMENT_CHAR = "�"


def _is_unlocatable(body: str) -> bool:
    """True when a chunk's content cannot be located in the source by any means.

    See :data:`_REPLACEMENT_CHAR`: a chunk whose decoded content carries U+FFFD lost
    bytes at a multi-byte token boundary, so neither its span nor its text can be
    matched against the verbatim ``blocks.jsonl`` content. Callers skip provenance for
    such chunks instead of marking the document FAILED.
    """
    return _REPLACEMENT_CHAR in body


def _load_content_blocks(blocks_path: str) -> list[tuple[str, str]]:
    """Read ``type == "content"`` rows from a blocks.jsonl file in order.

    Returns ``(blockid, raw_content)`` pairs, skipping the meta header and any
    malformed lines. Mirrors ``paragraph_semantic._load_blocks_from_jsonl`` but
    keeps the raw (un-stripped) content needed to reproduce the chunker input.

    Note: ``utils_pipeline.load_lightrag_document_content`` (which produces the
    merged text the chunker actually received) skips line 0 *by index*; here we
    skip *by type* instead. The two agree only because ``writer.write_sidecar``
    always emits the meta header as the very first line — under that invariant
    skip-by-type is equivalent, and it stays correct even if a stray non-content
    row ever appears mid-file.
    """
    blocks: list[tuple[str, str]] = []
    with Path(blocks_path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or obj.get("type") != "content":
                continue
            content = obj.get("content", "")
            if not isinstance(content, str):
                continue
            blockid = str(obj.get("blockid") or "").strip()
            blocks.append((blockid, content))
    return blocks


def _build_block_spans(
    blocks: list[tuple[str, str]],
) -> tuple[str, list[tuple[int, int, str]]]:
    """Reconstruct the merged text and each kept block's char span.

    Only rows whose ``content.strip()`` is truthy are kept (matching
    ``utils_pipeline.load_lightrag_document_content``). Returns ``(merged, spans)``
    where each span is ``(start, end, blockid)`` over ``merged`` char offsets. The
    ``"\\n\\n"`` separators between blocks belong to no span.
    """
    spans: list[tuple[int, int, str]] = []
    parts: list[str] = []
    cursor = 0
    for blockid, content in blocks:
        if not content.strip():
            continue
        if parts:
            cursor += len(_BLOCK_SEPARATOR)
        start = cursor
        end = start + len(content)
        spans.append((start, end, blockid))
        parts.append(content)
        cursor = end
    return _BLOCK_SEPARATOR.join(parts), spans


def _normalize_text(text: str) -> str:
    """Whitespace-stripped form of a string (every whitespace char removed).

    Used by :func:`_chunk_source_span` to validate a span whose text differs from the
    chunk content by whitespace only — V's ``SemanticChunker`` rejoins sentences with
    a single space, so its chunk content is not byte-verbatim against the source span.
    Removing all whitespace (rather than collapsing runs to a single space) keeps the
    two sides aligned even when V inserts a space between originally-adjacent
    sentences.
    """
    return "".join(text.split())


def _covered_blockids(
    spans: list[tuple[int, int, str]], o_start: int, o_end: int
) -> list[str]:
    """Blockids whose content span overlaps ``[o_start, o_end)``, in order, deduped."""
    covered: list[str] = []
    seen: set[str] = set()
    for start, end, blockid in spans:
        # Overlap test on half-open intervals; separators (gaps) match nothing.
        if start < o_end and o_start < end and blockid and blockid not in seen:
            seen.add(blockid)
            covered.append(blockid)
    return covered


def _chunk_source_span(
    chunk: dict[str, Any],
    merged: str,
) -> tuple[int, int] | None:
    span = chunk.get("_source_span")
    if not isinstance(span, dict):
        return None
    try:
        start = int(span["start"])
        end = int(span["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end <= start or end > len(merged):
        return None
    body = chunk.get("content", "")
    if not isinstance(body, str):
        return None
    source_text = merged[start:end]
    if source_text != body and _normalize_text(source_text) != _normalize_text(body):
        return None
    return start, end


def backfill_chunk_sidecars(
    chunking_result: list[dict[str, Any]],
    blocks_path: str,
) -> None:
    """Attach a ``sidecar`` to each F/R/V chunk via its ``_source_span``, in place.

    No-op when ``blocks_path`` is empty/unreadable or carries no content rows.
    Chunks that already have a valid sidecar (P / multimodal) and empty-content
    chunks are skipped. Every remaining chunk must carry a ``_source_span`` that
    resolves to its content in the reconstructed merged text and overlaps at least
    one block; one that does not raises :class:`ChunkBlockMatchError`, marking the
    document FAILED.

    Exception: a chunk whose content carries the Unicode replacement character
    (:data:`_REPLACEMENT_CHAR`) is inherently unlocatable — a multi-byte UTF-8
    character was split at a token-window boundary, corrupting both its span probe
    and its own content. Such a chunk is skipped (no sidecar) instead of failing the
    document.
    """
    if not blocks_path:
        return

    try:
        blocks = _load_content_blocks(blocks_path)
    except OSError as exc:
        logger.warning(
            f"[sidecar-backfill] cannot read blocks.jsonl at {blocks_path}: {exc}; "
            "skipping sidecar backfill"
        )
        return

    merged, spans = _build_block_spans(blocks)
    if not spans:
        return

    for chunk in chunking_result:
        if not isinstance(chunk, dict):
            continue
        if normalize_chunk_sidecar(chunk) is not None:
            continue
        body = chunk.get("content", "")
        if not isinstance(body, str) or not body.strip():
            continue

        source_span = _chunk_source_span(chunk, merged)
        if source_span is None:
            # No usable span: provenance is impossible. Degrade silently only for the
            # inherently-unlocatable replacement-char case; otherwise FAIL the document
            # rather than guess (the old text-matching fallback could not disambiguate
            # a genuine cross-block match from a whitespace-glue artifact).
            if _is_unlocatable(body):
                logger.warning(
                    f"[sidecar-backfill] chunk #{chunk.get('chunk_order_index', -1)} "
                    "contains replacement characters from a multi-byte token-boundary "
                    "split; skipping provenance for it"
                )
                continue
            raise ChunkBlockMatchError(
                chunk_order_index=int(chunk.get("chunk_order_index", -1)),
                chunk_preview=body,
                blocks_path=blocks_path,
            )

        o_start, o_end = source_span
        covered = _covered_blockids(spans, o_start, o_end)
        if not covered:
            # The span landed entirely on separator gaps — should not happen for
            # non-empty content, but guard rather than emit an empty ref.
            raise ChunkBlockMatchError(
                chunk_order_index=int(chunk.get("chunk_order_index", -1)),
                chunk_preview=body,
                blocks_path=blocks_path,
            )

        chunk["sidecar"] = {
            "type": "block",
            "id": covered[0],
            "refs": [{"type": "block", "id": bid} for bid in covered],
        }
