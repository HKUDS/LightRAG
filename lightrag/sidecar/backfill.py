"""Backfill chunk ``sidecar`` provenance for the F/R/V chunking strategies.

The ``P`` (paragraph_semantic) strategy records a ``sidecar`` field on each chunk
mapping it back to its source row(s) in the parse-time ``*.blocks.jsonl`` sidecar
file. The ``F`` / ``R`` / ``V`` strategies chunk the merged document text without
that provenance. When a document was parsed into the LightRAG sidecar format
(``blocks.jsonl`` exists), :func:`backfill_chunk_sidecars` scans the blocks and
attaches a ``sidecar`` to each chunk that lacks one.

Matching contract
-----------------
The merged text a chunker received is exactly reproducible from ``blocks.jsonl``:
both :func:`lightrag.sidecar.writer.write_sidecar` and
:func:`lightrag.utils_pipeline.load_lightrag_document_content` build it as
``"\\n\\n".join(raw_content for content rows where content.strip())``. We rebuild
the same string here, record each block's character span, and locate each chunk's
content within it.

F decodes/strips token windows (verbatim substrings, with token overlap); R strips
LangChain pieces (verbatim, possible overlap); V strips ``SemanticChunker`` pieces
that rejoin sentences with a single space — so newlines collapse to spaces *and* a
space may be inserted between sentences that were originally adjacent with no
whitespace, making the text *not* byte-verbatim. To cover all three uniformly,
matching runs over a **whitespace-stripped** projection of both sides (every
whitespace char removed, not merely collapsed to a single space). Because whitespace
removal is monotonic, a chunk's non-whitespace characters are always a contiguous
substring of the merged text's non-whitespace characters — regardless of how either
side spaced them — so V's reflowing can never spuriously fail a match.

Multimodal placeholder tags (``<table …>…</table>``, ``<drawing …/>``,
``<equation …>…</equation>``) appear identically in block content and chunk content,
so they project identically under whitespace stripping and match — markup is never
stripped before matching.
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


def _normalize_projection(merged: str) -> tuple[str, list[int]]:
    """Drop every whitespace char; map each kept char back to its merged offset.

    Returns ``(norm_text, norm_to_orig)`` where ``norm_to_orig[i]`` is the offset
    in ``merged`` of the ``i``-th surviving (non-whitespace) character. Removing all
    whitespace — rather than collapsing runs to a single space — keeps the two sides
    aligned even when V inserts a space between originally-adjacent sentences. Because
    whitespace removal is monotonic, any contiguous region of ``merged`` projects to a
    contiguous substring of ``norm_text``, so chunk lookups stay exact.
    """
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    for idx, ch in enumerate(merged):
        if ch.isspace():
            continue
        norm_chars.append(ch)
        norm_to_orig.append(idx)
    return "".join(norm_chars), norm_to_orig


def _normalize_text(text: str) -> str:
    """Whitespace-stripped form of a chunk body (every whitespace char removed).

    Uses ``str.split()``'s whitespace definition, which matches ``str.isspace()``
    used by :func:`_normalize_projection`, so both sides agree on what is stripped.
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


def backfill_chunk_sidecars(
    chunking_result: list[dict[str, Any]],
    blocks_path: str,
) -> None:
    """Attach a ``sidecar`` to each chunk lacking one, in place.

    No-op when ``blocks_path`` is empty/unreadable or carries no content rows.
    Chunks that already have a valid sidecar (P / multimodal) and empty-content
    chunks are skipped. Raises :class:`ChunkBlockMatchError` when a sidecar-less,
    non-empty chunk cannot be located in the reconstructed merged text.
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

    norm_merged, norm_to_orig = _normalize_projection(merged)

    # Forward-only cursor keyed on each chunk's *start* offset in the normalized
    # text. Contiguous F/R/V chunking guarantees the next chunk begins strictly
    # after the previous chunk's start (positive step) and at/before its end (token
    # overlap or plain adjacency — never a gap). So the correct match is the first
    # occurrence strictly past ``prev_start``: it lands on the overlap position for
    # F/R chunks that begin inside the previous chunk and on the next occurrence for
    # adjacent / repeated content. A ``prev_end`` anchor is structurally wrong here —
    # an overlap match always sits before ``prev_end``, so searching from ``prev_end``
    # skips it and can grab a spurious *later* duplicate of the same text, which then
    # strands the following chunk and raises a false ChunkBlockMatchError.
    prev_start = -1
    for chunk in chunking_result:
        if not isinstance(chunk, dict):
            continue
        if normalize_chunk_sidecar(chunk) is not None:
            continue
        body = chunk.get("content", "")
        if not isinstance(body, str) or not body.strip():
            continue

        nq = _normalize_text(body)
        if not nq:
            continue

        pos = norm_merged.find(nq, prev_start + 1)
        if pos == -1:
            raise ChunkBlockMatchError(
                chunk_order_index=int(chunk.get("chunk_order_index", -1)),
                chunk_preview=body,
                blocks_path=blocks_path,
            )

        norm_end = pos + len(nq)
        o_start = norm_to_orig[pos]
        # Map the last matched normalized char back, then extend one past it.
        o_end = norm_to_orig[norm_end - 1] + 1
        prev_start = pos

        covered = _covered_blockids(spans, o_start, o_end)
        if not covered:
            # The match landed entirely on separator gaps — should not happen for
            # non-empty normalized content, but guard rather than emit an empty ref.
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
