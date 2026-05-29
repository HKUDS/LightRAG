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


def _within_single_block(
    spans: list[tuple[int, int, str]], o_start: int, o_end: int
) -> bool:
    """True if ``[o_start, o_end)`` lies entirely within one block's content span.

    A block's content is contiguous (no internal gaps), so an interval contained in a
    single ``(start, end)`` touches exactly that block. An interval that straddles a
    separator gap is contained in no span and returns ``False``.
    """
    for start, end, _ in spans:
        if start <= o_start and o_end <= end:
            return True
        if start > o_start:
            break  # spans are start-ordered; no later span can contain o_start
    return False


def _locate_chunk(
    norm_merged: str,
    nq: str,
    prev_start: int,
    prev_end: int,
    norm_to_orig: list[int],
    spans: list[tuple[int, int, str]],
) -> int:
    """Locate ``nq`` consistently with forward, contiguous chunking; -1 if absent.

    F/R/V chunks cover the merged text in order. With token overlap each chunk shares
    a prefix with the previous one but always adds new content *past* it, so its end
    advances forward. Starts are not a reliable signal: stripping the whitespace a
    window falls on can make two consecutive chunks share a start (e.g. a window whose
    only non-whitespace content begins right after a separator the next window also
    begins with). The chunk **end** is the dependable monotonic anchor, and we further
    prefer matches that don't straddle a block boundary:

    - Primary: the leftmost occurrence at/after ``prev_start`` that (a) ends past
      ``prev_end`` and (b) lies entirely within a single block. Requiring forward
      end-progress rejects a pure-suffix duplicate inside the previous chunk (no new
      coverage); requiring single-block containment rejects a *cross-block artifact* —
      a match that exists only because whitespace stripping glued the tail of one block
      to the head of the next across a now-removed separator. Leftmost then resolves an
      overlap chunk to its true position rather than a later duplicate.
    - Cross-block fallback: if no single-block occurrence advances coverage, the
      leftmost occurrence that merely ends past ``prev_end``. A chunk whose window
      genuinely spanned blocks (its raw content held the separator) matches only here,
      so real cross-block chunks still resolve.
    - Tail fallback: when nothing extends coverage — a chunk clamped at the document
      tail, or a window that reduced to a duplicate of the previous one — the leftmost
      occurrence at/after ``prev_start`` (or -1 when the text is absent entirely).
    """
    length = len(nq)
    first_advancing = -1
    search = prev_start
    while True:
        p = norm_merged.find(nq, search)
        if p == -1:
            break
        search = p + 1
        if p + length <= prev_end:
            continue  # does not extend coverage; skip suffix/duplicate matches
        if first_advancing == -1:
            first_advancing = p
        o_start = norm_to_orig[p]
        o_end = norm_to_orig[p + length - 1] + 1
        if _within_single_block(spans, o_start, o_end):
            return p
    if first_advancing != -1:
        return first_advancing  # genuine cross-block chunk (no single-block match)
    return norm_merged.find(nq, prev_start)


def backfill_chunk_sidecars(
    chunking_result: list[dict[str, Any]],
    blocks_path: str,
    *,
    require_source_span: bool = False,
) -> None:
    """Attach a ``sidecar`` to each chunk lacking one, in place.

    No-op when ``blocks_path`` is empty/unreadable or carries no content rows.
    Chunks that already have a valid sidecar (P / multimodal) and empty-content
    chunks are skipped. ``_source_span`` is preferred when present. Raises
    :class:`ChunkBlockMatchError` when a sidecar-less, non-empty chunk cannot be
    located in the reconstructed merged text.
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

    # Forward cursor over the normalized text. ``prev_start`` is the previous chunk's
    # matched start; ``prev_end`` the furthest offset covered so far (monotonic, since
    # each chunk's end advances). ``_locate_chunk`` anchors on end-progress to place
    # each chunk consistently with contiguous, overlapping chunking — see its docstring.
    prev_start = 0
    prev_end = 0
    for chunk in chunking_result:
        if not isinstance(chunk, dict):
            continue
        if normalize_chunk_sidecar(chunk) is not None:
            continue
        body = chunk.get("content", "")
        if not isinstance(body, str) or not body.strip():
            continue

        source_span = _chunk_source_span(chunk, merged)
        if source_span is not None:
            o_start, o_end = source_span
            covered = _covered_blockids(spans, o_start, o_end)
            if not covered:
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
            continue

        if require_source_span:
            raise ChunkBlockMatchError(
                chunk_order_index=int(chunk.get("chunk_order_index", -1)),
                chunk_preview=body,
                blocks_path=blocks_path,
            )

        nq = _normalize_text(body)
        if not nq:
            continue

        pos = _locate_chunk(norm_merged, nq, prev_start, prev_end, norm_to_orig, spans)
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
        prev_end = max(prev_end, norm_end)

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
