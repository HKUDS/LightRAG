"""Paragraph Semantic Chunking for LightRAG.

Reads a LightRAG ``.blocks.jsonl`` sidecar — produced by any sidecar-emitting
parser (native / mineru / docling; native uses ``fixlevel=0``): heading-driven
blocks, tables kept whole — and produces a chunk list compatible with
:func:`lightrag.chunker.chunking_by_token_size`.

The full algorithm and rationale are documented in
``docs/ParagraphSemanticChunking-zh.md``. This module re-implements the
post-HeadingBlocks pipeline (TableRowSplit/AnchorSplit/LevelMerge) on top of
blocks.jsonl input, parameterised on ``chunk_token_size`` so chunk-size targets
follow the user's RAG configuration rather than the audit-mode constants in
``lightrag/parser/docx/parse_document.py``.

Pipeline:
  - HeadingBlocks — heading-driven initial split: done at parse time and
    persisted as one row per block in ``blocks.jsonl``.
  - TableRowSplit — oversized-table re-split + first/middle/last gluing,
    invoked when an embedded ``<table … format="json">`` (or ``"html"``)
    exceeds ``table_max``. Splitting prefers structural row boundaries (JSON
    list items, HTML ``<tr>`` rows) so each fragment stays a legal ``<table>``
    tag; only when no row boundary is available, or one row alone exceeds the
    cap, does it fall back to ``chunking_by_recursive_character`` on that
    fragment. When two oversized tables are separated by text inside one
    heading block, the bridge text may be duplicated into both table boundary
    chunks so each table keeps nearby context. The source table's repeating
    header (lifted at parse time into the ``.tables.json`` sidecar, keyed by the
    slice's ``<table>`` id) is re-injected into the non-first slices *here*, with
    its tokens budgeted out of the per-slice cap *before* splitting — so every
    slice stays ``≤ target_max`` including its header. Because the header now
    lives in the slice from the start, a split table's slices are frozen against
    LevelMerge (re-merging two slices of one table would duplicate the header).
  - AnchorSplit — anchor-driven long-block re-split: short non-table
    paragraphs (≤ 100 chars) are promoted as split points and the block is
    rebalanced toward ``target_ideal``. With no anchor, table-aware fallback
    applies the same row-boundary-first strategy to any oversized table
    paragraph and only character-splits the residual prose (using the
    configured paragraph-semantic overlap).
  - HeadingGlue — body-less heading glue: a section heading with no body of
    its own is glued forward into its first strictly-deeper child block
    (keeping the shallower parent heading), so it is never separated from that
    child nor glued onto an unrelated same-level sibling. A heading with no
    deeper child following is left untouched for LevelMerge.
  - LevelMerge — bottom-up, level-aware small-block merging: undersized blocks
    get absorbed by same-level neighbours (Phase A), then shallower levels
    (Phase B), with a final tail-absorption pass for the trailing remainders.
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from lightrag.table_markup import (
    TABLE_TAG_RE as _TABLE_TAG_RE,
    detect_table_format as _detect_table_format,
    extract_table_id as _extract_table_id,
    serialize_html_rows as _serialize_rows_with_wrappers,
    split_html_rows as _split_html_rows,
)
from lightrag.utils import Tokenizer, logger


# ---------------------------------------------------------------------------
# Threshold ratios — derived from the audit-mode constants in
# lightrag/parser/docx/parse_document.py so the trade-off curves
# (table vs. block size, ideal vs. max, etc.) carry over verbatim. The
# absolute values scale with the user-configured ``chunk_token_size``.
# ---------------------------------------------------------------------------

# IDEAL/MAX = 6000/8000 = 0.75 in audit mode.
_IDEAL_RATIO = 0.75

# TABLE_MAX/MAX = 5000/8000 = 0.625 in audit mode.
_TABLE_MAX_RATIO = 0.625

# TABLE_IDEAL/MAX = 3000/8000 = 0.375 in audit mode.
_TABLE_IDEAL_RATIO = 0.375

# TABLE_MIN_LAST/TABLE_MAX = (TABLE_MAX-TABLE_IDEAL)*0.8/TABLE_MAX
#                          = (5000-3000)*0.8/5000 = 0.32 in audit mode.
_TABLE_MIN_LAST_RATIO = 0.32

# SMALL_TAIL_THRESHOLD/MAX = (MAX-IDEAL)/2/MAX = 1000/8000 = 0.125.
_SMALL_TAIL_RATIO = 0.125

# Anchor candidate length is a UI/readability constraint — keep absolute.
_MAX_ANCHOR_CANDIDATE_LENGTH = 100  # characters

# Table tag regex (``_TABLE_TAG_RE``) plus the ``_detect_table_format``,
# ``_split_html_rows`` and ``_serialize_rows_with_wrappers`` helpers are
# imported from :mod:`lightrag.table_markup` so the surrounding-context
# extractor can reuse the same primitives.

_LEGACY_TABLE_CHUNK_SUFFIX_RE = re.compile(r"\s*\[表格片段\d+\]\s*$")
_PART_SUFFIX_RE = re.compile(r"\s*\[part\s+\d+\]\s*$", re.IGNORECASE)

# Markdown heading-line pattern — 1-6 ``#`` followed by one or more spaces.
# Mirrors ``_MD_HEADING_RE`` in ``lightrag/parser/_markdown.py`` (the same
# pattern ``render_heading_line`` uses to render a heading content line) so a
# "heading-only" block — one whose content carries nothing but heading lines —
# can be detected without importing a private symbol across packages.
_HEADING_LINE_RE = re.compile(r"^#{1,6} +")


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _count_tokens(tokenizer: Tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def _bounded_overlap(target_max: int, chunk_overlap_token_size: int) -> int:
    """Return an overlap value safe for recursive-character splitting."""
    overlap = max(int(chunk_overlap_token_size), 0)
    if target_max <= 1:
        return 0
    return min(overlap, target_max - 1)


def _strip_generated_heading_suffixes(heading: str) -> str:
    """Remove generated split suffixes before assigning a fresh part number."""
    cleaned = (heading or "").rstrip()
    while True:
        next_cleaned = _PART_SUFFIX_RE.sub("", cleaned).rstrip()
        next_cleaned = _LEGACY_TABLE_CHUNK_SUFFIX_RE.sub("", next_cleaned).rstrip()
        if next_cleaned == cleaned:
            return cleaned
        cleaned = next_cleaned


def _append_part_suffix(heading: str, part_number: int) -> str:
    base = _strip_generated_heading_suffixes(heading)
    suffix = f"[part {part_number}]"
    return f"{base} {suffix}" if base else suffix


def _apply_part_suffixes(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag split fragments from one original block as ``[part n]``."""
    if len(blocks) <= 1:
        return blocks
    for idx, block in enumerate(blocks, start=1):
        block["heading"] = _append_part_suffix(block.get("heading", ""), idx)
    return blocks


def _is_table_paragraph(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("<table ") and stripped.endswith("</table>")


def _block_to_paragraphs(content: str) -> list[dict[str, Any]]:
    """Recover the per-paragraph view of a rewritten block.

    The sidecar writer joins paragraphs with ``\\n`` and emits
    tables/equations/drawings as single-line tags with no internal newlines,
    so ``split("\\n")`` faithfully recovers paragraph boundaries.
    """
    paragraphs: list[dict[str, Any]] = []
    for line in content.split("\n"):
        if not line.strip():
            continue
        paragraphs.append({"text": line, "is_table": _is_table_paragraph(line)})
    return paragraphs


def _load_blocks_from_jsonl(blocks_path: str) -> list[dict[str, Any]]:
    """Read ``type == "content"`` rows from a blocks.jsonl file in order."""
    rows: list[dict[str, Any]] = []
    with Path(blocks_path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "content":
                rows.append(obj)
    return rows


def _tables_sidecar_path(blocks_path: str) -> str | None:
    """Derive the ``.tables.json`` sidecar path beside a ``.blocks.jsonl`` file.

    Mirrors :func:`lightrag.utils_pipeline.sidecar_modality_path` naming
    (``<base>.blocks.jsonl`` → ``<base>.tables.json``). Returns ``None`` when
    the path does not follow the ``.blocks.jsonl`` convention.
    """
    suffix = ".blocks.jsonl"
    if not blocks_path.endswith(suffix):
        return None
    return blocks_path[: -len(suffix)] + ".tables.json"


def _load_table_headers(blocks_path: str) -> dict[str, str]:
    """Map ``table_id -> repeating-header body`` from the tables.json sidecar.

    Only tables that carry a ``table_header`` field (the cross-page repeating
    header the parser lifted from the source's header rows) are included; the
    value is the verbatim JSON 2-D array string the writer stored, e.g.
    ``'[["X", "Y"]]'``. A missing / unreadable / malformed sidecar degrades
    silently to ``{}`` — a missing header must never block chunking (same
    tolerance as :func:`_load_blocks_from_jsonl`).
    """
    path = _tables_sidecar_path(blocks_path)
    if not path:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    tables = data.get("tables") if isinstance(data, dict) else None
    if not isinstance(tables, dict):
        return {}
    headers: dict[str, str] = {}
    for table_id, entry in tables.items():
        if not isinstance(entry, dict):
            continue
        header = entry.get("table_header")
        if isinstance(header, str) and header.strip():
            headers[str(table_id)] = header
    return headers


def _header_rows_to_html_thead(header_rows: list[Any]) -> str:
    """Render recovered header rows (a JSON 2-D array) as an HTML ``<thead>``.

    ``table_header`` is always stored as a JSON 2-D array, so when the slice is
    an HTML table the header rows are emitted as ``<th>`` cells inside a single
    ``<thead>`` to be prepended to the slice body. Cell text is HTML-escaped.
    Returns ``""`` when no usable row is present.
    """
    trs: list[str] = []
    for row in header_rows:
        if not isinstance(row, list):
            continue
        cells = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in row)
        trs.append(f"<tr>{cells}</tr>")
    return f"<thead>{''.join(trs)}</thead>" if trs else ""


def _parse_header_rows(header_body: str) -> list[Any] | None:
    """Parse a stored ``table_header`` JSON 2-D array, or ``None`` if unusable."""
    try:
        header_rows = json.loads(header_body)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(header_rows, list) or not header_rows:
        return None
    return header_rows


def _header_reserve_tokens(
    header_rows: list[Any], fmt: str | None, tokenizer: Tokenizer
) -> int:
    """Tokens to reserve out of a slice's body budget for the recovered header.

    Conservative on purpose: for JSON we reserve the cost of the header rows
    serialized on their own (``json.dumps(header_rows)``); the in-place form
    ``header_rows + data`` is actually a couple of tokens cheaper (one shared
    set of brackets), so reserving the standalone cost can only over-budget,
    never under-budget — the safe direction for the cap.
    """
    if fmt == "json":
        return _count_tokens(tokenizer, json.dumps(header_rows, ensure_ascii=False))
    if fmt == "html":
        return _count_tokens(tokenizer, _header_rows_to_html_thead(header_rows))
    return 0


def _inject_header_into_table_slice(text: str, header_body: str) -> str | None:
    """Prepend the recovered header rows into a table slice's body, in place.

    Rebuilds ``<table {attrs}>{header rows}{original body}</table>`` keeping the
    slice's own ``attrs`` (hence its ``id``). For JSON tables the header rows are
    prepended to the row array; for HTML tables they are emitted as a leading
    ``<thead>``. Returns ``None`` when the slice or header cannot be parsed, so
    the caller leaves the slice untouched.
    """
    match = _TABLE_TAG_RE.match((text or "").strip())
    if not match:
        return None
    header_rows = _parse_header_rows(header_body)
    if header_rows is None:
        return None
    attrs = match.group("attrs")
    body = match.group("body")
    fmt = _detect_table_format(attrs, body)
    if fmt == "json":
        try:
            rows = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(rows, list):
            return None
        merged = json.dumps(header_rows + rows, ensure_ascii=False)
        return f"<table {attrs}>{merged}</table>"
    if fmt == "html":
        thead = _header_rows_to_html_thead(header_rows)
        if not thead:
            return None
        return f"<table {attrs}>{thead}{body}</table>"
    return None


def _split_html_rows_by_tokens(
    rows: list[tuple[str, str]],
    tokenizer: Tokenizer,
    *,
    target_max: int,
    target_ideal: int,
    last_min: int,
) -> list[list[tuple[str, str]]]:
    """HTML-tuple analog of :func:`_split_rows_by_tokens`.

    Same balanced-split + tail-merge algorithm; tokens are measured on
    the row payloads (``tr_str``) only — wrapper overhead is amortised
    later by the per-chunk serialiser plus the re-split-on-overflow
    safety net in :func:`_split_table_text`.
    """
    total = _count_tokens(tokenizer, "".join(tr for _, tr in rows))
    if total <= target_max or len(rows) <= 1:
        return [rows]

    target_chunks = max(
        math.ceil(total / target_ideal),
        math.ceil(total / target_max),
    )
    target_chunks = min(target_chunks, len(rows))
    target_rows = len(rows) / target_chunks

    chunks: list[list[tuple[str, str]]] = []
    start = 0
    for i in range(target_chunks):
        if i == target_chunks - 1:
            end = len(rows)
        else:
            end = max(start + 1, min(int((i + 1) * target_rows), len(rows)))
            remaining = len(rows) - end
            if remaining > 0 and remaining < target_rows * 0.3:
                end = len(rows)
        chunks.append(rows[start:end])
        start = end
        if start >= len(rows):
            break

    if len(chunks) >= 2:
        last_text = "".join(tr for _, tr in chunks[-1])
        if _count_tokens(tokenizer, last_text) < last_min:
            merged = chunks[-2] + chunks[-1]
            merged_tokens = _count_tokens(tokenizer, "".join(tr for _, tr in merged))
            if merged_tokens <= target_max:
                chunks[-2] = merged
                chunks.pop()
    return chunks


def _dedup_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _new_block(
    *,
    heading: str,
    parent_headings: list[str],
    level: int,
    paragraphs: list[dict[str, Any]],
    table_chunk_role: str,
    tokenizer: Tokenizer,
    blockids: list[str] | None = None,
) -> dict[str, Any]:
    content = "\n".join(p["text"] for p in paragraphs)
    return {
        "heading": heading,
        "parent_headings": list(parent_headings),
        "level": level,
        "paragraphs": list(paragraphs),
        "content": content,
        "tokens": _count_tokens(tokenizer, content),
        "table_chunk_role": table_chunk_role,
        # Ordered list of source blockids (deduped). Empty when the input
        # blocks.jsonl row did not carry a blockid (raw/legacy input).
        "blockids": _dedup_preserving_order(list(blockids or [])),
    }


# ---------------------------------------------------------------------------
# TableRowSplit — oversized-table re-split with first/middle/last gluing.
# ---------------------------------------------------------------------------


def _split_rows_by_tokens(
    rows: list[Any],
    tokenizer: Tokenizer,
    *,
    target_max: int,
    target_ideal: int,
    last_min: int,
) -> list[list[Any]]:
    """Split ``rows`` into balanced row-bounded chunks (TableRowSplit core)."""
    total = _count_tokens(tokenizer, json.dumps(rows, ensure_ascii=False))
    if total <= target_max or len(rows) <= 1:
        return [rows]

    target_chunks = max(
        math.ceil(total / target_ideal),
        math.ceil(total / target_max),
    )
    # Cap at len(rows) so target_rows >= 1; otherwise int((i+1)*target_rows)
    # can collapse to ``start`` and emit empty <table>[]</table> slices.
    target_chunks = min(target_chunks, len(rows))
    target_rows = len(rows) / target_chunks

    chunks: list[list[Any]] = []
    start = 0
    for i in range(target_chunks):
        if i == target_chunks - 1:
            end = len(rows)
        else:
            # max(start + 1, ...) guarantees forward progress (>= 1 row per
            # slice) even at fractional target_rows boundaries.
            end = max(start + 1, min(int((i + 1) * target_rows), len(rows)))
            remaining = len(rows) - end
            if remaining > 0 and remaining < target_rows * 0.3:
                end = len(rows)
        chunks.append(rows[start:end])
        start = end
        if start >= len(rows):
            break

    # Merge a tiny last chunk back into the previous chunk when feasible.
    if len(chunks) >= 2:
        last_json = json.dumps(chunks[-1], ensure_ascii=False)
        if _count_tokens(tokenizer, last_json) < last_min:
            merged = chunks[-2] + chunks[-1]
            merged_tokens = _count_tokens(
                tokenizer, json.dumps(merged, ensure_ascii=False)
            )
            if merged_tokens <= target_max:
                chunks[-2] = merged
                chunks.pop()
    return chunks


def _character_split_text(
    text: str,
    tokenizer: Tokenizer,
    *,
    target_max: int,
    chunk_overlap_token_size: int = 0,
) -> list[str]:
    """Character-level fallback wrapped to return plain-text pieces.

    Lazy import dodges the ``recursive_character`` ↔ ``paragraph_semantic``
    circular dependency (same pattern as the sidecar-missing fallback in
    :func:`chunking_by_paragraph_semantic`). Callers that split ordinary
    prose pass the paragraph-semantic overlap; table character fallbacks
    leave the default at zero so structured table row chunks do not gain
    implicit row-level overlap.
    """
    from lightrag.chunker.recursive_character import (
        chunking_by_recursive_character,
    )

    pieces = chunking_by_recursive_character(
        tokenizer,
        text,
        target_max,
        chunk_overlap_token_size=_bounded_overlap(target_max, chunk_overlap_token_size),
    )
    return [p["content"] for p in pieces if p.get("content")]


def _split_table_text(
    table_text: str,
    *,
    tokenizer: Tokenizer,
    target_max: int,
    target_ideal: int,
    last_min: int,
    header_body: str | None = None,
) -> list[str]:
    """Split a single oversized ``<table>...</table>`` text into ≤ target_max pieces.

    Strategy (mirrors the user-supplied contract in
    ``docs/ParagraphSemanticChunking-zh.md`` — row boundary first,
    character fallback last):

      1. Match the outer ``<table {attrs}>{body}</table>``. If the regex
         fails, character-split the original text and return.
      2. Detect the body format via :func:`_detect_table_format` (with
         body sniffing when ``attrs`` is silent).
      3. Row-boundary split: JSON via :func:`_split_rows_by_tokens`,
         HTML via :func:`_split_html_rows_by_tokens`. Re-wrap every
         row-chunk as ``<table {attrs}>{rows}</table>``.
      4. For any wrapped chunk still exceeding ``target_max``
         (single-row chunks where the row alone exceeds the cap, or
         row-split returned a single chunk because rows were ≤ 1),
         character-fallback that specific chunk's text.
      5. Unknown / unparseable format → character-fallback the entire
         original text.

    When ``header_body`` (a stored ``table_header`` JSON 2-D array string) is
    supplied, the source table carries a cross-page repeating header. The first
    slice keeps its own header rows, so the header is re-injected only into the
    *non-first* slices — and its token cost is budgeted out of the per-slice cap
    *before* splitting, so each emitted slice stays ``≤ target_max`` including
    the prepended header (HeaderRecovery, §3.3.3). A header-less table
    (``header_body is None`` / unparseable) leaves the split untouched.

    Output strings are either:
      - a re-wrapped ``<table {attrs}>{rows}</table>`` (legal markup,
        callers may keep ``is_table=True`` for these), or
      - a character-fallback fragment (no ``<table>`` wrapper, callers
        should mark ``is_table=False``).
    """
    match = _TABLE_TAG_RE.match((table_text or "").strip())
    if not match:
        return _character_split_text(table_text, tokenizer, target_max=target_max)
    attrs = match.group("attrs")
    body = match.group("body")
    fmt = _detect_table_format(attrs, body)

    # Budget the <table {attrs}></table> wrapper out of the per-chunk
    # caps before calling the row splitter — the splitter only measures
    # the body (json.dumps(rows) / "".join(rows)), so without this the
    # wrapped chunk can exceed target_max purely from the wrapper, which
    # would force a needless character-fallback below.
    wrapper_overhead = _count_tokens(tokenizer, f"<table {attrs}></table>")
    # HeaderRecovery budget: reserve room for the repeating header that will be
    # prepended into every non-first slice, so a slice + its header never
    # exceeds target_max. The first slice keeps its own header rows (already in
    # the body), so reserving uniformly only makes it marginally smaller — safe.
    header_rows = _parse_header_rows(header_body) if header_body else None
    header_overhead = (
        _header_reserve_tokens(header_rows, fmt, tokenizer) if header_rows else 0
    )
    body_budget = max(target_max - wrapper_overhead - header_overhead, 1)
    body_max = body_budget
    body_ideal = max(
        min(target_ideal, target_max) - wrapper_overhead - header_overhead, 1
    )
    body_last_min = max(last_min - wrapper_overhead - header_overhead, 1)
    row_chunks: list[list[Any]] | None = None
    serialize: Callable[[list[Any]], str] | None = None
    if fmt == "json":
        try:
            rows = json.loads(body)
        except json.JSONDecodeError:
            rows = None
        if isinstance(rows, list) and len(rows) > 1:
            row_chunks = _split_rows_by_tokens(
                rows,
                tokenizer,
                target_max=body_max,
                target_ideal=body_ideal,
                last_min=body_last_min,
            )

            def serialize(chunk_rows: list[Any]) -> str:
                return (
                    f"<table {attrs}>"
                    f"{json.dumps(chunk_rows, ensure_ascii=False)}"
                    f"</table>"
                )

    elif fmt == "html":
        rows_html = _split_html_rows(body)
        if rows_html and len(rows_html) > 1:
            row_chunks = _split_html_rows_by_tokens(
                rows_html,
                tokenizer,
                target_max=body_max,
                target_ideal=body_ideal,
                last_min=body_last_min,
            )

            def serialize(chunk_rows: list[tuple[str, str]]) -> str:
                return (
                    f"<table {attrs}>"
                    f"{_serialize_rows_with_wrappers(chunk_rows)}"
                    f"</table>"
                )

    if row_chunks is None or serialize is None:
        # No row boundary available (single-row table, parse failure,
        # unknown format) → character-fallback the whole text.
        return _character_split_text(table_text, tokenizer, target_max=target_max)

    # Re-split any chunk whose wrapped form still exceeds target_max
    # before resorting to character-level shredding. The row splitter's
    # balanced-cut heuristic can produce uneven chunks when row sizes
    # vary, and only a chunk that has collapsed to a single row (where
    # row-boundary splitting can no longer reduce it) belongs in the
    # character fallback.
    pieces: list[str] = []
    pending: list[list[Any]] = list(row_chunks)
    while pending:
        chunk_rows = pending.pop(0)
        wrapped = serialize(chunk_rows)
        if _count_tokens(tokenizer, wrapped) <= target_max:
            pieces.append(wrapped)
            continue
        if len(chunk_rows) <= 1:
            pieces.extend(
                _character_split_text(wrapped, tokenizer, target_max=target_max)
            )
            continue
        # Force a finer cut: cap the next-pass body budget at half the
        # current wrapped size so target_chunks >= 2 inside the splitter.
        # This guarantees forward progress (one row at minimum per
        # sub-chunk, see the splitter's len(rows) cap).
        halved = max(_count_tokens(tokenizer, wrapped) // 2, 1)
        sub_max = max(min(body_max, halved), 1)
        sub_ideal = max(sub_max // 2, 1)
        sub_last_min = max(min(body_last_min, sub_max // 2), 1)
        if fmt == "json":
            sub_chunks = _split_rows_by_tokens(
                chunk_rows,
                tokenizer,
                target_max=sub_max,
                target_ideal=sub_ideal,
                last_min=sub_last_min,
            )
        else:
            sub_chunks = _split_html_rows_by_tokens(
                chunk_rows,
                tokenizer,
                target_max=sub_max,
                target_ideal=sub_ideal,
                last_min=sub_last_min,
            )
        if len(sub_chunks) <= 1:
            # The splitter could not reduce further (e.g. one row already
            # dominates the body). Avoid an infinite loop and let the
            # character fallback handle this stubborn chunk.
            pieces.extend(
                _character_split_text(wrapped, tokenizer, target_max=target_max)
            )
            continue
        # Process the finer cuts before any remaining peer chunks so the
        # output keeps source order.
        pending[0:0] = sub_chunks

    # HeaderRecovery: re-inject the source table's repeating header into every
    # non-first slice (the first slice kept its own header rows). The header's
    # tokens were budgeted out above, so in the common case the rebuilt slice
    # stays ≤ target_max. The overflow-repair loop validates each piece against
    # target_max *before* this prepend, so a near-cap single-row slice the
    # splitter could not reduce further would otherwise exceed the cap once the
    # header is added — guard against that by re-checking the rebuilt size and
    # skipping injection (leaving the slice header-less but cap-compliant) when
    # it would overflow. A character-fallback fragment (no <table> wrapper)
    # returns None and is left untouched.
    if header_body and len(pieces) > 1:
        for i in range(1, len(pieces)):
            rebuilt = _inject_header_into_table_slice(pieces[i], header_body)
            if rebuilt is not None and _count_tokens(tokenizer, rebuilt) <= target_max:
                pieces[i] = rebuilt
    return pieces


def _expand_block_with_table_splits(
    block: dict[str, Any],
    *,
    tokenizer: Tokenizer,
    table_max: int,
    table_ideal: int,
    table_min_last: int,
    target_max: int | None = None,
    chunk_overlap_token_size: int = 0,
    table_headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Apply TableRowSplit to one heading-driven block.

    For every embedded table whose tokens exceed ``table_max``:
      - the first row-slice glues with paragraphs already accumulated in
        the current expansion (i.e. content *before* the table);
      - middle slices are emitted as standalone blocks tagged
        ``table_chunk_role == "middle"`` so LevelMerge refuses to merge them;
      - the last slice begins a fresh accumulation that will glue with
        paragraphs *after* the table.

    When a ``last`` table slice is followed by short bridge text and then
    another oversized table's ``first`` slice, the bridge text is split
    into table boundary context: a prefix may be duplicated into the
    previous table block and a suffix into the next table block. If the
    bridge is longer than both context budgets, the remaining middle text
    is emitted as a standalone text block. Tables within the size limit
    pass through untouched.
    """
    if target_max is None:
        target_max = table_max
    target_max = max(int(target_max), 1)
    context_overlap = _bounded_overlap(target_max, chunk_overlap_token_size)
    sep_tokens = _count_tokens(tokenizer, "\n")
    paragraphs = block["paragraphs"]
    has_oversized_table = any(
        p["is_table"] and _count_tokens(tokenizer, p["text"]) > table_max
        for p in paragraphs
    )
    if not has_oversized_table:
        return [block]

    out: list[dict[str, Any]] = []
    cur_paras: list[dict[str, Any]] = []
    # Role to assign to ``cur_paras`` when it next flushes. Tracks the
    # boundary semantics across split-table iterations so the merged
    # block carries "first" / "last" instead of defaulting to "none" —
    # otherwise LevelMerge's directional protections (a "first" block must
    # not absorb backward, a "last" block must not absorb forward) silently
    # disappear after the slice glues with surrounding paragraphs.
    cur_role = "none"

    def flush_cur() -> None:
        nonlocal cur_role
        if not cur_paras:
            cur_role = "none"
            return
        out.append(
            _new_block(
                heading=block["heading"],
                parent_headings=block["parent_headings"],
                level=block["level"],
                paragraphs=cur_paras,
                table_chunk_role=cur_role,
                tokenizer=tokenizer,
                blockids=block.get("blockids"),
            )
        )
        cur_paras.clear()
        cur_role = "none"

    def _append_bridge_block(
        paragraphs: list[dict[str, Any]],
        table_chunk_role: str,
    ) -> None:
        if not paragraphs:
            return
        out.append(
            _new_block(
                heading=block["heading"],
                parent_headings=block["parent_headings"],
                level=block["level"],
                paragraphs=paragraphs,
                table_chunk_role=table_chunk_role,
                tokenizer=tokenizer,
                blockids=block.get("blockids"),
            )
        )

    def _text_paragraph(text: str) -> dict[str, Any] | None:
        if not text or not text.strip():
            return None
        return {"text": text, "is_table": False}

    def _context_capacity(base_paras: list[dict[str, Any]]) -> int:
        if context_overlap <= 0:
            return 0
        base_text = "\n".join(p["text"] for p in base_paras)
        base_tokens = _count_tokens(tokenizer, base_text)
        if base_tokens >= target_max:
            return 0
        # The context paragraph is joined to the table fragment with "\n".
        # Cap each side at target_max // 2 as well so the duplicated bridge
        # text can never dominate the whole block (§3.3.2).
        return max(
            min(
                context_overlap,
                target_max - base_tokens - sep_tokens,
                target_max // 2,
            ),
            0,
        )

    def _flush_last_bridge_before_next_first(
        next_first_para: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Flush ``last + bridge`` before a following table ``first``.

        Returns context paragraphs to prepend to the following first-table
        block. Only non-table bridge paragraphs are duplicated/sliced; if
        the bridge contains tables we keep the prior non-overlapping flush.
        """
        nonlocal cur_role
        if not cur_paras:
            cur_role = "none"
            return []

        seed_paras = [cur_paras[0]]
        bridge_paras = cur_paras[1:]
        if (
            context_overlap <= 0
            or not bridge_paras
            or any(p.get("is_table", False) for p in bridge_paras)
        ):
            flush_cur()
            return []

        bridge_text = "\n".join(p["text"] for p in bridge_paras)
        bridge_tokens = tokenizer.encode(bridge_text)
        if not bridge_tokens:
            flush_cur()
            return []

        prev_budget = _context_capacity(seed_paras)
        next_budget = _context_capacity([next_first_para])
        bridge_len = len(bridge_tokens)

        if bridge_len <= prev_budget and bridge_len <= next_budget:
            prefix_text = bridge_text
            suffix_text = bridge_text
            middle_text = ""
        else:
            prefix_len = min(prev_budget, bridge_len)
            suffix_len = min(next_budget, bridge_len)
            middle_start = prefix_len
            middle_end = max(middle_start, bridge_len - suffix_len)

            prefix_text = (
                tokenizer.decode(bridge_tokens[:prefix_len]) if prefix_len else ""
            )
            suffix_text = (
                tokenizer.decode(bridge_tokens[bridge_len - suffix_len :])
                if suffix_len
                else ""
            )
            # The standalone middle block keeps R-style overlap with the text
            # that went left (into the previous table block) and right (into the
            # next table block): extend it by ``context_overlap`` tokens on each
            # side, clamped inside ``bridge_tokens``. Because the indices never
            # leave the bridge — which by this branch contains no table
            # paragraphs — the overlap is pure text and never duplicates any
            # ``<table>`` content into the middle block.
            mid_lo = max(0, middle_start - context_overlap)
            mid_hi = min(bridge_len, middle_end + context_overlap)
            middle_text = (
                tokenizer.decode(bridge_tokens[mid_lo:mid_hi])
                if mid_hi > mid_lo and middle_end > middle_start
                else ""
            )

        prev_paras = list(seed_paras)
        prefix_para = _text_paragraph(prefix_text)
        if prefix_para is not None:
            prev_paras.append(prefix_para)
        _append_bridge_block(prev_paras, "last")

        middle_para = _text_paragraph(middle_text)
        if middle_para is not None:
            _append_bridge_block([middle_para], "none")

        cur_paras.clear()
        cur_role = "none"

        suffix_para = _text_paragraph(suffix_text)
        return [suffix_para] if suffix_para is not None else []

    for para in paragraphs:
        text = para["text"]
        if not (para["is_table"] and _count_tokens(tokenizer, text) > table_max):
            cur_paras.append(para)
            continue

        # Trace this oversized table back to its tables.json entry via the
        # slice's <table> id, so the splitter can budget + re-inject the source
        # table's repeating header into the non-first slices (HeaderRecovery).
        header_body: str | None = None
        if table_headers:
            tag_match = _TABLE_TAG_RE.match(text.strip())
            if tag_match:
                table_id = _extract_table_id(tag_match.group("attrs"))
                if table_id:
                    header_body = table_headers.get(table_id)

        # Row-boundary first, character fallback last. ``_split_table_text``
        # returns one or more strings: row-wrapped ``<table>...</table>``
        # fragments where row-splitting succeeded, plain text where it
        # had to character-split (single-row tables, parse failures,
        # rows whose own size exceeded ``table_max``).
        pieces = _split_table_text(
            text,
            tokenizer=tokenizer,
            target_max=table_max,
            target_ideal=table_ideal,
            last_min=table_min_last,
            header_body=header_body,
        )
        if len(pieces) <= 1:
            # No reduction was possible (e.g. very small unparseable table
            # that already fits within ``table_max`` after a no-op character
            # fallback). Keep the original paragraph to preserve content.
            cur_paras.append(para)
            continue

        for chunk_idx, piece_text in enumerate(pieces):
            stripped = piece_text.strip()
            is_still_table = stripped.startswith("<table ") and stripped.endswith(
                "</table>"
            )
            chunk_para = {"text": piece_text, "is_table": is_still_table}
            is_first = chunk_idx == 0
            is_last = chunk_idx == len(pieces) - 1

            if is_first:
                # First slice glues with everything currently accumulated
                # (= the paragraphs that appeared before the table inside
                # this heading block). If the buffer still carries the
                # "last" tail of a previous oversized table, flush it first
                # so its protective role survives instead of being
                # overwritten by "first".
                if cur_role == "last":
                    cur_paras.extend(_flush_last_bridge_before_next_first(chunk_para))
                cur_paras.append(chunk_para)
                cur_role = "first"
            elif is_last:
                # Flush the accumulated "first-glued" block, then begin a
                # new accumulation seeded with this last slice — it will
                # absorb the paragraphs that appear after the table.
                flush_cur()
                cur_paras.append(chunk_para)
                cur_role = "last"
            else:
                # Middle slice: flush the first-glued block, then emit
                # this middle slice as a standalone block that LevelMerge
                # MUST keep intact (table_chunk_role == "middle").
                flush_cur()
                out.append(
                    _new_block(
                        heading=block["heading"],
                        parent_headings=block["parent_headings"],
                        level=block["level"],
                        paragraphs=[chunk_para],
                        table_chunk_role="middle",
                        tokenizer=tokenizer,
                        blockids=block.get("blockids"),
                    )
                )

    flush_cur()
    return out


# ---------------------------------------------------------------------------
# AnchorSplit — anchor-driven long-block re-split.
# ---------------------------------------------------------------------------


def _split_long_block(
    paragraphs: list[dict[str, Any]],
    heading: str,
    parent_headings: list[str],
    level: int,
    table_chunk_role: str,
    *,
    tokenizer: Tokenizer,
    target_max: int,
    target_ideal: int,
    chunk_overlap_token_size: int = 100,
    blockids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Split an oversized block into balanced sub-blocks at short-paragraph anchors.

    Mirrors :func:`lightrag.parser.docx.parse_document.split_long_block`,
    parameterised on ``target_max`` / ``target_ideal``. Tables (``is_table``)
    are excluded from the anchor candidate pool, so TableRowSplit's row-level
    splits stay intact. When no anchor exists (including the single-
    paragraph oversized case), the no-anchor branch below honors the cap
    via row-boundary splitting (for tables) or character-level splitting
    (for prose). The audit-mode parser would ``sys.exit(1)`` on no-anchor
    failure, but the RAG pipeline must never drop a document silently.
    Character-level splitting of ordinary prose uses
    ``chunk_overlap_token_size`` so long text under one JSONL content row
    keeps semantic continuity across adjacent chunks.
    """
    chunk_overlap_token_size = _bounded_overlap(target_max, chunk_overlap_token_size)
    content = "\n".join(p["text"] for p in paragraphs)
    total = _count_tokens(tokenizer, content)
    if total <= target_max:
        return [
            _new_block(
                heading=heading,
                parent_headings=parent_headings,
                level=level,
                paragraphs=paragraphs,
                table_chunk_role=table_chunk_role,
                tokenizer=tokenizer,
                blockids=blockids,
            )
        ]

    target_blocks = max(
        math.ceil(total / target_ideal),
        math.ceil(total / target_max),
    )
    target_size = total / target_blocks

    # Build anchor candidates with cumulative token offsets. Index 0 is
    # excluded: an anchor at the first paragraph yields an empty leading
    # slice and a tail equal to the input, so it cannot divide the block —
    # selecting it would re-enter this function with the same arguments
    # and recurse until RecursionError.
    candidates: list[dict[str, Any]] = []
    cumulative = 0
    for idx, para in enumerate(paragraphs):
        text = para["text"]
        if (
            idx > 0
            and not para.get("is_table", False)
            and 0 < len(text) <= _MAX_ANCHOR_CANDIDATE_LENGTH
        ):
            candidates.append({"index": idx, "text": text, "position": cumulative})
        cumulative += _count_tokens(tokenizer, text)

    if not candidates:
        # All paragraphs in the block are longer than the anchor-length
        # cap (typical for dense academic prose: every paragraph is a
        # full body section).  Anchor-driven splitting cannot proceed,
        # but we must NOT emit a single oversized chunk: the
        # embedding-time hard fallback uses ``embedding_token_limit``
        # (often 8K), not ``chunk_token_size``, so the chunk would
        # silently exceed the user-configured size.  Prefer
        # row-boundary splitting on any oversized table paragraph
        # before falling back to character-level splitting on residual
        # content — character splitting destroys ``<table>`` markup
        # mid-tag and produces fragments LLMs can't interpret as
        # tables.
        logger.warning(
            "[paragraph_semantic_chunking] block under heading %r exceeds "
            "target_max=%d tokens (~%d tokens) but has no eligible anchor "
            "paragraph (≤ %d chars); preferring table row-boundary split, "
            "falling back to recursive-character splitting on residual "
            "content.",
            heading,
            target_max,
            total,
            _MAX_ANCHOR_CANDIDATE_LENGTH,
        )

        # Step 1: expand each oversized table paragraph into row-bounded
        # pieces; non-table or in-budget paragraphs pass through verbatim.
        # ``last_min`` mirrors TableRowSplit's ratio (no separate constant — the
        # tail-merge threshold is purely a row-balancing heuristic).
        last_min = max(int(target_max * _TABLE_MIN_LAST_RATIO), 1)
        pieces: list[str] = []
        for para in paragraphs:
            text = para["text"]
            if (
                para.get("is_table", False)
                and _count_tokens(tokenizer, text) > target_max
            ):
                pieces.extend(
                    _split_table_text(
                        text,
                        tokenizer=tokenizer,
                        target_max=target_max,
                        target_ideal=target_ideal,
                        last_min=last_min,
                    )
                )
            else:
                pieces.append(text)

        # Step 2: greedy-pack pieces into chunks ≤ target_max. A piece
        # that is itself oversized (e.g. a single dense prose paragraph
        # without short anchors) is character-split via
        # :func:`chunking_by_recursive_character` after flushing the
        # current buffer. The "\n" separator inserted by ``"\n".join(buf)``
        # also costs tokens, so it must be debited from the budget —
        # otherwise two pieces that sum to exactly target_max would
        # overflow once joined.
        sep_tokens = _count_tokens(tokenizer, "\n")
        chunks_text: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for piece in pieces:
            piece_tokens = _count_tokens(tokenizer, piece)
            if piece_tokens > target_max:
                if buf:
                    chunks_text.append("\n".join(buf))
                    buf, buf_tokens = [], 0
                chunks_text.extend(
                    _character_split_text(
                        piece,
                        tokenizer,
                        target_max=target_max,
                        chunk_overlap_token_size=chunk_overlap_token_size,
                    )
                )
                continue
            addition = piece_tokens + (sep_tokens if buf else 0)
            if buf and buf_tokens + addition > target_max:
                chunks_text.append("\n".join(buf))
                buf, buf_tokens = [], 0
                addition = piece_tokens
            buf.append(piece)
            buf_tokens += addition
        if buf:
            chunks_text.append("\n".join(buf))

        if not chunks_text:
            # Defensive: every piece was empty after stripping. Emit the
            # original oversized block so the document is never silently
            # dropped (matches the prior behaviour of the empty-R branch).
            return [
                _new_block(
                    heading=heading,
                    parent_headings=parent_headings,
                    level=level,
                    paragraphs=paragraphs,
                    table_chunk_role=table_chunk_role,
                    tokenizer=tokenizer,
                    blockids=blockids,
                )
            ]

        sub_blocks: list[dict[str, Any]] = []
        for i, chunk_text in enumerate(chunks_text):
            stripped = chunk_text.strip()
            is_still_table = stripped.startswith("<table ") and stripped.endswith(
                "</table>"
            )
            sub_blocks.append(
                _new_block(
                    heading=heading,
                    parent_headings=parent_headings,
                    level=level,
                    paragraphs=[{"text": chunk_text, "is_table": is_still_table}],
                    # Only the first sub-block keeps the inbound
                    # table_chunk_role; the rest are text-only by
                    # construction (mirrors the anchor-split path below).
                    table_chunk_role=table_chunk_role if i == 0 else "none",
                    tokenizer=tokenizer,
                    blockids=blockids,
                )
            )
        return sub_blocks

    # Pick the anchors closest to evenly-spaced ideal positions.
    pool = list(candidates)
    selected: list[dict[str, Any]] = []
    for i in range(1, target_blocks):
        if not pool:
            break
        ideal_position = i * target_size
        best = min(pool, key=lambda c: abs(c["position"] - ideal_position))
        selected.append(best)
        pool.remove(best)
    selected.sort(key=lambda c: c["index"])

    sub_blocks: list[dict[str, Any]] = []
    prev_idx = 0
    cur_heading = heading
    cur_parents = list(parent_headings)
    # Only the first sub-block keeps the inbound table_chunk_role; the
    # post-anchor sub-blocks are text-only by construction.
    cur_role = table_chunk_role

    for anchor in selected:
        split_idx = anchor["index"]
        slice_paras = paragraphs[prev_idx:split_idx]
        if slice_paras:
            sub_blocks.append(
                _new_block(
                    heading=cur_heading,
                    parent_headings=cur_parents,
                    level=level,
                    paragraphs=slice_paras,
                    table_chunk_role=cur_role,
                    tokenizer=tokenizer,
                    blockids=blockids,
                )
            )
        # Anchor becomes the first paragraph (and heading) of the next sub-block.
        cur_parents = (
            list(parent_headings) + [heading]
            if heading and cur_heading == heading
            else list(cur_parents)
        )
        cur_heading = anchor["text"]
        cur_role = "none"
        prev_idx = split_idx

    tail = paragraphs[prev_idx:]
    if tail:
        sub_blocks.append(
            _new_block(
                heading=cur_heading,
                parent_headings=cur_parents,
                level=level,
                paragraphs=tail,
                table_chunk_role=cur_role,
                tokenizer=tokenizer,
                blockids=blockids,
            )
        )

    # Recursive guard: any sub-block still over target_max is re-split,
    # including single-paragraph subs — the no-anchor branch above honors
    # the cap via row-boundary or character-level splitting and is the
    # only path that can shrink them.
    out: list[dict[str, Any]] = []
    for sub in sub_blocks:
        if sub["tokens"] > target_max:
            out.extend(
                _split_long_block(
                    sub["paragraphs"],
                    sub["heading"],
                    sub["parent_headings"],
                    sub["level"],
                    sub["table_chunk_role"],
                    tokenizer=tokenizer,
                    target_max=target_max,
                    target_ideal=target_ideal,
                    chunk_overlap_token_size=chunk_overlap_token_size,
                    blockids=sub.get("blockids") or blockids,
                )
            )
        else:
            out.append(sub)
    return out


# ---------------------------------------------------------------------------
# LevelMerge — bottom-up, level-aware small-block merging.
# ---------------------------------------------------------------------------


def _can_merge_forward(role: str) -> bool:
    # A split-table slice (first/middle/last) is frozen against LevelMerge: its
    # repeating header is already injected into the slice at TableRowSplit time,
    # so re-merging two slices of one table would duplicate the header mid-body.
    # Only ordinary blocks (role "none") may absorb a following block.
    return role == "none"


def _can_merge_backward(role: str) -> bool:
    # Symmetric to _can_merge_forward — only role "none" may be absorbed by a
    # preceding block. Split-table slices never re-merge (see §3.6 / §3.3.3).
    return role == "none"


def _same_parent_path(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when two blocks share the identical parent-heading chain.

    Same-level merging (Phase A, tail absorption) is gated on this so two
    blocks at the same ``level`` are only fused when they are true siblings
    under one parent — blocks that merely happen to share a ``level`` but sit
    under different parents (e.g. ``2.4.1`` vs ``2.5.1``) are NOT merged,
    which is the documented anti-cross-topic-pollution guarantee (§3.6.1 #4).
    Blocks with no parents (preamble / non-hierarchical input) compare equal.
    """
    return list(a.get("parent_headings") or []) == list(b.get("parent_headings") or [])


def _is_descendant(shallow: dict[str, Any], deep: dict[str, Any]) -> bool:
    """True when ``deep`` is nested under ``shallow`` in the heading tree.

    Cross-level absorption (Phase B, shallower-absorbs-deeper) is gated on
    this: the shallow block's full heading path (its ``parent_headings`` plus
    its own ``heading``) must be a prefix of the deep block's
    ``parent_headings``. This prevents a shallow block from swallowing a deeper
    block that belongs to an unrelated branch of the document tree. The
    shallow heading is stripped of any generated ``[part n]`` suffix first
    because ``parent_headings`` never carry that suffix. A shallow block with
    no heading (preamble) yields an empty path that prefixes anything, so such
    blocks are not blocked from absorbing — they have no hierarchy to violate.
    """
    head = _strip_generated_heading_suffixes(shallow.get("heading") or "")
    shallow_full = list(shallow.get("parent_headings") or []) + ([head] if head else [])
    deep_parents = list(deep.get("parent_headings") or [])
    return deep_parents[: len(shallow_full)] == shallow_full


def _merged_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    keep: str,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    base = left if keep == "left" else right
    paragraphs = list(left["paragraphs"]) + list(right["paragraphs"])
    content = left["content"] + "\n\n" + right["content"]
    merged_blockids = _dedup_preserving_order(
        list(left.get("blockids") or []) + list(right.get("blockids") or [])
    )
    return {
        "heading": base["heading"],
        "parent_headings": list(base["parent_headings"]),
        "level": base["level"],
        "paragraphs": paragraphs,
        "content": content,
        "tokens": _count_tokens(tokenizer, content),
        "table_chunk_role": "none",
        "blockids": merged_blockids,
    }


def _is_heading_only(block: dict[str, Any]) -> bool:
    """Return True when ``block`` carries a heading but no body content.

    A blocks.jsonl content row always renders its heading as the first line
    (``render_heading_line`` → ``"#" * level + " " + text``) and appends body
    paragraphs after it, so a heading-only section's ``content`` consists
    solely of heading lines. This still holds after two heading-only blocks
    are glued together, letting :func:`_glue_heading_only_blocks` cascade
    down a chain of bare ancestor headings.

    The ``heading`` guard excludes preamble blocks (text before any heading)
    and AnchorSplit anchor sub-blocks, whose body paragraphs are prose rather
    than heading lines.

    Note: a body line that genuinely begins with ``#`` + space is the same
    accepted ambiguity documented in ``lightrag/parser/_markdown.py`` and is
    treated as a heading line here too — rare prose, tolerated.
    """
    if not block.get("heading"):
        return False
    saw_line = False
    for line in (block.get("content") or "").split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        saw_line = True
        if not _HEADING_LINE_RE.match(stripped):
            return False
    return saw_line


def _glue_heading_only_blocks(
    blocks: list[dict[str, Any]],
    *,
    tokenizer: Tokenizer,
    target_max: int,
    target_ideal: int,
    chunk_overlap_token_size: int = 0,
) -> list[dict[str, Any]]:
    """Glue each body-less heading block forward into its deeper child block.

    A section heading with no body of its own (e.g. ``## 2.4`` immediately
    followed by ``### 2.4.1``) must travel WITH its first child rather than be
    left as a lone trailing heading or glued onto an unrelated same-level
    sibling (e.g. ``## 2.3``). Running this before LevelMerge bonds the bare
    heading to its child first. See ``docs/ParagraphSemanticChunking-zh.md``
    §3.5 for the full rationale.

    For a block that :func:`_is_heading_only` recognises whose NEXT block is
    STRICTLY DEEPER and whose ``table_chunk_role`` is ``none`` or ``first``,
    the pair merges with ``keep="left"`` — preserving the shallower parent's
    identity (``heading`` / ``level`` / ``parent_headings``) while appending the
    child's content. The merge is re-evaluated in place, so a chain of bare
    ancestors (``# 2`` → ``## 2.4`` → ``### 2.4.1``) collapses into one block
    keeping the shallowest identity. (Only ``none`` / ``first`` can follow a
    heading-only row — ``middle`` / ``last`` occur only inside one row's table.
    Gluing into a ``first`` slice KEEPS the ``first`` role so LevelMerge still
    cannot absorb it backward, protecting the table boundary.)

    Gluing is FORWARD only: a body-less heading whose next block is not deeper
    (a shallower/sibling heading, or end of list) is left for LevelMerge, not
    pulled backward into a deeper previous block (which would invert the
    hierarchy and demote the heading's level).

    Prepending the heading line(s) can tip the bonded block past ``target_max``,
    and nothing downstream re-splits an oversized chunk, so :func:`_split_to_cap`
    re-splits it here while keeping the heading attached to real body; every
    emitted piece honours ``target_max``. Because ``keep="left"`` preserves the
    parent's ``level``, the bonded group remains an ordinary small block (NOT
    pinned independent) that LevelMerge may still merge backward by its usual
    peer/tail rules — this pre-pass only guarantees the heading is never
    separated FROM its child.
    """
    if len(blocks) <= 1:
        return blocks

    out: list[dict[str, Any]] = []

    def _split_full(paras: list[dict[str, Any]], block: dict[str, Any], **kw):
        args = {
            "target_max": target_max,
            "target_ideal": target_ideal,
            "chunk_overlap_token_size": chunk_overlap_token_size,
            "blockids": block.get("blockids"),
        }
        args.update(kw)
        return _split_long_block(
            paras,
            block["heading"],
            block["parent_headings"],
            block["level"],
            block.get("table_chunk_role", "none"),
            tokenizer=tokenizer,
            **args,
        )

    def _split_to_cap(block: dict[str, Any]) -> list[dict[str, Any]]:
        # The bonded block exceeds target_max. Peel the leading run of
        # heading-line paragraphs (the prepended parent heading plus the child's
        # own heading line, or a chain) off the body, split only the BODY, then
        # glue the heading prefix back onto the FIRST body piece. The prefix is
        # never handed to the splitter, so the bare heading can never be sliced
        # off as a content-less first chunk — a heading-only orphan that LevelMerge
        # would re-absorb backward into the previous sibling. (Fusing the prefix
        # into the body instead does NOT work: when the fused paragraph itself
        # exceeds the cap, char-splitting re-separates the headings at their
        # newline and the orphan returns.)
        paras = block["paragraphs"]
        n = 0
        for para in paras:
            if para.get("is_table", False) or not _HEADING_LINE_RE.match(
                para["text"].strip()
            ):
                break
            n += 1
        prefix, body = paras[:n], paras[n:]
        prefix_tokens = _count_tokens(tokenizer, "\n".join(p["text"] for p in prefix))
        sep_tokens = _count_tokens(tokenizer, "\n")

        if not prefix or not body or prefix_tokens + sep_tokens >= target_max:
            # No prefix to protect; no body to anchor on; OR the prefix alone
            # fills/exceeds the cap (a very long title, or a tiny
            # chunk_token_size) so it cannot be kept whole. The hard cap wins
            # over heading-intactness: split the whole block directly —
            # _split_long_block char-splits any oversized heading line so every
            # emitted piece still honours target_max.
            return _split_full(paras, block)

        # Split the body at the FULL target_max so later body pieces (which do
        # NOT carry the prefix) keep the full budget — never shrunk to the
        # leftover first-chunk budget. Reserve room for the prefix only on the
        # first piece, re-splitting that one piece if it cannot also hold it.
        pieces = _split_full(body, block)
        first, rest = pieces[0], list(pieces[1:])
        if prefix_tokens + sep_tokens + first["tokens"] > target_max:
            reduced_max = max(target_max - prefix_tokens - sep_tokens, 1)
            refit = _split_full(
                first["paragraphs"],
                block,
                target_max=reduced_max,
                target_ideal=min(target_ideal, reduced_max),
                blockids=first.get("blockids") or block.get("blockids"),
            )
            first, rest = refit[0], list(refit[1:]) + rest
        rebuilt = _new_block(
            heading=block["heading"],
            parent_headings=block["parent_headings"],
            level=block["level"],
            paragraphs=prefix + first["paragraphs"],
            table_chunk_role=block.get("table_chunk_role", "none"),
            tokenizer=tokenizer,
            blockids=first.get("blockids") or block.get("blockids"),
        )
        return [rebuilt, *rest]

    def _emit(block: dict[str, Any], *, glued: bool) -> None:
        # A forward-glued block can be tipped over target_max by its prepended
        # heading line(s); re-split via AnchorSplit so the hard cap still holds.
        if glued and block["tokens"] > target_max:
            out.extend(_split_to_cap(block))
        else:
            out.append(block)

    cur = blocks[0]
    cur_glued = False
    for nxt in blocks[1:]:
        nxt_role = nxt.get("table_chunk_role", "none")
        if (
            _is_heading_only(cur)
            and nxt.get("level", 1) > cur.get("level", 1)
            and nxt_role in ("none", "first")
        ):
            cur = _merged_pair(cur, nxt, keep="left", tokenizer=tokenizer)
            # Preserve a "first" table-slice role so the bonded block still
            # cannot be absorbed backward into the previous sibling by LevelMerge
            # (the prepended heading is exactly the preceding context a "first"
            # slice is meant to carry). "none" stays "none" — unchanged.
            cur["table_chunk_role"] = nxt_role
            cur_glued = True
        else:
            _emit(cur, glued=cur_glued)
            cur, cur_glued = nxt, False
    _emit(cur, glued=cur_glued)
    return out


def _merge_small_blocks(
    blocks: list[dict[str, Any]],
    *,
    tokenizer: Tokenizer,
    target_max: int,
    target_ideal: int,
    small_tail_threshold: int,
) -> list[dict[str, Any]]:
    """Bottom-up, level-aware small-block merging.

    Re-implementation of
    :func:`lightrag.parser.docx.parse_document.merge_small_blocks`,
    parameterised on the chunk-size targets and operating on internal
    block dicts (no ``uuid`` / ``table_header`` propagation needed: the
    chunking output schema does not carry them).
    """
    if len(blocks) <= 1:
        return blocks

    result = list(blocks)
    levels = sorted({b.get("level", 1) for b in result}, reverse=True)

    for current_level in levels:
        # Phase A — same-level merging.
        changed = True
        while changed:
            changed = False
            new_result: list[dict[str, Any]] = []
            i = 0
            while i < len(result):
                cur = result[i]
                cur_tokens = cur["tokens"]
                cur_level = cur.get("level", 1)
                cur_role = cur.get("table_chunk_role", "none")
                below_ideal = 0 < cur_tokens < target_ideal
                is_cur_lv = cur_level == current_level

                if below_ideal and is_cur_lv:
                    merged = False

                    if _can_merge_forward(cur_role) and i + 1 < len(result):
                        nxt = result[i + 1]
                        if (
                            nxt.get("level", 1) == current_level
                            and _can_merge_backward(nxt.get("table_chunk_role", "none"))
                            and _same_parent_path(cur, nxt)
                        ):
                            combined = _merged_pair(
                                cur, nxt, keep="left", tokenizer=tokenizer
                            )
                            if combined["tokens"] <= target_max:
                                new_result.append(combined)
                                i += 2
                                changed = True
                                merged = True

                    if not merged and _can_merge_backward(cur_role) and new_result:
                        prev = new_result[-1]
                        if (
                            prev.get("level", 1) == current_level
                            and _can_merge_forward(prev.get("table_chunk_role", "none"))
                            and prev["tokens"] < target_ideal
                            and _same_parent_path(prev, cur)
                        ):
                            combined = _merged_pair(
                                prev, cur, keep="left", tokenizer=tokenizer
                            )
                            if combined["tokens"] <= target_max:
                                new_result[-1] = combined
                                i += 1
                                changed = True
                                merged = True

                    if not merged:
                        new_result.append(cur)
                        i += 1
                else:
                    # Tail absorption: an at-or-above-IDEAL block can absorb
                    # a short run of subsequent same-level blocks if their
                    # combined size stays under SMALL_TAIL_THRESHOLD and
                    # fits within target_max — eliminates the document's
                    # trailing sliver of zero-content remainders.
                    if is_cur_lv and cur_tokens >= target_ideal and cur_role == "none":
                        tail_total = 0
                        end_idx = i + 1
                        for j in range(i + 1, len(result)):
                            nxt = result[j]
                            if nxt.get("level", 1) != current_level:
                                break
                            # A split-table slice (first/middle/last) is frozen —
                            # never pull one into a tail-absorption run, which
                            # would re-merge it and duplicate its header.
                            if nxt.get("table_chunk_role", "none") != "none":
                                break
                            # Same-level only is not enough — a sibling under a
                            # different parent would be cross-topic. Stop the run
                            # at the first block whose parent path diverges.
                            if not _same_parent_path(cur, nxt):
                                break
                            tail_total += nxt["tokens"]
                            end_idx = j + 1
                        if (
                            tail_total > 0
                            and tail_total < small_tail_threshold
                            and cur_tokens + tail_total <= target_max
                        ):
                            absorbed_paragraphs = list(cur["paragraphs"])
                            absorbed_content = cur["content"]
                            for j in range(i + 1, end_idx):
                                nxt = result[j]
                                absorbed_paragraphs.extend(nxt["paragraphs"])
                                absorbed_content += "\n\n" + nxt["content"]
                            # The cheap predicate above sums per-block
                            # tokens, but absorption joins blocks with
                            # ``"\n\n"`` — those separator tokens are
                            # real and can push the merged block over
                            # target_max. Re-measure the joined content
                            # before committing to absorb.
                            absorbed_tokens = _count_tokens(tokenizer, absorbed_content)
                            if absorbed_tokens <= target_max:
                                new_result.append(
                                    {
                                        "heading": cur["heading"],
                                        "parent_headings": list(cur["parent_headings"]),
                                        "level": cur["level"],
                                        "paragraphs": absorbed_paragraphs,
                                        "content": absorbed_content,
                                        "tokens": absorbed_tokens,
                                        "table_chunk_role": "none",
                                    }
                                )
                                i = end_idx
                                changed = True
                                continue
                    new_result.append(cur)
                    i += 1
            result = new_result

        # Phase B — cross-level absorption (shallower absorbs deeper).
        changed = True
        while changed:
            changed = False
            new_result = []
            i = 0
            while i < len(result):
                cur = result[i]
                cur_tokens = cur["tokens"]
                cur_level = cur.get("level", 1)
                cur_role = cur.get("table_chunk_role", "none")
                below_ideal = 0 < cur_tokens < target_ideal
                is_cur_lv = cur_level == current_level

                if below_ideal and is_cur_lv:
                    merged = False

                    if _can_merge_forward(cur_role) and i + 1 < len(result):
                        nxt = result[i + 1]
                        if (
                            nxt.get("level", 1) > current_level
                            and _can_merge_backward(nxt.get("table_chunk_role", "none"))
                            and _is_descendant(cur, nxt)
                        ):
                            combined = _merged_pair(
                                cur, nxt, keep="left", tokenizer=tokenizer
                            )
                            if combined["tokens"] <= target_max:
                                new_result.append(combined)
                                i += 2
                                changed = True
                                merged = True

                    if not merged and _can_merge_backward(cur_role) and new_result:
                        prev = new_result[-1]
                        if (
                            prev.get("level", 1) < current_level
                            and _can_merge_forward(prev.get("table_chunk_role", "none"))
                            and prev["tokens"] < target_ideal
                            and _is_descendant(prev, cur)
                        ):
                            combined = _merged_pair(
                                prev, cur, keep="left", tokenizer=tokenizer
                            )
                            if combined["tokens"] <= target_max:
                                new_result[-1] = combined
                                i += 1
                                changed = True
                                merged = True

                    if not merged:
                        new_result.append(cur)
                        i += 1
                else:
                    new_result.append(cur)
                    i += 1
            result = new_result

    return result


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def chunking_by_paragraph_semantic(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 2000,
    *,
    blocks_path: str | None = None,
    chunk_overlap_token_size: int = 100,
) -> list[dict[str, Any]]:
    """Paragraph Semantic Chunking — the ``chunking="P"`` strategy.

    Reads structured blocks from a ``.blocks.jsonl`` sidecar (HeadingBlocks
    output, emitted by any sidecar-producing parser — native / mineru /
    docling) and applies TableRowSplit (table re-split + glue), AnchorSplit
    (anchor-driven long-block re-split) and LevelMerge (bottom-up, level-aware
    merging). Output rows match the schema produced by
    :func:`lightrag.chunker.chunking_by_token_size`
    (``tokens``/``content``/``chunk_order_index``), enriched with a nested
    ``heading`` block (``{level, heading, parent_headings}``) and an optional
    ``sidecar`` tracing source blockids, so KG extraction can leverage the
    document hierarchy.

    Signature follows the LightRAG chunker contract — the standard
    prefix ``(tokenizer, content, chunk_token_size)`` is shared with
    every other chunker, while strategy-specific knobs are keyword-only:

      - ``blocks_path`` (this strategy's required input — the
        ``.blocks.jsonl`` sidecar produced at parse time)

    Knobs that ``chunking_by_token_size`` exposes for delimiter-based
    splitting (``split_by_character``, ``split_by_character_only``) are
    deliberately absent here because paragraph-semantic chunks are
    heading-aligned. ``chunk_overlap_token_size`` is supported for two
    paragraph-semantic cases where overlap preserves meaning inside one
    JSONL content row: recursive-character fallback for long prose, and
    bridge text duplicated around adjacent oversized table boundary chunks.
    When one original ``blocks.jsonl`` content row is split into multiple
    fragments, every fragment heading receives a row-local ``[part n]``
    suffix; unsplit rows keep their original heading.

    Args:
        tokenizer: LightRAG tokenizer (used for all token counting; matches
            the unit used by ``chunk_token_size``).
        content: Merged plain-text content of the document. Used as the
            fallback corpus when ``blocks_path`` is missing or unreadable
            so the pipeline never silently drops a document.
        chunk_token_size: Hard upper bound for each chunk in tokens. The
            ideal target is set at 75 % of this value (mirroring the
            audit-mode 6000/8000 ratio); see threshold ratio constants
            above for the full mapping.
        blocks_path: Path to the document's ``.blocks.jsonl`` sidecar
            (typically ``parsed_data["blocks_path"]``). When ``None``,
            unreadable, or empty, this function falls back to
            :func:`chunking_by_recursive_character` on ``content``
            (per ``docs/FileProcessingConfiguration-zh.md`` §3).
            That fallback hard-requires ``langchain-text-splitters``;
            an :class:`ImportError` is surfaced rather than silently
            degrading further.
        chunk_overlap_token_size: Token overlap used only when P must
            fall back to recursive-character splitting of ordinary text,
            and as the per-side budget for duplicating text between two
            adjacent oversized table chunks. Structural table row splits
            remain row-bounded and non-overlapping.

    Returns:
        Ordered list of chunk dicts, each shaped::

            {
                "tokens": int,
                "content": str,
                "chunk_order_index": int,
                "heading": {"level": int, "heading": str,
                            "parent_headings": list[str]},
                "sidecar": {"type": "block", "id": str, "refs": [...]},  # optional
            }

        ``heading`` is a nested block (``level`` / ``parent_headings`` live
        inside it, not at the top level; the ``[part n]`` suffix lands on
        ``heading["heading"]``). ``sidecar`` is present only when the source
        row carried a ``blockid``.

    Notes:
        blocks.jsonl field analysis vs. algorithm requirements:

          - ``content`` (``\\n``-joined, tags single-line) → split back into
            per-paragraph text via ``split("\\n")``; lossless because
            table/equation/drawing tags are single-line replacements.
          - ``heading`` / ``parent_headings`` / ``level`` → consumed by
            AnchorSplit/LevelMerge for hierarchy-aware merging. A row split into
            multiple fragments gets a ``[part n]`` suffix on ``heading`` after
            TableRowSplit/AnchorSplit and before LevelMerge; ``parent_headings``
            are untouched.
          - ``<table … format="json">{rows_json}</table>`` → JSON body
            row-level re-split in TableRowSplit when over the per-table cap;
            short text between two split tables may be repeated into both
            boundary chunks, with any longer middle remainder left standalone.
          - ``<equation>`` / ``<drawing>`` → atomic non-table paragraphs,
            neither splittable nor anchorable.
          - Per-paragraph paraIds are NOT preserved (only block-level
            ``positions[].range``); fine, the output schema does not need them.
          - ``table_slice`` is always ``"none"`` in blocks.jsonl (tables kept
            whole at parse time), so the ``table_chunk_role`` LevelMerge
            consumes is recomputed on the fly inside TableRowSplit.
    """
    target_max = max(int(chunk_token_size), 1)
    target_ideal = max(int(target_max * _IDEAL_RATIO), 1)
    table_max = max(int(target_max * _TABLE_MAX_RATIO), 1)
    table_ideal = max(int(target_max * _TABLE_IDEAL_RATIO), 1)
    table_min_last = max(int(table_max * _TABLE_MIN_LAST_RATIO), 1)
    small_tail_threshold = max(int(target_max * _SMALL_TAIL_RATIO), 1)
    overlap = _bounded_overlap(target_max, chunk_overlap_token_size)

    rows: list[dict[str, Any]] = []
    fallback_reason: str | None = None
    if not blocks_path:
        fallback_reason = "blocks_path is empty"
    else:
        try:
            rows = _load_blocks_from_jsonl(blocks_path)
        except OSError as exc:
            fallback_reason = f"cannot read blocks.jsonl at {blocks_path}: {exc}"
        else:
            if not rows:
                fallback_reason = (
                    f"blocks.jsonl at {blocks_path} contains no content rows"
                )

    if fallback_reason is not None:
        # Defer to recursive-character chunking when the sidecar is
        # absent — ensures non-structured documents and edge-case parses
        # still produce chunks instead of silently dropping content.  The
        # document contract (FileProcessingConfiguration-zh.md §3) is
        # explicit that P falls back to R; that contract requires
        # langchain-text-splitters to be installed, so an ImportError
        # here is intentional rather than a silent degrade to F.  Lazy
        # import dodges the recursive_character ↔ paragraph_semantic
        # circular dependency.
        logger.warning(
            "[paragraph_semantic_chunking] %s; falling back to "
            "recursive-character chunking with chunk_token_size=%d.",
            fallback_reason,
            target_max,
        )
        from lightrag.chunker.recursive_character import (
            chunking_by_recursive_character,
        )

        return chunking_by_recursive_character(
            tokenizer,
            content,
            target_max,
            chunk_overlap_token_size=overlap,
        )

    # Build initial blocks (HeadingBlocks output, already persisted).
    initial: list[dict[str, Any]] = []
    for row in rows:
        text = row.get("content", "") or ""
        if not text.strip():
            continue
        paragraphs = _block_to_paragraphs(text)
        if not paragraphs:
            continue
        row_blockid = str(row.get("blockid") or "").strip()
        initial.append(
            _new_block(
                heading=row.get("heading", "") or "",
                parent_headings=list(row.get("parent_headings") or []),
                level=int(row.get("level", 1) or 1),
                paragraphs=paragraphs,
                table_chunk_role="none",
                tokenizer=tokenizer,
                blockids=[row_blockid] if row_blockid else None,
            )
        )

    # Repeating-header lookup for table slices that lose their header during
    # row-splitting. Loaded once here (never on the missing-sidecar fallback
    # path, which returned above); an absent / unreadable tables.json degrades
    # to an empty map, so HeaderRecovery silently no-ops. Fed into TableRowSplit
    # so the header's tokens are budgeted *before* splitting (the slice + header
    # never exceeds target_max), rather than backfilled after the cap is set.
    table_headers = _load_table_headers(blocks_path) if blocks_path else {}

    # TableRowSplit/AnchorSplit are run per original blocks.jsonl content row so split
    # fragments can be labelled with [part n] using a row-local counter
    # before LevelMerge merges small neighbours.
    after_c: list[dict[str, Any]] = []
    for blk in initial:
        block_after_b = _expand_block_with_table_splits(
            blk,
            tokenizer=tokenizer,
            table_max=table_max,
            table_ideal=table_ideal,
            table_min_last=table_min_last,
            target_max=target_max,
            chunk_overlap_token_size=overlap,
            table_headers=table_headers,
        )

        block_after_c: list[dict[str, Any]] = []
        for split_blk in block_after_b:
            block_after_c.extend(
                _split_long_block(
                    split_blk["paragraphs"],
                    split_blk["heading"],
                    split_blk["parent_headings"],
                    split_blk["level"],
                    split_blk.get("table_chunk_role", "none"),
                    tokenizer=tokenizer,
                    target_max=target_max,
                    target_ideal=target_ideal,
                    chunk_overlap_token_size=overlap,
                    blockids=split_blk.get("blockids") or blk.get("blockids"),
                )
            )
        after_c.extend(_apply_part_suffixes(block_after_c))

    # HeadingGlue — glue each body-less heading block FORWARD into its
    # strictly-deeper child (role "none" or the "first" slice of a split table),
    # so the bare heading never reaches _merge_small_blocks detached from its
    # child content nor glued onto an unrelated same-level sibling. Gluing into a
    # "first" slice keeps the "first" role so LevelMerge still can't pull it back.
    # A body-less heading whose next block is not deeper is left for LevelMerge
    # (not pulled into a deeper previous block — that would invert the
    # hierarchy). A forward-glued block tipped past target_max is re-split via
    # AnchorSplit so the hard cap holds. Runs across original rows after [part n]
    # tagging is finalised (heading-only rows are never split, so no part suffix).
    after_c = _glue_heading_only_blocks(
        after_c,
        tokenizer=tokenizer,
        target_max=target_max,
        target_ideal=target_ideal,
        chunk_overlap_token_size=overlap,
    )

    # LevelMerge — bottom-up, level-aware small-block merging.
    final = _merge_small_blocks(
        after_c,
        tokenizer=tokenizer,
        target_max=target_max,
        target_ideal=target_ideal,
        small_tail_threshold=small_tail_threshold,
    )

    # Convert internal block dicts to the new chunk schema: nested heading
    # dict + sidecar block carrying source blockid refs so the multimodal
    # pipeline (and document-delete cache cleanup) can trace each chunk
    # back to its blocks.jsonl row(s).
    chunks: list[dict[str, Any]] = []
    for idx, blk in enumerate(final):
        body = blk["content"].strip()
        if not body:
            continue
        chunk_dict: dict[str, Any] = {
            "tokens": blk["tokens"],
            "content": body,
            "chunk_order_index": idx,
            "heading": {
                "level": int(blk.get("level") or 0),
                "heading": str(blk.get("heading") or ""),
                "parent_headings": list(blk.get("parent_headings") or []),
            },
        }
        blockids = blk.get("blockids") or []
        if blockids:
            chunk_dict["sidecar"] = {
                "type": "block",
                "id": blockids[0],
                "refs": [{"type": "block", "id": bid} for bid in blockids],
            }
        chunks.append(chunk_dict)
    return chunks
