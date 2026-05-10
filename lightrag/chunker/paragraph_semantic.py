"""Paragraph Semantic Chunking for LightRAG.

Reads a LightRAG ``.blocks.jsonl`` artifact (produced by the docx native
parser at ``fixlevel=0`` — heading-driven splits only, tables kept whole)
and produces a chunk list compatible with
:func:`lightrag.chunker.chunking_by_token_size`.

The full algorithm and rationale are documented in
``docs/ParagraphSemanticChunking-zh.md``. This module re-implements the
post-Stage-A pipeline (B/C/D) on top of blocks.jsonl input, parameterised
on ``chunk_token_size`` so chunk size targets follow the user's RAG
configuration rather than the audit-mode constants in
``lightrag/native_parser/docx/parse_document.py``.

Pipeline:
  - Stage A — heading-driven initial split: already done at parse time and
    persisted as one row per block in ``blocks.jsonl``.
  - Stage B — oversized-table re-split + first/middle/last gluing: invoked
    here when an embedded ``<table … format="json">`` (or
    ``format="html"``) exceeds ``TABLE_MAX_TOKENS``. Splitting prefers
    structural row boundaries (JSON list items, HTML ``<tr>`` rows) so
    each fragment remains a legal ``<table>`` tag; only when no row
    boundary is available, or a single row alone exceeds the cap, does
    the splitter fall back to ``chunking_by_recursive_character`` on
    that specific fragment.
  - Stage C — anchor-driven long-block re-split: short non-table
    paragraphs (≤ 100 chars) are promoted as split points and the block
    is rebalanced toward ``IDEAL_BLOCK_TOKENS``. When no anchor exists,
    table-aware fallback applies the same row-boundary-first strategy
    to any oversized table paragraph and only character-splits the
    residual non-table content.
  - Stage D — bottom-up, level-aware small-block merging: undersized
    blocks get absorbed by same-level neighbours (Phase A), shallower
    levels (Phase B), and a final tail-absorption pass eliminates the
    last few zero-content remainders.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from lightrag.utils import Tokenizer, logger


# ---------------------------------------------------------------------------
# Threshold ratios — derived from the audit-mode constants in
# lightrag/native_parser/docx/parse_document.py so the trade-off curves
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

# Heading suffix used when promoting middle table chunks to standalone blocks.
_TABLE_CHUNK_SUFFIX_LABEL = "表格片段"

# Strict regex for a post-rewrite table tag emitted by ``lightrag_adapter``:
#   <table id="tb-…" format="json"[ caption="…"]>{rows_json}</table>
# blocks.jsonl invariants guarantee the tag has no embedded newlines.
_TABLE_TAG_RE = re.compile(
    r"<table\s+(?P<attrs>[^>]*)>(?P<body>.*?)</table>",
    re.DOTALL,
)

# Format detection regex inside the attrs string, e.g. format="json".
_TABLE_FORMAT_RE = re.compile(r"""format\s*=\s*["'](?P<fmt>[^"']+)["']""")

# HTML <tr>...</tr> row extractor. Standard HTML disallows nested <tr>,
# so a non-greedy match is sufficient for well-formed input.
_HTML_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _count_tokens(tokenizer: Tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def _is_table_paragraph(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("<table ") and stripped.endswith("</table>")


def _block_to_paragraphs(content: str) -> list[dict[str, Any]]:
    """Recover the per-paragraph view of a rewritten block.

    The docx parser joins paragraphs with ``\\n`` inside
    ``_build_unsplit_block``; tables/equations/drawings are inserted as
    single-line tags with no internal newlines, so ``split("\\n")`` faithfully
    recovers paragraph boundaries.
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


def _detect_table_format(attrs: str, body: str) -> str | None:
    """Return ``"json"``, ``"html"`` or ``None`` for a parsed ``<table>`` tag.

    Prefers an explicit ``format="…"`` attribute. When silent, sniffs the
    body: a leading ``[`` / ``{`` (after whitespace) implies JSON; the
    presence of any ``<tr`` tag implies HTML. Anything else is unknown
    and falls back to character splitting at the call site.
    """
    match = _TABLE_FORMAT_RE.search(attrs or "")
    if match:
        fmt = match.group("fmt").strip().lower()
        if fmt in {"json", "html"}:
            return fmt
        return None
    stripped = (body or "").lstrip()
    if stripped.startswith(("[", "{")):
        return "json"
    if "<tr" in stripped.lower():
        return "html"
    return None


def _split_html_rows(body: str) -> list[str] | None:
    """Extract ``<tr>...</tr>`` rows from an HTML table body.

    Returns the list of row strings (each preserved with its ``<tr>``
    wrapper) or ``None`` if no row was found — the latter is the signal
    for the caller to fall through to character splitting.

    Text between rows (``<thead>`` / ``<tbody>`` wrappers, whitespace,
    captions outside ``<tr>``) is dropped: each output string is a
    self-contained row. This is a pragmatic simplification — full DOM
    parsing would require ``lxml`` / ``beautifulsoup4``, an unjustified
    dependency for a fallback path.
    """
    rows = _HTML_TR_RE.findall(body or "")
    if not rows:
        return None
    return rows


def _split_html_rows_by_tokens(
    rows: list[str],
    tokenizer: Tokenizer,
    *,
    target_max: int,
    target_ideal: int,
    last_min: int,
) -> list[list[str]]:
    """HTML-string analog of :func:`_split_rows_by_tokens`.

    Same balanced-split + tail-merge algorithm; rows are concatenated
    with ``""`` (rather than serialised as JSON) for token measurement.
    """
    total = _count_tokens(tokenizer, "".join(rows))
    if total <= target_max or len(rows) <= 1:
        return [rows]

    target_chunks = max(
        math.ceil(total / target_ideal),
        math.ceil(total / target_max),
    )
    target_chunks = min(target_chunks, len(rows))
    target_rows = len(rows) / target_chunks

    chunks: list[list[str]] = []
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
        last_text = "".join(chunks[-1])
        if _count_tokens(tokenizer, last_text) < last_min:
            merged = chunks[-2] + chunks[-1]
            merged_tokens = _count_tokens(tokenizer, "".join(merged))
            if merged_tokens <= target_max:
                chunks[-2] = merged
                chunks.pop()
    return chunks


def _new_block(
    *,
    heading: str,
    parent_headings: list[str],
    level: int,
    paragraphs: list[dict[str, Any]],
    table_chunk_role: str,
    tokenizer: Tokenizer,
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
    }


# ---------------------------------------------------------------------------
# Stage B — oversized-table re-split with first/middle/last gluing.
# ---------------------------------------------------------------------------


def _parse_table_tag(text: str) -> tuple[str, list[Any]] | None:
    """Return ``(attrs_str, rows)`` for a ``<table …>{rows_json}</table>``."""
    match = _TABLE_TAG_RE.match(text.strip())
    if not match:
        return None
    body = match.group("body")
    try:
        rows = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(rows, list):
        return None
    return match.group("attrs"), rows


def _split_rows_by_tokens(
    rows: list[Any],
    tokenizer: Tokenizer,
    *,
    target_max: int,
    target_ideal: int,
    last_min: int,
) -> list[list[Any]]:
    """Split ``rows`` into balanced row-bounded chunks (Stage B core)."""
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
) -> list[str]:
    """Character-level fallback wrapped to return plain-text pieces.

    Lazy import dodges the ``recursive_character`` ↔ ``paragraph_semantic``
    circular dependency (same pattern as the sidecar-missing fallback in
    :func:`chunking_by_paragraph_semantic`).
    """
    from lightrag.chunker.recursive_character import (
        chunking_by_recursive_character,
    )

    pieces = chunking_by_recursive_character(
        tokenizer,
        text,
        target_max,
        chunk_overlap_token_size=0,
    )
    return [p["content"] for p in pieces if p.get("content")]


def _split_table_text(
    table_text: str,
    *,
    tokenizer: Tokenizer,
    target_max: int,
    target_ideal: int,
    last_min: int,
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

    row_chunks_serialized: list[str] | None = None
    if fmt == "json":
        try:
            rows = json.loads(body)
        except json.JSONDecodeError:
            rows = None
        if isinstance(rows, list) and len(rows) > 1:
            row_chunks = _split_rows_by_tokens(
                rows,
                tokenizer,
                target_max=target_max,
                target_ideal=target_ideal,
                last_min=last_min,
            )
            row_chunks_serialized = [
                f"<table {attrs}>"
                f"{json.dumps(chunk_rows, ensure_ascii=False)}"
                f"</table>"
                for chunk_rows in row_chunks
            ]
    elif fmt == "html":
        rows_html = _split_html_rows(body)
        if rows_html and len(rows_html) > 1:
            row_chunks = _split_html_rows_by_tokens(
                rows_html,
                tokenizer,
                target_max=target_max,
                target_ideal=target_ideal,
                last_min=last_min,
            )
            row_chunks_serialized = [
                f"<table {attrs}>{''.join(chunk_rows)}</table>"
                for chunk_rows in row_chunks
            ]

    if not row_chunks_serialized:
        # No row boundary available (single-row table, parse failure,
        # unknown format) → character-fallback the whole text.
        return _character_split_text(table_text, tokenizer, target_max=target_max)

    pieces: list[str] = []
    for chunk_text in row_chunks_serialized:
        if _count_tokens(tokenizer, chunk_text) > target_max:
            # A single row exceeds the cap on its own; preserve as much
            # structure as possible by character-splitting only this chunk.
            pieces.extend(
                _character_split_text(chunk_text, tokenizer, target_max=target_max)
            )
        else:
            pieces.append(chunk_text)
    return pieces


def _expand_block_with_table_splits(
    block: dict[str, Any],
    *,
    tokenizer: Tokenizer,
    table_max: int,
    table_ideal: int,
    table_min_last: int,
) -> list[dict[str, Any]]:
    """Apply Stage B to one heading-driven block.

    For every embedded table whose tokens exceed ``table_max``:
      - the first row-slice glues with paragraphs already accumulated in
        the current expansion (i.e. content *before* the table);
      - middle slices are emitted as standalone blocks tagged
        ``table_chunk_role == "middle"`` with a ``[表格片段N]`` heading
        suffix so Stage D refuses to merge them;
      - the last slice begins a fresh accumulation that will glue with
        paragraphs *after* the table.

    Tables within the size limit pass through untouched.
    """
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
    # otherwise Stage D's directional protections (a "first" block must
    # not absorb backward, a "last" block must not absorb forward) silently
    # disappear after the slice glues with surrounding paragraphs.
    cur_role = "none"
    table_split_counter = 0

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
            )
        )
        cur_paras.clear()
        cur_role = "none"

    for para in paragraphs:
        text = para["text"]
        if not (para["is_table"] and _count_tokens(tokenizer, text) > table_max):
            cur_paras.append(para)
            continue

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
                    flush_cur()
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
                # this middle slice as a standalone block that Stage D
                # MUST keep intact (table_chunk_role == "middle").
                flush_cur()
                table_split_counter += 1
                middle_heading = (
                    f"{block['heading']} "
                    f"[{_TABLE_CHUNK_SUFFIX_LABEL}{table_split_counter}]"
                )
                out.append(
                    _new_block(
                        heading=middle_heading,
                        parent_headings=block["parent_headings"],
                        level=block["level"],
                        paragraphs=[chunk_para],
                        table_chunk_role="middle",
                        tokenizer=tokenizer,
                    )
                )

    flush_cur()
    return out


# ---------------------------------------------------------------------------
# Stage C — anchor-driven long-block re-split.
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
) -> list[dict[str, Any]]:
    """Split an oversized block into balanced sub-blocks at short-paragraph anchors.

    Mirrors :func:`lightrag.native_parser.docx.parse_document.split_long_block`,
    parameterised on ``target_max`` / ``target_ideal``. Tables (``is_table``)
    are excluded from the anchor candidate pool, so Stage B's row-level
    splits stay intact. When no anchor exists, returns the block as a
    single oversized chunk and lets the embedding-time hard fallback split
    handle the cap (the audit-mode parser would `sys.exit(1)` here, but
    the RAG pipeline must never drop a document silently).
    """
    content = "\n".join(p["text"] for p in paragraphs)
    total = _count_tokens(tokenizer, content)
    if total <= target_max or len(paragraphs) <= 1:
        return [
            _new_block(
                heading=heading,
                parent_headings=parent_headings,
                level=level,
                paragraphs=paragraphs,
                table_chunk_role=table_chunk_role,
                tokenizer=tokenizer,
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
        # ``last_min`` mirrors Stage B's ratio (no separate constant — the
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
        # current buffer.
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
                    _character_split_text(piece, tokenizer, target_max=target_max)
                )
                continue
            if buf_tokens + piece_tokens > target_max and buf:
                chunks_text.append("\n".join(buf))
                buf, buf_tokens = [], 0
            buf.append(piece)
            buf_tokens += piece_tokens
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
            )
        )

    # Recursive guard: if any sub-block still exceeds target_max with more
    # than one paragraph available, try to split it further.
    out: list[dict[str, Any]] = []
    for sub in sub_blocks:
        if sub["tokens"] > target_max and len(sub["paragraphs"]) > 1:
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
                )
            )
        else:
            out.append(sub)
    return out


# ---------------------------------------------------------------------------
# Stage D — bottom-up, level-aware small-block merging.
# ---------------------------------------------------------------------------


def _can_merge_forward(role: str, *, phase: str) -> bool:
    if phase == "A":
        return role in {"none", "first"}
    return role in {"none", "first", "last"}


def _can_merge_backward(role: str) -> bool:
    return role in {"none", "last"}


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
    return {
        "heading": base["heading"],
        "parent_headings": list(base["parent_headings"]),
        "level": base["level"],
        "paragraphs": paragraphs,
        "content": content,
        "tokens": _count_tokens(tokenizer, content),
        "table_chunk_role": "none",
    }


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
    :func:`lightrag.native_parser.docx.parse_document.merge_small_blocks`,
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

                    if _can_merge_forward(cur_role, phase="A") and i + 1 < len(result):
                        nxt = result[i + 1]
                        if nxt.get("level", 1) == current_level and _can_merge_backward(
                            nxt.get("table_chunk_role", "none")
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
                            and _can_merge_forward(
                                prev.get("table_chunk_role", "none"), phase="A"
                            )
                            and prev["tokens"] < target_ideal
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
                    if is_cur_lv and cur_tokens >= target_ideal:
                        tail_total = 0
                        end_idx = i + 1
                        for j in range(i + 1, len(result)):
                            nxt = result[j]
                            if nxt.get("level", 1) != current_level:
                                break
                            if nxt.get("table_chunk_role", "none") == "middle":
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
                            new_result.append(
                                {
                                    "heading": cur["heading"],
                                    "parent_headings": list(cur["parent_headings"]),
                                    "level": cur["level"],
                                    "paragraphs": absorbed_paragraphs,
                                    "content": absorbed_content,
                                    "tokens": _count_tokens(
                                        tokenizer, absorbed_content
                                    ),
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

                    if _can_merge_forward(cur_role, phase="B") and i + 1 < len(result):
                        nxt = result[i + 1]
                        if nxt.get("level", 1) > current_level and _can_merge_backward(
                            nxt.get("table_chunk_role", "none")
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
                            and _can_merge_forward(
                                prev.get("table_chunk_role", "none"), phase="B"
                            )
                            and prev["tokens"] < target_ideal
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
) -> list[dict[str, Any]]:
    """Paragraph Semantic Chunking — the ``chunking="P"`` strategy.

    Reads structured blocks emitted by the docx native parser at
    ``fixlevel=0`` (Stage A, persisted to ``blocks.jsonl``) and applies
    Stage B (table re-split + glue), Stage C (anchor-driven long-block
    re-split) and Stage D (bottom-up, level-aware merging). Output rows
    match the schema produced by
    :func:`lightrag.chunker.chunking_by_token_size`
    (``tokens``/``content``/``chunk_order_index``), enriched with
    ``heading``, ``parent_headings`` and ``level`` so KG extraction can
    leverage the document hierarchy.

    Signature follows the LightRAG chunker contract — the standard
    prefix ``(tokenizer, content, chunk_token_size)`` is shared with
    every other chunker, while strategy-specific knobs are keyword-only:

      - ``blocks_path`` (this strategy's required input — the
        ``.blocks.jsonl`` sidecar produced at parse time)

    Knobs that ``chunking_by_token_size`` exposes (``split_by_character``,
    ``split_by_character_only``, ``chunk_overlap_token_size``) are
    deliberately absent here because paragraph-semantic chunks are
    heading-aligned and non-overlapping by construction; surfacing those
    knobs would invite misuse.

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
            (per ``docs/FileProcessingConfiguration-zh.md`` line 120 / 146).
            That fallback hard-requires ``langchain-text-splitters``;
            an :class:`ImportError` is surfaced rather than silently
            degrading further.

    Returns:
        Ordered list of chunk dicts, each shaped:
        ``{"tokens", "content", "chunk_order_index", "heading",
        "parent_headings", "level"}``.

    Notes:
        blocks.jsonl field analysis vs. algorithm requirements:

          - ``content`` (``\\n``-joined per ``_build_unsplit_block``) →
            split back into per-paragraph text via ``split("\\n")``;
            lossless because table/equation/drawing tags are emitted as
            single-line replacements.
          - ``heading`` / ``parent_headings`` / ``level`` → consumed
            directly by Stage C/D for hierarchy-aware merging.
          - ``<table id="…" format="json">{rows_json}</table>`` tags →
            JSON body parsed in Stage B for row-level re-split when the
            tag exceeds the per-table token cap.
          - ``<equation>`` / ``<drawing>`` tags → treated as atomic
            non-table paragraphs — neither splittable nor anchorable.
          - Per-paragraph paraIds are NOT preserved in blocks.jsonl
            (only block-level ``positions[].range`` is). Acceptable
            because the chunking output schema does not require them.
          - ``table_slice`` is always ``"none"`` in blocks.jsonl
            (parse-time ``fixlevel=0`` keeps tables whole), so any
            ``table_chunk_role`` consumed by Stage D is recomputed
            on-the-fly inside Stage B.
    """
    target_max = max(int(chunk_token_size), 1)
    target_ideal = max(int(target_max * _IDEAL_RATIO), 1)
    table_max = max(int(target_max * _TABLE_MAX_RATIO), 1)
    table_ideal = max(int(target_max * _TABLE_IDEAL_RATIO), 1)
    table_min_last = max(int(table_max * _TABLE_MIN_LAST_RATIO), 1)
    small_tail_threshold = max(int(target_max * _SMALL_TAIL_RATIO), 1)

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
        # absent — ensures non-docx documents and edge-case parses still
        # produce chunks instead of silently dropping content.  Document
        # contract (FileProcessingConfiguration-zh.md L120 / L146) is
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
        )

    # Build initial blocks (Stage A output, already persisted).
    initial: list[dict[str, Any]] = []
    for row in rows:
        text = row.get("content", "") or ""
        if not text.strip():
            continue
        paragraphs = _block_to_paragraphs(text)
        if not paragraphs:
            continue
        initial.append(
            _new_block(
                heading=row.get("heading", "") or "",
                parent_headings=list(row.get("parent_headings") or []),
                level=int(row.get("level", 1) or 1),
                paragraphs=paragraphs,
                table_chunk_role="none",
                tokenizer=tokenizer,
            )
        )

    # Stage B — oversized-table re-split + first/middle/last gluing.
    after_b: list[dict[str, Any]] = []
    for blk in initial:
        after_b.extend(
            _expand_block_with_table_splits(
                blk,
                tokenizer=tokenizer,
                table_max=table_max,
                table_ideal=table_ideal,
                table_min_last=table_min_last,
            )
        )

    # Stage C — anchor-driven long-block re-split.
    after_c: list[dict[str, Any]] = []
    for blk in after_b:
        after_c.extend(
            _split_long_block(
                blk["paragraphs"],
                blk["heading"],
                blk["parent_headings"],
                blk["level"],
                blk.get("table_chunk_role", "none"),
                tokenizer=tokenizer,
                target_max=target_max,
                target_ideal=target_ideal,
            )
        )

    # Stage D — bottom-up, level-aware small-block merging.
    final = _merge_small_blocks(
        after_c,
        tokenizer=tokenizer,
        target_max=target_max,
        target_ideal=target_ideal,
        small_tail_threshold=small_tail_threshold,
    )

    # Convert internal block dicts to the chunking_by_token_size schema,
    # enriched with heading metadata so KG extraction has access to the
    # document hierarchy.
    chunks: list[dict[str, Any]] = []
    for idx, blk in enumerate(final):
        body = blk["content"].strip()
        if not body:
            continue
        chunks.append(
            {
                "tokens": blk["tokens"],
                "content": body,
                "chunk_order_index": idx,
                "heading": blk["heading"],
                "parent_headings": list(blk["parent_headings"]),
                "level": blk["level"],
            }
        )
    return chunks
