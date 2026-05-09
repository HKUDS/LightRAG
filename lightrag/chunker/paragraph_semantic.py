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
    here when an embedded ``<table … format="json">`` exceeds
    ``TABLE_MAX_TOKENS``.
  - Stage C — anchor-driven long-block re-split: short non-table
    paragraphs (≤ 100 chars) are promoted as split points and the block
    is rebalanced toward ``IDEAL_BLOCK_TOKENS``.
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
    r'<table\s+(?P<attrs>[^>]*)>(?P<body>.*?)</table>',
    re.DOTALL,
)


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
    target_rows = len(rows) / target_chunks

    chunks: list[list[Any]] = []
    start = 0
    for i in range(target_chunks):
        if i == target_chunks - 1:
            end = len(rows)
        else:
            end = min(int((i + 1) * target_rows), len(rows))
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
    table_split_counter = 0

    def flush_cur(role: str = "none") -> None:
        if not cur_paras:
            return
        out.append(
            _new_block(
                heading=block["heading"],
                parent_headings=block["parent_headings"],
                level=block["level"],
                paragraphs=cur_paras,
                table_chunk_role=role,
                tokenizer=tokenizer,
            )
        )
        cur_paras.clear()

    for para in paragraphs:
        text = para["text"]
        if not (para["is_table"] and _count_tokens(tokenizer, text) > table_max):
            cur_paras.append(para)
            continue

        parsed = _parse_table_tag(text)
        if parsed is None:
            cur_paras.append(para)
            continue
        attrs, rows = parsed
        row_chunks = _split_rows_by_tokens(
            rows,
            tokenizer,
            target_max=table_max,
            target_ideal=table_ideal,
            last_min=table_min_last,
        )
        if len(row_chunks) <= 1:
            cur_paras.append(para)
            continue

        for chunk_idx, chunk_rows in enumerate(row_chunks):
            chunk_text = (
                f"<table {attrs}>"
                f"{json.dumps(chunk_rows, ensure_ascii=False)}"
                f"</table>"
            )
            chunk_para = {"text": chunk_text, "is_table": True}
            is_first = chunk_idx == 0
            is_last = chunk_idx == len(row_chunks) - 1

            if is_first:
                # First slice glues with everything currently accumulated
                # (= the paragraphs that appeared before the table inside
                # this heading block).
                cur_paras.append(chunk_para)
            elif is_last:
                # Flush the accumulated "first-glued" block, then begin a
                # new accumulation seeded with this last slice — it will
                # absorb the paragraphs that appear after the table.
                flush_cur()
                cur_paras.append(chunk_para)
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

    # Build anchor candidates with cumulative token offsets.
    candidates: list[dict[str, Any]] = []
    cumulative = 0
    for idx, para in enumerate(paragraphs):
        text = para["text"]
        if (
            not para.get("is_table", False)
            and 0 < len(text) <= _MAX_ANCHOR_CANDIDATE_LENGTH
        ):
            candidates.append({"index": idx, "text": text, "position": cumulative})
        cumulative += _count_tokens(tokenizer, text)

    if not candidates:
        logger.warning(
            "[paragraph_semantic_chunking] block under heading %r exceeds "
            "target_max=%d tokens (~%d tokens) but has no eligible anchor "
            "paragraph (≤ %d chars). Emitting as single oversized chunk; "
            "the embedding-time hard fallback will tokenize-split as needed.",
            heading,
            target_max,
            total,
            _MAX_ANCHOR_CANDIDATE_LENGTH,
        )
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
                        if (
                            nxt.get("level", 1) == current_level
                            and _can_merge_backward(
                                nxt.get("table_chunk_role", "none")
                            )
                        ):
                            combined = _merged_pair(
                                cur, nxt, keep="left", tokenizer=tokenizer
                            )
                            if combined["tokens"] <= target_max:
                                new_result.append(combined)
                                i += 2
                                changed = True
                                merged = True

                    if (
                        not merged
                        and _can_merge_backward(cur_role)
                        and new_result
                    ):
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
                        if (
                            nxt.get("level", 1) > current_level
                            and _can_merge_backward(
                                nxt.get("table_chunk_role", "none")
                            )
                        ):
                            combined = _merged_pair(
                                cur, nxt, keep="left", tokenizer=tokenizer
                            )
                            if combined["tokens"] <= target_max:
                                new_result.append(combined)
                                i += 2
                                changed = True
                                merged = True

                    if (
                        not merged
                        and _can_merge_backward(cur_role)
                        and new_result
                    ):
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
    chunk_token_size: int = 1200,
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
            (typically ``parsed_data["blocks_path"]``). When ``None`` or
            unreadable, this function falls back to
            :func:`chunking_by_token_size` on ``content``.

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
        # Defer to the fixed-token strategy when blocks.jsonl is absent —
        # ensures non-docx documents and edge-case parses still produce
        # chunks instead of silently dropping content. The fixed-token
        # strategy's delimiter / overlap knobs are deliberately not
        # plumbed through here: paragraph-semantic chunking is opted into
        # explicitly per-document, and the fallback only fires when the
        # required sidecar is unavailable, so default token-window
        # behaviour is the right thing to do.
        logger.warning(
            "[paragraph_semantic_chunking] %s; falling back to fixed-token "
            "chunking with chunk_token_size=%d.",
            fallback_reason,
            target_max,
        )
        from lightrag.chunker.token_size import chunking_by_token_size

        return chunking_by_token_size(
            tokenizer,
            content,
            chunk_token_size=target_max,
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
