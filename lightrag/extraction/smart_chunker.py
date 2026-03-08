"""Four-stage smart chunking pipeline – docx-extraction-guide-zh.md §3.

Stage A: Heading-driven initial chunking
Stage B: Large table row slicing
Stage C: Long block anchor splitting
Stage D: Small block merging (with tail absorption)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    EXTRACTION_SAFE_MAX_BLOCK_TOKENS,
    IDEAL_BLOCK_CONTENT_TOKENS,
    MAX_ANCHOR_CANDIDATE_LENGTH,
    MAX_BLOCK_CONTENT_TOKENS,
    SMALL_TAIL_THRESHOLD,
    TABLE_IDEAL_TOKENS,
    TABLE_MAX_TOKENS,
    TABLE_MIN_LAST_CHUNK_TOKENS,
)
from .docx_extractor import Paragraph
from .token_estimation import estimate_tokens


# ── Block data structure ─────────────────────────────────────────────
@dataclass
class Block:
    """A chunk of document content, ready for output."""

    heading: str = ""
    parent_headings: list[str] = field(default_factory=list)
    level: int = 0
    content_parts: list[str] = field(default_factory=list)
    uuid: str = ""
    uuid_end: str = ""
    table_chunk_role: str = "none"
    table_header: Optional[list[list[str]]] = None

    @property
    def content(self) -> str:
        return "\n\n".join(p for p in self.content_parts if p)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)


# ── Stage A: Heading-driven initial chunking ─────────────────────────
def _stage_a_heading_split(paragraphs: list[Paragraph]) -> list[Block]:
    """Split by heading paragraphs. Heading text also enters content."""
    blocks: list[Block] = []
    current_block: Block | None = None
    heading_stack: list[tuple[int, str]] = []  # (level, title)

    for para in paragraphs:
        is_heading = para.outline_level <= 8

        if is_heading:
            # Flush current block
            if current_block is not None and current_block.content_parts:
                blocks.append(current_block)

            title = para.text.strip()
            level = para.outline_level + 1  # 0-based → 1-based

            # Maintain heading stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent_headings = [h[1] for h in heading_stack]
            heading_stack.append((level, title))

            current_block = Block(
                heading=title,
                parent_headings=list(parent_headings),
                level=level,
                content_parts=[title],  # heading text enters content
                uuid=para.para_id,
                uuid_end=para.para_id,
            )
        else:
            # Body / table paragraph
            if current_block is None:
                current_block = Block(
                    heading="Preface/Uncategorized",
                    parent_headings=[],
                    level=0,
                    content_parts=[],
                    uuid=para.para_id,
                    uuid_end=para.para_id,
                )

            content = para.text
            current_block.content_parts.append(content)

            # Update uuid_end
            if para.para_id:
                current_block.uuid_end = para.para_id
            # For tables, the end id may be stored in drawing_id
            if para.is_table and para.drawing_id:
                current_block.uuid_end = para.drawing_id

            # Attach table metadata
            if para.is_table and para.table_header_rows:
                current_block.table_header = para.table_header_rows

    # Flush last block
    if current_block is not None and current_block.content_parts:
        blocks.append(current_block)

    return blocks


# ── Stage B: Large table row slicing ─────────────────────────────────
def _slice_table_rows(
    table_json: list[list[str]],
    header_rows: list[list[str]] | None,
) -> list[tuple[list[list[str]], str]]:
    """Slice table by rows. Returns list of (rows, role)."""
    total_tokens = estimate_tokens(json.dumps(table_json, ensure_ascii=False))
    if total_tokens <= TABLE_MAX_TOKENS:
        return [(table_json, "none")]

    slices: list[tuple[list[list[str]], str]] = []
    current_rows: list[list[str]] = []
    current_tokens = 0
    header_tokens = 0
    if header_rows:
        header_tokens = estimate_tokens(json.dumps(header_rows, ensure_ascii=False))

    for row in table_json:
        row_tokens = estimate_tokens(json.dumps(row, ensure_ascii=False))
        if current_tokens + row_tokens > TABLE_IDEAL_TOKENS and current_rows:
            slices.append((current_rows, ""))
            current_rows = []
            current_tokens = header_tokens  # account for header prepend
        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        slices.append((current_rows, ""))

    # Merge tiny last slice
    if len(slices) > 1:
        last_tokens = estimate_tokens(json.dumps(slices[-1][0], ensure_ascii=False))
        if last_tokens < TABLE_MIN_LAST_CHUNK_TOKENS:
            merged = slices[-2][0] + slices[-1][0]
            slices[-2] = (merged, slices[-2][1])
            slices.pop()

    # Assign roles
    if len(slices) == 1:
        slices[0] = (slices[0][0], "none")
    else:
        result = []
        for i, (rows, _) in enumerate(slices):
            if i == 0:
                role = "first"
            elif i == len(slices) - 1:
                role = "last"
            else:
                role = "middle"
            result.append((rows, role))
        slices = result

    return slices


def _stage_b_table_slice(blocks: list[Block]) -> list[Block]:
    """Slice large tables within blocks."""
    result: list[Block] = []

    for block in blocks:
        # Find table content in block
        has_big_table = False
        table_json = None
        table_header = None
        table_content_idx = -1

        for idx, part in enumerate(block.content_parts):
            if part.startswith("<table>") and part.endswith("</table>"):
                inner = part[7:-8]  # strip <table></table>
                try:
                    parsed = json.loads(inner)
                    tokens = estimate_tokens(part)
                    if tokens > TABLE_MAX_TOKENS:
                        has_big_table = True
                        table_json = parsed
                        table_header = block.table_header
                        table_content_idx = idx
                except Exception:
                    pass
                break

        if not has_big_table or table_json is None:
            result.append(block)
            continue

        # Split: before-table parts, table slices, after-table parts
        before_parts = block.content_parts[:table_content_idx]
        after_parts = block.content_parts[table_content_idx + 1 :]
        slices = _slice_table_rows(table_json, table_header)

        for si, (rows, role) in enumerate(slices):
            table_content = f"<table>{json.dumps(rows, ensure_ascii=False)}</table>"

            if si == 0:
                # First slice merges with before-table content
                new_parts = list(before_parts) + [table_content]
                heading = block.heading
            elif si == len(slices) - 1:
                # Last slice merges with after-table content
                new_parts = [table_content] + list(after_parts)
                heading = f"{block.heading} [表格片段{si + 1}]"
                after_parts = []  # consumed
            else:
                # Middle slices are standalone
                new_parts = [table_content]
                heading = f"{block.heading} [表格片段{si + 1}]"

            new_block = Block(
                heading=heading,
                parent_headings=list(block.parent_headings),
                level=block.level,
                content_parts=new_parts,
                uuid=block.uuid,
                uuid_end=block.uuid_end,
                table_chunk_role=role,
                table_header=table_header if role in ("middle", "last") else None,
            )
            result.append(new_block)

        # If after_parts remain (no last slice consumed them)
        if after_parts:
            result[-1].content_parts.extend(after_parts)

    return result


# ── Stage C: Long block anchor splitting ─────────────────────────────
def _stage_c_anchor_split(blocks: list[Block]) -> list[Block]:
    """Split blocks exceeding MAX_BLOCK_CONTENT_TOKENS at anchor points."""
    result: list[Block] = []

    for block in blocks:
        if block.tokens <= MAX_BLOCK_CONTENT_TOKENS:
            result.append(block)
            continue

        # Don't re-split table-only blocks
        if block.table_chunk_role == "middle":
            result.append(block)
            continue

        # Find anchor candidates: short paragraphs
        sub_blocks: list[Block] = []
        current = Block(
            heading=block.heading,
            parent_headings=list(block.parent_headings),
            level=block.level,
            content_parts=[],
            uuid=block.uuid,
            uuid_end=block.uuid_end,
            table_chunk_role=block.table_chunk_role,
            table_header=block.table_header,
        )

        for part in block.content_parts:
            current.content_parts.append(part)

            if current.tokens > MAX_BLOCK_CONTENT_TOKENS:
                # Try to find an anchor point
                is_anchor = (
                    len(part) <= MAX_ANCHOR_CANDIDATE_LENGTH
                    and not part.startswith("<table>")
                    and part.strip()
                )

                if is_anchor and len(current.content_parts) > 1:
                    # Split before this anchor
                    anchor_part = current.content_parts.pop()

                    if current.content_parts:
                        sub_blocks.append(current)

                    # New block starts with anchor as heading
                    current = Block(
                        heading=anchor_part.strip(),
                        parent_headings=list(block.parent_headings)
                        + ([block.heading] if block.heading else []),
                        level=block.level + 1 if block.level < 9 else block.level,
                        content_parts=[anchor_part],
                        uuid=block.uuid,
                        uuid_end=block.uuid_end,
                        table_chunk_role="none",
                    )

        if current.content_parts:
            sub_blocks.append(current)

        # Hard fallback split: if anchor strategy still leaves oversize blocks,
        # split inside block content by sentence/char boundaries.
        refined_sub_blocks: list[Block] = []
        for sb in sub_blocks:
            if sb.tokens <= MAX_BLOCK_CONTENT_TOKENS or sb.table_chunk_role == "middle":
                refined_sub_blocks.append(sb)
                continue
            refined_sub_blocks.extend(_force_split_oversize_block(sb))
        sub_blocks = refined_sub_blocks

        if sub_blocks:
            result.extend(sub_blocks)
        else:
            result.append(block)

    return result


def _force_split_oversize_block(
    block: Block, max_tokens: int = MAX_BLOCK_CONTENT_TOKENS
) -> list[Block]:
    """Force split oversize block when no suitable anchor is found.

    Strategy:
    1) sentence-level split by punctuation
    2) if a sentence itself is too long, char-level fallback
    """
    content = block.content
    if not content:
        return [block]

    units = _split_text_units(content)
    merged_units = _merge_units_by_token_limit(units, max_tokens)

    if len(merged_units) <= 1 and estimate_tokens(merged_units[0]) > max_tokens:
        # Char-level emergency split
        merged_units = _char_level_split(merged_units[0], max_tokens)

    if len(merged_units) <= 1:
        return [block]

    out: list[Block] = []
    for i, part in enumerate(merged_units):
        nb = Block(
            heading=block.heading if i == 0 else f"{block.heading} [续{i}]",
            parent_headings=list(block.parent_headings),
            level=block.level,
            content_parts=[part],
            uuid=block.uuid,
            uuid_end=block.uuid_end,
            table_chunk_role=block.table_chunk_role,
            table_header=block.table_header,
        )
        out.append(nb)
    return out


def _split_text_units(text: str) -> list[str]:
    """Split text into sentence-like units while preserving punctuation."""
    # First split by paragraph breaks
    paras = [p for p in text.split("\n\n") if p.strip()]
    units: list[str] = []
    for p in paras:
        # Chinese/English sentence punctuation
        pieces = re.split(r"(?<=[。！？；.!?])", p)
        for s in pieces:
            ss = s.strip()
            if ss:
                units.append(ss)
    return units if units else [text]


def _merge_units_by_token_limit(units: list[str], max_tokens: int) -> list[str]:
    merged: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    for u in units:
        ut = estimate_tokens(u)
        if cur and cur_tokens + ut > max_tokens:
            merged.append("\n\n".join(cur))
            cur = [u]
            cur_tokens = ut
        else:
            cur.append(u)
            cur_tokens += ut

    if cur:
        merged.append("\n\n".join(cur))
    return merged


def _char_level_split(text: str, max_tokens: int) -> list[str]:
    """Emergency split by chars when sentence units are still too large."""
    if not text:
        return [text]
    # Conservative char window for Chinese-heavy text
    window = max(200, int(max_tokens / 1.0))
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + window, n)
        # try backtrack to punctuation
        k = max(text.rfind("。", i, j), text.rfind("；", i, j), text.rfind("\n", i, j))
        if k > i + 50:
            j = k + 1
        out.append(text[i:j].strip())
        i = j
    return [x for x in out if x]


# ── Stage D: Small block merging ─────────────────────────────────────
def _is_heading_only_block(block: Block) -> bool:
    if not block.content_parts:
        return False
    content = block.content.strip()
    heading = (block.heading or "").strip()
    if not heading:
        return False
    # Exactly heading-only or heading + tiny punctuation tail
    if content == heading:
        return True
    if len(content) <= len(heading) + 4 and content.startswith(heading):
        return True
    return False


def _is_isolated_small_block(block: Block) -> bool:
    """True only for absorbable tiny fragments (not normal正文块)."""
    if _is_heading_only_block(block):
        return True
    # Aggressively absorb tiny directory/title fragments (<100 tokens).
    if (
        block.tokens < 100
        and block.table_chunk_role == "none"
        and not any(
            isinstance(part, str)
            and part.startswith("<table>")
            and part.endswith("</table>")
            for part in block.content_parts
        )
    ):
        return True
    if block.tokens >= SMALL_TAIL_THRESHOLD:
        return False

    # Must have heading-style structure and very short body tail.
    heading = (block.heading or "").strip()
    if not heading or not block.content_parts:
        return False
    first = (block.content_parts[0] or "").strip()
    if first != heading:
        return False

    # Allow absorbing only when body is extremely short (e.g., stray sentence)
    if len(block.content_parts) <= 2:
        body = (block.content_parts[1] if len(block.content_parts) == 2 else "").strip()
        return estimate_tokens(body) <= 200

    return False


def _is_parent_child(prev: Block, nxt: Block) -> bool:
    """Whether nxt is likely a child section block of prev."""
    if nxt.level != prev.level + 1:
        return False
    if not prev.heading:
        return False
    return prev.heading in (nxt.parent_headings or [])


def _merge_into_left(left: Block, right: Block) -> Block:
    """Merge right block into left block (append content)."""
    left.content_parts.extend(right.content_parts)
    left.uuid_end = right.uuid_end or left.uuid_end
    # Keep table metadata conservative
    if left.table_header is None and right.table_header is not None:
        left.table_header = right.table_header
    return left


def _merge_into_right(left: Block, right: Block) -> Block:
    """Merge left block into right block (prepend content)."""
    right.content_parts = left.content_parts + right.content_parts
    right.uuid = left.uuid or right.uuid
    if right.table_header is None and left.table_header is not None:
        right.table_header = left.table_header
    return right


def _target_distance(tokens: int) -> int:
    return abs(tokens - IDEAL_BLOCK_CONTENT_TOKENS)


def _absorb_isolated_small_blocks(blocks: list[Block]) -> list[Block]:
    """Absorb heading-only or very small blocks, preferring child-neighbor.

    This directly addresses fragments like:
      3.3 (tiny heading-only block) + 3.3.1 (large child block)
    """
    if len(blocks) < 2:
        return blocks

    work = list(blocks)
    i = 0
    while i < len(work):
        cur = work[i]
        if not _is_isolated_small_block(cur) or cur.table_chunk_role == "middle":
            i += 1
            continue

        left_idx = i - 1 if i - 1 >= 0 else None
        right_idx = i + 1 if i + 1 < len(work) else None

        left = work[left_idx] if left_idx is not None else None
        right = work[right_idx] if right_idx is not None else None

        # Build candidates
        candidates: list[tuple[str, int]] = []  # (direction, score)
        if left is not None and left.table_chunk_role != "middle":
            if left.tokens + cur.tokens <= MAX_BLOCK_CONTENT_TOKENS:
                score = _target_distance(left.tokens + cur.tokens)
                candidates.append(("left", score))
        if right is not None and right.table_chunk_role != "middle":
            if right.tokens + cur.tokens <= MAX_BLOCK_CONTENT_TOKENS:
                score = _target_distance(right.tokens + cur.tokens)
                # Prefer parent->child absorption (e.g., 3.3 -> 3.3.1)
                if _is_parent_child(cur, right):
                    score -= 10_000
                candidates.append(("right", score))

        if not candidates:
            i += 1
            continue

        direction = min(candidates, key=lambda x: x[1])[0]
        if direction == "right" and right_idx is not None:
            work[right_idx] = _merge_into_right(cur, work[right_idx])
            del work[i]
            # keep i at current index to re-check merged right
        elif direction == "left" and left_idx is not None:
            work[left_idx] = _merge_into_left(work[left_idx], cur)
            del work[i]
            i = max(left_idx, 0)
        else:
            i += 1

    return work


def _stage_d_merge(blocks: list[Block]) -> list[Block]:
    """Merge small consecutive blocks while respecting constraints."""
    if not blocks:
        return blocks

    # Pass 1: absorb isolated tiny/heading-only blocks
    premerged = _absorb_isolated_small_blocks(blocks)
    if not premerged:
        return premerged

    # Pass 2: regular same-level merge with IDEAL target preference (left-fold)
    result: list[Block] = [premerged[0]]

    for i in range(1, len(premerged)):
        current = premerged[i]
        prev = result[-1]

        # Never merge into/from middle table slices
        if prev.table_chunk_role == "middle" or current.table_chunk_role == "middle":
            result.append(current)
            continue

        # Check if mergeable: same level or compatible levels
        can_merge = (
            prev.tokens + current.tokens <= MAX_BLOCK_CONTENT_TOKENS
            and prev.level == current.level
        )

        if can_merge:
            prev.content_parts.extend(current.content_parts)
            prev.uuid_end = current.uuid_end or prev.uuid_end
        else:
            result.append(current)

    # Tail absorption: absorb tiny trailing blocks
    if len(result) >= 2:
        last = result[-1]
        second_last = result[-2]
        if (
            last.tokens < SMALL_TAIL_THRESHOLD
            and second_last.tokens + last.tokens <= MAX_BLOCK_CONTENT_TOKENS
            and last.table_chunk_role != "middle"
            and second_last.table_chunk_role != "middle"
        ):
            second_last.content_parts.extend(last.content_parts)
            second_last.uuid_end = last.uuid_end or second_last.uuid_end
            result.pop()

    return result


def _stage_e_extraction_safety_split(blocks: list[Block]) -> list[Block]:
    """Final split pass for extraction safety while preserving heading hierarchy."""
    if EXTRACTION_SAFE_MAX_BLOCK_TOKENS <= 0:
        return blocks

    result: list[Block] = []
    for block in blocks:
        if block.tokens <= EXTRACTION_SAFE_MAX_BLOCK_TOKENS:
            result.append(block)
            continue

        # Keep table chunks as-is to avoid breaking table payload structure.
        if any(
            isinstance(part, str)
            and part.startswith("<table>")
            and part.endswith("</table>")
            for part in block.content_parts
        ):
            result.append(block)
            continue

        result.extend(
            _force_split_oversize_block(
                block=block, max_tokens=EXTRACTION_SAFE_MAX_BLOCK_TOKENS
            )
        )
    return result


# ── Public API ───────────────────────────────────────────────────────
def smart_chunk(paragraphs: list[Paragraph]) -> list[Block]:
    """Run the full smart chunking pipeline."""
    blocks = _stage_a_heading_split(paragraphs)
    blocks = _stage_b_table_slice(blocks)
    blocks = _stage_c_anchor_split(blocks)
    blocks = _stage_d_merge(blocks)
    blocks = _stage_e_extraction_safety_split(blocks)
    return blocks
