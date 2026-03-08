"""End-to-end document parser: .docx → _blocks.jsonl content.

Public API:
    parse_docx_to_jsonl(file_bytes, source_file) → str  (JSONL text)
    parse_docx_to_blocks(file_bytes)              → list[dict]
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from .docx_extractor import extract_docx_paragraphs
from .smart_chunker import Block, smart_chunk
from .token_estimation import estimate_tokens


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _block_to_dict(block: Block, index: int) -> dict[str, Any]:
    """Convert a Block to the _blocks.jsonl chunk dict."""
    return {
        "uuid": block.uuid,
        "uuid_end": block.uuid_end,
        "heading": block.heading,
        "content": block.content,
        "type": "text",
        "parent_headings": block.parent_headings,
        "level": block.level,
        "table_chunk_role": block.table_chunk_role,
        **({"table_header": block.table_header} if block.table_header else {}),
    }


def _build_meta(source_file: str, source_hash: str) -> dict[str, Any]:
    """Build the meta line for _blocks.jsonl."""
    return {
        "type": "meta",
        "source_file": source_file,
        "source_hash": f"sha256:{source_hash}",
        "parsed_at": datetime.now().isoformat(),
    }


# ── Public API ───────────────────────────────────────────────────────
def parse_docx_to_blocks(file_bytes: bytes) -> list[dict[str, Any]]:
    """Parse .docx bytes into a list of block dicts (without meta line).

    Each dict matches the _blocks.jsonl chunk schema.
    """
    paragraphs = extract_docx_paragraphs(file_bytes)
    blocks = smart_chunk(paragraphs)
    return [_block_to_dict(b, i) for i, b in enumerate(blocks)]


def parse_docx_to_jsonl(
    file_bytes: bytes,
    source_file: str = "document.docx",
) -> str:
    """Parse .docx bytes into a complete JSONL string (meta + chunks).

    The returned string can be:
    - Written directly to a .jsonl file
    - Fed into LightRAG interchange format consumer
    """
    source_hash = _sha256_hex(file_bytes)
    meta = _build_meta(source_file, source_hash)

    paragraphs = extract_docx_paragraphs(file_bytes)
    blocks = smart_chunk(paragraphs)

    lines: list[str] = [json.dumps(meta, ensure_ascii=False)]
    for i, block in enumerate(blocks):
        chunk_dict = _block_to_dict(block, i)
        lines.append(json.dumps(chunk_dict, ensure_ascii=False))

    return "\n".join(lines)


def parse_docx_to_interchange_jsonl(
    file_bytes: bytes,
    source_file: str = "document.docx",
    doc_id: str = "",
) -> str:
    """Parse .docx into LightRAG 2.0 interchange format JSONL.

    Compatible with extraction_interchange.py consumer.
    Adds chunk_id, chunk_order_index, tokens, content_type fields.
    """
    source_hash = _sha256_hex(file_bytes)

    meta = {
        "type": "meta",
        "format_version": "2.0",
        "source_file": source_file,
        "source_hash": f"sha256:{source_hash}",
        "doc_id": doc_id,  # may be empty before enqueue
        "engine": "default",
        "engine_capabilities": ["t"],  # tables supported
        "chunking_method": "heading_semantic",
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }

    paragraphs = extract_docx_paragraphs(file_bytes)
    blocks = smart_chunk(paragraphs)

    meta["total_chunks"] = len(blocks)
    chunk_id_prefix = doc_id.strip() if isinstance(doc_id, str) and doc_id.strip() else f"doc-{source_hash[:12]}"

    lines: list[str] = [json.dumps(meta, ensure_ascii=False)]
    table_group_ids: dict[str, str] = {}

    for i, block in enumerate(blocks):
        content = block.content
        is_table_block = (
            block.table_chunk_role != "none"
            or (content.startswith("<table>") and content.endswith("</table>"))
        )

        table_fragment_num = None
        table_meta = None
        content_md = None

        if is_table_block:
            rows = _extract_table_rows_from_content(content)
            content_md = _table_rows_to_markdown(rows) if rows is not None else None

            # For fragmented tables, try to keep one ID shared across fragments
            heading_base = re.sub(r"\s*\[表格片段\d+\]\s*$", "", block.heading or "")
            group_key = f"{block.uuid}|{heading_base}|{'/'.join(block.parent_headings)}"
            if group_key not in table_group_ids:
                table_group_ids[group_key] = _make_table_id(
                    group_key, source_hash, i
                )
            table_id = table_group_ids[group_key]

            if block.table_chunk_role in {"first", "middle", "last"}:
                table_fragment_num = _extract_fragment_num(block.heading)
            else:
                table_fragment_num = None

            num_rows = len(rows) - 1 if rows else 0
            num_cols = len(rows[0]) if rows and rows[0] else 0
            has_header = bool(block.table_header)

            table_meta = {
                "table_id": table_id,
                "table_title": heading_base,
                "format": "json",
                "num_rows": max(num_rows, 0),
                "num_cols": max(num_cols, 0),
                "has_header": has_header,
                "table_header": block.table_header or [],
                "summary": "",
            }

        chunk = {
            "type": "text",
            "chunk_id": f"{chunk_id_prefix}-chunk-{i:03d}",
            "chunk_order_index": i,
            "content": content,
            "content_md": content_md,
            "tokens": estimate_tokens(content),
            "heading": block.heading,
            "parent_headings": block.parent_headings,
            "level": block.level,
            "content_type": "table" if is_table_block else "body",
            "uuid": block.uuid,
            "uuid_end": block.uuid_end,
            "table_chunk_role": block.table_chunk_role,
            "table_fragment_num": table_fragment_num,
            "table_meta": table_meta,
        }
        if block.table_header:
            chunk["table_header"] = block.table_header

        lines.append(json.dumps(chunk, ensure_ascii=False))

    return "\n".join(lines)


def _extract_table_rows_from_content(content: str) -> list[list[str]] | None:
    if not (content.startswith("<table>") and content.endswith("</table>")):
        return None
    raw = content[7:-8]
    try:
        rows = json.loads(raw)
        if isinstance(rows, list):
            return rows
    except Exception:
        return None
    return None


def _table_rows_to_markdown(rows: list[list[str]] | None) -> str | None:
    if not rows or not rows[0]:
        return None
    header = [str(x) for x in rows[0]]
    body = rows[1:] if len(rows) > 1 else []
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in body:
        cells = [str(x) for x in r]
        if len(cells) < len(header):
            cells.extend([""] * (len(header) - len(cells)))
        md_lines.append("| " + " | ".join(cells[: len(header)]) + " |")
    return "\n".join(md_lines)


def _make_table_id(group_key: str, source_hash: str, idx: int) -> str:
    digest = hashlib.sha256(f"{group_key}|{source_hash}|{idx}".encode("utf-8")).hexdigest()
    return f"tb-{digest[:8].upper()}"


def _extract_fragment_num(heading: str) -> int | None:
    m = re.search(r"\[表格片段(\d+)\]\s*$", heading or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None
