"""Interchange JSONL parser and validator.

Implements core rules from docs/extraction-interchange-format-spec-zh.md §8
without multimodal post-processing.
"""

from __future__ import annotations

import json
from typing import Any

from lightrag.utils import Tokenizer


VALID_CONTENT_TYPES = {
    "body",
    "table",
    "image",
    "equation",
    "summary",
    "toc",
    "references",
}


def parse_interchange_jsonl(
    raw_content: str,
    tokenizer: Tokenizer | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Parse and validate interchange JSONL.

    Returns:
      (meta, chunks) on success
      None when input is not valid interchange format.
    """
    if not raw_content:
        return None

    lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    try:
        meta = json.loads(lines[0])
    except Exception:
        return None

    if not _is_valid_meta(meta):
        return None

    raw_chunks: list[dict[str, Any]] = []
    for line in lines[1:]:
        try:
            item = json.loads(line)
        except Exception:
            return None
        if isinstance(item, dict):
            raw_chunks.append(item)

    chunks = _normalize_validate_chunks(raw_chunks, tokenizer)
    if chunks is None or not chunks:
        return None

    return meta, chunks


def _is_valid_meta(meta: dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("type") != "meta":
        return False
    # format_version is required by spec
    fv = meta.get("format_version")
    if fv is None:
        return False
    return True


def _normalize_validate_chunks(
    raw_chunks: list[dict[str, Any]],
    tokenizer: Tokenizer | None,
) -> list[dict[str, Any]] | None:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_chunks):
        if item.get("type") != "text":
            continue

        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            return None

        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            chunk_id = f"chunk-{idx:03d}"
        chunk_id = chunk_id.strip()
        if chunk_id in seen_ids:
            return None
        seen_ids.add(chunk_id)

        chunk_order_index = item.get("chunk_order_index", idx)
        if not isinstance(chunk_order_index, int):
            try:
                chunk_order_index = int(chunk_order_index)
            except Exception:
                return None

        content_type = item.get("content_type", "body")
        if content_type not in VALID_CONTENT_TYPES:
            return None

        tokens = item.get("tokens")
        if not isinstance(tokens, int):
            if tokenizer is not None:
                tokens = len(tokenizer.encode(content))
            else:
                tokens = max(1, int(len(content) * 0.5))

        table_role = item.get("table_chunk_role", "none")
        table_fragment_num = item.get("table_fragment_num")
        table_meta = item.get("table_meta")

        # Spec constraints
        if content_type == "table" and table_meta is None:
            return None
        if table_role != "none" and table_fragment_num is None:
            return None

        parsed = dict(item)
        parsed.pop("type", None)
        parsed["chunk_id"] = chunk_id
        parsed["chunk_order_index"] = chunk_order_index
        parsed["content_type"] = content_type
        parsed["tokens"] = tokens
        parsed["table_chunk_role"] = table_role
        normalized.append(parsed)

    # chunk_order_index continuity check (0..n-1)
    normalized.sort(key=lambda c: c["chunk_order_index"])
    for i, c in enumerate(normalized):
        if c["chunk_order_index"] != i:
            return None

    return normalized
