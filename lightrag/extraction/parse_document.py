"""End-to-end document parser: .docx → LightRAG content_list.

Public API:
    parse_docx_to_lightrag_content_list(file_bytes, source_file, doc_id)
        → (content_list, asset_blobs)  — feed into
        ``LightRAG._write_lightrag_document_from_content_list`` to emit the
        canonical ``.blocks.jsonl`` + ``tables.json`` + ``drawings.json``
        + ``.blocks.assets/`` artifacts.
    paragraphs_to_content_list(paragraphs, images, asset_dir_rel)
        → list[dict]  — pure conversion helper (used internally and by tests).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pathlib import Path

from .docx_extractor import extract_docx_images, extract_docx_paragraphs


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# LightRAG Document content_list builders
#
# These build the same ``content_list`` schema that the mineru/docling
# bindings emit, so the native docx parser can hand off to
# ``LightRAG._write_lightrag_document_from_content_list`` and produce
# the standard ``.blocks.jsonl`` + ``tables.json`` + ``drawings.json``
# + ``.blocks.assets/`` artifacts.
# ─────────────────────────────────────────────────────────────────────

# Inline tags emitted into paragraph text by ``docx_extractor``:
#   - ``<drawing id="N" name="..." />`` for each picture
#   - ``<equation>...</equation>`` for OMML formulas
_INLINE_DRAWING_OR_EQUATION = re.compile(
    r'<drawing\s+id="(?P<dr_id>[^"]*)"\s+name="(?P<dr_name>[^"]*)"\s*/>'
    r"|<equation>(?P<eq>.*?)</equation>"
)


def _split_inline_paragraph(
    text: str,
    drawing_rids: list[str],
    images: dict[str, tuple[str, bytes]],
    asset_dir_rel: str,
) -> list[dict[str, Any]]:
    """Split a body paragraph into ordered text/image/equation items.

    ``drawing_rids`` is consumed in document order — ``<drawing>`` tag #i
    inside the paragraph maps to ``drawing_rids[i]``.
    """
    items: list[dict[str, Any]] = []
    last_end = 0
    drawing_idx = 0
    for m in _INLINE_DRAWING_OR_EQUATION.finditer(text):
        before = text[last_end : m.start()].strip()
        if before:
            items.append({"type": "text", "text": before})
        if m.group("dr_id") is not None:
            dr_id = m.group("dr_id") or ""
            dr_name = m.group("dr_name") or ""
            rid = drawing_rids[drawing_idx] if drawing_idx < len(drawing_rids) else ""
            drawing_idx += 1
            img_info = images.get(rid)
            if img_info is not None:
                filename, _ = img_info
                items.append(
                    {
                        "type": "image",
                        "id": dr_id,
                        "img_path": (
                            f"{asset_dir_rel}/{filename}" if asset_dir_rel else filename
                        ),
                        "image_caption": [dr_name] if dr_name else [],
                    }
                )
        elif m.group("eq") is not None:
            eq_text = (m.group("eq") or "").strip()
            if eq_text:
                items.append({"type": "equation", "text": eq_text})
        last_end = m.end()
    trailing = text[last_end:].strip()
    if trailing:
        items.append({"type": "text", "text": trailing})
    if not items and text and text.strip():
        items.append({"type": "text", "text": text.strip()})
    return items


def paragraphs_to_content_list(
    paragraphs: list,
    images: dict[str, tuple[str, bytes]],
    asset_dir_rel: str,
) -> list[dict[str, Any]]:
    """Convert docx paragraphs (+ extracted images) into a mineru-style content_list.

    Mapping rules:
      * ``outline_level <= 8``  → ``section_header`` (heading text);
        ``_write_lightrag_document_from_content_list`` will manage
        the heading stack and emit ``parent_headings`` automatically.
      * ``is_table`` w/ ``table_json``  → ``table`` item with parsed rows.
      * Body paragraph text containing ``<drawing>`` / ``<equation>`` tags
        is split in document order so picture and formula items appear
        between the surrounding text fragments (preserving reading order).
      * Empty / whitespace-only paragraphs are skipped.
    """
    content_list: list[dict[str, Any]] = []
    for para in paragraphs:
        if getattr(para, "is_table", False) and getattr(para, "table_json", None):
            rows = para.table_json or []
            content_list.append(
                {
                    "type": "table",
                    "rows": rows,
                    "table_caption": [],
                    "num_rows": len(rows),
                    "num_cols": max((len(r) for r in rows), default=0),
                }
            )
            continue

        text = (getattr(para, "text", "") or "").strip()
        if not text:
            continue

        # Note: don't use ``or 9`` here — outline_level==0 (top heading) is
        # falsy in Python and would be silently demoted to body level.
        raw_level = getattr(para, "outline_level", 9)
        outline_level = int(9 if raw_level is None else raw_level)
        if outline_level <= 8:
            content_list.append(
                {
                    "type": "section_header",
                    "text": text,
                    # docx outline_level is 0-based (0 = top heading).
                    # content_list ``text_level`` is 1-based.
                    "text_level": outline_level + 1,
                }
            )
            continue

        drawing_rids = list(getattr(para, "drawing_rIds", []) or [])
        items = _split_inline_paragraph(text, drawing_rids, images, asset_dir_rel)
        content_list.extend(items)

    return content_list


def parse_docx_to_lightrag_content_list(
    file_bytes: bytes,
    source_file: str = "document.docx",
    doc_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Parse a .docx into a LightRAG-style content_list plus its asset blobs.

    Returns:
        ``(content_list, asset_filename_to_bytes)``. The caller is responsible
        for writing the asset bytes into the parsed-artifact directory's
        ``<base>.blocks.assets/`` folder, and for handing the content_list
        to ``LightRAG._write_lightrag_document_from_content_list`` which
        produces the canonical ``.blocks.jsonl`` + sidecar files.

    Asset directory naming matches the convention used by
    ``_write_lightrag_document_from_content_list``: ``<source-stem>.blocks.assets``.
    """
    source_hash = _sha256_hex(file_bytes)
    images = extract_docx_images(file_bytes)
    paragraphs = extract_docx_paragraphs(file_bytes)

    base_stem = Path(source_file).stem if source_file else ""
    if not base_stem:
        base_stem = (
            doc_id.strip()
            if isinstance(doc_id, str) and doc_id.strip()
            else f"doc-{source_hash[:12]}"
        )
    asset_dir_rel = f"{base_stem}.blocks.assets"

    content_list = paragraphs_to_content_list(paragraphs, images, asset_dir_rel)

    referenced: set[str] = set()
    for item in content_list:
        if item.get("type") == "image":
            img_path = str(item.get("img_path") or "")
            if img_path:
                referenced.add(Path(img_path).name)

    asset_blobs: dict[str, bytes] = {}
    for filename, blob in images.values():
        if filename in referenced and filename not in asset_blobs:
            asset_blobs[filename] = blob

    return content_list, asset_blobs
