"""PaddleOCR-VL JSON output -> LightRAG sidecar IR."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from lightrag.parser._html_table import (
    extract_html_table_info,
    html_table_inner_body,
    looks_like_html_table_payload,
    unwrap_html_table,
)
from lightrag.parser._markdown import (
    render_heading_line,
    strip_heading_markdown_prefix,
)
from lightrag.parser.external.paddleocr_vl.cache import (
    CONTENT_LIST_FILENAME,
    DEFAULT_PADDLEOCR_VL_ENGINE_VERSION,
)
from lightrag.sidecar.ir import (
    AssetSpec,
    IRBlock,
    IRDoc,
    IRDrawing,
    IREquation,
    IRPosition,
    IRTable,
)
from lightrag.utils import logger

PREFACE_HEADING = "Preface/Uncategorized"
_IMG_SRC_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.I)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_IMG_IN_IMAGE_BOX_RE = re.compile(r"^img_in_image_box_(\d+)_(\d+)_(\d+)_(\d+)$", re.I)
_SKIP_LABELS = {
    "number",
    "formula_number",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
}
_TEXT_LABELS = {"text", "abstract", "reference_content", "footnote", "figure_title"}
_MEDIA_LABELS = {"table", "image"}
_FIGURE_TITLE_LABEL = "figure_title"


class PaddleOCRVLIRBuilder:
    def __init__(self) -> None:
        self.engine_version = (
            os.getenv(
                "PADDLEOCR_VL_ENGINE_VERSION", DEFAULT_PADDLEOCR_VL_ENGINE_VERSION
            ).strip()
            or DEFAULT_PADDLEOCR_VL_ENGINE_VERSION
        )
        self.bbox_attributes = {"origin": "LEFTTOP"}

    def normalize_from_workdir(self, raw_dir: Path, *, document_name: str) -> IRDoc:
        result_path = raw_dir / CONTENT_LIST_FILENAME
        if not result_path.is_file():
            raise FileNotFoundError(
                f"PaddleOCR-VL raw bundle missing {CONTENT_LIST_FILENAME}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        pages = _extract_pages(payload)
        if not pages:
            raise ValueError(f"PaddleOCR-VL {CONTENT_LIST_FILENAME} contains no pages")
        return self._normalize_pages(pages, raw_dir, document_name=document_name)

    def _normalize_pages(
        self, pages: list[dict[str, Any]], raw_dir: Path, *, document_name: str
    ) -> IRDoc:
        blocks: list[IRBlock] = []
        assets: list[AssetSpec] = []
        asset_refs: set[str] = set()
        doc_title = ""
        placeholder_counter = 0
        heading_stack: list[str] = []

        cb_lines: list[str] = []
        cb_tables: list[IRTable] = []
        cb_drawings: list[IRDrawing] = []
        cb_equations: list[IREquation] = []
        cb_positions: list[IRPosition] = []
        cb_heading = PREFACE_HEADING
        cb_level = 0
        cb_parents: list[str] = []

        def next_key(prefix: str) -> str:
            nonlocal placeholder_counter
            placeholder_counter += 1
            return f"{prefix}{placeholder_counter}"

        def flush() -> None:
            nonlocal cb_lines, cb_tables, cb_drawings, cb_equations, cb_positions
            if not (cb_lines or cb_tables or cb_drawings or cb_equations):
                return
            content = "\n".join(line for line in cb_lines if line)
            if not content.strip() and not (cb_tables or cb_drawings or cb_equations):
                cb_lines = []
                cb_positions = []
                return
            blocks.append(
                IRBlock(
                    content_template=content,
                    heading=cb_heading,
                    level=cb_level,
                    parent_headings=list(cb_parents),
                    positions=list(cb_positions),
                    tables=list(cb_tables),
                    drawings=list(cb_drawings),
                    equations=list(cb_equations),
                )
            )
            cb_lines = []
            cb_tables = []
            cb_drawings = []
            cb_equations = []
            cb_positions = []

        def open_block(heading: str, level: int, raw_heading: str) -> None:
            nonlocal cb_heading, cb_level, cb_parents
            flush()
            parents = [h for h in heading_stack[: max(level - 1, 0)] if h]
            cb_heading = heading
            cb_level = level
            cb_parents = parents
            cb_lines.append(render_heading_line(level, raw_heading))

        def record_position(item: dict[str, Any], page_index: int) -> None:
            position = _position_from_item(item, page_index)
            if position is not None:
                cb_positions.append(position)

        for page_index, page in enumerate(pages):
            # Each item in items contains the following key-value pairs:
            # - block_bbox: (np.ndarray) Bounding box of the layout region
            # - block_label: (str) Label of the layout region, e.g., text, table, etc.
            # - block_content: (str) Content within the layout region
            # - block_id: (int) Index of the layout region, used for displaying layout sorting results
            # - block_order: (int) Reading order of the layout region, None for unsorted parts
            items = _page_items(page)
            captions_by_index, consumed_caption_indexes = _caption_map_for_items(items)
            # Build bbox -> image path mapping from markdown.images
            page_images = _page_images_map(page)
            for item_index, item in enumerate(items):
                label = _item_label(item)
                if label in _SKIP_LABELS:
                    continue
                if (
                    label == _FIGURE_TITLE_LABEL
                    and item_index in consumed_caption_indexes
                ):
                    continue
                text = _item_content(item)

                # Handle images first since they may have empty content
                if label == "image":
                    bbox = item.get("block_bbox") or item.get("bbox")
                    src = page_images.get(tuple(bbox) if bbox else None, "")
                    if not src:
                        src = _extract_image_src(text)
                    drawing, asset = _build_drawing(
                        item,
                        src,
                        raw_dir,
                        page_index=page_index,
                        item_index=item_index,
                        caption=captions_by_index.get(item_index, ""),
                    )
                    drawing.placeholder_key = next_key("im")
                    cb_drawings.append(drawing)
                    if asset is not None and asset.ref not in asset_refs:
                        assets.append(asset)
                        asset_refs.add(asset.ref)
                    cb_lines.append(f"{{{{IMG:{drawing.placeholder_key}}}}}")
                    record_position(item, page_index)
                    continue

                if not text:
                    continue

                heading_level = _heading_level(label, text)
                if heading_level is not None:
                    clean_heading = strip_heading_markdown_prefix(text)
                    heading_stack[:] = heading_stack[: max(heading_level - 1, 0)]
                    heading_stack.append(clean_heading)
                    open_block(clean_heading, heading_level, text)
                    record_position(item, page_index)
                    if not doc_title and heading_level == 1:
                        doc_title = clean_heading
                    continue

                if label == "table":
                    table = _build_table(
                        item, caption=captions_by_index.get(item_index, "")
                    )
                    if table is None:
                        continue
                    table.placeholder_key = next_key("tb")
                    table.self_ref = f"{CONTENT_LIST_FILENAME}#/{page_index}/prunedResult/parsing_res_list/{item_index}"
                    cb_tables.append(table)
                    cb_lines.append(f"{{{{TBL:{table.placeholder_key}}}}}")
                    record_position(item, page_index)
                    continue

                if label == "display_formula":
                    equation = IREquation(
                        placeholder_key=next_key("eq"),
                        latex=text.strip(),
                        is_block=True,
                        self_ref=(
                            f"{CONTENT_LIST_FILENAME}#/{page_index}/prunedResult/"
                            f"parsing_res_list/{item_index}"
                        ),
                    )
                    cb_equations.append(equation)
                    cb_lines.append(f"{{{{EQ:{equation.placeholder_key}}}}}")
                    record_position(item, page_index)
                    continue

                if label in _TEXT_LABELS or not label:
                    cb_lines.append(text)
                    record_position(item, page_index)
                    continue

                cb_lines.append(text)
                record_position(item, page_index)

        flush()
        if not doc_title:
            doc_title = Path(document_name).stem or document_name

        split_option = (
            {"engine_version": self.engine_version} if self.engine_version else {}
        )
        return IRDoc(
            document_name=document_name,
            document_format=Path(document_name).suffix.lower().lstrip("."),
            doc_title=doc_title,
            split_option=split_option,
            blocks=blocks,
            assets=assets,
            bbox_attributes=dict(self.bbox_attributes),
        )


def _extract_pages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(
            result.get("layoutParsingResults"), list
        ):
            return [p for p in result["layoutParsingResults"] if isinstance(p, dict)]
        if isinstance(payload.get("layoutParsingResults"), list):
            return [p for p in payload["layoutParsingResults"] if isinstance(p, dict)]
    return []


def _page_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    pruned = page.get("prunedResult")
    items = pruned.get("parsing_res_list") if isinstance(pruned, dict) else None
    return (
        [item for item in items if isinstance(item, dict)]
        if isinstance(items, list)
        else []
    )


def _page_images_map(page: dict[str, Any]) -> dict[tuple[int, int, int, int], str]:
    """Build mapping from bbox tuple to image path from markdown.images.

    PaddleOCR-VL stores image paths in ``page.markdown.images``, where keys
    are paths like ``imgs/img_in_image_box_X1_Y1_X2_Y2.jpg``. This function
    extracts bbox coordinates from those filenames and creates a lookup.
    """
    result = {}
    markdown = page.get("markdown") if isinstance(page, dict) else None
    images: dict[str, str] | None = (
        markdown.get("images") if isinstance(markdown, dict) else None
    )
    if not isinstance(images, dict):
        return result
    for path in images.keys():
        bbox = _parse_bbox_from_image_path(path)
        if bbox:
            result[bbox] = path
    return result


def _parse_bbox_from_image_path(path: str) -> tuple[int, int, int, int] | None:
    """Extract bbox coordinates from PaddleOCR-VL image paths.

    Path format: ``imgs/img_in_image_box_X1_Y1_X2_Y2.jpg``
    """
    try:
        stem = Path(path).stem
        match = _IMG_IN_IMAGE_BOX_RE.fullmatch(stem)
        if match:
            return tuple(int(x) for x in match.groups())  # type: ignore
    except (ValueError, TypeError):
        pass
    return None


def _heading_level(label: str, text: str) -> int | None:
    if label == "doc_title":
        return 1
    if label == "paragraph_title":
        stripped = text.lstrip()
        hashes = len(stripped) - len(stripped.lstrip("#"))
        return max(2, min(hashes or 2, 6))
    return None


def _item_content(item: dict[str, Any]) -> str:
    content = item.get("block_content")
    if content is None:
        content = item.get("text", item.get("content"))
    return str(content or "").strip()


def _item_label(item: dict[str, Any]) -> str:
    label = item.get("block_label")
    if label is None:
        label = item.get("label", "")
    return str(label or "").strip().lower()


def _caption_map_for_items(
    items: list[dict[str, Any]],
) -> tuple[dict[int, str], set[int]]:
    """Build mapping from media elements (images/tables) to their captions.

    PaddleOCR-VL treats figure/table titles as independent text blocks labeled
    'figure_title' rather than attaching them directly to their corresponding
    image/table elements. This function finds which title belongs to which media
    by examining adjacent items and applying layout heuristics (e.g., table
    captions typically appear above tables, figure captions below images).

    Returns:
        captions_by_index: Dict mapping media item index to its caption text
        consumed_caption_indexes: Set of title indices that were successfully
            matched and should not be rendered as separate text blocks
    """
    captions_by_index: dict[int, str] = {}
    consumed_caption_indexes: set[int] = set()
    for index, item in enumerate(items):
        if _item_label(item) != _FIGURE_TITLE_LABEL:
            continue
        caption = _item_content(item)
        if not caption:
            continue
        target_index = _adjacent_caption_target(items, index, caption)
        if target_index is None or target_index in captions_by_index:
            continue
        captions_by_index[target_index] = caption
        consumed_caption_indexes.add(index)
    return captions_by_index, consumed_caption_indexes


def _adjacent_caption_target(
    items: list[dict[str, Any]], caption_index: int, caption: str
) -> int | None:
    # Adjacent positions relative to caption in the rendered layout flow. OCR
    # may insert page numbers or headers between a caption and its media, so
    # skip ignorable layout noise while preserving real content boundaries.
    prev_index = _nearest_neighbor(items, caption_index, step=-1)
    next_index = _nearest_neighbor(items, caption_index, step=1)

    candidates: list[int] = []
    for index in (prev_index, next_index):
        if index is not None and _item_label(items[index]) in _MEDIA_LABELS:
            candidates.append(index)
    if not candidates:
        return None

    # - Tables: captions typically appear ABOVE the table → table after caption
    # - Images: captions typically appear BELOW the image → image before caption
    preferred_label = _caption_preferred_media_label(caption)
    if preferred_label == "table" and next_index in candidates:
        return next_index
    if preferred_label == "image" and prev_index in candidates:
        return prev_index
    if preferred_label:
        for index in candidates:
            if _item_label(items[index]) == preferred_label:
                return index

    candidate_labels = {_item_label(items[i]) for i in candidates}
    if len(candidate_labels) == 1:
        only_label = next(iter(candidate_labels))
        if only_label == "table" and next_index in candidates:
            return next_index
        if only_label == "image" and prev_index in candidates:
            return prev_index

    if next_index in candidates:
        return next_index
    return candidates[0]


def _nearest_neighbor(
    items: list[dict[str, Any]], caption_index: int, *, step: int
) -> int | None:
    index = caption_index + step
    while 0 <= index < len(items):
        item = items[index]
        label = _item_label(item)
        if label in _SKIP_LABELS:
            index += step
            continue
        return index
    return None


def _caption_preferred_media_label(caption: str) -> str:
    normalized = caption.lstrip().lower()
    if normalized.startswith(("table", "tab.", "tbl.", "表", "表格")):
        return "table"
    if normalized.startswith(("figure", "fig", "plate", "图", "图片", "插图", "附图")):
        return "image"
    return ""


def _position_from_item(item: dict[str, Any], page_index: int) -> IRPosition | None:
    bbox = item.get("block_bbox") or item.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            return IRPosition(
                type="bbox",
                anchor=str(page_index + 1),
                range=[int(x) for x in bbox],
            )
        except (TypeError, ValueError):
            logger.debug("[paddleocr_vl] skipping malformed bbox %r", bbox)
    return IRPosition(type="bbox", anchor=str(page_index + 1))


def _build_table(item: dict[str, Any], *, caption: str = "") -> IRTable | None:
    raw = _item_content(item)
    if not raw:
        return None
    html = unwrap_html_table(raw) if looks_like_html_table_payload(raw) else raw
    html = html.strip()
    if not html:
        return None
    info = (
        extract_html_table_info(html) if looks_like_html_table_payload(html) else None
    )
    return IRTable(
        placeholder_key="",
        html=html,
        num_rows=info.num_rows if info else 0,
        num_cols=info.num_cols if info else 0,
        caption=caption,
        body_override=html_table_inner_body(html) or None,
    )


def _extract_image_src(text: str) -> str:
    m = _IMG_SRC_RE.search(text)
    if m:
        return m.group(1)
    m = _MARKDOWN_IMAGE_RE.search(text)
    return m.group(1) if m else ""


def _build_drawing(
    item: dict[str, Any],
    src: str,
    raw_dir: Path,
    *,
    page_index: int,
    item_index: int,
    caption: str = "",
) -> tuple[IRDrawing, AssetSpec | None]:
    asset_ref = src or f"paddleocr_vl_image_{page_index}_{item_index}"
    source = _resolve_asset_source(raw_dir, asset_ref)
    suggested = Path(asset_ref.replace("\\", "/")).name or f"image_{page_index}.jpg"
    fmt = Path(suggested).suffix.lower().lstrip(".") or "jpg"
    asset = AssetSpec(asset_ref, suggested, source) if source is not None else None
    drawing = IRDrawing(
        placeholder_key="",
        asset_ref=asset_ref,
        fmt=fmt,
        src=src,
        caption=caption,
        self_ref=f"{CONTENT_LIST_FILENAME}#/{page_index}/prunedResult/parsing_res_list/{item_index}",
    )
    return drawing, asset


def _resolve_asset_source(raw_dir: Path, asset_ref: str) -> Path | None:
    if not asset_ref or "://" in asset_ref:
        return None
    candidate = raw_dir / Path(asset_ref.replace("\\", "/"))
    try:
        candidate.resolve().relative_to(raw_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


__all__ = ["PaddleOCRVLIRBuilder"]
