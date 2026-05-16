"""MinerU adapter: ``content_list.json`` (+ images/) → :class:`IRDoc`.

Input contract: a ``*.mineru_raw/`` directory containing at least
``content_list.json``. Optional sibling resources (``images/``,
``middle.json``, ``full.md``, ``layout.pdf``) are kept as-is; this adapter
only reads the content list and image asset bytes.

Conversion rules (informed by spec §3-§六):

- ``text`` items with ``text_level>0`` and ``title`` / ``section_header``
  start a NEW block. The heading text is rendered with a markdown ``#``
  prefix matching the level (``# foo``, ``## bar`` …) as the first line of
  the new block's content.
- All other items (``text``, ``list``, ``code``, ``table``, ``image``,
  ``equation``) are MERGED into the current block — their text / placeholder
  is appended (newline-separated) to the heading's block. This mirrors the
  native docx parser's "split-by-heading, merge-everything-under-heading"
  behavior (see ``native_parser/docx/parse_document.py``).
- Content emitted before the first heading lands in a synthetic
  ``Preface/Uncategorized`` block at level 0.
- ``list`` items joined with ``\n``; ``code`` body taken from ``code_body``
  if present.
- ``table`` → IRTable + ``{{TBL:k}}`` placeholder. ``table_body`` (HTML) or
  the ``rows`` field (2D array) become ``html`` / ``rows`` on IRTable.
  ``num_rows`` / ``num_cols`` are taken from MinerU if present, otherwise
  inferred. ``header`` populates ``table_header`` (per spec §5).
- ``image`` / ``picture`` / ``drawing`` → IRDrawing + ``{{IMG:k}}`` placeholder.
  Asset bytes are referenced via ``img_path`` relative to the raw dir.
- ``equation`` → IREquation. ``is_block`` is decided by whether
  ``text_format=="block"`` (MinerU explicit flag) OR ``text_level==0`` with
  no inline neighbours; otherwise inline. The latex string is preserved
  verbatim (including any ``$$``/``$`` wrappers) so ``blocks.jsonl``'s
  ``<equation>`` body matches MinerU's raw output; the writer strips the
  wrappers when persisting ``equations.json`` content.
- ``page_idx`` + ``bbox`` → ``IRPosition(type="bbox", anchor=page, range=[x0,y0,x1,y1])``.
  Empty/missing bbox is acceptable; positions accumulate on the merged block.
- ``IRDoc.split_option`` records the MinerU engine version when available.
- ``IRDoc.bbox_attributes`` defaults to ``{"origin":"LEFTTOP","max":1000}``
  reflecting MinerU's PDF coordinate convention. Operators may override
  via ``MINERU_BBOX_ATTRIBUTES`` (JSON string).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


class MinerUAdapter:
    """Stateless except for env-driven config. Reusable across calls."""

    def __init__(self) -> None:
        self.engine_version = os.getenv("MINERU_ENGINE_VERSION", "").strip()
        # Mirror MinerURawClient.__init__: when this is set, the downloader
        # stores ALL referenced images (including relative ones) under
        # ``images/<basename>``. The adapter has to look in the same place.
        self.image_url_template = os.getenv("MINERU_IMAGE_URL_TEMPLATE", "").strip()
        self.bbox_attributes = self._load_bbox_attributes_env()

    def _load_bbox_attributes_env(self) -> dict[str, Any]:
        default = {"origin": "LEFTTOP", "max": 1000}
        raw = os.getenv("MINERU_BBOX_ATTRIBUTES", "").strip()
        if not raw:
            return default
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[mineru_adapter] MINERU_BBOX_ATTRIBUTES is not valid JSON "
                "(%s); falling back to default %s",
                exc,
                default,
            )
            return default
        if not isinstance(parsed, dict):
            logger.warning(
                "[mineru_adapter] MINERU_BBOX_ATTRIBUTES must decode to a JSON "
                "object, got %s; falling back to default %s",
                type(parsed).__name__,
                default,
            )
            return default
        return parsed

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def normalize_from_workdir(
        self,
        raw_dir: Path,
        *,
        document_name: str,
    ) -> IRDoc:
        """Read ``raw_dir/content_list.json`` and emit an IRDoc.

        ``document_name`` is the canonical filename (e.g. ``foo.pdf``) used
        for ``meta.document_name``; resolved by the caller from the parser
        hint chain.
        """
        content_list_path = raw_dir / "content_list.json"
        if not content_list_path.is_file():
            raise FileNotFoundError(
                f"MinerU raw bundle missing content_list.json at {raw_dir}"
            )
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
        if not isinstance(content_list, list):
            raise ValueError(
                f"MinerU content_list.json malformed (not a JSON array) at {raw_dir}"
            )
        return self._normalize_content_list(
            content_list, raw_dir, document_name=document_name
        )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _normalize_content_list(
        self,
        content_list: list[Any],
        raw_dir: Path,
        *,
        document_name: str,
    ) -> IRDoc:
        document_format = Path(document_name).suffix.lower().lstrip(".")

        blocks: list[IRBlock] = []
        assets: list[AssetSpec] = []
        seen_assets: dict[str, str] = {}  # ref → suggested_name
        doc_title = ""
        placeholder_counter = 0

        def _next_key(prefix: str) -> str:
            nonlocal placeholder_counter
            placeholder_counter += 1
            return f"{prefix}{placeholder_counter}"

        # Heading hierarchy stack — index = level-1 (level 1 lives at [0]).
        heading_stack: list[str] = []

        # Current-block accumulator. The block is materialized when the next
        # heading arrives (or at end-of-document). The initial block is the
        # synthetic "Preface/Uncategorized" container at level 0.
        cb_lines: list[str] = []
        cb_tables: list[IRTable] = []
        cb_drawings: list[IRDrawing] = []
        cb_equations: list[IREquation] = []
        cb_positions: list[IRPosition] = []
        cb_heading = PREFACE_HEADING
        cb_level = 0
        cb_parents: list[str] = []
        # ``cb_has_body`` flips True the moment we accumulate any non-heading
        # payload into the current block. While it stays False, an adjacent
        # deeper heading is folded into this block as a body line (aligning
        # with the native docx parser's behaviour for back-to-back headings).
        cb_has_body = False

        def _flush_block() -> None:
            """Emit the in-flight block if it carries any content."""
            nonlocal cb_lines, cb_tables, cb_drawings, cb_equations, cb_positions
            nonlocal cb_has_body
            has_payload = bool(
                cb_lines or cb_tables or cb_drawings or cb_equations
            )
            if not has_payload:
                return
            content = "\n".join(line for line in cb_lines if line)
            if not content.strip() and not (
                cb_tables or cb_drawings or cb_equations
            ):
                # Reset and skip — nothing meaningful to emit.
                cb_lines = []
                cb_positions = []
                cb_has_body = False
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
            cb_has_body = False

        def _open_block(
            heading: str,
            level: int,
            parents: list[str],
            position: IRPosition | None,
        ) -> None:
            nonlocal cb_heading, cb_level, cb_parents
            cb_heading = heading
            cb_level = level
            cb_parents = parents
            # Render the heading line into the block body so the merged
            # text reads like markdown (``# Foo`` / ``## Bar`` / …).
            md_prefix = "#" * max(level, 1)
            cb_lines.append(f"{md_prefix} {heading}")
            if position is not None:
                cb_positions.append(position)

        def _append_text(text: str, position: IRPosition | None) -> None:
            nonlocal cb_has_body
            if text:
                cb_lines.append(text)
                cb_has_body = True
            if position is not None:
                cb_positions.append(position)

        def _merge_heading_as_body(
            heading: str, level: int, position: IRPosition | None
        ) -> None:
            """Fold an adjacent deeper heading into the current block.

            The line keeps its markdown ``#`` prefix so the rendered block
            still reads as ``# Section\n## Subsection``. Does NOT flip
            ``cb_has_body`` — successive headings can keep folding until a
            real body item lands.
            """
            md_prefix = "#" * max(level, 1)
            cb_lines.append(f"{md_prefix} {heading}")
            if position is not None:
                cb_positions.append(position)

        for item in content_list:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or item.get("label") or "").lower()
            position = _extract_position(item)

            heading_text, heading_level = _detect_heading(item, item_type)
            if heading_text:
                # Heading hierarchy is updated unconditionally so deeper
                # parents resolve correctly once the next real body item
                # opens a fresh block.
                heading_stack = heading_stack[: max(heading_level - 1, 0)]
                parents = [h for h in heading_stack if h]
                heading_stack.append(heading_text)

                # Adjacency merge: previous block is a real heading with no
                # body yet AND the new heading is strictly deeper — append
                # this heading as body to the existing block instead of
                # flushing. (Preface, level=0, is never merged into.)
                if (
                    cb_level > 0
                    and not cb_has_body
                    and heading_level > cb_level
                ):
                    _merge_heading_as_body(heading_text, heading_level, position)
                    if not doc_title and heading_level == 1:
                        doc_title = heading_text
                    continue

                _flush_block()
                _open_block(heading_text, heading_level, parents, position)

                if not doc_title and heading_level == 1:
                    doc_title = heading_text
                continue

            if item_type == "text":
                _append_text(_coerce_text(item), position)
                continue

            if item_type == "list":
                items = item.get("list_items")
                if isinstance(items, list):
                    text = "\n".join(str(x) for x in items if str(x).strip())
                else:
                    text = _coerce_text(item)
                _append_text(text, position)
                continue

            if item_type == "code":
                _append_text(item.get("code_body") or _coerce_text(item), position)
                continue

            if item_type == "equation":
                latex_raw = _coerce_text(item)
                if not latex_raw:
                    # Spec compliance fix: empty equation must not enter sidecar.
                    continue
                # Preserve MinerU's raw latex (including any ``$$``/``$``
                # wrappers); the writer strips them when emitting
                # equations.json so blocks.jsonl shows the raw form while
                # the per-equation sidecar holds clean latex.
                latex = latex_raw.strip()
                is_block = _is_block_equation(item)
                caption = str(item.get("caption") or "")
                placeholder = _next_key("eq")
                token = "EQ" if is_block else "EQI"
                cb_equations.append(
                    IREquation(
                        placeholder_key=placeholder,
                        latex=latex,
                        is_block=is_block,
                        caption=caption,
                        footnotes=_as_str_list(item.get("footnotes")),
                    )
                )
                cb_lines.append(f"{{{{{token}:{placeholder}}}}}")
                cb_has_body = True
                if position is not None:
                    cb_positions.append(position)
                continue

            if item_type == "table":
                table = self._build_ir_table(item)
                placeholder = _next_key("tb")
                table.placeholder_key = placeholder
                cb_tables.append(table)
                cb_lines.append(f"{{{{TBL:{placeholder}}}}}")
                cb_has_body = True
                if position is not None:
                    cb_positions.append(position)
                continue

            if item_type in {"image", "picture", "drawing"}:
                drawing, asset = self._build_ir_drawing(item, raw_dir, seen_assets)
                placeholder = _next_key("im")
                drawing.placeholder_key = placeholder
                if asset is not None and asset.ref not in {a.ref for a in assets}:
                    assets.append(asset)
                cb_drawings.append(drawing)
                cb_lines.append(f"{{{{IMG:{placeholder}}}}}")
                cb_has_body = True
                if position is not None:
                    cb_positions.append(position)
                continue

            # Fallback: serialize unknown items as plain text so we don't
            # silently drop information.
            _append_text(_coerce_text(item), position)

        _flush_block()

        if not doc_title:
            doc_title = Path(document_name).stem or document_name

        split_option: dict[str, Any] = {}
        if self.engine_version:
            split_option["engine_version"] = self.engine_version
        # Reserved hook for later: detect OCR flag from middle.json / config.

        return IRDoc(
            document_name=document_name,
            document_format=document_format,
            doc_title=doc_title,
            split_option=split_option,
            blocks=blocks,
            assets=assets,
            bbox_attributes=dict(self.bbox_attributes),
        )

    # ------------------------------------------------------------------
    # Tables / drawings
    # ------------------------------------------------------------------

    def _build_ir_table(self, item: dict) -> IRTable:
        rows: list[list[str]] | None = None
        html: str | None = None
        body_field = item.get("rows")
        body = body_field if body_field is not None else item.get("table_body")

        if isinstance(body, list):
            rows = _normalize_grid(body)
        elif isinstance(body, str):
            stripped = body.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    decoded = json.loads(stripped)
                    if isinstance(decoded, list):
                        rows = _normalize_grid(decoded)
                except json.JSONDecodeError:
                    pass
            if rows is None:
                html = stripped or None
        elif isinstance(body, dict):
            grid = body.get("grid") or body.get("rows")
            if isinstance(grid, list):
                rows = _normalize_grid(grid)
            else:
                html = json.dumps(body, ensure_ascii=False)

        num_rows = int(item.get("num_rows") or (len(rows) if rows else 0) or 0)
        num_cols_default = max((len(r) for r in rows), default=0) if rows else 0
        num_cols = int(item.get("num_cols") or num_cols_default or 0)

        captions = item.get("table_caption")
        caption = str(item.get("caption") or "")
        if not caption and isinstance(captions, list) and captions:
            caption = str(captions[0])

        table_header_raw = item.get("header")
        table_header: list[list[str]] | None = None
        if isinstance(table_header_raw, list) and table_header_raw:
            table_header = _normalize_grid(table_header_raw)

        return IRTable(
            placeholder_key="",  # filled by caller
            rows=rows,
            html=html,
            num_rows=num_rows,
            num_cols=num_cols,
            caption=caption,
            footnotes=_as_str_list(item.get("table_footnote") or item.get("footnotes")),
            table_header=table_header,
        )

    def _build_ir_drawing(
        self,
        item: dict,
        raw_dir: Path,
        seen: dict[str, str],
    ) -> tuple[IRDrawing, AssetSpec | None]:
        img_path = str(item.get("img_path") or item.get("path") or "")
        src_val = str(item.get("src") or "")
        captions = item.get("image_caption") or item.get("captions")
        caption = str(item.get("caption") or "")
        if not caption and isinstance(captions, list) and captions:
            caption = str(captions[0])

        fmt = Path(img_path).suffix.lower().lstrip(".") if img_path else ""
        if not fmt:
            fmt = str(item.get("format") or "")

        asset: AssetSpec | None = None
        ref = ""
        if img_path:
            ref = img_path
            if ref in seen:
                # Already declared by a previous block; reuse name.
                pass
            else:
                # Asset source: file on disk inside raw_dir. ``img_path`` is
                # untrusted (it comes from MinerU's content_list.json or a
                # downloaded zip), so we go through a safe resolver that
                # refuses to escape ``raw_dir`` and mirrors the downloader's
                # storage layout for absolute-URL / templated references.
                local_path = _safe_local_asset_path(
                    raw_dir,
                    img_path,
                    image_url_template=self.image_url_template,
                )
                suggested_name = _suggested_asset_name(img_path, fmt, len(seen))
                asset = AssetSpec(
                    ref=ref,
                    suggested_name=suggested_name,
                    source=local_path
                    if local_path is not None and local_path.is_file()
                    else None,
                )
                seen[ref] = suggested_name

        drawing = IRDrawing(
            placeholder_key="",  # filled by caller
            asset_ref=ref,
            fmt=fmt,
            caption=caption,
            footnotes=_as_str_list(item.get("image_footnote") or item.get("footnotes")),
            src=src_val,
        )
        return drawing, asset


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _detect_heading(item: dict, item_type: str) -> tuple[str, int]:
    """Return ``(heading_text, level)`` if ``item`` is a heading, else ``("", 0)``.

    A heading is either an explicit ``title``/``section_header`` block, or a
    ``text`` block whose ``text_level`` is positive (MinerU's convention).
    """
    if item_type in {"title", "section_header"}:
        text = _coerce_text(item).strip()
        level = max(int(item.get("text_level") or item.get("level") or 1), 1)
        return text, level
    if item_type == "text":
        try:
            tl = int(item.get("text_level") or 0)
        except (TypeError, ValueError):
            tl = 0
        if tl > 0:
            return _coerce_text(item).strip(), tl
    return "", 0


def _coerce_text(item: dict) -> str:
    for key in ("text", "content", "body", "code_body"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    s = str(value).strip()
    return [s] if s else []


def _normalize_grid(grid: Any) -> list[list[str]]:
    out: list[list[str]] = []
    if not isinstance(grid, list):
        return out
    for row in grid:
        if not isinstance(row, list):
            continue
        out_row: list[str] = []
        for cell in row:
            if isinstance(cell, dict):
                out_row.append(str(cell.get("text", "")).strip())
            else:
                out_row.append(str(cell).strip())
        out.append(out_row)
    return out


def _is_block_equation(item: dict) -> bool:
    """Heuristic: MinerU's ``text_format`` distinguishes block vs inline.

    Fallback when absent: treat as block (most MinerU equation items in
    PDF context represent display equations); inline equations are usually
    embedded inside ``text`` items rather than first-class ``equation``
    items.
    """
    fmt = str(item.get("text_format") or "").lower()
    if fmt in {"inline", "inline_equation"}:
        return False
    if fmt in {"block", "block_equation", "display"}:
        return True
    return True


def _extract_position(item: dict) -> IRPosition | None:
    """Build an ``IRPosition`` from MinerU page_idx + bbox if available.

    Returns ``None`` when no positional information is present, so blocks
    that legitimately lack a bbox don't get an empty position object.
    """
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        coords = [float(x) for x in bbox[:4]]
    except (TypeError, ValueError):
        return None

    page_raw = item.get("page_idx")
    if page_raw is None:
        page_raw = item.get("page")
    anchor: Any
    if isinstance(page_raw, int):
        anchor = page_raw + 1 if page_raw >= 0 else page_raw
    elif isinstance(page_raw, str) and page_raw.strip():
        anchor = page_raw
    else:
        anchor = None

    return IRPosition(type="bbox", anchor=anchor, range=coords)


def _safe_local_asset_path(
    raw_dir: Path,
    img_path: str,
    *,
    image_url_template: str = "",
) -> Path | None:
    """Resolve ``img_path`` to a concrete file location inside ``raw_dir``.

    ``img_path`` comes from MinerU's ``content_list.json`` and is therefore
    untrusted. This resolver mirrors :meth:`MinerURawClient._fetch_one_image`
    storage rules so the adapter always looks where the downloader wrote
    the file:

    - absolute http(s) URLs and absolute filesystem paths
      → ``raw_dir/images/<basename>``;
    - any ref when ``MINERU_IMAGE_URL_TEMPLATE`` is configured (the
      downloader routes ALL refs — including relative ones — through
      :meth:`_image_dest_rel`) → ``raw_dir/images/<basename>``;
    - otherwise relative paths resolve under ``raw_dir`` with ``..``
      traversal refused and a final ``Path.relative_to`` check.

    Returns ``None`` when the candidate is unsafe or cannot be expressed
    inside ``raw_dir``. The caller treats ``None`` the same as "file missing"
    — the drawing tag still gets written, but no bytes are copied.
    """
    if not img_path:
        return None

    if img_path.startswith(("http://", "https://")):
        name = Path(urlparse(img_path).path).name
        return raw_dir / "images" / name if name else None

    if os.path.isabs(img_path):
        # Absolute filesystem path in img_path is never trusted to point
        # outside raw_dir; mirror the downloader's basename rule.
        name = Path(img_path).name
        return raw_dir / "images" / name if name else None

    if image_url_template:
        # Templated mode: downloader stored every ref (incl. relative) at
        # images/<basename>, so we must look there too.
        name = Path(img_path).name
        return raw_dir / "images" / name if name else None

    normalized = os.path.normpath(img_path)
    if normalized.startswith("..") or os.path.isabs(normalized):
        return None
    candidate = (raw_dir / normalized).resolve()
    try:
        candidate.relative_to(raw_dir.resolve())
    except ValueError:
        return None
    return candidate


def _suggested_asset_name(img_path: str, fmt: str, seen_count: int) -> str:
    """Pick an in-assets-dir filename for an asset.

    For URL refs, use the URL path's basename so we get a useful filename
    (``foo.png`` rather than the whole URL). For local refs, the regular
    basename. Falls back to ``image-<n>[.fmt]`` when nothing usable.
    """
    if img_path.startswith(("http://", "https://")):
        name = Path(urlparse(img_path).path).name
    else:
        name = Path(img_path).name
    if name:
        return name
    return f"image-{seen_count + 1}{('.' + fmt) if fmt else ''}"


__all__ = ["MinerUAdapter"]
