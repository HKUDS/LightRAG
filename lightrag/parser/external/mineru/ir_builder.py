"""MinerU IR builder: ``content_list.json`` (+ images/) → :class:`IRDoc`.

Input contract: a ``*.mineru_raw/`` directory containing at least
``content_list.json``. Optional sibling resources (``images/``,
``middle.json``, ``full.md``, ``layout.pdf``) are kept as-is; this builder
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
  behavior (see ``parser/docx/parse_document.py``).
- Content emitted before the first heading lands in a synthetic
  ``Preface/Uncategorized`` block at level 0.
- ``list`` items joined with ``\n``; ``code`` body taken from ``code_body``
  if present.
- ``table`` → IRTable + ``{{TBL:k}}`` placeholder. MinerU HTML tables are
  preserved verbatim on ``IRTable.html`` so merged cells (``rowspan`` /
  ``colspan``) survive in ``tables.json``; the block placeholder receives
  only the table's inner HTML to avoid nested ``<table>`` wrappers. ``rows``
  is reserved for explicit 2D-array / non-HTML compatibility inputs. A real
  HTML ``<thead>`` populates ``table_header`` (per spec §5); otherwise the
  adapter does not guess a header row.
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

from dataclasses import dataclass
from html.parser import HTMLParser
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lightrag.parser._markdown import (
    render_heading_line,
    strip_heading_markdown_prefix,
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
CONTENT_LIST_FILENAME = "content_list.json"


class MinerUIRBuilder:
    """Stateless except for env-driven config. Reusable across calls."""

    def __init__(self) -> None:
        self.engine_version = os.getenv("MINERU_ENGINE_VERSION", "").strip()
        # Mirror MinerURawClient.__init__: when this is set, the downloader
        # stores ALL referenced images (including relative ones) under
        # ``images/<basename>``. The builder has to look in the same place.
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
                "[mineru_ir_builder] MINERU_BBOX_ATTRIBUTES is not valid JSON "
                "(%s); falling back to default %s",
                exc,
                default,
            )
            return default
        if not isinstance(parsed, dict):
            logger.warning(
                "[mineru_ir_builder] MINERU_BBOX_ATTRIBUTES must decode to a JSON "
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
        # Positions are split into two channels:
        # - ``cb_page_set`` collects ``page_idx`` of bbox-less items; at flush
        #   each unique page becomes one anchor-only summary ``IRPosition``.
        # - ``cb_bbox_positions`` keeps one fine-grained position per item that
        #   carried a parseable bbox (anchor + range), in source order, with
        #   no deduplication.
        cb_page_set: set[str] = set()
        cb_bbox_positions: list[IRPosition] = []
        cb_heading = PREFACE_HEADING
        cb_level = 0
        cb_parents: list[str] = []

        def _record_position(item: dict) -> None:
            """Route an item's positional info into the right channel.

            Items with a parseable ``bbox`` produce one fine-grained
            IRPosition appended to ``cb_bbox_positions`` (no dedupe).
            Otherwise, ``page_idx`` (if any) is added to ``cb_page_set``
            and emitted as a single anchor-only summary entry at flush.
            """
            bbox_pos = _extract_bbox_position(item)
            if bbox_pos is not None:
                cb_bbox_positions.append(bbox_pos)
                return
            page = _extract_page_anchor(item)
            if page is not None:
                cb_page_set.add(page)

        def _flush_block() -> None:
            """Emit the in-flight block if it carries any content."""
            nonlocal cb_lines, cb_tables, cb_drawings, cb_equations
            nonlocal cb_page_set, cb_bbox_positions
            has_payload = bool(cb_lines or cb_tables or cb_drawings or cb_equations)
            if not has_payload:
                return
            content = "\n".join(line for line in cb_lines if line)
            if not content.strip() and not (cb_tables or cb_drawings or cb_equations):
                # Reset and skip — nothing meaningful to emit.
                cb_lines = []
                cb_page_set = set()
                cb_bbox_positions = []
                return
            positions = [
                IRPosition(type="bbox", anchor=p)
                for p in _sort_page_anchors(cb_page_set)
            ] + list(cb_bbox_positions)
            blocks.append(
                IRBlock(
                    content_template=content,
                    heading=cb_heading,
                    level=cb_level,
                    parent_headings=list(cb_parents),
                    positions=positions,
                    tables=list(cb_tables),
                    drawings=list(cb_drawings),
                    equations=list(cb_equations),
                )
            )
            cb_lines = []
            cb_tables = []
            cb_drawings = []
            cb_equations = []
            cb_page_set = set()
            cb_bbox_positions = []

        def _open_block(
            heading: str, level: int, parents: list[str], raw_heading: str | None = None
        ) -> None:
            nonlocal cb_heading, cb_level, cb_parents
            cb_heading = heading
            cb_level = level
            cb_parents = parents
            # Render the heading line into the block body so the merged
            # text reads like markdown (``# Foo`` / ``## Bar`` / …). Levels
            # are capped at 6 ``#`` and headings already carrying a markdown
            # prefix are left untouched (see ``render_heading_line``).
            cb_lines.append(render_heading_line(level, raw_heading or heading))

        def _append_text(text: str) -> bool:
            """Append ``text`` to the current block body and return whether
            anything was actually written. Callers use the return value to
            decide whether to also record the item's source position — an
            empty text item must NOT leak its ``page_idx`` to the block.
            """
            if not text:
                return False
            cb_lines.append(text)
            return True

        for item_index, item in enumerate(content_list):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or item.get("label") or "").lower()

            heading_text, heading_level = _detect_heading(item, item_type)
            if heading_text:
                clean_heading = strip_heading_markdown_prefix(heading_text)
                # Heading hierarchy is updated unconditionally so deeper
                # parents resolve correctly once the next real body item
                # opens a fresh block.
                heading_stack = heading_stack[: max(heading_level - 1, 0)]
                parents = [h for h in heading_stack if h]
                heading_stack.append(clean_heading)

                # Every recognized heading starts its own block: flush the
                # in-flight block (whether it had body or was a bare heading)
                # and open a fresh one. A heading with no following body thus
                # becomes a standalone block whose content is just the heading
                # line, matching the native docx parser's behaviour.
                _flush_block()
                _open_block(clean_heading, heading_level, parents, heading_text)
                _record_position(item)

                if not doc_title and heading_level == 1:
                    doc_title = clean_heading
                continue

            if item_type == "text":
                if _append_text(_coerce_text(item)):
                    _record_position(item)
                continue

            if item_type == "list":
                items = item.get("list_items")
                if isinstance(items, list):
                    text = "\n".join(str(x) for x in items if str(x).strip())
                else:
                    text = _coerce_text(item)
                if _append_text(text):
                    _record_position(item)
                continue

            if item_type == "code":
                if _append_text(item.get("code_body") or _coerce_text(item)):
                    _record_position(item)
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
                        self_ref=_content_list_self_ref(item_index) if is_block else "",
                    )
                )
                cb_lines.append(f"{{{{{token}:{placeholder}}}}}")
                _record_position(item)
                continue

            if item_type == "table":
                table = self._build_ir_table(item)
                if table is None:
                    # Empty body — _build_ir_table already logged the drop.
                    # Skip placeholder allocation and position recording so
                    # the misidentified item leaves no trace in the IR.
                    continue
                placeholder = _next_key("tb")
                table.placeholder_key = placeholder
                table.self_ref = _content_list_self_ref(item_index)
                cb_tables.append(table)
                cb_lines.append(f"{{{{TBL:{placeholder}}}}}")
                _record_position(item)
                continue

            if item_type in {"image", "picture", "drawing"}:
                drawing, asset = self._build_ir_drawing(item, raw_dir, seen_assets)
                placeholder = _next_key("im")
                drawing.placeholder_key = placeholder
                drawing.self_ref = _content_list_self_ref(item_index)
                if asset is not None and asset.ref not in {a.ref for a in assets}:
                    assets.append(asset)
                cb_drawings.append(drawing)
                cb_lines.append(f"{{{{IMG:{placeholder}}}}}")
                _record_position(item)
                continue

            # Fallback: serialize unknown items as plain text so we don't
            # silently drop information. Position only recorded when the
            # fallback actually contributed text — empty unknown items must
            # not leak their page_idx into the current block.
            if _append_text(_coerce_text(item)):
                _record_position(item)

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

    def _build_ir_table(self, item: dict) -> IRTable | None:
        rows: list[list[str]] | None = None
        html: str | None = None
        body_override: str | None = None
        body_field = item.get("rows")
        body = body_field if body_field is not None else item.get("table_body")

        if isinstance(body, list):
            rows = _normalize_grid(body)
        elif isinstance(body, str):
            stripped = body.strip()
            if _looks_like_html_table_payload(stripped):
                # MinerU's table model sometimes wraps output in a
                # ``<html><body>…</body></html>`` document; unwrap to the bare
                # ``<table>…</table>`` so the sidecar ``content`` stays a single
                # clean table and the writer does not nest ``<table>`` wrappers.
                html = _unwrap_html_table(stripped) or None
                if html:
                    # ``or None`` so a degenerate ``<table></table>`` (empty
                    # inner body) falls back to rendering ``table.html`` in the
                    # writer instead of emitting an empty ``body_override``.
                    body_override = _html_table_inner_body(html) or None
            elif stripped.startswith("[") and stripped.endswith("]"):
                try:
                    decoded = json.loads(stripped)
                    if isinstance(decoded, list):
                        rows = _normalize_grid(decoded)
                except json.JSONDecodeError:
                    pass
            if rows is None and html is None:
                # Non-HTML, non-JSON string (or JSON that failed to parse):
                # fall back to the raw payload as the html body.
                html = stripped or None
        elif isinstance(body, dict):
            grid = body.get("grid") or body.get("rows")
            if isinstance(grid, list):
                rows = _normalize_grid(grid)
            else:
                html = json.dumps(body, ensure_ascii=False)

        # MinerU occasionally emits table items with no usable body (e.g. when
        # a page number or blank region is misidentified as a table). Dropping
        # them here keeps the sidecar free of items that would later trip the
        # analyze worker's "missing table content" hard-failure path.
        if not _ir_table_body_has_content(rows, html):
            logger.debug(
                "[mineru_ir_builder] dropping empty table item "
                "(body type=%s, num_rows=%s, num_cols=%s)",
                type(body).__name__,
                item.get("num_rows"),
                item.get("num_cols"),
            )
            return None

        num_rows = int(item.get("num_rows") or (len(rows) if rows else 0) or 0)
        num_cols_default = max((len(r) for r in rows), default=0) if rows else 0
        num_cols = int(item.get("num_cols") or num_cols_default or 0)
        html_table_info: _HTMLTableInfo | None = None
        if html and (num_rows <= 0 or num_cols <= 0):
            html_table_info = _extract_html_table_info(html)
            if num_rows <= 0:
                num_rows = html_table_info.num_rows
            if num_cols <= 0:
                num_cols = html_table_info.num_cols

        captions = item.get("table_caption")
        caption = str(item.get("caption") or "")
        if not caption and isinstance(captions, list) and captions:
            caption = str(captions[0])

        # The header representation follows the table's format so merged-cell
        # semantics survive: HTML tables keep the raw ``<thead>…</thead>``
        # (preserving rowspan/colspan); grid/JSON tables keep a 2-D grid.
        table_header_raw = item.get("header")
        table_header: list[list[str]] | str | None = None
        if html:
            table_header = _extract_thead_html(html)
        elif isinstance(table_header_raw, list) and table_header_raw:
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
            body_override=body_override,
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


@dataclass
class _HTMLTableInfo:
    num_rows: int = 0
    num_cols: int = 0


class _HTMLTableInfoParser(HTMLParser):
    """Count ``<tr>`` rows and their (colspan-aware) column widths.

    Used only to recover ``num_rows`` / ``num_cols`` when MinerU did not supply
    them; the ``<thead>`` header itself is preserved verbatim by
    :func:`_extract_thead_html`, not reconstructed here.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # ``col_count`` (sum of colspans) for each completed top-level ``<tr>``.
        self.row_col_counts: list[int] = []
        self._tr_depth = 0
        self._cell_depth = 0
        self._row_col_count = 0
        self._cell_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            if self._tr_depth == 0:
                self._row_col_count = 0
            self._tr_depth += 1
            return
        if tag in {"td", "th"} and self._tr_depth > 0:
            if self._cell_depth == 0:
                self._cell_colspan = _cell_span(attrs, "colspan")
            self._cell_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_depth > 0:
            self._cell_depth -= 1
            if self._cell_depth == 0:
                self._row_col_count += self._cell_colspan
                self._cell_colspan = 1
            return
        if tag == "tr" and self._tr_depth > 0:
            self._tr_depth -= 1
            # ``col_count > 0`` ⇔ the row held at least one cell (colspan ≥ 1),
            # so an empty ``<tr></tr>`` is skipped exactly as before.
            if self._tr_depth == 0 and self._row_col_count > 0:
                self.row_col_counts.append(self._row_col_count)
                self._row_col_count = 0


def _cell_span(attrs: list[tuple[str, str | None]], name: str) -> int:
    """Read a ``colspan``/``rowspan`` attribute as an int ``>= 1`` (default 1)."""
    for key, value in attrs:
        if key.lower() != name:
            continue
        try:
            return max(int(value or "1"), 1)
        except ValueError:
            return 1
    return 1


def _extract_html_table_info(html: str) -> _HTMLTableInfo:
    parser = _HTMLTableInfoParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception as exc:  # pragma: no cover - HTMLParser is forgiving.
        logger.debug("[mineru_ir_builder] failed to parse table HTML: %s", exc)
        return _HTMLTableInfo()
    return _HTMLTableInfo(
        num_rows=len(parser.row_col_counts),
        num_cols=max(parser.row_col_counts, default=0),
    )


def _extract_thead_html(html: str) -> str | None:
    """Return the first top-level ``<thead …>…</thead>`` substring verbatim.

    The raw markup is kept so merged-cell semantics (``rowspan`` / ``colspan``)
    survive into ``tables.json`` and, later, into every repeated header chunk
    of a split table. Returns ``None`` when the table has no ``<thead>`` or the
    ``<thead>`` carries no visible text (a blank spacer row, which would
    otherwise emit empty ``<th>`` headers).
    """
    stripped = (html or "").strip()
    lower = stripped.lower()
    start = _find_html_tag(lower, "thead")
    if start < 0:
        return None
    close = lower.find("</thead>", start)
    if close < 0:
        return None
    thead = stripped[start : close + len("</thead>")]
    # Blank check: drop a header whose cells hold no non-whitespace text.
    if not re.sub(r"<[^>]+>", "", thead).strip():
        return None
    return thead


def _looks_like_html_table_payload(body: str) -> bool:
    lower = (body or "").lstrip().lower()
    return any(
        _starts_with_html_tag(lower, tag)
        for tag in ("table", "thead", "tbody", "tfoot", "tr", "html", "body")
    )


def _unwrap_html_table(payload: str) -> str:
    """Strip a ``<html>/<body>`` document wrapper that MinerU's table model
    sometimes emits, returning the outermost ``<table…>…</table>`` span. Keeps
    a single clean ``<table>`` so the writer does not nest tables and the
    non-greedy ``TABLE_TAG_RE`` is not truncated at an inner ``</table>``.
    Falls back to the stripped payload when no ``<table>`` element exists."""
    stripped = (payload or "").strip()
    lower = stripped.lower()
    start = _find_table_open(lower)
    if start < 0:
        return stripped
    close = lower.rfind("</table>")
    if close < start:
        return stripped
    return stripped[start : close + len("</table>")]


def _find_table_open(lower: str) -> int:
    """First index of a real ``<table`` start tag (not e.g. ``<tablefoo``).
    Returns -1 when none is present."""
    return _find_html_tag(lower, "table")


def _find_html_tag(lower: str, tag: str) -> int:
    """First index of a real ``<tag`` start tag (not e.g. ``<tablefoo`` for
    ``tag="table"``). ``lower`` must already be lower-cased. Returns -1 when
    none is present."""
    needle = f"<{tag}"
    idx = 0
    while True:
        idx = lower.find(needle, idx)
        if idx < 0:
            return -1
        nxt = idx + len(needle)
        if nxt >= len(lower) or lower[nxt] in {" ", "\t", "\r", "\n", ">", "/"}:
            return idx
        idx = nxt


def _starts_with_html_tag(lower: str, tag: str) -> bool:
    prefix = f"<{tag}"
    if not lower.startswith(prefix):
        return False
    if len(lower) == len(prefix):
        return True
    return lower[len(prefix)] in {" ", "\t", "\r", "\n", ">", "/"}


def _html_table_inner_body(html: str) -> str:
    stripped = (html or "").strip()
    lower = stripped.lower()
    if not _starts_with_html_tag(lower, "table"):
        return stripped
    open_end = _open_tag_end(stripped)
    close_start = lower.rfind("</table>")
    if open_end < 0 or close_start <= open_end:
        return stripped
    return stripped[open_end + 1 : close_start].strip()


def _open_tag_end(html: str) -> int:
    """Index of the ``>`` closing the leading tag, skipping quoted attribute
    values so a ``>`` inside an attribute (e.g. ``<table data-x="a>b">``) does
    not terminate the tag early. Returns -1 when no closing ``>`` is found."""
    quote: str | None = None
    for idx, ch in enumerate(html):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in {'"', "'"}:
            quote = ch
        elif ch == ">":
            return idx
    return -1


def _content_list_self_ref(index: int) -> str:
    return f"{CONTENT_LIST_FILENAME}#/{index}"


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


def _ir_table_body_has_content(rows: list[list[str]] | None, html: str | None) -> bool:
    """True iff the parsed table body carries any visible cell text or HTML."""
    if html and html.strip():
        return True
    if rows:
        for row in rows:
            for cell in row:
                if isinstance(cell, str) and cell.strip():
                    return True
    return False


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


def _extract_page_anchor(item: dict) -> str | None:
    """Return a 1-based page anchor from MinerU's ``page_idx`` / ``page``.

    Always returns a string so ``blocks.jsonl`` carries a uniform anchor
    type across Roman / letter / numeric page labels. Integers are bumped
    to 1-based (``page_idx=0`` → ``"1"``); strings are stripped and passed
    through verbatim. Returns ``None`` when no usable page info is present.
    """
    page_raw = item.get("page_idx")
    if page_raw is None:
        page_raw = item.get("page")
    if isinstance(page_raw, bool):
        # bool is a subclass of int — guard so True/False don't sneak in.
        return None
    if isinstance(page_raw, int):
        return str(page_raw + 1 if page_raw >= 0 else page_raw)
    if isinstance(page_raw, str) and page_raw.strip():
        return page_raw.strip()
    return None


def _sort_page_anchors(pages: set[str]) -> list[str]:
    """Order page anchors using book pagination convention.

    Non-numeric labels (Roman preface pages ``i``/``ii``/``iv``…, letter
    pages like ``A``, ``B-1``) come first in lexical order; numeric labels
    follow, sorted by their integer value so ``"2"`` precedes ``"10"``.
    Mixing both kinds is safe — the bucketed key avoids the ``TypeError``
    that ``sorted({"ii", "1"})`` raises when ints and strings mix.
    """
    non_numeric = sorted(p for p in pages if not p.isdigit())
    numeric = sorted((p for p in pages if p.isdigit()), key=int)
    return non_numeric + numeric


def _extract_bbox_position(item: dict) -> IRPosition | None:
    """Build a fine-grained ``IRPosition`` when ``bbox`` is parseable.

    Returns ``None`` when ``bbox`` is missing or malformed; the caller then
    falls back to page-only tracking via :func:`_extract_page_anchor`.
    """
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        coords = [float(x) for x in bbox[:4]]
    except (TypeError, ValueError):
        return None
    return IRPosition(type="bbox", anchor=_extract_page_anchor(item), range=coords)


def _safe_local_asset_path(
    raw_dir: Path,
    img_path: str,
    *,
    image_url_template: str = "",
) -> Path | None:
    """Resolve ``img_path`` to a concrete file location inside ``raw_dir``.

    ``img_path`` comes from MinerU's ``content_list.json`` and is therefore
    untrusted. This resolver mirrors :meth:`MinerURawClient._fetch_one_image`
    storage rules so the builder always looks where the downloader wrote
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


__all__ = ["MinerUIRBuilder"]
