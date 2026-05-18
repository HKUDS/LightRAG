"""Docling adapter: ``DoclingDocument`` JSON → :class:`IRDoc`.

Input contract: a ``*.docling_raw/`` directory containing a ``<stem>.json``
produced by docling-serve with ``to_formats=[json,md]`` +
``image_export_mode=referenced``. Companion ``<stem>.md`` and
``artifacts/`` are not read by the adapter (markdown stays for human
inspection; image bytes are referenced by relative URI).

Conversion rules (informed by
``docs/DoclingSidecarRefactorPlan-zh.md`` §5):

- **Faithful** mapping. We do NOT correct heading levels from numbering,
  do NOT bind orphan ``caption`` / ``footnote`` text to neighbouring
  tables/pictures via proximity, do NOT merge continuation tables, do NOT
  invent captions or refer to inline neighbours. If docling didn't make
  the link, the sidecar doesn't make it either.
- ``content_layer != "body"`` is filtered everywhere (top-level traversal,
  group expansion, picture children). Furniture / background never leaks
  into blocks, positions, or consumed_refs.
- ``texts[*].label="title"`` → heading level 1; ``"section_header"`` →
  Docling ``level + 1`` (default 2 when level missing).
- ``texts[*].label="caption"|"footnote"`` are dropped from the reading
  stream **iff** their ref is referenced by a table/picture (via
  ``captions`` / ``footnotes`` refs, or as a direct ``children`` ref
  whose target is itself a caption/footnote). Otherwise they remain as
  regular text in the reading flow.
- ``pictures[*]`` without a usable image reference are skipped instead of
  emitting empty-path drawings. ``pictures[*].children`` references that
  are NOT caption/footnote are treated as inner-OCR text and excluded from
  the reading stream only for pictures that are emitted.
- ``IRPosition`` writes ``origin="LEFTTOP"`` only when the source
  ``prov.bbox.coord_origin == "TOPLEFT"``. ``BOTTOMLEFT`` inherits the
  doc-level meta (``{"origin":"LEFTBOTTOM"}`` by default). Coordinates
  are written verbatim — never flipped.
- ``DOCLING_BBOX_ATTRIBUTES`` env (JSON) can override the doc-level
  ``bbox_attributes``, mirroring MinerU's behaviour.
- Equations: ``texts[k].label == "formula"`` is treated as a structural
  formula signal whenever text/orig/content is non-empty. Top-level formulas
  become block equations; formulas inside inline groups become inline
  equations.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from lightrag.external_parser._common import env_json
from lightrag.external_parser.docling.manifest import select_main_json
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

# Docling JSON Pointer ``#/texts/3``, ``#/tables/2``, ``#/pictures/0``,
# ``#/groups/5``, or ``#/body``.
_REF_PATTERN = re.compile(r"^#/(?P<kind>[a-z_]+)(?:/(?P<index>\d+))?$")


class DoclingAdapter:
    """Stateless except for env-driven config. Reusable across calls."""

    def __init__(self) -> None:
        self.engine_version = os.getenv("DOCLING_ENGINE_VERSION", "").strip()
        self.bbox_attributes = self._load_bbox_attributes_env()

    @staticmethod
    def _load_bbox_attributes_env() -> dict[str, Any]:
        default = {"origin": "LEFTBOTTOM"}
        parsed = env_json("DOCLING_BBOX_ATTRIBUTES", default)
        if not isinstance(parsed, dict):
            logger.warning(
                "[docling_adapter] DOCLING_BBOX_ATTRIBUTES must decode to an object; "
                "falling back to %s",
                default,
            )
            return dict(default)
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
        main_json = select_main_json(raw_dir, Path(document_name))
        try:
            doc = json.loads(main_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Docling raw JSON malformed at {main_json}: {exc}"
            ) from exc
        if not isinstance(doc, dict):
            raise ValueError(f"Docling raw JSON is not an object at {main_json}")
        return self._normalize(doc, raw_dir, document_name=document_name)

    # ------------------------------------------------------------------
    # Core traversal
    # ------------------------------------------------------------------

    def _normalize(
        self,
        doc: dict,
        raw_dir: Path,
        *,
        document_name: str,
    ) -> IRDoc:
        document_format = Path(document_name).suffix.lower().lstrip(".")
        ref_index = _build_ref_index(doc)
        consumed_refs, picture_inner_refs = _precompute_consumed_refs(doc, raw_dir)

        blocks: list[IRBlock] = []
        assets: list[AssetSpec] = []
        seen_asset_refs: dict[str, str] = {}
        doc_title = ""
        placeholder_counter = 0

        def _next_key(prefix: str) -> str:
            nonlocal placeholder_counter
            placeholder_counter += 1
            return f"{prefix}{placeholder_counter}"

        # Heading stack + current block accumulator — identical structure
        # to MinerUAdapter so downstream P-chunking and provenance behave
        # the same way regardless of engine.
        heading_stack: list[str] = []
        cb_lines: list[str] = []
        cb_tables: list[IRTable] = []
        cb_drawings: list[IRDrawing] = []
        cb_equations: list[IREquation] = []
        cb_page_set: set[str] = set()
        cb_bbox_positions: list[IRPosition] = []
        cb_heading = PREFACE_HEADING
        cb_level = 0
        cb_parents: list[str] = []
        cb_has_body = False

        visited: set[str] = set()
        kv_count = len(doc.get("key_value_items") or [])
        form_count = len(doc.get("form_items") or [])

        # --- closures over the accumulator -----------------------------

        def _flush_block() -> None:
            nonlocal cb_lines, cb_tables, cb_drawings, cb_equations
            nonlocal cb_page_set, cb_bbox_positions, cb_has_body
            has_payload = bool(cb_lines or cb_tables or cb_drawings or cb_equations)
            if not has_payload:
                return
            content = "\n".join(line for line in cb_lines if line)
            if not content.strip() and not (cb_tables or cb_drawings or cb_equations):
                cb_lines = []
                cb_page_set = set()
                cb_bbox_positions = []
                cb_has_body = False
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
            cb_has_body = False

        def _open_block(heading: str, level: int, parents: list[str]) -> None:
            nonlocal cb_heading, cb_level, cb_parents
            cb_heading = heading
            cb_level = level
            cb_parents = parents
            md_prefix = "#" * max(level, 1)
            cb_lines.append(f"{md_prefix} {heading}")

        def _merge_heading_as_body(heading: str, level: int) -> None:
            md_prefix = "#" * max(level, 1)
            cb_lines.append(f"{md_prefix} {heading}")

        def _append_text(text: str) -> bool:
            nonlocal cb_has_body
            if not text:
                return False
            cb_lines.append(text)
            cb_has_body = True
            return True

        def _record_positions(item: dict) -> None:
            for prov in item.get("prov") or []:
                if not isinstance(prov, dict):
                    continue
                bbox = prov.get("bbox") or {}
                page_raw = prov.get("page_no")
                charspan = prov.get("charspan")
                if isinstance(bbox, dict) and all(
                    k in bbox for k in ("l", "t", "r", "b")
                ):
                    coord_origin = str(bbox.get("coord_origin") or "").upper()
                    origin_override: str | None = None
                    if coord_origin == "TOPLEFT":
                        origin_override = "LEFTTOP"
                    elif coord_origin == "BOTTOMLEFT":
                        origin_override = None
                    elif coord_origin:
                        logger.warning(
                            "[docling_adapter] unknown coord_origin %r; "
                            "writing through as override",
                            coord_origin,
                        )
                        origin_override = coord_origin
                    anchor = str(page_raw) if page_raw is not None else None
                    range_ = [
                        bbox["l"],
                        bbox["t"],
                        bbox["r"],
                        bbox["b"],
                    ]
                    cb_bbox_positions.append(
                        IRPosition(
                            type="bbox",
                            anchor=anchor,
                            range=range_,
                            charspan=(
                                list(charspan) if isinstance(charspan, list) else None
                            ),
                            origin=origin_override,
                        )
                    )
                elif page_raw is not None:
                    cb_page_set.add(str(page_raw))

        # --- main traversal -------------------------------------------

        def _visit_ref(ref: str) -> None:
            if not ref or ref in consumed_refs or ref in visited:
                return
            visited.add(ref)
            item = ref_index.get(ref)
            if item is None:
                return
            if _content_layer(item) != "body":
                return
            kind = _ref_kind(ref)

            if kind == "groups":
                _visit_group(item)
                return
            if kind == "texts":
                _handle_text(item)
                return
            if kind == "tables":
                _handle_table(item)
                return
            if kind == "pictures":
                _handle_picture(item)
                return
            # Unknown kind — log and ignore; falling through silently would
            # hide schema drift in future docling releases.
            logger.warning(
                "[docling_adapter] unknown ref kind %r (ref=%r); skipping", kind, ref
            )

        def _visit_group(group: dict) -> None:
            label = str(group.get("label") or "").lower()
            if label not in {
                "list",
                "inline",
                "picture_area",
                "section",
                "form_area",
                "key_value_area",
                "ordered_list",
                "unordered_list",
                "chapter",
            }:
                logger.warning(
                    "[docling_adapter] unrecognized group label %r; "
                    "expanding children as default reading order",
                    label,
                )
            if label == "inline":
                _handle_inline_group(group)
                return
            _visit_children(group)

        def _visit_children(item: dict) -> None:
            for child_ref in item.get("children") or []:
                ref = _ref_str(child_ref)
                _visit_ref(ref)

        def _handle_inline_group(group: dict) -> None:
            """``inline`` groups concatenate text and inline formulas on one line."""
            buf: list[str] = []
            pages_recorded = False
            for child_ref in group.get("children") or []:
                ref = _ref_str(child_ref)
                if ref in consumed_refs:
                    continue
                child = ref_index.get(ref)
                if not isinstance(child, dict):
                    continue
                if _content_layer(child) != "body":
                    continue
                if _ref_kind(ref) != "texts":
                    continue
                visited.add(ref)
                label = str(child.get("label") or "").lower()
                piece = (
                    _make_equation_placeholder(child, is_block=False)
                    if label == "formula"
                    else _text_of(child)
                )
                if piece:
                    buf.append(piece)
                    if not pages_recorded:
                        _record_positions(child)
                        pages_recorded = True
            line = " ".join(buf).strip()
            if line:
                _append_text(line)

        def _handle_text(item: dict) -> None:
            nonlocal doc_title, heading_stack, cb_has_body
            label = str(item.get("label") or "").lower()
            text = _text_of(item).strip()

            # Heading?
            heading_level = _docling_heading_level(label, item)
            if heading_level > 0 and text:
                heading_stack = heading_stack[: max(heading_level - 1, 0)]
                parents = [h for h in heading_stack if h]
                heading_stack.append(text)
                # Adjacency merge
                if cb_level > 0 and not cb_has_body and heading_level > cb_level:
                    _merge_heading_as_body(text, heading_level)
                    _record_positions(item)
                    if not doc_title and heading_level == 1:
                        doc_title = text
                    _visit_children(item)
                    return
                _flush_block()
                _open_block(text, heading_level, parents)
                _record_positions(item)
                if not doc_title and heading_level == 1:
                    doc_title = text
                _visit_children(item)
                return

            # Formula — Docling's label is the structural signal. For DOCX,
            # valid LaTeX may have text == orig, so do not use that equality
            # as an enrichment-off heuristic.
            if label == "formula":
                _handle_formula(item)
                _visit_children(item)
                return

            # list_item: keep the marker if Docling captured one
            if label == "list_item":
                marker = str(item.get("marker") or "").strip()
                line = f"{marker} {text}".strip() if marker else text
                if line and _append_text(line):
                    _record_positions(item)
                _visit_children(item)
                return

            # Caption/footnote not consumed by any table/picture → keep in
            # reading flow as ordinary text (preserves original prefixes).
            if label in {"caption", "footnote", "text", "code"}:
                if _append_text(text):
                    _record_positions(item)
                _visit_children(item)
                return

            # page_header / page_footer should have been filtered by
            # content_layer; reach here only if someone misuses the label.
            if label in {"page_header", "page_footer"}:
                return

            # Unknown label: fall back to writing the text and warn once.
            if text:
                logger.warning(
                    "[docling_adapter] unknown text label %r; treating as body",
                    label,
                )
                if _append_text(text):
                    _record_positions(item)
                _visit_children(item)

        def _handle_formula(item: dict) -> None:
            placeholder = _make_equation_placeholder(item, is_block=True)
            if not placeholder:
                return
            cb_lines.append(placeholder)
            _bump_has_body()
            _record_positions(item)

        def _make_equation_placeholder(item: dict, *, is_block: bool) -> str:
            latex_raw = _text_of(item).strip()
            if not latex_raw:
                return ""
            placeholder = _next_key("eq")
            token = "EQ" if is_block else "EQI"
            latex = f"$$ {latex_raw} $$" if is_block else latex_raw
            cb_equations.append(
                IREquation(
                    placeholder_key=placeholder,
                    latex=latex,
                    is_block=is_block,
                    self_ref=str(item.get("self_ref") or "") if is_block else "",
                )
            )
            return f"{{{{{token}:{placeholder}}}}}"

        def _bump_has_body() -> None:
            nonlocal cb_has_body
            cb_has_body = True

        def _handle_table(item: dict) -> None:
            table = _build_ir_table(item, ref_index)
            placeholder = _next_key("tb")
            table.placeholder_key = placeholder
            cb_tables.append(table)
            cb_lines.append(f"{{{{TBL:{placeholder}}}}}")
            _bump_has_body()
            _record_positions(item)

        def _handle_picture(item: dict) -> None:
            built = _build_ir_drawing(
                item,
                ref_index=ref_index,
                picture_inner_refs=picture_inner_refs,
                raw_dir=raw_dir,
                seen_asset_refs=seen_asset_refs,
            )
            if built is None:
                return
            drawing, asset = built
            placeholder = _next_key("im")
            drawing.placeholder_key = placeholder
            if asset is not None and asset.ref not in {a.ref for a in assets}:
                assets.append(asset)
            cb_drawings.append(drawing)
            cb_lines.append(f"{{{{IMG:{placeholder}}}}}")
            _bump_has_body()
            _record_positions(item)

        # Kick off traversal from body.children
        body = doc.get("body") or {}
        for child_ref in body.get("children") or []:
            _visit_ref(_ref_str(child_ref))

        _flush_block()

        if not doc_title:
            doc_title = Path(document_name).stem or document_name

        split_option: dict[str, Any] = {}
        if self.engine_version:
            split_option["engine_version"] = self.engine_version
        docling_extras: dict[str, Any] = {}
        if kv_count:
            docling_extras["key_value_items"] = kv_count
        if form_count:
            docling_extras["form_items"] = form_count
        if docling_extras:
            split_option["docling_extras"] = docling_extras

        return IRDoc(
            document_name=document_name,
            document_format=document_format,
            doc_title=doc_title,
            split_option=split_option,
            blocks=blocks,
            assets=assets,
            bbox_attributes=dict(self.bbox_attributes),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _ref_str(node: Any) -> str:
    """Normalize a Docling reference (``{"$ref": "#/texts/0"}`` or a bare
    string) to its string form. Returns ``""`` on garbage input."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        v = node.get("$ref") or node.get("ref")
        if isinstance(v, str):
            return v
    return ""


def _ref_kind(ref: str) -> str:
    m = _REF_PATTERN.match(ref)
    return m.group("kind") if m else ""


def _build_ref_index(doc: dict) -> dict[str, dict]:
    """Map every JSON-pointer-style ref to its target object.

    Builds entries for ``#/body``, ``#/texts/N``, ``#/tables/N``,
    ``#/pictures/N``, ``#/groups/N``. The body object is *not* a
    typical content item but we index it so callers don't need a
    special case when chasing arbitrary refs.
    """
    index: dict[str, dict] = {}
    body = doc.get("body")
    if isinstance(body, dict):
        index["#/body"] = body
    for key, prefix in (
        ("texts", "#/texts/"),
        ("tables", "#/tables/"),
        ("pictures", "#/pictures/"),
        ("groups", "#/groups/"),
    ):
        items = doc.get(key)
        if not isinstance(items, list):
            continue
        for i, obj in enumerate(items):
            if isinstance(obj, dict):
                index[f"{prefix}{i}"] = obj
    return index


def _precompute_consumed_refs(doc: dict, raw_dir: Path) -> tuple[set[str], set[str]]:
    """Return ``(consumed_refs, picture_inner_refs)``.

    ``consumed_refs`` enumerates text refs that must NOT enter the reading
    stream. The rules below apply only when the owning table/picture is
    itself in the body content layer — refs harvested from furniture or
    background items are ignored so they do not block legitimate body text
    that might be reachable through ``body.children``:

    - body ``tables[*].captions`` and ``tables[*].footnotes``
    - body ``pictures[*].captions`` and ``pictures[*].footnotes`` only when
      the picture has a usable image reference and will be emitted
    - body ``tables[*].children`` / ``pictures[*].children`` that resolve
      to ``texts[*]`` with ``label="caption"`` or ``"footnote"``
    - All body ``pictures[*].children`` that are non-caption/footnote texts
      (the picture's inner OCR text). These also land in
      ``picture_inner_refs`` so the adapter can attribute them to the
      drawing's extras.

    Sibling text nodes are NOT touched: only refs explicitly linked from a
    table/picture object qualify.
    """
    consumed: set[str] = set()
    picture_inner: set[str] = set()

    text_label_index: dict[str, str] = {}
    for i, obj in enumerate(doc.get("texts") or []):
        if isinstance(obj, dict):
            text_label_index[f"#/texts/{i}"] = str(obj.get("label") or "").lower()

    # Furniture/background tables/pictures must not consume refs that may
    # appear under body.children — the adapter contract is that non-body
    # items are filtered everywhere, including their outgoing refs.
    for table in doc.get("tables") or []:
        if not isinstance(table, dict):
            continue
        if _content_layer(table) != "body":
            continue
        for ref in _iter_refs(table.get("captions")):
            consumed.add(ref)
        for ref in _iter_refs(table.get("footnotes")):
            consumed.add(ref)
        for ref in _iter_refs(table.get("children")):
            label = text_label_index.get(ref)
            if label in {"caption", "footnote"}:
                consumed.add(ref)

    for pic in doc.get("pictures") or []:
        if not isinstance(pic, dict):
            continue
        if _content_layer(pic) != "body":
            continue
        if not _has_usable_picture_image(pic, raw_dir):
            continue
        for ref in _iter_refs(pic.get("captions")):
            consumed.add(ref)
        for ref in _iter_refs(pic.get("footnotes")):
            consumed.add(ref)
        for ref in _iter_refs(pic.get("children")):
            label = text_label_index.get(ref)
            if label in {"caption", "footnote"}:
                consumed.add(ref)
            elif ref.startswith("#/texts/"):
                consumed.add(ref)
                picture_inner.add(ref)

    return consumed, picture_inner


def _iter_refs(value: Any):
    """Yield refs from either a list of ref dicts/strings, or a single one."""
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            ref = _ref_str(item)
            if ref:
                yield ref
    else:
        ref = _ref_str(value)
        if ref:
            yield ref


def _content_layer(item: dict) -> str:
    return str(item.get("content_layer") or "body").lower()


def _text_of(item: dict) -> str:
    for key in ("text", "orig", "content"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _docling_heading_level(label: str, item: dict) -> int:
    """Map a Docling text item to its IR heading level.

    - ``title`` → level 1
    - ``section_header`` → ``item.level + 1`` (fallback 2)
    Returns 0 when the item is not a heading.
    """
    if label == "title":
        return 1
    if label == "section_header":
        raw = item.get("level")
        try:
            level = int(raw)
        except (TypeError, ValueError):
            level = 0
        if level <= 0:
            return 2
        return level + 1
    return 0


def _resolve_text_refs(refs: Any, ref_index: dict[str, dict]) -> list[str]:
    """Resolve a list of ``$ref`` entries to their text bodies.

    Skips targets whose ``content_layer`` is not ``"body"``. The adapter
    contract (see module docstring) is that furniture/background items
    never leak into sidecar metadata — even when a body table or picture
    explicitly references them, because such refs are typically the
    consequence of a page-header/footer being mislabeled as a caption.
    """
    out: list[str] = []
    for ref in _iter_refs(refs):
        target = ref_index.get(ref)
        if not isinstance(target, dict):
            continue
        if _content_layer(target) != "body":
            continue
        txt = _text_of(target).strip()
        if txt:
            out.append(txt)
    return out


def _build_ir_table(
    item: dict,
    ref_index: dict[str, dict],
) -> IRTable:
    data = item.get("data") or {}
    grid = data.get("grid") if isinstance(data, dict) else None
    rows = _rows_from_grid(grid)
    if not rows and isinstance(data, dict) and data.get("table_cells"):
        rows = _rows_from_table_cells(data)

    num_rows = (
        int(data.get("num_rows") or len(rows) or 0)
        if isinstance(data, dict)
        else len(rows)
    )
    num_cols = int(
        (data.get("num_cols") if isinstance(data, dict) else 0)
        or (max((len(r) for r in rows), default=0))
    )

    table_header = _extract_table_header(grid)

    captions = _resolve_text_refs(item.get("captions"), ref_index)
    if not captions:
        # Fallback: direct children with label="caption"
        captions = _resolve_children_with_label(
            item.get("children"), ref_index, "caption"
        )
    footnotes = _resolve_text_refs(item.get("footnotes"), ref_index)
    if not footnotes:
        footnotes = _resolve_children_with_label(
            item.get("children"), ref_index, "footnote"
        )

    extras: dict[str, Any] = {}
    if "parent" in item:
        extras["parent"] = item.get("parent")
    if item.get("children"):
        extras["children_refs"] = list(item.get("children") or [])
    if item.get("references"):
        extras["references"] = item.get("references")
    if item.get("annotations"):
        extras["annotations"] = item.get("annotations")
    cell_specs = _table_cell_specs(data)
    if cell_specs:
        extras["cells"] = cell_specs

    return IRTable(
        placeholder_key="",
        rows=rows or None,
        html=None,
        num_rows=num_rows,
        num_cols=num_cols,
        caption=" / ".join(captions),
        footnotes=footnotes,
        table_header=table_header,
        self_ref=str(item.get("self_ref") or ""),
        extras=extras,
    )


def _rows_from_grid(grid: Any) -> list[list[str]]:
    out: list[list[str]] = []
    if not isinstance(grid, list):
        return out
    for row in grid:
        if not isinstance(row, list):
            continue
        out.append(
            [str((c or {}).get("text", "") if isinstance(c, dict) else c) for c in row]
        )
    return out


def _rows_from_table_cells(data: dict) -> list[list[str]]:
    num_rows = int(data.get("num_rows") or 0)
    num_cols = int(data.get("num_cols") or 0)
    cells = data.get("table_cells") or []
    if num_rows <= 0 or num_cols <= 0 or not isinstance(cells, list):
        return []
    grid = [[""] * num_cols for _ in range(num_rows)]
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        text = str(cell.get("text") or "")
        rs = int(cell.get("start_row_offset_idx") or 0)
        re_ = int(cell.get("end_row_offset_idx") or rs + 1)
        cs = int(cell.get("start_col_offset_idx") or 0)
        ce_ = int(cell.get("end_col_offset_idx") or cs + 1)
        for r in range(max(rs, 0), min(re_, num_rows)):
            for c in range(max(cs, 0), min(ce_, num_cols)):
                grid[r][c] = text
    return grid


def _extract_table_header(grid: Any) -> list[list[str]] | None:
    """Return the contiguous top rows where every cell has
    ``column_header=True`` and ``start_row_offset_idx==0`` (the spec calls
    out both conditions to defeat false positives from spanning cells).
    """
    if not isinstance(grid, list):
        return None
    header_rows: list[list[str]] = []
    for row in grid:
        if not isinstance(row, list):
            break
        if (
            all(
                isinstance(c, dict)
                and bool(c.get("column_header"))
                and int(c.get("start_row_offset_idx") or 0) == 0
                for c in row
            )
            and row
        ):
            header_rows.append([str((c or {}).get("text", "")) for c in row])
        else:
            break
    return header_rows or None


def _table_cell_specs(data: dict) -> list[dict[str, Any]]:
    cells = data.get("table_cells") or [] if isinstance(data, dict) else []
    out: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        out.append(
            {
                k: cell.get(k)
                for k in (
                    "row_span",
                    "col_span",
                    "column_header",
                    "row_header",
                    "row_section",
                    "fillable",
                    "start_row_offset_idx",
                    "end_row_offset_idx",
                    "start_col_offset_idx",
                    "end_col_offset_idx",
                    "bbox",
                )
                if k in cell
            }
        )
    return out


def _resolve_children_with_label(
    children: Any, ref_index: dict[str, dict], expected_label: str
) -> list[str]:
    out: list[str] = []
    for ref in _iter_refs(children):
        target = ref_index.get(ref)
        if not isinstance(target, dict):
            continue
        # Same body-only filter as _resolve_text_refs; see its docstring.
        if _content_layer(target) != "body":
            continue
        if str(target.get("label") or "").lower() != expected_label:
            continue
        txt = _text_of(target).strip()
        if txt:
            out.append(txt)
    return out


def _build_ir_drawing(
    item: dict,
    *,
    ref_index: dict[str, dict],
    picture_inner_refs: set[str],
    raw_dir: Path,
    seen_asset_refs: dict[str, str],
) -> tuple[IRDrawing, AssetSpec | None] | None:
    image = item.get("image") or {}
    uri = ""
    mimetype = ""
    image_size: tuple[float, float] | None = None
    dpi: Any = None
    if isinstance(image, dict):
        uri = str(image.get("uri") or "")
        mimetype = str(image.get("mimetype") or "")
        size = image.get("size") or {}
        if isinstance(size, dict) and "width" in size and "height" in size:
            image_size = (float(size["width"]), float(size["height"]))
        dpi = image.get("dpi")

    fmt = _image_fmt_from_mimetype(mimetype) or (
        Path(uri).suffix.lstrip(".").lower() if uri else ""
    )

    captions = _resolve_text_refs(item.get("captions"), ref_index)
    if not captions:
        captions = _resolve_children_with_label(
            item.get("children"), ref_index, "caption"
        )
    footnotes = _resolve_text_refs(item.get("footnotes"), ref_index)
    if not footnotes:
        footnotes = _resolve_children_with_label(
            item.get("children"), ref_index, "footnote"
        )

    extras: dict[str, Any] = {}
    if image_size is not None:
        extras["intrinsic_size"] = list(image_size)
    if dpi is not None:
        extras["dpi"] = dpi
    if mimetype:
        extras["mimetype"] = mimetype
    if "parent" in item:
        extras["parent"] = item.get("parent")
    if item.get("children"):
        extras["children_refs"] = list(item.get("children") or [])
    inner_refs_for_this = [
        ref for ref in _iter_refs(item.get("children")) if ref in picture_inner_refs
    ]
    if inner_refs_for_this:
        extras["ocr_child_count"] = len(inner_refs_for_this)
    if item.get("annotations"):
        extras["annotations"] = item.get("annotations")
    if item.get("references"):
        extras["references"] = item.get("references")

    asset_ref = ""
    asset: AssetSpec | None = None
    path_override: str | None = None
    drawing_kwargs: dict[str, Any] = {}

    if not uri:
        return None
    if uri.startswith("data:"):
        decoded = _decode_data_uri(uri)
        if decoded is not None:
            payload, ext = decoded
            stem = (
                (item.get("self_ref") or "picture").replace("#/", "").replace("/", "_")
            )
            suggested = f"{stem}.{ext or fmt or 'bin'}"
            asset_ref = uri  # use the data URI as a stable ref
            if asset_ref not in seen_asset_refs:
                asset = AssetSpec(
                    ref=asset_ref,
                    suggested_name=suggested,
                    source=payload,
                )
                seen_asset_refs[asset_ref] = suggested
        else:
            logger.warning(
                "[docling_adapter] skipping picture %s because data URI could "
                "not be decoded",
                item.get("self_ref") or "<unknown>",
            )
            return None
    elif uri.startswith(("http://", "https://")):
        path_override = uri
        asset_ref = uri
    else:
        asset_ref = uri
        if asset_ref not in seen_asset_refs:
            # A malicious/corrupted bundle JSON could point at "../../etc/..."
            # or an absolute path; the zip extractor's traversal guard only
            # covers member names, not refs embedded in JSON metadata. Resolve
            # against raw_dir and require the result to stay inside.
            source_path = _resolve_local_image_path(raw_dir, uri)
            suggested = Path(uri).name or f"image_{len(seen_asset_refs):06d}"
            asset = AssetSpec(
                ref=asset_ref,
                suggested_name=suggested,
                source=source_path if source_path is not None else None,
            )
            if source_path is None:
                logger.warning(
                    "[docling_adapter] skipping picture %s because image URI "
                    "%r could not be resolved inside %s",
                    item.get("self_ref") or "<unknown>",
                    uri,
                    raw_dir,
                )
                return None
            seen_asset_refs[asset_ref] = suggested

    if path_override is not None:
        drawing_kwargs["path_override"] = path_override

    drawing = IRDrawing(
        placeholder_key="",
        asset_ref=asset_ref,
        fmt=fmt,
        caption=" / ".join(captions),
        footnotes=footnotes,
        src=str(item.get("src") or ""),
        self_ref=str(item.get("self_ref") or ""),
        extras=extras,
        **drawing_kwargs,
    )
    return drawing, asset


def _image_uri_of(item: dict) -> str:
    image = item.get("image")
    if not isinstance(image, dict):
        return ""
    return str(image.get("uri") or "")


def _has_usable_picture_image(item: dict, raw_dir: Path) -> bool:
    uri = _image_uri_of(item)
    if not uri:
        return False
    if uri.startswith("data:"):
        return _decode_data_uri(uri) is not None
    if uri.startswith(("http://", "https://")):
        return True
    return _resolve_local_image_path(raw_dir, uri) is not None


def _image_fmt_from_mimetype(mimetype: str) -> str:
    if not mimetype:
        return ""
    if mimetype == "image/jpeg":
        return "jpg"
    if mimetype.startswith("image/"):
        return mimetype[len("image/") :].lower()
    return ""


def _decode_data_uri(uri: str) -> tuple[bytes, str] | None:
    """Decode ``data:image/png;base64,...`` style URIs.

    Returns ``(bytes, extension)`` or ``None`` if the payload could not be
    decoded. Non-base64 payloads (extremely rare for images) are not
    supported and yield ``None``.
    """
    try:
        head, payload = uri.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in head:
        return None
    try:
        data = base64.b64decode(payload, validate=False)
    except (ValueError, TypeError):
        return None
    ext = ""
    if head.startswith("data:image/"):
        ext = head[len("data:image/") :].split(";", 1)[0].lower()
        if ext == "jpeg":
            ext = "jpg"
    return data, ext


def _resolve_local_image_path(raw_dir: Path, uri: str) -> Path | None:
    """Resolve a relative image URI against the bundle root and return it
    only if the result is a file *inside* ``raw_dir``.

    Returns ``None`` for: absolute URIs (``Path("foo") / "/etc/x"`` discards
    the left side and would escape), refs that resolve outside the bundle
    (``..``-traversal), and refs whose target does not exist. Symlinks are
    followed by ``resolve()`` and the post-resolution path is what's checked,
    so a symlink inside the bundle pointing outward is also refused.
    """
    if not uri or os.path.isabs(uri):
        return None
    try:
        base = raw_dir.resolve(strict=False)
        candidate = (raw_dir / uri).resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _sort_page_anchors(pages: set[str]) -> list[str]:
    non_numeric = sorted(p for p in pages if not p.isdigit())
    numeric = sorted((p for p in pages if p.isdigit()), key=int)
    return non_numeric + numeric


__all__ = ["DoclingAdapter"]
