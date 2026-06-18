"""Native Markdown IR builder: ``extract_markdown`` output → :class:`IRDoc`.

Input contract: the block dicts and side tables produced by
:func:`lightrag.parser.markdown.extract.extract_markdown`, threaded through
``NativeMarkdownParser.extract`` as ``(blocks, _, metadata)``. ``metadata``
carries the marker→payload side tables under the ``md_*`` keys.

Each block's ``content`` holds self-closing markers (``<mdtable ref=…/>`` /
``<mdequation ref=…/>`` / ``<mddrawing ref=…/>``). This builder rewrites them
into IR placeholder tokens (``{{TBL:k}}`` / ``{{EQ:k}}`` / ``{{IMG:k}}``) and
builds the matching :class:`IRTable` / :class:`IREquation` / :class:`IRDrawing`
from the side tables.

Image bytes are carried in ``metadata["md_assets"]`` (keyed by a stable
identity), so assets are declared as ``AssetSpec(source=bytes)`` and the writer
materializes + deduplicates them. External-link images carry no bytes and are
rendered verbatim through ``IRDrawing.path_override``.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lightrag.parser._html_table import (
    extract_html_table_info,
    extract_thead_html,
    html_table_inner_body,
    unwrap_html_table,
)
from lightrag.parser.markdown.extract import (
    DRAWING_MARKER_RE,
    EQUATION_MARKER_RE,
    PREFACE_HEADING,
    TABLE_MARKER_RE,
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


def _placeholder_keyspace() -> Callable[[str], str]:
    counter = itertools.count(1)
    return lambda prefix: f"{prefix}{next(counter)}"


class NativeMarkdownIRBuilder:
    """Translate ``extract_markdown`` output into an :class:`IRDoc`.

    Stateless — instantiate per call. ``asset_dir_name`` is the relative name
    of ``<base>.blocks.assets/``; the writer applies it as the on-disk prefix
    via ``block_drawing_path_style="with_prefix"``.
    """

    def normalize(
        self,
        blocks: list[dict[str, Any]],
        *,
        document_name: str,
        asset_dir_name: str,
        parse_metadata: dict[str, Any] | None = None,
    ) -> IRDoc:
        meta = parse_metadata or {}
        tables_meta: dict[str, dict] = meta.get("md_tables") or {}
        equations_meta: dict[str, str] = meta.get("md_equations") or {}
        drawings_meta: dict[str, dict] = meta.get("md_drawings") or {}
        assets_meta: dict[str, dict] = meta.get("md_assets") or {}

        next_key = _placeholder_keyspace()
        ir_blocks: list[IRBlock] = []
        assets: list[AssetSpec] = []
        seen_asset_refs: set[str] = set()
        doc_title = ""

        for block in blocks:
            content = block.get("content") or ""
            heading = block.get("heading") or ""
            level = int(block.get("level", 0) or 0)
            parent_headings = list(block.get("parent_headings") or [])

            tables: list[IRTable] = []
            equations: list[IREquation] = []
            drawings: list[IRDrawing] = []

            def _replace_table(match: "re.Match[str]") -> str:
                ref = match.group(1)
                spec = tables_meta.get(ref) or {}
                placeholder = next_key("tb")
                tables.append(_build_table(placeholder, spec))
                return f"{{{{TBL:{placeholder}}}}}"

            def _replace_equation(match: "re.Match[str]") -> str:
                ref = match.group(1)
                latex = equations_meta.get(ref, "")
                placeholder = next_key("eq")
                equations.append(
                    IREquation(placeholder_key=placeholder, latex=latex, is_block=True)
                )
                return f"{{{{EQ:{placeholder}}}}}"

            def _replace_drawing(match: "re.Match[str]") -> str:
                ref = match.group(1)
                spec = drawings_meta.get(ref) or {}
                placeholder = next_key("im")
                drawings.append(
                    _build_drawing(
                        placeholder, spec, assets_meta, assets, seen_asset_refs
                    )
                )
                return f"{{{{IMG:{placeholder}}}}}"

            content = TABLE_MARKER_RE.sub(_replace_table, content)
            content = EQUATION_MARKER_RE.sub(_replace_equation, content)
            content = DRAWING_MARKER_RE.sub(_replace_drawing, content)

            anchor = heading if heading and heading != PREFACE_HEADING else None
            positions = [IRPosition(type="heading", anchor=anchor)]

            ir_blocks.append(
                IRBlock(
                    content_template=content,
                    heading=heading,
                    level=level,
                    parent_headings=parent_headings,
                    positions=positions,
                    tables=tables,
                    drawings=drawings,
                    equations=equations,
                )
            )

            if not doc_title and level == 1 and heading and heading != PREFACE_HEADING:
                doc_title = heading

        if not doc_title:
            doc_title = Path(document_name).stem or document_name

        return IRDoc(
            document_name=document_name,
            document_format=Path(document_name).suffix.lower().lstrip("."),
            doc_title=doc_title,
            split_option={},
            blocks=ir_blocks,
            assets=assets,
            bbox_attributes=None,
        )


def _build_table(placeholder: str, spec: dict) -> IRTable:
    if spec.get("kind") == "html":
        raw = str(spec.get("html") or "")
        html = unwrap_html_table(raw) or None
        body_override = html_table_inner_body(html) or None if html else None
        info = extract_html_table_info(html or "")
        return IRTable(
            placeholder_key=placeholder,
            rows=None,
            html=html,
            num_rows=info.num_rows,
            num_cols=info.num_cols,
            table_header=extract_thead_html(html or ""),
            body_override=body_override,
        )
    # pipe table
    rows = [[str(c) for c in row] for row in (spec.get("rows") or [])]
    header = spec.get("header")
    num_rows = len(rows)
    num_cols = max((len(r) for r in rows), default=0)
    if header:
        num_cols = max(num_cols, max((len(h) for h in header), default=0))
    return IRTable(
        placeholder_key=placeholder,
        rows=rows,
        html=None,
        num_rows=num_rows,
        num_cols=num_cols,
        table_header=header if header else None,
    )


def _build_drawing(
    placeholder: str,
    spec: dict,
    assets_meta: dict[str, dict],
    assets: list[AssetSpec],
    seen_asset_refs: set[str],
) -> IRDrawing:
    fmt = str(spec.get("fmt") or "")
    src = str(spec.get("src") or "")
    if spec.get("kind") == "external":
        return IRDrawing(
            placeholder_key=placeholder,
            asset_ref="",
            fmt=fmt,
            src=src,
            path_override=str(spec.get("url") or src),
        )
    asset_ref = str(spec.get("asset_ref") or "")
    if asset_ref and asset_ref not in seen_asset_refs:
        asset = assets_meta.get(asset_ref) or {}
        assets.append(
            AssetSpec(
                ref=asset_ref,
                suggested_name=str(asset.get("suggested_name") or asset_ref),
                source=asset.get("data"),
            )
        )
        seen_asset_refs.add(asset_ref)
    return IRDrawing(
        placeholder_key=placeholder,
        asset_ref=asset_ref,
        fmt=fmt,
        src=src,
        path_override=None,
    )


__all__ = ["NativeMarkdownIRBuilder"]
