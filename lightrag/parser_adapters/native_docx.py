"""Native DOCX adapter: ``extract_docx_blocks`` output → :class:`IRDoc`.

Input contract: a list of block dicts as produced by
``lightrag.native_parser.docx.parse_document.extract_docx_blocks``. Each
block carries ``content`` text in which ``<table>``, ``<equation>`` and
``<drawing …/>`` placeholders are already embedded by the upstream parser.
The adapter rewrites those placeholders into IR placeholder tokens
(``{{TBL:k}} / {{EQ:k}} / {{EQI:k}} / {{IMG:k}}``) and builds the matching
``IRTable`` / ``IREquation`` / ``IRDrawing`` items.

Asset bytes are extracted to disk by the upstream parser *before* this
adapter runs (via ``DrawingExtractionContext`` passed to
``extract_docx_blocks``). The adapter therefore declares assets with
``AssetSpec.source=None`` — the writer records each entry's size without
copying.

Block-vs-inline equation distinction follows the legacy native rule: an
``<equation>…</equation>`` tag is *block* iff each side is either the
content boundary or a ``\\n`` character. Anything else stays inline,
keeps its tag in block text without an id, and never enters
``equations.json``.

Positions are always emitted as ``IRPosition(type="paraid", range=[start,
end])`` where each side may be ``None`` (legacy / non-Word docx authors
sometimes omit ``w14:paraId``). The writer's ``to_jsonable`` faithfully
preserves the per-side null so consumers can distinguish "start missing"
vs "both missing".
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lightrag.native_parser.docx.drawing_image_extractor import (
    DRAWING_TAG_PATTERN,
    parse_drawing_attributes,
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


_TABLE_TAG_RE = re.compile(r"<table>(.*?)</table>", re.DOTALL)
_EQUATION_TAG_RE = re.compile(r"<equation>(.*?)</equation>", re.DOTALL)


def _normalize_dimension(rows_value: Any) -> tuple[int, int]:
    if not isinstance(rows_value, list):
        return 0, 0
    num_rows = len(rows_value)
    num_cols = max((len(r) for r in rows_value if isinstance(r, list)), default=0)
    return num_rows, num_cols


def _placeholder_keyspace() -> Callable[[str], str]:
    """Return a fresh counter producing ``{prefix}{N}`` keys (1-indexed)."""
    counter = itertools.count(1)
    return lambda prefix: f"{prefix}{next(counter)}"


class NativeDocxAdapter:
    """Translate ``extract_docx_blocks`` output into an :class:`IRDoc`.

    The adapter is stateless — instantiate per call. ``asset_dir_name`` is
    the relative name (without trailing slash) of ``<base>.blocks.assets/``
    that the upstream parser used when emitting ``<drawing path>``
    attributes; the adapter strips that prefix when building
    :attr:`AssetSpec.ref` so the writer's ref↔filename mapping has
    predictable keys.
    """

    def normalize(
        self,
        blocks: list[dict[str, Any]],
        *,
        document_name: str,
        asset_dir_name: str,
        parse_metadata: dict[str, Any] | None = None,
    ) -> IRDoc:
        next_key = _placeholder_keyspace()
        ir_blocks: list[IRBlock] = []
        assets: list[AssetSpec] = []
        seen_asset_refs: set[str] = set()

        asset_prefix = f"{asset_dir_name}/" if asset_dir_name else ""

        for block in blocks:
            raw_content = block.get("content") or ""
            heading = block.get("heading") or ""
            level = int(block.get("level", 0) or 0)
            parent_headings = list(block.get("parent_headings") or [])
            # Preserve per-side nulls in [start, end].
            uuid_start = block.get("uuid") or None
            uuid_end = block.get("uuid_end") or None
            block_table_headers: list = list(block.get("table_headers") or [])

            tables: list[IRTable] = []
            drawings: list[IRDrawing] = []
            equations: list[IREquation] = []
            content_template = raw_content

            # --- Tables -------------------------------------------------
            table_position = 0

            def _replace_table(match: "re.Match[str]") -> str:
                nonlocal table_position
                table_body_raw = match.group(1)
                try:
                    rows = json.loads(table_body_raw)
                    if not isinstance(rows, list):
                        rows = None
                except json.JSONDecodeError:
                    rows = None

                if rows is not None:
                    parsed_rows: list[list[str]] | None = [
                        [str(c) for c in r] if isinstance(r, list) else [str(r)]
                        for r in rows
                    ]
                    html: str | None = None
                else:
                    parsed_rows = None
                    html = table_body_raw

                num_rows, num_cols = _normalize_dimension(parsed_rows)

                header_pos = table_position
                table_position += 1
                header_rows = (
                    block_table_headers[header_pos]
                    if header_pos < len(block_table_headers)
                    else None
                )
                # Treat empty list / explicit None identically: no header
                # entry on the sidecar item.
                table_header = header_rows if header_rows else None

                placeholder = next_key("tb")
                tables.append(
                    IRTable(
                        placeholder_key=placeholder,
                        rows=parsed_rows,
                        html=html,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        caption="",
                        footnotes=[],
                        table_header=table_header,
                        body_override=table_body_raw,
                    )
                )
                return f"{{{{TBL:{placeholder}}}}}"

            content_template = _TABLE_TAG_RE.sub(_replace_table, content_template)

            # --- Equations (block vs inline) ----------------------------
            def _replace_equation(match: "re.Match[str]") -> str:
                latex = match.group(1)
                source = match.string
                start, end = match.start(), match.end()
                is_block = (start == 0 or source[start - 1] == "\n") and (
                    end == len(source) or source[end] == "\n"
                )
                placeholder = next_key("eq")
                equations.append(
                    IREquation(
                        placeholder_key=placeholder,
                        latex=latex,
                        is_block=is_block,
                        caption="",
                        footnotes=[],
                    )
                )
                token = "EQ" if is_block else "EQI"
                return f"{{{{{token}:{placeholder}}}}}"

            content_template = _EQUATION_TAG_RE.sub(_replace_equation, content_template)

            # --- Drawings -----------------------------------------------
            def _replace_drawing(match: "re.Match[str]") -> str:
                attrs = parse_drawing_attributes(match.group(0))
                path_val = attrs.get("path", "") or ""
                src_val = attrs.get("src", "") or ""
                fmt = attrs.get("format", "") or ""
                if not fmt and path_val:
                    fmt = Path(path_val).suffix.lower().lstrip(".")

                # Two flavours of <drawing path="…">:
                #   1. Local asset under <base>.blocks.assets/ — already
                #      extracted to disk by DrawingExtractionContext;
                #      register as AssetSpec(source=None) and let the
                #      writer resolve the path via asset_paths.
                #   2. External/linked path (URL, or any path that does
                #      not live under asset_prefix) — pass through
                #      verbatim via IRDrawing.path_override; do NOT emit
                #      an AssetSpec (no on-disk bytes to materialize).
                if asset_prefix and path_val.startswith(asset_prefix):
                    rel_inside_assets = path_val[len(asset_prefix) :]
                    asset_ref = rel_inside_assets
                    suggested_name = Path(rel_inside_assets).name or rel_inside_assets
                    if asset_ref and asset_ref not in seen_asset_refs:
                        assets.append(
                            AssetSpec(
                                ref=asset_ref,
                                suggested_name=suggested_name,
                                source=None,  # already extracted to disk
                            )
                        )
                        seen_asset_refs.add(asset_ref)
                    path_override: str | None = None
                else:
                    asset_ref = ""
                    # Only mark as an external/linked reference when the
                    # upstream parser actually emitted a path. An empty
                    # ``path=""`` should fall back to the regular asset-
                    # resolution path (which will also produce ``path=""``
                    # downstream) rather than masquerading as an explicit
                    # adapter override.
                    path_override = path_val or None

                placeholder = next_key("im")
                drawings.append(
                    IRDrawing(
                        placeholder_key=placeholder,
                        asset_ref=asset_ref,
                        fmt=fmt,
                        caption="",
                        footnotes=[],
                        src=src_val,
                        path_override=path_override,
                    )
                )
                return f"{{{{IMG:{placeholder}}}}}"

            content_template = DRAWING_TAG_PATTERN.sub(
                _replace_drawing, content_template
            )

            # --- Positions: always paraid -------------------------------
            positions = [
                IRPosition(type="paraid", range=[uuid_start, uuid_end]),
            ]

            ir_blocks.append(
                IRBlock(
                    content_template=content_template,
                    heading=heading,
                    level=level,
                    parent_headings=parent_headings,
                    positions=positions,
                    tables=tables,
                    drawings=drawings,
                    equations=equations,
                )
            )

        # doc_title: parse_metadata["first_heading"] when present, else file
        # stem fallback (resolved here so the writer doesn't have to know).
        first_heading = ""
        if isinstance(parse_metadata, dict):
            first_heading = str(parse_metadata.get("first_heading") or "")
        doc_title = first_heading or (Path(document_name).stem or document_name)

        return IRDoc(
            document_name=document_name,
            document_format=Path(document_name).suffix.lower().lstrip("."),
            doc_title=doc_title,
            split_option={"fixlevel": 0},
            blocks=ir_blocks,
            assets=assets,
            bbox_attributes=None,
        )


__all__ = ["NativeDocxAdapter"]
