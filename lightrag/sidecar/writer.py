"""Spec-compliant sidecar writer.

This module is the *single executable specification* of the LightRAG sidecar
format (``docs/LightRAGSidecarFormat-zh.md``). Engine adapters hand it an
:class:`IRDoc`; it emits the ``*.parsed/`` directory.

Responsibilities (none of these belong in adapters):

- id allocation: ``tb-/im-/eq-<doc_hash>-NNNN`` (4-digit zero-padded,
  global per-doc sequence)
- placeholder rendering: ``{{TBL:k}}`` / ``{{IMG:k}}`` / ``{{EQ:k}}`` /
  ``{{EQI:k}}`` → spec-shaped XML-style tags
- blockid computation: ``md5(doc_id:block_index:heading:content)``
- assets dir creation and file copying; ``asset_dir`` flag in meta is
  derived from "directory exists and is non-empty"
- merged_text + document_hash
- meta line shape (spec §3.1)
- conditional writes: ``tables.json`` / ``drawings.json`` / ``equations.json``
  appear only when their dict is non-empty
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.sidecar.ir import (
    AssetSpec,
    IRBlock,
    IRDoc,
    IRDrawing,
    IREquation,
    IRTable,
)
from lightrag.sidecar.placeholders import (
    render_drawing_tag,
    render_equation_tag,
    render_table_tag,
    render_template,
    table_body_for_rows,
)
from lightrag.utils import logger


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def write_sidecar(
    ir: IRDoc,
    *,
    parsed_dir: Path,
    doc_id: str,
    engine: str,
) -> dict[str, Any]:
    """Emit a spec-compliant ``*.parsed/`` directory from an IR.

    Args:
        ir: Document IR produced by an engine adapter.
        parsed_dir: Output directory. Cleared and recreated; caller is
            responsible for placing it under ``__parsed__/<base>.parsed/``.
        doc_id: ``doc-<md5>``; ``doc_hash`` for sidecar ids is the 32-char
            tail after stripping the ``doc-`` prefix.
        engine: One of ``native`` / ``mineru`` / ``docling`` / ``legacy``;
            written verbatim to ``meta.parse_engine``.

    Returns:
        Dict shaped like the pipeline's existing ``parsed_data`` payload:
        ``{doc_id, file_path, parse_format, content, blocks_path}``.
        ``file_path`` is ``ir.document_name``; the caller resolves it to the
        actual on-disk path it wants persisted.
    """
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(ir.document_name).stem or ir.document_name
    blocks_path = parsed_dir / f"{base_name}.blocks.jsonl"
    tables_path = parsed_dir / f"{base_name}.tables.json"
    drawings_path = parsed_dir / f"{base_name}.drawings.json"
    equations_path = parsed_dir / f"{base_name}.equations.json"
    assets_dir = parsed_dir / f"{base_name}.blocks.assets"

    # Stage 1: realize assets first so drawings can carry resolved paths.
    asset_paths = _materialize_assets(ir.assets, assets_dir)

    # Stage 2: walk blocks, allocate ids, render templates, accumulate
    # sidecar item dicts and blocks.jsonl lines.
    doc_hash = doc_id.removeprefix("doc-")
    tables: dict[str, dict[str, Any]] = {}
    drawings: dict[str, dict[str, Any]] = {}
    equations: dict[str, dict[str, Any]] = {}
    blocks_lines: list[str] = []
    merged_parts: list[str] = []

    table_seq = 0
    drawing_seq = 0
    equation_seq = 0
    block_index = 0

    asset_prefix = f"{assets_dir.name}/"

    for block in ir.blocks:
        # Allocate ids for items declared on this block. Order: tables ->
        # drawings -> equations (per-block deterministic; the global
        # sequence advances across blocks).
        table_id_by_key: dict[str, str] = {}
        for table in block.tables:
            table_seq += 1
            tb_id = f"tb-{doc_hash}-{table_seq:04d}"
            table_id_by_key[table.placeholder_key] = tb_id

        drawing_id_by_key: dict[str, str] = {}
        for drawing in block.drawings:
            drawing_seq += 1
            im_id = f"im-{doc_hash}-{drawing_seq:04d}"
            drawing_id_by_key[drawing.placeholder_key] = im_id

        equation_id_by_key: dict[str, str] = {}
        for equation in block.equations:
            if not equation.is_block:
                continue
            equation_seq += 1
            eq_id = f"eq-{doc_hash}-{equation_seq:04d}"
            equation_id_by_key[equation.placeholder_key] = eq_id

        # Render placeholder template.
        rendered = _render_block_content(
            block,
            table_id_by_key=table_id_by_key,
            drawing_id_by_key=drawing_id_by_key,
            equation_id_by_key=equation_id_by_key,
            asset_paths=asset_paths,
            asset_prefix=asset_prefix,
        )

        rendered = rendered.strip()
        if not rendered:
            # Drop empty blocks entirely — neither blocks.jsonl entry nor
            # sidecar items (the items were tied to the placeholder; if it
            # vanished, the items are orphans). This mirrors the existing
            # native_docx behaviour and ensures merged_text is contiguous.
            continue

        blockid = hashlib.md5(
            f"{doc_id}:{block_index}:{block.heading}:{rendered}".encode("utf-8")
        ).hexdigest()

        # Realize per-block sidecar item dicts now that blockid is known.
        for table in block.tables:
            tb_id = table_id_by_key[table.placeholder_key]
            tables[tb_id] = _table_item_dict(tb_id, blockid, block.heading, table)
        for drawing in block.drawings:
            im_id = drawing_id_by_key[drawing.placeholder_key]
            drawings[im_id] = _drawing_item_dict(
                im_id, blockid, block.heading, drawing, asset_paths, asset_prefix
            )
        for equation in block.equations:
            if not equation.is_block:
                continue
            eq_id = equation_id_by_key[equation.placeholder_key]
            equations[eq_id] = _equation_item_dict(
                eq_id, blockid, block.heading, equation
            )

        row: dict[str, Any] = {
            "type": "content",
            "blockid": blockid,
            "format": "plain_text",
            "content": rendered,
            "heading": block.heading,
            "parent_headings": list(block.parent_headings),
            "level": int(block.level),
            "session_type": block.session_type or "body",
            "table_slice": block.table_slice or "none",
            "positions": [p.to_jsonable() for p in block.positions],
        }
        if block.table_header:
            row["table_header"] = block.table_header
        blocks_lines.append(json.dumps(row, ensure_ascii=False))
        merged_parts.append(rendered)
        block_index += 1

    # Stage 3: doc-level metadata.
    merged_text = "\n\n".join(p for p in merged_parts if p.strip())
    document_hash = hashlib.sha256(merged_text.encode("utf-8")).hexdigest()
    parse_time = datetime.now(timezone.utc).isoformat()

    asset_dir_present = assets_dir.exists() and any(assets_dir.iterdir())
    if not asset_dir_present and assets_dir.exists():
        try:
            assets_dir.rmdir()
        except OSError:
            pass

    meta: dict[str, Any] = {
        "type": "meta",
        "format": "lightrag",
        "version": "1.0",
        "document_name": ir.document_name,
        "document_format": ir.document_format,
        "document_hash": f"sha256:{document_hash}",
        "table_file": bool(tables),
        "equation_file": bool(equations),
        "drawing_file": bool(drawings),
        "asset_dir": asset_dir_present,
        "split_option": dict(ir.split_option or {}),
        "blocks": len(blocks_lines),
        "doc_id": doc_id,
        "parse_engine": engine,
        "parse_time": parse_time,
        "doc_title": ir.doc_title,
    }
    if ir.bbox_attributes is not None:
        meta["bbox_attributes"] = dict(ir.bbox_attributes)

    blocks_path.write_text(
        "\n".join([json.dumps(meta, ensure_ascii=False)] + blocks_lines) + "\n",
        encoding="utf-8",
    )

    if tables:
        tables_path.write_text(
            json.dumps(
                {"version": "1.0", "tables": tables},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if drawings:
        drawings_path.write_text(
            json.dumps(
                {"version": "1.0", "drawings": drawings},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if equations:
        equations_path.write_text(
            json.dumps(
                {"version": "1.0", "equations": equations},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    logger.info(
        "[sidecar] wrote %d blocks for doc_id=%s "
        "(%d tables, %d drawings, %d equations, assets=%s, engine=%s)",
        len(blocks_lines),
        doc_id,
        len(tables),
        len(drawings),
        len(equations),
        asset_dir_present,
        engine,
    )

    return {
        "doc_id": doc_id,
        "file_path": ir.document_name,
        "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
        "content": merged_text,
        "blocks_path": str(blocks_path),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _materialize_assets(
    assets: list[AssetSpec],
    assets_dir: Path,
) -> dict[str, str]:
    """Materialize :class:`AssetSpec` objects into ``assets_dir``.

    Returns: ``{ref: filename_inside_assets_dir}``.

    Collision policy: if two specs map to the same target name, the second
    gets a ``-2``, ``-3``, ... suffix on the stem. We never overwrite a file
    we've already produced.
    """
    if not assets:
        return {}

    assets_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    used_names: set[str] = set()

    for spec in assets:
        target_name = _allocate_unique_name(spec.suggested_name, used_names)
        target_path = assets_dir / target_name
        if isinstance(spec.source, (str, Path)):
            src_path = Path(spec.source)
            if not src_path.exists():
                logger.warning(
                    "[sidecar] asset source missing for ref=%s (%s); "
                    "skipping copy",
                    spec.ref,
                    src_path,
                )
                continue
            if src_path.resolve() != target_path.resolve():
                shutil.copyfile(src_path, target_path)
        elif isinstance(spec.source, bytes):
            target_path.write_bytes(spec.source)
        elif spec.source is None:
            # Assumed already on disk at the target location (native_docx
            # writes assets during extraction). Verify presence; warn if
            # missing.
            if not target_path.exists():
                logger.warning(
                    "[sidecar] asset ref=%s declared in place but %s "
                    "is absent",
                    spec.ref,
                    target_path,
                )
                continue
        else:
            logger.warning(
                "[sidecar] unsupported AssetSpec.source type for ref=%s: %s",
                spec.ref,
                type(spec.source).__name__,
            )
            continue
        used_names.add(target_name)
        out[spec.ref] = target_name

    return out


def _allocate_unique_name(suggested: str, used: set[str]) -> str:
    """Make ``suggested`` unique within ``used``: ``foo.png`` → ``foo-2.png``."""
    if suggested not in used:
        return suggested
    stem = Path(suggested).stem
    suffix = Path(suggested).suffix
    n = 2
    while True:
        cand = f"{stem}-{n}{suffix}"
        if cand not in used:
            return cand
        n += 1


def _render_block_content(
    block: IRBlock,
    *,
    table_id_by_key: dict[str, str],
    drawing_id_by_key: dict[str, str],
    equation_id_by_key: dict[str, str],
    asset_paths: dict[str, str],
    asset_prefix: str,
) -> str:
    """Expand placeholder tokens in ``block.content_template``."""

    tables_by_key = {t.placeholder_key: t for t in block.tables}
    drawings_by_key = {d.placeholder_key: d for d in block.drawings}
    equations_by_key = {e.placeholder_key: e for e in block.equations}

    def _table(key: str) -> str:
        table = tables_by_key.get(key)
        if table is None:
            return ""
        tb_id = table_id_by_key.get(key, "")
        if table.rows is not None:
            return render_table_tag(tb_id, "json", table_body_for_rows(table.rows))
        return render_table_tag(tb_id, "html", table.html or "")

    def _drawing(key: str) -> str:
        drawing = drawings_by_key.get(key)
        if drawing is None:
            return ""
        im_id = drawing_id_by_key.get(key, "")
        filename = asset_paths.get(drawing.asset_ref, "")
        path = f"{asset_prefix}{filename}" if filename else ""
        return render_drawing_tag(
            im_id,
            drawing.fmt,
            drawing.caption,
            path,
            drawing.src,
        )

    def _equation(key: str) -> str:
        eq = equations_by_key.get(key)
        if eq is None:
            return ""
        if not eq.is_block:
            # Adapter mistake: an EQ token should only be used for block
            # equations. Treat as inline to avoid a dangling token.
            return render_equation_tag(None, eq.latex, eq.caption)
        eq_id = equation_id_by_key.get(key, "")
        return render_equation_tag(eq_id, eq.latex, eq.caption)

    def _inline_equation(key: str) -> str:
        eq = equations_by_key.get(key)
        if eq is None:
            return ""
        return render_equation_tag(None, eq.latex, eq.caption)

    return render_template(
        block.content_template,
        table_renderer=_table,
        drawing_renderer=_drawing,
        equation_renderer=_equation,
        inline_equation_renderer=_inline_equation,
    )


def _table_item_dict(
    table_id: str,
    blockid: str,
    heading: str,
    table: IRTable,
) -> dict[str, Any]:
    if table.rows is not None:
        fmt = "json"
        content = table_body_for_rows(table.rows)
    else:
        fmt = "html"
        content = table.html or ""

    item: dict[str, Any] = {
        "id": table_id,
        "blockid": blockid,
        "heading": heading,
        "dimension": [int(table.num_rows), int(table.num_cols)],
        "format": fmt,
        "content": content,
        "caption": table.caption,
        "footnotes": list(table.footnotes),
    }
    if table.table_header is not None:
        # Spec §5: stored as JSON string.
        item["table_header"] = json.dumps(table.table_header, ensure_ascii=False)
    if table.extras:
        item["extras"] = dict(table.extras)
    return item


def _drawing_item_dict(
    drawing_id: str,
    blockid: str,
    heading: str,
    drawing: IRDrawing,
    asset_paths: dict[str, str],
    asset_prefix: str,
) -> dict[str, Any]:
    filename = asset_paths.get(drawing.asset_ref, "")
    path = f"{asset_prefix}{filename}" if filename else ""
    item: dict[str, Any] = {
        "id": drawing_id,
        "blockid": blockid,
        "heading": heading,
        "format": drawing.fmt,
        "path": path,
        "src": drawing.src,
        "caption": drawing.caption,
        "footnotes": list(drawing.footnotes),
    }
    if drawing.extras:
        item["extras"] = dict(drawing.extras)
    return item


def _equation_item_dict(
    eq_id: str,
    blockid: str,
    heading: str,
    equation: IREquation,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": eq_id,
        "blockid": blockid,
        "heading": heading,
        "format": "latex",
        "content": equation.latex,
        "caption": equation.caption,
        "footnotes": list(equation.footnotes),
    }
    if equation.extras:
        item["extras"] = dict(equation.extras)
    return item
