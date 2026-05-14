"""LightRAG Document writer for the native DOCX parser.

The adapter calls :func:`extract_docx_blocks` with ``fixlevel=0`` and writes
the LightRAG Document artifacts directly to disk:

- ``<base>.blocks.jsonl``           — main file (meta line + content lines)
- ``<base>.tables.json``            — table sidecar (only when non-empty)
- ``<base>.equations.json``         — equation sidecar (block equations only)
- ``<base>.drawings.json``          — drawing sidecar (only when non-empty)
- ``<base>.blocks.assets/``         — exported image bytes (only when non-empty)

Inline ``<table>{json}</table>``, ``<equation>{latex}</equation>`` and
``<drawing .../>`` placeholders produced by the upstream extractor are rewritten
in-place to carry stable LightRAG ids and captions, while sidecar entries hold
the full structured payload required by downstream multimodal stages.

Equations are split into two classes by surrounding whitespace: a tag wedged
between two newlines (or at the content boundary on a side that would otherwise
require ``\n``) is a *block* equation and lands in ``equations.json``; any other
``<equation>`` is *inline* and is left untouched in the block text without a
sidecar entry, so multimodal stages skip it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG, PARSER_ENGINE_NATIVE
from lightrag.utils import logger
from lightrag.utils_pipeline import (
    canonicalize_parser_hinted_basename,
    parsed_artifact_dir_for_source,
)

from .drawing_image_extractor import (
    DRAWING_TAG_PATTERN,
    DrawingExtractionContext,
    _load_relationships,
    parse_drawing_attributes,
)
from .parse_document import extract_docx_blocks


_TABLE_TAG_RE = re.compile(r"<table>(.*?)</table>", re.DOTALL)
_EQUATION_TAG_RE = re.compile(r"<equation>(.*?)</equation>", re.DOTALL)


def _xml_attr_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _caption_attr(caption: str) -> str:
    """Render a leading-space ``caption="..."`` attribute, empty when absent."""
    return f' caption="{_xml_attr_escape(caption)}"' if caption else ""


def _normalize_dimension(rows_value: Any) -> tuple[int, int]:
    if not isinstance(rows_value, list):
        return 0, 0
    num_rows = len(rows_value)
    num_cols = max((len(r) for r in rows_value if isinstance(r, list)), default=0)
    return num_rows, num_cols


async def parse_docx_to_lightrag_document(
    file_bytes: bytes,
    file_path: str,
    doc_id: str,
    source_path: str,
    output_dir: Path | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Extract a DOCX and write LightRAG Document artifacts to disk.

    Args:
        file_bytes: Raw DOCX bytes already loaded by the caller.
        file_path: Document path as known to the pipeline (may carry a
            ``[parser-hint]`` segment that needs canonicalization).
        doc_id: LightRAG document id used for stable sidecar ids and blockid.
        source_path: Concrete on-disk path to the source DOCX. Used to compute
            the parsed artifact directory via
            :func:`parsed_artifact_dir_for_source` when ``output_dir`` is None.
        output_dir: Optional override for the parsed-artifact directory. When
            provided, the writer drops ``.blocks.jsonl`` and sidecars under
            this directory instead of the production ``__parsed__/`` location.
            Used by the debugging CLI.
        debug: When True, the upstream ``extract_docx_blocks`` emits split
            traces to stderr (CLI debugging only).

    Returns:
        ``{doc_id, file_path, parse_format, content, blocks_path}`` — the same
        shape ``parse_native`` previously consumed from
        :func:`_write_lightrag_document_from_content_list`. When the document
        triggered any non-fatal parse warnings, a ``parse_warnings`` dict is
        also included (e.g. ``{"tables_without_paraid": 3}``); the pipeline
        forwards this into ``doc_status.metadata.parse_warnings``.
    """
    return await asyncio.to_thread(
        _parse_docx_sync,
        file_bytes,
        file_path,
        doc_id,
        source_path,
        output_dir,
        debug,
    )


def _parse_docx_sync(
    file_bytes: bytes,
    file_path: str,
    doc_id: str,
    source_path: str,
    output_dir: Path | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    parsed_dir = (
        Path(output_dir)
        if output_dir is not None
        else parsed_artifact_dir_for_source(source_path, file_path)
    )
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    canonical_basename = (
        canonicalize_parser_hinted_basename(file_path)
        or Path(source_path).name
        or f"{doc_id}.bin"
    )
    base_name = Path(canonical_basename).stem or canonical_basename

    blocks_path = parsed_dir / f"{base_name}.blocks.jsonl"
    tables_path = parsed_dir / f"{base_name}.tables.json"
    equations_path = parsed_dir / f"{base_name}.equations.json"
    drawings_path = parsed_dir / f"{base_name}.drawings.json"
    asset_dir = parsed_dir / f"{base_name}.blocks.assets"

    # extract_docx_blocks() and DrawingExtractionContext both work against a
    # filesystem path (the latter opens the docx as a zip), so persist the
    # caller-provided bytes to a temp file and clean up at the end.
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        temp_docx = Path(tmp.name)

    parse_warnings: dict[str, Any] = {}
    parse_metadata: dict[str, Any] = {}

    try:
        asset_dir.mkdir(parents=True, exist_ok=True)
        ctx = DrawingExtractionContext(
            docx_path=temp_docx,
            blocks_output_path=blocks_path,
            export_dir_name=asset_dir.name,
            export_dir_path=asset_dir,
        )
        _load_relationships(ctx)

        blocks = extract_docx_blocks(
            str(temp_docx),
            debug=debug,
            fixlevel=0,
            drawing_context=ctx,
            parse_warnings=parse_warnings,
            parse_metadata=parse_metadata,
        )
    finally:
        try:
            temp_docx.unlink()
        except OSError:
            pass

    if not blocks:
        raise ValueError(f"DOCX parser returned empty content for {file_path}")

    missing_paraid_count = int(parse_warnings.get("missing_paraid_count", 0) or 0)
    if missing_paraid_count > 0:
        # Surface once per document — the parser may encounter many missing
        # paraIds (legacy / non-Word docx authors omit ``w14:paraId``), but a
        # single warning with the count is enough for the user. Affected
        # blocks emit ``positions: [{"type": "paraid", "range": null}]``.
        logger.warning(
            "[native_parser/docx] %s: %d paragraphs lack paraId; "
            "Re-saving file in Word 2013+ to regenerate ids.",
            Path(file_path).name,
            missing_paraid_count,
        )

    tables: dict[str, Any] = {}
    equations: dict[str, Any] = {}
    drawings: dict[str, Any] = {}
    table_idx = 0
    equation_idx = 0
    drawing_idx = 0

    blocks_lines: list[str] = []
    merged_parts: list[str] = []

    for block_idx, block in enumerate(blocks):
        raw_content = block.get("content") or ""
        heading = block.get("heading") or ""
        level = int(block.get("level", 0) or 0)
        parent_headings = list(block.get("parent_headings") or [])
        # Keep ``None`` (rather than collapsing to a peer / empty string) so
        # downstream consumers can distinguish "start missing", "end missing"
        # and "both missing" via the per-side null in ``range``.
        uuid_start = block.get("uuid") or None
        uuid_end = block.get("uuid_end") or None

        # Sidecar entries for this block are accumulated locally so a block
        # that strips to empty does not leak orphan ids into the global maps.
        pending_tables: dict[str, dict[str, Any]] = {}
        pending_equations: dict[str, dict[str, Any]] = {}
        pending_drawings: dict[str, dict[str, Any]] = {}
        local_table_count = 0
        local_equation_count = 0
        local_drawing_count = 0
        # Cross-page repeating header rows for each <table> in this block, in
        # placeholder order. Each entry is either a list of header rows or
        # None when the table has no repeating header.
        block_table_headers: list = list(block.get("table_headers") or [])

        def _replace_table(match: re.Match) -> str:
            nonlocal local_table_count
            local_table_count += 1
            idx = table_idx + local_table_count
            tb_id = f"tb-{doc_id.removeprefix('doc-')}-{idx:04d}"
            caption = ""
            table_json = match.group(1)
            try:
                rows = json.loads(table_json)
            except Exception:
                rows = []
            num_rows, num_cols = _normalize_dimension(rows)
            content = json.dumps(rows, ensure_ascii=False) if rows else table_json
            header_pos = local_table_count - 1
            header_rows = (
                block_table_headers[header_pos]
                if header_pos < len(block_table_headers)
                else None
            )
            sidecar_entry: dict[str, Any] = {
                "id": tb_id,
                "blockid": "",
                "heading": heading,
                "dimension": [num_rows, num_cols],
                "format": "json",
                "content": content,
                "caption": caption,
                "footnotes": [],
            }
            if header_rows:
                # Sidecar format stores the header as a JSON string (see
                # docs/LightRAGSidecarFormat-zh.md §5).
                sidecar_entry["table_header"] = json.dumps(
                    header_rows, ensure_ascii=False
                )
            pending_tables[tb_id] = sidecar_entry
            return (
                f'<table id="{_xml_attr_escape(tb_id)}" format="json"'
                f"{_caption_attr(caption)}>{table_json}</table>"
            )

        def _replace_equation(match: re.Match) -> str:
            nonlocal local_equation_count
            latex = match.group(1)

            # Block equation = tag wedged between newlines (or content edge).
            # Inline equations stay verbatim in the block text and are not
            # promoted to sidecar entries.
            source = match.string
            start, end = match.start(), match.end()
            is_block = (start == 0 or source[start - 1] == "\n") and (
                end == len(source) or source[end] == "\n"
            )
            if not is_block:
                return f'<equation format="latex">{latex}</equation>'

            local_equation_count += 1
            idx = equation_idx + local_equation_count
            eq_id = f"eq-{doc_id.removeprefix('doc-')}-{idx:04d}"
            caption = ""
            pending_equations[eq_id] = {
                "id": eq_id,
                "blockid": "",
                "heading": heading,
                "format": "latex",
                "content": latex,
                "caption": caption,
                "footnotes": [],
            }
            return (
                f'<equation id="{_xml_attr_escape(eq_id)}" format="latex"'
                f"{_caption_attr(caption)}>{latex}</equation>"
            )

        def _replace_drawing(match: re.Match) -> str:
            nonlocal local_drawing_count
            local_drawing_count += 1
            idx = drawing_idx + local_drawing_count
            placeholder = match.group(0)
            attrs = parse_drawing_attributes(placeholder)
            dr_id = f"im-{doc_id.removeprefix('doc-')}-{idx:04d}"
            caption = ""
            path_val = attrs.get("path", "") or ""
            src_val = attrs.get("src", "") or ""
            fmt = attrs.get("format", "") or ""
            if not fmt and path_val:
                fmt = Path(path_val).suffix.lower().lstrip(".")
            pending_drawings[dr_id] = {
                "id": dr_id,
                "blockid": "",
                "heading": heading,
                "format": fmt,
                "path": path_val,
                "src": src_val,
                "caption": caption,
                "footnotes": [],
            }
            # Strip the "<base>.blocks.assets/" prefix from the path embedded in
            # the rewritten <drawing /> tag for blocks.jsonl. The sidecar entry
            # keeps the parsed_dir-relative path used by VLM image loading.
            block_path_val = path_val
            asset_prefix = f"{asset_dir.name}/"
            if block_path_val.startswith(asset_prefix):
                block_path_val = block_path_val[len(asset_prefix) :]
            return (
                f'<drawing id="{_xml_attr_escape(dr_id)}" '
                f'format="{_xml_attr_escape(fmt)}"'
                f"{_caption_attr(caption)} "
                f'path="{_xml_attr_escape(block_path_val)}" '
                f'src="{_xml_attr_escape(src_val)}" />'
            )

        rewritten = _TABLE_TAG_RE.sub(_replace_table, raw_content)
        rewritten = _EQUATION_TAG_RE.sub(_replace_equation, rewritten)
        rewritten = DRAWING_TAG_PATTERN.sub(_replace_drawing, rewritten)

        content_text = rewritten.strip()
        if not content_text:
            continue

        blockid = hashlib.md5(
            f"{doc_id}:{block_idx}:{heading}:{content_text}".encode("utf-8")
        ).hexdigest()

        for entry in pending_tables.values():
            entry["blockid"] = blockid
        for entry in pending_equations.values():
            entry["blockid"] = blockid
        for entry in pending_drawings.values():
            entry["blockid"] = blockid

        tables.update(pending_tables)
        equations.update(pending_equations)
        drawings.update(pending_drawings)
        table_idx += local_table_count
        equation_idx += local_equation_count
        drawing_idx += local_drawing_count

        # Always emit a paraid position entry for docx blocks. ``range`` is a
        # ``[start, end]`` pair where each side is the source paragraph /
        # table cell paraId, or ``null`` when the source lacked
        # ``w14:paraId``. Per-side nulls let consumers distinguish
        # "start missing" / "end missing" / "both missing" without relying
        # on an outer null.
        positions: list[dict[str, Any]] = [
            {
                "type": "paraid",
                "range": [uuid_start, uuid_end],
            }
        ]

        blocks_lines.append(
            json.dumps(
                {
                    "type": "content",
                    "blockid": blockid,
                    "format": "plain_text",
                    "content": content_text,
                    "heading": heading,
                    "parent_headings": parent_headings,
                    "level": level,
                    "session_type": "body",
                    "table_slice": "none",
                    "positions": positions,
                },
                ensure_ascii=False,
            )
        )
        merged_parts.append(content_text)

    merged_text = "\n\n".join(p for p in merged_parts if p.strip())
    doc_hash = hashlib.sha256(merged_text.encode("utf-8")).hexdigest()
    parse_time = datetime.now(timezone.utc).isoformat()

    asset_dir_present = bool(asset_dir.exists() and any(asset_dir.iterdir()))

    # doc_title prefers the document's first heading (captured by
    # extract_docx_blocks regardless of level); fall back to the file stem
    # when the document has no headings at all.
    first_heading = parse_metadata.get("first_heading") or ""
    doc_title = first_heading or (Path(canonical_basename).stem or canonical_basename)

    meta = {
        "type": "meta",
        "format": "lightrag",
        "version": "1.0",
        "document_name": canonical_basename,
        "document_format": Path(canonical_basename).suffix.lower().lstrip("."),
        "document_hash": f"sha256:{doc_hash}",
        "table_file": bool(tables),
        "equation_file": bool(equations),
        "drawing_file": bool(drawings),
        "asset_dir": asset_dir_present,
        "split_option": {"fixlevel": 0},
        "blocks": len(blocks_lines),
        "doc_id": doc_id,
        "parse_engine": PARSER_ENGINE_NATIVE,
        "parse_time": parse_time,
        "doc_title": doc_title,
    }

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
    if equations:
        equations_path.write_text(
            json.dumps(
                {"version": "1.0", "equations": equations},
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

    if not asset_dir_present and asset_dir.exists():
        try:
            asset_dir.rmdir()
        except OSError:
            pass

    logger.info(
        "[native_parser/docx] parsed %d blocks for doc_id=%s "
        "(%d tables, %d equations, %d drawings, assets=%s)",
        len(blocks_lines),
        doc_id,
        len(tables),
        len(equations),
        len(drawings),
        asset_dir_present,
    )

    result: dict[str, Any] = {
        "doc_id": doc_id,
        "file_path": file_path,
        "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
        "content": merged_text,
        "blocks_path": str(blocks_path),
    }
    if missing_paraid_count > 0:
        # Pipeline reads this from the parsed_data dict and writes it to
        # ``doc_status.metadata.parse_warnings`` so admin/list APIs can
        # surface the issue alongside the document record.
        result["parse_warnings"] = {"missing_paraid_count": missing_paraid_count}
    return result


# ---------------------------------------------------------------------------
# Debugging CLI: produce LightRAG sidecar artifacts directly from a DOCX so
# the on-disk format can be inspected without spinning up the full pipeline.
# ---------------------------------------------------------------------------


def _cli_load_blocks_summary(blocks_path: Path) -> tuple[dict, list[dict]]:
    """Read meta + content rows from a freshly written .blocks.jsonl."""
    meta: dict = {}
    rows: list[dict] = []
    with blocks_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if idx == 0 and obj.get("type") == "meta":
                meta = obj
            else:
                rows.append(obj)
    return meta, rows


def _cli_print_stats(meta: dict, rows: list[dict], parsed_dir: Path) -> None:
    headings = {row.get("heading", "") for row in rows}
    total_chars = sum(len(row.get("content", "")) for row in rows)
    avg = total_chars // len(rows) if rows else 0

    print("\n--- Document Statistics ---")
    print(f"Blocks: {len(rows)}")
    print(f"Unique headings: {len(headings)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average block size: {avg:,} chars")
    print(f"Tables sidecar:    {meta.get('table_file', False)}")
    print(f"Equations sidecar: {meta.get('equation_file', False)}")
    print(f"Drawings sidecar:  {meta.get('drawing_file', False)}")
    print(f"Asset dir present: {meta.get('asset_dir', False)}")
    if meta.get("asset_dir"):
        asset_dir = next(parsed_dir.glob("*.blocks.assets"), None)
        if asset_dir is not None:
            assets = sorted(p.name for p in asset_dir.iterdir())
            print(f"Asset files ({len(assets)}): {', '.join(assets)}")


def _cli_print_preview(rows: list[dict], limit: int = 5) -> None:
    print(f"\n--- Block Preview (first {limit}) ---")
    for i, row in enumerate(rows[:limit]):
        heading = row.get("heading") or "(no heading)"
        level = row.get("level", 0)
        content = row.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"\n[Block {i+1}] level={level} heading={heading!r}")
        print(f"blockid:  {row.get('blockid')}")
        print(f"positions: {row.get('positions')}")
        print(f"content:  {content}")


def main() -> None:
    """Entry point for ``python -m lightrag.native_parser.docx <file.docx>``.

    Writes ``<base>.blocks.jsonl`` plus ``.tables.json`` / ``.equations.json``
    / ``.drawings.json`` and a ``<base>.blocks.assets/`` directory under
    ``./parse_output/<stem>.parsed/`` (or the directory passed via
    ``--output-dir``). The output is the canonical LightRAG Document layout
    that ``parse_native`` produces in production — useful for verifying the
    sidecar format on a real DOCX without running the full pipeline.
    """
    import argparse

    from lightrag.utils import compute_mdhash_id

    parser = argparse.ArgumentParser(
        prog="python -m lightrag.native_parser.docx",
        description=(
            "Parse a DOCX file into the LightRAG Document format "
            "(blocks.jsonl + sidecar JSONs + assets dir) for debugging."
        ),
    )
    parser.add_argument(
        "document",
        type=str,
        help="Path to the DOCX file to parse",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help=(
            "Directory to write artifacts into. "
            "Default: ./parse_output/<stem>.parsed/"
        ),
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help=(
            "Override doc_id. Default: compute_mdhash_id(<basename>, "
            "prefix='doc-'), matching the value parse_native would use."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Forward debug flag to extract_docx_blocks for split traces.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print first 5 blocks of the resulting .blocks.jsonl",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics about the parsed document.",
    )

    args = parser.parse_args()

    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Error: File not found: {args.document}", file=sys.stderr)
        sys.exit(1)
    if doc_path.suffix.lower() != ".docx":
        print(
            f"Warning: File does not have .docx extension: {args.document}",
            file=sys.stderr,
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("parse_output") / f"{doc_path.stem}.parsed"

    file_bytes = doc_path.read_bytes()
    doc_id = args.doc_id or compute_mdhash_id(doc_path.name, prefix="doc-")

    print(f"Parsing document: {doc_path}")
    print(f"doc_id: {doc_id}")
    print(f"Output dir: {output_dir}")

    parsed_data = asyncio.run(
        parse_docx_to_lightrag_document(
            file_bytes=file_bytes,
            file_path=str(doc_path),
            doc_id=doc_id,
            source_path=str(doc_path),
            output_dir=output_dir,
            debug=args.debug,
        )
    )

    blocks_path = Path(parsed_data["blocks_path"])
    parsed_dir = blocks_path.parent

    print(f"Wrote: {blocks_path}")
    sidecar_files = sorted(
        p.name
        for p in parsed_dir.iterdir()
        if p.is_file() and p.name != blocks_path.name
    )
    if sidecar_files:
        print(f"Sidecars: {', '.join(sidecar_files)}")
    asset_dir = next(parsed_dir.glob("*.blocks.assets"), None)
    if asset_dir is not None and asset_dir.is_dir():
        asset_count = sum(1 for _ in asset_dir.iterdir())
        print(f"Assets ({asset_count}): {asset_dir}")

    if args.stats or args.preview:
        meta, rows = _cli_load_blocks_summary(blocks_path)
        if args.stats:
            _cli_print_stats(meta, rows, parsed_dir)
        if args.preview:
            _cli_print_preview(rows)


if __name__ == "__main__":
    main()
