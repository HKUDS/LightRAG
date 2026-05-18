"""Debug CLI: convert a ``*.docling_raw/`` bundle directly into a sidecar dir.

Skips the production pipeline (docling-serve, cache validation, ``full_docs``,
storage updates) so the adapter / writer can be iterated against an existing
raw bundle without round-tripping through the live service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from lightrag.external_parser._manifest import MANIFEST_FILENAME
from lightrag.external_parser.docling import (
    DOCLING_RAW_DIR_SUFFIX,
    DoclingAdapter,
)
from lightrag.sidecar.ir import IRBlock
from lightrag.sidecar.writer import write_sidecar
from lightrag.utils import compute_mdhash_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lightrag-docling-sidecar",
        description=(
            "Convert an existing Docling raw bundle into a LightRAG sidecar "
            "directory. Debug-only — does not touch storage, cache, or the "
            "docling-serve endpoint."
        ),
    )
    parser.add_argument(
        "raw_dir",
        type=Path,
        help="Path to a *.docling_raw/ bundle directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Sidecar output directory. Default: sibling of raw_dir named "
            "<document_name>.parsed/. Always cleaned and overwritten."
        ),
    )
    parser.add_argument(
        "--document-name",
        default=None,
        help=(
            "Override the document_name written into the IR / meta. "
            "Default: derived from raw_dir name (strip '.docling_raw'), "
            "falling back to the main JSON's origin.filename."
        ),
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help=(
            "Override the doc-<md5> id. Default: "
            "compute_mdhash_id(document_name, prefix='doc-')."
        ),
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        metavar="N",
        help="Number of blocks to preview in the summary (0 disables).",
    )
    return parser


def _find_unique_main_json(raw_dir: Path) -> Path | None:
    candidates = sorted(
        p for p in raw_dir.glob("*.json") if p.is_file() and p.name != MANIFEST_FILENAME
    )
    return candidates[0] if len(candidates) == 1 else None


def _infer_document_name(raw_dir: Path) -> str:
    if raw_dir.name.endswith(DOCLING_RAW_DIR_SUFFIX):
        stripped = raw_dir.name[: -len(DOCLING_RAW_DIR_SUFFIX)]
        if stripped:
            return stripped

    main_json = _find_unique_main_json(raw_dir)
    if main_json is None:
        raise ValueError(
            f"cannot infer document_name from {raw_dir}: directory name does "
            f"not end with '{DOCLING_RAW_DIR_SUFFIX}' and the bundle does not "
            f"contain exactly one non-manifest JSON. Pass --document-name."
        )
    try:
        doc: Any = json.loads(main_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read main JSON at {main_json}: {exc}. " f"Pass --document-name."
        ) from exc

    origin = doc.get("origin") if isinstance(doc, dict) else None
    filename = origin.get("filename") if isinstance(origin, dict) else None
    if not isinstance(filename, str) or not filename:
        raise ValueError(
            f"main JSON at {main_json} has no usable origin.filename. "
            f"Pass --document-name."
        )
    return filename


def _format_block_preview(block: IRBlock, max_chars: int = 80) -> str:
    body = " ".join(block.content_template.split())
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    heading = block.heading or "(no heading)"
    return f"{heading} | {body}"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    raw_dir: Path = args.raw_dir
    if not raw_dir.exists():
        print(f"error: raw dir does not exist: {raw_dir}", file=sys.stderr)
        return 1
    if not raw_dir.is_dir():
        print(f"error: raw dir is not a directory: {raw_dir}", file=sys.stderr)
        return 1

    try:
        document_name = args.document_name or _infer_document_name(raw_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    doc_id = args.doc_id or compute_mdhash_id(document_name, prefix="doc-")
    output_dir: Path = args.output_dir or raw_dir.parent / f"{document_name}.parsed"

    try:
        ir = DoclingAdapter().normalize_from_workdir(
            raw_dir, document_name=document_name
        )
        result = write_sidecar(
            ir,
            parsed_dir=output_dir,
            doc_id=doc_id,
            engine="docling",
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    n_tables = sum(len(b.tables) for b in ir.blocks)
    n_drawings = sum(len(b.drawings) for b in ir.blocks)
    n_equations = sum(1 for b in ir.blocks for eq in b.equations if eq.is_block)
    n_assets = len(ir.assets)

    print(f"output_dir:    {output_dir}")
    print(f"blocks_path:   {result['blocks_path']}")
    print(f"document_name: {document_name}")
    print(f"doc_title:     {ir.doc_title}")
    print(f"doc_id:        {doc_id}")
    print(
        f"counts:        blocks={len(ir.blocks)} tables={n_tables} "
        f"drawings={n_drawings} equations={n_equations} assets={n_assets}"
    )

    preview_count = max(0, args.preview)
    if preview_count and ir.blocks:
        print("preview:")
        for block in ir.blocks[:preview_count]:
            print(f"  - {_format_block_preview(block)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
