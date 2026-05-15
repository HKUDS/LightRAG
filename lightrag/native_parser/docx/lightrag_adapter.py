"""LightRAG Document writer for the native DOCX parser.

The adapter calls :func:`extract_docx_blocks` with ``fixlevel=0`` and
routes the result through the unified
:class:`lightrag.parser_adapters.NativeDocxAdapter` +
:func:`lightrag.sidecar.write_sidecar` pipeline. The on-disk artifacts:

- ``<base>.blocks.jsonl``           — main file (meta line + content lines)
- ``<base>.tables.json``            — table sidecar (only when non-empty)
- ``<base>.equations.json``         — equation sidecar (block equations only)
- ``<base>.drawings.json``          — drawing sidecar (only when non-empty)
- ``<base>.blocks.assets/``         — exported image bytes (only when non-empty)

are byte-equivalent to the legacy inline-regex implementation; see
``tests/parser_adapters/golden/native_docx/`` for the captured snapshots
that gate the migration.

The :class:`DrawingExtractionContext` extracts image bytes into the
``<base>.blocks.assets/`` directory *before* the IR adapter runs;
:func:`write_sidecar` is therefore invoked with ``clean_parsed_dir=False``
so those bytes survive — the writer just records them via
``AssetSpec.source=None``.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG, PARSER_ENGINE_NATIVE
from lightrag.utils import logger
from lightrag.utils_pipeline import (
    canonicalize_parser_hinted_basename,
    parsed_artifact_dir_for_source,
)

from .drawing_image_extractor import (
    DrawingExtractionContext,
    _load_relationships,
)
from .parse_document import extract_docx_blocks


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
    # Deferred imports break the otherwise-circular path
    # ``lightrag.parser_adapters/__init__.py`` →
    # ``lightrag.parser_adapters.native_docx`` →
    # ``lightrag.native_parser.docx.drawing_image_extractor`` →
    # ``lightrag.native_parser.docx/__init__.py`` →
    # ``lightrag.native_parser.docx.lightrag_adapter``.
    from lightrag.parser_adapters import NativeDocxAdapter
    from lightrag.sidecar import write_sidecar

    parsed_dir = (
        Path(output_dir)
        if output_dir is not None
        else parsed_artifact_dir_for_source(source_path, file_path)
    )
    # Pre-clean parsed_dir here (rather than letting write_sidecar do it)
    # so the DrawingExtractionContext can write asset bytes into
    # ``<base>.blocks.assets/`` BEFORE we call write_sidecar — the writer
    # is then invoked with ``clean_parsed_dir=False`` to keep them.
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    canonical_basename = (
        canonicalize_parser_hinted_basename(file_path)
        or Path(source_path).name
        or f"{doc_id}.bin"
    )
    base_name = Path(canonical_basename).stem or canonical_basename
    asset_dir = parsed_dir / f"{base_name}.blocks.assets"

    # ``extract_docx_blocks`` and ``DrawingExtractionContext`` both work
    # against a filesystem path (the latter opens the docx as a zip), so
    # persist the caller-provided bytes to a temp file and clean up at the
    # end.
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        temp_docx = Path(tmp.name)

    parse_warnings: dict[str, Any] = {}
    parse_metadata: dict[str, Any] = {}

    try:
        asset_dir.mkdir(parents=True, exist_ok=True)
        ctx = DrawingExtractionContext(
            docx_path=temp_docx,
            blocks_output_path=parsed_dir / f"{base_name}.blocks.jsonl",
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
        # paraIds (legacy / non-Word docx authors omit ``w14:paraId``), but
        # a single warning with the count is enough for the user. Affected
        # blocks emit ``positions: [{"type": "paraid", "range": null}]``.
        logger.warning(
            "[native_parser/docx] %s: %d paragraphs lack paraId; "
            "Re-saving file in Word 2013+ to regenerate ids.",
            Path(file_path).name,
            missing_paraid_count,
        )

    ir = NativeDocxAdapter().normalize(
        blocks,
        document_name=canonical_basename,
        asset_dir_name=asset_dir.name,
        parse_metadata=parse_metadata,
    )
    parsed_data = write_sidecar(
        ir,
        parsed_dir=parsed_dir,
        doc_id=doc_id,
        engine=PARSER_ENGINE_NATIVE,
        clean_parsed_dir=False,  # adapter already pre-created + populated
        block_drawing_path_style="basename_only",  # legacy native convention
    )

    # write_sidecar leaves the asset dir empty/removed when no assets were
    # declared; otherwise it stays put. Keep parity with the legacy adapter
    # by treating ``parse_format`` and the return shape identically.
    result: dict[str, Any] = {
        "doc_id": doc_id,
        "file_path": file_path,
        "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
        "content": parsed_data["content"],
        "blocks_path": parsed_data["blocks_path"],
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
    import json

    with blocks_path.open("r", encoding="utf-8") as fh:
        meta_line = fh.readline().strip()
        if not meta_line:
            raise SystemExit(f"empty blocks file at {blocks_path}")
        meta = json.loads(meta_line)
        rows = [json.loads(line) for line in fh if line.strip()]
    return meta, rows


def _cli_print_stats(meta: dict, rows: list[dict], parsed_dir: Path) -> None:
    print(f"parsed dir : {parsed_dir}")
    print(f"document   : {meta.get('document_name')}")
    print(f"doc_title  : {meta.get('doc_title')!r}")
    print(f"doc_id     : {meta.get('doc_id')}")
    print(f"blocks     : {meta.get('blocks')}")
    print(
        f"sidecars   : tables={meta.get('table_file')} "
        f"drawings={meta.get('drawing_file')} "
        f"equations={meta.get('equation_file')} "
        f"asset_dir={meta.get('asset_dir')}"
    )
    print(f"hash       : {meta.get('document_hash')}")
    print(f"row count  : {len(rows)}")


def _cli_print_preview(rows: list[dict], limit: int = 5) -> None:
    for row in rows[:limit]:
        heading = row.get("heading") or ""
        content = row.get("content") or ""
        snippet = content if len(content) <= 80 else content[:77] + "..."
        print(f"  [{row.get('blockid', '')[:8]}] heading={heading!r} :: {snippet}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the native DOCX adapter against a docx file and dump "
        "the resulting LightRAG sidecar artifacts to disk."
    )
    parser.add_argument("docx", type=Path, help="Path to source .docx")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--doc-id",
        default="doc-cli0000000000000000000000000000",
        help="Override doc id (defaults to a CLI placeholder)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--preview", type=int, default=5, help="Number of content rows to preview"
    )
    args = parser.parse_args()

    if not args.docx.is_file():
        raise SystemExit(f"input not found: {args.docx}")

    output_dir = args.output_dir or args.docx.with_suffix(".parsed")
    file_bytes = args.docx.read_bytes()

    async def _run() -> dict[str, Any]:
        return await parse_docx_to_lightrag_document(
            file_bytes=file_bytes,
            file_path=str(args.docx),
            doc_id=args.doc_id,
            source_path=str(args.docx),
            output_dir=output_dir,
            debug=args.debug,
        )

    result = asyncio.run(_run())
    blocks_path = Path(result["blocks_path"])
    parsed_dir = blocks_path.parent
    meta, rows = _cli_load_blocks_summary(blocks_path)
    _cli_print_stats(meta, rows, parsed_dir)
    if args.preview:
        print("--- content preview ---")
        _cli_print_preview(rows, limit=args.preview)
    if result.get("parse_warnings"):
        print("--- parse warnings ---")
        print(result["parse_warnings"])


if __name__ == "__main__":
    sys.exit(main() or 0)
