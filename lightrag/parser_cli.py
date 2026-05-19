"""Unified sidecar debug CLI for native / mineru / docling parsers.

Drives ``LightRAG.parse_<engine>`` against a single source file and writes
the resulting sidecar (and raw bundle, for mineru/docling) into a flat
layout — no ``__parsed__/`` middle layer, source file never archived —
so the artifacts can be inspected next to the input file.

Invocation::

    python -m lightrag.parser_cli path/to/sample.docx --engine native
    python -m lightrag.parser_cli path/to/sample.pdf  --engine mineru
    python -m lightrag.parser_cli path/to/sample.pdf  --engine docling --force-reparse

See ``docs/ParserDebugCLI-zh.md`` for the full reference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock

ENGINES = ("native", "mineru", "docling")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parse_sidecar",
        description=(
            "Run LightRAG.parse_<engine> on a single file and emit sidecar "
            "artifacts (plus a raw bundle for mineru/docling) into a flat "
            "layout alongside the source. No __parsed__/ middle layer; the "
            "source file is never moved."
        ),
    )
    parser.add_argument("input_file", type=Path, help="Source file to parse.")
    parser.add_argument(
        "--engine",
        required=True,
        choices=ENGINES,
        help="Parser engine to drive.",
    )
    parser.add_argument(
        "-o",
        "--sidecar-parent-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for <name>.parsed/ and <name>.<engine>_raw/. "
            "Default: the source file's parent directory."
        ),
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Override the doc id. Default: doc-<md5(absolute input path)>.",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help=(
            "Only affects mineru/docling. By default a non-empty raw_dir is "
            "treated as a valid cache and reused without manifest checks; "
            "this flag clears raw_dir and forces a fresh download/parse."
        ),
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        metavar="N",
        help="Number of block rows to preview after parsing (0 disables).",
    )
    return parser


def _print_summary(blocks_path: Path, raw_dir: Path | None, preview: int) -> None:
    with blocks_path.open("r", encoding="utf-8") as fh:
        meta_line = fh.readline().strip()
        if not meta_line:
            raise SystemExit(f"empty blocks file at {blocks_path}")
        meta = json.loads(meta_line)
        rows = [json.loads(line) for line in fh if line.strip()]
    parsed_dir = blocks_path.parent
    print(f"parsed dir : {parsed_dir} (exists={parsed_dir.exists()})")
    if raw_dir is not None:
        print(f"raw dir    : {raw_dir} (exists={raw_dir.exists()})")
    print(f"document   : {meta.get('document_name')}")
    print(f"doc_id     : {meta.get('doc_id')}")
    print(f"engine     : {meta.get('parse_engine')}")
    print(f"blocks     : {meta.get('blocks')}")
    print(
        f"sidecars   : tables={meta.get('table_file')} "
        f"drawings={meta.get('drawing_file')} "
        f"equations={meta.get('equation_file')} "
        f"asset_dir={meta.get('asset_dir')}"
    )
    if preview > 0 and rows:
        shown = min(preview, len(rows))
        print(f"--- preview (first {shown} of {len(rows)} blocks) ---")
        for row in rows[:preview]:
            heading = row.get("heading") or ""
            content = (row.get("content") or "").replace("\n", " ")
            snippet = content if len(content) <= 80 else content[:77] + "..."
            print(
                f"  [{row.get('blockid', '')[:8]}] " f"heading={heading!r} :: {snippet}"
            )


async def _run(args: argparse.Namespace) -> int:
    # Pipeline + heavy parser imports are deferred so ``--help`` and the
    # input-file existence check don't pay for them.
    from lightrag.constants import (
        FULL_DOCS_FORMAT_PENDING_PARSE,
        PARSER_ENGINE_SUFFIX_CAPABILITIES,
    )
    from lightrag.parser_debug import build_debug_rag
    from lightrag.utils import compute_mdhash_id
    import lightrag.pipeline as pipeline_mod
    import lightrag.utils_pipeline as utils_pipeline_mod

    source = args.input_file.resolve()
    if not source.is_file():
        print(f"error: input file does not exist: {source}", file=sys.stderr)
        return 1

    # Reject suffix/engine mismatches up-front: the pipeline would otherwise
    # fail deep inside the IR builder with a less helpful message.
    suffix = source.suffix.lstrip(".").lower()
    supported = PARSER_ENGINE_SUFFIX_CAPABILITIES.get(args.engine, frozenset())
    if suffix not in supported:
        print(
            f"error: engine '{args.engine}' does not support .{suffix or '<no suffix>'} "
            f"files (supported: {', '.join(sorted(supported))})",
            file=sys.stderr,
        )
        return 1

    sidecar_parent = (args.sidecar_parent_dir or source.parent).resolve()
    sidecar_parent.mkdir(parents=True, exist_ok=True)

    parsed_dir = sidecar_parent / f"{source.name}.parsed"
    raw_dir = (
        sidecar_parent / f"{source.name}.{args.engine}_raw"
        if args.engine in ("mineru", "docling")
        else None
    )

    doc_id = args.doc_id or compute_mdhash_id(str(source), prefix="doc-")

    def _patched_artifact_dir(
        file_path: str | None = None,
        *,
        parent_hint: Any | None = None,
    ) -> Path:
        # Flatten the production "<INPUT_DIR>/__parsed__/<base>.parsed/"
        # layout to "<sidecar_parent>/<source.name>.parsed/" so the sidecar
        # and the source file sit side by side.
        return parsed_dir

    def _lenient_bundle(raw_dir_arg: Path, _source_file: Path) -> bool:
        return raw_dir_arg.exists() and any(raw_dir_arg.iterdir())

    def _force_miss(*_args: Any, **_kwargs: Any) -> bool:
        return False

    bundle_check = _force_miss if args.force_reparse else _lenient_bundle

    async def _noop_archive(*_args: Any, **_kwargs: Any) -> None:
        return None

    rag = build_debug_rag()
    parse_method = getattr(rag, f"parse_{args.engine}")

    with ExitStack() as stack:
        # Patch 1: redirect sidecar output to the flat layout.
        # parsed_artifact_dir_for is from-imported into pipeline at
        # module load, so patch both namespaces.
        stack.enter_context(
            mock.patch.object(
                utils_pipeline_mod,
                "parsed_artifact_dir_for",
                _patched_artifact_dir,
            )
        )
        stack.enter_context(
            mock.patch.object(
                pipeline_mod,
                "parsed_artifact_dir_for",
                _patched_artifact_dir,
            )
        )

        # Patch 2: raw cache strategy. parse_mineru / parse_docling do a
        # function-local ``from lightrag.external_parser.<eng> import
        # is_bundle_valid``, so we replace the name on the facade module.
        if args.engine == "mineru":
            import lightrag.external_parser.mineru as mineru_pkg

            stack.enter_context(
                mock.patch.object(mineru_pkg, "is_bundle_valid", bundle_check)
            )
        elif args.engine == "docling":
            import lightrag.external_parser.docling as docling_pkg

            stack.enter_context(
                mock.patch.object(docling_pkg, "is_bundle_valid", bundle_check)
            )

        # Patch 3: keep the source file in place. All three parse_* methods
        # call archive_docx_source_after_full_docs_sync at the end.
        stack.enter_context(
            mock.patch.object(
                pipeline_mod,
                "archive_docx_source_after_full_docs_sync",
                _noop_archive,
            )
        )

        result = await parse_method(
            doc_id,
            str(source),
            {
                "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                "content": "",
            },
        )

    blocks_path = Path(result["blocks_path"])
    _print_summary(blocks_path, raw_dir, args.preview)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
