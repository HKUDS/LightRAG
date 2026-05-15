"""LightRAG-format debug CLI for the native DOCX parser.

Invocation::

    python -m lightrag.native_parser.docx <file.docx> [-o OUT_DIR]

Runs the production code path — :meth:`lightrag.LightRAG.parse_native` —
against a minimal in-memory RAG stand-in. Sidecar artifacts land under
``<OUT_DIR>/__parsed__/<stem>.parsed/`` (default OUT_DIR is the source
file's parent). Because ``parse_native`` archives the source after
parsing, the CLI works on a copy in a temp directory; the original docx
is left untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


class _DebugFullDocs:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    async def upsert(self, payload: dict[str, Any]) -> None:
        self.data.update(payload)

    async def get_by_id(self, doc_id: str) -> Any:
        return self.data.get(doc_id)

    async def index_done_callback(self) -> None:
        return None


class _DebugDocStatus:
    async def get_by_id(self, doc_id: str) -> Any:
        return None

    async def upsert(self, data: dict[str, Any]) -> None:
        return None


class _DebugRag:
    """Minimal LightRAG-shaped object exposing what ``parse_native`` reads."""

    # Bind the real method off LightRAG so we exercise the production path.
    from lightrag import LightRAG  # noqa: E402  (deferred to avoid circular)

    _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs
    parse_native = LightRAG.parse_native

    def __init__(self) -> None:
        self.full_docs = _DebugFullDocs()
        self.doc_status = _DebugDocStatus()

    def _resolve_source_file_for_parser(self, file_path: str) -> str:
        return file_path


def _print_summary(blocks_path: Path, preview: int) -> None:
    with blocks_path.open("r", encoding="utf-8") as fh:
        meta_line = fh.readline().strip()
        if not meta_line:
            raise SystemExit(f"empty blocks file at {blocks_path}")
        meta = json.loads(meta_line)
        rows = [json.loads(line) for line in fh if line.strip()]
    print(f"parsed dir : {blocks_path.parent}")
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
    if preview:
        print("--- content preview ---")
        for row in rows[:preview]:
            heading = row.get("heading") or ""
            content = row.get("content") or ""
            snippet = content if len(content) <= 80 else content[:77] + "..."
            print(f"  [{row.get('blockid', '')[:8]}] heading={heading!r} :: {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run LightRAG.parse_native on a docx file and dump the resulting "
            "LightRAG sidecar artifacts for inspection."
        )
    )
    parser.add_argument("docx", type=Path, help="Path to source .docx")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to receive the ``__parsed__/<stem>.parsed/`` tree. "
            "Defaults to the source file's parent."
        ),
    )
    parser.add_argument(
        "--doc-id",
        default="doc-cli0000000000000000000000000000",
        help=(
            "Override doc id (defaults to a CLI placeholder). The default "
            "shares its ``<doc_hash>`` prefix across invocations, so when "
            "inspecting multiple docx files in one session pass distinct "
            "values to keep sidecar ids unique."
        ),
    )
    parser.add_argument(
        "--preview", type=int, default=5, help="Number of content rows to preview"
    )
    args = parser.parse_args()

    if not args.docx.is_file():
        raise SystemExit(f"input not found: {args.docx}")

    output_dir = (args.output_dir or args.docx.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ``parse_native`` archives the source after parsing and resolves the
    # sidecar dir from ``INPUT_DIR``. Copy the docx into a temp inputs dir
    # so the original is preserved and the archived copy gets cleaned up
    # with the temp tree.
    from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE

    rag = _DebugRag()
    with tempfile.TemporaryDirectory(prefix="native_docx_cli_") as tmp_root:
        input_dir = Path(tmp_root) / "inputs"
        input_dir.mkdir()
        os.environ["INPUT_DIR"] = str(input_dir)
        scratch_docx = input_dir / args.docx.name
        shutil.copyfile(args.docx, scratch_docx)

        async def _run() -> dict[str, Any]:
            return await rag.parse_native(
                args.doc_id,
                str(scratch_docx),
                {
                    "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                    "content": "",
                    "source_path": str(scratch_docx),
                },
            )

        result = asyncio.run(_run())
        produced_dir = Path(result["blocks_path"]).parent
        target_dir = output_dir / produced_dir.name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(produced_dir, target_dir)

    blocks_path = target_dir / Path(result["blocks_path"]).name
    _print_summary(blocks_path, args.preview)
    if result.get("parse_warnings"):
        print("--- parse warnings ---")
        print(result["parse_warnings"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
