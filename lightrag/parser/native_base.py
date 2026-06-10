"""Shared template for native (local, in-process) parser engines.

``NativeParserBase.parse`` fixes the common local-parse flow once:

    resolve + validate source → compute parsed_dir/asset_dir
    → pre-clean (rmtree parsed_dir + mkdir + mkdir asset_dir, with rollback)
    → extract() in a thread → build_ir() → write_sidecar(clean_parsed_dir=False)
    → persist full_docs (lightrag) → archive source

Subclasses implement ``extract`` (sync, runs in a thread) and ``build_ir``.
Currently only :class:`NativeDocxParser`; xlsx/pptx/md land later as new
subclasses implementing the same two hooks.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.parser.base import BaseParser, ParseContext, ParseResult

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


class NativeParserBase(BaseParser):
    """Base for engines that parse a file locally into a sidecar."""

    # ``write_sidecar`` block_drawing_path_style; docx keeps the legacy
    # "basename_only" shape for byte-equivalence.
    sidecar_path_style: str = "with_prefix"
    # Prefix used in the "empty content" error message.
    empty_content_label: str = "Native"

    # --- engine-private hooks ------------------------------------------------
    def validate_source(self, source: Path, file_path: str) -> None:
        """Validate the resolved source (default: must be an existing file)."""
        if not (source.exists() and source.is_file()):
            raise FileNotFoundError(
                f"{self.engine_name} source file not found: {source}"
            )

    @abstractmethod
    def extract(
        self, source: Path, *, parsed_dir: Path, asset_dir: Path, base_name: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Extract ``(blocks, warnings, metadata)`` (sync; runs in a thread).

        ``parsed_dir`` and ``asset_dir`` are pre-created by the template; the
        hook may write side artifacts (e.g. image bytes) into ``asset_dir``
        before :func:`write_sidecar` runs with ``clean_parsed_dir=False``.
        """
        ...

    @abstractmethod
    def build_ir(
        self,
        blocks: list[dict[str, Any]],
        *,
        document_name: str,
        asset_dir_name: str,
        metadata: dict[str, Any],
    ) -> "IRDoc": ...

    def surface_warnings(
        self, warnings: dict[str, Any], source: Path
    ) -> dict[str, Any] | None:
        """Map parser warnings to the ``parse_warnings`` result field (opt)."""
        return None

    # --- template ------------------------------------------------------------
    async def parse(self, ctx: ParseContext) -> ParseResult:
        from lightrag.sidecar import write_sidecar
        from lightrag.utils_pipeline import (
            make_lightrag_doc_content,
            sidecar_uri_for,
        )

        rs = ctx.resolve(self.engine_name)
        source = rs.source_path
        self.validate_source(source, ctx.file_path)

        document_name = rs.document_name
        base_name = Path(document_name).stem or document_name
        parsed_dir = rs.parsed_dir
        asset_dir = parsed_dir / f"{base_name}.blocks.assets"

        def _extract_sync():
            # Pre-clean parsed_dir and pre-create asset_dir so the extractor
            # can write image bytes BEFORE write_sidecar (clean_parsed_dir=False
            # then keeps them). parsed_artifact_dir_for returns a unique dir per
            # source, so this rmtree only clobbers a prior attempt's artifacts.
            if parsed_dir.exists():
                shutil.rmtree(parsed_dir)
            parsed_dir.mkdir(parents=True, exist_ok=True)
            asset_dir.mkdir(parents=True, exist_ok=True)
            return self.extract(
                source, parsed_dir=parsed_dir, asset_dir=asset_dir, base_name=base_name
            )

        try:
            blocks, warnings, metadata = await asyncio.to_thread(_extract_sync)
        except BaseException:
            # Roll back the pre-created (possibly partial) dirs on any failure.
            if parsed_dir.exists():
                shutil.rmtree(parsed_dir, ignore_errors=True)
            raise
        if not blocks:
            if parsed_dir.exists():
                shutil.rmtree(parsed_dir, ignore_errors=True)
            raise ValueError(
                f"{self.empty_content_label} parser returned empty content "
                f"for {ctx.file_path}"
            )

        parse_warnings = self.surface_warnings(warnings, source)
        ir = self.build_ir(
            blocks,
            document_name=document_name,
            asset_dir_name=asset_dir.name,
            metadata=metadata,
        )
        parsed_data = write_sidecar(
            ir,
            parsed_dir=parsed_dir,
            doc_id=ctx.doc_id,
            engine=self.engine_name,
            clean_parsed_dir=False,  # asset dir pre-populated above
            block_drawing_path_style=self.sidecar_path_style,
        )

        await ctx.rag._persist_parsed_full_docs(
            ctx.doc_id,
            {
                "content": make_lightrag_doc_content(parsed_data["content"]),
                "file_path": ctx.file_path,
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                "sidecar_location": sidecar_uri_for(parsed_dir),
                "parse_engine": self.engine_name,
                "update_time": int(time.time()),
            },
        )
        await ctx.archive_source(str(source))
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_LIGHTRAG,
            content=parsed_data["content"],
            blocks_path=parsed_data["blocks_path"],
            parse_engine=self.engine_name,
            parse_warnings=parse_warnings,
        )
