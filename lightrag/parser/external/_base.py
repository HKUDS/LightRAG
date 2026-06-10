"""Shared template for external (download + raw-bundle cache) parser engines.

``ExternalParserBase.parse`` fixes the common MinerU/Docling flow once:

    resolve → raw_dir → force-reparse check → cache-hit skip
    else (mkdir + clear_dir_contents + download_into) → build_ir
    → write_sidecar → persist full_docs (lightrag) → archive source

Subclasses implement three engine-private hooks (``is_bundle_valid`` /
``download_into`` / ``build_ir``) and set ``raw_dir_suffix`` /
``force_reparse_env``.  This is the reshaped #3207 contract — now an
*internal* template rather than the top-level parser interface — with its two
gaps fixed: ``clear_dir_contents`` runs inside the template (cache-miss only),
and the per-engine upload-name divergence is normalised to a single
``upload_name`` hook parameter.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.parser.base import BaseParser, ParseContext, ParseResult
from lightrag.utils import logger

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


class ExternalParserBase(BaseParser):
    """Base for engines that fetch a raw bundle from an external service."""

    raw_dir_suffix: str
    force_reparse_env: str

    # --- engine-private hooks ------------------------------------------------
    @abstractmethod
    def is_bundle_valid(self, raw_dir: Path, source_path: Path) -> bool:
        """Cheap cache-hit check against the raw bundle on disk."""
        ...

    @abstractmethod
    async def download_into(
        self, raw_dir: Path, source_path: Path, *, upload_name: str
    ) -> None:
        """Fetch the raw bundle into ``raw_dir`` (called on cache miss only)."""
        ...

    @abstractmethod
    def build_ir(self, raw_dir: Path, document_name: str) -> "IRDoc":
        """Convert the raw bundle to an :class:`IRDoc`."""
        ...

    def validate_ir(self, ir: "IRDoc", *, file_path: str, raw_dir: Path) -> None:
        """Optional post-build validation hook (default no-op)."""

    # --- template ------------------------------------------------------------
    async def parse(self, ctx: ParseContext) -> ParseResult:
        from lightrag.parser.external._common import (
            clear_dir_contents,
            env_bool,
            raw_dir_for_parsed_dir,
        )
        from lightrag.sidecar import write_sidecar
        from lightrag.utils_pipeline import (
            archive_source_after_full_docs_sync,
            make_lightrag_doc_content,
            sidecar_uri_for,
        )

        rs = ctx.resolve(self.engine_name)
        source = rs.source_path
        if not source.is_file():
            raise FileNotFoundError(
                f"{self.engine_name} source file not found: {source}"
            )
        raw_dir = raw_dir_for_parsed_dir(rs.parsed_dir, suffix=self.raw_dir_suffix)
        force_reparse = env_bool(self.force_reparse_env, False)

        parse_stage_skipped = False
        if not force_reparse and self.is_bundle_valid(raw_dir, source):
            # Cache hit: stay purely local so a re-parse still works when the
            # external endpoint is temporarily unavailable.
            parse_stage_skipped = True
            logger.info("[%s] raw cache hit doc_id=%s", self.engine_name, ctx.doc_id)
        else:
            if force_reparse and raw_dir.exists():
                logger.info(
                    "[%s] %s set; discarding bundle at %s",
                    self.engine_name,
                    self.force_reparse_env,
                    raw_dir,
                )
            # download_into mkdir's raw_dir; we wipe stale contents first so a
            # previous bundle cannot leak into the new one (cache-miss only).
            raw_dir.mkdir(parents=True, exist_ok=True)
            clear_dir_contents(raw_dir)
            logger.info(
                "[%s] Parsing %s %s (may take a few minutes)",
                self.engine_name,
                ctx.doc_id,
                source.name,
            )
            await self.download_into(raw_dir, source, upload_name=rs.document_name)

        ir = self.build_ir(raw_dir, rs.document_name)
        self.validate_ir(ir, file_path=ctx.file_path, raw_dir=raw_dir)
        parsed_data = write_sidecar(
            ir,
            parsed_dir=rs.parsed_dir,
            doc_id=ctx.doc_id,
            engine=self.engine_name,
        )

        await ctx.rag._persist_parsed_full_docs(
            ctx.doc_id,
            {
                "content": make_lightrag_doc_content(parsed_data["content"]),
                "file_path": ctx.file_path,
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                "sidecar_location": sidecar_uri_for(rs.parsed_dir),
                "parse_engine": self.engine_name,
                "update_time": int(time.time()),
            },
        )
        await archive_source_after_full_docs_sync(str(source))
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_LIGHTRAG,
            content=parsed_data["content"],
            blocks_path=parsed_data["blocks_path"],
            parse_engine=self.engine_name,
            parse_stage_skipped=parse_stage_skipped,
        )
