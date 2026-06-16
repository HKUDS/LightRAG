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
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    def is_bundle_valid(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        engine_params: Mapping[str, Any] | None = None,
    ) -> bool:
        """Cheap cache-hit check against the raw bundle on disk.

        ``engine_params`` is the per-file engine-parameter override (decoded
        from ``parse_engine``); it MUST participate in the cache signature so an
        overridden document does not spuriously hit a bundle parsed with
        different params.
        """
        ...

    @abstractmethod
    async def download_into(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        upload_name: str,
        engine_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Fetch the raw bundle into ``raw_dir`` (called on cache miss only).

        ``engine_params`` is the per-file engine-parameter override applied to
        both the request payload and the recorded cache signature.
        """
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
        from lightrag.parser.routing import decode_parse_engine, encode_parse_engine
        from lightrag.sidecar import write_sidecar
        from lightrag.utils_pipeline import (
            make_lightrag_doc_content,
            sidecar_uri_for,
        )

        # Per-file engine params are encoded in the stored ``parse_engine``
        # directive (e.g. ``mineru(page_range=1-3)``); decode them once and
        # thread the SAME dict into both the cache-hit check and the download so
        # an overridden doc can never hit a bundle parsed with different params.
        # A malformed/corrupt directive fails this doc loudly rather than
        # silently parsing with no params.
        _engine, engine_params, decode_errs = decode_parse_engine(
            ctx.content_data.get("parse_engine")
            if isinstance(ctx.content_data, dict)
            else None
        )
        if decode_errs:
            raise ValueError(
                f"{self.engine_name}: invalid parse_engine for doc_id={ctx.doc_id}: "
                + "; ".join(decode_errs)
            )
        engine_params = engine_params or None

        rs = ctx.resolve(self.engine_name)
        source = rs.source_path
        if not source.is_file():
            raise FileNotFoundError(
                f"{self.engine_name} source file not found: {source}"
            )
        raw_dir = raw_dir_for_parsed_dir(rs.parsed_dir, suffix=self.raw_dir_suffix)
        force_reparse = env_bool(self.force_reparse_env, False)

        parse_stage_skipped = False
        if not force_reparse and self.is_bundle_valid(
            raw_dir, source, engine_params=engine_params
        ):
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
            await self.download_into(
                raw_dir,
                source,
                upload_name=rs.document_name,
                engine_params=engine_params,
            )

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
                # Re-encode the engine + params so the persisted directive keeps
                # the per-file params (the `{**existing, **record}` merge in
                # _persist_parsed_full_docs would otherwise revert it to the
                # bare engine name).
                "parse_engine": encode_parse_engine(self.engine_name, engine_params),
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
            parse_stage_skipped=parse_stage_skipped,
        )
