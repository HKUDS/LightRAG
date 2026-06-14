"""Legacy engine adapter: worker-stage plain-text extraction (RAW output)."""

from __future__ import annotations

import asyncio
import os
import time

from lightrag.constants import FULL_DOCS_FORMAT_RAW, PARSER_ENGINE_LEGACY
from lightrag.parser.base import BaseParser, ParseContext, ParseResult


class LegacyParser(BaseParser):
    """Extract plain text in-process and store it as a ``raw`` document.

    Also serves as the dispatch fallback engine: an unknown/unsupported
    suffix raises (caught at the parse stage → doc_status FAILED) rather than
    silently producing empty content.
    """

    engine_name = PARSER_ENGINE_LEGACY

    async def parse(self, ctx: ParseContext) -> ParseResult:
        from lightrag.parser.legacy.extractors import (
            LegacyExtractionError,
            extract_text,
        )
        from lightrag.parser.registry import suffix_capabilities

        rs = ctx.resolve(self.engine_name)
        source = rs.source_path
        if not source.is_file():
            raise FileNotFoundError(f"legacy source file not found: {source}")

        suffix = source.suffix.lower().lstrip(".")
        if suffix not in suffix_capabilities(self.engine_name):
            raise ValueError(
                f"legacy parser does not support .{suffix or '<no suffix>'}: "
                f"doc_id={ctx.doc_id} file={ctx.file_path}"
            )

        file_bytes = await asyncio.to_thread(source.read_bytes)
        # The PDF password is sourced from env only: the parser layer reads
        # ``PDF_DECRYPT_PASSWORD`` directly rather than from the API layer's
        # ``global_args`` (which would invert the parser -> API dependency
        # direction).
        pdf_password = os.getenv("PDF_DECRYPT_PASSWORD") or None
        text = await asyncio.to_thread(
            extract_text, file_bytes, suffix, pdf_password=pdf_password
        )
        # The binary extractors (pdf/docx/pptx/xlsx) return whatever the
        # library yields — a scanned PDF with no text layer extracts to pure
        # whitespace. Fail the parse (like the text-decode path already does)
        # instead of persisting an empty document into chunking.
        if not text.strip():
            raise LegacyExtractionError(
                f"extracted no usable text from {ctx.file_path} (doc_id={ctx.doc_id})"
            )

        await ctx.rag._persist_parsed_full_docs(
            ctx.doc_id,
            {
                "content": text,
                "file_path": ctx.file_path,
                "parse_format": FULL_DOCS_FORMAT_RAW,
                "parse_engine": self.engine_name,
                "update_time": int(time.time()),
            },
        )
        await ctx.archive_source(str(source))
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_RAW,
            content=text,
            blocks_path="",
            parse_engine=self.engine_name,
        )
