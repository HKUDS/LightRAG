"""No-op format handlers for the two "nothing to parse" document formats.

These are internal (``user_selectable=False``) parsers dispatched by *format*,
not by a user engine choice:

- :class:`ReuseParser` handles ``lightrag`` rows — a document already parsed
  by some engine whose sidecar exists.  Reached only on resume/retry (a doc
  that finished parsing but failed/interrupted at a later stage and is pulled
  back through the parse queue).  Re-uses the stored content + sidecar
  instead of re-parsing (the original source may already be archived).
- :class:`PassthroughParser` handles ``raw`` rows — content supplied verbatim
  at insert time (e.g. ``ainsert`` of a plain string).

Both mirror the corresponding branches of the former ``parse_native`` exactly
(no disk writes, ``parse_stage_skipped=True``, no ``parse_engine`` key).
"""

from __future__ import annotations

from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG, FULL_DOCS_FORMAT_RAW
from lightrag.parser.base import BaseParser, ParseContext, ParseResult


class ReuseParser(BaseParser):
    """Reuse an already-parsed (``lightrag``-format) document's sidecar."""

    engine_name = "reuse"

    async def parse(self, ctx: ParseContext) -> ParseResult:
        from lightrag.utils_pipeline import (
            sidecar_blocks_path,
            strip_lightrag_doc_prefix,
        )

        doc_format = ctx.content_data.get("parse_format", FULL_DOCS_FORMAT_LIGHTRAG)
        merged_text = strip_lightrag_doc_prefix(
            ctx.content_data.get("content"), doc_format
        )
        # ``sidecar_location`` may be absent on historical/abnormal rows; tolerate
        # it (blocks_path="") rather than failing or re-routing to extraction.
        blocks_path = (
            sidecar_blocks_path(ctx.content_data.get("sidecar_location")) or ""
        )
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=doc_format,
            content=merged_text,
            blocks_path=blocks_path,
            parse_stage_skipped=True,
        )


class PassthroughParser(BaseParser):
    """Pass ``raw``-format content through verbatim (no parser ran)."""

    engine_name = "passthrough"

    async def parse(self, ctx: ParseContext) -> ParseResult:
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_RAW,
            content=ctx.content_data.get("content", ""),
            blocks_path="",
            parse_stage_skipped=True,
        )
