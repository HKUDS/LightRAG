"""Native engine entry point — dispatches by source suffix.

The registry exposes a single ``native`` engine with one ``impl``; routing
yields the engine key and ``get_parser("native")`` returns this one instance.
Suffix capabilities (``docx`` / ``md`` / ``textpack``) are declared on the
registry spec but do NOT pick an implementation — that is this dispatcher's
job.

It delegates the **entire** ``parse(ctx)`` to the matching concrete parser
(rather than subclassing :class:`NativeParserBase` and only forwarding the
``extract`` / ``build_ir`` hooks) so each concrete parser keeps its own
``sidecar_path_style`` — docx stays ``basename_only`` (byte-equivalent golden
output) while markdown uses ``with_prefix``.
"""

from __future__ import annotations

from pathlib import Path

from lightrag.constants import PARSER_ENGINE_NATIVE
from lightrag.parser.base import BaseParser, ParseContext, ParseResult

_MARKDOWN_SUFFIXES = {".md", ".textpack"}


class NativeParser(BaseParser):
    """Routes a document to the docx or markdown native parser by suffix."""

    engine_name = PARSER_ENGINE_NATIVE

    def __init__(self) -> None:
        self._docx: BaseParser | None = None
        self._markdown: BaseParser | None = None

    def _docx_parser(self) -> BaseParser:
        parser = self._docx
        if parser is None:
            from lightrag.parser.docx.parser import NativeDocxParser

            parser = NativeDocxParser()
            self._docx = parser
        return parser

    def _markdown_parser(self) -> BaseParser:
        parser = self._markdown
        if parser is None:
            from lightrag.parser.markdown.parser import NativeMarkdownParser

            parser = NativeMarkdownParser()
            self._markdown = parser
        return parser

    async def parse(self, ctx: ParseContext) -> ParseResult:
        suffix = Path(ctx.file_path).suffix.lower()
        if suffix in _MARKDOWN_SUFFIXES:
            return await self._markdown_parser().parse(ctx)
        # Default to docx; its validate_source raises a clear error for any
        # other suffix the native engine was (mis)routed for.
        return await self._docx_parser().parse(ctx)
