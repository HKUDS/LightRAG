"""Routing + registry wiring for the native markdown engine.

``parser_rules=""`` is passed explicitly so these assertions are independent of
any ambient ``LIGHTRAG_PARSER`` in the dev environment.
"""

from __future__ import annotations

import asyncio

from lightrag.parser import registry
from lightrag.parser.native_dispatch import NativeParser
from lightrag.parser.routing import resolve_file_parser_directives


def _engine(path: str, rules: str = "") -> str:
    return resolve_file_parser_directives(path, parser_rules=rules)[0]


def test_native_declares_markdown_suffixes():
    assert registry.suffix_capabilities("native") == frozenset(
        {"docx", "md", "textpack"}
    )


def test_textpack_and_md_in_upload_allowlist():
    suffixes = registry.available_engine_suffixes()
    assert "textpack" in suffixes
    assert "md" in suffixes


def test_get_parser_native_is_dispatcher():
    parser = registry.get_parser("native")
    assert type(parser).__name__ == "NativeParser"


def test_textpack_defaults_to_native_without_hint():
    assert _engine("note.textpack") == "native"


def test_md_defaults_to_legacy_opt_in_for_native():
    assert _engine("doc.md") == "legacy"
    assert _engine("doc.[native].md") == "native"
    assert _engine("doc.md", rules="md:native") == "native"


def test_docx_default_unchanged():
    assert _engine("report.docx") == "legacy"


def test_textpack_falls_back_to_native_when_rule_engine_cannot_handle_it():
    # legacy cannot parse .textpack; the per-suffix default still wins.
    assert _engine("note.textpack", rules="textpack:legacy") == "native"


class _SentinelParser:
    def __init__(self, name: str) -> None:
        self.name = name

    async def parse(self, ctx):
        return self.name


class _Ctx:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path


def _dispatch(file_path: str) -> str:
    dispatcher = NativeParser()
    dispatcher._docx = _SentinelParser("docx")
    dispatcher._markdown = _SentinelParser("markdown")
    return asyncio.run(dispatcher.parse(_Ctx(file_path)))


def test_dispatcher_routes_by_suffix():
    assert _dispatch("a.md") == "markdown"
    assert _dispatch("a.textpack") == "markdown"
    assert _dispatch("a.[native-iet].md") == "markdown"
    assert _dispatch("a.docx") == "docx"
    assert _dispatch("report.[native].docx") == "docx"
