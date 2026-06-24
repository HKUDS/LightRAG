"""Unit tests for the legacy engine adapter (lightrag.parser.legacy.parser).

Covers the worker-stage extraction contract directly: success path
(persist + archive + RAW ParseResult), the unsupported-suffix gate, the
whitespace-only extraction guard (scanned-PDF case) and the
``PDF_DECRYPT_PASSWORD`` plumbing.
"""

import pytest

import lightrag.pipeline as _pipeline
from lightrag.constants import FULL_DOCS_FORMAT_RAW
from lightrag.parser.base import ParseContext
from lightrag.parser.legacy.extractors import LegacyExtractionError
from lightrag.parser.legacy.parser import LegacyParser

pytestmark = pytest.mark.offline


class _FakeRag:
    def __init__(self):
        self.persisted = []

    async def _persist_parsed_full_docs(self, doc_id, payload):
        self.persisted.append((doc_id, payload))

    def _resolve_source_file_for_parser(
        self, file_path, *, source_file=None, parser_engine=None
    ):
        return file_path


@pytest.fixture
def archived(monkeypatch):
    """Record archive calls instead of moving files into __parsed__."""
    calls = []

    async def _record(source_path):
        calls.append(source_path)
        return source_path

    monkeypatch.setattr(_pipeline, "archive_docx_source_after_full_docs_sync", _record)
    return calls


def _ctx(rag, source_path):
    return ParseContext(rag, "doc-legacy", str(source_path), {})


async def test_legacy_parse_txt_persists_and_archives(tmp_path, archived):
    source = tmp_path / "notes.txt"
    source.write_text("plain text body", encoding="utf-8")
    rag = _FakeRag()

    result = await LegacyParser().parse(_ctx(rag, source))

    assert result.parse_format == FULL_DOCS_FORMAT_RAW
    assert result.content == "plain text body"
    assert result.parse_engine == "legacy"
    assert result.blocks_path == ""
    doc_id, payload = rag.persisted[0]
    assert doc_id == "doc-legacy"
    assert payload["content"] == "plain text body"
    assert payload["parse_format"] == FULL_DOCS_FORMAT_RAW
    assert archived == [str(source)]
    # to_dict stays byte-compatible: no spurious skip/warning keys.
    assert "parse_stage_skipped" not in result.to_dict()


async def test_legacy_parse_unsupported_suffix_raises(tmp_path, archived):
    source = tmp_path / "image.xyz"
    source.write_bytes(b"not parseable")
    rag = _FakeRag()

    with pytest.raises(ValueError, match=r"does not support \.xyz"):
        await LegacyParser().parse(_ctx(rag, source))

    assert rag.persisted == []
    assert archived == []


async def test_legacy_parse_whitespace_only_extraction_raises(
    tmp_path, archived, monkeypatch
):
    # A scanned PDF (no text layer) extracts to pure whitespace; the parser
    # must fail the doc instead of persisting an empty document.
    monkeypatch.setattr(
        "lightrag.parser.legacy.extractors.extract_text",
        lambda file_bytes, suffix, *, pdf_password=None: "\n \n\t\n",
    )
    source = tmp_path / "scanned.pdf"
    source.write_bytes(b"%PDF-fake")
    rag = _FakeRag()

    with pytest.raises(LegacyExtractionError, match="no usable text"):
        await LegacyParser().parse(_ctx(rag, source))

    assert rag.persisted == []
    assert archived == []


async def test_legacy_parse_passes_pdf_password_from_env(
    tmp_path, archived, monkeypatch
):
    seen = {}

    def _capture(file_bytes, suffix, *, pdf_password=None):
        seen["suffix"] = suffix
        seen["pdf_password"] = pdf_password
        return "decrypted text"

    monkeypatch.setattr("lightrag.parser.legacy.extractors.extract_text", _capture)
    monkeypatch.setenv("PDF_DECRYPT_PASSWORD", "s3cret")
    source = tmp_path / "locked.pdf"
    source.write_bytes(b"%PDF-fake")
    rag = _FakeRag()

    result = await LegacyParser().parse(_ctx(rag, source))

    assert seen == {"suffix": "pdf", "pdf_password": "s3cret"}
    assert result.content == "decrypted text"


async def test_extract_text_strips_null_bytes(tmp_path):
    """Null bytes from PDF extractors must be stripped to avoid PostgreSQL
    ``invalid byte sequence for encoding "UTF8": 0x00`` errors (#3308)."""
    from lightrag.parser.legacy.extractors import _strip_null_bytes

    assert _strip_null_bytes("hello\x00world") == "helloworld"
    assert _strip_null_bytes("\x00\x00") == ""
    assert _strip_null_bytes("no nulls") == "no nulls"
    assert _strip_null_bytes("") == ""
