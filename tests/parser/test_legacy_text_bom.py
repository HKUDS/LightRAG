"""A UTF-8 signature is encoding metadata, not document content."""

import codecs

import pytest

from lightrag.parser.legacy.extractors import LegacyExtractionError, extract_text

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("suffix", ["txt", "md", "csv", ".TXT"])
def test_utf8_bom_is_not_included_in_extracted_text(suffix):
    text = "中文 document\nsecond line"
    assert extract_text(codecs.BOM_UTF8 + text.encode(), suffix) == text


@pytest.mark.parametrize("content", [b"", b" \r\n\t"])
def test_bom_only_or_whitespace_file_is_rejected(content):
    with pytest.raises(LegacyExtractionError, match="no content or only whitespace"):
        extract_text(codecs.BOM_UTF8 + content, "txt")


def test_inner_bom_character_is_preserved():
    text = "first\ufeffsecond"
    assert extract_text(text.encode(), "txt") == text


def test_invalid_utf8_remains_an_error():
    with pytest.raises(LegacyExtractionError, match="not valid UTF-8"):
        extract_text(codecs.BOM_UTF8 + b"\xff", "txt")
