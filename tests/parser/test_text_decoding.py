"""BOM-aware text decoding shared by the legacy and native markdown parsers."""

import codecs

import pytest

from lightrag.parser.legacy.extractors import LegacyExtractionError, extract_text
from lightrag.parser.text_decoding import TextDecodeError, decode_text_bytes

pytestmark = pytest.mark.offline

TEXT = "中文 document\nsecond line \U0001f600"

BOM_ENCODED = {
    "utf-8-bom": codecs.BOM_UTF8 + TEXT.encode("utf-8"),
    "utf-16-le": codecs.BOM_UTF16_LE + TEXT.encode("utf-16-le"),
    "utf-16-be": codecs.BOM_UTF16_BE + TEXT.encode("utf-16-be"),
    "utf-32-le": codecs.BOM_UTF32_LE + TEXT.encode("utf-32-le"),
    "utf-32-be": codecs.BOM_UTF32_BE + TEXT.encode("utf-32-be"),
}


@pytest.mark.parametrize("data", BOM_ENCODED.values(), ids=BOM_ENCODED.keys())
def test_bom_selects_codec_and_is_dropped(data):
    assert decode_text_bytes(data) == TEXT


def test_plain_utf8_is_unchanged():
    assert decode_text_bytes(TEXT.encode("utf-8")) == TEXT


def test_inner_bom_character_is_kept_in_utf16():
    text = "first﻿second"
    assert decode_text_bytes(codecs.BOM_UTF16_LE + text.encode("utf-16-le")) == text


def test_utf32_le_is_not_mistaken_for_utf16_le():
    # FF FE 00 00 also starts with the UTF-16 LE BOM; decoded as UTF-16 it
    # would come out with a NUL after every BMP character.
    data = codecs.BOM_UTF32_LE + "ab".encode("utf-32-le")
    assert decode_text_bytes(data) == "ab"


@pytest.mark.parametrize(
    ("data", "label"),
    [
        (codecs.BOM_UTF16_LE + "ab".encode("utf-16-le") + b"x", "UTF-16"),
        (codecs.BOM_UTF16_LE + b"\x00\xd8a\x00", "UTF-16"),
        (codecs.BOM_UTF32_LE + b"\x00\x00\x11\x00", "UTF-32"),
        (codecs.BOM_UTF8 + b"\xff", "UTF-8"),
    ],
    ids=["utf16-truncated", "utf16-lone-surrogate", "utf32-out-of-range", "utf8"],
)
def test_invalid_bytes_name_the_bom_encoding(data, label):
    with pytest.raises(TextDecodeError, match=f"not valid {label} encoded text"):
        decode_text_bytes(data)


def test_bomless_non_utf8_mentions_the_bom_requirement():
    data = "中文".encode("utf-16-le")  # no BOM: not detected, must fail
    with pytest.raises(TextDecodeError) as exc_info:
        decode_text_bytes(data)
    message = str(exc_info.value)
    assert "not valid UTF-8 encoded text" in message
    assert "only with a byte order mark" in message
    assert "Please convert it to UTF-8" in message


@pytest.mark.parametrize("suffix", ["txt", "csv", "md"])
@pytest.mark.parametrize("data", BOM_ENCODED.values(), ids=BOM_ENCODED.keys())
def test_legacy_extract_text_accepts_bom_encodings(data, suffix):
    assert extract_text(data, suffix) == TEXT


@pytest.mark.parametrize("body", ["", " \r\n\t"])
@pytest.mark.parametrize(
    ("bom", "codec"),
    [
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (codecs.BOM_UTF32_LE, "utf-32-le"),
    ],
)
def test_legacy_bom_only_or_whitespace_file_is_rejected(bom, codec, body):
    with pytest.raises(LegacyExtractionError, match="no content or only whitespace"):
        extract_text(bom + body.encode(codec), "txt")


def test_legacy_decode_failure_is_a_legacy_extraction_error():
    with pytest.raises(LegacyExtractionError, match="not valid UTF-16 encoded text"):
        extract_text(codecs.BOM_UTF16_LE + b"\x00\xd8a\x00", "txt")
