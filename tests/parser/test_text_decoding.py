"""BOM-aware text decoding shared by the legacy and native markdown parsers."""

import codecs

import pytest

from lightrag.parser.legacy.extractors import extract_text
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


@pytest.mark.parametrize(
    ("data", "label"),
    [
        (codecs.BOM_UTF16_LE + b"\x00\xd8a\x00", "UTF-16"),
        (codecs.BOM_UTF32_LE + b"\x00\x00\x11\x00", "UTF-32"),
    ],
    ids=["utf16-lone-surrogate", "utf32-out-of-range"],
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


def test_legacy_extract_text_accepts_a_bom_encoding():
    # The codec sweep is test_bom_selects_codec_and_is_dropped, on the shared
    # decode_text_bytes; this is the legacy extractor's wiring to it, and the
    # only legacy-entry case beyond UTF-8.
    assert extract_text(BOM_ENCODED["utf-16-le"], "txt") == TEXT
