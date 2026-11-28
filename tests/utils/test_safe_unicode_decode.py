"""Unicode escape decoding must keep valid supplementary characters."""

import pytest

from lightrag.utils import StreamingUnicodeDecoder, safe_unicode_decode

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (rb"\ud83d\ude00", "😀"),
        (rb"\uD834\uDD1E", "𝄞"),
        (rb"\u4f60\ud83d\ude00\u597d", "你😀好"),
        ("Already decoded: 😀".encode("utf-8"), "Already decoded: 😀"),
        (rb"Keep \utils, \n and \uZZZZ", r"Keep \utils, \n and \uZZZZ"),
        (b"", ""),
        # A surrogate with no partner in its run cannot be encoded: dropped.
        (rb"prefix \udc00 suffix", "prefix  suffix"),
        (rb"\ud83d\ud83d\ude00", "😀"),
        (rb"\ude00\ud83d", ""),
    ],
)
def test_safe_unicode_decode_keeps_utf8_text(content: bytes, expected: str):
    result = safe_unicode_decode(content)

    assert result == expected
    assert result.encode("utf-8").decode("utf-8") == expected


@pytest.mark.parametrize(
    ("chunks", "expected_outputs"),
    [
        # A pair split between chunks is emitted whole with the second chunk.
        ([r"smile \ud83d", r"\ude00 end"], ["smile ", "😀 end", ""]),
        # An escape cut before its last hex digit is completed by the next one.
        ([r"a\u00", r"41b"], ["a", "Ab", ""]),
        (["a\\", "u0041"], ["a", "A", ""]),
        # Held text the stream never completes: decoded as is on flush.
        ([r"cut \u00"], ["cut ", r"\u00"]),
        ([r"lone \ud83d"], ["lone ", ""]),
        # A chunk without a backslash passes through untouched.
        (["plain", " text"], ["plain", " text", ""]),
    ],
)
def test_streaming_decoder_holds_back_an_open_escape(chunks, expected_outputs):
    decoder = StreamingUnicodeDecoder()

    outputs = [decoder.feed(chunk) for chunk in chunks] + [decoder.flush()]

    assert outputs == expected_outputs


@pytest.mark.parametrize(
    "text",
    [
        r"x\ud83d\ude00y\u4f60\uD834\uDD1E",
        r"\ud83d\ud83d\ude00\\u\u00zz\ude00",
        r"tail \ud83d\u00",
    ],
)
def test_streaming_decoder_matches_whole_text_decode_for_every_split(text):
    expected = safe_unicode_decode(text.encode("utf-8"))
    mismatches = []
    for first in range(len(text) + 1):
        for second in range(first, len(text) + 1):
            decoder = StreamingUnicodeDecoder()
            parts = (text[:first], text[first:second], text[second:])
            joined = "".join(decoder.feed(part) for part in parts) + decoder.flush()
            if joined != expected:
                mismatches.append((first, second, joined))

    assert mismatches == []
