"""Unicode escape decoding must keep valid supplementary characters."""

import pytest

from lightrag.utils import safe_unicode_decode

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (rb"\ud83d\ude00", "😀"),
        (rb"\uD834\uDD1E", "𝄞"),
        (rb"\ud840\udc00", "𠀀"),
        (rb"\u4f60\ud83d\ude00\u597d", "你😀好"),
        (rb"\u4f60\u597d", "你好"),
        ("Already decoded: 😀".encode("utf-8"), "Already decoded: 😀"),
        (rb"Keep \utils, \n and \uZZZZ", r"Keep \utils, \n and \uZZZZ"),
        (b"", ""),
    ],
)
def test_safe_unicode_decode_keeps_utf8_text(content: bytes, expected: str):
    result = safe_unicode_decode(content)

    assert result == expected
    assert result.encode("utf-8").decode("utf-8") == expected
