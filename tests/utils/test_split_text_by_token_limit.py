"""Regression tests for :func:`lightrag.utils.split_text_by_token_limit`."""

import pytest

from lightrag.utils import Tokenizer, TokenizerInterface, split_text_by_token_limit

pytestmark = pytest.mark.offline


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) % 1000 for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _tok() -> Tokenizer:
    return Tokenizer(model_name="dummy", tokenizer=_DummyTokenizer())


def test_max_tokens_zero_returns_empty_not_valueerror():
    text = "Hello world. This is a second sentence that needs splitting!"
    assert split_text_by_token_limit(text, _tok(), 0) == []


def test_max_tokens_negative_returns_empty():
    text = "Hello world. Another sentence."
    assert split_text_by_token_limit(text, _tok(), -5) == []


def test_positive_limit_still_splits_oversize_units():
    text = "abcdefghij"
    pieces = split_text_by_token_limit(text, _tok(), 4)
    assert pieces == ["abcd", "efgh", "ij"]
    assert all(len(_tok().encode(p)) <= 4 for p in pieces)


def test_empty_text_returns_empty():
    assert split_text_by_token_limit("", _tok(), 10) == []
