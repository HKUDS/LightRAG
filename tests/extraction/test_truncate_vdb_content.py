"""``operate._truncate_vdb_content`` uses the safe truncate contract.

Entity/relation VDB payloads used to be clamped with a fixed heuristic safety
margin (``threshold - min(256, max(32, threshold // 16))``) and
``tokenizer.decode(tokens[:effective_limit])`` with no independent
re-verification -- unsafe for any tokenizer/content combination where
decode(encode(x)[:k]) doesn't round-trip to <= k tokens. It now delegates to
``Tokenizer.truncate_by_token_limit``, which re-encodes and verifies the
actual candidate substring.
"""

from __future__ import annotations

import pytest

from lightrag.operate import _truncate_vdb_content
from lightrag.utils import Tokenizer, TokenBudgetError, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str) -> list[int]:
        return [ord(ch) % 1000 for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char", _CharTokenizer())


def test_no_op_without_a_configured_limit_or_tokenizer():
    assert _truncate_vdb_content("hello", {}, "entity:x") == "hello"
    assert (
        _truncate_vdb_content("hello", {"embedding_token_limit": 5}, "entity:x")
        == "hello"
    )  # no tokenizer -> no-op


def test_empty_content_is_a_no_op():
    gc = {"embedding_token_limit": 5, "tokenizer": _tok()}
    assert _truncate_vdb_content("", gc, "entity:x") == ""


def test_content_within_limit_is_unchanged():
    tok = _tok()
    gc = {"embedding_token_limit": 100, "tokenizer": tok}
    content = "entity_name\nshort description"
    assert _truncate_vdb_content(content, gc, "entity:x") == content


def test_oversized_content_is_truncated_to_an_independently_verified_prefix():
    tok = _tok()
    gc = {"embedding_token_limit": 10, "tokenizer": tok}
    content = "x" * 500
    out = _truncate_vdb_content(content, gc, "entity:x")
    assert out == content[:10]
    assert len(tok.encode(out)) <= 10


def test_truncation_result_is_always_reencode_safe_even_with_bpe_growth():
    """Pin that the result is verified, not just heuristically clamped -- a
    tokenizer where a candidate's re-encoded length can exceed its char
    length must still yield a safe result."""

    class _GrowingTokenizer(TokenizerInterface):
        # Every character costs 2 tokens once content is long, to simulate a
        # tokenizer whose token density is not simply 1:1.
        def encode(self, content: str) -> list[int]:
            return [0] * (2 * len(content))

        def decode(self, tokens: list[int]) -> str:
            return "y" * (len(tokens) // 2)

    tok = Tokenizer("growing", _GrowingTokenizer())
    gc = {"embedding_token_limit": 9, "tokenizer": tok}
    out = _truncate_vdb_content("z" * 100, gc, "entity:x")
    assert len(tok.encode(out)) <= 9


def test_non_positive_limit_is_a_no_op():
    gc = {"embedding_token_limit": 0, "tokenizer": _tok()}
    content = "x" * 50
    assert _truncate_vdb_content(content, gc, "entity:x") == content


def test_impossible_budget_raises_token_budget_error():
    class _HeavyTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [0] * (5 * len(content))

        def decode(self, tokens: list[int]) -> str:
            return ""

    tok = Tokenizer("heavy", _HeavyTokenizer())
    gc = {"embedding_token_limit": 1, "tokenizer": tok}
    with pytest.raises(TokenBudgetError):
        _truncate_vdb_content("hello", gc, "entity:x")
