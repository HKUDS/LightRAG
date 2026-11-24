"""Argument floors of ``chunking_by_recursive_character``.

- ``chunk_token_size < 1`` raises. It used to be floored to 1, which turned a
  misconfigured size into one chunk per token. The configuration-level minimum
  (``MIN_CHUNK_TOKEN_SIZE``) is enforced at ingress, not here, so a direct call
  may still use a small positive size.
- A negative ``chunk_overlap_token_size`` is floored to 0, and the
  ``overlap >= size`` guard compares that floored overlap.

The sibling fixed-token chunker raises on a negative overlap instead
(``tests/chunker/test_chunking.py::test_negative_overlap_raises``).
"""

from __future__ import annotations

import pytest

from lightrag.chunker import chunking_by_recursive_character
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    """1 char == 1 token; lets assertions reason in terms of input length."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char-tokenizer", _CharTokenizer())


@pytest.mark.parametrize(
    "chunk_token_size",
    [
        0,
        -1,
    ],
)
def test_non_positive_chunk_token_size_raises(chunk_token_size):
    """A non-positive size is rejected, never floored to one-token chunks.

    The overlap is 0 so the overlap guard cannot be what fires.
    """
    with pytest.raises(
        ValueError, match=rf"chunk_token_size must be >= 1, got {chunk_token_size}"
    ):
        chunking_by_recursive_character(
            _tok(), "abc", chunk_token_size=chunk_token_size, chunk_overlap_token_size=0
        )


def test_small_positive_chunk_token_size_is_allowed_on_direct_call():
    """Size 1 is the direct-call floor; ``MIN_CHUNK_TOKEN_SIZE`` is not applied."""
    chunks = chunking_by_recursive_character(
        _tok(), "abc", chunk_token_size=1, chunk_overlap_token_size=0
    )

    assert [c["content"] for c in chunks] == ["a", "b", "c"]
    assert [c["chunk_order_index"] for c in chunks] == [0, 1, 2]


@pytest.mark.parametrize("chunk_overlap_token_size", [-1, -(10**6)])
def test_negative_chunk_overlap_token_size_floors_to_zero(chunk_overlap_token_size):
    """Any negative overlap gives exactly the output of an explicit 0."""
    body = "abcdefghij"

    floored = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=5,
        chunk_overlap_token_size=chunk_overlap_token_size,
    )
    explicit_zero = chunking_by_recursive_character(
        _tok(), body, chunk_token_size=5, chunk_overlap_token_size=0
    )

    assert [c["content"] for c in floored] == ["abcde", "fghij"]
    assert floored == explicit_zero


def test_overlap_guard_compares_the_floored_overlap():
    """A negative overlap passes the guard even against size 1 (0 < 1)."""
    chunks = chunking_by_recursive_character(
        _tok(), "ab", chunk_token_size=1, chunk_overlap_token_size=-7
    )

    assert [c["content"] for c in chunks] == ["a", "b"]


def test_overlap_guard_still_fires_on_small_size():
    """The default overlap (100) against a small size raises from the guard."""
    with pytest.raises(
        ValueError,
        match=r"chunk_overlap_token_size \(100\) must be < chunk_token_size \(5\)",
    ):
        chunking_by_recursive_character(_tok(), "abc", chunk_token_size=5)
