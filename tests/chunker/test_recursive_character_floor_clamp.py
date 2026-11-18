"""Floor-clamping of non-positive ``chunk_token_size`` / negative
``chunk_overlap_token_size`` in ``chunking_by_recursive_character``.

The function silently floors both values before building the LangChain
splitter (``resolved_chunk_size = max(int(chunk_token_size), 1)``,
``resolved_chunk_overlap = max(int(chunk_overlap_token_size), 0)``) and the
``overlap >= size`` guard added by the "reject overlap >= chunk size" fix
compares the *resolved* (clamped) values, not the raw arguments the caller
passed in -- the docstring calls this out explicitly ("once both are resolved
to their effective (floor-clamped) values").

This is reachable with real-world inputs: ``addon_params``, a direct SDK
call, or a persisted per-document chunking snapshot can all carry a
``chunk_token_size`` of ``0`` (an unset/blank env var coerced through
``int()``) or a negative ``chunk_overlap_token_size`` -- none of them pass
through the REST layer's ``_OverlapChunkParams`` validator first.

The sibling fixed-token chunker (``lightrag/chunker/token_size.py``) makes the
opposite choice for a negative overlap -- it raises
("chunk_overlap_token_size must be non-negative"), per
``tests/chunker/test_chunking.py::test_negative_overlap_raises``. Recursive
character chunking clamps instead. Both are intentional, documented
behaviours of their own function; this file pins the recursive-character side,
which `tests/chunker/test_chunker_recursive_character.py` and
`tests/chunker/test_recursive_character_bounds.py` do not cover.
"""

from __future__ import annotations

import pytest

from lightrag.chunker import chunking_by_recursive_character
from lightrag.utils import Tokenizer, TokenizerInterface


class _CharTokenizer(TokenizerInterface):
    """1 char == 1 token; lets assertions reason in terms of input length."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char-tokenizer", _CharTokenizer())


@pytest.mark.offline
@pytest.mark.parametrize("chunk_token_size", [0, -1, -5])
def test_non_positive_chunk_token_size_clamps_to_one(chunk_token_size):
    """``chunk_token_size <= 0`` floors to 1 instead of raising or crashing.

    With an explicit zero overlap the clamped size (1) stays strictly above
    the clamped overlap (0), so the "overlap >= size" guard does not fire and
    every token in the content ends up in its own chunk.
    """
    chunks = chunking_by_recursive_character(
        _tok(), "abc", chunk_token_size=chunk_token_size, chunk_overlap_token_size=0
    )

    assert [c["content"] for c in chunks] == ["a", "b", "c"]
    for index, chunk in enumerate(chunks):
        assert chunk["tokens"] == 1
        assert chunk["chunk_order_index"] == index


@pytest.mark.offline
@pytest.mark.parametrize("chunk_token_size", [0, -5])
def test_non_positive_chunk_token_size_with_default_overlap_still_raises(
    chunk_token_size,
):
    """The clamp does not defeat the overlap guard -- it feeds it.

    Leaving ``chunk_overlap_token_size`` at its default (100) while
    ``chunk_token_size`` floors to 1 means the resolved overlap (100) is still
    ``>=`` the resolved size (1), so this must raise with the *clamped*
    numbers in the message, not the caller's raw ``chunk_token_size``.
    """
    with pytest.raises(
        ValueError,
        match=r"chunk_overlap_token_size \(100\) must be < chunk_token_size \(1\)",
    ):
        chunking_by_recursive_character(
            _tok(), "abc", chunk_token_size=chunk_token_size
        )


@pytest.mark.offline
def test_negative_chunk_overlap_token_size_clamps_to_zero():
    """A negative overlap floors to 0 rather than raising.

    Unlike ``chunking_by_token_size`` (the fixed-size "Token" strategy), which
    explicitly rejects a negative ``chunk_overlap_token_size``, the recursive
    character strategy clamps it -- so the result must be byte-for-byte
    identical to passing ``chunk_overlap_token_size=0`` explicitly.
    """
    body = "abcdefghij"

    clamped = chunking_by_recursive_character(
        _tok(), body, chunk_token_size=5, chunk_overlap_token_size=-3
    )
    explicit_zero = chunking_by_recursive_character(
        _tok(), body, chunk_token_size=5, chunk_overlap_token_size=0
    )

    assert [c["content"] for c in clamped] == [c["content"] for c in explicit_zero]
    assert [c["content"] for c in clamped] == ["abcde", "fghij"]


@pytest.mark.offline
def test_negative_chunk_overlap_token_size_does_not_raise_even_deeply_negative():
    """The clamp has no floor of its own below zero -- any negative value
    collapses to the same resolved overlap of 0."""
    body = "abcdefghij"

    very_negative = chunking_by_recursive_character(
        _tok(), body, chunk_token_size=5, chunk_overlap_token_size=-(10**6)
    )
    slightly_negative = chunking_by_recursive_character(
        _tok(), body, chunk_token_size=5, chunk_overlap_token_size=-1
    )

    assert [c["content"] for c in very_negative] == [
        c["content"] for c in slightly_negative
    ]


@pytest.mark.offline
def test_both_non_positive_at_once_compares_clamped_not_raw_values():
    """``chunk_token_size=0`` and a negative overlap together must not raise.

    Naively, a caller might expect ``0`` and a negative number to compare as
    "overlap >= size" (e.g. 0 >= 0, or -1 >= 0 being false but close to the
    boundary either way). The guard actually runs on the resolved values
    (size floored to 1, overlap floored to 0), and 0 < 1, so this must
    succeed exactly like the single-sided non-positive case.
    """
    chunks = chunking_by_recursive_character(
        _tok(), "abc", chunk_token_size=0, chunk_overlap_token_size=-7
    )

    assert [c["content"] for c in chunks] == ["a", "b", "c"]
