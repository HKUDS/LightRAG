"""Regression tests for paragraph-semantic Stage C anchor selection."""

import pytest

from lightrag.chunker.paragraph_semantic import _split_long_block
from lightrag.utils import Tokenizer, TokenizerInterface


class _CharTokenizer(TokenizerInterface):
    """1:1 character-to-token mapping — keeps math obvious in assertions."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char", tokenizer=_CharTokenizer())


@pytest.mark.offline
def test_split_long_block_short_lead_then_huge_does_not_recurse():
    # Reproduces the case where the only ≤100-char paragraph is at index 0:
    # before the fix, the anchor at idx=0 was selected, slice_paras was
    # empty, the tail was the original input, and the recursive guard
    # re-entered _split_long_block with the same arguments forever.
    tokenizer = _make_tokenizer()
    paragraphs = [
        {"text": "Short lead anchor."},  # idx 0 — short, but unusable as a divider
        {"text": "x" * 4000},  # idx 1 — huge, no anchor inside
    ]

    blocks = _split_long_block(
        paragraphs,
        heading="Heading",
        parent_headings=[],
        level=2,
        table_chunk_role="none",
        tokenizer=tokenizer,
        target_max=1000,
        target_ideal=750,
    )

    # Falls through to the "no eligible anchor" branch and now defers to
    # recursive-character splitting so ``target_max`` is honored without
    # relying on the embedding-time hard fallback (which uses a different
    # threshold).  The original recursion-guard contract still holds: the
    # function returns a finite list rather than recursing forever.
    assert len(blocks) > 1
    assert all(b["tokens"] <= 1000 for b in blocks)
    # Heading hierarchy is preserved on every R-derived sub-block.
    assert all(b["heading"] == "Heading" for b in blocks)


@pytest.mark.offline
def test_split_long_block_no_anchor_pack_accounts_for_separator():
    # The no-anchor greedy pack joins pieces with ``"\n"``, which costs
    # tokens on its own. Without debiting that separator from the buffer
    # budget, two pieces summing to exactly target_max produced a final
    # chunk of ``target_max + 1`` tokens — silently violating the cap.
    tokenizer = _make_tokenizer()
    # Two paragraphs both > _MAX_ANCHOR_CANDIDATE_LENGTH (100 chars), so
    # neither qualifies as an anchor and the no-anchor branch fires.
    # Their lengths sum exactly to ``target_max`` (101 + 101 = 202),
    # so before the fix the joined output overflowed by the "\n" token.
    paragraphs = [
        {"text": "a" * 101},
        {"text": "b" * 101},
    ]

    blocks = _split_long_block(
        paragraphs,
        heading="Heading",
        parent_headings=[],
        level=2,
        table_chunk_role="none",
        tokenizer=tokenizer,
        target_max=202,
        target_ideal=150,
    )

    assert blocks, "expected at least one sub-block"
    assert all(b["tokens"] <= 202 for b in blocks), [b["tokens"] for b in blocks]


@pytest.mark.offline
def test_split_long_block_uses_later_short_anchor():
    # Sanity check: a short paragraph at idx>0 IS still a valid divider.
    tokenizer = _make_tokenizer()
    paragraphs = [
        {"text": "x" * 1500},  # idx 0 — huge
        {"text": "Mid anchor."},  # idx 1 — short, eligible
        {"text": "y" * 1500},  # idx 2 — huge
    ]

    blocks = _split_long_block(
        paragraphs,
        heading="Heading",
        parent_headings=[],
        level=2,
        table_chunk_role="none",
        tokenizer=tokenizer,
        target_max=1000,
        target_ideal=750,
    )

    assert len(blocks) >= 2
    # Anchor paragraph becomes the heading of the post-split sub-block.
    assert any(b["heading"] == "Mid anchor." for b in blocks)
