"""Regression tests for paragraph-semantic Stage D merging and the top-level R fallback."""

import pytest

from lightrag.chunker.paragraph_semantic import (
    _merge_small_blocks,
    chunking_by_paragraph_semantic,
)
from lightrag.utils import Tokenizer, TokenizerInterface


class _CharTokenizer(TokenizerInterface):
    """1:1 character-to-token mapping — keeps math obvious in assertions."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char", tokenizer=_CharTokenizer())


def _make_block(text: str, *, tokenizer: Tokenizer, level: int = 1) -> dict:
    return {
        "heading": "H",
        "parent_headings": [],
        "level": level,
        "paragraphs": [{"text": text, "is_table": False}],
        "content": text,
        "tokens": len(tokenizer.encode(text)),
        "table_chunk_role": "none",
    }


@pytest.mark.offline
def test_tail_absorption_rejects_when_separator_pushes_over_cap():
    # Tail absorption joins blocks with ``"\n\n"`` but the original
    # predicate only summed per-block tokens. With cur=99 and tail=1
    # the raw sum equals target_max=100, but the actual joined
    # ``"x"*99 + "\n\n" + "y"*1`` measures 102 tokens — the absorbed
    # block silently overflowed before the fix re-measured the joined
    # content.
    tokenizer = _make_tokenizer()
    blocks = [
        _make_block("x" * 99, tokenizer=tokenizer),
        _make_block("y" * 1, tokenizer=tokenizer),
    ]

    merged = _merge_small_blocks(
        blocks,
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )

    assert all(b["tokens"] <= 100 for b in merged), [b["tokens"] for b in merged]


@pytest.mark.offline
def test_tail_absorption_still_fires_when_joined_size_fits():
    # Sanity check: when the joined content (including separators)
    # genuinely fits target_max, absorption still happens. cur=80 +
    # "\n\n" (2 tokens) + tail=1 = 83 ≤ 100.
    tokenizer = _make_tokenizer()
    blocks = [
        _make_block("x" * 80, tokenizer=tokenizer),
        _make_block("y" * 1, tokenizer=tokenizer),
    ]

    merged = _merge_small_blocks(
        blocks,
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )

    assert len(merged) == 1
    assert merged[0]["tokens"] == 83
    assert merged[0]["content"] == "x" * 80 + "\n\n" + "y" * 1


@pytest.mark.offline
def test_paragraph_semantic_fallback_passes_configured_recursive_overlap(monkeypatch):
    # When ``blocks_path`` is missing, paragraph-semantic chunking
    # delegates to ``chunking_by_recursive_character``. P now permits
    # overlap for long text under one JSONL row, so the fallback must
    # pass through the configured overlap rather than forcing zero.
    captured: dict[str, object] = {}

    def fake_chunker(
        tokenizer,
        content,
        chunk_token_size: int = 1200,
        *,
        chunk_overlap_token_size: int = 100,
        separators=None,
    ):
        captured["chunk_overlap_token_size"] = chunk_overlap_token_size
        captured["chunk_token_size"] = chunk_token_size
        return [
            {
                "tokens": len(tokenizer.encode(content)),
                "content": content,
                "chunk_order_index": 0,
            }
        ]

    import lightrag.chunker.recursive_character as rc_mod

    monkeypatch.setattr(rc_mod, "chunking_by_recursive_character", fake_chunker)

    tokenizer = _make_tokenizer()
    chunking_by_paragraph_semantic(
        tokenizer,
        "fallback corpus",
        chunk_token_size=500,
        blocks_path=None,
        chunk_overlap_token_size=37,
    )

    assert (
        captured.get("chunk_overlap_token_size") == 37
    ), "P→R fallback must pass the configured chunk_overlap_token_size"
    assert captured.get("chunk_token_size") == 500
