"""Regression tests for paragraph-semantic AnchorSplit anchor selection."""

import json

import pytest

from lightrag.chunker.paragraph_semantic import (
    _split_long_block,
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


def _write_blocks_jsonl(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "doc.blocks.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return str(path)


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
def test_split_long_block_single_paragraph_oversized_is_character_split():
    # A single oversized paragraph used to trigger the early-return at
    # ``len(paragraphs) <= 1`` and the recursive-guard's ``> 1`` clause,
    # so the function emitted one ~total-token block that silently
    # blew past target_max. With both gates relaxed, the no-anchor
    # branch's character fallback honors the cap on this case too.
    tokenizer = _make_tokenizer()
    paragraphs = [{"text": "x" * 4000}]

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

    assert len(blocks) > 1, "single oversized paragraph must be split, not kept whole"
    assert all(b["tokens"] <= 1000 for b in blocks), [b["tokens"] for b in blocks]
    # Heading hierarchy is preserved on every R-derived sub-block.
    assert all(b["heading"] == "Heading" for b in blocks)


@pytest.mark.offline
def test_split_long_block_character_fallback_keeps_configured_overlap(monkeypatch):
    tokenizer = _make_tokenizer()
    captured: dict[str, int] = {}

    def fake_chunker(
        tokenizer,
        content,
        chunk_token_size: int = 1200,
        *,
        chunk_overlap_token_size: int = 100,
        separators=None,
    ):
        captured["chunk_overlap_token_size"] = chunk_overlap_token_size
        step = max(chunk_token_size - chunk_overlap_token_size, 1)
        tokens = tokenizer.encode(content)
        chunks = []
        for start in range(0, len(tokens), step):
            piece = tokenizer.decode(tokens[start : start + chunk_token_size])
            chunks.append(
                {
                    "tokens": len(tokenizer.encode(piece)),
                    "content": piece,
                    "chunk_order_index": len(chunks),
                }
            )
        return chunks

    import lightrag.chunker.recursive_character as rc_mod

    monkeypatch.setattr(rc_mod, "chunking_by_recursive_character", fake_chunker)

    blocks = _split_long_block(
        [{"text": "x" * 260}],
        heading="Heading",
        parent_headings=[],
        level=2,
        table_chunk_role="none",
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=75,
        chunk_overlap_token_size=25,
    )

    assert captured["chunk_overlap_token_size"] == 25
    assert len(blocks) > 1
    assert blocks[0]["content"][-25:] == blocks[1]["content"][:25]


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


@pytest.mark.offline
def test_public_chunking_keeps_unsplit_heading_without_part_suffix(tmp_path):
    tokenizer = _make_tokenizer()
    blocks_path = _write_blocks_jsonl(
        tmp_path,
        [
            {
                "type": "content",
                "heading": "Heading",
                "parent_headings": [],
                "level": 2,
                "content": "short body",
            }
        ],
    )

    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        "short body",
        chunk_token_size=100,
        blocks_path=blocks_path,
    )

    assert len(chunks) == 1
    assert chunks[0]["heading"]["heading"] == "Heading"


@pytest.mark.offline
def test_public_chunking_adds_part_suffixes_for_anchor_split(tmp_path):
    tokenizer = _make_tokenizer()
    body = "\n".join(["x" * 800, "Mid anchor.", "y" * 800])
    blocks_path = _write_blocks_jsonl(
        tmp_path,
        [
            {
                "type": "content",
                "heading": "Heading",
                "parent_headings": [],
                "level": 2,
                "content": body,
            }
        ],
    )

    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        body,
        chunk_token_size=1000,
        blocks_path=blocks_path,
        chunk_overlap_token_size=0,
    )

    assert [chunk["heading"]["heading"] for chunk in chunks] == [
        "Heading [part 1]",
        "Mid anchor. [part 2]",
    ]
    assert all(
        all("[part " not in parent for parent in chunk["heading"]["parent_headings"])
        for chunk in chunks
    )


@pytest.mark.offline
def test_public_chunking_adds_part_suffixes_for_long_text_fallback(
    tmp_path, monkeypatch
):
    tokenizer = _make_tokenizer()

    def fake_chunker(
        tokenizer,
        content,
        chunk_token_size: int = 1200,
        *,
        chunk_overlap_token_size: int = 100,
        separators=None,
    ):
        tokens = tokenizer.encode(content)
        chunks = []
        for start in range(0, len(tokens), chunk_token_size):
            piece = tokenizer.decode(tokens[start : start + chunk_token_size])
            chunks.append(
                {
                    "tokens": len(tokenizer.encode(piece)),
                    "content": piece,
                    "chunk_order_index": len(chunks),
                }
            )
        return chunks

    import lightrag.chunker.recursive_character as rc_mod

    monkeypatch.setattr(rc_mod, "chunking_by_recursive_character", fake_chunker)

    body = "z" * 260
    blocks_path = _write_blocks_jsonl(
        tmp_path,
        [
            {
                "type": "content",
                "heading": "Heading",
                "parent_headings": [],
                "level": 2,
                "content": body,
            }
        ],
    )

    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        body,
        chunk_token_size=100,
        blocks_path=blocks_path,
        chunk_overlap_token_size=0,
    )

    assert [chunk["heading"]["heading"] for chunk in chunks] == [
        "Heading [part 1]",
        "Heading [part 2]",
        "Heading [part 3]",
    ]
