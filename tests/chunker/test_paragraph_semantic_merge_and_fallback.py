"""Regression tests for paragraph-semantic Stage D merging and the top-level R fallback."""

import pytest

from lightrag.chunker.paragraph_semantic import (
    _glue_heading_only_blocks,
    _is_heading_only,
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


# ---------------------------------------------------------------------------
# Pre-Stage-D — body-less heading glue (forward into child / backward into prev).
# ---------------------------------------------------------------------------


def _hblock(
    content: str,
    *,
    heading: str,
    level: int,
    tokenizer: Tokenizer,
    table_chunk_role: str = "none",
) -> dict:
    """Build a block whose ``content`` keeps the markdown heading line(s).

    Unlike ``_make_block`` (heading-less ``content``), heading-only detection
    needs the ``#``-prefixed heading line preserved verbatim in ``content``.
    """
    return {
        "heading": heading,
        "parent_headings": [],
        "level": level,
        "paragraphs": [
            {"text": line, "is_table": False}
            for line in content.split("\n")
            if line.strip()
        ],
        "content": content,
        "tokens": len(tokenizer.encode(content)),
        "table_chunk_role": table_chunk_role,
        "blockids": [],
    }


@pytest.mark.offline
def test_is_heading_only_detection():
    tokenizer = _make_tokenizer()
    assert _is_heading_only(
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer)
    )
    # A glued accumulation of bare headings is still heading-only.
    assert _is_heading_only(
        _hblock("## 2.4\n\n### 2.4.1", heading="2.4", level=2, tokenizer=tokenizer)
    )
    # Heading + body is NOT heading-only.
    assert not _is_heading_only(
        _hblock("## 2.3\nbody text", heading="2.3", level=2, tokenizer=tokenizer)
    )
    # Preamble (no heading) is excluded by the heading guard.
    assert not _is_heading_only(
        _hblock("preamble text", heading="", level=1, tokenizer=tokenizer)
    )


@pytest.mark.offline
def test_heading_only_glues_forward_into_deeper_child():
    # `## 2.4` (heading-only) must bond with its deeper child `### 2.4.1`,
    # NOT get appended to the previous same-level block `## 2.3`.
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("## 2.3\n" + "a" * 40, heading="2.3", level=2, tokenizer=tokenizer),
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
        _hblock(
            "### 2.4.1\n" + "b" * 40, heading="2.4.1", level=3, tokenizer=tokenizer
        ),
    ]

    out = _glue_heading_only_blocks(
        blocks, tokenizer=tokenizer, target_max=10000, target_ideal=7500
    )

    assert len(out) == 2
    # 2.3 stays untouched — the lone heading was NOT glued onto its tail.
    assert out[0]["heading"] == "2.3"
    assert "## 2.4" not in out[0]["content"]
    # The bonded group keeps the shallower parent identity (2.4 / level 2)
    # but carries the child content.
    assert out[1]["heading"] == "2.4"
    assert out[1]["level"] == 2
    assert "## 2.4" in out[1]["content"]
    assert "### 2.4.1" in out[1]["content"]


@pytest.mark.offline
def test_heading_only_glue_respects_target_max_when_child_near_cap():
    # The child fits target_max on its own, but prepending the heading-only
    # parent line would tip the bonded block over the hard cap. The pre-pass
    # must re-split so every emitted piece stays within target_max, while the
    # parent heading still rides with the first piece (never detached).
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
        _hblock(
            "### 2.4.1\n" + "b" * 86, heading="2.4.1", level=3, tokenizer=tokenizer
        ),
    ]
    # child alone = 10 + 86 = 96 ≤ 100; bonded = 6 + 2 + 96 = 104 > 100.
    assert blocks[1]["tokens"] <= 100

    out = _glue_heading_only_blocks(
        blocks,
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=75,
        chunk_overlap_token_size=0,
    )

    assert len(out) >= 2
    assert all(b["tokens"] <= 100 for b in out), [b["tokens"] for b in out]
    # Parent heading is not detached — it leads the first emitted piece.
    assert "## 2.4" in out[0]["content"]


@pytest.mark.offline
def test_heading_only_cap_split_does_not_orphan_when_body_has_no_anchor():
    # Regression: child is near the cap and its body is ONE long paragraph
    # (> _MAX_ANCHOR_CANDIDATE_LENGTH chars), so the only anchor candidate in
    # the glued block is the child heading at index 1. The naive
    # split-the-whole-block path sliced off `[## 2.4]` alone — a heading-only
    # orphan that Stage D then re-absorbs backward, recreating the separation.
    # The prefix-aware re-split must keep the heading with real body content.
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
        _hblock(
            "### 2.4.1\n" + "b" * 110, heading="2.4.1", level=3, tokenizer=tokenizer
        ),
    ]
    # child alone = 10 + 110 = 120 (near cap); bonded = 6 + 2 + 120 = 128 > 120.
    assert blocks[1]["tokens"] <= 120

    out = _glue_heading_only_blocks(
        blocks,
        tokenizer=tokenizer,
        target_max=120,
        target_ideal=90,
        chunk_overlap_token_size=0,
    )

    assert all(b["tokens"] <= 120 for b in out), [b["tokens"] for b in out]
    # No piece is a heading-only orphan.
    assert not any(_is_heading_only(b) for b in out)
    # The heading lines ride with real body content in the first piece.
    assert "## 2.4" in out[0]["content"]
    assert "### 2.4.1" in out[0]["content"]
    assert "b" in out[0]["content"]


@pytest.mark.offline
def test_heading_only_chain_collapses_to_shallowest_identity():
    # `# 2` -> `## 2.4` -> `### 2.4.1` (body) collapses into one block whose
    # identity is the shallowest heading (level 1).
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("# 2", heading="2", level=1, tokenizer=tokenizer),
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
        _hblock(
            "### 2.4.1\n" + "c" * 30, heading="2.4.1", level=3, tokenizer=tokenizer
        ),
    ]

    out = _glue_heading_only_blocks(
        blocks, tokenizer=tokenizer, target_max=10000, target_ideal=7500
    )

    assert len(out) == 1
    assert out[0]["heading"] == "2"
    assert out[0]["level"] == 1
    content = out[0]["content"]
    assert "# 2" in content and "## 2.4" in content and "### 2.4.1" in content


@pytest.mark.offline
def test_heading_only_no_glue_when_next_not_deeper():
    # Next block is same level -> no forced forward glue; left for Stage D.
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
        _hblock("## 2.5\nbody", heading="2.5", level=2, tokenizer=tokenizer),
    ]

    out = _glue_heading_only_blocks(
        blocks, tokenizer=tokenizer, target_max=10000, target_ideal=7500
    )

    assert len(out) == 2


@pytest.mark.offline
def test_heading_only_no_glue_into_protected_table_slice():
    # A deeper next block that is a protected table slice (role != "none")
    # must not absorb the heading-only block.
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
        _hblock(
            '<table id="t" format="json">[[1]]</table>',
            heading="2.4.1",
            level=3,
            tokenizer=tokenizer,
            table_chunk_role="first",
        ),
    ]

    out = _glue_heading_only_blocks(
        blocks, tokenizer=tokenizer, target_max=10000, target_ideal=7500
    )

    assert len(out) == 2


@pytest.mark.offline
def test_heading_only_group_stays_separate_when_prev_is_saturated():
    # Rule 1 end-to-end: `## 2.3` already reached target_ideal AND the bonded
    # `2.4 + 2.4.1` group exceeds small_tail_threshold, so neither peer merging
    # nor tail absorption pulls it backward — it stays its own chunk, with 2.4
    # bonded to 2.4.1 (not to 2.3). (A group below small_tail_threshold could
    # still be tail-absorbed into a saturated 2.3, which is acceptable since it
    # would carry 2.4.1 along.)
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("## 2.3\n" + "a" * 200, heading="2.3", level=2, tokenizer=tokenizer),
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
        _hblock(
            "### 2.4.1\n" + "b" * 40, heading="2.4.1", level=3, tokenizer=tokenizer
        ),
    ]

    glued = _glue_heading_only_blocks(
        blocks, tokenizer=tokenizer, target_max=10000, target_ideal=7500
    )
    final = _merge_small_blocks(
        glued,
        tokenizer=tokenizer,
        target_max=2000,
        target_ideal=150,
        small_tail_threshold=12,
    )

    assert len(final) == 2
    assert "## 2.4" not in final[0]["content"]
    assert "## 2.4" in final[1]["content"] and "### 2.4.1" in final[1]["content"]


@pytest.mark.offline
def test_heading_only_group_backfills_into_unsaturated_prev():
    # Rule 2 end-to-end: when `## 2.3` is still below target_ideal and the
    # join fits target_max, Stage D packs 2.3 + 2.4 + 2.4.1 into one chunk.
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("## 2.3\n" + "a" * 40, heading="2.3", level=2, tokenizer=tokenizer),
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
        _hblock(
            "### 2.4.1\n" + "b" * 40, heading="2.4.1", level=3, tokenizer=tokenizer
        ),
    ]

    glued = _glue_heading_only_blocks(
        blocks, tokenizer=tokenizer, target_max=10000, target_ideal=7500
    )
    final = _merge_small_blocks(
        glued,
        tokenizer=tokenizer,
        target_max=200,
        target_ideal=150,
        small_tail_threshold=12,
    )

    assert len(final) == 1
    content = final[0]["content"]
    assert "## 2.3" in content
    assert "## 2.4" in content
    assert "### 2.4.1" in content


@pytest.mark.offline
def test_heading_only_not_glued_into_deeper_prev():
    # `## 2.4` (L2) has no deeper child after it; its previous block is the
    # DEEPER `### 2.3.9` (L3). It must NOT be pulled backward into that deeper
    # block — absorbing a shallower heading into a deeper chunk would invert the
    # hierarchy. It stays separate, left for Stage D.
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock(
            "### 2.3.9\n" + "a" * 40, heading="2.3.9", level=3, tokenizer=tokenizer
        ),
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
    ]

    out = _glue_heading_only_blocks(
        blocks, tokenizer=tokenizer, target_max=10000, target_ideal=7500
    )

    assert len(out) == 2
    assert out[0]["heading"] == "2.3.9"
    assert "## 2.4" not in out[0]["content"]
    assert out[1]["heading"] == "2.4"


@pytest.mark.offline
def test_heading_only_not_glued_into_same_level_prev():
    # The previous block `## 2.3` is the SAME level (a sibling), not deeper, so
    # the body-less `## 2.4` is not glued backward into it — that is the original
    # mis-merge. It stays standalone for Stage D to handle.
    tokenizer = _make_tokenizer()
    blocks = [
        _hblock("## 2.3\n" + "a" * 40, heading="2.3", level=2, tokenizer=tokenizer),
        _hblock("## 2.4", heading="2.4", level=2, tokenizer=tokenizer),
    ]

    out = _glue_heading_only_blocks(
        blocks, tokenizer=tokenizer, target_max=10000, target_ideal=7500
    )

    assert len(out) == 2
    assert "## 2.4" not in out[0]["content"]
    assert out[1]["heading"] == "2.4"
