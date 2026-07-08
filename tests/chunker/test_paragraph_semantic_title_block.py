"""G10 tests: smart-heading title blocks through the P chunker.

A title block is a sidecar row carrying ``is_title_block: true`` at
``level: 0``. The chunker keeps it at ``level 0`` and gives it a MINIMAL pin:
HeadingGlue must not fold it into its sub-document, and Phase A same-level
merging / tail absorption must never fuse two title blocks or cross a title
boundary. It is NOT frozen against Phase B cross-level absorption, though: a
level-0 title block MAY pull its level-1 descendants in (shallower absorbs
deeper), and the absorbed result stays pinned so it never then merges with an
adjacent title. Pinning keys off the explicit flag ONLY — a bare ``level: 0``
row without the flag (markdown prefaces today) must keep the historical
behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.chunker.paragraph_semantic import (
    _expand_block_with_table_splits,
    _glue_heading_only_blocks,
    _merge_small_blocks,
    _split_long_block,
    chunking_by_paragraph_semantic,
)
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char", tokenizer=_CharTokenizer())


def _block(
    text: str,
    *,
    tokenizer: Tokenizer,
    heading: str = "H",
    parents: list[str] | None = None,
    level: int = 1,
    title: bool = False,
) -> dict:
    return {
        "heading": heading,
        "parent_headings": list(parents or []),
        "level": level,
        "paragraphs": [{"text": text, "is_table": False}],
        "content": text,
        "tokens": len(tokenizer.encode(text)),
        "table_chunk_role": "none",
        "is_title_block": title,
    }


# ---------------------------------------------------------------------------
# G10-1: HeadingGlue must not fold a title block into its child
# ---------------------------------------------------------------------------


def test_title_block_not_glued_into_child() -> None:
    tokenizer = _make_tokenizer()
    # Content shaped like a pure heading line — the exact form
    # _is_heading_only() would glue if the pin were missing.
    title = _block(
        "# 公司年度报告",
        tokenizer=tokenizer,
        heading="公司年度报告",
        level=0,
        title=True,
    )
    child = _block(
        "正文内容开始。" * 10,
        tokenizer=tokenizer,
        heading="第一章",
        parents=["公司年度报告"],
        level=1,
    )
    control_parent = _block("# 第二章", tokenizer=tokenizer, heading="第二章", level=1)
    control_child = _block(
        "第二章正文。" * 10,
        tokenizer=tokenizer,
        heading="小节",
        parents=["第二章"],
        level=2,
    )

    out = _glue_heading_only_blocks(
        [title, child, control_parent, control_child],
        tokenizer=tokenizer,
        target_max=1000,
        target_ideal=750,
    )

    # Title stays standalone; the control heading-only block glues as before.
    assert out[0]["heading"] == "公司年度报告"
    assert out[0]["content"] == "# 公司年度报告"
    assert len(out) == 3
    assert out[2]["content"].startswith("# 第二章\n\n")


# ---------------------------------------------------------------------------
# G10-2: LevelMerge never merges / absorbs a title block
# ---------------------------------------------------------------------------


def test_adjacent_title_blocks_do_not_merge_phase_a() -> None:
    tokenizer = _make_tokenizer()
    # Two spliced articles: both title blocks are level 0 with no parents —
    # exactly the shape Phase A would merge without the pin.
    blocks = [
        _block(
            "文章一标题", tokenizer=tokenizer, heading="文章一标题", level=0, title=True
        ),
        _block(
            "文章二标题", tokenizer=tokenizer, heading="文章二标题", level=0, title=True
        ),
    ]
    control = [
        _block("aaa", tokenizer=tokenizer, heading="A"),
        _block("bbb", tokenizer=tokenizer, heading="A"),
    ]

    merged = _merge_small_blocks(
        blocks,
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )
    merged_control = _merge_small_blocks(
        control,
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )

    assert len(merged) == 2  # pinned: no merge
    assert len(merged_control) == 1  # same shape without the flag merges


def test_title_block_absorbs_descendant_phase_b() -> None:
    """A level-0 title block IS the shallow absorber in Phase B: it pulls its
    level-1 descendant in (shallower-absorbs-deeper), and the merged result
    stays pinned/level-0 so it never later fuses with a sibling title."""
    tokenizer = _make_tokenizer()
    title = _block("主标题", tokenizer=tokenizer, heading="主标题", level=0, title=True)
    child = _block(
        "子文档正文", tokenizer=tokenizer, heading="第一章", parents=["主标题"], level=1
    )

    merged = _merge_small_blocks(
        [title, child],
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )

    assert len(merged) == 1
    title_out = merged[0]
    assert title_out["heading"] == "主标题"
    assert title_out["level"] == 0
    assert "子文档正文" in title_out["content"]
    assert title_out.get("is_title_block") is True


def test_absorbed_title_stays_pinned_against_adjacent_title() -> None:
    """After absorbing its descendant a title block keeps ``is_title_block`` —
    so two title blocks (spliced sub-documents) never fuse under Phase A even
    once each has pulled its own child in."""
    tokenizer = _make_tokenizer()
    title1 = _block(
        "标题一", tokenizer=tokenizer, heading="标题一", level=0, title=True
    )
    child1 = _block(
        "正文一", tokenizer=tokenizer, heading="节一", parents=["标题一"], level=1
    )
    title2 = _block(
        "标题二", tokenizer=tokenizer, heading="标题二", level=0, title=True
    )

    merged = _merge_small_blocks(
        [title1, child1, title2],
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )

    assert [b["heading"] for b in merged] == ["标题一", "标题二"]
    assert "正文一" in merged[0]["content"]
    assert merged[0].get("is_title_block") is True
    assert merged[1]["content"] == "标题二"
    assert merged[1].get("is_title_block") is True


def test_title_block_does_not_absorb_non_descendant_phase_b() -> None:
    """Phase B still gates on ``_is_descendant``: a level-1 block that is NOT
    under the title (different parent chain) is not pulled in."""
    tokenizer = _make_tokenizer()
    title = _block("主标题", tokenizer=tokenizer, heading="主标题", level=0, title=True)
    # parent chain does NOT start with the title's heading → not a descendant.
    stranger = _block(
        "别处正文", tokenizer=tokenizer, heading="别节", parents=["另一主标题"], level=1
    )

    merged = _merge_small_blocks(
        [title, stranger],
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )

    assert len(merged) == 2
    title_out = next(b for b in merged if b["heading"] == "主标题")
    assert title_out["content"] == "主标题"
    assert title_out.get("is_title_block") is True


def test_title_block_stops_tail_absorption_run() -> None:
    tokenizer = _make_tokenizer()
    big = _block("x" * 80, tokenizer=tokenizer, heading="大块", level=0)
    title = _block("标题块", tokenizer=tokenizer, heading="标题块", level=0, title=True)
    tail = _block("尾巴", tokenizer=tokenizer, heading="尾巴", level=0)

    merged = _merge_small_blocks(
        [big, title, tail],
        tokenizer=tokenizer,
        target_max=200,
        target_ideal=60,
        small_tail_threshold=50,
    )

    # The run breaks at the pinned title block: nothing was absorbed into
    # ``big`` across it.
    title_out = next(b for b in merged if b["heading"] == "标题块")
    assert title_out["content"] == "标题块"
    big_out = next(b for b in merged if b["heading"] == "大块")
    assert "标题块" not in big_out["content"]


# ---------------------------------------------------------------------------
# G10-3: end-to-end — sidecar row flag → pinned chunk at level 0
# ---------------------------------------------------------------------------


def _write_blocks_jsonl(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "doc.blocks.jsonl"
    lines = [json.dumps({"type": "meta", "doc_title": "t"}, ensure_ascii=False)]
    lines += [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_sidecar_flag_survives_to_chunk_schema(tmp_path) -> None:
    tokenizer = _make_tokenizer()
    rows = [
        {
            "type": "content",
            "blockid": "b1",
            "content": "年度工作报告 — 副标题",
            "heading": "年度工作报告 — 副标题",
            "parent_headings": [],
            "level": 0,
            "is_title_block": True,
        },
        {
            "type": "content",
            "blockid": "b2",
            "content": "# 第一章\n正文内容。",
            "heading": "第一章",
            "parent_headings": ["年度工作报告 — 副标题"],
            "level": 1,
        },
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        "unused fallback",
        chunk_token_size=2000,
        blocks_path=blocks_path,
    )

    # The title block (level 0) absorbs its small level-1 child (Phase B) into a
    # single chunk that KEEPS level 0 — the flag survived the chunker. Contrast
    # the bare-level-0 preface below, which coerces to level 1 instead.
    assert len(chunks) == 1, chunks
    assert chunks[0]["heading"]["level"] == 0
    assert chunks[0]["heading"]["heading"] == "年度工作报告 — 副标题"
    assert "年度工作报告 — 副标题" in chunks[0]["content"]
    assert "第一章" in chunks[0]["content"]
    assert chunks[0]["sidecar"]["refs"] == [
        {"type": "block", "id": "b1"},
        {"type": "block", "id": "b2"},
    ]


# ---------------------------------------------------------------------------
# G10-4: markdown preface (level 0 WITHOUT the flag) keeps legacy behavior
# ---------------------------------------------------------------------------


def test_bare_level_zero_without_flag_keeps_legacy_coercion(tmp_path) -> None:
    tokenizer = _make_tokenizer()
    rows = [
        {
            "type": "content",
            "blockid": "b1",
            "content": "preface text before any heading",
            "heading": "",
            "parent_headings": [],
            "level": 0,
        },
        {
            "type": "content",
            "blockid": "b2",
            "content": "# Chapter\nbody",
            "heading": "Chapter",
            "parent_headings": [],
            "level": 1,
        },
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        "unused fallback",
        chunk_token_size=2000,
        blocks_path=blocks_path,
    )

    # Legacy: level 0 coerces to 1, both tiny blocks merge into one chunk.
    assert len(chunks) == 1, chunks
    assert chunks[0]["heading"]["level"] == 1
    assert "preface text" in chunks[0]["content"]
    assert "# Chapter" in chunks[0]["content"]


# ---------------------------------------------------------------------------
# G10-5: title + long intro + level-1 child — absorption is size-gated
# ---------------------------------------------------------------------------


def test_title_with_long_intro_absorbs_child_only_within_cap() -> None:
    """A title block already carrying a long intro (parse-time merge) absorbs a
    following level-1 child only when the combined size stays within the cap;
    an over-cap child is left as a separate block."""
    tokenizer = _make_tokenizer()
    intro = "封面主标题\n" + "引言" * 20  # sizable but under target_max=100
    title = _block(intro, tokenizer=tokenizer, heading="主标题", level=0, title=True)
    small_child = _block(
        "小节正文", tokenizer=tokenizer, heading="第一节", parents=["主标题"], level=1
    )
    big_child = _block(
        "巨" * 200, tokenizer=tokenizer, heading="第二节", parents=["主标题"], level=1
    )

    within = _merge_small_blocks(
        [title, small_child],
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )
    assert len(within) == 1
    assert "小节正文" in within[0]["content"]
    assert within[0].get("is_title_block") is True

    over = _merge_small_blocks(
        [title, big_child],
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )
    assert len(over) == 2
    title_out = next(b for b in over if b.get("is_title_block"))
    assert "巨" not in title_out["content"]


# ---------------------------------------------------------------------------
# G10-6: over-cap title split — only the first fragment keeps the pin, and the
# title boundary is never crossed by a neighbouring title block.
# ---------------------------------------------------------------------------


def _title_paras(*texts: str) -> list[dict]:
    return [{"text": t, "is_table": False} for t in texts]


def test_oversized_title_anchor_split_pins_only_first_fragment() -> None:
    tokenizer = _make_tokenizer()
    # Short body paras => anchor-split path.
    paras = _title_paras("封面主标题", *[f"第{i}段正文内容" for i in range(9)])
    out = _split_long_block(
        paras,
        "主标题",
        [],
        0,
        "none",
        tokenizer=tokenizer,
        target_max=20,
        target_ideal=15,
        is_title_block=True,
    )
    assert len(out) > 1
    assert out[0].get("is_title_block") is True
    assert "封面主标题" in out[0]["content"]
    assert all(not b.get("is_title_block") for b in out[1:])


def test_oversized_title_no_anchor_split_pins_only_first_fragment() -> None:
    tokenizer = _make_tokenizer()
    # One dense paragraph (> _MAX_ANCHOR_CANDIDATE_LENGTH) => no-anchor path.
    paras = _title_paras("封面主标题", "正" * 200)
    out = _split_long_block(
        paras,
        "主标题",
        [],
        0,
        "none",
        tokenizer=tokenizer,
        target_max=50,
        target_ideal=40,
        is_title_block=True,
    )
    assert len(out) > 1
    assert out[0].get("is_title_block") is True
    assert all(not b.get("is_title_block") for b in out[1:])


def test_oversized_title_boundary_not_crossed(tmp_path) -> None:
    """Two huge back-to-back title blocks: after size-splitting, no chunk mixes
    content across the original title boundary (the pinned first fragment of
    each title acts as the separator)."""
    tokenizer = _make_tokenizer()
    rows = [
        {
            "type": "content",
            "blockid": "b1",
            "content": "甲" * 300,
            "heading": "标题一",
            "parent_headings": [],
            "level": 0,
            "is_title_block": True,
        },
        {
            "type": "content",
            "blockid": "b2",
            "content": "乙" * 300,
            "heading": "标题二",
            "parent_headings": [],
            "level": 0,
            "is_title_block": True,
        },
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        "unused fallback",
        chunk_token_size=50,
        blocks_path=blocks_path,
    )

    assert len(chunks) > 2  # both titles split into several fragments
    for chunk in chunks:
        content = chunk["content"]
        assert not ("甲" in content and "乙" in content), (
            f"chunk crossed the title boundary: {content!r}"
        )


def _oversized_table_text(num_rows: int, payload: int) -> str:
    rows = [[f"r{idx}-" + "x" * payload] for idx in range(num_rows)]
    return f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'


def test_table_split_title_pins_only_first_fragment() -> None:
    """A title block whose body carries an oversized table loses the whole-block
    pin during TableRowSplit, but the FIRST emitted fragment (the cover) keeps
    it so the title boundary is still protected."""
    tokenizer = _make_tokenizer()
    block = {
        "heading": "主标题",
        "parent_headings": [],
        "level": 0,
        "is_title_block": True,
        "paragraphs": [
            {"text": "封面主标题", "is_table": False},
            {"text": _oversized_table_text(num_rows=6, payload=200), "is_table": True},
            {"text": "尾段正文", "is_table": False},
        ],
    }

    out = _expand_block_with_table_splits(
        block,
        tokenizer=tokenizer,
        table_max=400,
        table_ideal=300,
        table_min_last=128,
    )

    assert len(out) > 1
    assert out[0].get("is_title_block") is True
    assert "封面主标题" in out[0]["content"]
    assert all(not b.get("is_title_block") for b in out[1:])
