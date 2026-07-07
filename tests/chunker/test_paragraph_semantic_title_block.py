"""G10 tests: smart-heading title blocks are pinned through the P chunker.

A title block is a sidecar row carrying ``is_title_block: true`` at
``level: 0``. The chunker pins it: HeadingGlue must not fold it into its
sub-document, LevelMerge (Phase A / Phase B / tail absorption) must neither
merge, absorb, nor let it absorb, and the output chunk keeps ``level 0``.
Pinning keys off the explicit flag ONLY — a bare ``level: 0`` row without
the flag (markdown prefaces today) must keep the historical behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.chunker.paragraph_semantic import (
    _glue_heading_only_blocks,
    _merge_small_blocks,
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


def test_title_block_does_not_absorb_descendants_phase_b() -> None:
    tokenizer = _make_tokenizer()
    title = _block("主标题", tokenizer=tokenizer, heading="主标题", level=0, title=True)
    child = _block(
        "子文档正文", tokenizer=tokenizer, heading="第一章", parents=["主标题"], level=1
    )
    control_parent = _block("概述", tokenizer=tokenizer, heading="概述", level=1)
    control_child = _block(
        "深层内容", tokenizer=tokenizer, heading="细节", parents=["概述"], level=2
    )

    merged = _merge_small_blocks(
        [title, child, control_parent, control_child],
        tokenizer=tokenizer,
        target_max=100,
        target_ideal=80,
        small_tail_threshold=12,
    )

    headings = [b["heading"] for b in merged]
    # Title block absorbed nothing and was absorbed by nothing.
    assert "主标题" in headings
    title_out = next(b for b in merged if b["heading"] == "主标题")
    assert title_out["content"] == "主标题"
    # The control pair (ordinary shallow+deep) did merge.
    assert not any(b["heading"] == "细节" for b in merged)


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

    assert len(chunks) == 2, chunks
    assert chunks[0]["heading"]["level"] == 0
    assert chunks[0]["heading"]["heading"] == "年度工作报告 — 副标题"
    assert chunks[0]["content"] == "年度工作报告 — 副标题"
    assert chunks[1]["heading"]["level"] == 1


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
