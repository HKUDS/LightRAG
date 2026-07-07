"""G8 tests: heading merge, strong-body sweep with CB2, skeleton correction,
smoothing and clamping (§2.2.7 / §2.2.8)."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading.heading_flow import (
    HeadingDecision,
    align_numbering_series,
    clamp_deep_levels,
    correct_numbering_skeleton,
    demote_strong_body_headings,
    merge_split_headings,
)
from lightrag.parser.docx.smart_heading.style_key import classify_numbering

pytestmark = pytest.mark.offline


def _d(
    text: str,
    level: int,
    *,
    idx: int,
    size: float = 14.0,
    numbered: bool = False,
    outline: int | None = None,
    anchored: bool = False,
) -> HeadingDecision:
    return HeadingDecision(
        record_index=idx,
        text=text,
        is_heading=True,
        level=level,
        font_size_pt=size,
        outline_level=outline,
        anchored=anchored,
        numbering=classify_numbering(text) if numbered else None,
    )


def _stub_strong_body(text: str) -> str | None:
    stripped = text.strip()
    if stripped.endswith(("。", "？", "！")) or len(stripped) > 60:
        return "strong_body_stub"
    return None


# ---------------------------------------------------------------------------
# merge (G8-1 / G8-2 / G8-3)
# ---------------------------------------------------------------------------


def _records(n: int, empty_at: set[int] = frozenset()) -> list[ParagraphRecord]:
    return [
        ParagraphRecord(kind="empty_para" if i in empty_at else "para", text=f"r{i}")
        for i in range(n)
    ]


def test_adjacent_same_level_headings_merge_across_one_blank() -> None:
    """G8-2: same level + same size, one blank between → merged; a numbered
    heading is never absorbed."""
    records = _records(6, empty_at={1})
    ds = [
        _d("中华人民共和国", 2, idx=0),
        _d("某某管理办法", 2, idx=2),
        _d("一、总则", 2, idx=3, numbered=True),  # numbered: never absorbed
        _d("正文块", 3, idx=4),
    ]
    warnings: dict = {}
    out = merge_split_headings(ds, records, warnings=warnings)
    texts = [d.text for d in out]
    assert "中华人民共和国某某管理办法" in texts  # CJK join, no space
    assert "一、总则" in texts
    assert warnings["smart_heading_merges"] == 1
    merged = next(d for d in out if "管理办法" in d.text)
    assert merged.member_indices == (0, 2)


def test_merge_respects_four_line_cap() -> None:
    """G8-3: a 5-line "heading" chain stops merging at 4 lines."""
    records = _records(5)
    ds = [_d(f"标题行{i}", 2, idx=i) for i in range(5)]
    out = merge_split_headings(ds, records, warnings={})
    assert [d.text.count("\n") + d.text.count("标题行") for d in out]
    first = out[0]
    assert first.text.count("标题行") == 4  # capped at 4 lines
    assert len(out) == 2  # the 5th line stays standalone


def test_merge_requires_same_size_and_level() -> None:
    records = _records(4)
    ds = [
        _d("大字号行", 2, idx=0, size=16.0),
        _d("小字号行", 2, idx=1, size=14.0),  # size differs → no merge
        _d("同层同字号甲", 3, idx=2, size=14.0),
        _d("同层同字号乙", 3, idx=3, size=14.0),
    ]
    out = merge_split_headings(ds, records, warnings={})
    assert [d.text for d in out][:2] == ["大字号行", "小字号行"]
    assert any(d.text == "同层同字号甲同层同字号乙" for d in out)


def test_softbreak_lines_count_toward_cap() -> None:
    """G8-1: a heading already holding soft-break lines merges within cap."""
    records = _records(2)
    ds = [
        _d("第一行\n第二行\n第三行", 2, idx=0),  # 3 lines
        _d("第四行", 2, idx=1),
    ]
    out = merge_split_headings(ds, records, warnings={})
    assert len(out) == 1
    assert out[0].text == "第一行\n第二行\n第三行第四行"


# ---------------------------------------------------------------------------
# strong-body sweep + CB2 (G8-4)
# ---------------------------------------------------------------------------


def test_demotion_propagates_to_series() -> None:
    """≥20% of a series hit + <50% outlined → the WHOLE series demotes."""
    cn = "一二三四五"
    ds = []
    for i in range(5):
        tail = "结尾带句号的编号标题。" if i < 2 else "正常编号标题"
        ds.append(_d(f"{cn[i]}、{tail}", 3, idx=i, numbered=True))
    warnings: dict = {}
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    assert all(not d.is_heading for d in ds)  # 2/5 = 40% ≥ 20% → propagate
    assert "smart_cb2_propagation_stopped" not in warnings
    assert warnings["smart_cb2_propagations"] == 1  # A14 metric


def test_cb2_low_hit_share_stops_propagation() -> None:
    """G8-4: 1/10 hits (10% < 20%) → only the hit demotes + warning."""
    cn = "一二三四五六七八九十"
    ds = []
    for i in range(10):
        tail = "结尾带句号。" if i == 0 else "正常标题"
        ds.append(_d(f"{cn[i]}、{tail}", 3, idx=i, numbered=True))
    warnings: dict = {}
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    assert not ds[0].is_heading
    assert all(d.is_heading for d in ds[1:])
    assert warnings["smart_cb2_propagation_stopped"] == 1


def test_cb2_outlined_series_stops_propagation() -> None:
    """G8-4: ≥50% outlined members block propagation even at high hit share."""
    cn = "一二三四"
    ds = []
    for i in range(4):
        tail = "结尾带句号。" if i < 2 else "正常标题"
        ds.append(_d(f"{cn[i]}、{tail}", 3, idx=i, numbered=True, outline=2))
    warnings: dict = {}
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    assert not ds[0].is_heading and not ds[1].is_heading
    assert ds[2].is_heading and ds[3].is_heading
    assert warnings["smart_cb2_propagation_stopped"] == 1


# ---------------------------------------------------------------------------
# skeleton correction (G8-7 / G8-8 / G8-9 / G8-10 / G8-13)
# ---------------------------------------------------------------------------


def test_skeleton_nests_cnnum_under_cnparent_scope() -> None:
    """G8-7: 一、(10.5pt) holds （一）(12pt) — size leveled them inverted;
    the nesting evidence pushes （一） below 一、 and entrains deeper
    unnumbered headings behind it."""
    ds = [
        _d("一、总纲", 3, idx=0, numbered=True),
        _d("（一）分项甲", 2, idx=1, numbered=True),  # inverted by size
        _d("无编号深标题", 3, idx=2),  # deeper than （一）'s snapshot → entrains
        _d("（二）分项乙", 2, idx=3, numbered=True),
        _d("二、次纲", 3, idx=4, numbered=True),
        _d("（一）分项丙", 2, idx=5, numbered=True),
    ]
    warnings: dict = {}
    audit = correct_numbering_skeleton(ds, warnings=warnings)
    parens = [d for d in ds if d.text.startswith("（")]
    assert all(d.level == 4 for d in parens)  # pushed to 一、+1
    assert ds[2].level == 5  # entrained (+2, was deeper than snapshot 2)
    assert any(a["rule"] == "skeleton_entrain" for a in audit)
    # CnNum itself untouched
    assert ds[0].level == 3 and ds[4].level == 3


def test_skeleton_unit_semantics_chapter_over_clause() -> None:
    """G8-8: 法规 — 条 runs through 章 without resetting; the unit hierarchy
    (章 > 条) provides the (ii-c) evidence, pushing 条 below 章."""
    ds = [
        _d("第一章 总则", 4, idx=0, numbered=True),  # 10.5pt sized deep
        _d("第一条 目的", 3, idx=1, numbered=True),  # 12pt sized shallow
        _d("第二条 适用范围", 3, idx=2, numbered=True),
        _d("第二章 管理", 4, idx=3, numbered=True),
        _d("第三条 职责", 3, idx=4, numbered=True),  # continuous numbering
    ]
    correct_numbering_skeleton(ds, warnings={})
    clauses = [d for d in ds if d.numbering.unit == "条"]
    chapters = [d for d in ds if d.numbering.unit == "章"]
    assert all(c.level == 4 for c in chapters)
    assert all(c.level == 5 for c in clauses)  # 章 + 1


def test_skeleton_no_edge_without_alternating_containment() -> None:
    """G8-9: a flat body list 1. 2. 3. inside ONE scope (no alternation)
    builds no nesting edge — nothing moves."""
    ds = [
        _d("一、唯一父级", 2, idx=0, numbered=True),
        _d("1. 平级列表甲", 3, idx=1, numbered=True),
        _d("2. 平级列表乙", 3, idx=2, numbered=True),
        _d("3. 平级列表丙", 3, idx=3, numbered=True),
    ]
    before = [d.level for d in ds]
    audit = correct_numbering_skeleton(ds, warnings={})
    assert [d.level for d in ds] == before
    assert audit == []


def test_skeleton_fixed_node_never_moves_only_warns() -> None:
    """G8-10: an anchored series demanded deeper by evidence stays put with
    a warning; the invariant I3 anchor semantics win."""
    ds = [
        _d("第一章 总则", 3, idx=0, numbered=True),
        _d("第一条 锚定条款", 3, idx=1, numbered=True, outline=2, anchored=True),
        _d("第二章 管理", 3, idx=2, numbered=True),
        _d("第二条 锚定条款乙", 3, idx=3, numbered=True, outline=2, anchored=True),
    ]
    warnings: dict = {}
    correct_numbering_skeleton(ds, warnings=warnings)
    # 章→条 edge demands the anchored 条 go to 4, but anchors never move.
    assert ds[1].level == 3 and ds[3].level == 3
    assert ds[0].level == 3 and ds[2].level == 3
    assert warnings.get("smart_skeleton_anchor_conflict", 0) >= 1


def test_skeleton_orphan_head_blocks_edge() -> None:
    """G8-13: a b-member BEFORE the first a-member is a counter-example —
    no edge, zero adjustment (healthy 一、 1.1 1.1.2 doc stays intact)."""
    ds = [
        _d("1.1 开头小节", 2, idx=0, numbered=True),  # orphan before 一、
        _d("一、之后的父级", 1, idx=1, numbered=True),
        _d("1.2 后续小节", 2, idx=2, numbered=True),
        _d("二、另一父级", 1, idx=3, numbered=True),
        _d("2.1 再一节", 2, idx=4, numbered=True),
    ]
    before = [d.level for d in ds]
    audit = correct_numbering_skeleton(ds, warnings={})
    assert [d.level for d in ds] == before
    assert audit == []


def test_skeleton_mln_intrinsic_edges_push_children() -> None:
    """MLN raw2 pushed below raw1 keeps raw3 below raw2 (intrinsic edges)."""
    ds = [
        _d("1. 顶层", 3, idx=0, numbered=True),
        _d("1.1 二层", 3, idx=1, numbered=True),  # collided with 顶层 by size
        _d("1.1.1 三层", 3, idx=2, numbered=True),
        _d("2. 顶层乙", 3, idx=3, numbered=True),
        _d("2.1 二层乙", 3, idx=4, numbered=True),
    ]
    # Convert "1." / "2." to MultiLevelNum raw 1 the way backfill would.
    from dataclasses import replace

    for d in (ds[0], ds[3]):
        d.numbering = replace(
            d.numbering,
            style_key="MultiLevelNum",
            raw_level=1,
            top_ordinal=d.numbering.ordinal,
        )
    correct_numbering_skeleton(ds, warnings={})
    assert ds[0].level == 3 and ds[3].level == 3
    assert ds[1].level == 4 and ds[4].level == 4
    assert ds[2].level == 5


# ---------------------------------------------------------------------------
# smoothing (§2.2.8 step 2) + clamping (G8-6)
# ---------------------------------------------------------------------------


def test_smoothing_aligns_unanchored_to_anchored_mode() -> None:
    """Post-anchor smoothing: skip_anchored keeps locked members fixed and
    pulls the unanchored stragglers to the anchored mode."""
    cn = "一二三"
    ds = []
    for i, lv, anch in ((0, 2, True), (1, 2, True), (2, 4, False)):
        ds.append(_d(f"{cn[i]}、条目", lv, idx=i, numbered=True, anchored=anch))
    align_numbering_series(ds, skip_anchored=True)
    assert [d.level for d in ds] == [2, 2, 2]


def test_clamp_beyond_nine_demotes_to_body() -> None:
    """G8-6: >9 levels demote to body instead of clamping to 9."""
    ds = [
        _d("九层标题", 9, idx=0),
        _d("十层标题", 10, idx=1),
    ]
    warnings: dict = {}
    clamp_deep_levels(ds, warnings=warnings)
    assert ds[0].is_heading
    assert not ds[1].is_heading and ds[1].use_raw_text
    assert warnings["smart_clamp_demotions"] == 1


def test_skeleton_snapshot_takes_shallowest_and_floor_writeback() -> None:
    """G8-12: a series split across windows (members at 3 and 5) solves from
    the SHALLOWEST snapshot (3); floor write-back deepens the 3-member to 4
    and never lifts the 5-member."""
    from dataclasses import replace

    ds = [
        _d("1. 顶层", 3, idx=0, numbered=True),
        _d("1.1 前窗成员", 3, idx=1, numbered=True),
        _d("2. 顶层乙", 3, idx=2, numbered=True),
        _d("2.1 被平移压深的成员", 5, idx=3, numbered=True),
    ]
    for d in (ds[0], ds[2]):
        d.numbering = replace(
            d.numbering,
            style_key="MultiLevelNum",
            raw_level=1,
            top_ordinal=d.numbering.ordinal,
        )
    correct_numbering_skeleton(ds, warnings={})
    # raw-2 node: snapshot = min(3, 5) = 3; solved = max(3, solved(raw1)+1=4)
    assert ds[1].level == 4  # deepened by the solve
    assert ds[3].level == 5  # floor semantics: never lifted shallower


def test_skeleton_suspected_inversion_counted_not_acted_on() -> None:
    """A14 (§2.3.5): an inversion against the habitual order that lacks the
    three-evidence proof is counted as suspected — levels stay put."""
    ds = [
        _d("一、总体要求", 3, idx=0, numbered=True),
        _d("（一）提高认识", 2, idx=1, numbered=True),  # inverted vs 一、
    ]
    warnings: dict = {}
    audit = correct_numbering_skeleton(ds, warnings=warnings)
    assert warnings.get("smart_skeleton_inversion_suspected", 0) >= 1
    assert [d.level for d in ds] == [3, 2]  # observed, never acted on
    assert audit == []


# ---------------------------------------------------------------------------
# assembler bookkeeping (review C3 / C4)
# ---------------------------------------------------------------------------


def test_merged_then_demoted_heading_keeps_member_text() -> None:
    """Review C4: a §2.2.7-merged heading later demoted by the sweep must
    still emit every member's text. The absorbed-member markers are only laid
    down while the merged heading survives; once demoted, the members fall
    back to their own paragraph rows, so I1 passes (no content loss)."""
    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_content_preservation,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult

    records = [
        ParagraphRecord(
            kind="para", text="标题被拆成两行，上半句在此，", full_text_raw="标题被拆成两行，上半句在此，"
        ),
        ParagraphRecord(
            kind="para",
            text="下半句在此，合并后成为带句号的完整句子。",
            full_text_raw="下半句在此，合并后成为带句号的完整句子。",
        ),
        ParagraphRecord(kind="para", text="正文段落。", full_text_raw="正文段落。"),
    ]
    ds = [
        HeadingDecision(record_index=0, text=records[0].text, is_heading=True, level=2, font_size_pt=14.0),
        HeadingDecision(record_index=1, text=records[1].text, is_heading=True, level=2, font_size_pt=14.0),
    ]
    ds = merge_split_headings(ds, records, warnings={})
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings={})

    # Reproduce run_smart_heading's absorbed-sentinel bookkeeping.
    decisions: dict[int, HeadingDecision] = {}
    for d in ds:
        decisions[d.record_index] = d
        if d.member_indices and not d.is_title_block and d.is_heading:
            for m in d.member_indices[1:]:
                decisions[m] = HeadingDecision(record_index=m, text="", absorbed=True)
    result = SmartHeadingResult(
        decisions=decisions, toc_indices=set(), doc_title=None, audit={}
    )
    blocks = _assemble_blocks_smart(records, result, None, {})
    assert verify_content_preservation(records, blocks) == []
    joined = "\n".join(b["content"] for b in blocks)
    assert "下半句在此" in joined  # the absorbed member survived


def test_title_block_members_emitted_exactly_once() -> None:
    """Review C3: non-lead members of a multi-paragraph title block are
    emitted once (inside the composite level-0 block), never re-emitted as
    standalone body rows."""
    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult

    records = [
        ParagraphRecord(kind="para", text="关于加强质量管理的通知"),
        ParagraphRecord(kind="para", text="质监发〔2026〕12号"),
        ParagraphRecord(kind="para", text="第一章 总则"),
        ParagraphRecord(kind="para", text="正文内容一。"),
    ]
    tb = HeadingDecision(
        record_index=0,
        text=records[0].text,
        is_heading=True,
        is_title_block=True,
        level=0,
        composed_heading="关于加强质量管理的通知 — 质监发〔2026〕12号",
        title_parts=("关于加强质量管理的通知", "质监发〔2026〕12号"),
        member_indices=(0, 1),
    )
    member_sentinel = HeadingDecision(record_index=1, text="")
    member_sentinel.note("title_block_member")
    heading = HeadingDecision(record_index=2, text=records[2].text, is_heading=True, level=1)
    result = SmartHeadingResult(
        decisions={0: tb, 1: member_sentinel, 2: heading},
        toc_indices=set(),
        doc_title="关于加强质量管理的通知",
        audit={},
    )
    blocks = _assemble_blocks_smart(records, result, None, {})
    all_content = "\n".join(b["content"] for b in blocks)
    assert all_content.count("质监发〔2026〕12号") == 1
    # The doc-number lives only inside the level-0 title block.
    title_block = next(b for b in blocks if b.get("is_title_block"))
    assert "质监发〔2026〕12号" in title_block["content"]
