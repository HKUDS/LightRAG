"""Unit tests for §2.2.8 numbered-list re-nesting (`nest_numbered_under_parent`)
and the `backfill_top_level` chapter-top guard.

The nesting pass repairs same-font-size layouts where `assign_levels_by_size`
places a non-MultiLevelNum numbered class (EnNum, EnAlpha, …) at a flat class
slot ABOVE the deep MultiLevelNum heading it should nest under. Decisions are
built directly with post-anchor levels so the pass is exercised in isolation.
"""

from __future__ import annotations

from lightrag.parser.docx.smart_heading.heading_flow import (
    HeadingDecision,
    backfill_top_level,
    nest_numbered_under_parent,
)
from lightrag.parser.docx.smart_heading.style_key import classify_numbering


def _d(idx, text, level, *, size=12.0, outline=None, plain=False, anchored=False):
    return HeadingDecision(
        record_index=idx,
        text=text,
        is_heading=True,
        level=level,
        font_size_pt=size,
        numbering=None if plain else classify_numbering(text),
        outline_level=outline,
        anchored=anchored or outline is not None,
    )


# ---------------------------------------------------------------------------
# nest_numbered_under_parent
# ---------------------------------------------------------------------------


def test_i_ennum_nests_under_deep_mln_parent() -> None:
    # EnNum mis-leveled shallower than the deep MLN it follows → parent + 1.
    ds = [_d(0, "2.1.1.1.1.2 密钥管理规则", 6, outline=5), _d(1, "1. 密钥产生", 5)]
    nest_numbered_under_parent(ds)
    assert ds[1].level == 7
    assert "nest_under_parent" in ds[1].rule_trail


def test_ii_two_lists_each_align_to_own_parent() -> None:
    ds = [
        _d(0, "2.1.1 A", 5, outline=4),
        _d(1, "1. x", 4),
        _d(2, "2. y", 4),
        _d(3, "2.1.1.1 B", 6, outline=5),
        _d(4, "1. p", 4),
        _d(5, "2. q", 4),
    ]
    nest_numbered_under_parent(ds)
    assert ds[1].level == 6 and ds[2].level == 6  # under A (L5)
    assert ds[4].level == 7 and ds[5].level == 7  # under B (L6)


def test_iii_already_deeper_is_not_raised() -> None:
    # EnNum already nested deeper than its parent → untouched.
    ds = [_d(0, "2.1.1 A", 5, outline=4), _d(1, "1. x", 6)]
    nest_numbered_under_parent(ds)
    assert ds[1].level == 6
    assert ds[1].rule_trail == []


def test_iv_outlined_ennum_is_not_touched() -> None:
    # An EnNum carrying an outline level is anchoring/I3's job, not this pass.
    ds = [_d(0, "2.1.1 A", 5, outline=4), _d(1, "1. x", 7, outline=6)]
    nest_numbered_under_parent(ds)
    assert ds[1].level == 7
    assert ds[1].rule_trail == []


def test_v_locked_descendant_not_dragged() -> None:
    # An outline-locked descendant under the EnNum must NOT move with it.
    ds = [
        _d(0, "2.1.1 A", 6, outline=5),
        _d(1, "1. x", 5),
        _d(2, "child", 7, outline=6, anchored=True),
    ]
    nest_numbered_under_parent(ds)
    assert ds[1].level == 7  # x nested under A (L6)
    assert ds[2].level == 7  # locked child stayed put (not shifted to 9)


def test_v2_nonoutline_anchored_child_moves_with_parent() -> None:
    # After anchoring round 2, ordinary NON-outline headings are flagged
    # anchored=True (bookkeeping, not an outline lock). A genuine non-outline
    # child MUST still move with its re-nested parent — only physical-outline
    # descendants stay pinned. Regression for the `anchored`-vs-`outline_level`
    # stop condition.
    ds = [
        _d(0, "2.1.1 A", 6, outline=5),
        _d(1, "1. x", 5),
        _d(2, "小节", 6, plain=True, anchored=True),  # non-outline, round-2 anchored
    ]
    nest_numbered_under_parent(ds)
    assert ds[1].level == 7  # x nested under A (L6)
    assert ds[2].level == 8  # non-outline child followed x (+2), still under it


def test_vi_unnumbered_parent_no_reattach_to_earlier_numbered() -> None:
    # Nearest parent is an unnumbered non-outline heading → nest under IT,
    # never walk back to the earlier MLN.
    ds = [
        _d(0, "2.1 MLN", 2, outline=1),
        _d(1, "小结", 5, plain=True),
        _d(2, "1. x", 4),
    ]
    nest_numbered_under_parent(ds)
    assert ds[2].level == 6  # under 小结 (L5), not under 2.1 (L2)


def test_vii_peer_aligns_across_deeper_child() -> None:
    # 1. A -> a. child (deeper) -> 2. B : B aligns to A, NOT nested under child.
    ds = [
        _d(0, "2.1.1 P", 6, outline=5),
        _d(1, "1. A", 5),
        _d(2, "a. child", 6),
        _d(3, "2. B", 5),
    ]
    nest_numbered_under_parent(ds)
    assert ds[3].level == ds[1].level  # 2. B == 1. A (peer)
    assert ds[2].level == ds[1].level + 1  # a. child under 1. A


def test_vii_b_nested_restart_nests_under_child_not_outer() -> None:
    # 1. A -> a. child -> 1. subitem : the RESET ordinal (1 after 1) starts a
    # new nested sublist under `a. child`, NOT a peer of the outer `1. A`.
    ds = [
        _d(0, "2.1.1 P", 5, outline=4),
        _d(1, "1. A", 5),
        _d(2, "a. child", 6),
        _d(3, "1. subitem", 5),
    ]
    nest_numbered_under_parent(ds)
    assert ds[1].level == 6  # 1. A under P (L5)
    assert ds[2].level == 7  # a. child under 1. A
    assert ds[3].level == 8  # 1. subitem RESTARTS under a. child (not peer of A)
    assert ds[3].level != ds[1].level


def test_viii_same_level_heading_breaks_the_list() -> None:
    # 1. first -> 小结(same level) -> 1. redo : the same-level heading closes the
    # scope, so 1. redo starts a NEW list nested under 小结 (not peer of first).
    ds = [
        _d(0, "2.1 P", 5, outline=4),
        _d(1, "1. first", 6),
        _d(2, "小结", 6, plain=True),
        _d(3, "1. redo", 6),
        _d(4, "2. redo2", 6),
    ]
    nest_numbered_under_parent(ds)
    assert ds[3].level == 7  # under 小结 (L6), not aligned to 1. first
    assert "nest_under_parent" in ds[3].rule_trail
    assert ds[4].level == ds[3].level  # redo2 aligns to redo


# ---------------------------------------------------------------------------
# backfill_top_level chapter-top guard (Fix 2)
# ---------------------------------------------------------------------------


def test_backfill_linked_chapter_beats_linked_ennum() -> None:
    # BOTH a CnChapter top AND an EnNum series carry ≥2 ordinal linkages (the
    # latent case the guard targets). The chapter must win as MLN level-1; EnNum
    # must NOT be rewritten. Layout keeps each series' scopes free of foreign
    # top ordinals so both reach linkage 2.
    from lightrag.parser.docx.smart_heading.style_key import EN_NUM, MULTI_LEVEL_NUM

    ds = [
        _d(0, "第一章 X", 2, size=14.0),  # CnChapter ord1
        _d(1, "1. foo", 2, size=14.0),  # EnNum ord1
        _d(2, "1.1 a", 3),  # MLN top1
        _d(3, "1.2 b", 3),  # MLN top1
        _d(4, "第二章 Y", 2, size=14.0),  # CnChapter ord2
        _d(5, "2. bar", 2, size=14.0),  # EnNum ord2
        _d(6, "2.1 c", 3),  # MLN top2
        _d(7, "2.2 d", 3),  # MLN top2
    ]
    backfill_top_level(ds)

    foo, bar = ds[1], ds[5]
    chap1, chap2 = ds[0], ds[4]
    # EnNum untouched (chapter wins on the priority tie-break)…
    assert foo.numbering.style_key == EN_NUM
    assert bar.numbering.style_key == EN_NUM
    assert "backfill_top_level" not in foo.rule_trail
    # …CnChapter absorbed as the MLN raw-1 top.
    assert chap1.numbering.style_key == MULTI_LEVEL_NUM
    assert "backfill_top_level" in chap1.rule_trail
    assert chap2.numbering.style_key == MULTI_LEVEL_NUM
