"""G7 tests: two-round physical-outline anchoring (§2.2.6) + CB3."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.smart_heading.heading_flow import (
    HeadingDecision,
    anchor_outline_levels,
)
from lightrag.parser.docx.smart_heading.style_key import classify_numbering

pytestmark = pytest.mark.offline


def _d(
    text: str,
    level: int,
    *,
    outline: int | None = None,
    numbered: bool = False,
    idx: int = 0,
) -> HeadingDecision:
    return HeadingDecision(
        record_index=idx,
        text=text,
        is_heading=True,
        level=level,
        outline_level=outline,
        numbering=classify_numbering(text) if numbered else None,
    )


def _build(specs) -> list[HeadingDecision]:
    return [
        _d(text, level, outline=outline, numbered=numbered, idx=i)
        for i, (text, level, outline, numbered) in enumerate(specs)
    ]


def test_no_outline_skips_everything() -> None:
    ds = _build([("标题甲", 2, None, False), ("标题乙", 3, None, False)])
    assert anchor_outline_levels(ds, warnings={}) is False
    assert [d.level for d in ds] == [2, 3]
    assert not any(d.anchored for d in ds)


def test_round1_non_numbered_snaps_to_outline() -> None:
    """G7-1: outlineLvl=2 → final level 3 regardless of the size level."""
    ds = _build([("样式标题", 5, 2, False)])
    assert anchor_outline_levels(ds, warnings={}) is True
    assert ds[0].level == 3 and ds[0].anchored
    assert ds[0].pre_anchor_level == 5


def test_round1_series_mode_unifies_outliers() -> None:
    """G7-2: 5 CnNum members, four outlined at 1 and one at 3 → all level 2."""
    specs = [
        (f"{'一二三四五'[i]}、条目", 4, 1 if i != 2 else 3, True) for i in range(5)
    ]
    ds = _build(specs)
    warnings: dict = {}
    anchor_outline_levels(ds, warnings=warnings)
    assert [d.level for d in ds] == [2] * 5
    assert all(d.anchored for d in ds)
    assert "smart_cb3_low_outline_ratio" not in warnings


def test_cb3_low_outline_ratio_still_propagates() -> None:
    """G7-3: 2/10 outlined (<50%) → propagate + warning."""
    specs = []
    cn = "一二三四五六七八九十"
    for i in range(10):
        specs.append((f"{cn[i]}、条目", 4, 1 if i < 2 else None, True))
    ds = _build(specs)
    warnings: dict = {}
    anchor_outline_levels(ds, warnings=warnings)
    assert [d.level for d in ds] == [2] * 10
    assert warnings["smart_cb3_low_outline_ratio"] == 1


def test_round2_subwindow_with_deep_levels_shifts_with_back_anchor() -> None:
    """G7-4: the sub-window still holds levels ≥ R1 → whole-window shift by
    R2 - R1 alongside the raised trailing anchor."""
    ds = _build(
        [
            ("锚点甲", 2, 1, False),  # anchored, unchanged (pre=2 → 2)
            ("未锁定深标题", 4, None, False),
            ("未锁定更深", 5, None, False),
            ("锚点乙", 4, 1, False),  # anchored: 4 → 2 (R1=4, R2=2)
        ]
    )
    warnings: dict = {}
    anchor_outline_levels(ds, warnings=warnings)
    # window holds level ≥ R1(4) → shift by R2-R1 = -2
    assert [d.level for d in ds] == [2, 2, 3, 2]
    assert all(d.anchored for d in ds)
    assert warnings["smart_anchor_window_shifts"] == 1  # A14 metric


def test_round2_gap_closing_when_all_shallower_than_r1() -> None:
    """G7-5: sub-window entirely shallower than R1 — three shapes: gap
    closes by max(0, L0-R2+1); exactly-one-level gap; already above R2."""
    # shape 1: L0=5, back anchor 7→3 → Δ = -(5-3+1) = -3
    ds1 = _build(
        [
            ("未锁定甲", 4, None, False),
            ("未锁定乙", 5, None, False),
            ("后锚点", 7, 2, False),  # 7 → 3
        ]
    )
    anchor_outline_levels(ds1, warnings={})
    assert [d.level for d in ds1] == [1, 2, 3]

    # shape 2: L0=3, back 5→3 → Δ = -(3-3+1) = -1 (continuous boundary)
    ds2 = _build(
        [
            ("未锁定甲", 3, None, False),
            ("后锚点", 5, 2, False),  # 5 → 3
        ]
    )
    anchor_outline_levels(ds2, warnings={})
    assert [d.level for d in ds2] == [2, 3]

    # shape 3: already shallower than R2 → no shift
    ds3 = _build(
        [
            ("未锁定甲", 2, None, False),
            ("后锚点", 6, 3, False),  # 6 → 4, window at 2 < R2 → untouched
        ]
    )
    anchor_outline_levels(ds3, warnings={})
    assert [d.level for d in ds3] == [2, 4]


def test_round2_floor_compression_warns() -> None:
    """G7-5 floor: a shift that would go below level 1 floors and warns."""
    ds = _build(
        [
            ("未锁定甲", 2, None, False),
            ("未锁定乙", 4, None, False),  # ≥ R1 → whole-window shift
            ("后锚点", 4, 0, False),  # 4 → 1 (R1=4, R2=1) → Δ=-3
        ]
    )
    warnings: dict = {}
    anchor_outline_levels(ds, warnings=warnings)
    assert [d.level for d in ds] == [1, 1, 1]
    assert warnings["smart_anchor_floor_compressions"] == 1


def test_round2_back_anchor_moving_deeper_adjusts_nothing() -> None:
    """G7-6a: a trailing anchor that moved DEEPER leaves the window alone."""
    ds = _build(
        [
            ("未锁定甲", 3, None, False),
            ("后锚点", 2, 4, False),  # 2 → 5 (deeper)
        ]
    )
    anchor_outline_levels(ds, warnings={})
    assert [d.level for d in ds] == [3, 5]


def test_round2_front_anchor_follow_and_stop() -> None:
    """Front-anchor shift drags its immediate subtree; stops at the first
    heading not deeper than the anchor's new level."""
    ds = _build(
        [
            ("前锚点", 4, 1, False),  # 4 → 2 (delta -2)
            ("子标题甲", 5, None, False),  # follows → 3
            ("子标题乙", 6, None, False),  # follows → 4
            ("同层标题", 2, None, False),  # ≤ new level 2 → stop, untouched
            ("后续深标题", 3, None, False),  # beyond the cut, no back anchor
        ]
    )
    anchor_outline_levels(ds, warnings={})
    assert [d.level for d in ds] == [2, 3, 4, 2, 3]


def test_round2_first_window_without_front_anchor() -> None:
    """G7-6b: the document-leading window has no front anchor — no follow
    adjustment; only the trailing anchor's raise applies."""
    ds = _build(
        [
            ("开头未锁定", 3, None, False),
            ("锚点", 3, 1, False),  # 3 → 2 (R1=3, R2=2)
        ]
    )
    anchor_outline_levels(ds, warnings={})
    # window holds level ≥ R1(3) → shifts by -1
    assert [d.level for d in ds] == [2, 2]


def test_round2_front_follow_stops_at_pre_adjust_level_when_raised() -> None:
    """A3 (§2.2.6): the front-anchor follow stops at the anchor's PRE-adjust
    level. Raised front (4→2): only the genuine child (5) follows; the
    sibling at the old level (4) and anything shallower stay put."""
    ds = _build(
        [
            ("前锚点", 4, 1, False),  # outline 1 → level 2 (pre=4, delta=-2)
            ("真子标题", 5, None, False),
            ("旧平级标题", 4, None, False),
            ("更浅标题", 3, None, False),
        ]
    )
    anchor_outline_levels(ds, warnings={})
    # Post-level stop (the bug) would drag 旧平级 to 2 and 更浅 to 1.
    assert [d.level for d in ds] == [2, 3, 4, 3]


def test_round2_front_follow_stops_at_pre_adjust_level_when_lowered() -> None:
    """A3 symmetric case. Lowered front (2→4): the child at 3 must follow to
    5 — a post-level stop (3 ≤ 4) would strand it above its parent."""
    ds = _build(
        [
            ("前锚点", 2, 3, False),  # outline 3 → level 4 (pre=2, delta=+2)
            ("真子标题", 3, None, False),
            ("旧平级标题", 2, None, False),
        ]
    )
    anchor_outline_levels(ds, warnings={})
    assert [d.level for d in ds] == [4, 5, 2]
