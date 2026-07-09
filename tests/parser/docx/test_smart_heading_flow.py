"""G5/G6 tests: candidate gate, size leveling, backfill, series alignment."""

from __future__ import annotations

import logging

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading.heading_flow import (
    HeadingDecision,
    align_numbering_series,
    assign_levels_by_size,
    backfill_top_level,
    close_unnumbered_level_gaps,
    demote_strong_body_headings,
    document_fs_base,
    gate_candidates,
    gate_with_cb1,
    nest_numbered_under_parent,
)
from lightrag.parser.docx.smart_heading.style_key import classify_numbering

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _propagate_lightrag_logs():
    """The ``lightrag`` logger sets propagate=False; caplog needs it on."""
    lg = logging.getLogger("lightrag")
    old = lg.propagate
    lg.propagate = True
    try:
        yield
    finally:
        lg.propagate = old


def _para(text: str, *, size: float = 12.0, **kw) -> ParagraphRecord:
    return ParagraphRecord(kind="para", text=text, font_size_pt=size, **kw)


def _stub_strong_body(text: str) -> str | None:
    stripped = text.strip()
    if stripped.endswith(("。", "？", "！")) or len(stripped) > 60:
        return "strong_body_stub"
    return None


def _stub_no_veto(_c, _t) -> str | None:
    return None


def _stub_no_caption(_t) -> str | None:
    return None


def _gate(records, *, fs_base=None, **kw):
    indices = list(range(len(records)))
    fs = fs_base or document_fs_base(records, indices)
    kw.setdefault("strong_body", _stub_strong_body)
    kw.setdefault("numbering_veto", _stub_no_veto)
    kw.setdefault("caption_veto", _stub_no_caption)
    return gate_candidates(records, indices, fs_base=fs, **kw)


def _body(n: int = 20, size: float = 12.0) -> list[ParagraphRecord]:
    return [
        _para(f"这是第{i}段正常长度的正文内容，用来撑起基准字号的统计权重。", size=size)
        for i in range(n)
    ]


def _texts(result) -> list[str]:
    return [d.text for d in result.decisions]


# ---------------------------------------------------------------------------
# G5: candidate gate
# ---------------------------------------------------------------------------


def test_outline_candidate_survives_smaller_font() -> None:
    """G5-1: an outlineLvl paragraph at 10.5pt < FS_base=12 stays a candidate."""
    records = _body() + [_para("小字号样式标题", size=10.5, outline_level=0)]
    result = _gate(records)
    assert "小字号样式标题" in _texts(result)
    d = next(x for x in result.decisions if x.text == "小字号样式标题")
    assert d.rule_trail[0] == "outline"


def test_outline_strong_body_demotion_is_rule_tagged() -> None:
    """Review C2: an outline paragraph the recognition-time strong-body check
    demotes must leave a rule-tagged demoted decision (not a silent drop), so
    the I2 retention check sees an explicit demotion instead of a violation."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )

    records = _body(20) + [
        _para("这段正文被误加了大纲级别，它以句号结束。", size=12.0, outline_level=1)
    ]
    result = _gate(records)
    # Not admitted as a heading candidate…
    assert "这段正文被误加了大纲级别，它以句号结束。" not in _texts(result)
    # …but recorded as an explicit, rule-tagged demotion.
    assert len(result.demoted) == 1
    dem = result.demoted[0]
    assert dem.record_index == len(records) - 1
    assert dem.is_heading is False
    assert "strong_body_demoted" in dem.rule_trail
    # Merging the demoted decisions (as run_smart_heading does) makes I2 pass.
    merged = list(result.decisions) + result.demoted
    assert verify_baseline_heading_retention(records, merged) == []


def test_non_outline_strong_body_demotion_is_audited_output_neutral() -> None:
    """Every recognition-time strong-body demotion is audited, not just the
    I2 outline ones: a NON-outline candidate that the check demotes leaves a
    rule-tagged demoted decision so the per-paragraph ledger matches the
    demotion counter — but stays output-neutral (``use_raw_text`` off, no
    physical outline, so it renders as plain body just like no decision)."""
    records = _body(30) + [
        # +4pt over the 12pt body → a size_strong candidate, but strong-body
        # (ends with a period) and NOT carrying any physical outline level.
        _para(
            "这是一段被字号误判为标题候选但其实是正文的内容，它以句号收尾。", size=16.0
        )
    ]
    warnings: dict = {}
    result = _gate(records, warnings=warnings)
    # Not admitted as a heading…
    assert (
        "这是一段被字号误判为标题候选但其实是正文的内容，它以句号收尾。"
        not in _texts(result)
    )
    # …but recorded as an explicit, rule-tagged, audited demotion.
    assert warnings.get("smart_strong_body_demotions") == 1
    assert len(result.demoted) == 1
    dem = result.demoted[0]
    assert dem.record_index == len(records) - 1
    assert dem.is_heading is False
    assert "strong_body_demoted" in dem.rule_trail
    # Non-outline: output-neutral (renders rec.text, not full_text_raw) and no
    # physical-outline weight, so it never touches the I2 retention check.
    assert dem.use_raw_text is False
    assert dem.outline_level is None


def test_numbering_veto_suppression_is_audited_output_neutral() -> None:
    """A paragraph whose numbering identity a veto revokes, and which then
    matches no promotion rule, is NOT silently dropped: it leaves a rule-tagged
    output-neutral ledger row carrying the veto reason so the audit records WHY
    a would-be numbered candidate became body."""

    def _veto_the_item(_cls, text):
        return "homophone_ner_entity" if text.startswith("1. ") else None

    records = _body(20) + [_para("1. 建立上岗人员培训制度", size=12.0)]
    result = _gate(records, numbering_veto=_veto_the_item)

    # Not admitted as a heading (its numbering was revoked, no other signal)…
    assert "1. 建立上岗人员培训制度" not in _texts(result)
    # …but recorded in the veto-suppression ledger, rule-tagged.
    assert len(result.veto_suppressed) == 1
    supp = result.veto_suppressed[0]
    assert supp.record_index == len(records) - 1
    assert supp.is_heading is False
    assert "homophone_ner_entity" in supp.rule_trail
    assert "numbering_veto_suppressed" in supp.rule_trail
    # Non-outline → output-neutral: renders as plain body, same as no decision.
    assert supp.use_raw_text is False
    assert supp.outline_level is None


def test_high_confidence_size_tiers() -> None:
    """G5-6: +1pt alone stands; +0.5pt needs companions; isolated fails."""
    records = _body(30)
    records.append(_para("强信号标题", size=13.0))  # +1pt
    records.append(_para("孤立弱信号", size=12.5))  # +0.5pt, alone… but see below
    result = _gate(records)
    texts = _texts(result)
    assert "强信号标题" in texts
    assert "孤立弱信号" not in texts

    # With a same-size companion the weak tier passes (P2 pair).
    records2 = _body(30)
    records2.append(_para("弱信号甲", size=12.5))
    records2 += _body(3)
    records2.append(_para("弱信号乙", size=12.5))
    result2 = _gate(records2)
    texts2 = _texts(result2)
    assert "弱信号甲" in texts2 and "弱信号乙" in texts2


def test_weak_pair_ignores_strong_body_companion() -> None:
    """Review P2: a strong-body paragraph (about to be demoted) does NOT count
    as a weak-signal companion, so a lone +0.5pt paragraph paired only with a
    same-size sentence stays out (spec §2.3.4 '不含强正文特征的弱信号段落')."""
    records = _body(30)
    records.append(_para("孤立弱信号标题", size=12.5))  # +0.5pt, no real companion
    records += _body(3)
    # Same size but strong-body (ends in 。 → _stub_strong_body demotes it).
    records.append(_para("这是一句以句号结尾的同字号长正文。", size=12.5))
    result = _gate(records)
    assert "孤立弱信号标题" not in _texts(result)


def test_base_size_needs_numbering_or_bold() -> None:
    records = _body(30)
    records.append(_para("一、成套编号甲", size=12.0))
    records += _body(3)
    records.append(_para("二、成套编号乙", size=12.0))
    records.append(_para("整段加粗短语", size=12.0, all_bold=True))
    records.append(_para("普通同字号段落", size=12.0))
    result = _gate(records)
    texts = _texts(result)
    assert "一、成套编号甲" in texts and "二、成套编号乙" in texts
    assert "整段加粗短语" in texts
    assert "普通同字号段落" not in texts


def test_isolated_numbering_blocked_by_p2() -> None:
    """孤立编号不成章: a lone numbered line at base size is rejected."""
    records = _body(30) + [_para("一、只有这一个编号", size=12.0)]
    result = _gate(records)
    assert "一、只有这一个编号" not in _texts(result)


def test_smaller_than_base_never_heading_without_outline() -> None:
    records = _body(30) + [_para("一、小字号编号", size=10.5)]
    records += [_para("二、小字号编号乙", size=10.5)]
    result = _gate(records)
    # Necessary condition: size >= FS_base (high confidence).
    assert _texts(result) == []


def test_low_confidence_disables_bare_size_advantage() -> None:
    """G3-3: mixed sizes (<60% dominant) → bare size > FS_base is not enough;
    composite paths (series / bold / centered companions) still work."""
    records = []
    records += _body(10, size=10.0)
    records += _body(8, size=12.0)
    records += _body(7, size=14.0)
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert not fs.confidence_high

    records.append(_para("纯字号优势标题", size=16.0))
    records.append(_para("一、复合信号标题", size=16.0))
    records += _body(2, size=10.0)
    records.append(_para("二、复合信号标题乙", size=16.0))
    result = _gate(records)
    texts = _texts(result)
    assert "纯字号优势标题" not in texts
    assert "一、复合信号标题" in texts and "二、复合信号标题乙" in texts


def test_centered_companion_channel_low_confidence() -> None:
    """G5-7: centered same-size lines separated by body are companions;
    an isolated centered line and back-to-back centered runs are not."""
    records = []
    records += _body(6, size=10.0)
    records += _body(5, size=12.0)
    records += _body(4, size=14.0)  # low-confidence mix
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert not fs.confidence_high

    records.append(_para("居中标题一", size=14.0, alignment="center"))
    records += _body(3, size=10.0)
    records.append(_para("居中标题二", size=14.0, alignment="center"))
    records += _body(3, size=10.0)
    records.append(_para("孤立居中行", size=14.0, alignment="center"))
    # 孤立居中行 has companions? It IS same size/shape as 一/二 — so it
    # qualifies too (companions需不同run即可). Control: adjacent pair below.
    records += _body(2, size=10.0)
    records.append(_para("连续居中甲", size=13.0, alignment="center"))
    records.append(_para("连续居中乙", size=13.0, alignment="center"))

    result = _gate(records)
    texts = _texts(result)
    assert "居中标题一" in texts and "居中标题二" in texts
    # Same-run neighbours are each other's ONLY same-size shape lines →
    # no cross-run companion → both rejected.
    assert "连续居中甲" not in texts and "连续居中乙" not in texts


def test_centered_run_longer_than_four_loses_channel() -> None:
    """G5-8: an 8-line centered poem block never gains candidacy."""
    records = _body(30)
    poem = [_para(f"诗歌居中行{i}", size=12.0, alignment="center") for i in range(8)]
    records += poem
    records += _body(3)
    records.append(_para("居中标题一", size=12.0, alignment="center"))
    records += _body(2)
    records.append(_para("居中标题二", size=12.0, alignment="center"))
    result = _gate(records)
    texts = _texts(result)
    assert all(f"诗歌居中行{i}" not in texts for i in range(8))
    assert "居中标题一" in texts and "居中标题二" in texts


def test_cb1_reestimation_recovers_density(monkeypatch, caplog) -> None:
    """G3-4 branch 1: dominant fake-heading size folds into the body; the
    re-gated density collapses and the breaker does NOT trip."""
    records = _body(60, size=10.5)
    # 30 short 12pt lines with no composite signal — they clear the strong
    # +1.5pt tier at first, flooding density past 30%.
    for i in range(30):
        records.append(_para(f"伪标题短语{i}", size=12.0))
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert fs.confidence_high

    warnings: dict = {}
    with caplog.at_level(logging.INFO, logger="lightrag"):
        result = gate_with_cb1(
            records,
            indices,
            fs_base=fs,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    assert warnings.get("smart_cb1_reestimated") == 1
    assert result.cb1_reestimated and not result.cb1_tripped
    # 12pt became the body baseline; the bare lines lost candidacy.
    assert result.decisions == []
    # Re-gate outcome logged as a converged (INFO) event.
    regate = next(m for m in _log_messages(caplog) if "CB1 re-gate:" in m)
    assert "converged" in regate and "threshold" in regate


def _log_messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def test_cb1_extreme_sample_trips_breaker(monkeypatch, caplog) -> None:
    """G3-4 branch 2: a question-bank doc stays dense after one
    re-estimation → the breaker trips (sub-document falls back)."""
    records = []
    for i in range(40):
        records.append(_para(f"{i + 1}. 选择题题干第{i}题", size=12.0))
        records.append(_para("A. 选项甲  B. 选项乙", size=10.5))
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)

    warnings: dict = {}
    with caplog.at_level(logging.INFO, logger="lightrag"):
        result = gate_with_cb1(
            records,
            indices,
            fs_base=fs,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    assert result.cb1_reestimated
    assert result.cb1_tripped
    assert warnings.get("smart_cb1_tripped") == 1
    # Re-gate outcome logged as a still-over (WARNING) fallback event.
    regate = next(r for r in caplog.records if "CB1 re-gate:" in r.getMessage())
    assert regate.levelno == logging.WARNING
    assert "outline-only fallback" in regate.getMessage()


# ---------------------------------------------------------------------------
# G6: leveling / backfill / alignment
# ---------------------------------------------------------------------------


def _decision(
    text: str,
    *,
    size: float | None = None,
    centered: bool = False,
    numbered: bool = True,
    idx: int = 0,
) -> HeadingDecision:
    cls = classify_numbering(text) if numbered else None
    return HeadingDecision(
        record_index=idx,
        text=text,
        is_heading=True,
        font_size_pt=size,
        numbering=cls,
        centered=centered,
    )


def test_size_bands_consecutive_levels() -> None:
    """G6-1: 16/14/12pt headings → levels 1/2/3."""
    ds = [
        _decision("总览", size=16.0, numbered=False, centered=True, idx=0),
        _decision("第一章 分述", size=14.0, idx=1),
        _decision("一、细项", size=12.0, idx=2),
    ]
    assign_levels_by_size(ds)
    assert [d.level for d in ds] == [1, 2, 3]


def test_same_band_centered_numbered_uncentered_order() -> None:
    """G6-2: within one band, centered < numbered < uncentered."""
    ds = [
        _decision("居中无编号", size=14.0, numbered=False, centered=True, idx=0),
        _decision("一、编号标题", size=14.0, idx=1),
        _decision("非居中无编号", size=14.0, numbered=False, idx=2),
    ]
    assign_levels_by_size(ds)
    assert [d.level for d in ds] == [1, 2, 3]


def test_step2_spec_example_table() -> None:
    """G6-5 (part 1): the §2.2.5 step-2 example table, verbatim."""
    ds = [
        _decision("居中无编号标题", size=14.0, numbered=False, centered=True, idx=0),
        _decision("1.2 多级编号", size=14.0, idx=1),
        _decision("一、中文数字", size=14.0, idx=2),
        _decision("2.3 多级编号", size=14.0, idx=3),
        _decision("1.1.2 多级编号", size=14.0, idx=4),
        _decision("（一）中文括号", size=14.0, idx=5),
    ]
    assign_levels_by_size(ds)
    assert [d.level for d in ds] == [2 - 1, 3 - 1, 4 - 1, 3 - 1, 4 - 1, 5 - 1]
    # (the spec table starts at 2 because a shallower band exists there;
    # with a single band our levels start at 1 — relative order matches)


def test_step4_spec_example_table() -> None:
    """G6-5 (part 2): the §2.2.5 step-4 alignment example, verbatim."""
    rows = [
        ("居中无编号标题", 2, 2, False),
        ("1.2 节", 3, 3, True),
        ("一、目标", 4, 4, True),
        ("二、方法", 5, 4, True),  # same CnNum series as the row above
        ("1.1.2 小节", 4, 4, True),
        ("1.3 节", 4, 3, True),  # aligns with the raw-level-2 series
        ("三、总结", 3, 3, True),  # a NEW CnNum series (closed by the shallower 1.3)
        ("（一）分项", 5, 4, True),
        ("（二）分项", 4, 4, True),
    ]
    ds = []
    for i, (text, level, _expected, numbered) in enumerate(rows):
        d = _decision(text, size=14.0, numbered=numbered, idx=i)
        d.level = level
        ds.append(d)
    align_numbering_series(ds)
    assert [d.level for d in ds] == [r[2] for r in rows]


def test_alignment_shifts_subtree() -> None:
    """G8-5 precursor: aligning a parent drags its subtree along."""
    ds = []
    specs = [
        ("一、第一部分", 4, True),
        ("子标题甲", 5, False),
        ("二、第二部分", 5, True),  # same series → align to 4
        ("子标题乙", 6, False),  # subtree of 二、 → shifts to 5
        ("孙标题", 7, False),  # deeper subtree → shifts to 6
    ]
    for i, (text, level, numbered) in enumerate(specs):
        d = _decision(text, size=None, numbered=numbered, idx=i)
        d.level = level
        ds.append(d)
    align_numbering_series(ds)
    assert [d.level for d in ds] == [4, 5, 4, 5, 6]


def test_backfill_absorbs_ennum_with_ordinal_linkage() -> None:
    """G6-4: 第一章/1.1 style docs — EnNum with matching top ordinals is
    absorbed as MultiLevelNum raw 1; competing keys keep their class."""
    texts = [
        ("1. 总则", 14.0, 2),
        ("1.1 范围", 12.0, 3),
        ("1.2 定义", 12.0, 3),
        ("2. 要求", 14.0, 2),
        ("2.1 基本要求", 12.0, 3),
        ("第一章 干扰章节", 14.0, 2),  # competing CnChapter, no linkage
    ]
    ds = []
    for i, (text, size, level) in enumerate(texts):
        d = _decision(text, size=size, idx=i)
        d.level = level
        ds.append(d)
    warnings: dict = {}
    backfill_top_level(ds, warnings=warnings)
    en1, en2 = ds[0], ds[3]
    assert en1.numbering.style_key == "MultiLevelNum" and en1.numbering.raw_level == 1
    assert en2.numbering.style_key == "MultiLevelNum" and en2.numbering.raw_level == 1
    assert ds[5].numbering.style_key == "CnChapter"  # not absorbed
    # levels rise above the raw-2 layer
    assert en1.level == 2 and en2.level == 2  # already shallower than 1.x → kept


def test_backfill_linkage_waives_size_threshold() -> None:
    """G6-6 main group: parents typed SMALLER than children (10.5 < 12) are
    still absorbed when ≥2 ordinal linkages hold."""
    texts = [
        ("1. 总则", 10.5, 4),
        ("1.1 范围", 12.0, 3),
        ("1.2 定义", 12.0, 3),
        ("2. 要求", 10.5, 4),
        ("2.1 基本要求", 12.0, 3),
    ]
    ds = []
    for i, (text, size, level) in enumerate(texts):
        d = _decision(text, size=size, idx=i)
        d.level = level
        ds.append(d)
    warnings: dict = {}
    backfill_top_level(ds, warnings=warnings)
    assert ds[0].numbering.style_key == "MultiLevelNum"
    assert ds[0].numbering.raw_level == 1
    # Levels rose above the raw-2 layer (min raw level was 3 → parents at 2).
    assert ds[0].level == 2 and ds[3].level == 2


def test_backfill_single_linkage_pair_skipped_with_warning() -> None:
    """G6-6 control 1: only one linked ordinal → no waiver + warning."""
    texts = [
        ("1. 总则", 10.5, 4),  # below the 12pt size threshold
        ("1.1 范围", 12.0, 3),
        ("1.2 定义", 12.0, 3),
        ("2. 附则", 10.5, 4),  # scope holds no 2.x → only ordinal 1 links
    ]
    ds = []
    for i, (text, size, level) in enumerate(texts):
        d = _decision(text, size=size, idx=i)
        d.level = level
        ds.append(d)
    warnings: dict = {}
    backfill_top_level(ds, warnings=warnings)
    assert ds[0].numbering.style_key == "EnNum"
    assert warnings.get("smart_backfill_single_linkage") == 1


def test_backfill_counter_example_blocks_linkage() -> None:
    """A foreign top value inside a parent scope refutes the linkage; with
    the size channel also failing, nothing is absorbed."""
    texts = [
        ("1. 总则", 10.5, 4),
        ("2.1 错位小节", 12.0, 3),  # top=2 inside scope of 1 → counter-example
        ("2. 要求", 10.5, 4),
        ("2.2 又一节", 12.0, 3),
    ]
    ds = []
    for i, (text, size, level) in enumerate(texts):
        d = _decision(text, size=size, idx=i)
        d.level = level
        ds.append(d)
    backfill_top_level(ds, warnings={})
    assert ds[0].numbering.style_key == "EnNum"  # linkage refuted, size too low


def test_fs_base_excludes_toc_lines() -> None:
    records = _body(10, size=12.0)
    records += [
        ParagraphRecord(
            kind="para", text="目录行" * 10, font_size_pt=20.0, is_toc_link=True
        )
        for _ in range(20)
    ]
    fs = document_fs_base(records, range(len(records)))
    assert fs.size_pt == 12.0


# ---------------------------------------------------------------------------
# spec-vs-implementation audit regressions (A1 / A2 / A5 / A6 / A10)
# ---------------------------------------------------------------------------


def test_cb1_second_pass_keeps_sizes_above_reestimated_base() -> None:
    """A1 (§2.3.3): re-estimation disables ALL same-size composite paths but
    keeps auto-admitting sizes strictly above the re-estimated body size."""
    records = _body(60, size=10.5)
    for i in range(30):
        records.append(_para(f"伪标题短语{i}", size=12.0))
    records.append(_para("一、同字号编号甲", size=12.0))
    records.append(_para("二、同字号编号乙", size=12.0))
    records.append(_para("同字号加粗题", size=12.0, all_bold=True))
    records.append(_para("真标题一", size=16.0))
    records.append(_para("真标题二", size=16.0))
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)

    result = gate_with_cb1(
        records,
        indices,
        fs_base=fs,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result.cb1_reestimated
    texts = _texts(result)
    # Above the folded-in 12pt body size: auto-admitted without any signal.
    assert "真标题一" in texts and "真标题二" in texts
    # Same-size series / bold paths are off in the CB1 second pass.
    assert "一、同字号编号甲" not in texts and "二、同字号编号乙" not in texts
    assert "同字号加粗题" not in texts
    assert all(f"伪标题短语{i}" not in texts for i in range(30))


def test_cb1_sparse_body_average_triggers_reestimation() -> None:
    """A2 (§2.3.3 trigger #2): density stays under the cap, but the average
    CJK-weighted body chars between adjacent headings falls below 200."""
    records = []
    for i in range(4):
        records.append(_para(f"密集标题{i}", size=16.0))
        records += _body(2, size=12.0)  # normal-length body...
        records[-1] = _para("短", size=12.0)  # ...but keep gaps tiny
        records[-2] = _para("句", size=12.0)
    records += _body(60, size=12.0)  # tail body keeps density low
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert fs.confidence_high

    warnings: dict = {}
    result = gate_with_cb1(
        records,
        indices,
        fs_base=fs,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert warnings.get("smart_cb1_reestimated") == 1
    assert result.cb1_reestimated


def test_cb1_long_inter_heading_body_does_not_trigger() -> None:
    """A2 control: healthy documents (long body between headings) never
    trip the sparse-body trigger; the first-pass result is returned."""
    records = []
    for i in range(4):
        records.append(_para(f"正常标题{i}", size=16.0))
        records += _body(3, size=12.0)  # ~90 CJK chars ≈ 270 weighted per gap
    records += _body(40, size=12.0)
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)

    result = gate_with_cb1(
        records,
        indices,
        fs_base=fs,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert not result.cb1_reestimated
    assert all(f"正常标题{i}" in _texts(result) for i in range(4))


def _cb1_baseline_density_records() -> list:
    """4 headings (16pt) with long body gaps → 40% candidate density but the
    inter-heading spacing stays well over 200 weighted chars (no sparse-body
    trip). Layout: H B B H B B H B B H → 4 headings / 10 paras = 0.40."""
    long_body = "这是一段刻意写长的正文内容，用来把相邻标题之间的正文字符数撑到二百加权字符以上，从而避开稀疏正文触发条件的判定。"
    records: list = []
    for i in range(4):
        records.append(_para(f"标题第{i}节", size=16.0))
        if i < 3:
            records.append(_para(long_body, size=12.0))
            records.append(_para(long_body, size=12.0))
    return records


def test_cb1_threshold_is_baseline_aware() -> None:
    """The density ceiling is max(floor 0.35, baseline_density + 0.10). A
    richly-outlined sub-document (high baseline) tolerates a candidate density
    the flat floor would reject, so CB1 does NOT trip and every heading is
    kept."""
    records = _cb1_baseline_density_records()
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert fs.confidence_high

    # baseline outline density 0.35 → threshold max(0.35, 0.45) = 0.45;
    # the 40% candidate density sits under it.
    result = gate_with_cb1(
        records,
        indices,
        fs_base=fs,
        baseline_density=0.35,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert not result.cb1_reestimated
    assert all(f"标题第{i}节" in _texts(result) for i in range(4))


def test_cb1_low_baseline_still_trips_same_density() -> None:
    """Control for baseline-awareness: the SAME 40% candidate density, but a
    low baseline (threshold falls back to the 0.35 floor), trips CB1."""
    records = _cb1_baseline_density_records()
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)

    warnings: dict = {}
    result = gate_with_cb1(
        records,
        indices,
        fs_base=fs,
        baseline_density=0.0,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result.cb1_reestimated
    assert warnings.get("smart_cb1_reestimated") == 1


def test_min_inter_heading_chars_env_override(monkeypatch) -> None:
    """DOCX_SMART_MIN_INTER_HEADING_CHARS tunes the sparse-body trigger: a
    document whose inter-heading spacing clears the default 200 trips
    re-estimation once the floor is raised above that spacing."""
    records: list = []
    for i in range(4):
        records.append(_para(f"正常标题{i}", size=16.0))
        records += _body(3, size=12.0)  # ~250 weighted chars between headings
    records += _body(40, size=12.0)  # tail body keeps density low
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)

    def _run():
        return gate_with_cb1(
            records,
            indices,
            fs_base=fs,
            warnings={},
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )

    # Default 200: the gaps are wide enough → not sparse, no re-estimation.
    assert not _run().cb1_reestimated
    # Raise the floor above the gap width → the same doc now reads as sparse.
    monkeypatch.setenv("DOCX_SMART_MIN_INTER_HEADING_CHARS", "400")
    assert _run().cb1_reestimated


def test_caption_veto_spares_outline_paragraphs() -> None:
    """A5 (P3 × I2): an outline paragraph hitting the caption blacklist keeps
    its candidacy (silently dropping it would trip the I2 machine check);
    the caption veto still rejects non-outline paragraphs."""
    from lightrag.parser.docx.smart_heading import guardrails

    records = _body(30)
    records.append(_para("图 1 系统架构", size=12.0, outline_level=1))
    records.append(_para("图 2 数据流转示意", size=16.0))
    result = _gate(records, caption_veto=guardrails.caption_prefix_reason)
    texts = _texts(result)
    assert "图 1 系统架构" in texts  # outline is immune to P3
    assert "图 2 数据流转示意" not in texts  # P3 vetoes the size-tier path


def test_llm_body_veto_revokes_candidacy() -> None:
    """A10 (§2.2.4 赋予或撤销): an LLM body vote strips an otherwise
    strong-size candidate; without the vote it is admitted."""
    records = _body(30)
    records.append(_para("被判为正文的大字号行", size=16.0))
    idx = len(records) - 1

    admitted = _gate(records)
    assert "被判为正文的大字号行" in _texts(admitted)

    vetoed = _gate(records, llm_body_vetoes={idx})
    assert "被判为正文的大字号行" not in _texts(vetoed)


def test_title_block_gate_uses_char_weighted_mode_not_mean(monkeypatch) -> None:
    """A6 (§2.2.4): the title-block size baseline is the global FS_base
    initial value (char-weighted MODE, 12pt here) — a long large-font
    paragraph inflating the weighted mean must not raise the gate."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    records = [
        _para("行业观察白皮书", size=14.0),  # title line: 14 ≥ mode12+1
        _para("某某研究院", size=12.0),
    ]
    records += _body(30, size=12.0)
    records.append(_para("超大字号的引言长段落" * 40 + "。", size=30.0))

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {
                "is_title_block": True,
                "main_title": "行业观察白皮书",
                "sub_title": None,
                "doc_number": None,
                "classification": None,
                "publisher": "某某研究院",
                "date": None,
                "headings": [],
                "body": [],
            },
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.doc_title == "行业观察白皮书"
    assert any(d.is_title_block and d.level == 0 for d in result.decisions.values())


def test_seq_break_knob_splits_distant_series(monkeypatch) -> None:
    """A12 (DOCX_SMART_SEQ_BREAK_PARAS): a long body run between same-key
    members closes the open sequence; the default 0 keeps one series."""
    from lightrag.parser.docx.smart_heading.heading_flow import (
        align_numbering_series,
    )

    def _series(levels_and_idx):
        ds = []
        for text, level, idx in levels_and_idx:
            d = _decision(text, idx=idx)
            d.level = level
            ds.append(d)
        return ds

    spec = [("一、条目甲", 3, 0), ("二、条目乙", 4, 1), ("三、远端条目", 5, 60)]

    monkeypatch.delenv("DOCX_SMART_SEQ_BREAK_PARAS", raising=False)
    ds = _series(spec)
    align_numbering_series(ds)
    assert [d.level for d in ds] == [3, 3, 3]  # one series, ties → shallow

    monkeypatch.setenv("DOCX_SMART_SEQ_BREAK_PARAS", "5")
    ds = _series(spec)
    align_numbering_series(ds)
    # 三、 sits 58 record slots away → its sequence is closed and it aligns
    # alone (not pulled to 3); the 5→4 move is the pre-existing subtree
    # shift riding along when 二、 aligned 4→3.
    assert [d.level for d in ds] == [3, 3, 4]


def test_run_level_audit_exports_metrics(monkeypatch) -> None:
    """A14 (§2.3.5): llm call count, per-candidate verdicts and the full
    re-judgment ledger land in the audit payload."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = [
        _para("行业观察白皮书", size=16.0),
        _para("某某研究院", size=12.0),
    ]
    records += _body(30, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "行业观察白皮书"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    audit = result.audit
    assert audit["llm_calls"] >= 1
    assert [c["is_title_block"] for c in audit["title_block_candidates"]] == [True]
    rows = audit["decisions"]
    assert rows
    assert all(
        {
            "hash",
            "summary",
            "font_size_pt",
            "sub_fs_base",
            "rules",
            "level",
            "is_heading",
        }
        <= set(r)
        for r in rows
    )
    assert any(any("title_block" in rule for rule in r["rules"]) for r in rows)
    # Enriched fields present and the plaintext preview is capped. Value
    # assertions live in test_audit_decision_fields_enriched below.
    from lightrag.parser.docx.smart_heading.heading_flow import _AUDIT_SUMMARY_CHARS

    assert all(len(r["summary"]) <= _AUDIT_SUMMARY_CHARS + 3 for r in rows)  # +"..."
    # Every sub-document (fallbacks included) now records its fs_base.
    assert all("fs_base" in s for s in audit["sub_documents"])


def test_audit_decision_fields_enriched(monkeypatch) -> None:
    """Requirement 2: each decision row carries a plaintext summary, the
    paragraph mode font off the record, and its sub-document fs_base."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    # No title block (llm_judge=None) → the whole doc is one sub-document.
    records = [_para("第一章 绪论", size=16.0, outline_level=0)]
    records += _body(30, size=12.0)

    result = run_smart_heading(
        records,
        llm_judge=None,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    rows = result.audit["decisions"]
    heading_row = next(r for r in rows if r["summary"] == "第一章 绪论")
    assert heading_row["is_heading"] is True
    assert heading_row["font_size_pt"] == 16.0  # record mode, not FS_base
    assert heading_row["sub_fs_base"] == 12.0  # the sub-document body baseline


def test_run_logs_physical_feature_summary(monkeypatch, caplog) -> None:
    """Requirement 1: a document-level info line (FS_base, paragraph count,
    outline histogram) fires once physical features are extracted."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = [_para("第一章 绪论", size=16.0, outline_level=0)]
    records += _body(30, size=12.0)

    with caplog.at_level(logging.INFO, logger="lightrag"):
        run_smart_heading(
            records,
            llm_judge=None,
            warnings={},
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )

    summary = next(
        (m for m in _log_messages(caplog) if "physical features:" in m), None
    )
    assert summary is not None
    assert "FS_base=12.0pt" in summary
    assert "31 paragraphs" in summary
    assert "L1: 1" in summary and "none: 30" in summary


def test_table_title_block_end_to_end_assembly(monkeypatch) -> None:
    """§2.2.4 table channel, end to end: a cover laid out inside tables is
    judged a title block (level-0, heading = main title) whose content carries
    every absorbed cover-cell text verbatim (as plain text, not a ``<table>``
    placeholder). With no section heading in the document, the title block also
    OWNS the following body and the trailing data table (block-boundary merge),
    and the data table keeps its ``<table>`` placeholder; I1 passes (absorption
    is lossless)."""
    import json

    from lightrag.parser.docx.parse_document import (
        ParagraphRecord,
        _assemble_blocks_smart,
    )
    from lightrag.parser.docx.smart_heading import guardrails as g
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    def _table(rows, text="<table>[]</table>"):
        return ParagraphRecord(kind="table", text=text, table_cell_features=rows)

    cover_title = _table(
        [[("产品标准化大纲", 22.0, False)], [("某某模块", 22.0, False)]],
        text='<table>[["产品标准化大纲"], ["某某模块"]]</table>',
    )
    cover_publisher = _table(
        [[("某某电子股份有限公司", 16.0, False)]],
        text='<table>[["某某电子股份有限公司"]]</table>',
    )
    data_table = _table(
        [[("序号", 10.5, False), ("这一格是以句号结尾的数据。", 10.5, False)]],
        text='<table>[["序号", "这一格是以句号结尾的数据。"]]</table>',
    )
    records = [cover_title, cover_publisher] + _body(30, size=12.0) + [data_table]

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        if "产品标准化大纲" in prompt:
            return json.dumps(
                {
                    "is_title_block": True,
                    "main_title": "产品标准化大纲某某模块",
                    "publisher": "某某电子股份有限公司",
                },
                ensure_ascii=False,
            )
        return json.dumps({"is_title_block": False}, ensure_ascii=False)

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.doc_title == "产品标准化大纲某某模块"
    lead = result.decisions[0]
    assert lead.is_title_block and lead.level == 0
    assert lead.member_indices == (0, 1)

    meta: dict = {}
    blocks = _assemble_blocks_smart(records, result, {}, meta)
    title_blocks = [b for b in blocks if b.get("is_title_block")]
    assert len(title_blocks) == 1
    tb = title_blocks[0]
    assert tb["level"] == 0
    # heading is the plain main title; the publisher stays in the content.
    assert tb["heading"] == "产品标准化大纲某某模块"
    for cell in ("产品标准化大纲", "某某模块", "某某电子股份有限公司"):
        assert cell in tb["content"]  # absorption is lossless
    assert meta["first_heading"] == "产品标准化大纲某某模块"

    # No section heading separates the cover from the rest, so the whole
    # document is one title block: cover cells carried as PLAIN text (no
    # <table> placeholder), while the ordinary data table keeps its placeholder.
    assert len(blocks) == 1
    assert '["产品标准化大纲"]' not in tb["content"]
    assert '<table>[["序号"' in tb["content"]

    # I1: absorbed cell texts never trip the paragraph-preservation check.
    assert g.verify_content_preservation(records, blocks, toc_indices=set()) == []


def test_title_block_empty_members_defense() -> None:
    """Defensive: a title verdict whose members yield no emittable content (an
    empty-cell cover table) must warn + skip the title treatment rather than
    crash (``_build_unsplit_block`` indexes ``paragraphs[-1]``) or mis-label.
    The degenerate lead never becomes a level-1 heading and no stale pin leaks
    to the following body block. (Unreachable via ``run_smart_heading`` — its
    locate-back requires a non-empty window — so it is exercised directly.)"""
    from lightrag.parser.docx.parse_document import (
        ParagraphRecord,
        _assemble_blocks_smart,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult

    empty_cover = ParagraphRecord(
        kind="table",
        text='<table>[[""]]</table>',
        table_cell_features=[[("   ", 22.0, False)]],
        para_id="t0",
    )
    body = ParagraphRecord(
        kind="para", text="正文一句。", font_size_pt=12.0, para_id="p1"
    )
    records = [empty_cover, body]
    result = SmartHeadingResult(
        decisions={
            0: HeadingDecision(
                record_index=0,
                text="<table>…",
                is_heading=True,
                level=0,
                is_title_block=True,
                member_indices=(0,),
            )
        },
        toc_indices=set(),
        doc_title=None,
        audit={},
    )
    warnings: dict = {}
    blocks = _assemble_blocks_smart(records, result, warnings, {})

    assert warnings.get("title_block_empty_members") == 1
    assert not any(b.get("is_title_block") for b in blocks)
    # The following body survived as an ordinary (non-title) block — no crash,
    # no level-1 mis-labelling of the degenerate title decision.
    joined = "\n".join(b["content"] for b in blocks)
    assert "正文一句" in joined


# ---------------------------------------------------------------------------
# §2.3.3 CB1 look-ahead: density evaluated AFTER strong-body + series propagation
# ---------------------------------------------------------------------------


def test_cb1_strong_body_lookahead_keeps_cnnum_headings(caplog) -> None:
    """A flat, same-size 公文 whose body is ``CnParentNum`` "(一)…。" clauses
    floods raw density (all numbered candidates at one size), but projecting the
    downstream strong-body + CB2 series propagation drops it back under the bar.
    CB1 must withhold the trip WITHOUT re-estimating, so the genuine same-size
    ``CnNum`` headings survive the normal demotion sweep instead of being wiped
    out by the blanket "same-size composite off" re-gate."""
    clause = "加强组织领导，落实工作责任，明确任务分工，确保各项措施落地见效。"
    records = [
        _para("一、总体要求"),
        _para(f"（一）{clause}"),
        _para(f"（二）{clause}"),
        _para(f"（三）{clause}"),
        _para("二、主要任务"),
        _para(f"（四）{clause}"),
        _para(f"（五）{clause}"),
        _para(f"（六）{clause}"),
    ]
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert fs.size_pt == 12.0 and fs.confidence_high

    warnings: dict = {}
    with caplog.at_level(logging.INFO, logger="lightrag"):
        gate = gate_with_cb1(
            records,
            indices,
            fs_base=fs,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    # Look-ahead spared the trip WITHOUT re-estimation or a fallback.
    assert warnings.get("smart_cb1_strong_body_recovered") == 1
    assert gate.cb1_strong_body_recovered
    assert not gate.cb1_reestimated and not gate.cb1_tripped
    regate = next(r for r in caplog.records if "not tripping" in r.getMessage())
    assert regate.levelno == logging.INFO
    assert not any("re-estimated FS_base" in m for m in _log_messages(caplog))

    # gate_with_cb1 defers the real demotion — run the downstream sweep exactly
    # as run_smart_heading does; the CnParentNum body collapses, the CnNum
    # headings remain.
    ds = gate.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    assert {d.text for d in ds if d.is_heading} == {"一、总体要求", "二、主要任务"}


def test_cb1_lookahead_propagates_demotion_across_partial_series(caplog) -> None:
    """CB2 propagation inside the look-ahead: only one member of each
    ``CnParentNum`` series is sentence-like, but a 33% hit share (≥ the 20%
    body ratio, no physical outline) propagates the body demotion to the whole
    series — so the projected density still falls under threshold and CB1 is
    spared, keeping the ``CnNum`` headings."""
    sentence = "加强组织领导，落实工作责任，明确任务分工，确保各项措施落地见效。"
    phrase = (
        "完善制度机制，强化监督问责，明确任务分工，压实各方责任"  # no 。, < 60 chars
    )
    records = [
        _para("一、总体要求"),
        _para(f"（一）{sentence}"),  # sentence-like -> individual hit
        _para(f"（二）{phrase}"),  # not an individual hit
        _para(f"（三）{phrase}"),  # not an individual hit
        _para("二、保障措施"),
        _para(f"（四）{sentence}"),  # sentence-like -> individual hit
        _para(f"（五）{phrase}"),
        _para(f"（六）{phrase}"),
    ]
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)

    warnings: dict = {}
    with caplog.at_level(logging.INFO, logger="lightrag"):
        gate = gate_with_cb1(
            records,
            indices,
            fs_base=fs,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    assert gate.cb1_strong_body_recovered
    assert not gate.cb1_reestimated and not gate.cb1_tripped

    ds = gate.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    # Whole CnParentNum series demoted via propagation, not just the two hits.
    assert {d.text for d in ds if d.is_heading} == {"一、总体要求", "二、保障措施"}


# ---------------------------------------------------------------------------
# 公文版记 (imprint): strong-body rule 0 in the gate/sweep, and the
# outline-only fallback exception
# ---------------------------------------------------------------------------
# These tests use the REAL guardrails.strong_body_reason: every record either
# opens with an imprint marker (regex, returns before any NLP) or ends with a
# CJK sentence terminator (string check), so no spaCy models are needed.


def test_imprint_candidate_demoted_at_recognition() -> None:
    """An imprint line admitted by a size rule is demoted at recognition time
    (unnumbered → check_now always applies) with the imprint rule id."""
    from lightrag.parser.docx.smart_heading import guardrails

    records = _body(20) + [_para("抄送：各区人民政府", size=16.0)]
    warnings: dict = {}
    result = _gate(
        records, strong_body=guardrails.strong_body_reason, warnings=warnings
    )
    assert "抄送：各区人民政府" not in _texts(result)
    assert len(result.demoted) == 1
    dem = result.demoted[0]
    assert dem.rule_trail == ["imprint_marker", "strong_body_demoted"]
    assert dem.use_raw_text is False  # never a baseline heading: output-neutral
    assert warnings["smart_strong_body_demotions"] == 1


def test_imprint_outline_demotion_passes_i2() -> None:
    """An OUTLINE imprint line demotes with use_raw_text and an I2-recognized
    rule trail, so the retention check stays green."""
    from lightrag.parser.docx.smart_heading import guardrails

    records = _body(20) + [_para("抄送：各区人民政府", size=12.0, outline_level=1)]
    result = _gate(records, strong_body=guardrails.strong_body_reason)
    assert len(result.demoted) == 1
    dem = result.demoted[0]
    assert dem.is_heading is False and dem.use_raw_text is True
    merged = list(result.decisions) + result.demoted
    assert guardrails.verify_baseline_heading_retention(records, merged) == []


def test_postmerge_sweep_demotes_imprint_heading() -> None:
    """Leak path (e.g. llm_grant): a surviving imprint heading is caught by
    the §2.2.7 sweep; unnumbered → no series propagation side effects."""
    from lightrag.parser.docx.smart_heading import guardrails

    d = _decision("抄送：各成员单位", numbered=False)
    demote_strong_body_headings([d], strong_body=guardrails.strong_body_reason)
    assert d.is_heading is False
    assert "imprint_marker" in d.rule_trail
    assert "strong_body_demoted" in d.rule_trail


def test_outline_only_fallback_demotes_imprint() -> None:
    """The sub-document fallback keeps outlineLvl headings EXCEPT imprint
    lines, which get an explicit rule-tagged demotion (a silent skip would be
    an I2 violation). Both fallback call sites share this helper."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import (
        _outline_only_decisions,
    )

    records = [
        _para("第一章 总体要求", size=12.0, outline_level=0),
        _para("正文段落，以句号结尾。", size=12.0),
        _para("抄送：各成员单位", size=12.0, outline_level=1),
    ]
    warnings: dict = {}
    out = _outline_only_decisions(records, range(len(records)), warnings=warnings)
    assert len(out) == 2
    normal = next(d for d in out if d.record_index == 0)
    assert normal.is_heading and normal.level == 1
    assert "subdoc_fallback_outline_only" in normal.rule_trail
    dem = next(d for d in out if d.record_index == 2)
    assert dem.is_heading is False and dem.use_raw_text is True
    assert dem.rule_trail == ["imprint_marker", "strong_body_demoted"]
    assert warnings["smart_strong_body_demotions"] == 1
    assert verify_baseline_heading_retention(records, out) == []


def test_cb4_short_subdoc_fallback_wires_imprint(monkeypatch) -> None:
    """Wiring: run_smart_heading's cb4_short_subdoc fallback demotes an
    outline imprint line and counts it — even with an injected stub
    strong_body that does not know imprint (the fallback uses the guardrails
    default marker, not the injected strong_body)."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "1000000")

    records = _body(20) + [
        _para("附则说明", size=12.0, outline_level=0),
        _para("抄送：各成员单位", size=12.0, outline_level=1),
    ]
    imprint_idx = len(records) - 1
    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=None,  # uniform 12pt: no title-block candidates, no LLM
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.audit["sub_documents"][0]["fallback"] == "cb4_short_subdoc"
    dem = result.decisions[imprint_idx]
    assert dem.is_heading is False and dem.use_raw_text is True
    assert "imprint_marker" in dem.rule_trail
    assert result.decisions[imprint_idx - 1].is_heading  # plain outline kept
    assert warnings["smart_strong_body_demotions"] == 1


def test_whole_doc_cb4_skip_leaves_imprint_to_baseline(monkeypatch) -> None:
    """Guarantee boundary: below the whole-document CB4 gate smart never runs
    (returns None) — the imprint rules do NOT reach the untouched baseline
    output, by design."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "1000000")
    records = _body(5) + [_para("抄送：各成员单位", size=12.0, outline_level=1)]
    result = run_smart_heading(
        records,
        llm_judge=None,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is None


# ---------------------------------------------------------------------------
# 版记 region conditional demotion: a 抄送…印发 span followed by a valid title
# block (a 公文汇编 boundary) is force-demoted to body; otherwise veto-only.
# ---------------------------------------------------------------------------


def test_imprint_region_demoted_when_title_block_follows(monkeypatch) -> None:
    """公文汇编 boundary: a 抄送…印发 region whose closer is immediately followed
    by a valid title block is confirmed 版记 — its outline lines are force-
    demoted to body (use_raw_text, rule-tagged, I2-green). The 抄送 anchor is
    body already (no outline) and is left alone; the closer is a TRAILING 印发
    line the injected strong_body stub cannot see — only the region does."""
    import json

    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    region = [
        _para("抄送：各区人民政府", size=12.0),  # anchor (body)
        _para("中间说明行不带大纲", size=12.0),  # middle (plain body)
        _para("某某办公室 2026年6月30日 印发", size=12.0, outline_level=1),  # closer
        ParagraphRecord(kind="empty_para"),
        _para("数字政府建设白皮书", size=18.0, page_break_before=True),  # doc2 cover
    ]
    records = doc1 + region + _body(20)
    closer_idx = len(doc1) + 2
    cover_idx = len(doc1) + 4

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "数字政府建设白皮书"},
            ensure_ascii=False,
        )

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.decisions[cover_idx].is_title_block  # doc2 cover confirmed
    # The TRAILING 印发 outline line — invisible to the strong_body stub — is
    # force-demoted to body because the title block immediately follows.
    dem = result.decisions[closer_idx]
    assert dem.is_heading is False and dem.use_raw_text is True
    assert dem.rule_trail[-2:] == ["imprint_region", "strong_body_demoted"]
    assert warnings["smart_imprint_region_demotions"] == 1
    assert (
        verify_baseline_heading_retention(records, list(result.decisions.values()))
        == []
    )


def test_imprint_region_veto_only_without_following_title_block(monkeypatch) -> None:
    """No valid title block after the 印发 closer → the region is NOT confirmed:
    veto-only, the outline lines are kept as headings (never force-demoted)."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    records = (
        doc1
        + [
            _para("抄送：各区人民政府", size=12.0),
            _para("中间说明行不带大纲", size=12.0),
            _para("某某办公室 2026年6月30日 印发", size=12.0, outline_level=1),
        ]
        + _body(20)
    )
    closer_idx = len(doc1) + 2

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=None,  # nothing to confirm a following title block
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.decisions[closer_idx].is_heading is True
    assert "smart_imprint_region_demotions" not in warnings


def test_imprint_region_below_cb4_gate_untouched(monkeypatch) -> None:
    """Guarantee boundary: below the whole-document CB4 gate run_smart_heading
    returns None — the region / closer rules never reach the baseline output."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "1000000")
    records = _body(5) + [
        _para("抄送：各区", size=12.0),
        _para("某某办公室 2026年 印发", size=12.0, outline_level=1),
    ]
    result = run_smart_heading(
        records,
        llm_judge=None,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is None


# ---------------------------------------------------------------------------
# mixed table/paragraph covers, end to end: absorbed paragraphs stay members
# in SOURCE ORDER through judgment and assembly
# ---------------------------------------------------------------------------


def test_mixed_cover_assembly_preserves_source_order(monkeypatch) -> None:
    """表A + 封面材料段 + 表B: the absorbed paragraph rides in member_indices
    between the tables, assembly renders 表A cells → 段落 → 表B cells in source
    order with no duplication, and I1 passes."""
    import json

    from lightrag.parser.docx.parse_document import (
        ParagraphRecord,
        _assemble_blocks_smart,
    )
    from lightrag.parser.docx.smart_heading import guardrails as g
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    def _table(rows, text="<table>[]</table>"):
        return ParagraphRecord(kind="table", text=text, table_cell_features=rows)

    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("研究策划部编写", size=16.0),  # absorbed cover-material paragraph
        _table([[("某某电子股份有限公司", 16.0, False)]]),
    ] + _body(30, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "产品标准化大纲"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    lead = result.decisions[0]
    assert lead.is_title_block
    assert lead.member_indices == (0, 1, 2)  # 表A, 段, 表B — source order

    blocks = _assemble_blocks_smart(records, result, {}, {})
    tb = next(b for b in blocks if b.get("is_title_block"))
    content = tb["content"]
    # Source order preserved, each fragment exactly once.
    for frag in ("产品标准化大纲", "研究策划部编写", "某某电子股份有限公司"):
        assert content.count(frag) == 1
    assert (
        content.index("产品标准化大纲")
        < content.index("研究策划部编写")
        < content.index("某某电子股份有限公司")
    )
    assert g.verify_content_preservation(records, blocks, toc_indices=set()) == []


def test_paragraph_tail_cover_end_to_end(monkeypatch) -> None:
    """档号表 + 主标题段 + body(无后置表格): the paragraph-tail cover completes
    the title-block path — the candidate stands on the paragraph's size, the
    heading is the paragraph-borne main title, assembly is lossless."""
    import json

    from lightrag.parser.docx.parse_document import (
        ParagraphRecord,
        _assemble_blocks_smart,
    )
    from lightrag.parser.docx.smart_heading import guardrails as g
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    form_table = ParagraphRecord(
        kind="table",
        text="<table>[]</table>",
        table_cell_features=[
            [("档 号", 10.5, False), ("", None, False)],
            [("密 级", 10.5, False), ("公开", 10.5, False)],
        ],
    )
    records = [form_table, _para("某某管理办法", size=22.0)] + _body(30, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "某某管理办法"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    lead = result.decisions[0]
    assert lead.is_title_block
    assert lead.member_indices == (0, 1)

    blocks = _assemble_blocks_smart(records, result, {}, {})
    tb = next(b for b in blocks if b.get("is_title_block"))
    assert tb["heading"] == "某某管理办法"
    assert tb["content"].count("某某管理办法") == 1  # member, not re-emitted
    assert g.verify_content_preservation(records, blocks, toc_indices=set()) == []


def test_rejected_mixed_cover_paragraph_regains_heading_path(monkeypatch) -> None:
    """LLM 否决取舍固化: when the LLM rejects the mixed-cover candidate, the
    absorbed 22pt paragraph re-enters the normal gate and becomes an ORDINARY
    heading (not a title block) — bounded loss, mirroring multi-window."""
    import json

    from lightrag.parser.docx.parse_document import ParagraphRecord
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "10")

    form_table = ParagraphRecord(
        kind="table",
        text="<table>[]</table>",
        table_cell_features=[[("档 号", 10.5, False), ("", None, False)]],
    )
    records = [
        form_table,
        _para("某某管理办法", size=22.0),
        ParagraphRecord(
            kind="table",
            text="<table>[]</table>",
            table_cell_features=[[("某某电子股份有限公司", 16.0, False)]],
        ),
    ] + _body(30, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps({"is_title_block": False}, ensure_ascii=False)

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    d = result.decisions.get(1)
    assert d is not None and d.is_heading and not d.is_title_block


def test_trailing_document_date_not_absorbed_into_next_cover(monkeypatch) -> None:
    """End-to-end (mirrors test5-红头文件): a 成文日期 mis-ordered right after the
    版记 must NOT become the leading member of the following 附件 cover — it
    stays body under the previous section; the cover roots at the real title."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    tail = [
        _para("抄送：各设区市城乡规划局", size=12.0),  # anchor
        _para("河北省住房和城乡建设厅办公室  2009年7月6日印发", size=12.0),  # closer
        _para("二○○九年七月六日", size=12.0),  # 成文日期 (mis-ordered)
        _para("附件：", size=12.0),  # 附件 cover root
        _para("城市控制性详细规划备案工作规程", size=18.0),  # cover title
    ]
    records = doc1 + tail + _body(20)
    date_idx = len(doc1) + 2

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "城市控制性详细规划备案工作规程"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    tb_roots = [d for d in result.decisions.values() if d.is_title_block]
    assert tb_roots  # 附件 cover confirmed
    # The date is never a title-block member, and is body (not a heading).
    assert all(date_idx not in d.member_indices for d in tb_roots)
    d_date = result.decisions.get(date_idx)
    assert d_date is None or not (d_date.is_heading or d_date.is_title_block)


# ---------------------------------------------------------------------------
# §2.2.8: close_unnumbered_level_gaps (post-demotion unnumbered lift)
# ---------------------------------------------------------------------------


def _leveled(
    idx: int,
    text: str,
    level: int,
    *,
    plain: bool = False,
    outline: int | None = None,
    title_block: bool = False,
    anchored: bool = False,
) -> HeadingDecision:
    """A surviving post-leveling decision (numbering derived from text)."""
    return HeadingDecision(
        record_index=idx,
        text=text,
        is_heading=True,
        level=level,
        font_size_pt=12.0,
        numbering=None if plain else classify_numbering(text),
        outline_level=outline,
        anchored=anchored or outline is not None,
        is_title_block=title_block,
    )


def test_gap_close_test7_shape() -> None:
    """The test7-专利说明书 shape: an EnNum claims series occupied the class
    slot (L2) then got demoted wholesale — sections must lift 3 → 2."""
    warnings: dict = {}
    ds = [
        _leveled(0, "一种层级感知的文档语义分块方法及系统", 1, plain=True),
        _leveled(1, "技术领域", 3, plain=True),
        _leveled(2, "背景技术", 3, plain=True),
    ]
    close_unnumbered_level_gaps(ds, warnings=warnings)
    assert [d.level for d in ds] == [1, 2, 2]
    assert "unnumbered_gap_closed" in ds[1].rule_trail
    assert "unnumbered_gap_closed" in ds[2].rule_trail
    # Unmoved levels leave no note; the counter matches moved headings only.
    assert "unnumbered_gap_closed" not in ds[0].rule_trail
    assert warnings["smart_unnumbered_gap_closed"] == 2


def test_gap_close_preserves_mln_deliberate_gaps() -> None:
    """MultiLevelNum raw-level gaps are the author's choice: pinned levels
    keep their values and the trailing unnumbered heading stays put."""
    ds = [
        _leveled(0, "总标题", 1, plain=True),
        _leveled(1, "1.1 概述", 2),
        _leveled(2, "1.1.1.1 深层小节", 4),  # deliberate gap at 3
        _leveled(3, "普通小节", 5, plain=True),
    ]
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == [1, 2, 4, 5]
    assert all("unnumbered_gap_closed" not in d.rule_trail for d in ds)


def test_gap_close_single_gap_below_surviving_numbered() -> None:
    """A vacated level between a surviving numbered class and an unnumbered
    heading: the unnumbered one snaps up; the numbered one is pinned."""
    ds = [
        _leveled(0, "居中主标题", 1, plain=True),
        _leveled(1, "一、编号章", 2),
        _leveled(2, "加粗小节", 4, plain=True),  # L3 vacated by demotion
    ]
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == [1, 2, 3]


def test_gap_close_cascading_gaps() -> None:
    """Several movable levels re-pack consecutively (1/3/4 → 1/2/3)."""
    ds = [
        _leveled(0, "主标题", 1, plain=True),
        _leveled(1, "加粗章", 3, plain=True),
        _leveled(2, "加粗节", 4, plain=True),
    ]
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == [1, 2, 3]


def test_gap_close_numbered_numbered_gap_out_of_scope() -> None:
    """A gap BETWEEN two pinned numbered levels is out of scope: numbered
    levels never move (series equality), so the gap stays."""
    ds = [
        _leveled(0, "第一章 总则", 1),
        _leveled(1, "（一）分项", 3),  # gap at 2 between numbered classes
        _leveled(2, "加粗小节", 4, plain=True),
    ]
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == [1, 3, 4]


def test_gap_close_mixed_level_is_pinned() -> None:
    """A level holding both a numbered and an unnumbered heading is
    conservatively pinned — moving only the unnumbered half would split it."""
    ds = [
        _leveled(0, "主标题", 1, plain=True),
        _leveled(1, "一、编号章", 3),
        _leveled(2, "加粗章", 3, plain=True),  # same level as the numbered one
    ]
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == [1, 3, 3]


def test_gap_close_outline_pinned_round2_anchored_still_movable() -> None:
    """Physical-outline headings are pinned via ``outline_level`` — but a
    plain heading that anchoring round 2 flagged ``anchored=True`` (mere
    bookkeeping, not an outline lock) must still be movable."""
    ds = [
        _leveled(0, "样式标题", 1, outline=0),
        _leveled(1, "加粗小节", 4, plain=True, anchored=True),
    ]
    close_unnumbered_level_gaps(ds)
    assert ds[0].level == 1  # outline pinned
    assert ds[1].level == 2  # round-2 anchored plain heading still lifts


def test_gap_close_title_block_is_pinned() -> None:
    """A title-block root never moves, even off a vacated shallower level."""
    ds = [
        _leveled(0, "封面主标题", 2, plain=True, title_block=True),
        _leveled(1, "加粗章", 4, plain=True),
    ]
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == [2, 3]


def test_gap_close_then_nest_child_lands_on_corrected_parent() -> None:
    """Nest interaction (a): a numbered child nest WILL move (child level ≤
    parent level) lands on the parent's corrected level, not the stale one."""
    ds = [
        _leveled(0, "主标题", 1, plain=True),
        _leveled(1, "加粗章", 4, plain=True),  # stranded deep by a demoted class
        _leveled(2, "1. 子项", 2),  # EnNum class slot above the section
    ]
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == [1, 3, 2]  # section lifted 4 → 3
    nest_numbered_under_parent(ds)
    assert ds[2].level == 4  # corrected parent (3) + 1, not stale 4 + 1


def test_gap_close_then_nest_deeper_child_keeps_documented_gap() -> None:
    """Nest interaction (b), the documented scope boundary: nest is
    one-directional (``d.level > parent.level`` skips), so a child already
    deeper than its lifted parent keeps its level — the numeric gap stays,
    heading ORDER (child strictly deeper) and series equality hold."""
    ds = [
        _leveled(0, "主标题", 1, plain=True),
        _leveled(1, "加粗章", 3, plain=True),
        _leveled(2, "1. 子项", 4),
    ]
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == [1, 2, 4]  # gap between parent and child
    nest_numbered_under_parent(ds)
    assert ds[2].level == 4  # nest never pulls a deeper child up
    assert ds[2].level > ds[1].level  # order invariant holds


def test_gap_close_end_to_end_bold_sections_reach_level_2(monkeypatch) -> None:
    """End-to-end mirror of test7-专利说明书: no outline, single font size,
    a centered+bold main title, an EnNum claims series that the post-merge
    sweep demotes wholesale, and bold left-aligned section headings — the
    sections must land at L2 (not L3) after the vacated class slot closes."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = (
        [
            _para(
                "一种层级感知的文档语义分块方法及系统",
                size=12.0,
                all_bold=True,
                alignment="center",
            )
        ]
        + [
            _para(
                f"{i}、根据权利要求所述的方法，其特征在于，包括对应的处理步骤。",
                size=12.0,
            )
            for i in range(1, 9)
        ]
        + [_para("技术领域", size=12.0, all_bold=True)]
        + _body(10)
        + [_para("背景技术", size=12.0, all_bold=True)]
        + _body(10)
    )
    title_idx = 0
    section_indices = [9, 20]

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps({"is_title_block": False}, ensure_ascii=False)

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    d_title = result.decisions[title_idx]
    assert d_title.is_heading and d_title.level == 1
    for i in section_indices:
        d = result.decisions[i]
        assert d.is_heading, records[i].text
        assert d.level == 2, (records[i].text, d.level, d.rule_trail)
        assert "unnumbered_gap_closed" in d.rule_trail
    # The claims series is body (demoted by the post-merge sweep).
    for i in range(1, 9):
        d = result.decisions.get(i)
        assert d is None or not d.is_heading
