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


def test_outline_nlp_only_multisentence_is_spared_contextually() -> None:
    """An outline candidate beats an NLP-only multi-sentence vote, while the
    same text without outline and an outline paragraph with visible sentence
    punctuation retain the original strong-body demotion semantics."""

    def _multi(_text: str) -> str:
        return "strong_body_multi_sentence"

    spared_text = "2.3   投标单位廉洁自律承诺书"
    non_outline_text = "投标单位廉洁自律承诺书"
    explicit_text = "项目背景已经说明。后续要求继续执行"
    records = _body(20) + [
        _para(spared_text, size=14.0, outline_level=1),
        _para(non_outline_text, size=14.0),
        _para(explicit_text, size=14.0, outline_level=1),
    ]
    result = _gate(
        records,
        strong_body=_multi,
        numbering_veto=lambda _cls, text: (
            "homophone_ner_entity" if text == spared_text else None
        ),
    )

    spared = next(d for d in result.decisions if d.text == spared_text)
    assert spared.is_heading
    assert "outline_multisentence_spared" in spared.rule_trail
    assert "strong_body_demoted" not in spared.rule_trail

    demoted = {d.text: d for d in result.demoted}
    assert not demoted[non_outline_text].is_heading
    assert "strong_body_demoted" in demoted[non_outline_text].rule_trail
    assert not demoted[explicit_text].is_heading
    assert "strong_body_demoted" in demoted[explicit_text].rule_trail


def test_outline_other_strong_body_reasons_keep_full_force() -> None:
    """The outline guard is exact to multi-sentence: length and terminal
    punctuation remain direct, explicit demotion evidence."""
    from lightrag.parser.docx.smart_heading import guardrails

    length_text = "这是一段确实超过标题长度上限的正文内容" * 8
    terminal_text = "这段正文被误加了物理大纲级别，但仍然以句号结束。"
    records = _body(30) + [
        _para(length_text, size=14.0, outline_level=1),
        _para(terminal_text, size=14.0, outline_level=1),
    ]
    result = _gate(records, strong_body=guardrails.strong_body_reason)
    demoted = {d.text: d for d in result.demoted}

    assert "strong_body_length" in demoted[length_text].rule_trail
    assert "strong_body_sentence_end" in demoted[terminal_text].rule_trail
    assert all("strong_body_demoted" in d.rule_trail for d in demoted.values())


def test_cb1_projection_keeps_outline_multisentence_spared_candidates(
    monkeypatch,
) -> None:
    """CB1's scratch demotion uses the same outline guard: spared candidates
    remain in projected density instead of falsely recovering the overflow by
    projecting them all away."""

    def _multi_for_candidates(text: str) -> str | None:
        return "strong_body_multi_sentence" if text.startswith("候选标题") else None

    monkeypatch.setenv("DOCX_SMART_MIN_INTER_HEADING_CHARS", "0")
    records = _body(5) + [
        _para(f"候选标题{i}", size=14.0, outline_level=1) for i in range(5)
    ]
    fs = document_fs_base(records, range(len(records)))
    warnings: dict = {}
    result = gate_with_cb1(
        records,
        list(range(len(records))),
        fs_base=fs,
        warnings=warnings,
        strong_body=_multi_for_candidates,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )

    assert result.cb1_tripped
    assert not result.cb1_strong_body_recovered
    assert len(result.decisions) == 5
    assert all("outline_multisentence_spared" in d.rule_trail for d in result.decisions)


def test_post_merge_outline_nlp_only_multisentence_is_spared() -> None:
    """The same outline guard applies when numbered headings defer their
    strong-body judgment to the authoritative post-merge sweep."""

    def _multi(_text: str) -> str:
        return "strong_body_multi_sentence"

    spared = HeadingDecision(
        record_index=0,
        text="2.3 投标单位廉洁自律承诺书",
        is_heading=True,
        level=2,
        outline_level=1,
        numbering=classify_numbering("2.3 投标单位廉洁自律承诺书"),
    )
    explicit = HeadingDecision(
        record_index=1,
        text="2.4 项目背景已经说明。后续要求继续执行",
        is_heading=True,
        level=2,
        outline_level=1,
        numbering=classify_numbering("2.4 项目背景已经说明。后续要求继续执行"),
    )

    demote_strong_body_headings([spared, explicit], strong_body=_multi, warnings={})

    assert spared.is_heading
    assert spared.rule_trail == ["outline_multisentence_spared"]
    assert not explicit.is_heading
    assert "strong_body_demoted" in explicit.rule_trail


def test_source_multisentence_gate_real_spacy_end_to_end(monkeypatch) -> None:
    """test11 regression through real NLP and the block assembler, now via the
    source-level multi-sentence gate in ``strong_body_reason`` (a spaCy ≥2 vote
    counts only WITH a visible internal terminator):

    - the outline-1 ``2.3   投标单位廉洁自律承诺书`` survives the model's
      dependency-ROOT pseudo split (承诺|书) and stays a level-2 heading — the
      pseudo split now yields ``strong_body_reason is None`` at the source, so it
      is never demoted (no ``outline_multisentence_spared`` detour needed);
    - a non-outline body lead-in (no visible terminator) is likewise not strong
      body and simply stays plain body content (it is body-font, so it is not
      promoted to a heading either);
    - an outline paragraph carrying a genuine internal ``。`` still trips
      ``strong_body_multi_sentence`` and is demoted.
    """
    import json

    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading import guardrails, nlp
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    if nlp.missing_spacy_models():
        pytest.skip("spaCy models not installed")
    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    target = "2.3   投标单位廉洁自律承诺书"
    body_lead = (
        "为确保本项目产品的售后服务需求得以及时响应和解决，公司将组建该项目"
        "售后服务团队，具体负责人员事项如下表："
    )
    explicit = "项目背景已经说明。后续要求继续执行"

    # target/body_lead: spaCy pseudo-splits without any visible terminator →
    # the source gate returns None (previously target was a false multi_sentence
    # rescued only by the outline-context wrapper).
    assert not guardrails.has_explicit_internal_sentence_boundary(target)
    assert guardrails.strong_body_reason(target) is None
    assert guardrails.strong_body_reason(body_lead) is None
    # explicit: a genuine internal terminator corroborates the ≥2 vote.
    assert guardrails.has_explicit_internal_sentence_boundary(explicit)
    assert guardrails.strong_body_reason(explicit) == "strong_body_multi_sentence"

    records = _body(20) + [
        _para("2 商务文件", size=16.0, outline_level=0),
        *_body(5),
        _para("2.2   中标服务费承诺函", size=14.0, outline_level=1),
        *_body(5),
        _para(body_lead),  # body-font lead-in (a real one is not heading-sized)
        _para(target, size=14.0, outline_level=1),
        *_body(5),
        _para("2.4   公司实力", size=14.0, outline_level=1),
        *_body(5),
        _para(explicit, size=14.0, outline_level=1),
        *_body(5),
    ]

    def _llm(_prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": False, "headings": [], "body": []},
            ensure_ascii=False,
        )

    warnings: dict = {}
    result = run_smart_heading(records, llm_judge=_llm, warnings=warnings)
    assert result is not None
    blocks = _assemble_blocks_smart(records, result, {}, {})
    # No spared detour anymore — the source gate handles it silently.
    assert "smart_outline_multisentence_spared" not in warnings

    target_decision = next(d for d in result.decisions.values() if d.text == target)
    assert target_decision.is_heading and target_decision.level == 2
    assert "homophone_ner_entity" in target_decision.rule_trail
    assert "strong_body_demoted" not in target_decision.rule_trail
    assert "outline_multisentence_spared" not in target_decision.rule_trail

    target_block = next(b for b in blocks if b.get("heading") == target)
    assert target_block["level"] == 2
    assert target_block["parent_headings"] == ["2 商务文件"]
    for sibling in ("2.2   中标服务费承诺函", "2.4   公司实力"):
        block = next(b for b in blocks if b.get("heading") == sibling)
        assert block["level"] == 2
        assert block["parent_headings"] == ["2 商务文件"]

    # body_lead is not strong and body-font → plain body, never a heading, and
    # not fetched from the demotion ledger (it produces no HeadingDecision).
    assert not any(d.text == body_lead for d in result.decisions.values())
    assert not any(b.get("heading") == body_lead for b in blocks)
    assert any(body_lead in (b.get("content") or "") for b in blocks)

    # explicit: genuine multi-sentence prose is still demoted.
    explicit_decision = next(d for d in result.decisions.values() if d.text == explicit)
    assert not explicit_decision.is_heading
    assert "strong_body_multi_sentence" in explicit_decision.rule_trail
    assert "strong_body_demoted" in explicit_decision.rule_trail


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


def test_solo_centered_line_is_heading_under_high_confidence() -> None:
    """§2.2.5 solo centered channel: under a high-confidence FS_base a
    centered base-size line free of every body signal is a heading on its
    own — no cross-run companion needed."""
    records = _body(30)
    records.append(_para("孤立居中标题", size=12.0, alignment="center"))
    result = _gate(records)
    d = next(x for x in result.decisions if x.text == "孤立居中标题")
    assert d.rule_trail[0] == "base_center"


def test_two_line_centered_title_both_admitted() -> None:
    """A two-line centered title (adjacent lines = one run, so it never had
    a cross-run companion) is admitted whole; merge_split_headings later
    joins the pair."""
    records = _body(30)
    records.append(
        _para("广州市城市安全风险综合监测预警平台", size=12.0, alignment="center")
    )
    records.append(_para("建设的工作方案", size=12.0, alignment="center"))
    result = _gate(records)
    texts = _texts(result)
    assert "广州市城市安全风险综合监测预警平台" in texts
    assert "建设的工作方案" in texts


def test_solo_centered_channel_exclusions() -> None:
    """The solo channel's safety exclusions: a centered 成文日期 (every
    format), a letter-free decoration line, and a caption never enter it."""
    from lightrag.parser.docx.smart_heading import guardrails

    excluded = (
        "二〇二六年七月六日",
        "2026.7.31",
        "2026/7/31",
        "2026-7-31",
        "- 1 -",
        "***",
        "——",
        "图 1 系统架构",
    )
    records = _body(30)
    for t in excluded:
        records.append(_para(t, size=12.0, alignment="center"))
        records += _body(2)  # separate runs: exclusion, not anti-poetry
    result = _gate(records, caption_veto=guardrails.caption_prefix_reason)
    texts = _texts(result)
    for t in excluded:
        assert t not in texts


def test_centered_run_of_four_loses_channel_but_not_size_path() -> None:
    """Anti-poetry tightened to >= 4: a 4-line centered cluster loses the
    centered channel as a WHOLE run, but losing the channel is not a veto —
    a big line inside the cluster still rises via size_strong."""
    records = _body(30)
    records.append(_para("居中簇一", size=12.0, alignment="center"))
    records.append(_para("居中簇二", size=12.0, alignment="center"))
    records.append(_para("大字号居中行", size=16.0, alignment="center"))
    records.append(_para("居中簇四", size=12.0, alignment="center"))
    result = _gate(records)
    texts = _texts(result)
    assert "居中簇一" not in texts
    assert "居中簇二" not in texts
    assert "居中簇四" not in texts
    d = next(x for x in result.decisions if x.text == "大字号居中行")
    assert d.rule_trail[0] == "size_strong"


def test_lowconf_truly_isolated_centered_line_needs_companion() -> None:
    """Low confidence keeps the companion requirement: a globally UNIQUE
    centered-shape line (no same-size centered line in ANY other run —
    unlike G5-7's "isolated" line, which has cross-run companions) is not
    admitted."""
    records = []
    records += _body(6, size=10.0)
    records += _body(5, size=12.0)
    records += _body(4, size=14.0)
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert not fs.confidence_high

    records.append(_para("全局唯一居中行", size=14.0, alignment="center"))
    result = _gate(records)
    assert "全局唯一居中行" not in _texts(result)


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
# CB1 graduated demotion (§2.3.3): peel same-size candidates one evidence tier
# at a time instead of the blanket re-estimation wiping every same-size heading.
# ---------------------------------------------------------------------------

# ~70-CJK body paragraph — long enough that a couple between candidates keeps
# average inter-heading spacing above DOCX_SMART_MIN_INTER_HEADING_CHARS (200),
# so density (not sparsity) is the binding CB1 constraint in these fixtures.
_GRAD_BODY = (
    "本段为用于支撑基准字号统计权重并拉开相邻标题间距的正文内容占位文字总长约七十"
    "余个汉字确保标题间平均正文字符数稳定高于阈值不触发稀疏条件仅让密度成为约束若干"
)


def _grad_body_para() -> ParagraphRecord:
    return _para(_GRAD_BODY, size=12.0)


def _gbt_shape_records(
    *, ennum_root: bool = True, strong_body_clauses: int = 0
) -> list[ParagraphRecord]:
    """A GB/T-shaped sub-document: same-size (12pt = FS_base) numbered headings
    of several tiers, spread among long body so density is ~0.49 (over the 0.35
    bar) but spacing is healthy. ``ennum_root`` controls whether the EnNum
    ordinals cover the MultiLevelNum leading components (making EnNum the
    family root) or not.

    ``strong_body_clauses`` adds N CnClause candidates whose text ends in a
    period, so they are admitted first-pass (clause defers strong-body to the
    post-merge sweep) but removed by the look-ahead projection — used to force
    ``density`` (real) and ``cb1_graduated_density`` (projection) apart.
    """
    cands: list[ParagraphRecord] = []
    ennum_labels = range(1, 11) if ennum_root else range(101, 111)
    for n in ennum_labels:  # 10 EnNum "1 …" (root parents 5.1 / 5.1.1 when 1..10)
        cands.append(_para(f"{n} 概述章节{n}", size=12.0))
    for i in range(8):  # 8 MultiLevelNum raw-level 2, tops 1..8
        cands.append(_para(f"{i + 1}.1 子节标题{i}", size=12.0))
    for i in range(14):  # 14 MultiLevelNum raw-level 3, tops cycle 1..7
        cands.append(_para(f"{(i % 7) + 1}.1.{i + 1} 细则条目{i}", size=12.0))
    for i in range(6):  # 6 EnSingleParen list items "a) …"
        cands.append(_para(f"a) 列表项目内容{i}", size=12.0))
    for i in range(strong_body_clauses):  # strong-body CnClause (body-in-disguise)
        cands.append(
            _para(
                f"第{'一二三四五六'[i]}条 本条为伪装成标题的正文内容以句号结尾说明。",
                size=12.0,
            )
        )
    recs = [_grad_body_para(), _grad_body_para()]
    for c in cands:
        recs.append(c)
        recs.append(_grad_body_para())
    return recs


def _grad_gate(records, warnings):
    idx = list(range(len(records)))
    return gate_with_cb1(
        records,
        idx,
        fs_base=document_fs_base(records, idx),
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )


def test_cb1_graduated_recovers_same_size_numbered_headings(caplog) -> None:
    """Fix-proof: same-size numbered headings that the blanket re-estimation
    would wipe are recovered by peeling the weak tiers (list items + deepest
    MLN). The EnNum root and the mln raw-2 tier survive; CB1 does NOT trip."""
    records = _gbt_shape_records(ennum_root=True)
    warnings: dict = {}
    with caplog.at_level(logging.INFO, logger="lightrag"):
        result = _grad_gate(records, warnings)

    # No blanket re-estimation / trip / look-ahead recovery — the graduated
    # path handled it.
    assert not result.cb1_reestimated
    assert not result.cb1_tripped
    assert not result.cb1_strong_body_recovered
    # Weakest body-prior tier first, then deepest MLN; EnNum root deferred so it
    # is NOT reached (stops before it).
    assert result.cb1_graduated_stages == ["en_single_paren", "mln_raw3"]
    assert result.cb1_graduated_ennum_root is True
    assert result.cb1_graduated_demoted == 20  # 6 list items + 14 mln raw-3

    kept = {d.record_index: d for d in result.decisions}
    styles = sorted(d.numbering.style_key for d in kept.values())
    assert styles == ["EnNum"] * 10 + ["MultiLevelNum"] * 8  # root + raw-2 kept

    # Demoted members: original admitting rule kept, plus the two audit marks,
    # output-neutral (never re-rendered from raw text — they are not outline).
    demoted = [d for d in result.demoted if "cb1_graduated_demoted" in d.rule_trail]
    assert len(demoted) == 20
    for d in demoted:
        assert not d.is_heading
        assert d.use_raw_text is False
        assert any(t.startswith("cb1_stage:") for t in d.rule_trail)
        assert d.rule_trail[0] in {"base_series", "base_bold", "base_center"}
    assert warnings["smart_cb1_graduated_demotions"] == 20

    # With no strong-body candidates the real ratio and the projection basis
    # coincide (18/78); their DIVERGENCE is exercised separately in
    # test_cb1_graduated_density_reflects_real_decisions_not_projection.
    assert result.density == pytest.approx(18 / 78, abs=1e-4)
    assert result.cb1_graduated_density == pytest.approx(18 / 78, abs=1e-4)
    assert result.cb1_graduated_inter_chars is not None
    assert result.cb1_graduated_inter_chars >= 200
    assert any("CB1 graduated" in m for m in _log_messages(caplog))


def test_cb1_graduated_density_reflects_real_decisions_not_projection() -> None:
    """``result.density`` keeps its real-decisions definition; the projection
    basis is a SEPARATE field. Strong-body clause candidates are admitted
    first-pass (so they stay in ``decisions``) but removed by the look-ahead
    projection — so the two densities genuinely diverge, and writing the
    projection value back to ``density`` would be caught here. The authoritative
    post-merge sweep then demotes the clauses, converging the final heading
    count on the projection."""
    records = _gbt_shape_records(ennum_root=True, strong_body_clauses=4)
    warnings: dict = {}
    result = _grad_gate(records, warnings)

    assert result.cb1_graduated_stages == ["en_single_paren", "mln_raw3"]
    # The 4 strong-body clauses survive the first-pass gate (clause defers
    # strong-body) and are NOT touched by the ladder, so the real ratio counts
    # them while the projection ratio does not — a strict inequality.
    clauses = [
        d
        for d in result.decisions
        if d.numbering is not None and d.numbering.style_key == "CnClause"
    ]
    assert len(clauses) == 4 and all(d.is_heading for d in clauses)
    assert result.cb1_graduated_density is not None
    assert result.density > result.cb1_graduated_density

    # The authoritative post-merge sweep demotes the body-in-disguise clauses;
    # the final heading count matches the projection the ladder decided on.
    ds = result.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    assert not any(
        d.numbering is not None and d.numbering.style_key == "CnClause" and d.is_heading
        for d in ds
    )
    final_headings = sum(1 for d in ds if d.is_heading)
    assert final_headings == round(result.cb1_graduated_density * result.non_empty)


def test_cb1_graduated_ladder_orders_mln_last_and_ennum_root_deferred() -> None:
    """The ladder fixes the demotion order: weakest body-prior first, then MLN
    deepest-raw-level-first, and the EnNum root relocated to the very end."""
    from lightrag.parser.docx.smart_heading.heading_flow import _cb1_ladder

    present = {
        "en_single_paren": [],
        "en_num": [],
        "cn_num": [],
        "mln_raw2": [],
        "mln_raw3": [],
    }
    assert _cb1_ladder(present, ennum_root=False) == [
        "en_single_paren",
        "en_num",
        "cn_num",
        "mln_raw3",
        "mln_raw2",
    ]
    assert _cb1_ladder(present, ennum_root=True) == [
        "en_single_paren",
        "cn_num",
        "mln_raw3",
        "mln_raw2",
        "en_num",
    ]


def test_cb1_graduated_ennum_root_detection() -> None:
    """EnNum is the MLN root only with >= 2 distinct overlapping ordinals AND
    >= half the MLN leading components covered — a lone {1} overlap is too weak.
    """
    from lightrag.parser.docx.smart_heading.heading_flow import (
        _cb1_ennum_is_mln_root,
    )

    def _mln(top: int) -> HeadingDecision:
        return HeadingDecision(
            record_index=top,
            text=f"{top}.1 子节",
            is_heading=True,
            numbering=classify_numbering(f"{top}.1 子节"),
        )

    def _en(ordinal: int) -> HeadingDecision:
        return HeadingDecision(
            record_index=1000 + ordinal,
            text=f"{ordinal} 章节",
            is_heading=True,
            numbering=classify_numbering(f"{ordinal} 章节"),
        )

    mlns = [_mln(t) for t in (1, 2, 3, 4)]
    assert _cb1_ennum_is_mln_root(mlns, [_en(o) for o in (1, 2, 3, 4)]) is True
    assert _cb1_ennum_is_mln_root(mlns, [_en(1)]) is False  # single {1} overlap
    assert _cb1_ennum_is_mln_root(mlns, [_en(o) for o in (90, 91)]) is False  # none
    assert _cb1_ennum_is_mln_root([], [_en(1), _en(2)]) is False  # no MLN family


def test_cb1_graduated_ennum_root_not_demoted_before_mln() -> None:
    """Contrast: when the EnNum ordinals do NOT cover the MLN tops, EnNum is not
    the root and is demoted at its base-ladder position (before MLN); the MLN
    tiers survive instead."""
    records = _gbt_shape_records(ennum_root=False)
    warnings: dict = {}
    result = _grad_gate(records, warnings)

    assert result.cb1_graduated_ennum_root is False
    assert result.cb1_graduated_stages == ["en_single_paren", "en_num"]
    styles = sorted(d.numbering.style_key for d in result.decisions)
    assert styles == ["MultiLevelNum"] * 22  # all MLN kept, EnNum demoted


def test_cb1_graduated_sparse_only_trigger() -> None:
    """CB1 can trigger on spacing alone (density under the bar). A packed run of
    same-size list items makes the mean inter-heading spacing sparse; peeling
    that one tier restores spacing without touching the real headings."""
    records = [_grad_body_para(), _grad_body_para()]
    for i in range(4):  # 4 well-spaced MLN headings
        records += [
            _para(f"{i + 1}.1 子节标题{i}", size=12.0),
            _grad_body_para(),
            _grad_body_para(),
        ]
    records.append(_grad_body_para())
    for i in range(6):  # a packed run of list items — no body between them
        records.append(_para(f"a) 列表项{i}", size=12.0))
    records += [_grad_body_para(), _grad_body_para()]

    warnings: dict = {}
    result = _grad_gate(records, warnings)

    assert not result.cb1_reestimated and not result.cb1_tripped
    assert result.cb1_graduated_stages == ["en_single_paren"]
    # Density was already within bounds; spacing was the trigger and recovered.
    assert result.cb1_graduated_density is not None
    assert result.cb1_graduated_density <= 0.35
    assert result.cb1_graduated_inter_chars >= 200
    assert sorted(d.numbering.style_key for d in result.decisions) == (
        ["MultiLevelNum"] * 4
    )


def test_cb1_graduated_exhausted_falls_back_to_blanket() -> None:
    """When no ladder prefix converges (exempt outline candidates hold density
    over the bar), the ladder leaves the result untouched and CB1 falls back to
    blanket re-estimation — proving the simulation is side-effect free."""
    records = [_para(_GRAD_BODY, size=10.5) for _ in range(20)]
    for i in range(30):  # exempt outline headings keep density high
        records.append(_para(f"大纲标题{i}", size=10.5, outline_level=0))
    for i in range(10):  # the only demotable tier
        records.append(_para(f"a) 列表项{i}", size=10.5))

    warnings: dict = {}
    result = _grad_gate(records, warnings)

    # Ladder ran (a demotable tier existed) but found no converging prefix:
    # no graduated state, no graduated demotions, blanket path taken instead.
    assert result.cb1_graduated_stages == []
    assert result.cb1_graduated_demoted == 0
    assert "smart_cb1_graduated_demotions" not in warnings
    assert result.cb1_reestimated
    assert not any("cb1_graduated_demoted" in d.rule_trail for d in result.demoted)


def test_cb1_graduated_outline_and_larger_size_exempt() -> None:
    """Outline headings and candidates whose size is above FS_base are never
    demoted by the ladder — only same-size non-outline tiers are."""
    from lightrag.parser.docx.smart_heading.heading_flow import _cb1_stage_tag

    outline = HeadingDecision(
        record_index=0,
        text="1 大纲",
        is_heading=True,
        font_size_pt=12.0,
        outline_level=0,
        numbering=classify_numbering("1 大纲"),
    )
    outline.note("outline")
    bigger = HeadingDecision(
        record_index=1,
        text="1) 列表",
        is_heading=True,
        font_size_pt=14.0,
        numbering=classify_numbering("1) 列表"),
    )
    bigger.note("size_strong")
    same_size = HeadingDecision(
        record_index=2,
        text="a) 列表",
        is_heading=True,
        font_size_pt=12.0,
        numbering=classify_numbering("a) 列表"),
    )
    same_size.note("base_series")

    assert _cb1_stage_tag(outline, 12.0) is None
    assert _cb1_stage_tag(bigger, 12.0) is None
    assert _cb1_stage_tag(same_size, 12.0) == "en_single_paren"


def test_cb1_graduated_flows_through_downstream_leveling() -> None:
    """After graduated demotion the kept candidates are ordinary headings: the
    normal leveling / backfill pipeline runs over them without incident and the
    EnNum root is promoted to the shallowest level."""
    records = _gbt_shape_records(ennum_root=True)
    warnings: dict = {}
    result = _grad_gate(records, warnings)

    ds = result.decisions
    assign_levels_by_size(ds)
    backfill_top_level(ds, warnings=warnings)
    align_numbering_series(ds)
    assert ds  # non-empty
    assert all(d.is_heading and d.level is not None and d.level >= 1 for d in ds)
    root_levels = [d.level for d in ds if d.text.startswith("1 概述")]
    assert root_levels and min(d.level for d in ds) in root_levels


def test_cb1_graduated_audit_fields(monkeypatch) -> None:
    """run_smart_heading surfaces the graduated-demotion audit block."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "10")
    records = _gbt_shape_records(ennum_root=True)

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=None,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    subs = result.audit["sub_documents"]
    grad = next(s["cb1_graduated"] for s in subs if "cb1_graduated" in s)
    assert grad["stages_applied"] == ["en_single_paren", "mln_raw3"]
    assert grad["ennum_root"] is True
    assert grad["demoted_count"] == 20
    assert grad["projected_density"] == pytest.approx(18 / 78, abs=1e-3)
    assert grad["projected_inter_chars"] >= 200


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


def test_backfill_only_promotes_top_level_roots() -> None:
    """test11 bug: backfill must absorb ONLY the genuine top-level parents,
    not a nested body list sharing the bare numbering key. The gate keys the
    linkage channel on the parent RECORD IDENTITY (not the ordinal value): a
    body list ``1./2.`` COLLIDES ordinals with the real roots ``1./2.``, so
    an ordinal-membership test would wrongly promote it. Here the roots are
    10.5pt (< the 14pt child-size threshold) and carry no outline, so ONLY
    the record-index linkage channel can lift them — isolating the fix."""
    specs = [
        # (text, size, outline_level, level)
        ("1. 价格文件", 10.5, None, 4),  # root: linked (scope holds 1.1), no outline
        ("1.1 开标一览表", 14.0, None, 3),  # MLN child (top_ordinal 1)
        ("2. 商务文件", 10.5, None, 4),  # root: linked (scope holds 2.1)
        ("2.1 投标函", 14.0, None, 3),  # MLN child (top_ordinal 2)
        (
            "1. 我方按招标文件递交投标文件正本。",
            10.5,
            None,
            6,
        ),  # body, ordinal 1 COLLIDES
        (
            "2. 我方承认招标人有权决定中标人。",
            10.5,
            None,
            6,
        ),  # body, ordinal 2 COLLIDES
    ]
    ds = []
    for i, (text, size, outline, level) in enumerate(specs):
        d = _decision(text, size=size, idx=i)
        d.level = level
        d.outline_level = outline
        ds.append(d)
    backfill_top_level(ds, warnings={})
    # The two roots (linked by record identity) are absorbed as MLN raw 1.
    for r in (ds[0], ds[2]):
        assert r.numbering.style_key == "MultiLevelNum" and r.numbering.raw_level == 1
        assert "backfill_top_level" in r.rule_trail
    # The body list — colliding ordinals 1/2 — is NOT promoted (would be, if
    # the gate matched ordinal VALUES instead of parent record identity).
    for b in (ds[4], ds[5]):
        assert b.numbering.style_key == "EnNum"
        assert "backfill_top_level" not in b.rule_trail
    assert sum("backfill_top_level" in d.rule_trail for d in ds) == 2


def test_outlined_part_root_survives_end_to_end(monkeypatch) -> None:
    """test11 shape end to end (through the assembler, where parent_headings
    is generated): outline-0 EnNum part roots with same-key body lists under
    their sub-sections must SURVIVE as headings — the body lists demote, the
    roots do not. Sizes are cleanly banded (root 16 > child 14 > body 12) so
    the nesting is deterministic; the same-size test11 shape is covered by the
    real-doc rerun in verification."""
    import json

    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    def _sents(n, p):
        return [
            _para(
                f"{p}第{i}段正常长度的正文内容，用来撑起基准字号统计，本段以句号结尾。",
                size=12.0,
            )
            for i in range(n)
        ]

    records = (
        _sents(15, "引")
        + [_para("1. 价格文件", size=16.0, outline_level=0)]
        + _sents(6, "价")
        + [_para("1.1 开标一览表", size=14.0)]
        + _sents(6, "开")
        + [_para("2. 商务文件", size=16.0, outline_level=0)]
        + _sents(6, "商")
        + [_para("2.1 投标函", size=14.0)]
        + [
            _para("1. 我方按招标文件递交投标文件正本以及相关材料。", size=12.0),
            _para("2. 我方承认招标人有权决定中标人的相关权利。", size=12.0),
        ]
        + _sents(6, "尾")
    )

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": False, "headings": [], "body": []}, ensure_ascii=False
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
    meta: dict = {}
    blocks = _assemble_blocks_smart(records, result, {}, meta)

    def _block(needle):
        return next(b for b in blocks if needle in (b.get("heading") or ""))

    assert _block("1. 价格文件")["level"] == 1
    assert _block("2. 商务文件")["level"] == 1
    assert _block("1.1 开标一览表")["parent_headings"] == ["1. 价格文件"]
    assert _block("2.1 投标函")["parent_headings"] == ["2. 商务文件"]
    # The body EnNum lists never became headings, and never got the top-level
    # backfill / outline-anchor rules.
    assert not any(
        (b.get("heading") or "").startswith(("1. 我方", "2. 我方")) for b in blocks
    )
    for idx, r in enumerate(records):
        if r.text.startswith(("1. 我方", "2. 我方")):
            d = result.decisions.get(idx)
            assert d is None or (
                not d.is_heading
                and "backfill_top_level" not in d.rule_trail
                and "anchor_outline_series" not in d.rule_trail
            )


def test_backfill_noop_without_mln() -> None:
    """No MultiLevelNum headings → backfill is a no-op (guards the legal-doc
    corpus test8/test9, which have no MLN and must be untouched)."""
    ds = [
        _decision("1. 总则", size=14.0, idx=0),
        _decision("2. 附则", size=14.0, idx=1),
    ]
    for d in ds:
        d.level = 2
    backfill_top_level(ds, warnings={})
    assert all(d.numbering.style_key == "EnNum" for d in ds)
    assert all("backfill_top_level" not in d.rule_trail for d in ds)


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
    """A10 (§2.2.4, the veto side — the only partition side with force): an
    LLM body vote strips an otherwise strong-size candidate; without the
    vote it is admitted."""
    records = _body(30)
    records.append(_para("被判为正文的大字号行", size=16.0))
    idx = len(records) - 1

    admitted = _gate(records)
    assert "被判为正文的大字号行" in _texts(admitted)

    vetoed = _gate(records, llm_body_vetoes={idx})
    assert "被判为正文的大字号行" not in _texts(vetoed)


def test_llm_grant_carries_no_admission_force() -> None:
    """An LLM heading vote is audit-only (the test9 regression: a
    right-aligned running-header line the LLM named a heading became a
    level-2 ghost): a granted base-size line with no normal signal stays
    body and is recorded in ``grant_rejected``, output-neutral."""
    records = _body(30)
    records.append(_para("GB/T 9704—2012", size=12.0, alignment="right"))
    idx = len(records) - 1

    result = _gate(records, llm_heading_grants={idx})
    assert "GB/T 9704—2012" not in _texts(result)
    assert [d.record_index for d in result.grant_rejected] == [idx]
    rej = result.grant_rejected[0]
    assert rej.is_heading is False
    assert rej.level is None
    assert rej.use_raw_text is False  # output-neutral ledger row
    assert "llm_grant_rejected" in rej.rule_trail


def test_llm_grant_does_not_block_normal_admission() -> None:
    """A granted paragraph that passes a normal rule is admitted under that
    rule (the grant neither helps nor hinders) and is not 'rejected'."""
    records = _body(30)
    records.append(_para("真正的大字号标题", size=16.0))
    idx = len(records) - 1

    result = _gate(records, llm_heading_grants={idx})
    d = next(x for x in result.decisions if x.record_index == idx)
    assert d.rule_trail[0] == "size_strong"
    assert result.grant_rejected == []


def test_llm_grant_rejected_with_numbering_veto_single_ledger_row() -> None:
    """A granted paragraph whose numbering was ALSO homophone-vetoed leaves
    ONE ledger decision carrying both marks (a second row would be dropped
    by the caller's setdefault merge), in ``grant_rejected`` — not doubled
    into ``veto_suppressed``."""
    records = _body(30)
    records.append(_para("一、被撤销编号且被点名的行", size=12.0))
    idx = len(records) - 1

    result = _gate(
        records,
        llm_heading_grants={idx},
        numbering_veto=lambda _c, _t: "homophone_stub",
    )
    assert "一、被撤销编号且被点名的行" not in _texts(result)
    assert result.veto_suppressed == []
    assert [d.record_index for d in result.grant_rejected] == [idx]
    trail = result.grant_rejected[0].rule_trail
    assert "homophone_stub" in trail
    assert "numbering_veto_suppressed" in trail
    assert "llm_grant_rejected" in trail


def test_llm_grant_rejected_end_to_end_audit(monkeypatch) -> None:
    """test9 国标范例 end-to-end: the LLM rejects the 目次-page window as a
    title block but names every line a heading (headings=[0,1,2]). Only the
    16pt 目次 line survives (size_strong, on its own signal); the base-size
    running-header and TOC lines are rejected grants — counted once against
    the final gate result and ledgered in ``audit["decisions"]`` as
    output-neutral non-headings."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    records = [
        _para("GB/T 9704—2012", size=10.5, alignment="right"),  # running header
        _para("目  次", size=16.0, alignment="center"),
        _para("前言", size=10.5),  # a plain-text TOC entry
    ]
    records += _body(30, size=10.5)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {
                "is_title_block": False,
                "main_title": None,
                "sub_title": None,
                "doc_number": None,
                "classification": None,
                "publisher": None,
                "date": None,
                "headings": [0, 1, 2],
                "body": [],
            },
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
    assert result.decisions[1].is_heading  # 目次: admitted by size_strong
    for idx in (0, 2):  # granted base-size lines stay body, rule-tagged
        d = result.decisions[idx]
        assert d.is_heading is False
        assert "llm_grant_rejected" in d.rule_trail
    assert warnings["smart_llm_grant_rejected"] == 2
    rows = [r for r in result.audit["decisions"] if "llm_grant_rejected" in r["rules"]]
    assert len(rows) == 2
    assert all(r["is_heading"] is False and r["level"] is None for r in rows)


def test_merge_ledger_only_merges_and_counts_once() -> None:
    """The ledger merge shared by the normal path and the CB1 fallback:
    rows ``setdefault`` in (an existing real decision wins) and the
    grant-rejected warning counts once per FINAL gate result."""
    from lightrag.parser.docx.smart_heading.heading_flow import (
        GateResult,
        _merge_ledger_only,
    )

    records = _body(5)
    fs = document_fs_base(records, range(len(records)))
    rej = HeadingDecision(record_index=3, text="x", is_heading=False)
    rej.note("llm_grant_rejected")
    supp = HeadingDecision(record_index=4, text="y", is_heading=False)
    supp.note("numbering_veto_suppressed")
    gate = GateResult(
        decisions=[],
        fs_base=fs,
        density=0.0,
        veto_suppressed=[supp],
        grant_rejected=[rej],
    )
    existing = HeadingDecision(record_index=3, text="real", is_heading=True)
    decisions = {3: existing}
    warnings: dict = {}
    _merge_ledger_only(decisions, gate, warnings)
    assert decisions[3] is existing
    assert decisions[4] is supp
    assert warnings["smart_llm_grant_rejected"] == 1


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


def test_title_block_multiline_main_title_flattened(monkeypatch) -> None:
    """A soft-break cover title echoed by the LLM with its ``\\n`` intact must
    land single-line everywhere it fans out: result.doc_title, the meta
    doc_title key, the block heading, and every descendant's
    parent_headings."""
    import json

    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    records = [
        _para("行业观察\n白皮书", size=14.0),
        _para("某某研究院", size=12.0),
    ]
    records += _body(15, size=12.0)
    records.append(_para("第一章 总则", size=12.0, outline_level=0))
    records += _body(15, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "行业观察\n白皮书"},
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

    meta: dict = {}
    blocks = _assemble_blocks_smart(records, result, {}, meta)
    assert meta["doc_title"] == "行业观察白皮书"
    title = next(b for b in blocks if b.get("is_title_block"))
    assert title["heading"] == "行业观察白皮书"
    chapter = next(b for b in blocks if b["heading"] == "第一章 总则")
    assert chapter["parent_headings"] == ["行业观察白皮书"]
    for b in blocks:
        assert "\n" not in b["heading"]
        assert all("\n" not in h for h in b["parent_headings"])


def test_assembler_doc_title_empty_without_title_block() -> None:
    """Smart mode: the title block is the ONLY doc_title source. With no
    title verdict the assembler records an explicitly EMPTY doc_title (a
    "前言"-style first heading must not masquerade as the document title),
    while first_heading keeps its legacy any-heading semantics."""
    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult

    records = [_para("前言", outline_level=0), _para("正文一句。")]
    result = SmartHeadingResult(
        decisions={
            0: HeadingDecision(record_index=0, text="前言", is_heading=True, level=1)
        },
        toc_indices=set(),
        doc_title=None,
        audit={},
    )
    meta: dict = {}
    _assemble_blocks_smart(records, result, {}, meta)
    assert meta["doc_title"] == ""
    assert meta["first_heading"] == "前言"


def test_title_block_judge_receives_fs_base(monkeypatch) -> None:
    """Call-site guard: run_smart_heading must hand the global FS_base
    initial value to judge_title_block — the font-size evidence legend
    depends on it, and a dropped kwarg silently renders the legend-free
    prompt (the M1212 regression would come back unnoticed)."""
    import json

    from lightrag.parser.docx.smart_heading import title_block as tb
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    records = [
        _para("行业观察白皮书", size=14.0),
        _para("某某研究院", size=12.0),
    ]
    records += _body(30, size=12.0)

    captured: dict = {}
    real_judge = tb.judge_title_block

    def _spy(candidate, recs, judge, **kw):
        captured["fs_base_pt"] = kw.get("fs_base_pt")
        return real_judge(candidate, recs, judge, **kw)

    # run_smart_heading imports title_block function-locally, so patching the
    # module attribute intercepts its call.
    monkeypatch.setattr(tb, "judge_title_block", _spy)

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
    assert captured["fs_base_pt"] == 12.0  # the char-weighted body mode


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
    assert meta["doc_title"] == "产品标准化大纲某某模块"

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
# clause-class numbering (条/款/项, Article/§) defers to the post-merge sweep
# ---------------------------------------------------------------------------


def test_early_strong_body_keys_covers_only_chapter_classes() -> None:
    """Chapter classes are strong-body-checked at recognition (per-paragraph,
    no propagation); clause classes are NOT — they defer to the post-merge
    sweep so CB2 can demote a body-in-disguise run as a whole series."""
    from lightrag.parser.docx.smart_heading.heading_flow import (
        EARLY_STRONG_BODY_KEYS,
    )
    from lightrag.parser.docx.smart_heading.style_key import (
        CN_CHAPTER,
        CN_CLAUSE,
        EN_CHAPTER,
        EN_CLAUSE,
    )

    assert CN_CHAPTER in EARLY_STRONG_BODY_KEYS
    assert EN_CHAPTER in EARLY_STRONG_BODY_KEYS
    assert CN_CLAUSE not in EARLY_STRONG_BODY_KEYS
    assert EN_CLAUSE not in EARLY_STRONG_BODY_KEYS


def test_body_clause_series_demoted_via_post_merge_sweep() -> None:
    """Regression (test8 应急管理部令第11号): a mostly-body 第X条 series is
    demoted as a WHOLE by the post-merge CB2 sweep — including short colon-lead
    survivors (第六/八条) that trip no strong-body rule on their own.

    Must enter from the gate, NOT hand-built decisions: the bug is that CnClause
    was strong-body-demoted at RECOGNITION (per-paragraph, no propagation), so
    the body clauses left the candidate list before the sweep and could not act
    as the CB2 hits that drag the survivors down. With CnClause early (buggy)
    ``result.decisions`` keeps only the two colon survivors → no hits → they
    leak as headings; with CnClause deferred (fixed) all clauses stay → the
    。-ending ones are hits → CB2 (hit share 60% ≥ 20%, no outline) demotes the
    whole series. This test is RED on the pre-fix code and GREEN after.
    """
    sentence = (
        "为了加强安全生产领域信用体系建设，规范严重失信主体名单管理，依法制定本办法。"
    )
    survivor_six = "第六条  下列单位及其人员应当列入严重失信主体名单："
    survivor_eight = "第八条  应急管理部门可以采取下列管理措施："
    records = _body(20) + [
        _para(f"第一条  {sentence}"),  # 。-ending → individual hit
        _para(f"第二条  {sentence}"),  # 。-ending → individual hit
        _para(f"第三条  {sentence}"),  # 。-ending → individual hit
        _para(survivor_six),  # colon, < 60 chars → NOT an individual hit
        _para(survivor_eight),  # colon, < 60 chars → NOT an individual hit
    ]
    warnings: dict = {}
    result = _gate(records, warnings=warnings)
    ds = result.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)

    by_text = {d.text: d for d in ds}
    # The colon survivors are exactly what leaked pre-fix; CB2 must demote them.
    assert by_text[survivor_six].is_heading is False
    assert by_text[survivor_eight].is_heading is False
    # No 第X条 survives as a heading at all.
    assert not any(d.is_heading and "条" in d.text for d in ds)


def test_short_bare_clause_series_survive_as_headings() -> None:
    """Guard against over-demotion: a 第X条 series whose every member is a short
    bare heading (no strong-body feature) has zero hits, so CB2 never
    propagates and all survive. Passes both before and after the fix — the
    safety net for the deferral change (cf. e2e regulation clauses)."""
    records = _body(20) + [
        _para("第一条"),
        _para("第二条"),
        _para("第三条"),
        _para("第四条"),
    ]
    result = _gate(records)
    ds = result.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings={})
    assert {d.text for d in ds if d.is_heading} == {
        "第一条",
        "第二条",
        "第三条",
        "第四条",
    }


def test_body_article_series_demoted_via_post_merge_sweep() -> None:
    """EnClause parity with CnClause: a mostly-body ``Article`` series demotes
    as a whole, dragging a short colon-lead survivor down. Uses ``Article``
    (not ``Section``/``Sec`` — those classify as EnChapter, which stays early)."""
    long_en = (
        "This article establishes the comprehensive requirements and the "
        "detailed obligations applicable to all covered entities herein."
    )  # > 60 chars → _stub length hit
    survivor = "Article 5  introduces the following measures:"  # colon, < 60
    records = _body(20) + [
        _para(f"Article 1  {long_en}"),
        _para(f"Article 2  {long_en}"),
        _para(f"Article 3  {long_en}"),
        _para(survivor),
    ]
    result = _gate(records)
    ds = result.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings={})
    by_text = {d.text: d for d in ds}
    assert by_text[survivor].is_heading is False
    assert not any(d.is_heading and d.text.startswith("Article") for d in ds)


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
    """Leak path (e.g. an outline/size-admitted line, or a merge whose joined
    text newly reads as imprint): a surviving imprint heading is caught by
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


def test_imprint_region_demoted_when_multi_line_title_block_follows(
    monkeypatch,
) -> None:
    """A later multi-paragraph cover remains valid evidence for a 公文汇编
    boundary, so the preceding 抄送…印发 region is force-demoted as before."""
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
        _para("（2026年版）", size=12.0),  # makes the cover a multi-window
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


def test_imprint_region_demoted_for_later_single_line_cover(monkeypatch) -> None:
    """用户裁决翻转（原 test_imprint_region_not_demoted_for_later_single_line）：
    版记收尾是强文档边界信号，其后的单行大字号标题（版记吸收中间的换页/
    空段）经 ``imprint_single`` 通道成为候选；LLM 判真 → title block 确认、
    版记区域按 §版记 条件降级——:1806 多行封面链路的单行镜像。无版记的
    "换页+单行"仍不成候选（title_block 层 B8 测试承接原负面语义）。"""
    import json

    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    records = (
        doc1
        + [
            _para("抄送：各区人民政府", size=12.0),
            _para("中间说明行不带大纲", size=12.0),
            _para("某某办公室 2026年6月30日 印发", size=12.0, outline_level=1),
            ParagraphRecord(kind="empty_para"),
            _para("数字政府建设白皮书", size=18.0, page_break_before=True),
        ]
        + _body(20)
    )
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
    d_cover = result.decisions[cover_idx]
    assert d_cover.is_title_block and d_cover.level == 0
    assert "title_block:imprint_single" in d_cover.rule_trail
    # The trailing 印发 outline line is force-demoted: the confirmed title
    # block immediately follows the region (the 公文汇编 boundary).
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


# ---------------------------------------------------------------------------
# Zero-visible-char placeholder paragraphs (pure <drawing>/<object>/<equation>)
# ---------------------------------------------------------------------------
# Production records for placeholder-only paragraphs carry
# ``visible_char_count == 0`` (placeholder markup never enters run_features).
# Manually built test records MUST set it explicitly: the default None makes
# ``_record_weight`` fall back to ``len(text.strip())`` and the placeholder
# tag text would count as visible characters.


def test_zero_weight_placeholder_not_size_promoted() -> None:
    """The test11 offender: a 14pt drawing-only paragraph over FS_base=12
    must not take size_strong — its size is synthesized style metadata, not
    measured text."""
    records = _body(20) + [
        _para('<drawing id="1" path="x.emf" />', size=14.0, visible_char_count=0)
    ]
    result = _gate(records)
    assert result.decisions == []
    assert result.demoted == []
    assert result.veto_suppressed == []


def test_zero_weight_placeholder_not_center_promoted() -> None:
    """Even at FS_base (the post-fix test11 size), a centered short
    placeholder must not take the base_center channel."""
    records = _body(20) + [
        _para('<drawing id="1" />', size=12.0, alignment="center", visible_char_count=0)
    ]
    result = _gate(records)
    assert result.decisions == []


def test_pure_equation_paragraph_never_candidate() -> None:
    """Invariant pin: a pure-OMML paragraph (no w-namespace runs → size=None,
    zero visible chars) is never a candidate on any channel."""
    records = _body(20) + [
        _para(
            "<equation>E=mc^2</equation>",
            size=None,
            alignment="center",
            visible_char_count=0,
        )
    ]
    result = _gate(records)
    assert result.decisions == []
    assert result.demoted == []


def test_zero_weight_not_weak_companion() -> None:
    """A placeholder at fs+0.5 is no weak-pair companion: the lone real
    12.5pt line must NOT be admitted via weak_pair on the placeholder's
    synthesized-size second vote."""
    records = _body(20) + [
        _para("候选标题甲", size=12.5),
        _para('<drawing id="1" />', size=12.5, visible_char_count=0),
    ]
    result = _gate(records)
    assert "候选标题甲" not in _texts(result)


def test_zero_weight_placeholder_not_numbering_series_companion() -> None:
    """An auto-numbered image line must not lend the second series vote: the
    lone real numbered line stays a singleton (no base_series admission)."""
    records = _body(20) + [
        _para("3.1 真实编号标题", size=12.0),
        _para('3.2 <drawing id="1" />', size=12.0, visible_char_count=0),
    ]
    result = _gate(records)
    assert "3.1 真实编号标题" not in _texts(result)
    assert "3.2 " not in " ".join(_texts(result))


def test_zero_weight_placeholder_transparent_in_centered_run() -> None:
    """A decorative centered placeholder between centered title lines is
    TRANSPARENT: it neither joins the centered run (which would push a
    3-line cluster over the anti-poetry cap) nor breaks it."""
    records = _body(20) + [
        _para("居中标题一", size=12.0, alignment="center"),
        _para(
            '<drawing id="1" />', size=12.0, alignment="center", visible_char_count=0
        ),
        _para("居中标题二", size=12.0, alignment="center"),
        _para("居中标题三", size=12.0, alignment="center"),
    ]
    result = _gate(records)
    texts = _texts(result)
    for t in ("居中标题一", "居中标题二", "居中标题三"):
        assert t in texts, t
    assert '<drawing id="1" />' not in texts


def test_outline_zero_weight_placeholder_demoted() -> None:
    """User ruling: an author-styled (outlineLvl) placeholder paragraph is
    NOT a heading either — it takes an explicit I2 demotion decision, and the
    retention check accepts the ``placeholder_demoted`` tag."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )

    records = _body(20) + [
        _para('<drawing id="1" />', size=14.0, outline_level=0, visible_char_count=0)
    ]
    result = _gate(records)
    assert result.decisions == []
    assert len(result.demoted) == 1
    dem = result.demoted[0]
    assert dem.is_heading is False
    assert dem.use_raw_text is True
    assert "placeholder_demoted" in dem.rule_trail
    assert "zero_visible_chars" in dem.rule_trail
    merged = list(result.decisions) + result.demoted
    assert verify_baseline_heading_retention(records, merged) == []


def test_outline_only_fallback_demotes_placeholder(caplog) -> None:
    """The CB4/CB1 fallback path (_outline_only_decisions) demotes a
    zero-weight outline paragraph instead of reverting it to a heading, and
    counts + logs exactly once (it is a terminal path)."""
    from lightrag.parser.docx.smart_heading.heading_flow import (
        _outline_only_decisions,
    )

    records = _body(5) + [
        _para("正常大纲标题", size=12.0, outline_level=0),
        _para('<drawing id="1" />', size=12.0, outline_level=0, visible_char_count=0),
    ]
    warnings: dict = {}
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        out = _outline_only_decisions(records, range(len(records)), warnings=warnings)
    by_idx = {d.record_index: d for d in out}
    real = by_idx[len(records) - 2]
    assert real.is_heading and "subdoc_fallback_outline_only" in real.rule_trail
    dem = by_idx[len(records) - 1]
    assert dem.is_heading is False
    assert dem.use_raw_text is True
    assert "placeholder_demoted" in dem.rule_trail
    assert warnings["smart_placeholder_demotions"] == 1
    logs = [m for m in _log_messages(caplog) if "placeholder" in m]
    assert len(logs) == 1


@pytest.mark.parametrize("cb1_terminal", ["converged", "tripped"])
def test_placeholder_demotion_counted_once_across_cb1(
    monkeypatch, caplog, cb1_terminal
) -> None:
    """``smart_placeholder_demotions`` counts at the FINAL adoption point
    only: CB1 runs gate_candidates twice with the same warnings dict, and an
    eager in-gate count would double. Both CB1 terminal states — (i) the
    re-estimated second pass is adopted, (ii) the breaker trips and the
    sub-document falls back to _outline_only_decisions — must count the one
    outline placeholder exactly once and emit exactly one I2 log line."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    if cb1_terminal == "converged":
        # 30 bare 12pt lines over a 10.5pt body: CB1 re-estimates FS_base to
        # 12 and the second pass converges (test_cb1_reestimation shape).
        records = _body(60, size=10.5)
        records += [_para(f"伪标题短语{i}", size=12.0) for i in range(30)]
    else:
        # Question-bank shape: density stays over the bar after re-estimation
        # → the breaker trips and the sub-document falls back.
        records = []
        for i in range(40):
            records.append(_para(f"{i + 1}. 选择题题干第{i}题", size=12.0))
            records.append(_para("A. 选项甲  B. 选项乙", size=10.5))
    records.append(
        _para('<drawing id="1" />', size=12.0, outline_level=0, visible_char_count=0)
    )
    placeholder_idx = len(records) - 1

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps({"is_title_block": False}, ensure_ascii=False)

    warnings: dict = {}
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = run_smart_heading(
            records,
            llm_judge=_llm,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    assert result is not None
    # Pin the ACTUAL terminal state: a future drift in the fixture shapes
    # that stops reaching the target branch must fail loudly, not pass
    # vacuously. Both states prove CB1 re-estimated (two gate passes ran —
    # the double-count premise); they differ in the adoption path.
    sub_docs = result.audit["sub_documents"]
    assert any("cb1_reestimated_fs" in s for s in sub_docs)
    if cb1_terminal == "tripped":
        assert any(s.get("fallback") == "cb1_density" for s in sub_docs)
    else:
        assert not any(s.get("fallback") == "cb1_density" for s in sub_docs)
    assert warnings["smart_placeholder_demotions"] == 1
    dem = result.decisions[placeholder_idx]
    assert dem.is_heading is False
    assert "placeholder_demoted" in dem.rule_trail
    i2_logs = [
        m for m in _log_messages(caplog) if "zero-visible-char outline paragraph" in m
    ]
    assert len(i2_logs) == 1


def test_zero_weight_llm_grant_leaves_rejected_ledger() -> None:
    """LLM audit contract for zero-weight paragraphs: a granted placeholder
    line must land in ``grant_rejected`` (not vanish silently), and the
    final-adoption merge counts it once and inserts the ledger row.

    Layered on purpose: a pure-drawing line cannot obtain a grant through
    the full run_smart_heading — the title-block window strips drawing tags
    (no semantic text → not in the LLM index_map) — so the grant is injected
    at the gate layer and the merge layer is exercised directly."""
    from lightrag.parser.docx.smart_heading.heading_flow import _merge_ledger_only

    records = _body(20) + [_para('<drawing id="1" />', size=14.0, visible_char_count=0)]
    idx = len(records) - 1
    result = _gate(records, llm_heading_grants={idx})
    assert result.decisions == []
    assert [d.record_index for d in result.grant_rejected] == [idx]
    trail = result.grant_rejected[0].rule_trail
    assert "zero_visible_chars" in trail
    assert "llm_grant_rejected" in trail

    decisions: dict = {}
    warnings: dict = {}
    _merge_ledger_only(decisions, result, warnings)
    assert warnings["smart_llm_grant_rejected"] == 1
    assert decisions[idx] is result.grant_rejected[0]


# ---------------------------------------------------------------------------
# mid-document title-block gate: end-to-end + audit enrichment
# ---------------------------------------------------------------------------


def test_mid_document_isolated_title_never_reaches_llm(monkeypatch) -> None:
    """The test11 shape end to end: a page-break 16pt caption line beside a
    metadata line must not form any title-block window — zero LLM calls, no
    candidates, no doc_title; the line stands as an ordinary size_strong
    level-1 heading and the document stays ONE sub-document."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = (
        _body(20)
        + [
            _para("填报单位：某某公司", size=12.0),
            _para("外购外协价格明细表", size=16.0, page_break_before=True),
        ]
        + _body(20)
    )
    title_idx = 21

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        raise AssertionError("LLM must not be called for a gated document")

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
    assert result.audit["llm_calls"] == 0
    assert result.audit["title_block_candidates"] == []
    assert result.doc_title is None
    d = result.decisions[title_idx]
    assert d.is_heading and not d.is_title_block
    assert d.level == 1
    assert "size_strong" in d.rule_trail
    assert len(result.audit["sub_documents"]) == 1


def test_title_block_candidate_audit_records_verdict_content(monkeypatch) -> None:
    """Audit enrichment: each candidate row carries the window end, the LLM's
    main_title (true verdicts) and the partition strength (false verdicts)."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = [
        _para("产品发布白皮书", size=22.0),
        _para("（2026年版）", size=12.0),
    ] + _body(30)

    def _true_llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "产品发布白皮书"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_true_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    (row,) = result.audit["title_block_candidates"]
    assert row["trigger"] == "multi_window"
    assert (row["start"], row["end"]) == (0, 2)
    assert row["is_title_block"] is True
    assert row["main_title"] == "产品发布白皮书"
    assert row["heading_grants"] == 0 and row["body_vetoes"] == 0

    def _false_llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": False, "headings": [], "body": [0, 1]},
            ensure_ascii=False,
        )

    result2 = run_smart_heading(
        records,
        llm_judge=_false_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result2 is not None
    (row2,) = result2.audit["title_block_candidates"]
    assert row2["is_title_block"] is False
    assert row2["main_title"] is None
    assert row2["body_vetoes"] == 2


def test_imprint_single_cover_behind_logo_confirms_region(monkeypatch) -> None:
    """Review regression: the imprint confirm scan must walk with the SAME
    eyes as the imprint_single channel — a pure logo paragraph + section
    break between the 版记 tail and the single-line cover must not stop the
    scan, or the confirmed cover leaves the region undemoted."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    records = (
        doc1
        + [
            _para("抄送：各区人民政府", size=12.0),
            _para("中间说明行不带大纲", size=12.0),
            _para("某某办公室 2026年6月30日 印发", size=12.0, outline_level=1),
            _para('<drawing id="seal-1" />', size=12.0, visible_char_count=0),
            ParagraphRecord(kind="section_break"),
        ]
        + [_para("数字政府建设白皮书", size=18.0)]
        + _body(20)
    )
    closer_idx = len(doc1) + 2
    cover_idx = len(doc1) + 5

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
    d_cover = result.decisions[cover_idx]
    assert d_cover.is_title_block and d_cover.level == 0
    assert "title_block:imprint_single" in d_cover.rule_trail
    dem = result.decisions[closer_idx]
    assert dem.is_heading is False and dem.use_raw_text is True
    assert dem.rule_trail[-2:] == ["imprint_region", "strong_body_demoted"]
    assert warnings["smart_imprint_region_demotions"] == 1


def test_logo_led_multi_cover_still_confirms_region(monkeypatch) -> None:
    """A multi window that OPENS on a seal/logo image paragraph keys its
    decision on that image; the confirm scan (which skips non-content
    records) lands on the block's title LINE — matching the full member set
    keeps the 版记 confirmation working (matching starts only would not)."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    records = (
        doc1
        + [
            _para("抄送：各区人民政府", size=12.0),
            _para("中间说明行不带大纲", size=12.0),
            _para("某某办公室 2026年6月30日 印发", size=12.0, outline_level=1),
            ParagraphRecord(kind="empty_para"),
            _para('<drawing id="seal-1" />', size=12.0, visible_char_count=0),
            _para("数字政府建设白皮书", size=18.0),
        ]
        + _body(20)
    )
    closer_idx = len(doc1) + 2
    logo_idx = len(doc1) + 4

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
    d_cover = result.decisions[logo_idx]  # image-led window keys on the logo
    assert d_cover.is_title_block and d_cover.level == 0
    assert "title_block:multi_window" in d_cover.rule_trail
    dem = result.decisions[closer_idx]
    assert dem.is_heading is False
    assert dem.rule_trail[-2:] == ["imprint_region", "strong_body_demoted"]
    assert warnings["smart_imprint_region_demotions"] == 1


# ---------------------------------------------------------------------------
# merged doc-title length bounding — four-way consistency (H1 / doc_title /
# first_heading / descendant parent_headings root) under truncation
# ---------------------------------------------------------------------------


def _assemble_bounded_title(monkeypatch, cap_env: str, main: str, sub):
    """Drive the assembler for a level-0 title block whose composed doc-title is
    (potentially) truncated, plus one child heading, and return
    (doc_heading, blocks, meta)."""
    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult
    from lightrag.parser.docx.smart_heading.title_block import compose_doc_title

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", cap_env)
    doc_heading = compose_doc_title(main, sub)

    records = [
        _para(main, size=22.0),  # 0: title-block lead
        _para("封面说明正文一句。", size=12.0),  # 1: body owned by the title block
        _para("第一章 总则", size=14.0),  # 2: child heading
        _para("章节正文一句。", size=12.0),  # 3: body under the child
    ]
    result = SmartHeadingResult(
        decisions={
            0: HeadingDecision(
                record_index=0,
                text=main,
                is_heading=True,
                is_title_block=True,
                level=0,
                doc_title_heading=doc_heading,
                member_indices=(0,),
            ),
            2: HeadingDecision(
                record_index=2, text="第一章 总则", is_heading=True, level=1
            ),
        },
        toc_indices=set(),
        doc_title=doc_heading,
        audit={},
    )
    meta: dict = {}
    blocks = _assemble_blocks_smart(records, result, {}, meta)
    return doc_heading, blocks, meta


def _assert_four_way(doc_heading, blocks, meta) -> None:
    title = next(b for b in blocks if b.get("is_title_block"))
    child = next(b for b in blocks if b["heading"] == "第一章 总则")
    assert title["heading"] == doc_heading
    assert meta["doc_title"] == doc_heading
    assert meta["first_heading"] == doc_heading
    assert child["parent_headings"][0] == doc_heading


def test_merged_doc_title_weighted_truncation_four_way(monkeypatch) -> None:
    """A merged CJK doc title over the default weighted cap truncates with '...'
    and the same value lands at all four sites."""
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    from lightrag.parser.docx.smart_heading.guardrails import weighted_char_length

    doc_heading, blocks, meta = _assemble_bounded_title(
        monkeypatch,
        str(DEFAULT_DOCX_SMART_HEADING_MAX_CHARS),
        "标" * 60,  # weighted 180
        "案" * 30,  # merged weighted 180 + 2 + 90 = 272 -> truncated
    )
    assert doc_heading.endswith("...")
    assert weighted_char_length(doc_heading) <= DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    _assert_four_way(doc_heading, blocks, meta)


def test_merged_doc_title_raw_ceiling_above_cap_four_way(monkeypatch) -> None:
    """Even with the env cap set above 200, a long ASCII title is bounded by the
    raw MAX_HEADING_LENGTH ceiling so the H1 truncate_heading stays a no-op and
    the four sites never split (the pre-fix bug)."""
    from lightrag.constants import MAX_HEADING_LENGTH

    doc_heading, blocks, meta = _assemble_bounded_title(
        monkeypatch,
        "300",
        "x" * 130,
        "y" * 130,  # merged 262 raw ASCII
    )
    assert len(doc_heading) == MAX_HEADING_LENGTH
    assert doc_heading.endswith("...")
    _assert_four_way(doc_heading, blocks, meta)


def test_main_only_long_doc_title_truncated_four_way(monkeypatch) -> None:
    """Unified handling (user ruling): a concatenated main_title with no sub is
    bounded the same way, and stays consistent across the four sites."""
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    from lightrag.parser.docx.smart_heading.guardrails import weighted_char_length

    doc_heading, blocks, meta = _assemble_bounded_title(
        monkeypatch,
        str(DEFAULT_DOCX_SMART_HEADING_MAX_CHARS),
        "则" * 70,  # weighted 210 > 180, no sub
        None,
    )
    assert doc_heading.endswith("...")
    assert weighted_char_length(doc_heading) <= DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    _assert_four_way(doc_heading, blocks, meta)


def test_run_smart_heading_rejects_invalid_max_chars_env(monkeypatch) -> None:
    """Parse-time entry check: a structurally invalid DOCX_SMART_HEADING_MAX_CHARS
    hard-fails the parse on EVERY entry point (library / per-file hint), matching
    the API startup check, instead of silently defaulting."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "2")
    with pytest.raises(ValueError, match="DOCX_SMART_HEADING_MAX_CHARS"):
        run_smart_heading([], llm_judge=None, warnings={})

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "abc")
    with pytest.raises(ValueError, match="DOCX_SMART_HEADING_MAX_CHARS"):
        run_smart_heading([], llm_judge=None, warnings={})
