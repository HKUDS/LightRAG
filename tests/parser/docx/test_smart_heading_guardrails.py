"""G9 tests: TOC detection, content preservation (I1), I2/I3 checks,
canonicalization, and the 30% length gate."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading.guardrails import (
    TOC_ELLIPSIS,
    TocOutputPlan,
    canonicalize_paragraph_text,
    detect_toc_records,
    plan_toc_output,
    shadow_baseline_diff,
    smart_output_length_ok,
    toc_audit_entries,
    verify_anchor_semantics,
    verify_baseline_heading_retention,
    verify_content_preservation,
)
from lightrag.parser.docx.smart_heading.heading_flow import HeadingDecision

pytestmark = pytest.mark.offline


def _para(text: str, **kw) -> ParagraphRecord:
    return ParagraphRecord(kind="para", text=text, **kw)


# ---------------------------------------------------------------------------
# canonicalization (G9-3 normalization side)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("## 第一章 绪论", "第一章绪论"),  # markdown prefix + spaces
        ("  中 文  分 词 ", "中文分词"),  # ALL whitespace (CJK zero-width splits)
        ("# Heading Line", "HeadingLine"),
        ("plain\ttext", "plaintext"),
    ],
)
def test_canonicalization_pairs(a: str, b: str) -> None:
    assert canonicalize_paragraph_text(a) == canonicalize_paragraph_text(b)


def test_canonicalization_hash_only_strips_leading_hashes() -> None:
    assert canonicalize_paragraph_text("A # B") == "A#B"  # inner # kept


# ---------------------------------------------------------------------------
# English trailing-period disambiguation (review D1)
# ---------------------------------------------------------------------------


class _FakeSent:
    def __init__(self, start: int, end: int) -> None:
        self.start_char = start
        self.end_char = end


class _FakeDoc:
    def __init__(self, sents) -> None:
        self.sents = sents


def test_ends_with_sentence_period_detects_terminal_dot(monkeypatch) -> None:
    """Review D1: a single English sentence ending in '.' must be recognized
    as a sentence terminator. The phantom 'Next' becomes its OWN sentence, so
    the function returns True. (The old ``>=`` matched the original sentence
    first and always returned False.)"""
    from lightrag.parser.docx.smart_heading import nlp

    text = "This is a sentence."  # len 19; phantom yields two sentences
    monkeypatch.setattr(
        nlp, "analyze", lambda t: _FakeDoc([_FakeSent(0, 19), _FakeSent(20, 24)])
    )
    assert nlp.ends_with_sentence_period(text) is True


def test_ends_with_sentence_period_spares_abbreviation(monkeypatch) -> None:
    """Review D1: an abbreviation dot keeps 'Next' inside the same sentence,
    so it is NOT a terminator."""
    from lightrag.parser.docx.smart_heading import nlp

    text = "See Fig."  # len 8; phantom absorbed into one sentence (0..13)
    monkeypatch.setattr(nlp, "analyze", lambda t: _FakeDoc([_FakeSent(0, 13)]))
    assert nlp.ends_with_sentence_period(text) is False


# ---------------------------------------------------------------------------
# strong-body: a numbering label must not read as a sentence break (I2 bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3.1.1 桌面及办公设备运维服务", "桌面及办公设备运维服务"),
        ("3.1.1桌面及办公设备运维服务", "桌面及办公设备运维服务"),  # no space
        ("2.4.5 系统安全与容灾", "系统安全与容灾"),
        ("1. 概述", "概述"),
        ("第3章 项目背景", "项目背景"),
        ("（一）建设目标", "建设目标"),
        ("没有编号的标题", "没有编号的标题"),  # no numbering → no-op
        ("第二章 \xa0列入条件和管理措施", "列入条件和管理措施"),  # NBSP separator
        ("第3章　项目背景", "项目背景"),  # full-width space separator
    ],
)
def test_strip_leading_numbering(text: str, expected: str) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import _strip_leading_numbering

    assert _strip_leading_numbering(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2.3   投标单位廉洁自律承诺书", False),
        ("第一章\xa0总 则", False),
        ("3.2.2.2.2.4   禁限用工艺", False),
        ("3.2.2.1.1.1.3   4路AD_1～AD_4信号高速采集电路设计", False),
        ("第一阶段已经完成。下一阶段继续推进", True),
        ("第一阶段已经完成。” 下一阶段继续推进", True),
        ("Version 2.0 design overview", False),
        ("项目范围：", False),
    ],
)
def test_explicit_internal_sentence_boundary(text: str, expected: bool) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        has_explicit_internal_sentence_boundary,
    )

    assert has_explicit_internal_sentence_boundary(text) is expected


def test_strong_body_spares_multi_level_numbered_heading() -> None:
    """Regression: zh spaCy splits a dotted multi-level number ("3.1.1 X" →
    "3.1." | "1 X") into two pseudo-sentences, which falsely demoted numbered
    headings as multi-sentence body. strong_body_reason must judge the title
    prose (label stripped), so a numbered heading is spared while genuine body
    still trips."""
    from lightrag.parser.docx.smart_heading import nlp
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    if nlp.missing_spacy_models():
        pytest.skip("spaCy models not installed")

    assert strong_body_reason("3.1.1 桌面及办公设备运维服务") is None
    assert strong_body_reason("3.1.2 政务信息化基础设施运维服务") is None
    assert strong_body_reason("2.4.5 系统安全与容灾") is None
    # Real body (ends with a CJK terminator) is still demoted.
    assert strong_body_reason("本节介绍运维范围。") == "strong_body_sentence_end"


def test_strong_body_spares_heading_with_nbsp_separator() -> None:
    """Regression (test8 应急管理部令第11号): an NBSP after the numbering label
    survived the separator strip, and zh spaCy counted the leading \\xa0 as its
    own "sentence" — falsely demoting the chapter heading as multi-sentence
    body. Both the strip (all Unicode whitespace) and sentence_count (ignore
    whitespace-only sentences) must spare it."""
    from lightrag.parser.docx.smart_heading import nlp
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    if nlp.missing_spacy_models():
        pytest.skip("spaCy models not installed")

    assert strong_body_reason("第二章 \xa0列入条件和管理措施") is None
    assert strong_body_reason("第三章 \xa0列入和移出程序") is None
    # A whitespace-only pseudo-sentence never inflates the count on its own.
    assert nlp.sentence_count("\xa0列入条件和管理措施") == 1
    # Real body (ends with a CJK terminator) is still demoted.
    assert strong_body_reason("本节介绍列入条件。") == "strong_body_sentence_end"


def test_strong_body_multi_sentence_needs_explicit_terminator() -> None:
    """Regression (test13 事故调查报告): the pinned zh model's parser
    hallucinates a mid-word sentence break on short title fragments — a cover
    lead-in (广州市增城区"7.19"索菲亚定制家居项目, split at 索菲|亚) and a bare
    title phrase (投标单位廉洁自律承诺书, split at 承诺|书) both score ≥2 spaCy
    sentences without any visible terminator. The multi-sentence verdict is now
    gated on a deterministic internal terminator, so these are NOT strong body
    (the cover lead-in then joins its title block instead of being demoted),
    while genuine prose carrying an internal "。" still trips."""
    from lightrag.parser.docx.smart_heading import nlp
    from lightrag.parser.docx.smart_heading.guardrails import (
        has_explicit_internal_sentence_boundary,
        strong_body_reason,
    )

    if nlp.missing_spacy_models():
        pytest.skip("spaCy models not installed")

    lead_in = "广州市增城区“7.19”索菲亚定制家居项目"
    phrase = "投标单位廉洁自律承诺书"
    # The pseudo-split is real (≥2 spaCy sentences) but has no visible terminator.
    assert nlp.sentence_count(lead_in) >= 2
    assert not has_explicit_internal_sentence_boundary(lead_in)
    assert strong_body_reason(lead_in) is None
    assert strong_body_reason(phrase) is None
    # Genuine multi-sentence prose (internal "。") is still demoted.
    assert (
        strong_body_reason("项目背景已经说明。后续要求继续执行")
        == "strong_body_multi_sentence"
    )


# ---------------------------------------------------------------------------
# TOC two-channel detection (G9-1 / G9-2 / G9-5)
# ---------------------------------------------------------------------------


def test_structural_toc_evidence() -> None:
    records = [
        _para("目录", is_toc_field=True),
        _para("第一章 绪论……3", is_toc_link=True),
        _para("普通正文段落。"),
    ]
    toc, _events = detect_toc_records(records)
    assert toc == {0, 1}
    entries = toc_audit_entries(records, toc)
    assert len(entries) == 2 and all("hash" in e for e in entries)


def test_heuristic_toc_needs_three_consecutive_leader_lines() -> None:
    """G9-2: a 4-line dot-leader run is TOC; an isolated line is not."""
    run = [
        _para("第一章 绪论............3"),
        _para("第二章 方法............12"),
        _para("第三章 实验……25"),
        _para("第四章 结论·····31"),
    ]
    records = run + [_para("正文开始。"), _para("附录 A............99")]
    toc, _events = detect_toc_records(records)
    assert toc == {0, 1, 2, 3}  # the isolated trailing line survives


def test_heuristic_toc_single_paragraph_soft_breaks() -> None:
    """Review M2 (§2.2.2): a TOC typed as ONE paragraph with ≥3 soft-break
    dot-leader lines is detected — lines are counted, not paragraphs."""
    records = [
        _para(
            "第一章 绪论............3\n第二章 方法............12\n第三章 实验............25"
        ),
        _para("正文开始。"),
    ]
    toc, _events = detect_toc_records(records)
    assert toc == {0}


def test_heuristic_toc_single_paragraph_two_lines_not_enough() -> None:
    """A 2-line soft-break paragraph is below the 3-line threshold."""
    records = [
        _para("第一章 绪论............3\n第二章 方法............12"),
        _para("正文开始。"),
    ]
    assert detect_toc_records(records)[0] == set()


def test_toc_similar_body_not_whitelisted() -> None:
    """G9-5: body text adjacent to a TOC but without leader shape stays."""
    records = [
        _para("第一章 绪论............3"),
        _para("第二章 方法............12"),
        _para("第三章 实验............25"),
        _para("以下正文引用了第一章 绪论的内容。"),
    ]
    toc, _events = detect_toc_records(records)
    assert 3 not in toc


# ---------------------------------------------------------------------------
# TOC third channel: plain (leaderless) numbered run under a 目次 title
# ---------------------------------------------------------------------------

_LONG_BODY = "这是用于分隔目录与正文的较长正文段落其加权字符数明显超过标题长度上限以确保它不会被误判为目录成员并作为一个天然的分隔项目描述若干"


def _p(text: str, size: float = 10.5) -> ParagraphRecord:
    return ParagraphRecord(kind="para", text=text, font_size_pt=size)


def test_toc_third_channel_detects_plain_numbered_run() -> None:
    """Fix-proof (GB/T shape): a leaderless numbered run under 目次, every entry
    duplicated later in the body, is removed; the 目次 title itself stays."""
    records = [
        _p("目  次", 16.0),
        _p("1 范围"),
        _p("2 规范性引用文件"),
        _p("3 术语和定义"),
        _p(_LONG_BODY),  # separator: long → ends the run
        _p("1 范围", 14.0),
        _p("本章规定了适用范围。"),
        _p("2 规范性引用文件", 14.0),
        _p("下列引用文件适用。"),
        _p("3 术语和定义", 14.0),
        _p("下列术语和定义适用。"),
    ]
    toc, events = detect_toc_records(records)
    assert toc == {1, 2, 3}
    assert 0 not in toc  # the 目次 title is a real heading
    assert events == [
        {"rule": "toc_plain_numbered_run", "anchor": 0, "start": 1, "end": 3}
    ]


def test_toc_third_channel_truncates_before_body_first_heading() -> None:
    """Review regression: the TOC is immediately followed by the body's first
    heading with NO long separator, at the SAME size as the TOC entries. The
    body heading's only copy is the TOC entry ABOVE it, so it has no forward
    duplicate and must END the run rather than be swallowed."""
    records = [
        _p("目  次", 16.0),
        _p("1 范围"),
        _p("2 规范性引用文件"),
        _p("3 术语和定义"),
        # body starts immediately, same 10.5pt size (size termination is inert)
        _p("1 范围"),
        _p("本章规定了适用范围。"),
        _p("2 规范性引用文件"),
        _p("下列引用文件适用。"),
        _p("3 术语和定义"),
        _p("下列术语和定义适用。"),
    ]
    toc, events = detect_toc_records(records)
    assert toc == {1, 2, 3}
    assert 4 not in toc and 6 not in toc and 8 not in toc  # body headings stay
    assert events[0]["end"] == 3


def test_toc_third_channel_font_size_terminates_run() -> None:
    """A member strictly larger than the TOC's established max size (the body's
    first heading, e.g. a 16pt 前言) ends the run."""
    records = [
        _p("目  次", 16.0),
        _p("1 范围"),
        _p("2 规范性引用文件"),
        _p("3 术语和定义"),
        _p("前  言", 16.0),  # bigger → run ends here
        _p("前  言", 16.0),
        _p("1 范围"),
        _p("2 规范性引用文件"),
        _p("3 术语和定义"),
    ]
    toc, events = detect_toc_records(records)
    assert toc == {1, 2, 3}
    assert 4 not in toc


def test_toc_third_channel_all_or_nothing() -> None:
    """If any run member has no forward duplicate, the WHOLE block is rejected
    (a partial TOC is not confidently a TOC)."""
    records = [
        _p("目  次", 16.0),
        _p("1 范围"),
        _p("2 规范性引用文件"),
        _p("3 术语和定义"),  # this one has NO body copy
        _p(_LONG_BODY),
        _p("1 范围", 14.0),
        _p("正文。"),
        _p("2 规范性引用文件", 14.0),
        _p("正文。"),
    ]
    toc, events = detect_toc_records(records)
    assert toc == set()
    assert events == []


def test_toc_third_channel_anchor_front_part_limit() -> None:
    """A 目录/Contents heading deep in the body (after several long body
    paragraphs) does not anchor the channel."""
    records = [_p(_LONG_BODY) for _ in range(3)]  # 3 long body paras up front
    records += [
        _p("Contents", 16.0),  # a mid-document "Contents" — too late to anchor
        _p("1 范围"),
        _p("2 规范性引用文件"),
        _p("3 术语和定义"),
        _p(_LONG_BODY),
        _p("1 范围", 14.0),
        _p("2 规范性引用文件", 14.0),
        _p("3 术语和定义", 14.0),
    ]
    toc, events = detect_toc_records(records)
    assert toc == set()
    assert events == []


def test_toc_third_channel_soft_break_member_counts_lines() -> None:
    """A run member typed as one paragraph with soft-break lines is matched and
    counted line-by-line (line-level inverted index)."""
    records = [
        _p("目  次", 16.0),
        _p("1 范围\n2 规范性引用文件\n3 术语和定义"),  # one paragraph, 3 lines
        _p(_LONG_BODY),
        _p("1 范围", 14.0),
        _p("2 规范性引用文件", 14.0),
        _p("3 术语和定义", 14.0),
    ]
    toc, events = detect_toc_records(records)
    assert toc == {1}  # the single multi-line paragraph
    assert events[0]["start"] == 1 and events[0]["end"] == 1


def test_toc_third_channel_below_min_lines_not_detected() -> None:
    """Fewer than DOCX_SMART_TOC_MIN_LINES members → not a TOC (false negatives
    are acceptable here — CB1 handles a small stray run)."""
    records = [
        _p("目  次", 16.0),
        _p("1 范围"),
        _p("2 规范性引用文件"),
        _p(_LONG_BODY),
        _p("1 范围", 14.0),
        _p("2 规范性引用文件", 14.0),
    ]
    toc, events = detect_toc_records(records)
    assert toc == set()


def test_toc_third_channel_needs_title_anchor() -> None:
    """A numbered run with body duplicates but NO 目次/Contents title is not
    removed — the anchor is mandatory."""
    records = [
        _p("1 范围"),
        _p("2 规范性引用文件"),
        _p("3 术语和定义"),
        _p(_LONG_BODY),
        _p("1 范围", 14.0),
        _p("2 规范性引用文件", 14.0),
        _p("3 术语和定义", 14.0),
    ]
    toc, events = detect_toc_records(records)
    assert toc == set()


def test_toc_third_channel_strips_trailing_pageno() -> None:
    """A TOC entry with a trailing page number matches the body heading without
    it (the page number is stripped before the duplicate comparison)."""
    records = [
        _p("目  次", 16.0),
        _p("1 范围　3"),
        _p("2 规范性引用文件　5"),
        _p("3 术语和定义　8"),
        _p(_LONG_BODY),
        _p("1 范围", 14.0),
        _p("2 规范性引用文件", 14.0),
        _p("3 术语和定义", 14.0),
    ]
    toc, events = detect_toc_records(records)
    assert toc == {1, 2, 3}


def test_toc_third_channel_english_contents_casefold() -> None:
    """English TOCs anchor on Table of Contents / CONTENTS and match body
    headings case-insensitively."""
    records = [
        _p("Table of Contents", 16.0),
        _p("1 Scope"),
        _p("2 Normative References"),
        _p("3 Terms and Definitions"),
        _p(_LONG_BODY),
        _p("1 SCOPE", 14.0),  # different case in the body
        _p("2 normative references", 14.0),
        _p("3 Terms and Definitions", 14.0),
    ]
    toc, events = detect_toc_records(records)
    assert toc == {1, 2, 3}


# ---------------------------------------------------------------------------
# TOC retention: plan_toc_output (§2.3)
# ---------------------------------------------------------------------------


def _toc(*texts: str) -> tuple[list[ParagraphRecord], set[int]]:
    """Para records that are ALL in the TOC set (document index order)."""
    recs = [_p(t) for t in texts]
    return recs, set(range(len(recs)))


def test_plan_toc_output_under_budget_keeps_all_no_ellipsis() -> None:
    recs, toc = _toc("第一章 绪论……3", "第二章 方法……12")
    plan = plan_toc_output(recs, toc, keep_lines=5)
    assert isinstance(plan, TocOutputPlan)
    assert plan.kept_text == {0: "第一章 绪论……3", 1: "第二章 方法……12"}
    assert plan.ellipsis_anchor is None
    assert (plan.kept_lines, plan.removed_lines) == (2, 0)


def test_plan_toc_output_exactly_at_budget_no_ellipsis() -> None:
    recs, toc = _toc("1", "2", "3", "4", "5")
    plan = plan_toc_output(recs, toc, keep_lines=5)
    assert set(plan.kept_text) == {0, 1, 2, 3, 4}
    assert plan.ellipsis_anchor is None
    assert (plan.kept_lines, plan.removed_lines) == (5, 0)


def test_plan_toc_output_over_budget_truncates_with_ellipsis() -> None:
    recs, toc = _toc(*[f"第{i} 条……{i}" for i in range(8)])
    plan = plan_toc_output(recs, toc, keep_lines=5)
    assert set(plan.kept_text) == {0, 1, 2, 3, 4}
    assert plan.ellipsis_anchor == 4  # one "……" after the last kept record
    assert (plan.kept_lines, plan.removed_lines) == (5, 3)
    assert all(i not in plan.kept_text for i in (5, 6, 7))
    assert plan.fully_removed_records == 3  # records 5-7 kept nothing


def test_plan_toc_output_straddling_soft_break_paragraph() -> None:
    """A single paragraph with 8 soft-break lines, budget 5: keep the first 5
    lines, count the 3-line tail as removed, anchor is that same record. This
    is why the budget must be per visible line, not per record."""
    recs, toc = _toc("\n".join(f"第{i} 条……{i}" for i in range(8)))
    plan = plan_toc_output(recs, toc, keep_lines=5)
    assert plan.kept_text == {0: "\n".join(f"第{i} 条……{i}" for i in range(5))}
    assert plan.ellipsis_anchor == 0
    assert (plan.kept_lines, plan.removed_lines) == (5, 3)
    assert plan.fully_removed_records == 0  # partly kept ≠ fully removed


def test_plan_toc_output_keep_zero_collapses_to_ellipsis() -> None:
    recs, toc = _toc("第一章……3", "第二章……12", "第三章……25")
    plan = plan_toc_output(recs, toc, keep_lines=0)
    assert plan.kept_text == {}
    assert plan.ellipsis_anchor == 0  # the first visible record
    assert (plan.kept_lines, plan.removed_lines) == (0, 3)
    assert plan.fully_removed_records == 3


def test_plan_toc_output_negative_budget_clamps_to_zero() -> None:
    recs, toc = _toc("第一章……3", "第二章……12")
    plan = plan_toc_output(recs, toc, keep_lines=-3)
    assert plan.kept_text == {}
    assert plan.ellipsis_anchor == 0
    assert plan.removed_lines == 2


def test_plan_toc_output_non_contiguous_segments_single_anchor() -> None:
    """Two TOC runs (目录 + 图目录) share ONE global budget and ONE ellipsis;
    a body record between them is excluded from toc_indices."""
    recs = [_p(t) for t in ("A", "B", "sep-body", "C", "D", "E")]
    toc = {0, 1, 3, 4, 5}  # index 2 is body
    plan = plan_toc_output(recs, toc, keep_lines=3)
    assert set(plan.kept_text) == {0, 1, 3}  # first 3 lines across BOTH runs
    assert plan.ellipsis_anchor == 3
    assert (plan.kept_lines, plan.removed_lines) == (3, 2)


def test_plan_toc_output_skips_whitespace_and_strips() -> None:
    recs = [_p("   "), _p("  第一章 绪论……3  "), _p("第二章……12")]
    plan = plan_toc_output(recs, {0, 1, 2}, keep_lines=5)
    assert 0 not in plan.kept_text  # whitespace-only record skipped entirely
    assert plan.kept_text[1] == "第一章 绪论……3"  # visible text, surrounding space gone
    assert (plan.kept_lines, plan.removed_lines) == (2, 0)
    assert plan.ellipsis_anchor is None
    assert plan.fully_removed_records == 0  # a whitespace record is not "removed"


def test_plan_toc_output_counts_demoted_tail_as_removed() -> None:
    """A TOC record carrying a ``demoted_body_text`` tail (oversize soft-break
    remainder from the read pass): the tail is never re-emitted by the TOC
    branch, so its visible lines count as removed and raise the ellipsis even
    when every ``text`` line fits the budget."""
    rec = ParagraphRecord(
        kind="para",
        text="第一章 绪论……3",
        demoted_body_text="尾部第一行\n尾部第二行",
    )
    plan = plan_toc_output([rec], {0}, keep_lines=5)
    assert plan.kept_text == {0: "第一章 绪论……3"}
    assert (plan.kept_lines, plan.removed_lines) == (1, 2)  # tail lines removed
    assert plan.ellipsis_anchor == 0  # elided tail alone raises the marker
    assert plan.fully_removed_records == 0


# ---------------------------------------------------------------------------
# I1 content preservation (G9-3)
# ---------------------------------------------------------------------------


def _blocks(*contents: str) -> list[dict]:
    return [{"content": c, "heading": "", "level": 1} for c in contents]


def test_i1_passes_on_faithful_output() -> None:
    records = [
        _para("第一章 绪论"),
        _para("正文第一段。"),
        _para("正文第二段。"),
    ]
    blocks = _blocks("# 第一章 绪论\n正文第一段。", "正文第二段。")
    assert verify_content_preservation(records, blocks) == []


def test_i1_detects_injected_loss() -> None:
    """G9-3: dropping a paragraph is caught."""
    records = [
        _para("第一章 绪论"),
        _para("正文第一段。"),
        _para("被丢失的段落。"),
    ]
    blocks = _blocks("# 第一章 绪论\n正文第一段。")
    missing = verify_content_preservation(records, blocks)
    assert missing == [canonicalize_paragraph_text("被丢失的段落。")]


def test_i1_tolerates_heading_merge_and_oversize_rebuild() -> None:
    """Merged headings (two records → one line) and use_raw_text rebuilds
    (one line = first-line + demoted tail) both pass via the ordered
    concatenation tolerance pass."""
    records = [
        _para("中华人民共和国"),
        _para("某某管理办法"),
        ParagraphRecord(
            kind="para",
            text="超长标题首行",
            demoted_body_text="被拆出的余部内容",
        ),
    ]
    blocks = _blocks(
        "# 中华人民共和国某某管理办法",
        "超长标题首行被拆出的余部内容",
    )
    assert verify_content_preservation(records, blocks) == []


def test_i1_whitelists_toc_indices_only() -> None:
    records = [
        _para("第一章……3", is_toc_link=True),
        _para("正文段落。"),
    ]
    blocks = _blocks("正文段落。")
    assert verify_content_preservation(records, blocks, toc_indices={0}) == []
    # Without the whitelist the same absence is a violation.
    assert verify_content_preservation(records, blocks) != []


def test_i1_subtracts_injected_toc_copy_reveals_body_loss() -> None:
    """§2.3 TOC retention: a retained TOC copy of a body heading must NOT mask
    that heading actually going missing from the output. The injected copy is
    subtracted from the output multiset first, so the lost body line surfaces.
    (Multiset SUBTRACTION, not mere canonicalization: proven by the paired
    'body present' case below returning [].)"""
    records = [
        _para("第一章 绪论", is_toc_link=True),  # leaderless TOC entry (same canon)
        _para("第一章 绪论"),  # the real body heading
        _para("正文段落。"),
    ]
    # Output has ONLY ONE "第一章 绪论" line (the retained TOC copy); the body
    # heading was lost by an assembler bug.
    blocks = _blocks("第一章 绪论\n正文段落。")
    missing = verify_content_preservation(
        records, blocks, toc_indices={0}, ignored_output_texts=["第一章 绪论"]
    )
    assert missing == [canonicalize_paragraph_text("第一章 绪论")]


def test_i1_injected_toc_copy_ok_when_body_copy_present() -> None:
    """Paired control: with the body heading present the output has two copies,
    the injected one is subtracted, and the body one still satisfies I1."""
    records = [
        _para("第一章 绪论", is_toc_link=True),
        _para("第一章 绪论"),
        _para("正文段落。"),
    ]
    blocks = _blocks("第一章 绪论\n# 第一章 绪论\n正文段落。")
    assert (
        verify_content_preservation(
            records, blocks, toc_indices={0}, ignored_output_texts=["第一章 绪论"]
        )
        == []
    )


def test_i1_subtracts_ellipsis_without_masking_a_source_ellipsis() -> None:
    """The injected ellipsis is subtracted; a genuine body '……' line still
    needs its own output copy."""
    records = [
        _para("目录条目……3", is_toc_link=True),
        _para(TOC_ELLIPSIS),  # a REAL body paragraph that happens to be "……"
    ]
    # One ellipsis in output = the injected one; the source ellipsis is lost.
    assert verify_content_preservation(
        records,
        _blocks(TOC_ELLIPSIS),
        toc_indices={0},
        ignored_output_texts=[TOC_ELLIPSIS],
    ) == [canonicalize_paragraph_text(TOC_ELLIPSIS)]
    # Two ellipses (injected + real source) → subtract one, pass.
    assert (
        verify_content_preservation(
            records,
            _blocks(f"{TOC_ELLIPSIS}\n{TOC_ELLIPSIS}"),
            toc_indices={0},
            ignored_output_texts=[TOC_ELLIPSIS],
        )
        == []
    )


# ---------------------------------------------------------------------------
# I2 / I3
# ---------------------------------------------------------------------------


def test_i2_outline_paragraph_must_stay_or_be_rule_tagged() -> None:
    records = [
        _para("样式标题甲", outline_level=0),
        _para("样式标题乙", outline_level=0),
        _para("样式标题丙", outline_level=0),
    ]
    kept = HeadingDecision(record_index=0, text="样式标题甲", is_heading=True, level=1)
    demoted = HeadingDecision(record_index=1, text="样式标题乙", is_heading=False)
    demoted.note("strong_body_demoted")
    silently_dropped = HeadingDecision(
        record_index=2, text="样式标题丙", is_heading=False
    )
    assert verify_baseline_heading_retention(
        records, [kept, demoted, silently_dropped]
    ) == [2]


def test_i3_non_title_block_level_zero_is_violation() -> None:
    """Review I3: level 0 is reserved for title-block roots; a plain heading
    that somehow landed at level 0 is a construction bug (single-root
    assertion)."""
    title = HeadingDecision(
        record_index=0, text="主标题", is_heading=True, is_title_block=True, level=0
    )
    stray = HeadingDecision(record_index=1, text="普通标题", is_heading=True, level=0)
    assert verify_anchor_semantics([title, stray]) == [1]


def test_i3_non_numbered_anchor_must_match_outline() -> None:
    good = HeadingDecision(
        record_index=0,
        text="锚定标题",
        is_heading=True,
        level=3,
        outline_level=2,
        anchored=True,
    )
    bad = HeadingDecision(
        record_index=1,
        text="漂移标题",
        is_heading=True,
        level=5,
        outline_level=2,
        anchored=True,
    )
    assert verify_anchor_semantics([good, bad]) == [1]


# ---------------------------------------------------------------------------
# 30% gate + shadow diff (G9-4)
# ---------------------------------------------------------------------------


def test_length_gate_triggers_below_30_percent() -> None:
    baseline = _blocks("x" * 1000)
    assert smart_output_length_ok(_blocks("x" * 300), baseline)
    assert not smart_output_length_ok(_blocks("x" * 299), baseline)


def test_shadow_diff_summary() -> None:
    smart = [
        {"content": "abc", "level": 0},
        {"content": "def", "level": 2},
    ]
    base = [{"content": "abcdef", "level": 1}]
    diff = shadow_baseline_diff(smart, base)
    assert diff["heading_count_smart"] == 2
    assert diff["heading_count_baseline"] == 1
    assert diff["level_distribution_smart"] == {0: 1, 2: 1}
    assert diff["content_char_delta"] == 0


def test_i1_concatenation_tolerance_rejects_substring_hiding() -> None:
    """A13: a genuinely lost short paragraph must not pass just because its
    text is a substring of an unmatched merged line; ordered concatenation
    consumes each source piece at most once."""
    records = [
        _para("第一部分"),
        _para("总体要求"),
        _para("要求"),  # lost: substring of the merged line below
    ]
    blocks = _blocks("# 第一部分总体要求")
    missing = verify_content_preservation(records, blocks)
    assert missing == [canonicalize_paragraph_text("要求")]


def test_i1_concatenation_tolerance_accepts_ordered_merge() -> None:
    """A13 positive path: consecutive source pieces composing the merged
    output line exactly (in document order) all pass."""
    records = [
        _para("第一部分"),
        _para("总体要求"),
        _para("独立正文段。"),
    ]
    blocks = _blocks("# 第一部分总体要求", "独立正文段。")
    assert verify_content_preservation(records, blocks) == []


def test_i1_tolerates_body_paragraph_soft_break() -> None:
    """Review C1 (G9-3 control group): a body paragraph carrying a soft line
    break (``w:br`` → ``\\n``) is emitted verbatim as multiple output content
    lines. The source side is split on ``\\n`` too, so the two sides compare
    symmetrically instead of the whole paragraph canonicalizing to one piece
    that no single output line equals (which forced a spurious fallback)."""
    records = [
        _para("标题甲"),
        _para("正文第一行\n正文第二行"),  # soft break inside body
    ]
    blocks = _blocks("# 标题甲\n正文第一行\n正文第二行")
    assert verify_content_preservation(records, blocks) == []


def test_i1_tolerates_demoted_body_text_soft_break() -> None:
    """Review C1: the oversize soft-break remainder (``demoted_body_text``)
    may itself carry further soft breaks and still pass I1."""
    records = [
        ParagraphRecord(
            kind="para",
            text="超长标题首行",
            demoted_body_text="余部第一行\n余部第二行",
        ),
    ]
    blocks = _blocks("超长标题首行\n余部第一行\n余部第二行")
    assert verify_content_preservation(records, blocks) == []


# ---------------------------------------------------------------------------
# heading length cap reader + weighted truncation (merged doc-title support)
# ---------------------------------------------------------------------------


def test_heading_max_chars_reads_env_and_floors_below_three(monkeypatch) -> None:
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    from lightrag.parser.docx.smart_heading.guardrails import heading_max_chars

    monkeypatch.delenv("DOCX_SMART_HEADING_MAX_CHARS", raising=False)
    assert heading_max_chars() == DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "250")
    assert heading_max_chars() == 250
    # A cap below 3 cannot hold a "..." and is treated as invalid -> default.
    for bad in ("2", "0", "-5"):
        monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", bad)
        assert heading_max_chars() == DEFAULT_DOCX_SMART_HEADING_MAX_CHARS


def test_truncate_to_heading_length_short_string_unchanged(monkeypatch) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        truncate_to_heading_length,
    )

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "180")
    assert truncate_to_heading_length("广州市平台  建设方案") == "广州市平台  建设方案"


def test_truncate_to_heading_length_weighted_cap(monkeypatch) -> None:
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    from lightrag.parser.docx.smart_heading.guardrails import (
        truncate_to_heading_length,
        weighted_char_length,
    )

    monkeypatch.setenv(
        "DOCX_SMART_HEADING_MAX_CHARS", str(DEFAULT_DOCX_SMART_HEADING_MAX_CHARS)
    )
    out = truncate_to_heading_length("标" * 80)  # weighted 240 > 180
    assert weighted_char_length(out) <= DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    assert out.endswith("...")


def test_truncate_to_heading_length_floors_tiny_cap(monkeypatch) -> None:
    """A cap below 3 must not make the helper emit a bare '...' whose weighted
    length exceeds the cap; heading_max_chars floors it to the default."""
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    from lightrag.parser.docx.smart_heading.guardrails import (
        truncate_to_heading_length,
        weighted_char_length,
    )

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "2")
    out = truncate_to_heading_length("标" * 100)
    assert out != "..."
    assert weighted_char_length(out) <= DEFAULT_DOCX_SMART_HEADING_MAX_CHARS


def test_truncate_to_heading_length_raw_ceiling_above_cap(monkeypatch) -> None:
    """When the env cap exceeds MAX_HEADING_LENGTH, a long ASCII heading is
    still bounded by the hard raw 200-char ceiling (so the H1 truncate_heading
    stays a no-op and the four title-block landing sites never split)."""
    from lightrag.constants import MAX_HEADING_LENGTH
    from lightrag.parser.docx.smart_heading.guardrails import (
        truncate_to_heading_length,
    )

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "300")
    out = truncate_to_heading_length("x" * 250)  # weighted=raw=250 <= 300, > 200 raw
    assert len(out) == MAX_HEADING_LENGTH
    assert out.endswith("...")


def test_strong_body_length_floors_tiny_cap(monkeypatch) -> None:
    """strong_body_reason shares heading_max_chars: a nonsensical cap<3 must
    not demote an ordinary short heading via the length rule."""
    from lightrag.parser.docx.smart_heading import nlp
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    if nlp.missing_spacy_models():
        pytest.skip("spaCy models not installed")

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "2")
    # Weighted 30 (10 CJK); with the floor (180) the length rule spares it.
    assert strong_body_reason("某某管理办法实施细则条例") is None


def test_validate_heading_max_chars_env_returns_parsed(monkeypatch) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        validate_heading_max_chars_env,
    )

    monkeypatch.delenv("DOCX_SMART_HEADING_MAX_CHARS", raising=False)
    assert validate_heading_max_chars_env() is None  # unset -> default downstream
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "180")
    assert validate_heading_max_chars_env() == 180
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "50")
    assert validate_heading_max_chars_env() == 50  # usable-but-tiny: no raise here


@pytest.mark.parametrize("bad", ["abc", "1.5", "2", "0", "-5"])
def test_validate_heading_max_chars_env_raises_on_invalid(monkeypatch, bad) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        validate_heading_max_chars_env,
    )

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", bad)
    with pytest.raises(ValueError, match="DOCX_SMART_HEADING_MAX_CHARS"):
        validate_heading_max_chars_env()
