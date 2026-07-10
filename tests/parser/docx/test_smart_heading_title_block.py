"""G4 tests (judgment layer): title-block candidates and LLM verdict parsing.

Strong-body / homophone judgments are injected as deterministic stubs so
these tests run without spaCy; the real implementations are covered by
test_smart_heading_guards.py.
"""

from __future__ import annotations

import json

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading import guardrails
from lightrag.parser.docx.smart_heading.title_block import (
    TitleBlockCandidate,
    TitleBlockLLMError,
    compose_title_heading,
    detect_imprint_regions,
    find_title_block_candidates,
    flatten_heading_line,
    judge_title_block,
)

pytestmark = pytest.mark.offline


def _para(text: str, *, size: float = 12.0, **kw) -> ParagraphRecord:
    return ParagraphRecord(kind="para", text=text, font_size_pt=size, **kw)


def _empty() -> ParagraphRecord:
    return ParagraphRecord(kind="empty_para")


def _stub_strong_body(text: str) -> str | None:
    """Deterministic stand-in: sentence enders / long text are body."""
    stripped = text.strip()
    if stripped.endswith(("。", "？", "！")) or len(stripped) > 60:
        return "strong_body_stub"
    return None


def _stub_no_veto(_classification, _text) -> str | None:
    return None


def _stub_always_veto(_classification, _text) -> str | None:
    return "homophone_stub"


def _find(records, **kw):
    kw.setdefault("fs_base_pt", 12.0)
    kw.setdefault("strong_body", _stub_strong_body)
    kw.setdefault("numbering_veto", _stub_no_veto)
    return find_title_block_candidates(records, **kw)


# ---------------------------------------------------------------------------
# candidate discovery
# ---------------------------------------------------------------------------


def test_multi_window_candidate_found() -> None:
    records = [
        _para("公司数字化转型白皮书", size=22.0),  # big main title
        _para("（2026年版）", size=12.0),  # metadata line, body-sized
        _empty(),
        _para("某某咨询公司发布", size=12.0),
        _para("正文从这里开始，介绍研究的背景与目标。", size=12.0),  # strong body
    ]
    cands = _find(records)
    assert len(cands) == 1
    c = cands[0]
    assert not c.single
    assert (c.start, c.end) == (0, 4)  # stops before the strong-body line


def test_multi_window_requires_two_paragraphs_and_big_line() -> None:
    # Only one non-body line → no multi window; also no boundary evidence
    # beyond doc-start for a single candidate when size is not dominant.
    records = [
        _para("孤立小字号行", size=12.0),
        _para("这是很长的正文，它明确地以句号结尾。", size=12.0),
    ]
    assert _find(records) == []


def test_single_candidate_first_paragraph(monkeypatch) -> None:
    """G4-9: a lone big first line is a single-paragraph candidate; a
    digit-led title survives when the homophone veto revokes its numbering."""
    records = [
        _para("2026年度工作报告", size=18.0),
        _para("正文第一段说明了年度工作的总体情况。", size=12.0),
        _para("正文第二段继续说明。", size=12.0),
    ]
    cands = _find(records, numbering_veto=_stub_always_veto)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 0

    # Without the veto the same line counts as genuinely numbered → excluded.
    assert _find(records, numbering_veto=_stub_no_veto) == []


def test_single_candidate_requires_boundary_evidence() -> None:
    """G4-11: mid-document big line without any hard boundary is rejected."""
    body = [_para(f"正文段落{i}，以句号结尾。", size=12.0) for i in range(3)]
    records = (
        body
        + [_para("突然的大字号行", size=18.0)]
        + [_para(f"后续正文{i}，以句号结尾。", size=12.0) for i in range(3)]
    )
    assert _find(records) == []


def test_single_candidate_centered_with_blank_flanks() -> None:
    records = [
        _para("开头正文，以句号结尾。", size=12.0),
        _empty(),
        _para("居中的独立标题", size=18.0, alignment="center"),
        _empty(),
        _para("后续正文，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 2


def test_single_candidate_after_page_break() -> None:
    records = [
        _para("前一篇正文结束。", size=12.0),
        _para("另一篇文章的标题", size=18.0, page_break_before=True),
        _para("这一篇正文开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 1


def test_single_candidate_with_genuine_numbering_excluded() -> None:
    """A line opening with a REAL chapter number (第二篇) is chapter-level
    material, not a per-article title — it must not split sub-documents."""
    records = [
        _para("前文正文结束。", size=12.0),
        _para("第二篇 分论", size=18.0, page_break_before=True),
        _para("正文继续，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []


def test_single_candidate_cap_truncates(monkeypatch) -> None:
    """G4-11: candidates beyond the per-document cap are skipped + warned."""
    monkeypatch.setenv("DOCX_SMART_SINGLE_TITLE_LLM_MAX", "2")
    records = []
    for i in range(4):
        records.append(_para(f"独立文章标题{i}", size=18.0, page_break_before=True))
        records.append(_para(f"文章{i}的正文，以句号结尾。", size=12.0))
    warnings: dict = {}
    cands = _find(records, warnings=warnings)
    assert len(cands) == 2
    assert warnings["title_block_single_candidates_truncated"] == 2


# ---------------------------------------------------------------------------
# LLM verdict parsing + validation
# ---------------------------------------------------------------------------

_TITLE_RECORDS = [
    _para("中华人民共和国某某管理办法", size=22.0),
    _para("（2026年修订）", size=12.0),
    _empty(),
    _para("国某发〔2026〕12号", size=12.0),
    _para("正文第一条从这里开始。", size=12.0),
]
_CANDIDATE = TitleBlockCandidate(start=0, end=4, single=False, trigger="multi_window")


def _judge_with(payload: dict | str, records=None, candidate=None, warnings=None):
    raw = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return raw

    return judge_title_block(
        candidate or _CANDIDATE,
        records or _TITLE_RECORDS,
        _llm,
        warnings=warnings if warnings is not None else {},
    )


def test_title_verdict_parsed_and_composed() -> None:
    decision = _judge_with(
        {
            "is_title_block": True,
            "main_title": "中华人民共和国某某管理办法",
            "sub_title": "（2026年修订）",
            "doc_number": "国某发〔2026〕12号",
            "classification": None,
            "publisher": None,
            "date": "2026",
        }
    )
    assert decision.is_title_block
    assert decision.member_indices == (0, 1, 3)  # blank line carries no index
    assert (
        compose_title_heading(decision)
        == "中华人民共和国某某管理办法 — （2026年修订） — 国某发〔2026〕12号(2026)"
    )


def test_title_verdict_concatenated_main_title_locates() -> None:
    records = [
        _para("中华人民共和国", size=22.0),
        _para("某某管理办法", size=22.0),
        _para("正文开始。", size=12.0),
    ]
    decision = _judge_with(
        {"is_title_block": True, "main_title": "中华人民共和国某某管理办法"},
        records=records,
        candidate=TitleBlockCandidate(start=0, end=2, single=False, trigger="t"),
    )
    assert decision.main_title == "中华人民共和国某某管理办法"


def test_title_verdict_multiline_fields_flattened() -> None:
    """Title material renders single-line at verdict construction: an LLM
    echoing a soft-break/multi-paragraph title must not ride ``\\n`` into the
    heading stack (and thence every descendant's parent_headings). CJK
    boundaries join without a space, others with one; locate-back still
    passes (_canon strips ALL whitespace)."""
    records = [
        _para("中华人民共和国\n某某管理办法", size=22.0),
        _para("Implementation\nGuide", size=16.0),
        _para("正文开始。", size=12.0),
    ]
    decision = _judge_with(
        {
            "is_title_block": True,
            "main_title": "中华人民共和国\n某某管理办法",
            "sub_title": "Implementation\r\nGuide",
        },
        records=records,
        candidate=TitleBlockCandidate(start=0, end=2, single=False, trigger="t"),
    )
    assert decision.main_title == "中华人民共和国某某管理办法"
    assert decision.sub_title == "Implementation Guide"
    assert "\n" not in compose_title_heading(decision)


def test_flatten_heading_line_unicode_breaks() -> None:
    assert flatten_heading_line("总则\u2028细则") == "总则细则"
    assert flatten_heading_line("  Part A \n\n Part B ") == "Part A Part B"
    assert flatten_heading_line("单行标题") == "单行标题"
    assert flatten_heading_line("\n \n") == ""


def test_title_verdict_hallucinated_title_hard_fails() -> None:
    with pytest.raises(TitleBlockLLMError, match="cannot be located"):
        _judge_with({"is_title_block": True, "main_title": "完全虚构的标题"})


def test_invalid_json_hard_fails() -> None:
    with pytest.raises(TitleBlockLLMError, match="no JSON object"):
        _judge_with("I think this is a title block.")


def test_non_title_verdict_classifies_every_paragraph() -> None:
    decision = _judge_with({"is_title_block": False, "headings": [0], "body": [1, 2]})
    assert not decision.is_title_block
    assert decision.heading_indices == (0,)
    assert decision.body_indices == (1, 3)


def test_non_title_verdict_incomplete_partition_recovers() -> None:
    """A well-formed but UNDER-specified partition (window index 2 left in
    neither list) no longer hard-fails: the unmentioned paragraph abstains
    (neither granted nor vetoed) and re-enters the normal flow; the omission
    is counted via ``title_block_partition_incomplete``, not fatal."""
    warnings: dict = {}
    decision = _judge_with(
        {"is_title_block": False, "headings": [0], "body": [1]},
        warnings=warnings,
    )
    assert not decision.is_title_block
    assert decision.heading_indices == (0,)  # granted
    assert decision.body_indices == (1,)  # vetoed; window idx 2 (rec 3) abstains
    assert warnings["title_block_partition_incomplete"] == 1


def test_non_title_verdict_metadata_dropped_index_recovers() -> None:
    """Regression (test9 国标范例): the LLM correctly rejects the title block
    but routes the standard number into ``doc_number`` and forgets to place
    its index in headings/body (headings=[1], body=[]). The dropped index now
    abstains instead of failing the whole document."""
    records = [
        _para("GB/T 9704—2012", size=12.0),
        _para("前　言", size=14.0),
    ]
    warnings: dict = {}
    decision = _judge_with(
        {
            "is_title_block": False,
            "headings": [1],
            "body": [],
            "doc_number": "GB/T 9704—2012",
        },
        records=records,
        candidate=TitleBlockCandidate(
            start=0, end=2, single=False, trigger="multi_window"
        ),
        warnings=warnings,
    )
    assert not decision.is_title_block
    assert decision.heading_indices == (1,)  # 前言 granted
    assert decision.body_indices == ()  # GB/T line (index 0) abstains, not vetoed
    assert warnings["title_block_partition_incomplete"] == 1


def test_non_title_verdict_malformed_partition_hard_fails() -> None:
    """Only OMISSION is forgiven; malformed output is still loud. A missing or
    null field, a non-int (incl. bool) index, an out-of-range index, or the
    same index in both lists cannot be reconciled and hard-fails."""
    with pytest.raises(TitleBlockLLMError, match="malformed"):  # heading∩body
        _judge_with({"is_title_block": False, "headings": [0, 1], "body": [1, 2]})
    with pytest.raises(TitleBlockLLMError, match="malformed"):  # out of range
        _judge_with({"is_title_block": False, "headings": [99], "body": []})
    with pytest.raises(TitleBlockLLMError, match="must be"):  # missing fields
        _judge_with({"is_title_block": False})
    with pytest.raises(TitleBlockLLMError, match="must be"):  # explicit null
        _judge_with({"is_title_block": False, "headings": None, "body": None})
    with pytest.raises(TitleBlockLLMError, match="must be"):  # bool is not an index
        _judge_with({"is_title_block": False, "headings": [True], "body": [1]})


def test_outline_paragraph_never_demoted_by_llm(monkeypatch) -> None:
    """G4-5 / I2: an outlineLvl paragraph voted 'body' keeps heading standing."""
    records = [
        _para("大标题", size=22.0),
        _para("带物理大纲的段落", size=12.0, outline_level_raw=1),
        _para("正文开始。", size=12.0),
    ]
    warnings: dict = {}
    decision = _judge_with(
        {"is_title_block": False, "headings": [0], "body": [1]},
        records=records,
        candidate=TitleBlockCandidate(start=0, end=2, single=False, trigger="t"),
        warnings=warnings,
    )
    assert decision.heading_indices == (0, 1)
    assert decision.body_indices == ()
    assert warnings["title_block_llm_outline_demotion_blocked"] == 1


def test_window_token_cap_truncates(monkeypatch) -> None:
    """G4-4: an over-long window is tail-truncated with a warning."""
    monkeypatch.setenv("DOCX_SMART_LLM_WINDOW_TOKENS", "30")
    records = [
        _para("标题块第一行内容比较长" * 3, size=22.0),
        _para("标题块第二行内容也比较长" * 3, size=12.0),
        _para("标题块第三行内容同样很长" * 3, size=12.0),
        _para("正文开始。", size=12.0),
    ]
    seen_prompt: dict = {}

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        seen_prompt["prompt"] = prompt
        return json.dumps(
            {"is_title_block": True, "main_title": records[0].text},
            ensure_ascii=False,
        )

    warnings: dict = {}
    decision = judge_title_block(
        TitleBlockCandidate(start=0, end=3, single=False, trigger="t"),
        records,
        _llm,
        warnings=warnings,
    )
    assert warnings["title_block_window_truncated"] == 1
    # cap=30 only fits the first (always-kept) line; the tail is dropped.
    assert "[1]" not in seen_prompt["prompt"]
    assert decision.member_indices == (0,)


def test_missing_llm_hard_fails() -> None:
    with pytest.raises(TitleBlockLLMError, match="none is configured"):
        judge_title_block(_CANDIDATE, _TITLE_RECORDS, None)


def test_single_candidate_window_includes_context() -> None:
    records = [
        _para("上文一。", size=12.0),
        _para("上文二。", size=12.0),
        _para("独立主标题", size=18.0, page_break_before=True),
        _para("下文一。", size=12.0),
        _para("下文二。", size=12.0),
    ]
    seen: dict = {}

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        seen["prompt"] = prompt
        return json.dumps(
            {"is_title_block": True, "main_title": "独立主标题"}, ensure_ascii=False
        )

    decision = judge_title_block(
        TitleBlockCandidate(start=2, end=3, single=True, trigger="single_line"),
        records,
        _llm,
    )
    # Window shows ±2 context paragraphs; the block itself is only the line.
    assert "上文一。" in seen["prompt"] and "下文二。" in seen["prompt"]
    assert decision.is_title_block
    assert decision.member_indices == (2,)


def test_single_candidate_line_survives_truncation(monkeypatch) -> None:
    """Review D2: when huge context would blow the token cap, the candidate
    line must still be shown to the LLM (it is pre-reserved) — otherwise a
    true verdict composes a level-0 heading from context alone."""
    monkeypatch.setenv("DOCX_SMART_LLM_WINDOW_TOKENS", "30")
    records = [
        _para("很长的上文内容" * 20, size=12.0),  # ~oversized context before
        _para("很长的上文内容之二" * 20, size=12.0),
        _para("独立主标题", size=18.0, page_break_before=True),  # the candidate
        _para("下文。", size=12.0),
    ]
    seen: dict = {}

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        seen["prompt"] = prompt
        return json.dumps(
            {"is_title_block": True, "main_title": "独立主标题"}, ensure_ascii=False
        )

    decision = judge_title_block(
        TitleBlockCandidate(start=2, end=3, single=True, trigger="single_line"),
        records,
        _llm,
        warnings={},
    )
    assert "独立主标题" in seen["prompt"]  # candidate never truncated away
    assert decision.is_title_block and decision.member_indices == (2,)


def test_single_candidate_main_title_must_be_the_candidate_line() -> None:
    """Review D4: the single-candidate hallucination guard is scoped to the
    candidate paragraph — a main_title lifted from reference-only context
    must NOT validate."""
    records = [
        _para("上下文里的另一句话", size=12.0),
        _para("真正的独立主标题", size=18.0, page_break_before=True),
        _para("下文。", size=12.0),
    ]
    # LLM (wrongly) returns a context paragraph as the main title.
    with pytest.raises(TitleBlockLLMError, match="cannot be located"):
        _judge_with(
            {"is_title_block": True, "main_title": "上下文里的另一句话"},
            records=records,
            candidate=TitleBlockCandidate(
                start=1, end=2, single=True, trigger="single_line"
            ),
        )


def test_single_candidate_false_verdict_leaves_context_untouched() -> None:
    """Review D3: a non-title verdict on a single candidate neither grants
    nor vetoes the reference-only context — the candidate re-enters the
    normal flow. The LLM's headings/body partition over context is ignored,
    and the candidate is NOT demoted."""
    records = [
        _para("上文一。", size=12.0),
        _para("上文二。", size=12.0),
        _para("独立大字号行", size=18.0, page_break_before=True),
        _para("下文一。", size=12.0),
        _para("下文二。", size=12.0),
    ]
    decision = _judge_with(
        # Even a hostile partition that votes the candidate + context as body
        # must not propagate.
        {"is_title_block": False, "headings": [], "body": [0, 1, 2, 3, 4]},
        records=records,
        candidate=TitleBlockCandidate(
            start=2, end=3, single=True, trigger="single_line"
        ),
    )
    assert not decision.is_title_block
    assert decision.heading_indices == ()
    assert decision.body_indices == ()  # candidate not demoted; context untouched


def test_multi_verdict_rejects_duplicate_indices() -> None:
    """Review D5: a duplicate index inside one list is not a valid partition
    even though the deduped sets would appear to cover the window."""
    with pytest.raises(TitleBlockLLMError, match="malformed"):
        _judge_with({"is_title_block": False, "headings": [0, 0], "body": [1, 2]})


# ---------------------------------------------------------------------------
# spec-vs-implementation audit regressions (A7 / A9)
# ---------------------------------------------------------------------------


def test_multi_window_requires_two_pt_over_body() -> None:
    """A7 tightened (§2.2.4 conservative preference): the multi-paragraph
    window title line now needs +2pt over FS_base, matching the single
    channel. The old +1pt strong-signal tier let ordinary section-heading
    sizes open windows, so it is no longer sufficient on its own."""
    # +1pt over base: below the (now unified) +2pt bar → no title line.
    plus_one = [
        _para("产品技术白皮书", size=13.0),
        _para("研发中心", size=12.0),
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
    ]
    assert _find(plus_one) == []

    # +2pt over base clears the bar → a multi_window candidate.
    plus_two = [
        _para("产品技术白皮书", size=14.0),
        _para("研发中心", size=12.0),
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
    ]
    assert [c.trigger for c in _find(plus_two)] == ["multi_window"]

    # Control: at exactly base size the window carries no title line.
    flat = [
        _para("产品技术白皮书", size=12.0),
        _para("研发中心", size=12.0),
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
    ]
    assert _find(flat) == []


def test_title_line_uses_first_line_size_for_split_records() -> None:
    """A9 (§2.2.2 / §3.1): a soft-break-split heading record is judged by
    its FIRST line's size, not the whole-paragraph dominant size that the
    demoted body remainder dominates."""
    split = _para(
        "被拆分的超长主标题首行",
        size=12.0,  # whole-paragraph dominant — swamped by the remainder
        first_line_font_size_pt=15.0,
        demoted_body_text="降级为正文的余部内容",
    )
    records = [
        split,
        _para("编制单位", size=12.0),
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
    ]
    cands = _find(records)
    assert [c.trigger for c in cands] == ["multi_window"]

    # Single-paragraph channel honours the same effective size (+2pt tier).
    single = _para(
        "整篇文档开头的孤立大标题",
        size=12.0,
        first_line_font_size_pt=14.5,
        demoted_body_text="软回车拆出的正文余部",
    )
    solo = [single] + [
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0)
        for _ in range(3)
    ]
    cands = _find(solo)
    assert [c.trigger for c in cands] == ["single_line"]


# ---------------------------------------------------------------------------
# multi-window boundary: real section headings are never title-block members
# ---------------------------------------------------------------------------


def test_multi_window_excludes_numbered_outline_headings() -> None:
    """Repro (M1212 标准化大纲): body-sized reference lines followed by real
    numbered+outlined section headings must NOT form a title block. The only
    lines reaching the +1pt title tier were the section headings themselves;
    excluding them leaves the reference run with no title line → no candidate,
    so the headings keep their standing and the reference text stays body."""
    records = [
        _para("GJB/Z 106A-2005  工艺标准化大纲编制指南", size=12.0),
        _para("GJB/Z 114A-2015  产品标准化大纲编制指南", size=12.0),
        _para("《M1212(ML001) 模块技术协议书》", size=12.0),
        _para("1.4   产品简介", size=14.0, outline_level_raw=1),
        _para("1.4.1   产品组成", size=14.0, outline_level_raw=2),
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
    ]
    assert _find(records) == []


def test_multi_window_stops_before_outline_heading() -> None:
    """A real section heading TERMINATES the window rather than nuking the
    whole candidate: a genuine cover run before it still yields a candidate
    whose ``end`` stops at the heading."""
    records = [
        _para("公司数字化转型白皮书", size=22.0),  # title line
        _para("研发中心发布", size=12.0),
        _para("概述", size=14.0, outline_level_raw=0),  # outline heading — boundary
        _para("这一段是以句号结尾的正文内容。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1 and not cands[0].single
    assert (cands[0].start, cands[0].end) == (0, 2)  # stops before the heading


def test_multi_window_outline_heading_not_a_window_start() -> None:
    """A window never STARTS on a real section heading (even a big one).

    Without the start guard, ``[0, 2)`` would be a multi_window candidate
    (18pt title line + a body-sized companion); the outline heading blocks it.
    """
    records = [
        _para("引言", size=18.0, outline_level_raw=0),  # outline heading — big
        _para("研发中心", size=12.0),
        _para("这一段是以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
    ]
    assert _find(records) == []


def test_multi_window_stops_before_numbered_heading() -> None:
    """Genuine numbering (no physical outline) is also a boundary — the
    single-line channel already excludes it; multi-window now matches."""
    records = [
        _para("公司数字化转型白皮书", size=22.0),
        _para("研发中心发布", size=12.0),
        _para("第二章 总体设计", size=14.0),  # genuine numbering, no outline
        _para("这一段是以句号结尾的正文内容。", size=12.0),
    ]
    # numbering_veto defaults to _stub_no_veto → the numbering stays genuine.
    cands = _find(records)
    assert len(cands) == 1 and (cands[0].start, cands[0].end) == (0, 2)


def test_multi_window_boundary_uses_raw_outline_level() -> None:
    """Field-choice pin: a length-demoted heading (``outline_level`` None,
    ``outline_level_raw`` set) is still a boundary. Using the post-policy
    ``outline_level`` would miss it and sweep its text into the block, where
    a title-block member's text is lost — so the raw level is the right test.
    """
    records = [
        _para("公司数字化转型白皮书", size=22.0),
        _para("研发中心发布", size=12.0),
        _para(
            "某个被样式标成标题却因过长而被降级的段落标题行内容",
            size=14.0,
            outline_level=None,
            outline_level_raw=1,
        ),
        _para("这一段是以句号结尾的正文内容。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1 and (cands[0].start, cands[0].end) == (0, 2)


def test_single_candidate_outline_heading_excluded() -> None:
    """The single-paragraph channel excludes real section headings too: an
    outline-carrying heading at a boundary (here doc start) is a top-level
    heading for the outline anchor path, not a level-0 title block — mirroring
    the numbering exclusion so a genuine heading never splits sub-documents."""
    records = [
        _para("引言", size=18.0, outline_level_raw=0),  # Heading-1 at doc start
        _para("正文第一段说明了总体情况。", size=12.0),
        _para("正文第二段继续说明。", size=12.0),
    ]
    assert _find(records) == []

    # Control: the same lone big first line WITHOUT an outline is still a
    # single-paragraph candidate (the feature is intact for non-headings).
    plain = [
        _para("某某产品发布公告", size=18.0),
        _para("正文第一段说明了总体情况。", size=12.0),
        _para("正文第二段继续说明。", size=12.0),
    ]
    cands = _find(plain)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 0


def test_multi_window_plain_cover_still_found() -> None:
    """Over-suppression guard: a legitimate cover block (big main title plus
    body-sized companions, no outline, no numbering) still forms a candidate."""
    records = [
        _para("某某产品标准化大纲", size=22.0),
        _para("副标题：模块化设计", size=14.0),
        _para("某某研究所 发布", size=12.0),
        _para("这一段是以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1 and not cands[0].single
    assert (cands[0].start, cands[0].end) == (0, 3)


# ---------------------------------------------------------------------------
# visual-dominance gate (Rule 2): a title block must out-size a neighbouring
# section heading on at least one flank
# ---------------------------------------------------------------------------


def test_multi_window_rejected_when_not_dominant_over_headings() -> None:
    """A cluster whose biggest line is SMALLER than the neighbouring section
    headings is a mid-body emphasis block, not a cover title — rejected even
    though the line clears +2pt over body."""
    records = [
        _para("第一章 绪论", size=18.0, outline_level_raw=0),  # heading before
        _para("一段以句号结尾的正文内容，用来隔开上一个标题。", size=12.0),
        _para("强调小标题", size=14.0),  # +2pt over body, but < 18pt headings
        _para("配套的说明行", size=12.0),
        _para("一段以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
        _para("第二章 概览", size=18.0, outline_level_raw=0),  # heading after
    ]
    # window is [2, 4): 强调小标题(14) + 配套的说明行(12); flanked by 18pt
    # headings on both sides → 14 is not dominant → no candidate.
    assert _find(records) == []


def test_multi_window_admitted_when_dominant_over_one_flank() -> None:
    """OR semantics: dominating the heading on EITHER flank is enough."""
    records = [
        _para("第一章 绪论", size=20.0, outline_level_raw=0),  # bigger flank
        _para("一段以句号结尾的正文内容，用来隔开上一个标题。", size=12.0),
        _para("某某产品发布公告", size=18.0),  # title line
        _para("配套的说明行", size=12.0),
        _para("一段以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
        _para("第二章 概览", size=16.0, outline_level_raw=0),  # smaller flank
    ]
    # 18pt beats the 16pt following heading (though not the 20pt preceding
    # one) → dominant on one flank → candidate over [2, 4).
    cands = _find(records)
    assert len(cands) == 1 and (cands[0].start, cands[0].end) == (2, 4)


def test_multi_window_dominance_passes_without_neighbor_headings() -> None:
    """No comparable section heading flanks the window (unstructured doc) →
    the relative rule is inapplicable and passes; the +2pt absolute bar still
    guards the candidate."""
    records = [
        _para("公司数字化转型白皮书", size=14.0),  # +2pt, no heading neighbours
        _para("研发中心发布", size=12.0),
        _para("这一段是以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1 and (cands[0].start, cands[0].end) == (0, 2)


def test_dominance_reference_includes_numbered_headings() -> None:
    """The neighbour reference is a real heading = physical outline OR genuine
    numbering; a numbered-but-un-outlined heading still counts."""
    records = [
        _para("第一章 绪论", size=18.0),  # genuine numbering, NO outline level
        _para("一段以句号结尾的正文内容，用来隔开上一个标题。", size=12.0),
        _para("强调小标题", size=14.0),  # +2pt, but < 18pt numbered heading
        _para("配套的说明行", size=12.0),
        _para("一段以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
    ]
    # Only flank heading is numbered (no outline) but still bounds dominance;
    # 14 < 18 and no other neighbour → not dominant → no candidate.
    assert _find(records) == []


def test_dominance_ignores_headings_beyond_flank_window(monkeypatch) -> None:
    """Section headings farther than K=20 paragraphs from the window are not
    compared against; with a small K the distant big heading is ignored and
    the window (no near neighbour) passes on the fallback."""
    import lightrag.parser.docx.smart_heading.title_block as tb

    monkeypatch.setattr(tb, "_FLANK_WINDOW", 1)
    records = [
        _para("第一章 绪论", size=20.0, outline_level_raw=0),  # 2 paras away
        _para("一段以句号结尾的正文内容，用来隔开。", size=12.0),
        _para("某某产品发布公告", size=14.0),  # title line, +2pt over body
        _para("配套的说明行", size=12.0),
        _para("一段以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
    ]
    # With K=1 the 20pt heading (2 body paras before the window start) is out
    # of reach → no comparable neighbour → dominance passes → candidate.
    cands = _find(records)
    assert len(cands) == 1 and (cands[0].start, cands[0].end) == (2, 4)


def test_single_candidate_after_previous_paragraph_break_run() -> None:
    """A8 (§2.2.4 evidence b): a w:br type="page" run in the PREVIOUS
    paragraph is boundary evidence for the next paragraph; the paragraph
    carrying only a trailing break run gains no self-evidence."""
    records = [
        _para("上一篇的收尾正文，以句号结尾。", size=12.0, has_page_break_run=True),
        _para("下一篇文章的标题", size=18.0),
        _para("下一篇正文开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 1


# ---------------------------------------------------------------------------
# table windows (§2.2.4 table channel): a cover laid out inside tables
# ---------------------------------------------------------------------------


def _table(rows, **kw) -> ParagraphRecord:
    """Fake table record; ``rows`` = per-row lists of (text, size, outline)."""
    return ParagraphRecord(
        kind="table", text="<table>[]</table>", table_cell_features=rows, **kw
    )


def _cover_tables() -> list[ParagraphRecord]:
    """M1212-style cover: form tables (multi-cell rows, short text) + a title
    table (single-cell big rows) + a publisher table."""
    return [
        _table(
            [
                [("档 号", 10.5, False), ("", None, False)],
                [("版 本 号", 10.5, False), ("1V1.0.0", 10.5, False)],
            ]
        ),
        _empty(),
        _table(
            [
                [("产品标准化大纲", 22.0, False)],  # title rows: single cell
                [("某某模块", 22.0, False)],
            ]
        ),
        _table([[("某某电子股份有限公司", 16.0, False)]]),
    ]


def test_table_window_merges_adjacent_qualifying_tables() -> None:
    """Adjacent member-qualifying tables (blank lines between are fine) merge
    into ONE table_window candidate spanning form + title + publisher."""
    records = _cover_tables() + [
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    c = cands[0]
    assert c.table and c.trigger == "table_window" and not c.single
    assert (c.start, c.end) == (0, 4)


def test_table_window_prefix_breaks_at_disqualified_table() -> None:
    """Rule 3: the first non-qualifying table ends the window AND the run —
    a later qualifying table in the same run never seeds a new window."""
    data_table = _table(
        [[("序号", 10.5, False), ("这一格是以句号结尾的数据内容。", 10.5, False)]]
    )
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        data_table,
        _table([[("另一份大字标题", 22.0, False)]]),  # after the break: ignored
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert (cands[0].start, cands[0].end) == (0, 1)  # prefix only


def test_table_window_requires_qualifying_first_table() -> None:
    """A run whose FIRST table is disqualified yields no candidate at all,
    even though a qualifying title table follows in the same run."""
    records = [
        _table(
            [[("序号", 10.5, False), ("这一格是以句号结尾的数据内容。", 10.5, False)]]
        ),
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []


@pytest.mark.parametrize(
    "bad_cell",
    [
        ("这一格是以句号结尾的正文语句。", 10.5, False),  # strong body
        ("超" * 40, 10.5, False),  # 40 CJK = 120 weighted > 90 cap
        ("带物理大纲的格", 10.5, True),  # physical outline
    ],
)
def test_table_member_gate_rejects_bad_cells(bad_cell) -> None:
    """Rule 2: ONE offending cell disqualifies the whole table."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)], [bad_cell]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []


def test_table_title_row_requires_single_big_unnumbered_cell() -> None:
    """Rule 1: the title row is a single-PHYSICAL-cell row at the +2pt tier
    whose text has no genuine numbering."""
    # (a) all sizes below fs_base+2 → no title row.
    records = [
        _table([[("产品标准化大纲", 13.0, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []

    # (b) big text but in a MULTI-cell row → not a title row.
    records = [
        _table([[("产品标准化大纲", 22.0, False), ("旁边一格", 10.5, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []

    # (c) genuine numbering disqualifies the title row (consistent with the
    # paragraph channels)…
    records = [
        _table([[("第二章 总体设计", 22.0, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []

    # …but the homophone veto keeps a date-opener title alive.
    records = [
        _table([[("2026年度工作报告", 22.0, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records, numbering_veto=_stub_always_veto)
    assert len(cands) == 1 and cands[0].table


def test_table_window_dominance_gate() -> None:
    """The table window shares the visual-dominance gate: a bigger real
    heading on both flanks rejects the candidate."""
    records = [
        _para("第一章 绪论", size=20.0, outline_level_raw=0),
        _para("一段以句号结尾的正文，用来隔开标题。", size=12.0),
        _table([[("强调框标题", 14.0, False)]]),  # +2pt but < 20pt heading
        _para("一段以句号结尾的正文。", size=12.0),
        _para("第二章 概览", size=20.0, outline_level_raw=0),
    ]
    assert _find(records) == []


# --- judge: table candidates ------------------------------------------------

_TABLE_RECORDS = [
    _table(
        [
            [("产品标准化大纲", 22.0, False)],
            [("某某模块", 22.0, False)],
        ]
    ),
    _table([[("某某电子股份有限公司", 16.0, False)]]),
    _para("正文从这里开始，以句号结尾。", size=12.0),
]
_TABLE_CANDIDATE = TitleBlockCandidate(
    start=0, end=2, single=False, trigger="table_window", table=True
)


def test_table_verdict_members_are_table_records() -> None:
    decision = _judge_with(
        {
            "is_title_block": True,
            "main_title": "产品标准化大纲某某模块",  # concatenated cells
            "publisher": "某某电子股份有限公司",
        },
        records=_TABLE_RECORDS,
        candidate=_TABLE_CANDIDATE,
    )
    assert decision.is_title_block
    assert decision.member_indices == (0, 1)  # the TABLE record indices
    assert (
        compose_title_heading(decision)
        == "产品标准化大纲某某模块(某某电子股份有限公司)"
    )


def test_table_verdict_locates_against_cell_texts() -> None:
    """Locate-back runs against the CELL texts (the record text is a <table>
    placeholder); a hallucinated title hard-fails."""
    with pytest.raises(TitleBlockLLMError):
        _judge_with(
            {"is_title_block": True, "main_title": "凭空捏造的标题"},
            records=_TABLE_RECORDS,
            candidate=_TABLE_CANDIDATE,
        )


def test_table_non_title_verdict_ignored() -> None:
    """A non-title verdict for a table window grants/vetoes nothing (D3):
    cells are not paragraph records, so the partition is ignored even when
    it is malformed."""
    decision = _judge_with(
        {"is_title_block": False, "headings": [0, 0], "body": []},  # malformed
        records=_TABLE_RECORDS,
        candidate=_TABLE_CANDIDATE,
    )
    assert not decision.is_title_block
    assert decision.heading_indices == () and decision.body_indices == ()


# ---------------------------------------------------------------------------
# font-size evidence legend (the M1212 regression: a metadata-heavy table
# cover whose 22pt title the text-only window could not distinguish from
# the surrounding form labels)
# ---------------------------------------------------------------------------


def _prompt_for(records, candidate, *, fs_base_pt=None, payload=None):
    seen: dict = {}

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        seen["prompt"] = prompt
        return json.dumps(
            payload or {"is_title_block": False, "headings": [], "body": []},
            ensure_ascii=False,
        )

    judge_title_block(candidate, records, _llm, warnings={}, fs_base_pt=fs_base_pt)
    return seen["prompt"]


def test_table_window_legend_tiers_largest_over_enlarged() -> None:
    """M1212 regression shape: form-metadata tables + a 22pt title table +
    a 16pt publisher table + an absorbed 16pt next-page heading. The legend
    lists the 22pt lines as LARGEST and the 16pt lines one tier below, so
    the enlarged-but-not-title lines never read as strongly."""
    records = [
        _table(
            [
                [("档 号", 10.5, False), ("", None, False)],
                [("版 本 号", 10.5, False), ("1V1.0.0", 10.5, False)],
            ]
        ),
        _table([[("产品标准化大纲", 22.0, False)], [("某某模块", 22.0, False)]]),
        _table([[("某某电子股份有限公司", 16.0, False)]]),
        _para("更 改 记 录", size=16.0),  # absorbed next-page heading
    ]
    candidate = TitleBlockCandidate(
        start=0,
        end=4,
        single=False,
        trigger="table_window",
        table=True,
        members=(0, 1, 2, 3),
    )
    prompt = _prompt_for(records, candidate, fs_base_pt=12.0)
    # Window lines: [0] 档 号, [1] 版 本 号, [2] 1V1.0.0, [3] 产品标准化大纲,
    # [4] 某某模块, [5] 某某电子股份有限公司, [6] 更 改 记 录.
    assert "Font-size evidence — largest lines: [3]=22pt, [4]=22pt" in prompt
    assert "other enlarged lines: [5]=16pt, [6]=16pt" in prompt
    assert "(body ≈ 12pt)" in prompt


def test_paragraph_window_legend_uses_indexed_lines() -> None:
    """[BLANK] rows carry no index, so the legend must number by indexed
    lines — a raw-row count would point at the wrong text."""
    records = [
        _para("公司数字化转型白皮书", size=22.0),
        _empty(),
        _para("（2026年版）", size=12.0),
        _para("某某咨询公司发布", size=16.0),
    ]
    candidate = TitleBlockCandidate(
        start=0, end=4, single=False, trigger="multi_window"
    )
    prompt = _prompt_for(records, candidate, fs_base_pt=12.0)
    assert "Font-size evidence — largest lines: [0]=22pt" in prompt
    # record 3 renders as indexed line [2] (the blank row shifts raw rows).
    assert "other enlarged lines: [2]=16pt" in prompt


def test_legend_absent_without_fs_base_or_dominant_line() -> None:
    """No baseline (hand-built candidates) or no dominant line → the prompt
    is legend-free, byte-identical to the pre-evidence form."""
    records = [
        _para("普通一行", size=12.0),
        _para("另一行", size=12.0),
    ]
    candidate = TitleBlockCandidate(
        start=0, end=2, single=False, trigger="multi_window"
    )
    assert "Font-size evidence" not in _prompt_for(records, candidate)
    assert "Font-size evidence" not in _prompt_for(records, candidate, fs_base_pt=12.0)


def test_single_window_legend_covers_only_the_candidate_line() -> None:
    """Single candidates: the ±context lines are reference-only and a true
    verdict's main_title is locate-back-scoped to the candidate line alone —
    a LARGER context line must never be advertised as the largest line, or
    the LLM would follow the cue into a guaranteed locate-back hard fail."""
    records = [
        _para("上一篇文档的巨大结尾行", size=26.0),  # bigger, but context-only
        _para("独立主标题", size=18.0, page_break_before=True),
        _para("正文从这里开始。", size=12.0),
    ]
    candidate = TitleBlockCandidate(start=1, end=2, single=True, trigger="single_line")
    prompt = _prompt_for(
        records,
        candidate,
        fs_base_pt=12.0,
        payload={"is_title_block": True, "main_title": "独立主标题"},
    )
    assert "Font-size evidence — largest lines: [1]=18pt" in prompt
    assert "26pt" not in prompt  # the context line carries no evidence


def test_legend_skips_truncated_lines(monkeypatch) -> None:
    """A token-truncated line is not emitted, so it must not surface in the
    legend either — the legend only ever cites lines the LLM can see."""
    monkeypatch.setenv("DOCX_SMART_LLM_WINDOW_TOKENS", "30")
    records = [
        _para("标题块第一行内容比较长" * 3, size=22.0),
        _para("被截掉的大字号尾行" * 3, size=22.0),
    ]
    candidate = TitleBlockCandidate(
        start=0, end=2, single=False, trigger="multi_window"
    )
    prompt = _prompt_for(
        records,
        candidate,
        fs_base_pt=12.0,
        payload={"is_title_block": True, "main_title": records[0].text},
    )
    assert "largest lines: [0]=22pt" in prompt
    assert "[1]" not in prompt  # neither as window line nor in the legend


def test_prompt_front_matter_negative_list() -> None:
    """Guard the front-matter negative list against prompt refactors.

    The list deliberately lives INSIDE the user template's "If true:
    main_title" field rule — as a selection constraint only — and NOT in the
    system prompt as a window-verdict rule: A/B probes against the deployed
    judge showed any verdict-level phrasing flips genuine covers that merely
    CONTAIN such a heading (the M1212 table window absorbs a trailing
    更 改 记 录 line) to false, while the field constraint keeps every
    verdict intact. The needles cover the listed names, the Chinese
    whitespace-insensitivity note, and the substring-acceptable escape."""
    from lightrag.parser.docx.smart_heading.title_block import (
        _SYSTEM_PROMPT,
        _USER_TEMPLATE,
    )

    for needle in (
        "NEVER a front-matter/bookkeeping heading",
        "目次",
        "更改记录",
        "Revision History",
        "ignoring whitespace between Chinese characters",
        "更改记录管理规范",
    ):
        assert needle in _USER_TEMPLATE, needle
    # The verdict-level system prompt must stay list-free (see above).
    assert "更改记录" not in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 公文版记 (imprint) veto: the marker line and its 2 preceding non-blank
# paragraphs are barred from every title-block channel
# ---------------------------------------------------------------------------


def test_imprint_breaks_multi_window_and_vetoes_two_neighbors() -> None:
    """The window ends 2 non-blank paragraphs BEFORE the imprint line; with
    the veto injected away, the imprint line and its neighbours would be
    absorbed as cover-title material (the bug being fixed)."""
    records = [
        _para("公司数字化转型白皮书", size=22.0),  # 0 big main title
        _para("（2026年版）", size=12.0),  # 1
        _para("某某市人民政府办公室", size=12.0),  # 2 ← vetoed (2nd preceding)
        _para("研究策划部编写", size=12.0),  # 3 ← vetoed (1st preceding)
        _para("抄送：各区人民政府", size=12.0),  # 4 anchor
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert not cands[0].single
    assert (cands[0].start, cands[0].end) == (0, 2)

    # Injectable: a no-op marker restores the pre-fix absorption behaviour.
    cands = _find(records, imprint_marker=lambda t: None)
    assert len(cands) == 1
    assert (cands[0].start, cands[0].end) == (0, 5)


def test_imprint_line_never_single_candidate() -> None:
    """Mirror of test_single_candidate_centered_with_blank_flanks: the same
    boundary evidence yields a candidate for a plain line but never for an
    imprint line (stub strong_body does NOT know imprint — the title_block
    veto must hold on its own)."""
    records = [
        _para("开头正文，以句号结尾。", size=12.0),
        _empty(),
        _para("抄送：各市人民政府", size=18.0, alignment="center"),
        _empty(),
        _para("后续正文，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []
    cands = _find(records, imprint_marker=lambda t: None)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 2


def test_imprint_neighbor_veto_blocks_single_candidate() -> None:
    """Tail signature/date lines above an imprint line stop being title
    candidates; without the imprint line the same area yields one."""
    body = [_para(f"正文段落{i}，以句号结尾。", size=12.0) for i in range(3)]
    records = body + [
        _empty(),
        _para("某某市人民政府办公室", size=18.0, alignment="center"),  # 4
        _empty(),
        _para("二〇二六年七月一日", size=12.0),  # 6 ← vetoed
        _para("抄送：各县区人民政府", size=12.0),  # 7 anchor
    ]
    assert _find(records) == []
    # Control: drop the imprint line → the signature line is a candidate.
    cands = _find(records[:-1])
    assert len(cands) == 1 and cands[0].start == 4


def test_imprint_neighbor_walk_skips_blank_paras() -> None:
    """The backward walk skips empty/whitespace-only paragraphs without
    consuming the 2-paragraph budget."""
    records = [
        _para("某某公司发文稿纸", size=22.0),  # 0 ← vetoed (2nd preceding)
        _empty(),
        _para("某某办公室", size=12.0),  # 2 ← vetoed (1st preceding)
        _para("   ", size=12.0),  # whitespace-only para: skipped, not counted
        _para("抄送：各成员单位", size=12.0),  # 4 anchor
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []
    assert len(_find(records, imprint_marker=lambda t: None)) == 1


def test_imprint_neighbor_walk_stops_at_table() -> None:
    """A non-paragraph record is a structural boundary: the veto never leaks
    across it, so the window above the table survives intact."""
    records = [
        _para("产品发布白皮书", size=22.0),  # 0
        _para("某某公司编", size=12.0),  # 1
        _table([[("附件清单", 10.5, False)]]),  # 2 boundary
        _para("某某办公室代拟", size=12.0),  # 3 ← vetoed (walk stops at table)
        _para("抄送：各部门", size=12.0),  # 4 anchor
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert (cands[0].start, cands[0].end) == (0, 2)


def test_table_member_rejects_imprint_cell() -> None:
    """One imprint cell disqualifies the whole table from the table channel
    (a short imprint cell is not caught by the strong-body stub)."""
    records = [
        _table(
            [
                [("产品标准化大纲", 22.0, False)],
                [("抄送：市委各部门", 10.5, False)],
            ]
        ),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []


def test_table_member_rejects_closer_cell() -> None:
    """A 印发-family CLOSER cell also disqualifies the table (no anchor-above
    context in a table, so the closer must veto on its own here)."""
    records = [
        _table(
            [
                [("产品标准化大纲", 22.0, False)],
                [("某某办公室 2026年6月30日 印发", 10.5, False)],
            ]
        ),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []


# ---------------------------------------------------------------------------
# mixed table/paragraph covers: absorbable paragraphs join the table run
# (image-only, cover-material, paragraph-borne main title, paragraph tail)
# ---------------------------------------------------------------------------

_FORM_TABLE_ROWS = [
    [("档 号", 10.5, False), ("", None, False)],
    [("密 级", 10.5, False), ("公开", 10.5, False)],
]
_PUB_TABLE_ROWS = [[("某某电子股份有限公司", 16.0, False)]]


def test_image_para_no_longer_breaks_table_run() -> None:
    """A pure <drawing/> paragraph (印章/logo) between cover tables is absorbed
    instead of splitting the cover; members carry it in source order."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para('<drawing id="1" name="图"/>', size=12.0),
        _table(_PUB_TABLE_ROWS),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    c = cands[0]
    assert c.table and (c.start, c.end) == (0, 3)
    assert c.members == (0, 1, 2)


def test_mixed_cover_paragraph_carries_main_title() -> None:
    """档号表(小字,无 title row) + 主标题段(22pt) + 发文机关表: the candidate
    stands on the PARAGRAPH's size — the table channel is no longer
    tables-only."""
    records = [
        _table(_FORM_TABLE_ROWS),
        _para("某某管理办法", size=22.0),
        _table(_PUB_TABLE_ROWS),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert cands[0].members == (0, 1, 2)


def test_mixed_image_and_title_para_stands_on_semantic_text() -> None:
    """logo 与主标题同段: the long drawing tag must not push the title line
    over the length cap — length judges the semantic text, size the real
    paragraph size."""
    records = [
        _table(_FORM_TABLE_ROWS),
        _para(
            '<drawing id="1" name="logo" path="assets/very-long-path-to-'
            'image-file-name-here.png"/> 某某管理办法',
            size=22.0,
        ),
        _table(_PUB_TABLE_ROWS),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert cands[0].members == (0, 1, 2)


def test_cover_material_para_with_formula_absorbed() -> None:
    """A short cover-material paragraph carrying an inline formula joins the
    cover (情形 2 user ruling); the formula text rides along losslessly."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("副标题 <equation>x</equation>", size=16.0),
        _table(_PUB_TABLE_ROWS),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert cands[0].members == (0, 1, 2)


def test_body_para_still_breaks_table_run() -> None:
    """Control: a strong-body paragraph between tables still splits the run —
    absorption is for cover material only."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("这是一段以句号结尾的正文。", size=12.0),
        _table(_PUB_TABLE_ROWS),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert [(c.start, c.end) for c in cands] == [(0, 1), (2, 3)]


def test_real_heading_para_still_breaks_table_run() -> None:
    """Control: a physical-outline paragraph between tables is a real section
    heading, never cover material — the run splits."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("第一章 总则", size=12.0, outline_level_raw=0),
        _table(_PUB_TABLE_ROWS),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert all(1 not in c.members for c in cands)


def test_drawing_prefixed_numbered_heading_still_breaks_table_run() -> None:
    """A leading <drawing/> tag must not smuggle a genuinely NUMBERED heading
    into the cover: raw-text numbering classification is defeated by the tag,
    so the absorbable gate re-checks real numbering on the semantic text."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para('<drawing id="1" name="装饰"/> 第一章 总则', size=12.0),
        _table(_PUB_TABLE_ROWS),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert all(1 not in c.members for c in cands)


def test_paragraph_tail_cover() -> None:
    """A cover ending in a paragraph (档号表 + 主标题段, no trailing table)
    still forms one candidate — no tail trimming (user ruling)."""
    records = [
        _table(_FORM_TABLE_ROWS),
        _para("某某管理办法", size=22.0),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert (cands[0].start, cands[0].end) == (0, 2)
    assert cands[0].members == (0, 1)


def test_trailing_image_para_joins_cover() -> None:
    """标题表 + 尾随印章图片段: the trailing image joins the cover members."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para('<drawing id="9" name="印章"/>', size=12.0),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert cands[0].members == (0, 1)


def test_imprint_preceding_line_not_absorbed_by_table_run() -> None:
    """版记 region neighbourhood protection reaches the table channel: the
    signature line PRECEDING a 抄送 anchor sits in imprint_excluded and is
    never absorbed as cover material."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("某某市人民政府办公室", size=12.0),  # preceding of the anchor below
        _para("抄送：各区人民政府", size=12.0),  # imprint anchor
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert cands[0].members == (0,)  # the signature line stayed out


def test_table_verdict_locates_paragraph_main_title() -> None:
    """judge 层: a members-bearing candidate renders the absorbed paragraph
    into the LLM window and canon, so a paragraph-borne main_title passes
    locate-back and lands in member_indices in source order."""
    records = [
        _table(_FORM_TABLE_ROWS),
        _para("某某管理办法", size=22.0),
        _table(_PUB_TABLE_ROWS),
    ]
    candidate = TitleBlockCandidate(
        start=0,
        end=3,
        single=False,
        trigger="table_window",
        table=True,
        members=(0, 1, 2),
    )
    decision = _judge_with(
        {"is_title_block": True, "main_title": "某某管理办法"},
        records=records,
        candidate=candidate,
    )
    assert decision.is_title_block
    assert decision.member_indices == (0, 1, 2)


def test_mixed_para_locates_semantic_main_title() -> None:
    """A mid-line drawing tag must not break locate-back: the canon carries
    the para's SEMANTIC text, so the LLM's tag-free main_title ("某某管理办法"
    out of "某某<drawing/>管理办法") still matches contiguously."""
    records = [
        _table(_FORM_TABLE_ROWS),
        _para('某某<drawing id="1" name="纹章"/>管理办法', size=22.0),
        _table(_PUB_TABLE_ROWS),
    ]
    candidate = TitleBlockCandidate(
        start=0,
        end=3,
        single=False,
        trigger="table_window",
        table=True,
        members=(0, 1, 2),
    )
    decision = _judge_with(
        {"is_title_block": True, "main_title": "某某管理办法"},
        records=records,
        candidate=candidate,
    )
    assert decision.is_title_block
    assert decision.member_indices == (0, 1, 2)


def test_llm_window_contains_no_image_placeholders() -> None:
    """Images are removed from the LLM window ENTIRELY (least noise, nothing
    for the LLM to echo back): a pure-image member contributes no window line,
    a mixed line reads as its clean semantic text — while both stay members
    so assembly keeps them in source order."""
    prompts: list[str] = []

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        prompts.append(prompt)
        return json.dumps(
            {"is_title_block": True, "main_title": "某某管理办法"},
            ensure_ascii=False,
        )

    # Table channel: pure-image member + mixed title member.
    records = [
        _table(_FORM_TABLE_ROWS),
        _para('<drawing id="9" name="印章" path="a/b.png"/>', size=12.0),
        _para('某某<drawing id="1" name="纹章"/>管理办法', size=22.0),
        _table(_PUB_TABLE_ROWS),
    ]
    candidate = TitleBlockCandidate(
        start=0,
        end=4,
        single=False,
        trigger="table_window",
        table=True,
        members=(0, 1, 2, 3),
    )
    decision = judge_title_block(candidate, records, _llm, warnings={})
    assert decision.is_title_block
    assert decision.member_indices == (0, 1, 2, 3)  # image kept as a member
    assert "<drawing" not in prompts[-1]
    assert "[IMAGE]" not in prompts[-1]
    assert "某某管理办法" in prompts[-1]  # mixed line reads clean

    # Paragraph channel (multi-window): same stripping + semantic locate; the
    # pure-image paragraph contributes no window line at all.
    records2 = [
        _para('某某<drawing id="1" name="纹章"/>管理办法', size=22.0),
        _para('<drawing id="9" name="印章"/>', size=12.0),
        _para("（2026年版）", size=12.0),
    ]
    cand2 = TitleBlockCandidate(start=0, end=3, single=False, trigger="multi_window")
    decision2 = judge_title_block(cand2, records2, _llm, warnings={})
    assert decision2.is_title_block
    assert "<drawing" not in prompts[-1]
    assert "[IMAGE]" not in prompts[-1]
    assert "某某管理办法" in prompts[-1]
    assert "（2026年版）" in prompts[-1]  # image line skipped, context intact
    # The unrendered image paragraph STAYS a member (source-order assembly);
    # dropping it would re-emit the image after the block's members.
    assert decision2.member_indices == (0, 1, 2)


# ---------------------------------------------------------------------------
# 版记 region model: detect_imprint_regions (抄送 anchor → 印发 closer, forward
# window, TOC/blank skip, structural boundary)
# ---------------------------------------------------------------------------


def _regions(records, **kw):
    kw.setdefault("imprint_marker", guardrails.imprint_marker_reason)
    kw.setdefault("imprint_closer", guardrails.imprint_closer_reason)
    return detect_imprint_regions(records, **kw)


def test_imprint_region_closer_within_forward_window() -> None:
    """A 抄送 anchor closes on the first 印发 line inside the 3-paragraph
    forward window; the middle lines join the region."""
    records = [
        _para("抄送：各区人民政府", size=12.0),  # 0 anchor
        _para("中间行一", size=12.0),  # 1 middle
        _para("某某办公室 2026年6月30日 印发", size=12.0),  # 2 closer (trailing)
    ]
    regs = _regions(records)
    assert len(regs) == 1
    assert regs[0].anchor == 0 and regs[0].closer == 2
    assert regs[0].members == {0, 1, 2}


def test_imprint_region_closer_beyond_window_falls_back() -> None:
    """A 印发 line as the 4th non-blank paragraph is out of window → no closer,
    region degrades to the anchor alone (pre-region behaviour)."""
    records = [
        _para("抄送：各区人民政府", size=12.0),  # 0 anchor
        _para("中间行一", size=12.0),  # 1
        _para("中间行二", size=12.0),  # 2
        _para("中间行三", size=12.0),  # 3
        _para("某某办公室 2026年 印发", size=12.0),  # 4 closer (too far)
    ]
    regs = _regions(records)
    assert len(regs) == 1
    assert regs[0].closer is None
    assert regs[0].members == {0}


def test_imprint_region_walks_skip_toc(monkeypatch) -> None:
    """N2: TOC lines (skip_indices) are stepped over WITHOUT spending the
    backward-2 / forward budget and never join a region."""
    records = [
        _para("上文", size=12.0),  # 0 ← preceding (2nd)
        _para("目录行", size=12.0),  # 1 TOC — skipped, not counted
        _para("署名行", size=12.0),  # 2 ← preceding (1st)
        _para("抄送：各区", size=12.0),  # 3 anchor
        _para("目录行二", size=12.0),  # 4 TOC — skipped, not counted
        _para("某某 2026年 印发", size=12.0),  # 5 closer
    ]
    regs = _regions(records, skip_indices={1, 4})
    assert len(regs) == 1
    r = regs[0]
    assert r.preceding == {0, 2}  # TOC(1) skipped without eating budget
    assert r.closer == 5 and r.members == {3, 5}  # TOC(4) not a member


def test_imprint_region_forward_walk_stops_at_table() -> None:
    """A non-paragraph record ends the forward walk: a closer beyond a table
    is unreachable, region falls back to the anchor."""
    records = [
        _para("抄送：各部门", size=12.0),  # 0 anchor
        _table([[("附件清单", 10.5, False)]]),  # 1 boundary
        _para("某某办公室 2026年 印发", size=12.0),  # 2 (past the table)
    ]
    regs = _regions(records)
    assert regs[0].closer is None and regs[0].members == {0}


def test_imprint_region_start_marker_is_middle_content() -> None:
    """A 主题词-opened region runs THROUGH a following 抄送 (middle content, not
    a closer) to reach the 印发 closer; both anchors are vetoed."""
    records = [
        _para("主题词：经济 管理", size=12.0),  # 0 anchor (主题词)
        _para("抄送：各区人民政府", size=12.0),  # 1 middle content (own anchor too)
        _para("某某办公室 2026年 印发", size=12.0),  # 2 closer
    ]
    by_anchor = {r.anchor: r for r in _regions(records)}
    assert by_anchor[0].closer == 2 and by_anchor[0].members == {0, 1, 2}
    assert by_anchor[1].closer == 2 and by_anchor[1].members == {1, 2}


def test_imprint_region_closer_is_yinfa_jiguan() -> None:
    """印发机关 is a CLOSER (not an anchor): it ends a 抄送-opened region — the
    old space-class anchor knob (DOCX_SMART_IMPRINT_SPACE_PREFIXES) is gone."""
    from lightrag.parser.docx.smart_heading.guardrails import imprint_marker_reason

    records = [
        _para("抄送：各区", size=12.0),  # 0 anchor
        _para("印发机关　某某市人民政府办公厅", size=12.0),  # 1 closer
    ]
    assert imprint_marker_reason(records[1].text) is None  # never an anchor
    regs = _regions(records)
    assert len(regs) == 1
    assert regs[0].closer == 1 and regs[0].members == {0, 1}


def test_imprint_region_absorbs_trailing_document_date() -> None:
    """A mis-ordered 成文日期 right after the 印发 closer (版记 THEN date — not the
    GB/T order) is pulled into the region; it belongs to THIS document."""
    records = [
        _para("抄送：各设区市城乡规划局", size=12.0),  # 0 anchor
        _para("河北省住房和城乡建设厅办公室   2009年7月6日印发", size=12.0),  # 1 closer
        _para("二○○九年七月六日", size=12.0),  # 2 成文日期 → absorbed
        _para("正文从这里开始，以句号结尾。", size=12.0),  # 3 not a date → stop
    ]
    regs = _regions(records)
    assert len(regs) == 1
    assert regs[0].closer == 1 and regs[0].members == {0, 1, 2}


def test_imprint_region_date_absorb_stops_at_non_date() -> None:
    """Only bare date-only lines right after the closer are pulled in; the walk
    stops at the first non-date paragraph (a later date does not leak in)."""
    records = [
        _para("抄送：各区", size=12.0),  # 0 anchor
        _para("某某办公室 2009年 印发", size=12.0),  # 1 closer
        _para("二○○九年七月六日", size=12.0),  # 2 date → absorbed
        _para("附件：", size=12.0),  # 3 NOT a date → stop
        _para("2009年7月7日", size=12.0),  # 4 date, but past the stop → not absorbed
    ]
    regs = _regions(records)
    assert regs[0].members == {0, 1, 2}


def test_trailing_document_date_vetoed_from_next_cover() -> None:
    """End-to-end (mirrors test5-红头文件): the absorbed 成文日期 no longer seeds
    the following 附件 cover — the window starts at the real title, not the
    date. Injecting a no-op document_date restores the bug (date seeds)."""
    records = [
        _para("抄送：各设区市城乡规划局", size=12.0),  # 0 anchor
        _para("河北省住房和城乡建设厅办公室  2009年7月6日印发", size=12.0),  # 1 closer
        _para("二○○九年七月六日", size=12.0),  # 2 成文日期
        _para("附件：", size=12.0),  # 3
        _para("河北省城市控制性详细规划备案工作规程", size=22.0),  # 4 cover title
        _para("第一条 为规范……以句号结尾。", size=12.0),  # 5 body
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert (cands[0].start, cands[0].end) == (3, 5)  # window skips the date (2)

    # Control: date not recognized → it seeds the window (the pre-fix bug).
    cands2 = _find(records, document_date=lambda t: False)
    assert cands2 and cands2[0].start == 2


def test_imprint_forward_region_vetoes_middle_single_candidate() -> None:
    """The middle line of a 抄送…印发 region is barred from the single-line
    channel; disabling the closer detector restores it as a candidate (proving
    the forward region — not strong_body — is what vetoes it)."""
    records = [
        _para("正文一，以句号结尾。", size=12.0),  # 0
        _para("抄送：各区人民政府", size=12.0),  # 1 anchor
        _empty(),  # 2
        _para("通知标题", size=18.0, alignment="center"),  # 3 middle (would-be)
        _empty(),  # 4
        _para("某某办公室2026年6月30日印发", size=12.0),  # 5 closer (trailing)
        _para("正文二，以句号结尾。", size=12.0),  # 6
    ]
    assert _find(records) == []
    # Control: no-op closer → region is anchor-only → idx 3 leads a candidate
    # (a tail window absorbing the 版记 material — exactly the bug being fixed).
    cands = _find(records, imprint_closer=lambda t: None)
    assert len(cands) == 1 and cands[0].start == 3
