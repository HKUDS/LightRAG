"""G4 tests (judgment layer): title-block candidates and LLM verdict parsing.

Strong-body / homophone judgments are injected as deterministic stubs so
these tests run without spaCy; the real implementations are covered by
test_smart_heading_guards.py.
"""

from __future__ import annotations

import json

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading.title_block import (
    TitleBlockCandidate,
    TitleBlockLLMError,
    compose_title_heading,
    find_title_block_candidates,
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


def test_non_title_verdict_incomplete_classification_hard_fails() -> None:
    with pytest.raises(TitleBlockLLMError, match="exactly once"):
        _judge_with({"is_title_block": False, "headings": [0], "body": [1]})
    with pytest.raises(TitleBlockLLMError, match="exactly once"):
        _judge_with({"is_title_block": False, "headings": [0, 1], "body": [1, 2]})


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
    with pytest.raises(TitleBlockLLMError, match="exactly once"):
        _judge_with({"is_title_block": False, "headings": [0, 0], "body": [1, 2]})


# ---------------------------------------------------------------------------
# spec-vs-implementation audit regressions (A7 / A9)
# ---------------------------------------------------------------------------


def test_multi_window_admits_strong_tier_plus_one_pt() -> None:
    """A7 (§2.2.4): the multi-paragraph window title line sits at the
    strong-signal tier (+1pt over the FS_base initial value); only the
    single-paragraph channel keeps the stricter +2pt delta."""
    records = [
        _para("产品技术白皮书", size=13.0),  # +1pt — below the old +2pt bar
        _para("研发中心", size=12.0),
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
    ]
    cands = _find(records)
    assert [c.trigger for c in cands] == ["multi_window"]

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
