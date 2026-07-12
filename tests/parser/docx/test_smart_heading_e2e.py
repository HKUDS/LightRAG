"""G11/G12 end-to-end tests: real .docx fixtures through the full smart path.

Fixtures are committed binaries (decoupled from python-docx version drift);
regenerate deliberately with:

    python tests/parser/docx/test_smart_heading_e2e.py --regen

The LLM is a deterministic mock; spaCy judgments are the real ones, so these
tests skip when the pinned models are absent (see test_smart_heading_guards).
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent / "golden" / "smart_heading"


def _models_available() -> bool:
    if importlib.util.find_spec("spacy") is None:
        return False
    import spacy.util

    return spacy.util.is_package("zh_core_web_sm") and spacy.util.is_package(
        "en_core_web_sm"
    )


pytestmark = [
    pytest.mark.offline,
    pytest.mark.skipif(
        not _models_available(),
        reason="pinned spaCy models not installed (lightrag-download-cache --spacy)",
    ),
]


# ---------------------------------------------------------------------------
# fixture builders (used only by --regen)
# ---------------------------------------------------------------------------


def _p(
    doc,
    text: str,
    *,
    size: float = 12.0,
    bold: bool = False,
    center: bool = False,
    outline: int | None = None,
    page_break: bool = False,
):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    para = doc.add_paragraph()
    run = para.add_run(text)
    if page_break:
        # explicit page break at the start of the run (before the text)
        br = OxmlElement("w:br")
        br.set(qn("w:type"), "page")
        run._r.insert(0, br)
    run.font.size = Pt(size)
    run.bold = bold
    p_pr = para._p.get_or_add_pPr()
    if center:
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "center")
        p_pr.append(jc)
    if outline is not None:
        el = OxmlElement("w:outlineLvl")
        el.set(qn("w:val"), str(outline))
        p_pr.append(el)
    return para


def _body_filler(doc, n: int, *, prefix: str = "正文", size: float = 12.0) -> None:
    for i in range(n):
        _p(
            doc,
            f"{prefix}第{i}段，本段用于撑起文档的基准字号统计与篇幅门槛，内容以句号结尾。",
            size=size,
        )


def _build_redhead():
    from docx import Document

    doc = Document()
    _p(doc, "某某市人民政府文件", size=22.0, center=True)
    _p(doc, "关于加强某某管理的通知", size=16.0, center=True)
    _p(doc, "某政发〔2026〕5号", size=12.0, center=True)
    doc.add_paragraph("")
    _p(doc, "为了加强管理工作，现将有关事项通知如下。", size=12.0)
    _p(doc, "一、总体要求", size=14.0, bold=True)
    _body_filler(doc, 6)
    _p(doc, "（一）提高认识", size=12.0, bold=True)
    _body_filler(doc, 5)
    _p(doc, "（二）加强领导", size=12.0, bold=True)
    _body_filler(doc, 5)
    _p(doc, "二、工作重点", size=14.0, bold=True)
    _body_filler(doc, 6)
    _p(doc, "（一）突出重点任务", size=12.0, bold=True)
    _body_filler(doc, 5)
    return doc


def _build_regulation():
    from docx import Document

    doc = Document()
    _p(doc, "某某管理条例", size=20.0, center=True)
    _p(doc, "（2026年修订）", size=12.0, center=True)
    _p(doc, "本条例经相关会议审议通过，自发布之日起施行。", size=12.0)
    _p(doc, "第一章 总则", size=14.0, bold=True)
    _p(doc, "第一条", size=12.0, bold=True)
    _body_filler(doc, 5, prefix="总则条文")
    _p(doc, "第二条", size=12.0, bold=True)
    _body_filler(doc, 5, prefix="适用范围条文")
    _p(doc, "第二章 管理规范", size=14.0, bold=True)
    _p(doc, "第三条", size=12.0, bold=True)
    _body_filler(doc, 5, prefix="管理规范条文")
    _p(doc, "第四条", size=12.0, bold=True)
    _body_filler(doc, 5, prefix="监督检查条文")
    return doc


def _build_outline_intact():
    from docx import Document

    doc = Document()
    _p(doc, "系统设计说明书", size=12.0, outline=0)
    _body_filler(doc, 5, prefix="概述")
    _p(doc, "总体结构", size=12.0, outline=1)
    _body_filler(doc, 5, prefix="结构")
    _p(doc, "接口设计", size=12.0, outline=1)
    _body_filler(doc, 5, prefix="接口")
    _p(doc, "数据结构", size=12.0, outline=2)
    _body_filler(doc, 5, prefix="数据")
    return doc


def _build_question_bank():
    from docx import Document

    doc = Document()
    for i in range(60):
        _p(doc, f"{i + 1}. 下面关于某某概念的说法正确的是", size=12.0)
        _p(doc, "A. 选项甲的描述  B. 选项乙的描述", size=10.5)
    return doc


def _build_spliced():
    from docx import Document

    doc = Document()
    _p(doc, "数字化转型研究综述", size=18.0, center=True)
    _body_filler(doc, 8, prefix="第一篇正文")
    _p(doc, "供应链韧性分析报告", size=18.0, center=True, page_break=True)
    _body_filler(doc, 8, prefix="第二篇正文")
    return doc


def _build_oversize_outline():
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    # with soft break: first line stays a heading
    para = doc.add_paragraph()
    head = para.add_run("含软回车的超长大纲标题首行")
    head.font.size = Pt(14)
    head.add_break(WD_BREAK.LINE)
    tail = para.add_run("余部内容" * 60)
    tail.font.size = Pt(12)
    p_pr = para._p.get_or_add_pPr()
    el = OxmlElement("w:outlineLvl")
    el.set(qn("w:val"), "0")
    p_pr.append(el)
    _body_filler(doc, 4)
    # without soft break: whole paragraph demotes
    _p(doc, "无软回车的超长大纲标题" + "延长内容" * 60, size=12.0, outline=0)
    _body_filler(doc, 4)
    return doc


SCENARIOS = {
    "redhead": _build_redhead,
    "regulation": _build_regulation,
    "outline_intact": _build_outline_intact,
    "question_bank": _build_question_bank,
    "spliced": _build_spliced,
    "oversize_outline": _build_oversize_outline,
}


def _regen() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    for name, builder in sorted(SCENARIOS.items()):
        buf = io.BytesIO()
        builder().save(buf)
        (FIXTURE_ROOT / f"{name}.docx").write_bytes(buf.getvalue())
        print(f"regenerated {name}")


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class _Runtime:
    def __init__(self, llm):
        self.engine_params = {"smart_heading": True}
        self.llm_invoke = llm
        self.cancel_event = None


def _make_llm(title_responses: dict[str, dict], counter: list | None = None):
    """Deterministic judge: keyed on a needle found in the prompt; unmatched
    windows answer 'not a title block, everything is body'."""

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        if counter is not None:
            counter.append(prompt)
        for needle, resp in title_responses.items():
            if needle in prompt:
                return json.dumps(resp, ensure_ascii=False)
        ids = [int(m) for m in re.findall(r"^\[(\d+)\]", prompt, re.M)]
        return json.dumps(
            {"is_title_block": False, "headings": [], "body": ids},
            ensure_ascii=False,
        )

    return _llm


def _extract(
    name: str,
    llm,
    monkeypatch,
    *,
    min_tokens: int = 50,
    subdoc_min_tokens: int | None = None,
):
    from lightrag.parser.docx.parse_document import extract_docx_blocks

    # The whole-document and per-sub-document CB4 gates read separate env vars;
    # these fixtures are tiny, so force both low (sub defaults to the whole-doc
    # value) or the sub-document gate falls the fixture back to outline-only.
    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", str(min_tokens))
    monkeypatch.setenv(
        "DOCX_SMART_SUBDOC_MIN_TOKENS",
        str(min_tokens if subdoc_min_tokens is None else subdoc_min_tokens),
    )
    warnings: dict = {}
    metadata: dict = {}
    blocks = extract_docx_blocks(
        str(FIXTURE_ROOT / f"{name}.docx"),
        parse_warnings=warnings,
        parse_metadata=metadata,
        smart_heading_runtime=_Runtime(llm),
    )
    return blocks, warnings, metadata


def _summary(blocks) -> list[tuple]:
    return [
        (b["heading"], b["level"], bool(b.get("is_title_block", False))) for b in blocks
    ]


def _baseline(name: str):
    from lightrag.parser.docx.parse_document import extract_docx_blocks

    warnings: dict = {}
    metadata: dict = {}
    blocks = extract_docx_blocks(
        str(FIXTURE_ROOT / f"{name}.docx"),
        parse_warnings=warnings,
        parse_metadata=metadata,
    )
    return blocks, warnings, metadata


# ---------------------------------------------------------------------------
# G11 scenarios
# ---------------------------------------------------------------------------

_REDHEAD_TITLE = {
    # Red-header: the masthead 某某市人民政府文件 is the largest line but names
    # the issuing agency — it belongs in "publisher"; the real (smaller) title
    # line is the main title (the masthead clause steers the judge here).
    "关于加强某某管理的通知": {
        "is_title_block": True,
        "main_title": "关于加强某某管理的通知",
        "doc_number": "某政发〔2026〕5号",
        "publisher": "某某市人民政府文件",
    }
}


def test_redhead_document_structure(monkeypatch) -> None:
    """G11-1: title block + 一、/（一） hierarchy.

    Red-header masthead → publisher; the block heading is the plain main title
    (doc-number / publisher ride the composed heading, not the block heading).
    """
    blocks, warnings, metadata = _extract(
        "redhead", _make_llm(_REDHEAD_TITLE), monkeypatch
    )
    summary = _summary(blocks)
    assert summary[0] == ("关于加强某某管理的通知", 0, True)
    by_heading = {h: (lv, tb) for h, lv, tb in summary}
    assert by_heading["一、总体要求"] == (1, False)
    assert by_heading["二、工作重点"] == (1, False)
    assert by_heading["（一）提高认识"] == (2, False)
    assert by_heading["（二）加强领导"] == (2, False)
    # parent chains: （一） under 一、 under the main title
    sub = next(b for b in blocks if b["heading"] == "（一）提高认识")
    assert sub["parent_headings"] == ["关于加强某某管理的通知", "一、总体要求"]
    assert metadata["first_heading"] == "关于加强某某管理的通知"
    assert metadata["doc_title"] == "关于加强某某管理的通知"


def test_regulation_chapters_and_clauses(monkeypatch) -> None:
    """G11-3: 第X章 level 1, bare 第X条 (empty title allowed) level 2.

    The +1pt multi-window tier (A7) makes the 14pt chapter lines open LLM
    windows too; a faithful judge classifies them as headings — the blunt
    "everything is body" default would now revoke them for real (A10).
    """
    responses = {
        "某某管理条例": {
            "is_title_block": True,
            "main_title": "某某管理条例",
            "sub_title": "（2026年修订）",
        },
        "第一章 总则": {"is_title_block": False, "headings": [0, 1], "body": []},
        "第二章 管理规范": {"is_title_block": False, "headings": [0, 1], "body": []},
    }
    blocks, warnings, metadata = _extract(
        "regulation", _make_llm(responses), monkeypatch
    )
    by_heading = {b["heading"]: b["level"] for b in blocks}
    assert by_heading["第二章 管理规范"] == 1
    for clause in ("第一条", "第二条", "第三条", "第四条"):
        assert by_heading[clause] == 2, by_heading
    # The sub-title merges into the level-0 doc title with a double-space
    # separator, and fans out consistently to the meta doc_title and every
    # descendant's parent_headings root.
    merged = "某某管理条例  （2026年修订）"
    title = next(b for b in blocks if b.get("is_title_block"))
    assert (title["heading"], title["level"]) == (merged, 0)
    assert metadata["doc_title"] == merged
    chapter = next(b for b in blocks if b["heading"] == "第一章 总则")
    assert chapter["level"] == 1
    assert chapter["parent_headings"] == [merged]


def test_outline_intact_structure_equivalent(monkeypatch) -> None:
    """G11-6: a well-outlined doc keeps its baseline structure under smart."""
    base_blocks, _bw, _bm = _baseline("outline_intact")
    smart_blocks, warnings, _m = _extract("outline_intact", _make_llm({}), monkeypatch)
    base = [(b["heading"], b["level"], b["content"]) for b in base_blocks]
    smart = [(b["heading"], b["level"], b["content"]) for b in smart_blocks]
    assert smart == base
    assert "smart_fallback_baseline" not in warnings


def test_question_bank_cb1_yields_no_phantom_headings(monkeypatch) -> None:
    """G11-4: the CB1 breaker keeps a question bank heading-free.

    The invariant is "CB1 engages and no phantom heading survives", NOT the
    specific mechanism. Whether CB1 trips (falls back to outline-only) or the
    re-estimation converges to zero candidates depends on FS_base, which is a
    near-tie here (12pt stems vs 10.5pt option lines); the §2.2.2 whitespace-
    excluding weight fix tipped it to 12pt, so the re-gate now converges to 0
    candidates instead of tripping. Both keep the bank heading-free — assert
    the engagement + outcome, not the branch."""
    blocks, warnings, metadata = _extract("question_bank", _make_llm({}), monkeypatch)
    assert warnings.get("smart_cb1_reestimated") == 1
    # No phantom heading blocks: everything stays one preface block.
    assert all(not b.get("is_title_block") for b in blocks)
    assert {b["heading"] for b in blocks} == {"Preface/Uncategorized"}
    # Accepted smart output with no title block: doc_title is explicitly empty.
    assert metadata["doc_title"] == ""


def test_spliced_articles_only_opening_line_is_title_block(monkeypatch) -> None:
    """A page break in a spliced document does not create another level-0
    root; the later large line remains an ordinary structural heading."""
    responses = {
        "数字化转型研究综述": {
            "is_title_block": True,
            "main_title": "数字化转型研究综述",
        },
        "供应链韧性分析报告": {
            "is_title_block": True,
            "main_title": "供应链韧性分析报告",
        },
    }
    blocks, warnings, metadata = _extract("spliced", _make_llm(responses), monkeypatch)
    titles = [b for b in blocks if b.get("is_title_block")]
    assert [t["heading"] for t in titles] == ["数字化转型研究综述"]
    assert titles[0]["level"] == 0
    second = next(b for b in blocks if b["heading"] == "供应链韧性分析报告")
    assert not second.get("is_title_block")
    assert second["level"] == 1
    assert metadata["first_heading"] == "数字化转型研究综述"
    assert metadata["doc_title"] == "数字化转型研究综述"


# ---------------------------------------------------------------------------
# G12 environment / gates
# ---------------------------------------------------------------------------


def test_short_document_skips_smart_with_zero_llm_calls(monkeypatch) -> None:
    """G12-2: below the whole-doc token gate smart never runs."""
    calls: list = []
    blocks, warnings, metadata = _extract(
        "redhead",
        _make_llm(_REDHEAD_TITLE, counter=calls),
        monkeypatch,
        min_tokens=100000,
    )
    assert warnings.get("smart_skipped_short_document") == 1
    assert calls == []  # zero LLM calls
    # CB4 skip ships baseline output — baseline doc_title semantics with it.
    assert "doc_title" not in metadata
    base_blocks, _w, _m = _baseline("redhead")
    assert _summary(blocks) == _summary(base_blocks)


def test_audit_artifact_deterministic_across_runs(monkeypatch) -> None:
    """G12-3: the audit payload is byte-identical across repeated parses."""
    _b1, _w1, meta1 = _extract("redhead", _make_llm(_REDHEAD_TITLE), monkeypatch)
    _b2, _w2, meta2 = _extract("redhead", _make_llm(_REDHEAD_TITLE), monkeypatch)
    dump1 = json.dumps(meta1["smart_audit"], ensure_ascii=False, sort_keys=True)
    dump2 = json.dumps(meta2["smart_audit"], ensure_ascii=False, sort_keys=True)
    assert dump1 == dump2


def test_oversize_outline_paragraphs_never_crash(monkeypatch) -> None:
    """G12-4: >200-char outline paragraphs — soft-break keeps the first line
    as a heading; no soft break demotes to body; no DocxContentError."""
    blocks, warnings, _ = _extract("oversize_outline", _make_llm({}), monkeypatch)
    headings = {b["heading"] for b in blocks}
    assert "含软回车的超长大纲标题首行" in headings
    assert not any("无软回车" in h for h in headings)
    joined = "\n".join(b["content"] for b in blocks)
    assert "无软回车的超长大纲标题" in joined  # preserved as body (I1)
    assert "余部内容" in joined


def test_content_preservation_end_to_end(monkeypatch) -> None:
    """I1 sanity on a real fixture: nothing from the baseline body is lost."""
    base_blocks, _w, _m = _baseline("redhead")
    smart_blocks, warnings, _m2 = _extract(
        "redhead", _make_llm(_REDHEAD_TITLE), monkeypatch
    )
    from lightrag.parser.docx.smart_heading.guardrails import (
        canonicalize_paragraph_text,
    )

    def _canon_all(blocks) -> str:
        return "".join(
            canonicalize_paragraph_text(line)
            for b in blocks
            for line in b["content"].split("\n")
        )

    base_text = _canon_all(base_blocks)
    smart_text = _canon_all(smart_blocks)
    # every baseline character sequence survives (no TOC in this fixture)
    assert len(smart_text) >= len(base_text) * 0.99
    assert "smart_fallback_baseline" not in warnings


if __name__ == "__main__":
    import sys

    if "--regen" in sys.argv:
        _regen()
    else:
        print(__doc__)


def test_softbreak_heading_lands_single_line(monkeypatch, tmp_path) -> None:
    """A4 (§2.2.7): a heading paragraph that keeps its soft-break lines is
    ONE title — pass3 renders it as a single CJK-joined line. Multi-line
    headings could never match their I1 source paragraph and would fall the
    whole document back to baseline."""
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    from lightrag.parser.docx.parse_document import extract_docx_blocks

    doc = Document()
    para = doc.add_paragraph()
    r1 = para.add_run("年度工作总结")
    r1.add_break(WD_BREAK.LINE)
    r2 = para.add_run("与下年度展望")
    for r in (r1, r2):
        r.font.size = Pt(16)
    lvl = OxmlElement("w:outlineLvl")
    lvl.set(qn("w:val"), "0")
    para._p.get_or_add_pPr().append(lvl)
    _body_filler(doc, 6)
    path = tmp_path / "softbreak.docx"
    doc.save(str(path))

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    warnings: dict = {}
    blocks = extract_docx_blocks(
        str(path),
        parse_warnings=warnings,
        parse_metadata={},
        smart_heading_runtime=_Runtime(_make_llm({})),
    )
    assert "smart_fallback_baseline" not in warnings  # I1 held
    headings = [b["heading"] for b in blocks]
    assert "年度工作总结与下年度展望" in headings
    assert all("\n" not in h for h in headings)
    joined = "\n".join(b["content"] for b in blocks)
    assert "# 年度工作总结与下年度展望" in joined


def test_softbreak_title_block_lands_single_line(monkeypatch, tmp_path) -> None:
    """A soft-break COVER title (no outline level — the LLM title-block
    channel, not the plain-heading one) echoed by the LLM with its ``\\n``
    intact must land single-line in the block heading, the meta doc_title
    and every descendant's parent_headings."""
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    from lightrag.parser.docx.parse_document import extract_docx_blocks

    doc = Document()
    para = doc.add_paragraph()
    r1 = para.add_run("年度述职")
    r1.add_break(WD_BREAK.LINE)
    r2 = para.add_run("报告")
    for r in (r1, r2):
        r.font.size = Pt(22)
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    para._p.get_or_add_pPr().append(jc)
    # A strong-body line right after the title pins the single-paragraph
    # title-block channel (same trick as the mixed G11-7 fixture).
    _p(doc, "本篇为年度述职报告正文的开篇说明，请结合材料审阅。", size=12.0)
    _p(doc, "一、工作回顾", size=14.0, bold=True)
    _body_filler(doc, 6, prefix="工作回顾正文")
    _p(doc, "二、来年计划", size=14.0, bold=True)
    _body_filler(doc, 6, prefix="来年计划正文")
    path = tmp_path / "softbreak_title.docx"
    doc.save(str(path))

    responses = {"年度述职": {"is_title_block": True, "main_title": "年度述职\n报告"}}
    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "50")
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "50")
    warnings: dict = {}
    metadata: dict = {}
    blocks = extract_docx_blocks(
        str(path),
        parse_warnings=warnings,
        parse_metadata=metadata,
        smart_heading_runtime=_Runtime(_make_llm(responses)),
    )
    assert "smart_fallback_baseline" not in warnings
    title = next(b for b in blocks if b.get("is_title_block"))
    assert title["heading"] == "年度述职报告"
    assert metadata["doc_title"] == "年度述职报告"
    sub = next(b for b in blocks if b["heading"] == "一、工作回顾")
    assert sub["parent_headings"] == ["年度述职报告"]
    for b in blocks:
        assert "\n" not in b["heading"]
        assert all("\n" not in h for h in b["parent_headings"])


def test_extreme_length_fallback_g9_4(monkeypatch, tmp_path) -> None:
    """G9-4: smart output shrinking below 30% of the baseline (a TOC-
    dominated document) falls the WHOLE document back to baseline output —
    TOC lines included — with the fallback warning."""
    from docx import Document

    from lightrag.parser.docx.parse_document import extract_docx_blocks

    doc = Document()
    for i in range(40):
        _p(doc, f"第{i + 1}章 目录条目标题的完整章节文字............{i + 3}", size=12.0)
    _body_filler(doc, 4)
    path = tmp_path / "toc_dominated.docx"
    doc.save(str(path))

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    warnings: dict = {}
    metadata: dict = {}
    blocks = extract_docx_blocks(
        str(path),
        parse_warnings=warnings,
        parse_metadata=metadata,
        smart_heading_runtime=_Runtime(_make_llm({})),
    )
    assert warnings.get("smart_fallback_baseline") == 1
    # Fallback output keeps the TOC, so the removal claim must not land.
    assert "smart_toc_removed_paragraphs" not in warnings
    # The smart-only doc_title key must not survive the fallback either.
    assert "doc_title" not in metadata
    joined = "\n".join(b["content"] for b in blocks)
    assert "第1章 目录条目标题" in joined  # baseline keeps the TOC lines

    baseline = extract_docx_blocks(str(path), parse_warnings={}, parse_metadata={})
    assert [(b["heading"], b["level"], b["content"]) for b in blocks] == [
        (b["heading"], b["level"], b["content"]) for b in baseline
    ]


def test_toc_warning_only_on_accepted_smart_output(monkeypatch, tmp_path) -> None:
    """``smart_toc_removed_paragraphs`` is a content claim: it lands exactly
    when the smart output (which drops the TOC) ships, and never on the CB4
    short-document skip, whose baseline output keeps the TOC. (The guardrail
    fallback side is pinned by test_extreme_length_fallback_g9_4.)"""
    from docx import Document

    from lightrag.parser.docx.parse_document import extract_docx_blocks

    doc = Document()
    _p(doc, "第一章 绪论............3", size=12.0)
    _p(doc, "第二章 方法............12", size=12.0)
    _p(doc, "第三章 结论............25", size=12.0)
    _body_filler(doc, 12)
    path = tmp_path / "toc_small.docx"
    doc.save(str(path))

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    warnings: dict = {}
    blocks = extract_docx_blocks(
        str(path),
        parse_warnings=warnings,
        parse_metadata={},
        smart_heading_runtime=_Runtime(_make_llm({})),
    )
    assert "smart_fallback_baseline" not in warnings
    assert warnings.get("smart_toc_removed_paragraphs") == 3
    joined = "\n".join(b["content"] for b in blocks)
    assert "第一章 绪论" not in joined  # smart output dropped the TOC lines

    # CB4 skip on the same document: baseline output keeps the TOC, so the
    # removal claim must not appear even though detection saw the TOC run.
    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "100000")
    skip_warnings: dict = {}
    skip_blocks = extract_docx_blocks(
        str(path),
        parse_warnings=skip_warnings,
        parse_metadata={},
        smart_heading_runtime=_Runtime(_make_llm({})),
    )
    assert skip_warnings.get("smart_skipped_short_document") == 1
    assert "smart_toc_removed_paragraphs" not in skip_warnings
    assert "第一章 绪论" in "\n".join(b["content"] for b in skip_blocks)


def test_mixed_document_keeps_one_title_root(monkeypatch, tmp_path) -> None:
    """A later page-broken single line stays below the document title instead
    of splitting the document into a second level-0 sub-document."""
    from docx import Document

    from lightrag.parser.docx.parse_document import extract_docx_blocks

    doc = Document()
    _p(doc, "管理工作指引手册", size=18.0, center=True)
    # A strong-body line right after each big title pins the title-block
    # window to the single-paragraph channel (otherwise the multi window
    # would swallow the following headings/questions as block members).
    _p(doc, "本篇给出管理工作的总体指引，请结合实际执行。", size=12.0)
    _p(doc, "一、总体要求", size=14.0, bold=True)
    _body_filler(doc, 6, prefix="总体要求正文")
    _p(doc, "二、工作安排", size=14.0, bold=True)
    _body_filler(doc, 6, prefix="工作安排正文")
    _p(doc, "附录题库", size=18.0, center=True, page_break=True)
    _p(doc, "以下为附录题库内容，请按要求作答。", size=12.0)
    for i in range(60):
        _p(doc, f"{i + 1}. 下面关于某某概念的说法正确的是", size=12.0)
        _p(doc, "A. 选项甲的描述  B. 选项乙的描述", size=10.5)
    path = tmp_path / "mixed.docx"
    doc.save(str(path))

    responses = {
        "管理工作指引手册": {"is_title_block": True, "main_title": "管理工作指引手册"},
        "附录题库": {"is_title_block": True, "main_title": "附录题库"},
    }
    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "50")
    # The healthy sub-document (一、总体要求 / 二、工作安排) is short; keep the
    # per-sub-document CB4 gate low too or it falls back to outline-only and
    # drops these size/bold headings.
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "50")
    warnings: dict = {}
    metadata: dict = {}
    blocks = extract_docx_blocks(
        str(path),
        parse_warnings=warnings,
        parse_metadata=metadata,
        smart_heading_runtime=_Runtime(_make_llm(responses)),
    )

    titles = [b["heading"] for b in blocks if b.get("is_title_block")]
    assert titles == ["管理工作指引手册"]
    assert metadata["first_heading"] == "管理工作指引手册"
    assert metadata["doc_title"] == "管理工作指引手册"

    by_heading = {b["heading"]: b for b in blocks}
    assert "一、总体要求" in by_heading  # the healthy sub-doc kept smart
    assert by_heading["一、总体要求"]["level"] >= 1

    # Question lines remain body, owned by the ordinary appendix heading.
    assert "1. 下面关于某某概念的说法正确的是" not in by_heading
    appendix = by_heading["附录题库"]
    assert not appendix.get("is_title_block")
    assert appendix["level"] >= 1
    assert "下面关于某某概念" in appendix["content"]

    audit = metadata["smart_audit"]
    # CB1 still protects the question-bank portion from phantom headings, but
    # the whole document now has one structural scope rooted at the real title.
    assert warnings.get("smart_cb1_reestimated", 0) >= 1
    assert len(audit["sub_documents"]) == 1
    assert audit["sub_documents"][0].get("headings") == 3


def test_subdoc_gate_follows_lowered_whole_doc_gate(monkeypatch, tmp_path) -> None:
    """The per-sub-document CB4 gate DEFAULTS to min(1000, DOCX_SMART_MIN_TOKENS):
    lowering only DOCX_SMART_MIN_TOKENS (the "run smart on short documents" knob)
    must also pull the sub-document floor down. Otherwise a short document clears
    the whole-document gate only to have its sub-documents silently fall back to
    outline-only — with the old independent 1000 default this asserts-false."""
    from docx import Document

    from lightrag.parser.docx.parse_document import extract_docx_blocks

    doc = Document()
    _p(doc, "管理工作指引手册", size=18.0, center=True)
    # Strong body pins the title block to the single-paragraph channel so the
    # window does not swallow the headings below.
    _p(doc, "本篇给出管理工作的总体指引，请结合实际执行。", size=12.0)
    _p(doc, "一、总体要求", size=14.0, bold=True)
    _body_filler(doc, 4, prefix="总体要求正文")
    _p(doc, "二、工作安排", size=14.0, bold=True)
    _body_filler(doc, 4, prefix="工作安排正文")
    path = tmp_path / "short_subdoc.docx"
    doc.save(str(path))

    # The sub-document (everything under the title block) is far shorter than the
    # 1000-token default sub-gate; lower ONLY the whole-document gate and leave
    # the sub-gate env unset so the follow-down default (min(1000, 50)=50) applies.
    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "50")
    monkeypatch.delenv("DOCX_SMART_SUBDOC_MIN_TOKENS", raising=False)
    responses = {
        "管理工作指引手册": {"is_title_block": True, "main_title": "管理工作指引手册"}
    }
    warnings: dict = {}
    blocks = extract_docx_blocks(
        str(path),
        parse_warnings=warnings,
        parse_metadata={},
        smart_heading_runtime=_Runtime(_make_llm(responses)),
    )
    by_heading = {b["heading"] for b in blocks}
    # The short sub-document kept smart leveling (not outline-only), so its
    # size/bold headings survived.
    assert "一、总体要求" in by_heading
    assert "二、工作安排" in by_heading


def test_object_only_paragraph_stays_body_at_chain_sz(monkeypatch, tmp_path) -> None:
    """The test11 offender end to end, on a freshly built docx (no committed
    fixture needed): a paragraph whose style carries only ``w:szCs=28`` over
    a basedOn parent with ``w:sz=24``, holding a single bare ``w:object`` run
    (embedded OLE image, no rPr, no text). It must stay body content (I1) —
    never a heading block — and the audit must carry no promoted placeholder
    row: its size resolves through the sz TRACK to 12pt (= FS_base), and the
    zero-visible-char gate rejects every promotion channel regardless."""
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    from lightrag.parser.docx.parse_document import extract_docx_blocks

    doc = Document()
    base = doc.styles.add_style("SzBase", WD_STYLE_TYPE.PARAGRAPH)
    base.font.size = Pt(12)  # sz=24
    caption = doc.styles.add_style("CsCaption", WD_STYLE_TYPE.PARAGRAPH)
    caption.base_style = base
    rpr = caption.element.get_or_add_rPr()
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), "28")  # szCs only — the "10 图片及图题" shape
    rpr.append(szcs)
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    caption.element.get_or_add_pPr().append(jc)

    _p(doc, "装配流程说明", size=16.0, outline=0)
    _body_filler(doc, 6)
    obj_para = doc.add_paragraph(style="CsCaption")
    obj_para.add_run()._r.append(OxmlElement("w:object"))
    _body_filler(doc, 6, prefix="后续")
    path = tmp_path / "object_only.docx"
    doc.save(str(path))

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    warnings: dict = {}
    metadata: dict = {}
    blocks = extract_docx_blocks(
        str(path),
        parse_warnings=warnings,
        parse_metadata=metadata,
        smart_heading_runtime=_Runtime(_make_llm({})),
    )
    assert "smart_fallback_baseline" not in warnings  # I1 held
    # Never a heading block…
    assert all(not b["heading"].lstrip().startswith("<drawing") for b in blocks)
    # …but the placeholder tag itself survives in body content (I1).
    joined = "\n".join(b["content"] for b in blocks)
    assert "<drawing" in joined
    # No audit row promoted the placeholder (no size_strong/base_center row).
    audit = metadata["smart_audit"]
    placeholder_rows = [
        r
        for r in audit["decisions"]
        if r["summary"].lstrip().startswith(("<drawing", "<equation"))
    ]
    assert all(r["is_heading"] is False for r in placeholder_rows)
