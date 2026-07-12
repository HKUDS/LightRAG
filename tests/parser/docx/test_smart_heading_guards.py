"""G5 defensive-judgment tests: strong-body features, homophone vetoes, P3.

Positive-path cases need the real pinned spaCy models (installed in dev via
``lightrag-download-cache --spacy --spacy-install``); they skip when the
models are absent (e.g. a bare CI). The missing-model hard-error contract
(G12-1) is tested with mocks and always runs.
"""

from __future__ import annotations

import importlib.util

import pytest

from lightrag.parser.docx.smart_heading.style_key import classify_numbering

pytestmark = pytest.mark.offline


def _models_available() -> bool:
    if importlib.util.find_spec("spacy") is None:
        return False
    import spacy.util

    return spacy.util.is_package("zh_core_web_sm") and spacy.util.is_package(
        "en_core_web_sm"
    )


requires_models = pytest.mark.skipif(
    not _models_available(),
    reason="pinned spaCy models not installed (lightrag-download-cache --spacy)",
)


# route_language is a pure function (no model load), so it always runs.
@pytest.mark.parametrize(
    "text,lang",
    [
        ("这是一段中文标题", "zh"),
        ("This is an English heading", "en"),
        ("标　题", "zh"),  # review D8: full-width space must not dilute CJK share
        ("标题\t内容", "zh"),  # tabs excluded from the denominator too
    ],
)
def test_route_language_excludes_all_whitespace(text: str, lang: str) -> None:
    from lightrag.parser.docx.smart_heading.nlp import route_language

    assert route_language(text) == lang


# ---------------------------------------------------------------------------
# strong-body features
# ---------------------------------------------------------------------------


@requires_models
@pytest.mark.parametrize(
    "text,expected_rule",
    [
        # length: 70 CJK chars ≈ 210 en-equivalent > 180
        ("这是一段相当长的正文内容" * 7, "strong_body_length"),
        ("本办法自发布之日起施行。", "strong_body_sentence_end"),
        ("已经完成了吗？", "strong_body_sentence_end"),
        ("他说：“明天见。”", "strong_body_sentence_end"),  # closing-quote step-over
        ("第一步已经完成；", "strong_body_sentence_end"),  # trailing semicolon
        ("This is done. And more follows", "strong_body_multi_sentence"),
    ],
)
def test_strong_body_detected(text: str, expected_rule: str) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    assert strong_body_reason(text) == expected_rule


@requires_models
@pytest.mark.parametrize(
    "text",
    [
        "第一章 绪论",
        "第一章：绪论",  # full-width colon is not a terminator
        "项目背景与意义",
        "Report to Mr.",  # abbreviation dot, not a sentence end
        "Implementation Overview",
    ],
)
def test_not_strong_body(text: str) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    assert strong_body_reason(text) is None


# ---------------------------------------------------------------------------
# numbering homophone vetoes (G5-2 judgment layer)
# ---------------------------------------------------------------------------


@requires_models
def test_date_paragraph_vetoed_but_plain_numbering_not() -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        numbering_homophone_reason,
    )

    dated = "2026年3月5日召开会议"
    cls_dated = classify_numbering(dated)
    assert cls_dated is not None and cls_dated.style_key == "EnNum"
    assert numbering_homophone_reason(cls_dated, dated) is not None

    report = "2026年度工作报告"
    cls_report = classify_numbering(report)
    assert cls_report is not None
    assert numbering_homophone_reason(cls_report, report) == "homophone_unit_blacklist"

    plain = "1. 概念定义"
    cls_plain = classify_numbering(plain)
    assert cls_plain is not None and cls_plain.style_key == "EnNum"
    assert numbering_homophone_reason(cls_plain, plain) is None


def test_ennum_dot_ordinal_overrides_any_ner_homophone_label(monkeypatch) -> None:
    """An EnNum dot-ordinal ("4."/"12."/"2026.") is structurally never a
    homophone number-phrase (MultiLevelNum already claimed "N.N"), so a spaCy
    homophone label — whichever one it hallucinates — must NOT revoke its
    numbering identity. The NER label is forced, so this runs without models.
    """
    from lightrag.parser.docx.smart_heading import guardrails, nlp

    dot_ordinals = ["4. 制定实施方案", "12. 标题", "2026. 年度计划", "4、制定实施方案"]
    for bogus in nlp.HOMOPHONE_ENTITY_LABELS:
        monkeypatch.setattr(nlp, "leading_entity_label", lambda _t, _b=bogus: _b)
        for text in dot_ordinals:
            cls = classify_numbering(text)
            assert cls is not None and cls.style_key == "EnNum"
            assert guardrails.numbering_homophone_reason(cls, text) is None, (
                f"{text!r} + spaCy label {bogus} should be un-vetoed"
            )

    # The carve-out is scoped to EnNum dot-ordinals: a non-dot EnNum and a
    # MultiLevelNum keep the NER veto even with the same forced label.
    monkeypatch.setattr(nlp, "leading_entity_label", lambda _t: "DATE")
    glued = classify_numbering("2026计划说明")  # EnNum, no dot
    assert glued is not None and glued.style_key == "EnNum"
    assert guardrails.numbering_homophone_reason(glued, "2026计划说明") == (
        "homophone_ner_entity"
    )
    multi = classify_numbering("1.2.3 项目说明")  # MultiLevelNum
    assert multi is not None and multi.style_key == "MultiLevelNum"
    assert guardrails.numbering_homophone_reason(multi, "1.2.3 项目说明") == (
        "homophone_ner_entity"
    )


@requires_models
def test_ennum_dot_ordinal_not_vetoed_real_spacy() -> None:
    """Real-world: strings spaCy mislabels (observed: "4. …"→DATE,
    "1. …制度"→PERCENT) must resolve to None. Robust across model versions —
    the result is None whether spaCy vetoes-then-carves or labels CARDINAL."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        numbering_homophone_reason,
    )

    for text in ["4. 制定实施方案", "1. 建立上岗人员培训制度"]:
        cls = classify_numbering(text)
        assert cls is not None and cls.style_key == "EnNum"
        assert numbering_homophone_reason(cls, text) is None


@requires_models
def test_version_shape_vetoed() -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        numbering_homophone_reason,
    )

    # A bare version number "3.14 版" (unit word at end of line) still vetoes.
    for text in ["3.14 版", "3.14版"]:
        cls = classify_numbering(text)
        assert cls is not None and cls.style_key == "MultiLevelNum"
        assert numbering_homophone_reason(cls, text) == "homophone_version_shape"

    # Fix-proof: 公文 headings whose 版 heads a real CJK word (版面/版头/版记)
    # are NOT version numbers — the CJK negative lookahead keeps them out of
    # the veto so they can be recognized as same-size numbered headings.
    for text in ["5.2 版面", "7.2 版头", "7.4 版记", "7.2.7 版头中的分隔线"]:
        cls = classify_numbering(text)
        assert cls is not None and cls.style_key == "MultiLevelNum"
        assert numbering_homophone_reason(cls, text) is None, text

    # Contract change (documented): a version-release note "3.14 版更新说明"
    # is no longer regex-vetoed (a real heading; the CJK follows 版). The
    # token channel also does not fire (token after the number is ".").
    note = "3.14 版更新说明"
    cls = classify_numbering(note)
    assert cls is not None and cls.style_key == "MultiLevelNum"
    assert numbering_homophone_reason(cls, note) is None


def test_mln_ner_veto_quantity_escape(monkeypatch) -> None:
    """A MultiLevelNum with a small leading component ("7.2.1 份号") that spaCy
    mislabels QUANTITY is a real section number, not a measure — the veto is
    lifted so its size/bold/series channels can judge it. The escape is scoped
    to QUANTITY and to a leading component <= 99: every other homophone label
    (DATE/TIME/MONEY/PERCENT) and a large leading component (a real date like
    "2026.3.5") keep the veto. The NER label is forced, so no models needed.
    """
    from lightrag.parser.docx.smart_heading import guardrails, nlp

    # QUANTITY on a small-top MLN is lifted (rl2 and rl3 alike).
    monkeypatch.setattr(nlp, "leading_entity_label", lambda _t: "QUANTITY")
    for text in ["7.2 版头", "7.2.1 份号", "7.3.2 主送机关", "99.2.1 说明"]:
        cls = classify_numbering(text)
        assert cls is not None and cls.style_key == "MultiLevelNum"
        assert guardrails.numbering_homophone_reason(cls, text) is None, text

    # top > 99 keeps the veto even under QUANTITY (a real date shape).
    for text in ["100.2.3 说明", "2026.3.5 印发说明"]:
        cls = classify_numbering(text)
        assert cls is not None and cls.top_ordinal is not None
        assert cls.top_ordinal > 99
        assert guardrails.numbering_homophone_reason(cls, text) == (
            "homophone_ner_entity"
        ), text

    # Every OTHER homophone label keeps the veto on a small-top MLN — a
    # two-digit-year date / point-time is MLN top<=99 and structurally
    # indistinguishable from a section number, so only QUANTITY is exempted.
    for label in nlp.HOMOPHONE_ENTITY_LABELS - {"QUANTITY"}:
        monkeypatch.setattr(nlp, "leading_entity_label", lambda _t, _l=label: _l)
        for text in ["7.2.1 份号", "12.31.25 项目日期", "12.30.45 会议纪要"]:
            cls = classify_numbering(text)
            assert cls is not None and cls.style_key == "MultiLevelNum"
            assert guardrails.numbering_homophone_reason(cls, text) == (
                "homophone_ner_entity"
            ), f"{text!r} + {label} should keep veto"

    # A "%" phrase never even reaches the NER veto — classify_numbering
    # rejects it at the structural layer (MLN shape claims it, "%" is not a
    # legal separator, no title after → body).
    assert classify_numbering("7.2.1% 增长率") is None


@requires_models
def test_mln_section_number_quantity_not_vetoed_real_spacy() -> None:
    """Real-world: the 公文 headings whose double-space form spaCy tags
    QUANTITY ("7.2.1  份号", "7.3.2  主送机关") must resolve to None."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        numbering_homophone_reason,
    )

    for text in ["7.2.1  份号", "7.3.2  主送机关"]:
        cls = classify_numbering(text)
        assert cls is not None and cls.style_key == "MultiLevelNum"
        assert numbering_homophone_reason(cls, text) is None, text


@requires_models
def test_ennum_blacklist_env_override(monkeypatch) -> None:
    """G5-3: a custom env word takes effect."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        numbering_homophone_reason,
    )

    text = "1簇光纤"
    cls = classify_numbering(text)
    assert cls is not None and cls.style_key == "EnNum"
    assert numbering_homophone_reason(cls, text) is None  # not in default list

    monkeypatch.setenv("DOCX_SMART_ENNUM_BLACKLIST", "簇")
    assert numbering_homophone_reason(cls, text) == "homophone_unit_blacklist"


def test_ennum_blacklist_matches_multichar_unit(monkeypatch) -> None:
    """Review P1: a user-configured MULTI-char unit matches via startswith,
    not only the single-char defaults (a blacklist hit returns before NER, so
    this needs no spaCy model)."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        numbering_homophone_reason,
    )

    monkeypatch.setenv("DOCX_SMART_ENNUM_BLACKLIST", "小时,公斤")
    text = "3小时后召开"
    cls = classify_numbering(text)
    assert cls is not None and cls.style_key == "EnNum"
    assert numbering_homophone_reason(cls, text) == "homophone_unit_blacklist"


# ---------------------------------------------------------------------------
# P3 caption prefixes (no NLP needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,vetoed",
    [
        ("图1 系统架构", True),
        ("表 2-1 实验结果", True),
        ("Figure 3 shows the flow", True),
        ("figure 3 shows the flow", True),  # review P3: lowercase caption vetoed
        ("table 2-1 results", True),
        ("Fig. 4 detailed view", True),
        ("公式3 能量守恒", True),
        ("图书管理系统设计", False),  # word prefix without a numbering shape
        ("表达能力评估", False),
        ("第一章 绪论", False),
    ],
)
def test_caption_prefix(text: str, vetoed: bool) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import caption_prefix_reason

    assert (caption_prefix_reason(text) is not None) is vetoed


# ---------------------------------------------------------------------------
# 公文版记 (imprint) markers (no NLP needed)
#
# Q1 revision (product decision): the reliable ANCHOR set is 抄送 + 主题词 (both
# are formal GB/T 版记 fields in the "前缀：" shape). 主题词 being a DEFAULT
# anchor is deliberate, not accidental; the closer (印发-family, incl. 印发机关)
# is region-scoped and never an anchor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "抄送：市委各部门。",
        "抄送:各区人民政府",  # half-width colon
        "抄　送：省政府办公厅",  # justified label — ideographic space inside
        "　　抄送：市政府各委办局",  # leading indent
        "主题词：经济 管理 通知",  # 主题词 is a default anchor too (Q1 revision)
        "主题词:城市规划",  # half-width colon
    ],
)
def test_imprint_marker_detected(text: str) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import imprint_marker_reason

    assert imprint_marker_reason(text) == "imprint_marker"


@pytest.mark.parametrize(
    "text",
    [
        "抄送单位管理规定",  # no colon
        "主送：各处室",  # 主送 dropped by design (uncommon)
        "主题词经济管理",  # no colon
        "印发机关 某某厅",  # 印发机关 is a CLOSER now, never an anchor
        "印发机关：某某厅",
        "请及时抄送：相关单位",  # prefix not at line start
        "一、抄送：相关单位",  # numbering-led line is not an imprint opener
        "",
    ],
)
def test_imprint_marker_not_hit(text: str) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import imprint_marker_reason

    assert imprint_marker_reason(text) is None


def test_imprint_prefixes_env_override(monkeypatch) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import imprint_marker_reason

    monkeypatch.setenv("DOCX_SMART_IMPRINT_COLON_PREFIXES", "传阅")
    assert imprint_marker_reason("传阅：全体职工") == "imprint_marker"
    assert imprint_marker_reason("抄送：各区人民政府") is None  # default replaced


def test_strong_body_rule0_imprint() -> None:
    """The imprint ANCHOR is strong-body rule 0: it returns before the
    sentence-end (P4) check — the 。-terminated 抄送 line reports imprint, not
    sentence_end — and before any spaCy call, so no models are needed. (The
    印发-family CLOSER is region-scoped and deliberately absent from
    strong_body; see test_imprint_closer_absent_from_strong_body.)"""
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    assert strong_body_reason("抄送：市委各部门。") == "imprint_marker"
    assert strong_body_reason("主题词：经济 管理。") == "imprint_marker"


@pytest.mark.parametrize(
    "text",
    [
        "印发：某某集团公司",  # prefix + colon
        "印发 某某集团公司",  # prefix + space
        "印发　某某厅",  # prefix + ideographic space
        "印发机关 某某市人民政府办公厅",  # 印发机关 closer + space
        "印发机关：某某厅",  # 印发机关 closer + colon
        "印发机关\n某某办公厅",  # soft line break counts as whitespace
        "某某办公室    2026年6月30日 印发",  # trailing (GB/T layout)
        "某某办公室2026年6月30日印发",  # trailing, no separators
    ],
)
def test_imprint_closer_detected(text: str) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import imprint_closer_reason

    assert imprint_closer_reason(text) == "imprint_closer"


@pytest.mark.parametrize(
    "text",
    [
        "已于近日印发。",  # trailing period → body prose, not a closer
        "印发",  # bare label, nothing before/after
        "该文件印发范围包括",  # 印发 mid-line, no prefix/trailing shape
        "抄送：各区人民政府",  # an anchor, not a closer
        "",
    ],
)
def test_imprint_closer_not_hit(text: str) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import imprint_closer_reason

    assert imprint_closer_reason(text) is None


@requires_models
def test_imprint_closer_absent_from_strong_body() -> None:
    """The closer is region-scoped only: it must NOT demote a line per-line
    via strong_body (that would defeat the "印发 after 抄送" gate).

    Needs the pinned spaCy models: a bare prefix-印发 line has no sentence
    terminator and is short, so strong_body_reason falls through to the
    multi-sentence (spaCy) check to return None — exactly the path this test
    asserts stays blind to the closer."""
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    # A bare prefix-印发 line, no sentence terminator, short → strong_body is
    # blind to it (only the region scanner in title_block sees it as a closer).
    assert strong_body_reason("印发 某某集团公司") is None


def test_imprint_closer_env_override(monkeypatch) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import imprint_closer_reason

    monkeypatch.setenv("DOCX_SMART_IMPRINT_CLOSER_PREFIXES", "签发")
    assert imprint_closer_reason("签发：张三") == "imprint_closer"
    assert imprint_closer_reason("印发：某某厅") is None  # default replaced

    monkeypatch.setenv("DOCX_SMART_IMPRINT_CLOSER_TRAILING", "签章")
    assert imprint_closer_reason("某某办公室 签章") == "imprint_closer"
    assert imprint_closer_reason("某某办公室 2026年 印发") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("二○○九年七月六日", True),  # CJK numerals, ○ = circle zero
        ("二〇二六年十二月三十一日", True),  # 〇 = ideographic zero
        ("2009年7月6日", True),
        ("  2026 年 12 月 31 日 ", True),  # padded / spaced
        ("2026.7.31", True),  # separator-style: dots
        ("2026/7/31", True),  # slashes
        ("2026-12-31", True),  # hyphens
        ("2026.7/31", False),  # mixed separators are not a date
        ("35.240.20", False),  # an ICS code is not a date (2-digit "year")
        ("1.0.0", False),  # a version number is not a date
        ("会议纪要 2026.7.31", False),  # CONTAINS a date, not bare
        ("第十六条 本规程自2009年7月1日起施行。", False),  # CONTAINS a date, not bare
        ("2009年", False),  # no month/day
        ("某某办公室 2009年7月6日印发", False),  # a closer, not a bare date
        ("规划 备案 规程", False),
        ("", False),
    ],
)
def test_is_document_date(text: str, expected: bool) -> None:
    """A WHOLE-line 成文日期 only; a line that merely contains a date is not."""
    from lightrag.parser.docx.smart_heading.guardrails import is_document_date

    assert is_document_date(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("- 1 -", True),  # page number
        ("***", True),  # separator
        ("——", True),  # dash rule
        ("……", True),
        ("12 / 34", True),  # bare figures
        ("", True),  # nothing to judge
        ("完", False),  # a CJK ideograph is a letter
        ("第 1 章", False),
        ("Chapter 1", False),
        ("图-1", False),  # mixed symbol + CJK
    ],
)
def test_is_symbolic_line(text: str, expected: bool) -> None:
    """Letter-free decoration lines (positive detection: no CJK, no Latin)."""
    from lightrag.parser.docx.smart_heading.guardrails import is_symbolic_line

    assert is_symbolic_line(text) is expected


# ---------------------------------------------------------------------------
# G12-1 judgment layer: missing spaCy/model hard-fails with guidance
# ---------------------------------------------------------------------------


def test_missing_spacy_model_hard_errors(monkeypatch) -> None:
    from lightrag.parser.docx.smart_heading import nlp

    monkeypatch.setattr(nlp, "_pipelines", {})
    spacy = pytest.importorskip("spacy")

    def _boom(name, *a, **k):
        raise OSError(f"[E050] Can't find model '{name}'")

    monkeypatch.setattr(spacy, "load", _boom)
    with pytest.raises(nlp.SmartHeadingNLPError, match="lightrag-download-cache"):
        nlp.sentence_count("some text")


def test_missing_spacy_package_hard_errors(monkeypatch) -> None:
    import builtins

    from lightrag.parser.docx.smart_heading import nlp

    monkeypatch.setattr(nlp, "_pipelines", {})
    real_import = builtins.__import__

    def _no_spacy(name, *args, **kwargs):
        if name == "spacy":
            raise ImportError("No module named 'spacy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_spacy)
    with pytest.raises(nlp.SmartHeadingNLPError, match="lightrag-hku\\[api\\]"):
        nlp.sentence_count("some text")
