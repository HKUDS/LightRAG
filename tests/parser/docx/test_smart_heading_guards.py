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


@requires_models
def test_version_shape_vetoed() -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        numbering_homophone_reason,
    )

    text = "3.14 版更新说明"
    cls = classify_numbering(text)
    assert cls is not None and cls.style_key == "MultiLevelNum"
    assert numbering_homophone_reason(cls, text) == "homophone_version_shape"


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
