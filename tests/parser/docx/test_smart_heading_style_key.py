"""G2/G3 tests: styleKey classification and FS_base statistics."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.smart_heading.style_key import (
    ALLOW_EMPTY_TITLE,
    CN_CHAPTER,
    CN_CLAUSE,
    CN_NUM,
    CN_PARENT_NUM,
    EN_ALPHA,
    EN_CHAPTER,
    EN_CLAUSE,
    EN_DOUBLE_PAREN,
    EN_NUM,
    EN_SINGLE_PAREN,
    MULTI_LEVEL_NUM,
    ROMAN_NUM,
    STYLE_KEY_PRIORITY,
    classify_numbering,
    compute_fs_base,
    parse_cn_ordinal,
    parse_roman,
    reclassify_single_char_romans,
    unit_rank,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# G2-1: positive corpus (styleKey + expected label)
# ---------------------------------------------------------------------------

POSITIVE_CASES = [
    # CnChapter
    ("第一章 绪论", CN_CHAPTER, "第一章"),
    ("第 1 章 引言", CN_CHAPTER, "第 1 章"),
    ("第一章绪论", CN_CHAPTER, "第一章"),
    ("第一章：绪论", CN_CHAPTER, "第一章"),
    ("第一章", CN_CHAPTER, "第一章"),  # empty title allowed
    ("第十二篇 内容", CN_CHAPTER, "第十二篇"),
    ("第三节 方法", CN_CHAPTER, "第三节"),
    ("第2卷 上", CN_CHAPTER, "第2卷"),
    ("第五编 分则", CN_CHAPTER, "第五编"),
    ("第一部", CN_CHAPTER, "第一部"),
    # EnChapter
    ("Chapter 1 Introduction", EN_CHAPTER, "Chapter 1"),
    ("chapter 2: Basics", EN_CHAPTER, "chapter 2"),
    ("PART I", EN_CHAPTER, "PART I"),
    ("CHAPTER 1", EN_CHAPTER, "CHAPTER 1"),
    ("Section 3", EN_CHAPTER, "Section 3"),  # wins over EnClause
    ("Volume II Overview", EN_CHAPTER, "Volume II"),
    ("Part A", EN_CHAPTER, "Part A"),
    # MultiLevelNum
    ("1.2 节", MULTI_LEVEL_NUM, "1.2"),
    ("§ 1.1.4 节", MULTI_LEVEL_NUM, "§ 1.1.4"),
    ("§§ 2.4.1节", MULTI_LEVEL_NUM, "§§ 2.4.1"),
    ("1.1.2 标题", MULTI_LEVEL_NUM, "1.1.2"),
    ("2.3、内容", MULTI_LEVEL_NUM, "2.3"),
    ("3.4.5. Title", MULTI_LEVEL_NUM, "3.4.5"),
    ("1.2概述", MULTI_LEVEL_NUM, "1.2"),
    # CnClause
    ("第十二条", CN_CLAUSE, "第十二条"),
    ("第12条 罚则", CN_CLAUSE, "第12条"),
    ("第 12 条 罚则", CN_CLAUSE, "第 12 条"),
    ("第三款 内容", CN_CLAUSE, "第三款"),
    ("第五项", CN_CLAUSE, "第五项"),
    ("第二条规定了处罚", CN_CLAUSE, "第二条"),
    # EnClause
    ("Art. 2", EN_CLAUSE, "Art. 2"),
    ("Article 12 Scope", EN_CLAUSE, "Article 12"),
    ("Sec. 3 Rules", EN_CLAUSE, "Sec. 3"),
    ("Clause 7", EN_CLAUSE, "Clause 7"),
    ("§ 101", EN_CLAUSE, "§ 101"),
    ("¶ 12 text", EN_CLAUSE, "¶ 12"),
    # CnNum
    ("一、项目背景", CN_NUM, "一"),
    ("三 项目背景", CN_NUM, "三"),
    ("十二、内容", CN_NUM, "十二"),
    # CnParentNum
    ("（一）总则", CN_PARENT_NUM, "（一）"),
    ("(三）混搭括号", CN_PARENT_NUM, "(三）"),
    ("三）半括号", CN_PARENT_NUM, "三）"),
    ("（十二）内容", CN_PARENT_NUM, "（十二）"),
    # RomanNum
    ("II. Method", ROMAN_NUM, "II"),
    ("iii、结论", ROMAN_NUM, "iii"),
    ("Ⅲ、总则", ROMAN_NUM, "Ⅲ"),
    ("XI. Overview", ROMAN_NUM, "XI"),
    ("ⅻ、附录", ROMAN_NUM, "ⅻ"),
    ("IIX. broken but harmless", ROMAN_NUM, "IIX"),
    # EnNum
    ("1. 概念", EN_NUM, "1"),
    ("1.概念", EN_NUM, "1"),
    ("1概念", EN_NUM, "1"),
    ("12、内容", EN_NUM, "12"),
    ("3 Title", EN_NUM, "3"),
    # EnAlpha
    ("A. 概念", EN_ALPHA, "A"),
    ("a. Intro", EN_ALPHA, "a"),
    ("B、内容", EN_ALPHA, "B"),
    # EnDoubleParen
    ("(1) 内容", EN_DOUBLE_PAREN, "(1)"),
    ("（a）内容", EN_DOUBLE_PAREN, "（a）"),
    ("(A) Text", EN_DOUBLE_PAREN, "(A)"),
    ("（12）内容", EN_DOUBLE_PAREN, "（12）"),
    # EnSingleParen
    ("1) 内容", EN_SINGLE_PAREN, "1"),
    ("a) Intro", EN_SINGLE_PAREN, "a"),
    ("12）内容", EN_SINGLE_PAREN, "12"),
]


@pytest.mark.parametrize(
    "text,style_key,label", POSITIVE_CASES, ids=[c[0] for c in POSITIVE_CASES]
)
def test_positive_classification(text: str, style_key: str, label: str) -> None:
    result = classify_numbering(text)
    assert result is not None, f"expected {style_key} for {text!r}"
    assert result.style_key == style_key
    assert result.label_text == label


# ---------------------------------------------------------------------------
# G2-1: negative corpus (must classify as body / None)
# ---------------------------------------------------------------------------

NEGATIVE_CASES = [
    # keyword word-boundary defenses
    "Participants met yesterday",
    "Security is important",
    "Articulate the plan",
    "Chapters are numbered",
    "Sections of society",
    "Partition the disk",
    "Paradigm shift",
    # EnAlpha requires a dot/、 separator
    "A cat sat here",
    "I think so",
    "A股 上涨了",
    "B超 检查",
    # RomanNum: separator strictness and alphabet limits
    "XI'AN 城市",
    "VI 编号（空格分隔）",
    "CV. 简历缩写",
    "MD. 医生头衔",
    "ix regards",
    # bare numbering with mandatory-title styleKeys → body
    "1.2",
    "3.14",
    "1.2.3",
    "三、",
    "（一）",
    "1)",
    "(1)",
    "A.",
    "II.",
    # 第X + non-unit char is not a chapter/clause
    "第二天早上出发",
    "第一时间响应",
]


@pytest.mark.parametrize("text", NEGATIVE_CASES, ids=NEGATIVE_CASES)
def test_negative_classification(text: str) -> None:
    assert classify_numbering(text) is None


# ---------------------------------------------------------------------------
# G2-2 / G2-3: units, ordinals, priorities
# ---------------------------------------------------------------------------


def test_unit_extraction_and_suborder() -> None:
    chapter = classify_numbering("第一章 绪论")
    section = classify_numbering("第三节 方法")
    part = classify_numbering("第一篇 总论")
    assert (chapter.unit, section.unit, part.unit) == ("章", "节", "篇")
    assert unit_rank(CN_CHAPTER, "篇") < unit_rank(CN_CHAPTER, "章")
    assert unit_rank(CN_CHAPTER, "章") < unit_rank(CN_CHAPTER, "节")
    assert unit_rank(CN_CLAUSE, "条") < unit_rank(CN_CLAUSE, "款")
    assert unit_rank(CN_CLAUSE, "款") < unit_rank(CN_CLAUSE, "项")
    assert unit_rank(EN_CHAPTER, "volume") == unit_rank(EN_CHAPTER, "part")
    assert unit_rank(EN_CHAPTER, "part") < unit_rank(EN_CHAPTER, "chapter")
    assert unit_rank(EN_CHAPTER, "chapter") < unit_rank(EN_CHAPTER, "section")


def test_en_clause_unit_normalization() -> None:
    assert classify_numbering("Art. 2").unit == "article"
    assert classify_numbering("Article 2").unit == "article"
    assert classify_numbering("SEC. 3 Rules").unit == "section"
    assert classify_numbering("§ 101").unit == "§"
    assert classify_numbering("¶ 12 x").unit == "¶"


def test_series_key_same_unit_required() -> None:
    zh_arab = classify_numbering("第1章 引言")
    zh_cn = classify_numbering("第一章 绪论")
    zh_sec = classify_numbering("第一节 方法")
    assert zh_arab.series_key() == zh_cn.series_key()  # 第1章 ≡ 第一章
    assert zh_cn.series_key() != zh_sec.series_key()  # 章 ≠ 节


def test_ordinals() -> None:
    assert classify_numbering("第十二条").ordinal == 12
    assert classify_numbering("第 12 条").ordinal == 12
    assert classify_numbering("二十三、内容").ordinal == 23
    assert classify_numbering("（十）内容").ordinal == 10
    assert classify_numbering("XI. Overview").ordinal == 11
    assert classify_numbering("Ⅲ、总则").ordinal == 3
    assert classify_numbering("IIX. broken").ordinal is None
    assert classify_numbering("b) Intro").ordinal == 2
    assert classify_numbering("Chapter 4 x").ordinal == 4
    assert classify_numbering("PART I").ordinal == 1
    assert parse_cn_ordinal("一百二十") == 120
    assert parse_roman("XXXIX") == 39


def test_multilevel_raw_level_and_top() -> None:
    two = classify_numbering("1.2 概述")
    three = classify_numbering("§ 1.1.4 节")
    assert (two.raw_level, two.top_ordinal) == (2, 1)
    assert (three.raw_level, three.top_ordinal) == (3, 1)


def test_priority_table() -> None:
    assert STYLE_KEY_PRIORITY[CN_CHAPTER] == STYLE_KEY_PRIORITY[EN_CHAPTER] == 1
    assert STYLE_KEY_PRIORITY[MULTI_LEVEL_NUM] == 2
    assert STYLE_KEY_PRIORITY[CN_CLAUSE] == STYLE_KEY_PRIORITY[EN_CLAUSE] == 3
    assert (
        STYLE_KEY_PRIORITY[CN_NUM]
        < STYLE_KEY_PRIORITY[CN_PARENT_NUM]
        < STYLE_KEY_PRIORITY[ROMAN_NUM]
        < STYLE_KEY_PRIORITY[EN_NUM]
        < STYLE_KEY_PRIORITY[EN_ALPHA]
        < STYLE_KEY_PRIORITY[EN_DOUBLE_PAREN]
        < STYLE_KEY_PRIORITY[EN_SINGLE_PAREN]
    )
    assert ALLOW_EMPTY_TITLE == {CN_CHAPTER, EN_CHAPTER, CN_CLAUSE, EN_CLAUSE}


# ---------------------------------------------------------------------------
# G2-5: deferred single-char roman reclassification
# ---------------------------------------------------------------------------


def test_single_char_roman_promoted_with_companions() -> None:
    items = [
        classify_numbering("I. Intro"),
        classify_numbering("II. Method"),
        classify_numbering("III. Results"),
    ]
    assert items[0].style_key == EN_ALPHA  # default before the second scan
    out = reclassify_single_char_romans(items)
    assert [c.style_key for c in out] == [ROMAN_NUM, ROMAN_NUM, ROMAN_NUM]
    assert out[0].ordinal == 1


def test_single_char_roman_stays_alpha_without_companions() -> None:
    items = [
        classify_numbering("A. Alpha"),
        classify_numbering("B. Beta"),
        classify_numbering("I. Maybe roman"),
    ]
    out = reclassify_single_char_romans(items)
    assert [c.style_key for c in out] == [EN_ALPHA, EN_ALPHA, EN_ALPHA]


# ---------------------------------------------------------------------------
# G3-1 / G3-2: FS_base
# ---------------------------------------------------------------------------


def test_fs_base_char_weighted_not_paragraph_count() -> None:
    # 300 short list paragraphs at 10pt (10 chars each) vs 80 long body
    # paragraphs at 12pt (100 chars each): weight wins, not count.
    pairs = [(10.0, 10)] * 300 + [(12.0, 100)] * 80
    fs = compute_fs_base(pairs)
    assert fs.size_pt == 12.0
    assert fs.confidence_high is True


def test_fs_base_tie_prefers_larger() -> None:
    fs = compute_fs_base([(10.5, 500), (12.0, 500)])
    assert fs.size_pt == 12.0
    assert fs.dominant_ratio == 0.5
    assert fs.confidence_high is False


def test_fs_base_low_confidence_below_threshold() -> None:
    fs = compute_fs_base([(10.0, 40), (12.0, 35), (14.0, 25)])
    assert fs.size_pt == 10.0
    assert fs.confidence_high is False


def test_fs_base_empty_input() -> None:
    fs = compute_fs_base([])
    assert fs.size_pt is None and fs.confidence_high is False
