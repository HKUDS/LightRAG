"""G9 tests: TOC detection, content preservation (I1), I2/I3 checks,
canonicalization, and the 30% length gate."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading.guardrails import (
    canonicalize_paragraph_text,
    detect_toc_records,
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
# TOC two-channel detection (G9-1 / G9-2 / G9-5)
# ---------------------------------------------------------------------------


def test_structural_toc_evidence() -> None:
    records = [
        _para("目录", is_toc_field=True),
        _para("第一章 绪论……3", is_toc_link=True),
        _para("普通正文段落。"),
    ]
    warnings: dict = {}
    toc = detect_toc_records(records, warnings=warnings)
    assert toc == {0, 1}
    assert warnings["smart_toc_removed_paragraphs"] == 2
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
    toc = detect_toc_records(records, warnings={})
    assert toc == {0, 1, 2, 3}  # the isolated trailing line survives


def test_toc_similar_body_not_whitelisted() -> None:
    """G9-5: body text adjacent to a TOC but without leader shape stays."""
    records = [
        _para("第一章 绪论............3"),
        _para("第二章 方法............12"),
        _para("第三章 实验............25"),
        _para("以下正文引用了第一章 绪论的内容。"),
    ]
    toc = detect_toc_records(records, warnings={})
    assert 3 not in toc


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
