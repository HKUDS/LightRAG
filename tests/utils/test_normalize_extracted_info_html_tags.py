"""Tests for line break handling in ``normalize_extracted_info``.

The delimited extraction format is one record per line, so an LLM writes a line
break inside a value as ``<br>`` / ``<p>`` in any of their spellings, and
documents bring their own line break characters. What they become depends on
the field:

- single-line values (entity names, relation endpoints, types, keywords): tags
  are removed and every run of whitespace collapses into one space, as the last
  step. A name is an entity's identity in the graph and a rebuild re-parses
  cached LLM output with the current rules, so a rule change may only merge
  names that used to differ, never split names that used to be equal (see
  "Entity name normalization changes" in docs/design/PurgeRecoveryContract.md).
- free text (descriptions, ``multiline=True``): tags and line break characters
  become a line break, so the two lines are not glued into one word.
"""

from __future__ import annotations

import pytest

from lightrag.operate import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _process_json_extraction_result,
)
from lightrag.utils import (
    normalize_entity_name,
    normalize_extracted_info,
    sanitize_and_normalize_extracted_text,
)

pytestmark = pytest.mark.offline

LINE_BREAK_TAGS = ["<br>", "<br >", "<br/>", "<br />", "<br  />", "<BR />", "</br>"]
PARAGRAPH_TAGS = ["<p>", "<p >", "<p/>", "<p />", "<P />", "</p>"]
LINE_BREAK_CHARS = ["\n", "\r\n", "\r", "\x85", " ", " "]


# --- single-line values: only merge, never split ----------------------------


@pytest.mark.parametrize("tag", LINE_BREAK_TAGS + PARAGRAPH_TAGS)
def test_every_tag_spelling_is_removed_from_a_name(tag):
    assert normalize_extracted_info(f"Acme{tag}Corp") == "AcmeCorp"


@pytest.mark.parametrize(
    "spelling",
    [
        "Acme<br>Corp",
        "Acme<br />Corp",
        "Acme<p />Corp",
        "Acme&lt;br /&gt;Corp",
        "Acme\x0bCorp",  # deleted by the sanitizer as a control character
        "Acme\x0cCorp",
    ],
)
def test_a_tag_or_control_break_does_not_split_a_name_from_its_glued_form(spelling):
    # A stored "AcmeCorp" may have been extracted from any of these spellings.
    # A rebuild re-parses them and keeps only the records whose name equals
    # the stored key, so they must all keep landing on it.
    assert normalize_entity_name(spelling) == normalize_entity_name("AcmeCorp")


@pytest.mark.parametrize("char", LINE_BREAK_CHARS + ["\t", "  ", " \n\t "])
def test_whitespace_and_line_breaks_collapse_into_one_space_in_a_name(char):
    name = normalize_entity_name(f"Acme{char}Corp")

    assert name == "Acme Corp"
    assert name.splitlines() == ["Acme Corp"]


def test_non_breaking_spaces_merge_with_ordinary_spaces_in_a_name():
    # The collapse runs after the non-breaking space conversion; before it,
    # the first spelling kept two spaces and split from the second.
    assert normalize_entity_name("Acme&nbsp;&nbsp;Corp") == "Acme Corp"
    assert normalize_entity_name("Acme  Corp") == "Acme Corp"
    assert normalize_entity_name("Acme  Corp") == "Acme Corp"


@pytest.mark.parametrize(
    "spelling",
    [
        "Acme&nbsp;&nbsp;Corp",
        "Acme   Corp",
        "<p> Acme\t\tCorp </p>",
        "北京 \n 大学",
        "Acme  Corp",
    ],
)
def test_normalizing_a_name_twice_changes_no_whitespace(spelling):
    once = normalize_entity_name(spelling)

    assert normalize_entity_name(once) == once


def test_a_break_between_chinese_characters_leaves_no_space_in_a_name():
    assert normalize_entity_name("北京<br />大学") == "北京大学"
    assert normalize_entity_name("北京 大学") == "北京大学"
    assert normalize_entity_name("北京\x0b大学") == "北京大学"


def test_outer_quotes_next_to_a_paragraph_tag_are_still_removed():
    assert normalize_entity_name('<p>"Acme Corp"</p>') == "Acme Corp"


def test_narrow_non_breaking_space_after_a_digit_is_kept():
    assert normalize_entity_name("10 km") == "10 km"


# --- free text ---------------------------------------------------------------


@pytest.mark.parametrize("tag", LINE_BREAK_TAGS)
def test_every_line_break_tag_spelling_becomes_a_line_break_in_free_text(tag):
    assert (
        normalize_extracted_info(f"first line{tag}second line", multiline=True)
        == "first line\nsecond line"
    )


@pytest.mark.parametrize("char", LINE_BREAK_CHARS + ["\x0b", "\x0c"])
def test_every_line_break_character_becomes_a_line_break_in_free_text(char):
    text = f"first line{char}second line"

    assert (
        sanitize_and_normalize_extracted_text(text, multiline=True)
        == "first line\nsecond line"
    )
    assert normalize_extracted_info(text, multiline=True) == "first line\nsecond line"


def test_escaped_line_break_tag_becomes_a_line_break_too():
    # sanitize_text_for_encoding unescapes entities before the tag cleanup runs
    assert (
        sanitize_and_normalize_extracted_text(
            "first line&lt;br /&gt;second line", multiline=True
        )
        == "first line\nsecond line"
    )


def test_paragraphs_are_separated_by_one_blank_line_at_most():
    assert (
        normalize_extracted_info("<p>first</p><p>second</p>", multiline=True)
        == "first\n\nsecond"
    )
    assert (
        normalize_extracted_info("first<br><br><br><br>second", multiline=True)
        == "first\n\nsecond"
    )


def test_spaces_around_a_line_break_are_dropped():
    assert (
        normalize_extracted_info("first line <br />\t second line", multiline=True)
        == "first line\nsecond line"
    )


def test_a_line_break_between_chinese_characters_stays_in_free_text():
    assert normalize_extracted_info("第一行<br>第二行", multiline=True) == (
        "第一行\n第二行"
    )
    assert (
        sanitize_and_normalize_extracted_text("第一行\x0b第二行", multiline=True)
        == "第一行\n第二行"
    )
    # horizontal spaces between Chinese characters are still removed
    assert normalize_extracted_info("中文 字符", multiline=True) == "中文字符"


# --- both modes --------------------------------------------------------------


@pytest.mark.parametrize("multiline", [False, True])
@pytest.mark.parametrize("text", ["<pre>code</pre>", "<brand>", "<path />"])
def test_tags_that_only_start_like_p_or_br_are_kept(text, multiline):
    assert normalize_extracted_info(text, multiline=multiline) == text


# --- through the extraction parsers ------------------------------------------


def test_delimited_records_apply_the_rule_per_field():
    entity = _handle_single_entity_extraction(
        ["entity", "Acme<br />Corp", "organization", "Makes widgets.<br />Since 1990."],
        "chunk-1",
        1,
        "doc.txt",
    )
    relation = _handle_single_relationship_extraction(
        [
            "relation",
            "Acme  Corp",
            "Globex",
            "supply chain",
            "Acme supplies Globex.<br />Since 2001.",
        ],
        "chunk-1",
        1,
        "doc.txt",
    )

    assert entity["entity_name"] == "AcmeCorp"
    assert entity["description"] == "Makes widgets.\nSince 1990."
    assert relation["src_id"] == "Acme Corp"
    assert relation["description"] == "Acme supplies Globex.\nSince 2001."


@pytest.mark.asyncio
async def test_json_results_put_an_entity_and_its_relation_on_one_node():
    raw = (
        '{"entities":[{"name":"Acme&nbsp;&nbsp;Corp","type":"organization",'
        '"description":"Makes widgets.<br />Since 1990."}],'
        '"relationships":[{"source":"Acme\\nCorp","target":"Globex",'
        '"keywords":"supply chain","description":"Supplies Globex.\\u2028Since 2001."}]}'
    )

    nodes, edges = await _process_json_extraction_result(
        raw, chunk_key="chunk-1", timestamp=1, file_path="doc.txt"
    )

    assert list(nodes) == ["Acme Corp"]
    assert nodes["Acme Corp"][0]["description"] == "Makes widgets.\nSince 1990."
    assert list(edges) == [("Acme Corp", "Globex")]
    assert edges[("Acme Corp", "Globex")][0]["description"] == (
        "Supplies Globex.\nSince 2001."
    )
