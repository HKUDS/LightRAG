"""Tests for line break handling in ``normalize_extracted_info``.

The delimited extraction format is one record per line, so an LLM writes a line
break inside a value as ``<br>`` / ``<p>`` in any of their spellings. What that
tag becomes depends on the field:

- single-line values (entity names, relation endpoints, types, keywords): a
  space, and every run of whitespace collapses into one space. A name is an
  entity's identity in the graph, so every spelling of the same break has to
  produce the same name and none of them may leave a line break in it.
- free text (descriptions, ``multiline=True``): a line break, so the two lines
  are not glued into one word.
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


# --- single-line values ------------------------------------------------------


@pytest.mark.parametrize("tag", LINE_BREAK_TAGS)
def test_every_line_break_spelling_becomes_a_space_in_a_name(tag):
    assert normalize_extracted_info(f"Acme{tag}Corp") == "Acme Corp"


@pytest.mark.parametrize("tag", PARAGRAPH_TAGS)
def test_every_paragraph_tag_spelling_becomes_a_space_in_a_name(tag):
    assert normalize_extracted_info(f"Acme{tag}Corp") == "Acme Corp"


def test_every_spelling_of_a_break_gives_one_entity_name():
    spellings = [
        "Acme<br>Corp",
        "Acme<br/>Corp",
        "Acme<br />Corp",
        "Acme<p>Corp",
        "Acme\nCorp",
        "Acme\r\nCorp",
        "Acme\tCorp",
        "Acme  Corp",
        "Acme <br /> Corp",
        "<p>Acme Corp</p>",
    ]

    assert {normalize_entity_name(s) for s in spellings} == {"Acme Corp"}


def test_a_break_between_chinese_characters_leaves_no_space_in_a_name():
    assert normalize_entity_name("北京<br />大学") == "北京大学"
    assert normalize_entity_name("北京\n大学") == "北京大学"


def test_outer_quotes_next_to_a_paragraph_tag_are_still_removed():
    assert normalize_entity_name('<p>"Acme Corp"</p>') == "Acme Corp"


def test_keywords_and_types_get_a_space_by_default():
    assert sanitize_and_normalize_extracted_text("supply<br />chain") == (
        "supply chain"
    )


def test_single_line_mode_leaves_non_breaking_space_rules_in_charge():
    # U+202F after a digit is a thousands/unit separator the quote rules keep
    assert normalize_entity_name("10 km") == "10 km"


# --- free text ---------------------------------------------------------------


@pytest.mark.parametrize("tag", LINE_BREAK_TAGS)
def test_every_line_break_spelling_becomes_a_line_break_in_free_text(tag):
    assert (
        normalize_extracted_info(f"first line{tag}second line", multiline=True)
        == "first line\nsecond line"
    )


def test_escaped_line_break_becomes_a_line_break_too():
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
    assert normalize_extracted_info("first\r\nsecond", multiline=True) == (
        "first\nsecond"
    )


def test_a_line_break_between_chinese_characters_stays_in_free_text():
    assert normalize_extracted_info("第一行<br>第二行", multiline=True) == (
        "第一行\n第二行"
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
            "Acme<br />Corp",
            "Globex",
            "supply<br />chain",
            "Acme supplies Globex.<br />Since 2001.",
        ],
        "chunk-1",
        1,
        "doc.txt",
    )

    assert entity["entity_name"] == "Acme Corp"
    assert entity["description"] == "Makes widgets.\nSince 1990."
    assert relation["src_id"] == "Acme Corp"
    assert relation["keywords"] == "supply chain"
    assert relation["description"] == "Acme supplies Globex.\nSince 2001."


@pytest.mark.asyncio
async def test_json_results_apply_the_rule_per_field():
    raw = (
        '{"entities":[{"name":"Acme\\nCorp","type":"organization",'
        '"description":"Makes widgets.<br />Since 1990."}],'
        '"relationships":[{"source":"Acme<br>Corp","target":"Globex",'
        '"keywords":"supply chain","description":"Supplies Globex.\\nSince 2001."}]}'
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
