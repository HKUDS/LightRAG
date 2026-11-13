"""Tests for the HTML tag cleanup in ``normalize_extracted_info``.

``normalize_entity_name`` output is an entity's identity in the graph, so every
spelling of a line break or paragraph tag has to be removed the same way. The
self-closing form with a space before the slash (``<br />``, ``<p />``) was
left in place while ``<br>`` and ``<br/>`` were removed, so the same entity got
two names and the tag stayed in descriptions.
"""

from __future__ import annotations

import pytest

from lightrag.utils import (
    normalize_entity_name,
    normalize_extracted_info,
    sanitize_and_normalize_extracted_text,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "tag", ["<br>", "<br >", "<br/>", "<br />", "<br  />", "<BR />", "</br>"]
)
def test_every_line_break_spelling_is_removed(tag):
    assert normalize_extracted_info(f"Acme{tag}Corp") == "AcmeCorp"


@pytest.mark.parametrize("tag", ["<p>", "<p >", "<p/>", "<p />", "<P />", "</p>"])
def test_every_paragraph_tag_spelling_is_removed(tag):
    assert normalize_extracted_info(f"{tag}Acme Corp") == "Acme Corp"


def test_line_break_spelling_does_not_change_the_entity_name():
    names = {
        normalize_entity_name(f"Acme{tag}Corp") for tag in ("<br>", "<br/>", "<br />")
    }

    assert names == {"AcmeCorp"}


def test_escaped_line_break_with_a_space_is_removed_too():
    # sanitize_text_for_encoding unescapes entities before the tag cleanup runs
    assert (
        sanitize_and_normalize_extracted_text("first line&lt;br /&gt;second line")
        == "first linesecond line"
    )


@pytest.mark.parametrize("text", ["<pre>code</pre>", "<brand>", "<path />"])
def test_tags_that_only_start_like_p_or_br_are_kept(text):
    assert normalize_extracted_info(text) == text
