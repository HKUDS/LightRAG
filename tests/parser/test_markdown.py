"""Unit tests for the shared markdown heading renderer."""

from __future__ import annotations

import pytest

from lightrag.parser._markdown import (
    render_heading_line,
    strip_heading_markdown_prefix,
)


@pytest.mark.offline
@pytest.mark.parametrize(
    ("level", "text", "expected"),
    [
        (1, "Intro", "# Intro"),
        (2, "Sub", "## Sub"),
        (6, "Deep", "###### Deep"),
        # level >= 7 is clamped to six "#".
        (7, "Deeper", "###### Deeper"),
        (99, "Deepest", "###### Deepest"),
        # level < 1 falls back to a single "#".
        (0, "Zero", "# Zero"),
        (-3, "Neg", "# Neg"),
    ],
)
def test_render_heading_line_prefix_and_cap(level, text, expected) -> None:
    assert render_heading_line(level, text) == expected


@pytest.mark.offline
@pytest.mark.parametrize(
    "already",
    ["# Foo", "## Bar", "###### Six"],
)
def test_render_heading_line_keeps_existing_markdown(already) -> None:
    # Already a markdown heading (1-6 "#" + space) → returned unchanged,
    # regardless of the requested level (no double-prefixing).
    assert render_heading_line(3, already) == already


@pytest.mark.offline
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 7 "#" is NOT a valid markdown heading → still gets prefixed.
        ("####### Seven", "# ####### Seven"),
        # "#" with no following space → not a heading marker → prefixed.
        ("#NoSpace", "# #NoSpace"),
    ],
)
def test_render_heading_line_non_heading_hash_is_prefixed(text, expected) -> None:
    assert render_heading_line(1, text) == expected


@pytest.mark.offline
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("# Foo", "Foo"),
        ("## Bar", "Bar"),
        ("###### Six", "Six"),
        # The whole "#" run plus all following spaces are stripped — no
        # leading space leaks into the cleaned metadata.
        ("#  Extra space", "Extra space"),
        ("#NoSpace", "#NoSpace"),
        ("####### Seven", "####### Seven"),
    ],
)
def test_strip_heading_markdown_prefix(text, expected) -> None:
    assert strip_heading_markdown_prefix(text) == expected
