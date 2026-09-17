"""Unit tests for OMML (Word equation) delimiter -> LaTeX conversion.

Locks in that ``m:d`` always emits delimiters LaTeX accepts:
- curly braces are escaped (``\\left\\{``), because ``\\left{`` is a parse error,
- ``\\left`` and ``\\right`` stay paired, using ``.`` when Word only gives one side.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from lightrag.parser.docx.omml import convert_omml_to_latex

MATH_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'


def _delimited(beg: str, end: str) -> ET.Element:
    """Build an ``m:oMath`` holding a single delimited ``x``.

    Word writes the element with an empty value for a one-sided delimiter, so
    both are always emitted here; omitting them selects the ``(`` / ``)`` default.
    """
    return ET.fromstring(
        f"<m:oMath {MATH_NS}><m:d>"
        f'<m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr>'
        "<m:e><m:r><m:t>x</m:t></m:r></m:e></m:d></m:oMath>"
    )


@pytest.mark.offline
def test_curly_brace_delimiters_are_escaped():
    # ``\left{`` is not valid LaTeX; the delimiter has to be ``\{``.
    assert convert_omml_to_latex(_delimited("{", "}")) == r"\left\{ x \right\}"


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        ("(", ")", r"\left( x \right)"),
        ("[", "]", r"\left[ x \right]"),
        ("⌊", "⌋", r"\left\lfloor x \right\rfloor"),
    ],
)
@pytest.mark.offline
def test_other_delimiters_are_unchanged(beg, end, expected):
    assert convert_omml_to_latex(_delimited(beg, end)) == expected


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        ("{", "", r"\left\{ x \right."),
        ("", "}", r"\left. x \right\}"),
        ("[", "", r"\left[ x \right."),
    ],
)
@pytest.mark.offline
def test_one_sided_delimiters_are_paired_with_the_empty_delimiter(beg, end, expected):
    # Word writes only one of begChr/endChr for e.g. a single opening brace;
    # an unmatched \left or \right is a parse error.
    assert convert_omml_to_latex(_delimited(beg, end)) == expected


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        ("⟦", "⟧", r"[\![ x ]\!]"),
        ("⟦", "", r"[\![ x"),
        ("", "⟧", r"x ]\!]"),
        ("⟦", ")", r"\left. [\![ x \right)"),
        ("(", "⟧", r"\left( x ]\!] \right."),
    ],
)
@pytest.mark.offline
def test_fixed_size_double_brackets_only_pair_scalable_sides(beg, end, expected):
    # ``⟦``/``⟧`` render as fixed-size ``[\![``/``]\!]``, not \left/\right, so
    # the empty delimiter is added only opposite a side that is scalable.
    assert convert_omml_to_latex(_delimited(beg, end)) == expected


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        # An opening character as endChr: Word writes this for a half-open
        # interval. Only ``[`` used to be handled, by a patch table.
        ("[", "[", r"\left[ x \right["),
        ("(", "(", r"\left( x \right("),
        ("{", "{", r"\left\{ x \right\{"),
        ("⌈", "⌊", r"\left\lceil x \right\lfloor"),
        # A closing character as begChr, the mirror image. Only ``]`` was
        # handled, by the other patch table.
        ("]", "[", r"\left] x \right["),
        (")", ")", r"\left) x \right)"),
        ("}", "}", r"\left\} x \right\}"),
        ("⌋", "⌉", r"\left\rfloor x \right\rceil"),
        # One-sided, with the side the old map did not anticipate.
        ("", "(", r"\left. x \right("),
        (")", "", r"\left) x \right."),
        ("", "{", r"\left. x \right\{"),
        ("}", "", r"\left\} x \right."),
    ],
)
@pytest.mark.offline
def test_delimiter_side_follows_position_not_character(beg, end, expected):
    # The character selects the glyph; the side it lands on selects \left or
    # \right. Baking the side into the map emitted \left in the end position
    # (and \right in the start position) for every entry the patch tables
    # missed, which is an unmatched or out-of-order delimiter either way.
    assert convert_omml_to_latex(_delimited(beg, end)) == expected
