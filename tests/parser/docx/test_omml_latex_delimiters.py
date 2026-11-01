"""Unit tests for OMML (Word equation) delimiter -> LaTeX conversion.

Locks in that ``m:d`` always emits delimiters LaTeX accepts:
- curly braces are escaped (``\\left\\{``), because ``\\left{`` is a parse error,
- ``\\left`` and ``\\right`` stay paired, using ``.`` when Word only gives one side,
- CJK and fullwidth delimiter characters select their own glyph instead of
  silently falling back to ``(`` / ``)``.
"""

from __future__ import annotations

import unicodedata
from xml.etree import ElementTree as ET

import pytest

from lightrag.parser.docx.omml import convert_omml_to_latex
from lightrag.parser.docx.omml.ommlparser import (
    DELIMITER_MAP,
    FIXED_SIZE_DELIMITER_MAP,
)

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


def _matrix(beg: str, end: str) -> ET.Element:
    """Build an ``m:oMath`` holding a one-cell matrix inside a delimiter."""
    return ET.fromstring(
        f"<m:oMath {MATH_NS}><m:d>"
        f'<m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr>'
        "<m:e><m:m><m:mr><m:e><m:r><m:t>a</m:t></m:r></m:e></m:mr></m:m></m:e>"
        "</m:d></m:oMath>"
    )


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        ("（", "）", r"\left( x \right)"),  # U+FF08 / U+FF09
        ("［", "］", r"\left[ x \right]"),  # U+FF3B / U+FF3D
        ("｛", "｝", r"\left\{ x \right\}"),  # U+FF5B / U+FF5D
        ("｜", "｜", r"\left| x \right|"),  # U+FF5C
    ],
)
@pytest.mark.offline
def test_fullwidth_delimiters_fold_onto_their_ascii_counterparts(beg, end, expected):
    # Word stores the character the author typed, and a CJK IME types these.
    # They used to miss the map and fall back to the parenthesis pair - valid
    # LaTeX, wrong glyph. NFKC covers the group, including unenumerated forms.
    assert convert_omml_to_latex(_delimited(beg, end)) == expected


@pytest.mark.parametrize(
    ("beg", "end"),
    [
        ("\u3008", "\u3009"),  # U+3008/U+3009, what NFC text carries
        ("\u2329", "\u232a"),  # U+2329/U+232A, indistinguishable on screen
    ],
)
@pytest.mark.offline
def test_both_angle_bracket_spellings_reach_the_map(beg, end):
    # The map used to be keyed on U+2329, which canonically decomposes to
    # U+3008: any text that has been through NFC carries U+3008, so the entry
    # was unreachable and the angle brackets it served fell back to parentheses.
    assert (
        convert_omml_to_latex(_delimited(beg, end)) == r"\left\langle x \right\rangle"
    )


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        ("【", "】", r"\left[ x \right]"),  # U+3010 / U+3011
        ("〔", "〕", r"\left[ x \right]"),  # U+3014 / U+3015
        ("【", "", r"\left[ x \right."),
        ("", "〕", r"\left. x \right]"),
    ],
)
@pytest.mark.offline
def test_cjk_brackets_use_the_nearest_scalable_delimiter(beg, end, expected):
    # These have no \left-compatible command of their own. The square bracket
    # is an approximate glyph; the parenthesis they used to get was a wrong one.
    assert convert_omml_to_latex(_delimited(beg, end)) == expected


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        ("《", "》", r"\langle\!\langle x \rangle\!\rangle"),
        ("《", "", r"\langle\!\langle x"),
        ("", "》", r"x \rangle\!\rangle"),
        ("《", ")", r"\left. \langle\!\langle x \right)"),
    ],
)
@pytest.mark.offline
def test_double_angle_brackets_are_fixed_size(beg, end, expected):
    # U+300A/U+300B have no single-glyph delimiter to borrow, so they follow
    # ⟦/⟧: a fixed-size sequence that carries no \left / \right, and pairs the
    # empty delimiter only opposite a side that is scalable.
    assert convert_omml_to_latex(_delimited(beg, end)) == expected


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        ("（", "）", r"\begin{pmatrix} a \end{pmatrix}"),
        ("｜", "｜", r"\begin{vmatrix} a \end{vmatrix}"),
        ("(", ")", r"\begin{pmatrix} a \end{pmatrix}"),
        ("‖", "‖", r"\begin{Vmatrix} a \end{Vmatrix}"),
        # bmatrix is reached through the square bracket these map to.
        ("[", "]", r"\begin{bmatrix} a \end{bmatrix}"),
        ("［", "］", r"\begin{bmatrix} a \end{bmatrix}"),
        ("【", "】", r"\begin{bmatrix} a \end{bmatrix}"),
        ("〔", "〕", r"\begin{bmatrix} a \end{bmatrix}"),
    ],
)
@pytest.mark.offline
def test_matrix_flavour_follows_the_normalized_delimiter(beg, end, expected):
    # The matrix branch compares the character itself rather than looking it up,
    # so normalizing at the lookup alone would leave a fullwidth-delimited
    # matrix falling through to bmatrix.
    assert convert_omml_to_latex(_matrix(beg, end)) == expected


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        (
            "《",
            "》",
            r"\langle\!\langle \begin{matrix} a \end{matrix} \rangle\!\rangle",
        ),
        ("〈", "〉", r"\left\langle \begin{matrix} a \end{matrix} \right\rangle"),
        ("⟦", "⟧", r"[\![ \begin{matrix} a \end{matrix} ]\!]"),
        ("⌊", "⌋", r"\left\lfloor \begin{matrix} a \end{matrix} \right\rfloor"),
        ("[", "", r"\left[ \begin{matrix} a \end{matrix} \right."),
    ],
)
@pytest.mark.offline
def test_matrix_without_a_named_environment_keeps_its_own_delimiters(
    beg, end, expected
):
    # A named environment carries its own brackets, so choosing one for a
    # delimiter it does not match asserts a glyph the author did not write -
    # every pair here used to come out as bmatrix, and the one-sided case had
    # a closing bracket invented for it.
    assert convert_omml_to_latex(_matrix(beg, end)) == expected


@pytest.mark.parametrize(
    ("sep", "expected"),
    [("｜", "|"), ("，", ","), ("；", ";"), ("：", ":"), ("．", "."), ("／", "/")],
)
@pytest.mark.offline
def test_fullwidth_separator_is_normalized(sep, expected):
    # sepChr is emitted verbatim between the elements, so a fullwidth bar would
    # otherwise put a CJK codepoint into math mode.
    assert convert_omml_to_latex(_separated(sep)) == f"\\left( x{expected}y \\right)"


def _separated(sep: str) -> ET.Element:
    """Build an ``m:oMath`` holding two elements split by ``sep``."""
    return ET.fromstring(
        f"<m:oMath {MATH_NS}><m:d>"
        f'<m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/><m:sepChr m:val="{sep}"/></m:dPr>'
        "<m:e><m:r><m:t>x</m:t></m:r></m:e>"
        "<m:e><m:r><m:t>y</m:t></m:r></m:e>"
        "</m:d></m:oMath>"
    )


@pytest.mark.parametrize(
    "sep",
    [
        # LaTeX's own special characters.
        "＃",
        "＄",
        "％",
        "＆",
        "＼",
        "＾",
        "＿",
        "｛",
        "｝",
        "～",
        # Structural in math mode without being special: ＇ becomes a prime on
        # the preceding symbol, and a letter or digit becomes a variable.
        "＇",
        "｀",
        "＂",
        "Ｘ",
        "５",
    ],
)
@pytest.mark.offline
def test_separator_keeps_the_character_word_wrote_unless_the_fold_is_plain(sep):
    # Unlike a delimiter, the separator is emitted verbatim, so the fold is
    # allowed only onto a plain printing mark. The unsafe half of 94 fullwidth
    # folds cannot be enumerated - ＆ is an alignment tab, ％ comments out the
    # rest of the line, ＇ silently primes x - so the allowlist is the guard.
    assert convert_omml_to_latex(_separated(sep)) == f"\\left( x{sep}y \\right)"


@pytest.mark.offline
def test_delimiter_of_a_latex_special_character_never_reaches_the_output():
    # The guard above is only needed for the separator: a delimiter character
    # is a lookup key, and one that is not in either map falls back to the
    # parenthesis instead of being emitted.
    assert convert_omml_to_latex(_delimited("＆", "％")) == r"\left( x \right)"


@pytest.mark.offline
def test_delimiter_map_keys_are_nfkc_normalized():
    # Lookups are handed an NFKC-normalized character, so a key that is not in
    # NFKC form is unreachable - which is exactly how the U+2329 entry became
    # dead code. This guards every future entry, not just that one.
    keys = [*DELIMITER_MAP, *FIXED_SIZE_DELIMITER_MAP]
    assert [k for k in keys if unicodedata.normalize("NFKC", k) != k] == []
