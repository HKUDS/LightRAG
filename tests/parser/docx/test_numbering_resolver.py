"""Unit tests for NumberingResolver ilvl resolution and numFmt rendering.

Covers the ilvl-resolution priority when a paragraph's numPr omits w:ilvl:
(a) explicit ilvl inherited through the style basedOn chain, (b) the
abstractNum per-level w:pStyle link, (c) default 0, plus the two direct-numPr
edge cases: (d) direct numId without ilvl still borrows the chain ilvl, and
(e) an explicit direct ilvl=0 is preserved (NOT treated as missing).

Also covers numFmt rendering: the CJK counting families, the digit-by-digit
``ideographDigital`` family (which must NOT reuse the counting converter), and
the unknown-format diagnostic. These stay at unit level deliberately — the
resolver is the single chokepoint for every label the parser emits, and an
end-to-end fixture would mean hand-crafting a numbering.xml into a .docx zip
for no extra coverage of this logic.

The resolution logic is exercised directly on hand-built dicts + <w:p>
elements — no numbering.xml crafting — so the tests target the merge/fallback
behaviour rather than the XML parsers.
"""

from __future__ import annotations

import pytest
from lxml import etree

from lightrag.parser.docx.numbering_resolver import NumberingResolver
from lightrag.parser.docx.smart_heading.style_key import classify_numbering

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# One all-decimal abstractNum ("10"): ilvl i renders "1.1...."(i+1 components).
_LEVELS = {
    i: {
        "start": 1,
        "numFmt": "decimal",
        "lvlText": ".".join(f"%{j + 1}" for j in range(i + 1)),
        "isLgl": False,
    }
    for i in range(4)
}


def _resolver() -> NumberingResolver:
    """A resolver wired to a synthetic single-abstract numbering scheme.

    Style graph (all point at numId 100 → abstract 10):
      H4    numId=100, ilvl absent  → basedOn H4alt
      H4alt numId=100, ilvl=3       → basedOn H1
      H1    numId=100, ilvl=0       → basedOn Normal
      PS    numId=100, ilvl absent  → basedOn Normal   (only pStyle-linked)
      ListP numId=100, ilvl absent  → basedOn Normal   (no ilvl anywhere)
    abstract 10 pStyle link: level 3 → style "PS".
    """
    r = NumberingResolver.__new__(NumberingResolver)
    r.abstract_nums = {"10": dict(_LEVELS)}
    r.abstract_pstyle = {"10": {"PS": 3}}
    r.num_to_abstract = {"100": "10"}
    r.counters = {}
    r.start_overrides = {}
    r.style_numpr = {
        "H4": {"numId": "100", "ilvl": None},
        "H4alt": {"numId": "100", "ilvl": 3},
        "H1": {"numId": "100", "ilvl": 0},
        "PS": {"numId": "100", "ilvl": None},
        "ListP": {"numId": "100", "ilvl": None},
    }
    r.style_based_on = {
        "H4": "H4alt",
        "H4alt": "H1",
        "H1": "Normal",
        "PS": "Normal",
        "ListP": "Normal",
    }
    r.last_numId = None
    r.last_abstract_id = None
    r.last_style_id = None
    return r


def _para(*, style: str | None = None, num_id: str | None = None, ilvl=None):
    """Build a <w:p> with optional pStyle and a direct numPr.

    ``ilvl`` is only emitted when not None, so ``num_id`` set + ``ilvl=None``
    reproduces a direct numPr that carries numId but omits w:ilvl.
    """
    inner = []
    if style is not None:
        inner.append(f'<w:pStyle w:val="{style}"/>')
    if num_id is not None:
        numpr = []
        if ilvl is not None:
            numpr.append(f'<w:ilvl w:val="{ilvl}"/>')
        numpr.append(f'<w:numId w:val="{num_id}"/>')
        inner.append(f"<w:numPr>{''.join(numpr)}</w:numPr>")
    return etree.fromstring(
        f'<w:p xmlns:w="{W}"><w:pPr>{"".join(inner)}</w:pPr>'
        f"<w:r><w:t>x</w:t></w:r></w:p>"
    )


def test_a_basedon_chain_supplies_missing_ilvl() -> None:
    # H4's numPr omits ilvl; the explicit ilvl=3 is inherited from basedOn H4alt
    # (H4 is not in the pStyle map, so this isolates the basedOn path).
    assert _resolver().get_label(_para(style="H4")) == "1.1.1.1"


def test_b_pstyle_link_supplies_missing_ilvl() -> None:
    # PS has no explicit ilvl anywhere in its chain; the abstract's pStyle link
    # (level 3 → PS) supplies it.
    assert _resolver().get_label(_para(style="PS")) == "1.1.1.1"


def test_c_default_ilvl_zero_when_no_signal() -> None:
    # ListP: no explicit ilvl in the chain, no pStyle link → default 0.
    assert _resolver().get_label(_para(style="ListP")) == "1"


def test_d_direct_numid_without_ilvl_borrows_chain_ilvl() -> None:
    # Direct numPr carries numId but omits ilvl: the direct numId is kept and
    # the ilvl is borrowed from the style chain (H4alt → ilvl 3). Guards against
    # only calling the style fallback when num_id is None.
    assert _resolver().get_label(_para(style="H4alt", num_id="100")) == "1.1.1.1"


def test_e_explicit_direct_ilvl_zero_is_preserved() -> None:
    # Explicit direct ilvl=0 must NOT be treated as "missing" (the `x or None`
    # truthy trap): it renders level 0 ("1"), NOT the chain's level 3.
    assert _resolver().get_label(_para(style="H4alt", num_id="100", ilvl=0)) == "1"


def test_get_numbering_from_style_merges_numid_and_explicit_ilvl() -> None:
    # numId from the nearest ancestor defining it, ilvl from the nearest with an
    # EXPLICIT ilvl — inherited independently down the basedOn chain.
    r = _resolver()
    assert r._get_numbering_from_style("H4") == {"numId": "100", "ilvl": 3}
    # A chain with no explicit ilvl anywhere returns ilvl=None (not 0).
    assert r._get_numbering_from_style("ListP") == {"numId": "100", "ilvl": None}


def test_ilvl_outside_ooxml_domain_is_rejected_not_looped() -> None:
    """w:ilvl is defined for 0-8 (ECMA-376 ST_DecimalNumber). A malicious
    document can define a level at the same out-of-range ilvl in both
    numbering.xml and the paragraph's direct numPr, so the "ilvl not in
    levels" check alone would not catch it -- it would flow straight into
    range(ilvl) and blow up into a CPU-bound loop of that many iterations."""
    r = NumberingResolver.__new__(NumberingResolver)
    r.abstract_nums = {
        "10": {
            1000: {"start": 1, "numFmt": "decimal", "lvlText": "%1.", "isLgl": False}
        }
    }
    r.abstract_pstyle = {}
    r.num_to_abstract = {"100": "10"}
    r.counters = {}
    r.start_overrides = {}
    r.style_numpr = {}
    r.style_based_on = {}
    r.last_numId = None
    r.last_abstract_id = None
    r.last_style_id = None
    r._warnings = None

    assert r.get_label(_para(num_id="100", ilvl=1000)) == ""
    assert r.last_numId is None
    assert r.last_abstract_id is None


@pytest.mark.parametrize("ilvl", [-1, 9, 999999])
def test_out_of_range_ilvl_values_are_rejected(ilvl) -> None:
    r = _resolver()
    r.abstract_nums["10"][ilvl] = {
        "start": 1,
        "numFmt": "decimal",
        "lvlText": "%1.",
        "isLgl": False,
    }
    assert r.get_label(_para(num_id="100", ilvl=ilvl)) == ""


def test_boundary_ilvl_eight_still_renders() -> None:
    r = _resolver()
    r.abstract_nums["10"][8] = {
        "start": 1,
        "numFmt": "decimal",
        "lvlText": "%1.",
        "isLgl": False,
    }
    assert r.get_label(_para(num_id="100", ilvl=8)) == "1."


def test_resolve_ilvl_by_pstyle_walks_basedon_ancestors() -> None:
    r = _resolver()
    # direct style match
    assert r._resolve_ilvl_by_pstyle("100", "PS") == 3
    # a descendant of PS also matches via the basedOn walk
    r.style_based_on["Child"] = "PS"
    assert r._resolve_ilvl_by_pstyle("100", "Child") == 3
    # no link for H4 → None
    assert r._resolve_ilvl_by_pstyle("100", "H4") is None


# ---------------------------------------------------------------------------
# numFmt rendering
# ---------------------------------------------------------------------------


def _fmt_resolver(num_fmt: str, lvl_text: str = "（%1）") -> NumberingResolver:
    """A resolver whose single abstract level uses ``num_fmt``."""
    r = NumberingResolver.__new__(NumberingResolver)
    r.abstract_nums = {
        "10": {0: {"start": 1, "numFmt": num_fmt, "lvlText": lvl_text, "isLgl": False}}
    }
    r.abstract_pstyle = {}
    r.num_to_abstract = {"100": "10"}
    r.counters = {}
    r.start_overrides = {}
    r.style_numpr = {}
    r.style_based_on = {}
    r.last_numId = None
    r.last_abstract_id = None
    r.last_style_id = None
    r.unsupported_formats = set()
    r.out_of_range_formats = set()
    r._warnings = None
    return r


def _label(r: NumberingResolver, count: int) -> str:
    r.counters["100"] = {0: count}
    return r._format_label("100", 0, r.abstract_nums["10"])


@pytest.mark.parametrize("num_fmt", ["lowerLetter", "upperLetter"])
@pytest.mark.parametrize("lvl_text", ["%1.", "(%1)", "%1)"])
def test_letter_labels_preserve_ordinals_after_first_alphabet(num_fmt, lvl_text):
    from lightrag.parser.docx.smart_heading.style_key import classify_numbering

    resolver = _fmt_resolver(num_fmt, lvl_text)
    for count in range(1, 79):
        label = resolver.get_label(_para(num_id="100", ilvl=0))
        match = classify_numbering(
            f"{label} Heading", numbering_format=resolver.last_label_format
        )
        assert match is not None
        assert match.ordinal == count
        if count in {1, 26, 27, 28, 52, 53, 78}:
            letters = {
                1: "a",
                26: "z",
                27: "aa",
                28: "bb",
                52: "zz",
                53: "aaa",
                78: "zzz",
            }[count]
            if num_fmt == "upperLetter":
                letters = letters.upper()
            assert label == lvl_text.replace("%1", letters)


@pytest.mark.parametrize("num_fmt", ["lowerLetter", "upperLetter"])
@pytest.mark.parametrize("count", [-1, 0, 79, 2147483647])
@pytest.mark.parametrize("override", [False, True])
def test_large_letter_starts_fall_back_before_allocating(num_fmt, count, override):
    resolver = _fmt_resolver(num_fmt, "%1.")
    resolver._warnings = {}
    if override:
        resolver.start_overrides = {"100": {0: count}}
    else:
        resolver.abstract_nums["10"][0]["start"] = count
    assert resolver.get_label(_para(num_id="100", ilvl=0)) == f"{count}."
    assert resolver.out_of_range_formats == ({num_fmt} if count > 78 else set())
    assert resolver._warnings == (
        {"numbering_out_of_range_formats": 1} if count > 78 else {}
    )


def test_read_pass_retains_letter_provenance_and_clears_it_on_plain_text():
    from docx import Document

    from lightrag.parser.docx.parse_document import _read_document_records
    from lightrag.parser.docx.smart_heading.features import StyleAttributes

    doc = Document()
    para = doc.add_paragraph("Heading")
    para._p.get_or_add_pPr().append(
        _para(num_id="100", ilvl=0).find(f"{{{W}}}pPr/{{{W}}}numPr")
    )
    doc.add_paragraph("II. Typed Roman heading")
    resolver = _fmt_resolver("lowerLetter", "%1.")
    resolver.abstract_nums["10"][0]["start"] = 35
    records = _read_document_records(
        doc, resolver, {}, None, {}, style_attributes=StyleAttributes()
    )
    assert records[0].text == "ii. Heading"
    assert records[0].numbering_format == "lowerLetter"
    assert records[1].numbering_format is None


def test_label_format_provenance_is_readable_before_any_label(tmp_path):
    """A freshly constructed resolver must already expose the attribute.

    The read pass today always calls ``get_label`` before reading the
    provenance (empty paragraphs ``continue`` before the read), so no
    production path hits this. The defect is that the attribute is part of
    the resolver's read surface while being declared only inside
    ``get_label``: any other consumer, or a future reordering of the read
    pass, gets an AttributeError instead of "no numbering here".
    """
    from docx import Document

    path = tmp_path / "empty.docx"
    Document().save(str(path))
    assert NumberingResolver(str(path)).last_label_format is None


def _cross_level_resolver(parent_fmt: str, child_fmt: str, lvl_text: str):
    """A two-level abstractNum whose ilvl-1 template is `lvl_text`."""
    r = _fmt_resolver(child_fmt, lvl_text)
    r.abstract_nums["10"] = {
        0: {"start": 2, "numFmt": parent_fmt, "lvlText": "%1.", "isLgl": False},
        1: {"start": 35, "numFmt": child_fmt, "lvlText": lvl_text, "isLgl": False},
    }
    return r


@pytest.mark.parametrize(
    ("parent_fmt", "child_fmt", "expected_fmt", "expected"),
    [
        # Child is alphabetic but its template renders the Roman parent: the
        # visible "ii" is Roman 2, NOT the alphabetic 35 the child would give.
        ("lowerRoman", "lowerLetter", "lowerRoman", ("RomanNum", 2)),
        # The inverse: a decimal child rendering its lowerLetter parent. The
        # parent's counter is seeded to 35, which renders "ii" alphabetically.
        ("lowerLetter", "decimal", "lowerLetter", ("EnAlpha", 35)),
    ],
)
def test_provenance_follows_the_leading_placeholder_not_the_current_level(
    parent_fmt, child_fmt, expected_fmt, expected
) -> None:
    """lvlText may reference only an ancestor level.

    The classifier reads the label's LEADING token, so the provenance must
    name the placeholder that produced it. Taking the current level's numFmt
    instead makes a Roman-looking token classify as its own inverse.
    """
    r = _cross_level_resolver(parent_fmt, child_fmt, "%1.")
    if parent_fmt == "lowerLetter":
        r.abstract_nums["10"][0]["start"] = 35
    label = r.get_label(_para(num_id="100", ilvl=1))

    assert label == "ii."
    assert r.last_label_format == expected_fmt
    cls = classify_numbering(f"{label} Heading", numbering_format=r.last_label_format)
    assert (cls.style_key, cls.ordinal) == expected


def test_provenance_is_the_first_placeholder_of_a_multi_level_template() -> None:
    """Guards the single-level coincidence: with "%1.%2." the leading token
    comes from level 0, so "current level" and "first placeholder" differ."""
    r = _cross_level_resolver("lowerLetter", "decimal", "%1.%2.")
    r.abstract_nums["10"][0]["start"] = 27
    label = r.get_label(_para(num_id="100", ilvl=1))

    assert label == "aa.35."
    assert r.last_label_format == "lowerLetter"


def test_provenance_skips_placeholders_that_render_nothing() -> None:
    """A `none` level occupies a template slot but contributes no token.

    Picking the leftmost SUBSTITUTED placeholder is not enough: with
    lvlText "%1%2." and a numFmt "none" level 0, level 0 wins on position
    while rendering "". The visible leading token comes from level 1, so
    attributing the label to "none" drops the alpha provenance and sends
    "ii" back to the Roman branch.
    """
    r = _fmt_resolver("lowerLetter", "%1%2.")
    r.abstract_nums["10"] = {
        0: {"start": 1, "numFmt": "none", "lvlText": "%1", "isLgl": False},
        1: {"start": 35, "numFmt": "lowerLetter", "lvlText": "%1%2.", "isLgl": False},
    }
    label = r.get_label(_para(num_id="100", ilvl=1))

    assert label == "ii."
    assert r.last_label_format == "lowerLetter"
    cls = classify_numbering(f"{label} Heading", numbering_format=r.last_label_format)
    assert (cls.style_key, cls.ordinal) == ("EnAlpha", 35)


def test_provenance_is_none_when_nothing_renders() -> None:
    """An all-empty render has no token to attribute a format to."""
    r = _fmt_resolver("none", "%1")
    assert r.get_label(_para(num_id="100", ilvl=0)) == ""
    assert r.last_label_format is None


@pytest.mark.parametrize("lvl_text", ["(%1)", "%1)"])
@pytest.mark.parametrize("num_fmt", ["lowerRoman", "upperRoman"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_parenthesized_roman_lists_keep_roman_ordinals(lvl_text, num_fmt, count):
    """The widened paren patterns accept repeated letters, so "(ii)" from a
    Roman list reaches them. _P_ROMAN cannot claim it — it only matches a
    "." / "、" terminator — so the carried numFmt is the only evidence that
    "ii" is 2 and not the alphabetic 35."""
    r = _fmt_resolver(num_fmt, lvl_text)
    r.abstract_nums["10"][0]["start"] = count
    label = r.get_label(_para(num_id="100", ilvl=0))

    assert r.last_label_format == num_fmt
    cls = classify_numbering(f"{label} Heading", numbering_format=r.last_label_format)
    assert cls is not None
    assert cls.ordinal == count


# The counting families all render 一/二/十/十一/… — [MS-DOCX] gives
# japaneseCounting as 一,二,三 and chineseCounting / taiwaneseCounting as
# 一 (1) / 十 (10). Chinese-locale Word writes 一二三 auto-numbering as
# japaneseCounting, the value that made test21 emit （1） instead of （一）.
_COUNTING_FORMATS = (
    "japaneseCounting",
    "chineseCounting",
    "taiwaneseCounting",
    "chineseCountingThousand",
)


def test_japanese_counting_renders_chinese_numerals() -> None:
    """Regression: `numFmt="japaneseCounting"` + `lvlText="（%1）"` used to fall
    through to the decimal default and emit （1） where Word shows （一）."""
    r = _fmt_resolver("japaneseCounting")
    assert _label(r, 1) == "（一）"
    assert _label(r, 2) == "（二）"
    assert _label(r, 11) == "（十一）"
    assert r.unsupported_formats == set()


@pytest.mark.parametrize("num_fmt", _COUNTING_FORMATS)
@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, "一"), (2, "二"), (10, "十"), (11, "十一"), (20, "二十"), (99, "九十九")],
)
def test_counting_families_are_positional(num_fmt, count, expected) -> None:
    """10 must render 十, not 一〇: only values past 9 tell a positional counting
    system apart from the digit-by-digit ideograph one."""
    assert _label(_fmt_resolver(num_fmt, "%1"), count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "一"),
        (10, "一〇"),
        (11, "一一"),
        (20, "二〇"),
        (99, "九九"),
        (100, "一〇〇"),
    ],
)
def test_ideograph_digital_is_digit_by_digit(count, expected) -> None:
    """``ideographDigital`` is NOT a counting system: per [MS-DOCX] 1/10/100 are
    U+4E00 / U+4E00U+3007 / U+4E00U+3007U+3007 (一 / 一〇 / 一〇〇)."""
    assert _label(_fmt_resolver("ideographDigital", "%1"), count) == expected


def test_ideograph_digital_does_not_reuse_the_counting_converter() -> None:
    """Pins the two families apart, so ideographDigital cannot be "simplified"
    into the counting table: they only diverge from 10 upward."""
    assert NumberingResolver._to_ideograph_digital(10) == "一〇"
    assert NumberingResolver._to_chinese(10) == "十"
    assert NumberingResolver._to_ideograph_digital(10) != NumberingResolver._to_chinese(
        10
    )


def test_unknown_format_degrades_to_decimal_but_is_reported() -> None:
    """An unknown numFmt is a legitimate OOXML value we do not implement, so the
    label still degrades to decimal — but it is recorded instead of silently
    producing a plausible-looking wrong label."""
    warnings: dict = {}
    r = _fmt_resolver("koreanCounting")
    r._warnings = warnings
    assert _label(r, 1) == "（1）"
    assert r.unsupported_formats == {"koreanCounting"}
    assert warnings == {"numbering_unsupported_formats": 1}
    # Re-hitting the same format neither re-warns nor double-counts.
    assert _label(r, 2) == "（2）"
    assert warnings == {"numbering_unsupported_formats": 1}
    # A second unknown format bumps the count to the number of DISTINCT values.
    r.abstract_nums["10"][0]["numFmt"] = "thaiCounting"
    assert _label(r, 3) == "（3）"
    assert r.unsupported_formats == {"koreanCounting", "thaiCounting"}
    assert warnings == {"numbering_unsupported_formats": 2}
    # An unmapped format is NOT also reported as out-of-range: the two branches
    # are mutually exclusive (no converter at all vs. a converter with a domain).
    assert r.out_of_range_formats == set()


@pytest.mark.parametrize("num_fmt", _COUNTING_FORMATS)
def test_counting_family_past_its_domain_is_recorded(num_fmt) -> None:
    """``_to_chinese`` renders 1-99 and degrades to the decimal string above it.

    That degradation is legible (nobody reads `（100）` as a Chinese numeral, unlike
    `（1）` passing for `（一）`), so it is not corrected here — the families do NOT
    share one rendering past 99 and no corpus document reaches it. It IS recorded,
    so a real document that gets there becomes findable evidence.
    """
    warnings: dict = {}
    r = _fmt_resolver(num_fmt)
    r._warnings = warnings
    assert _label(r, 99) == "（九十九）"  # in domain: nothing recorded
    assert warnings == {}
    assert r.out_of_range_formats == set()

    assert _label(r, 100) == "（100）"  # out of domain: decimal, but noisy
    assert r.out_of_range_formats == {num_fmt}
    assert warnings == {"numbering_out_of_range_formats": 1}
    # Re-hitting the same format neither re-warns nor double-counts.
    assert _label(r, 101) == "（101）"
    assert warnings == {"numbering_out_of_range_formats": 1}
    # The format stays supported — nothing lands in the unsupported ledger.
    assert r.unsupported_formats == set()


def test_out_of_range_counts_distinct_formats() -> None:
    """The counter is the number of DISTINCT formats, like its unsupported twin."""
    warnings: dict = {}
    r = _fmt_resolver("chineseCounting")
    r._warnings = warnings
    assert _label(r, 100) == "（100）"
    r.abstract_nums["10"][0]["numFmt"] = "japaneseCounting"
    assert _label(r, 100) == "（100）"
    assert r.out_of_range_formats == {"chineseCounting", "japaneseCounting"}
    assert warnings == {"numbering_out_of_range_formats": 2}


def test_ideograph_digital_has_no_domain_limit() -> None:
    """``ideographDigital`` renders every count digit-by-digit, so it must NOT be
    in the limited-domain table: 100 is a correct 一〇〇, not a degraded label."""
    warnings: dict = {}
    r = _fmt_resolver("ideographDigital", "%1")
    r._warnings = warnings
    assert _label(r, 100) == "一〇〇"
    assert _label(r, 1000) == "一〇〇〇"
    assert r.out_of_range_formats == set()
    assert warnings == {}
    assert "ideographDigital" not in NumberingResolver.LIMITED_DOMAIN_FORMATS
