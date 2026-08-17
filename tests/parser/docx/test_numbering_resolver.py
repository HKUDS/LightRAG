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
