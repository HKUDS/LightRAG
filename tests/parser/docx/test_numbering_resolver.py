"""Unit tests for NumberingResolver ilvl resolution.

Covers the ilvl-resolution priority when a paragraph's numPr omits w:ilvl:
(a) explicit ilvl inherited through the style basedOn chain, (b) the
abstractNum per-level w:pStyle link, (c) default 0, plus the two direct-numPr
edge cases: (d) direct numId without ilvl still borrows the chain ilvl, and
(e) an explicit direct ilvl=0 is preserved (NOT treated as missing).

The resolution logic is exercised directly on hand-built dicts + <w:p>
elements — no numbering.xml crafting — so the tests target the merge/fallback
behaviour rather than the XML parsers.
"""

from __future__ import annotations

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
