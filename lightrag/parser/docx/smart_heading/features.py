"""Physical paragraph features for smart heading discovery.

Two layers:

- :func:`parse_styles_attributes` reads ``styles.xml`` once per document and
  resolves each style's effective run formatting (``w:sz``/``w:szCs``/``w:b``)
  and paragraph formatting (``w:jc``) along the ``basedOn`` inheritance chain,
  seeded by ``docDefaults``. It is a superset of, and independent from,
  ``parse_styles_outline_levels`` (whose return type the smart-off path
  consumes directly and must not change).
- :func:`extract_paragraph_physical_features` computes per-paragraph signals
  from the live lxml element: the character-weighted dominant font size on the
  0.5pt grid, whole-paragraph bold, resolved alignment, leading whitespace
  padding, explicit page-break evidence, and TOC structural evidence (field
  instructions / ``_Toc`` bookmark links).

Font sizes are stored in half-points exactly as OOXML does and only converted
to pt at the edge, so the 0.5pt grid comparison stays exact (no float
tolerance — precise grid equality is required).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from typing import Any

from lightrag.utils import logger

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


#: Subtrees the baseline treats as opaque placeholders — it never recurses into
#: them, so their inner runs / field codes / hyperlinks are NOT part of the
#: paragraph's visible text. 文本框 (textboxes) are likewise excluded from all
#: stats. ``extract_paragraph_physical_features`` must prune them too; otherwise
#: ``iter()`` would descend into a decorative textbox and let its font size,
#: bold state, or an embedded TOC field pollute the host paragraph's features
#: (cover pages / red-header docs are exactly the target corpus).
_PRUNE_SUBTREE_TAGS = frozenset(
    {_w("drawing"), _w("pict"), _w("object"), _w("txbxContent")}
)

#: Tracked-change / comment subtrees that are NOT part of the final revised
#: document, so nothing inside them is visible to the reader. MUST mirror
#: ``parse_document._SKIP_PARAGRAPH_TAGS`` — the body text extractor drops
#: exactly these, and a paragraph-geometry scan that disagreed would let content
#: the user already deleted change how the visible text is classified
#: (``test_revision_skip_matches_body_parser`` pins the two together).
#: ``w:ins`` / ``w:moveTo`` are deliberately absent: inserted and moved-in
#: content IS visible, and is reached by ordinary recursion.
_SKIP_REVISION_SUBTREES = frozenset(
    f"{{{W_NS}}}{tag}"
    for tag in (
        "del",
        "moveFrom",
        "commentRangeStart",
        "commentRangeEnd",
        "commentReference",
        "annotationRef",
    )
)

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

#: OMML equation roots. Read ONLY by :func:`leading_nontext_content` — never
#: pruned from the feature walk, which must keep descending exactly as before
#: (pruning here would change the size/bold statistics of every equation
#: paragraph). Their text lives in ``m:t``, so they contribute nothing to
#: ``run_features`` either way.
_OMML_TAGS = frozenset({f"{{{M_NS}}}oMath", f"{{{M_NS}}}oMathPara"})


# ---------------------------------------------------------------------------
# styles.xml resolution
# ---------------------------------------------------------------------------


@dataclass
class _RawStyle:
    based_on: str | None = None
    #: w:sz and w:szCs are SEPARATE tracks: each inherits independently along
    #: the basedOn chain (a child style carrying only szCs must not shadow an
    #: ancestor's sz — the ASCII/East-Asian size WPS/Word actually renders).
    sz_half_points: int | None = None
    szcs_half_points: int | None = None
    bold: bool | None = None
    alignment: str | None = None


@dataclass
class StyleAttributes:
    """Effective formatting per styleId plus document defaults."""

    _resolved_size: dict[str, int | None] = field(default_factory=dict)
    _resolved_bold: dict[str, bool | None] = field(default_factory=dict)
    _resolved_alignment: dict[str, str | None] = field(default_factory=dict)
    default_size_half_points: int | None = None
    default_bold: bool | None = None
    default_alignment: str | None = None
    #: styleId of the document's default paragraph style (``w:default="1"``,
    #: usually "Normal"). A paragraph with no explicit ``<w:pStyle>`` inherits
    #: this style's formatting before docDefaults (OOXML cascade), so its run
    #: sizes must resolve through it — else a body whose base size lives on the
    #: default style (not in docDefaults) reads as size-unknown.
    default_para_style_id: str | None = None

    def style_size_half_points(self, style_id: str | None) -> int | None:
        """Compatibility-synthesized size (half-points) for a style chain:
        the basedOn-chain-resolved ``w:sz``, falling back to the chain's
        ``w:szCs`` only when NO level defines sz. Not raw ``w:sz``."""
        if not style_id:
            return None
        return self._resolved_size.get(style_id)

    def style_bold(self, style_id: str | None) -> bool | None:
        if not style_id:
            return None
        return self._resolved_bold.get(style_id)

    def style_alignment(self, style_id: str | None) -> str | None:
        if not style_id:
            return None
        return self._resolved_alignment.get(style_id)


def _parse_bool_attr(elem) -> bool:
    """OOXML on/off value: absent val means on; "0"/"false"/"none" mean off."""
    val = elem.get(_w("val"))
    if val is None:
        return True
    return val not in ("0", "false", "none")


def _grid_half_points(val: str | None) -> int | None:
    """Parse a w:sz/w:szCs val to the nearest 0.5pt-grid half-point, or None.

    Nearest-grid rounding: theme sources may emit fractional
    half-points; truncation would bias 21.5pt down to 21pt.
    """
    if not val:
        return None
    try:
        return round(float(val))
    except (TypeError, ValueError):
        return None


def _rpr_size_half_points(rpr) -> int | None:
    """Effective run-size from an rPr: prefer a usable w:sz, else w:szCs.

    A bare ``<w:sz/>`` (element present, no ``w:val``) must NOT mask a valid
    ``<w:szCs w:val=…/>`` — the ASCII size being unspecified does not void the
    complex-script size."""
    if rpr is None:
        return None
    for tag in ("sz", "szCs"):
        node = rpr.find(_w(tag))
        if node is not None:
            size = _grid_half_points(node.get(_w("val")))
            if size is not None:
                return size
    return None


def _read_rpr(rpr, raw: _RawStyle) -> None:
    """Read sz and szCs into their SEPARATE _RawStyle tracks.

    A bare ``<w:sz/>`` (no usable val) writes nothing: it neither shadows this
    level's szCs track nor interrupts an ancestor's sz track."""
    if rpr is None:
        return
    for tag, attr in (("sz", "sz_half_points"), ("szCs", "szcs_half_points")):
        node = rpr.find(_w(tag))
        if node is not None:
            size = _grid_half_points(node.get(_w("val")))
            if size is not None:
                setattr(raw, attr, size)
    b = rpr.find(_w("b"))
    if b is not None:
        raw.bold = _parse_bool_attr(b)


def parse_styles_attributes(
    docx_path: str, *, warnings: dict | None = None
) -> StyleAttributes:
    """Parse styles.xml into effective per-style formatting.

    Missing/corrupt styles.xml yields an empty :class:`StyleAttributes`
    (every lookup falls through to docDefaults=None); per-paragraph trace
    failures are then counted by the caller toward the CB5 confidence gate.
    Unlike the legacy ``parse_styles_outline_levels`` this does NOT swallow a
    parse failure silently — it records a warning so a document-wide style
    degradation is observable.
    """
    try:
        from defusedxml import ElementTree as ET
    except ImportError:
        from xml.etree import ElementTree as ET

    attrs = StyleAttributes()
    raw_styles: dict[str, _RawStyle] = {}

    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            if "word/styles.xml" not in zf.namelist():
                return attrs
            root = ET.parse(zf.open("word/styles.xml")).getroot()

            doc_defaults = root.find(_w("docDefaults"))
            if doc_defaults is not None:
                rpr_default = doc_defaults.find(_w("rPrDefault"))
                if rpr_default is not None:
                    raw = _RawStyle()
                    _read_rpr(rpr_default.find(_w("rPr")), raw)
                    # docDefaults is a single level: within-level merge (sz
                    # preferred, szCs fallback) matches _rpr_size_half_points.
                    attrs.default_size_half_points = (
                        raw.sz_half_points
                        if raw.sz_half_points is not None
                        else raw.szcs_half_points
                    )
                    attrs.default_bold = raw.bold
                ppr_default = doc_defaults.find(_w("pPrDefault"))
                if ppr_default is not None:
                    ppr = ppr_default.find(_w("pPr"))
                    if ppr is not None:
                        jc = ppr.find(_w("jc"))
                        if jc is not None:
                            attrs.default_alignment = jc.get(_w("val"))

            for style in root.findall(f".//{_w('style')}"):
                style_id = style.get(_w("styleId"))
                if not style_id:
                    continue
                # Record the default paragraph style (``w:default`` is an
                # attribute on ``<w:style>``, OOXML on/off semantics — not a
                # child ``w:val``, so _parse_bool_attr does not apply).
                if style.get(_w("type")) == "paragraph" and style.get(
                    _w("default")
                ) in ("1", "true", "on"):
                    attrs.default_para_style_id = style_id
                raw = _RawStyle()
                based_on = style.find(_w("basedOn"))
                if based_on is not None:
                    raw.based_on = based_on.get(_w("val"))
                _read_rpr(style.find(_w("rPr")), raw)
                ppr = style.find(_w("pPr"))
                if ppr is not None:
                    jc = ppr.find(_w("jc"))
                    if jc is not None:
                        raw.alignment = jc.get(_w("val"))
                raw_styles[style_id] = raw
    except Exception:
        # A broken styles part degrades to "no style info" rather than failing
        # the parse — but, unlike parse_styles_outline_levels, it is surfaced:
        # every paragraph then loses its style-chain size and the whole doc
        # slides toward CB5 low confidence, which should not be silent.
        if warnings is not None:
            warnings["smart_styles_xml_parse_failed"] = (
                warnings.get("smart_styles_xml_parse_failed", 0) + 1
            )
        logger.warning(
            "[smart_heading] styles.xml could not be parsed for %s; "
            "style-chain font sizes unavailable (degrading to docDefaults)",
            docx_path,
        )
        return attrs

    def _resolve(style_id: str, attr: str) -> object:
        visited: set[str] = set()
        cur: str | None = style_id
        while cur and cur not in visited:
            visited.add(cur)
            raw = raw_styles.get(cur)
            if raw is None:
                return None
            value = getattr(raw, attr)
            if value is not None:
                return value
            cur = raw.based_on
        return None

    for style_id in raw_styles:
        # Two-track size resolution WITHIN the basedOn chain: the sz track is
        # resolved over the whole chain first, and only when NO level defines
        # sz does the szCs track apply (a mid-chain szCs-only style — e.g. a
        # CJK caption style — must not shadow an ancestor's sz, which is the
        # size Word/WPS actually renders for ASCII/East-Asian text). This is
        # deliberately NOT full OOXML property-wise cascading: the resolved
        # style-chain szCs still outranks docDefaults sz in _fallback_size,
        # preserving the existing cross-layer compatibility fallback.
        sz = _resolve(style_id, "sz_half_points")
        attrs._resolved_size[style_id] = (
            sz if sz is not None else _resolve(style_id, "szcs_half_points")
        )
        attrs._resolved_bold[style_id] = _resolve(style_id, "bold")
        attrs._resolved_alignment[style_id] = _resolve(style_id, "alignment")
    return attrs


# ---------------------------------------------------------------------------
# per-paragraph physical features
# ---------------------------------------------------------------------------


@dataclass
class RunFeature:
    """One run's visible text plus its effective formatting."""

    text: str  # w:t content; soft line breaks contribute "\n"
    size_half_points: int | None
    bold: bool


@dataclass
class ParagraphPhysicalFeatures:
    font_size_pt: float | None  # char-weighted dominant, 0.5pt grid
    all_bold: bool
    alignment: str | None  # resolved jc value or None
    page_break_before: bool  # w:pPr/w:pageBreakBefore only
    has_page_break_run: bool  # a w:br type="page" run INSIDE this paragraph
    # w:br type="page" BEFORE the first visible character — the Ctrl+Enter
    # then-keep-typing shape; equivalent to pageBreakBefore for THIS para.
    has_leading_page_break_run: bool
    # w:br type="page" AFTER visible text — "the NEXT paragraph starts a new
    # page". Kept separate from the aggregate has_page_break_run so a leading
    # break is never double-counted as both a before-THIS and after-THIS
    # boundary (title-block window breaking reads exactly one side).
    has_nonleading_page_break_run: bool
    is_toc_field: bool
    is_toc_link: bool
    size_trace_failed: bool  # no run had a resolvable size (CB5 input)
    #: Net leading whitespace of the FIRST line, in CJK em (see
    #: :func:`leading_pad_em`). 0.0 for the overwhelming majority of
    #: paragraphs; a large value on a ``w:jc=center`` paragraph means the line
    #: is not visually centered (``guardrails.is_visually_centered``).
    leading_pad_em: float = 0.0
    style_id: str | None = None  # paragraph pStyle id
    run_features: list[RunFeature] = field(default_factory=list)

    @property
    def visible_char_count(self) -> int:
        """Visible source-text characters (w:t only), for FS_base weighting.

        FS_base must not be weighted by parser-generated text — auto-
        numbering labels, ``<sup>`` wrappers and ``<equation>``/``<drawing>``
        /``<table>`` placeholders. ``run_features`` already holds only source
        ``w:t`` text (labels/placeholders never enter it), so the visible
        count is just the sum of per-run weights.
        """
        return sum(_weight(rf.text) for rf in self.run_features)


def _weight(text: str) -> int:
    """Character weight of a run: visible (non-whitespace) characters."""
    return sum(1 for ch in text if not ch.isspace())


def _is_toc_instr(instr_upper: str) -> bool:
    """True for a field instruction that marks a TOC paragraph.

    Two shapes: the ``TOC`` field itself (the generator), and a TOC-ENTRY
    field — an auto-generated entry references a ``_Toc`` bookmark via
    ``PAGEREF``/``HYPERLINK`` (Word/WPS reserve the ``_Toc`` prefix for TOC
    targets, so a body cross-reference points at ``_Ref…``/named bookmarks
    instead). The entry paragraph carries neither a ``TOC`` instruction nor a
    ``<w:hyperlink>`` element — only these field codes — so without this it
    evades detection and, once its runs resolve to the (heading-sized) TOC
    style, is mis-promoted to a heading. ``instr_upper`` is already uppercased.
    """
    return instr_upper.startswith("TOC") or "_TOC" in instr_upper


def effective_font_size_pt(rec: Any) -> float | None:
    """Candidate-facing paragraph size.

    A soft-break-split heading line re-stats its FIRST line's characters —
    the whole-paragraph dominant size would be swamped by the demoted body
    remainder. Everything else uses the paragraph dominant size.
    """
    if (
        getattr(rec, "demoted_body_text", None) is not None
        and rec.first_line_font_size_pt is not None
    ):
        return rec.first_line_font_size_pt
    return rec.font_size_pt


def _element_direct_size(rpr) -> int | None:
    # Shared sz/szCs resolution prefers a usable w:sz and falls back to szCs,
    # so a bare <w:sz/> cannot mask a valid <w:szCs>.
    return _rpr_size_half_points(rpr)


def _element_direct_bold(rpr) -> bool | None:
    if rpr is None:
        return None
    b = rpr.find(_w("b"))
    if b is None:
        return None
    return _parse_bool_attr(b)


def _run_visible_text(run) -> str:
    """Visible text of one run: w:t contents, soft breaks as newline.

    Counts only source OOXML text — numbering labels, ``<sup>`` wrappers and
    placeholder tokens the extractor synthesizes never appear here
    (rendered/synthetic characters are never counted).
    """
    parts: list[str] = []
    for child in run:
        tag = child.tag
        if tag == _w("t"):
            parts.append(child.text or "")
        elif tag == _w("br"):
            # Page/column breaks are invisible; line breaks split lines.
            if child.get(_w("type")) in (None, "textWrapping"):
                parts.append("\n")
        elif tag == _w("tab"):
            parts.append("\t")
    return "".join(parts)


#: Width of each whitespace variant in CJK em (one ideograph = 1.0 em) — the
#: unit :func:`leading_pad_em` reports and ``guardrails.CENTER_MAX_LEADING_PAD_EM``
#: budgets against. Absolute points are deliberately never involved: the
#: question is "how many character widths of padding", which is font-size
#: independent. Anything whitespace-ish NOT listed falls back to
#: ``_PAD_EM_FALLBACK`` (never 0.0 — an unknown space still occupies space, and
#: silently measuring it as zero would let one exotic variant defeat the rule).
#: Keys stay ``\uXXXX`` escapes, never literal characters: most of these render
#: identically to a plain space (or not at all), so a literal table could not be
#: read or reviewed, and an editor's whitespace normalization could rewrite it.
_PAD_EM_WIDTHS = {
    " ": 0.5,  # SPACE — half an em beside a CJK glyph (英文空格)
    " ": 0.5,  # NO-BREAK SPACE — WPS/Word 公文 padding (不换行空格)
    "　": 1.0,  # IDEOGRAPHIC SPACE — one full em (中文全角空格)
    " ": 0.5,  # EN QUAD
    " ": 1.0,  # EM QUAD
    " ": 0.5,  # EN SPACE
    " ": 1.0,  # EM SPACE
    " ": 1 / 3,  # THREE-PER-EM SPACE
    " ": 0.25,  # FOUR-PER-EM SPACE
    " ": 1 / 6,  # SIX-PER-EM SPACE
    " ": 0.5,  # FIGURE SPACE — one digit wide
    " ": 0.25,  # PUNCTUATION SPACE
    " ": 0.2,  # THIN SPACE
    " ": 0.1,  # HAIR SPACE
    " ": 0.2,  # NARROW NO-BREAK SPACE
    " ": 0.25,  # MEDIUM MATHEMATICAL SPACE
    " ": 0.5,  # OGHAM SPACE MARK
    # Zero-width invisibles: no width, but they MUST be listed — see
    # _PAD_SCAN_EXTRA below.
    "​": 0.0,  # ZERO WIDTH SPACE
    "‌": 0.0,  # ZERO WIDTH NON-JOINER
    "‍": 0.0,  # ZERO WIDTH JOINER
    "⁠": 0.0,  # WORD JOINER
    "﻿": 0.0,  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\t": 2.0,  # TAB — true width depends on tab stops; 2 em matches the
    # Word/WPS default 0.74cm stop at 小四 body size.
}

#: Table entries ``str.isspace()`` reports False for — the zero-width
#: invisibles. The pad scan must still step OVER them: a stray ZWSP/BOM ahead of
#: the real padding would otherwise terminate the scan at 0.0 and hide the
#: offset entirely. DERIVED from the table rather than listed again, so adding a
#: non-isspace entry above cannot forget to extend the scan.
_PAD_SCAN_EXTRA = frozenset(ch for ch in _PAD_EM_WIDTHS if not ch.isspace())

#: Width charged to a whitespace character absent from ``_PAD_EM_WIDTHS``.
_PAD_EM_FALLBACK = 0.5


def _occupies_no_width(ch: str) -> bool:
    """True for a character that contributes no visible glyph to a line.

    THE single definition of "not a visible character" for the centered-pad
    rule, shared by :func:`_pad_run_em` (what the scan may step over) and
    :func:`_has_visible_char` (what ends the scan). Splitting the two is a live
    bug source: ``str.strip()`` does NOT remove the ``_PAD_SCAN_EXTRA``
    zero-width characters, so a ``strip()``-based "is there text here" test
    reads a lone ZWSP as visible text while the pad scan steps straight over
    it — two paragraphs that look identical on screen then classify
    differently.
    """
    return ch.isspace() or ch in _PAD_SCAN_EXTRA


def _has_visible_char(text: str) -> bool:
    """True when ``text`` holds at least one width-occupying character."""
    return any(not _occupies_no_width(ch) for ch in text)


def _pad_run_em(text: str) -> float:
    """Total em width of ``text``'s leading whitespace run."""
    total = 0.0
    for ch in text:
        if not _occupies_no_width(ch):
            break
        total += _PAD_EM_WIDTHS.get(ch, _PAD_EM_FALLBACK)
    return total


def _visible_first_line(para_element) -> tuple[bool, str]:
    """``(non-text content leads, text of the first visible line)``.

    ONE strictly-document-order scan carrying the paragraph geometry the
    centered-pad rule needs, under the SAME visibility semantics as
    ``parse_document.extract_paragraph_content``:

    - ``_SKIP_REVISION_SUBTREES`` (``w:del`` / ``w:moveFrom`` / comment markers)
      are skipped whole. Content the author deleted is not on the page, so it can
      neither supply padding nor mask the real leading content.
    - a drawing / picture / OLE object / OMML equation reached before the first
      visible character sets the flag (see :func:`leading_pad_em`);
    - ``w:t`` text and ``w:tab`` build the line; a soft ``w:br`` ends it (page /
      column breaks are invisible and ignored).

    Why this does NOT read ``run_features``: that list is built by the feature
    walk, which has no revision filtering and maps every ``w:tab`` it meets —
    including a DELETED one — to ``"\\t"``. A deleted tab would then measure as
    padding the reader cannot see. Adding revision filtering to the feature walk
    instead would change its page-break and TOC-field evidence (a break or field
    code inside a deleted subtree), which is unrelated pre-existing behavior.
    """
    parts: list[str] = []
    nontext_leads = False
    line_done = False

    def visit(node) -> None:
        nonlocal nontext_leads, line_done
        tag = node.tag
        if tag in _SKIP_REVISION_SUBTREES:
            return
        if tag in _PRUNE_SUBTREE_TAGS or tag in _OMML_TAGS:
            if not _has_visible_char("".join(parts)):
                nontext_leads = True
            return
        if tag == _w("t"):
            parts.append(node.text or "")
            return
        if tag == _w("tab"):
            parts.append("\t")
            return
        if tag == _w("br"):
            # Only a soft (text-wrapping) break ends the line; page / column
            # breaks occupy no width and do not split the rendered line here.
            if node.get(_w("type")) in (None, "textWrapping"):
                line_done = True
            return
        for child in node:
            visit(child)
            if line_done:
                return

    for child in para_element:
        if child.tag == _w("pPr"):
            continue
        visit(child)
        if line_done:
            break
    return nontext_leads, "".join(parts)


def leading_nontext_content(para_element) -> bool:
    """True when non-``w:t`` content precedes the paragraph's first visible char.

    Diagnostic accessor over :func:`_visible_first_line`; see
    :func:`leading_pad_em` for what the flag means and why it exists.
    """
    return _visible_first_line(para_element)[0]


def leading_pad_em(para_element) -> float:
    """Net leading whitespace of the FIRST visible line, in CJK em.

    Word centers a line INCLUDING its leading whitespace, so a centered
    paragraph padded with spaces renders with its visible text pushed right by
    HALF this value — the 空格排版落款/署名 shape, visually right-aligned while
    ``w:jc`` still says ``center``.

    TRAILING whitespace of the same line is subtracted: symmetric padding
    (``"   标题   "``) leaves the text centered and must not be flagged. The
    result is signed, and only the positive direction is meaningful — a
    trailing-heavier line is left to render however Word renders it (Word
    generally drops trailing whitespace at a line end, so no rightward shift
    is claimed for it).

    Only the first line is measured, matching
    :func:`first_line_size_half_points`: the padding of a soft-break
    continuation line says nothing about the heading line's alignment.

    Reports 0.0 — "no measurable pad" — when non-text content LEADS the line
    (:func:`_visible_first_line`). A drawing / OLE object / OMML equation renders
    as a placeholder token but carries no ``w:t``, so whitespace that merely
    FOLLOWS it would read as leading padding. The shape that hits is a centered
    formula line, ``<equation>…</equation>\\t\\t（3）``, whose two layout tabs
    separate the formula from its number: measured naively it lands at exactly
    4.0 em and loses the centered channel for a reason that has nothing to do
    with padding. Reporting 0.0 keeps the do-no-harm direction — the rule only
    ever fires on evidence it actually holds.

    Both the pad and that flag come from ONE scan over the live element, with
    the body extractor's revision semantics. Splitting them across two inputs is
    what produced the two mirror-image defects this signature replaces: a
    DELETED drawing masked a real 14.5em signature pad (flag said "non-text
    leads" while the body text showed only the padded signature), and DELETED
    tabs manufactured a 4.0em pad on a genuinely centered line (they reached
    ``run_features`` because the feature walk maps every ``w:tab`` regardless of
    revision context). Either way invisible content decided the verdict.
    """
    nontext_leads, first_line = _visible_first_line(para_element)
    if nontext_leads:
        return 0.0
    return _pad_run_em(first_line) - _pad_run_em(first_line[::-1])


def dominant_size_half_points(
    run_features: list[RunFeature],
) -> int | None:
    """Char-weighted dominant size; ties break toward the LARGER size."""
    weights: dict[int, int] = {}
    for rf in run_features:
        if rf.size_half_points is None:
            continue
        w = _weight(rf.text)
        if w <= 0:
            continue
        weights[rf.size_half_points] = weights.get(rf.size_half_points, 0) + w
    if not weights:
        # No weighted text at all (e.g. whitespace-only runs): fall back to
        # the first sized run so a lone-run paragraph still reports a size.
        for rf in run_features:
            if rf.size_half_points is not None:
                return rf.size_half_points
        return None
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


def first_line_size_half_points(run_features: list[RunFeature]) -> int | None:
    """Dominant size restricted to text before the first soft line break."""
    clipped: list[RunFeature] = []
    for rf in run_features:
        head, sep, _rest = rf.text.partition("\n")
        clipped.append(RunFeature(head, rf.size_half_points, rf.bold))
        if sep:
            break
    return dominant_size_half_points(clipped)


def half_points_to_pt(half_points: int | None) -> float | None:
    """Half-points → pt on the 0.5pt grid (exact by construction)."""
    if half_points is None:
        return None
    return half_points / 2.0


def extract_paragraph_physical_features(
    para_element,
    styles: StyleAttributes,
) -> ParagraphPhysicalFeatures:
    """Compute the smart-heading physical features for one ``w:p`` element."""
    ppr = para_element.find(_w("pPr"))

    para_style_id: str | None = None
    para_mark_rpr = None
    page_break_before = False
    alignment: str | None = None
    if ppr is not None:
        pstyle = ppr.find(_w("pStyle"))
        if pstyle is not None:
            para_style_id = pstyle.get(_w("val"))
        para_mark_rpr = ppr.find(_w("rPr"))
        pbb = ppr.find(_w("pageBreakBefore"))
        if pbb is not None and _parse_bool_attr(pbb):
            page_break_before = True
        jc = ppr.find(_w("jc"))
        if jc is not None:
            alignment = jc.get(_w("val"))
    # No explicit <w:pStyle>: the document's default paragraph style still
    # applies (OOXML cascade), so size/bold/alignment must resolve through it
    # before falling to docDefaults.
    if para_style_id is None:
        para_style_id = styles.default_para_style_id
    if alignment is None:
        alignment = styles.style_alignment(para_style_id)
    if alignment is None:
        alignment = styles.default_alignment

    # Paragraph-level fallbacks shared by every run.
    para_mark_bold = _element_direct_bold(para_mark_rpr)
    para_style_size = styles.style_size_half_points(para_style_id)
    para_style_bold = styles.style_bold(para_style_id)

    def _fallback_size(run_style_id: str | None) -> int | None:
        # A text run with no direct w:sz resolves rStyle (character style) >
        # paragraph style > docDefaults — the OOXML run-property chain. The
        # paragraph-MARK rPr (``w:pPr/w:rPr``) is DELIBERATELY excluded: per
        # ECMA-376 §17.3.1.29 it formats the paragraph-mark glyph (¶) only,
        # NOT the text runs, and WPS/Word render such a run at the paragraph
        # style size accordingly. Consulting it here inflated a caption whose
        # text run was unsized but whose ¶ mark carried a larger sz, making it
        # the document's largest text and a phantom top-level heading.
        for candidate in (
            styles.style_size_half_points(run_style_id),
            para_style_size,
            styles.default_size_half_points,
        ):
            if candidate is not None:
                return candidate
        return None

    def _fallback_bold(run_style_id: str | None) -> bool | None:
        for candidate in (
            styles.style_bold(run_style_id),
            para_mark_bold,
            para_style_bold,
            styles.default_bold,
        ):
            if candidate is not None:
                return candidate
        return None

    run_features: list[RunFeature] = []
    is_toc_field = False
    is_toc_link = False
    has_page_break_run = False
    has_leading_page_break_run = False
    has_nonleading_page_break_run = False
    text_seen = False

    # Depth-first walk in document order, pruning opaque subtrees (drawings /
    # pictures / objects / textboxes) so their inner runs, field codes and
    # hyperlinks never contribute to THIS paragraph's features — baseline
    # parity plus textbox exclusion. Deliberately NOT ``iter()`` + an
    # id() skip-set: lxml element proxies are transient, so their id() is not
    # stable across passes and a skip-set silently mis-prunes.
    def _walk(node) -> None:
        nonlocal is_toc_field, is_toc_link
        nonlocal has_page_break_run, has_leading_page_break_run
        nonlocal has_nonleading_page_break_run, text_seen
        tag = node.tag
        if tag in _PRUNE_SUBTREE_TAGS:
            return
        if tag == _w("r"):
            # Skip the paragraph-mark rPr context: w:pPr/w:rPr is not a run.
            rpr = node.find(_w("rPr"))
            run_style_id = None
            if rpr is not None:
                rstyle = rpr.find(_w("rStyle"))
                if rstyle is not None:
                    run_style_id = rstyle.get(_w("val"))
            size = _element_direct_size(rpr)
            if size is None:
                size = _fallback_size(run_style_id)
            bold = _element_direct_bold(rpr)
            if bold is None:
                bold = _fallback_bold(run_style_id)
            text = _run_visible_text(node)
            run_features.append(RunFeature(text, size, bool(bold)))
            # Positional page-break detection: iterate the run's children in
            # order so a break before the first visible character reads as
            # "this paragraph starts the new page" (Ctrl+Enter then typing).
            for child in node:
                if child.tag == _w("br") and child.get(_w("type")) == "page":
                    has_page_break_run = True
                    if not text_seen:
                        has_leading_page_break_run = True
                    else:
                        has_nonleading_page_break_run = True
                elif child.tag == _w("t") and (child.text or "").strip():
                    text_seen = True
            if text.strip():
                text_seen = True
            # Fall through to descend so a field-code w:instrText nested in
            # this run is still seen; standalone w:t/w:br carry no branch.
        elif tag == _w("instrText"):
            instr = (node.text or "").strip().upper()
            if _is_toc_instr(instr):
                is_toc_field = True
        elif tag == _w("fldSimple"):
            instr = (node.get(_w("instr")) or "").strip().upper()
            if _is_toc_instr(instr):
                is_toc_field = True
        elif tag == _w("hyperlink"):
            anchor = node.get(_w("anchor")) or ""
            if anchor.startswith("_Toc"):
                is_toc_link = True
        elif tag == _w("docPartGallery"):
            # Structural evidence: an in-paragraph SDT whose
            # docPartObj gallery is "Table of Contents" marks a TOC field.
            # (Body-level TOC SDTs are not read at all — baseline invariant.)
            if (node.get(_w("val")) or "").strip() == "Table of Contents":
                is_toc_field = True
        for child in node:
            _walk(child)

    _walk(para_element)

    weighted = [rf for rf in run_features if _weight(rf.text) > 0]
    all_bold = bool(weighted) and all(rf.bold for rf in weighted)

    dominant = dominant_size_half_points(run_features)
    size_trace_failed = dominant is None and any(
        _weight(rf.text) > 0 for rf in run_features
    )

    return ParagraphPhysicalFeatures(
        font_size_pt=half_points_to_pt(dominant),
        all_bold=all_bold,
        alignment=alignment,
        # Kept apart on purpose (title-block evidence b): pageBreakBefore means
        # THIS paragraph starts a page; a page-break run inside a paragraph
        # means the NEXT one does — conflating them points the single-title
        # boundary evidence at the wrong paragraph.
        page_break_before=page_break_before,
        has_page_break_run=has_page_break_run,
        has_leading_page_break_run=has_leading_page_break_run,
        has_nonleading_page_break_run=has_nonleading_page_break_run,
        is_toc_field=is_toc_field,
        is_toc_link=is_toc_link,
        size_trace_failed=size_trace_failed,
        leading_pad_em=leading_pad_em(para_element),
        style_id=para_style_id,
        run_features=run_features,
    )
