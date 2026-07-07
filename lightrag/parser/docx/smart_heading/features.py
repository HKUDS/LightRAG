"""Physical paragraph features for smart heading discovery (spec §2.2.1/§2.2.2).

Two layers:

- :func:`parse_styles_attributes` reads ``styles.xml`` once per document and
  resolves each style's effective run formatting (``w:sz``/``w:szCs``/``w:b``)
  and paragraph formatting (``w:jc``) along the ``basedOn`` inheritance chain,
  seeded by ``docDefaults``. It is a superset of, and independent from,
  ``parse_styles_outline_levels`` (whose return type the smart-off path
  consumes directly and must not change).
- :func:`extract_paragraph_physical_features` computes per-paragraph signals
  from the live lxml element: the character-weighted dominant font size on the
  0.5pt grid, whole-paragraph bold, resolved alignment, explicit page-break
  evidence, and TOC structural evidence (field instructions / ``_Toc``
  bookmark links).

Font sizes are stored in half-points exactly as OOXML does and only converted
to pt at the edge, so the 0.5pt grid comparison stays exact (no float
tolerance — spec §2.2.1 mandates precise grid equality).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from typing import Any

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


#: Subtrees the baseline treats as opaque placeholders — it never recurses into
#: them, so their inner runs / field codes / hyperlinks are NOT part of the
#: paragraph's visible text. §2.2.2 likewise excludes 文本框 (textboxes) from all
#: stats. ``extract_paragraph_physical_features`` must prune them too; otherwise
#: ``iter()`` would descend into a decorative textbox and let its font size,
#: bold state, or an embedded TOC field pollute the host paragraph's features
#: (cover pages / red-header docs are exactly the target corpus).
_PRUNE_SUBTREE_TAGS = frozenset(
    {_w("drawing"), _w("pict"), _w("object"), _w("txbxContent")}
)


# ---------------------------------------------------------------------------
# styles.xml resolution
# ---------------------------------------------------------------------------


@dataclass
class _RawStyle:
    based_on: str | None = None
    size_half_points: int | None = None
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

    def style_size_half_points(self, style_id: str | None) -> int | None:
        """Effective ``w:sz`` (half-points) for a style chain, else None."""
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


def _read_rpr(rpr, raw: _RawStyle) -> None:
    if rpr is None:
        return
    sz = rpr.find(_w("sz"))
    if sz is None:
        sz = rpr.find(_w("szCs"))
    if sz is not None:
        val = sz.get(_w("val"))
        try:
            # Nearest half-point-grid value (§2.2.1): theme sources may emit
            # fractional half-points; truncation would bias 21.5pt down to 21pt.
            raw.size_half_points = round(float(val)) if val else None
        except (TypeError, ValueError):
            raw.size_half_points = None
    b = rpr.find(_w("b"))
    if b is not None:
        raw.bold = _parse_bool_attr(b)


def parse_styles_attributes(docx_path: str) -> StyleAttributes:
    """Parse styles.xml into effective per-style formatting.

    Missing/corrupt styles.xml yields an empty :class:`StyleAttributes`
    (every lookup falls through to docDefaults=None); per-paragraph trace
    failures are then counted by the caller toward the CB5 confidence gate.
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
                    attrs.default_size_half_points = raw.size_half_points
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
        # Same contract as parse_styles_outline_levels: a broken styles part
        # degrades to "no style info" rather than failing the parse.
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
        attrs._resolved_size[style_id] = _resolve(style_id, "size_half_points")
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
    is_toc_field: bool
    is_toc_link: bool
    size_trace_failed: bool  # no run had a resolvable size (CB5 input)
    style_id: str | None = None  # paragraph pStyle id
    run_features: list[RunFeature] = field(default_factory=list)

    @property
    def visible_char_count(self) -> int:
        """Visible source-text characters (w:t only), for FS_base weighting.

        §2.2.2 forbids weighting FS_base by parser-generated text — auto-
        numbering labels, ``<sup>`` wrappers and ``<equation>``/``<drawing>``
        /``<table>`` placeholders. ``run_features`` already holds only source
        ``w:t`` text (labels/placeholders never enter it), so the visible
        count is just the sum of per-run weights.
        """
        return sum(_weight(rf.text) for rf in self.run_features)


def _weight(text: str) -> int:
    """Character weight of a run: visible (non-whitespace) characters."""
    return sum(1 for ch in text if not ch.isspace())


def effective_font_size_pt(rec: Any) -> float | None:
    """Candidate-facing paragraph size (§2.2.2 / §3.1).

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
    if rpr is None:
        return None
    sz = rpr.find(_w("sz"))
    if sz is None:
        sz = rpr.find(_w("szCs"))
    if sz is None:
        return None
    val = sz.get(_w("val"))
    try:
        return round(float(val)) if val else None
    except (TypeError, ValueError):
        return None


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
    placeholder tokens the extractor synthesizes never appear here (spec
    §2.2.2 forbids counting rendered/synthetic characters).
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
    if alignment is None:
        alignment = styles.style_alignment(para_style_id)
    if alignment is None:
        alignment = styles.default_alignment

    # Paragraph-level fallbacks shared by every run (cascade: run rPr >
    # paragraph-mark rPr > style chain > docDefaults; spec §3.2).
    para_mark_size = _element_direct_size(para_mark_rpr)
    para_mark_bold = _element_direct_bold(para_mark_rpr)
    para_style_size = styles.style_size_half_points(para_style_id)
    para_style_bold = styles.style_bold(para_style_id)

    def _fallback_size(run_style_id: str | None) -> int | None:
        for candidate in (
            styles.style_size_half_points(run_style_id),
            para_mark_size,
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
    text_seen = False

    # Depth-first walk in document order, pruning opaque subtrees (drawings /
    # pictures / objects / textboxes) so their inner runs, field codes and
    # hyperlinks never contribute to THIS paragraph's features — baseline
    # parity plus §2.2.2 textbox exclusion. Deliberately NOT ``iter()`` + an
    # id() skip-set: lxml element proxies are transient, so their id() is not
    # stable across passes and a skip-set silently mis-prunes.
    def _walk(node) -> None:
        nonlocal is_toc_field, is_toc_link
        nonlocal has_page_break_run, has_leading_page_break_run, text_seen
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
                elif child.tag == _w("t") and (child.text or "").strip():
                    text_seen = True
            if text.strip():
                text_seen = True
            # Fall through to descend so a field-code w:instrText nested in
            # this run is still seen; standalone w:t/w:br carry no branch.
        elif tag == _w("instrText"):
            instr = (node.text or "").strip()
            if instr.upper().startswith("TOC"):
                is_toc_field = True
        elif tag == _w("fldSimple"):
            instr = (node.get(_w("instr")) or "").strip()
            if instr.upper().startswith("TOC"):
                is_toc_field = True
        elif tag == _w("hyperlink"):
            anchor = node.get(_w("anchor")) or ""
            if anchor.startswith("_Toc"):
                is_toc_link = True
        elif tag == _w("docPartGallery"):
            # §2.2.2 / §3.4 structural evidence: an in-paragraph SDT whose
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
        # Kept apart on purpose (§2.2.4 evidence b): pageBreakBefore means
        # THIS paragraph starts a page; a page-break run inside a paragraph
        # means the NEXT one does — conflating them points the single-title
        # boundary evidence at the wrong paragraph.
        page_break_before=page_break_before,
        has_page_break_run=has_page_break_run,
        has_leading_page_break_run=has_leading_page_break_run,
        is_toc_field=is_toc_field,
        is_toc_link=is_toc_link,
        size_trace_failed=size_trace_failed,
        style_id=para_style_id,
        run_features=run_features,
    )
