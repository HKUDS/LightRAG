"""G1 tests: physical paragraph features for smart heading discovery.

Covers the font-size cascade (run rPr > style chain > docDefaults; the
paragraph-mark rPr styles the ¶ glyph only and never feeds text runs),
szCs fallback, whole-paragraph bold, alignment resolution,
char-weighted dominant size with label/markup exclusion, first-line re-stat
for soft-break split headings, page-break and TOC evidence, and the
record-level wiring through ``_read_document_records``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

from lightrag.parser.docx.smart_heading.features import (
    StyleAttributes,
    extract_paragraph_physical_features,
    parse_styles_attributes,
)

pytestmark = pytest.mark.offline


def _save(doc: Document, tmp_path: Path) -> Path:
    path = tmp_path / "features.docx"
    doc.save(path)
    return path


def _set_doc_default_size(doc: Document, half_points: int) -> None:
    styles_el = doc.styles.element
    doc_defaults = styles_el.find(qn("w:docDefaults"))
    if doc_defaults is None:
        doc_defaults = OxmlElement("w:docDefaults")
        styles_el.insert(0, doc_defaults)
    rpr_default = doc_defaults.find(qn("w:rPrDefault"))
    if rpr_default is None:
        rpr_default = OxmlElement("w:rPrDefault")
        doc_defaults.insert(0, rpr_default)
    rpr = rpr_default.find(qn("w:rPr"))
    if rpr is None:
        rpr = OxmlElement("w:rPr")
        rpr_default.append(rpr)
    sz = rpr.find(qn("w:sz"))
    if sz is None:
        sz = OxmlElement("w:sz")
        rpr.append(sz)
    sz.set(qn("w:val"), str(half_points))


def _add_szcs_only_run(para, text: str, half_points: int) -> None:
    run = para.add_run(text)
    rpr = run._r.get_or_add_rPr()
    for tag in ("w:sz",):
        el = rpr.find(qn(tag))
        if el is not None:
            rpr.remove(el)
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), str(half_points))
    rpr.append(szcs)


def _features_for(docx_path: Path, para_index: int):
    doc = Document(str(docx_path))
    styles = parse_styles_attributes(str(docx_path))
    return extract_paragraph_physical_features(doc.paragraphs[para_index]._p, styles)


# ---------------------------------------------------------------------------
# G1-1: font-size cascade
# ---------------------------------------------------------------------------


def test_font_size_cascade_run_over_style_over_default(tmp_path) -> None:
    doc = Document()
    _set_doc_default_size(doc, 21)  # 10.5pt
    style = doc.styles.add_style("Big28", WD_STYLE_TYPE.PARAGRAPH)
    style.font.size = Pt(14)  # sz=28

    # (0) run-level sz=24 beats style sz=28 beats default sz=21
    p0 = doc.add_paragraph(style="Big28")
    p0.add_run("run level").font.size = Pt(12)
    # (1) style-level only
    doc.add_paragraph("style level", style="Big28")
    # (2) docDefaults only
    doc.add_paragraph("default level")
    # (3) a paragraph-MARK rPr size does NOT feed the text run: the run has no
    # direct size, so it resolves to the paragraph STYLE (Big28 = 14pt), not
    # the ¶-mark's 18pt. Per ECMA-376 §17.3.1.29 the paragraph-mark rPr styles
    # the ¶ glyph only; WPS/Word render the text at the style size.
    p3 = doc.add_paragraph("para mark level", style="Big28")
    ppr = p3._p.get_or_add_pPr()
    mark_rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "36")
    mark_rpr.append(sz)
    ppr.insert(0, mark_rpr)

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).font_size_pt == 12.0
    assert _features_for(path, 1).font_size_pt == 14.0
    assert _features_for(path, 2).font_size_pt == 10.5
    assert _features_for(path, 3).font_size_pt == 14.0  # style, NOT ¶-mark 18pt


def test_style_chain_based_on_inheritance(tmp_path) -> None:
    doc = Document()
    base = doc.styles.add_style("BaseSized", WD_STYLE_TYPE.PARAGRAPH)
    base.font.size = Pt(16)
    child = doc.styles.add_style("ChildUnsized", WD_STYLE_TYPE.PARAGRAPH)
    child.base_style = base
    doc.add_paragraph("inherits via basedOn", style="ChildUnsized")

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).font_size_pt == 16.0


# ---------------------------------------------------------------------------
# G1-2: szCs fallback
# ---------------------------------------------------------------------------


def test_szcs_only_run_uses_szcs(tmp_path) -> None:
    doc = Document()
    para = doc.add_paragraph()
    _add_szcs_only_run(para, "中文内容字号", 30)  # 15pt via szCs

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).font_size_pt == 15.0


def _style_set_szcs_only(style, half_points: int) -> None:
    """Give a style an rPr carrying ONLY ``w:szCs`` (no ``w:sz``)."""
    rpr = style.element.get_or_add_rPr()
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), str(half_points))
    rpr.append(szcs)


def test_style_chain_szcs_does_not_mask_ancestor_sz(tmp_path) -> None:
    """A mid-chain szCs-only style must NOT shadow an ancestor's sz — the sz
    track resolves independently along basedOn (the test11 "10 图片及图题"
    shape: child szCs=28 over Normal sz=24 renders 12pt in WPS, not 14pt)."""
    doc = Document()
    base = doc.styles.add_style("SzBase", WD_STYLE_TYPE.PARAGRAPH)
    base.font.size = Pt(12)  # sz=24
    child = doc.styles.add_style("CsChild", WD_STYLE_TYPE.PARAGRAPH)
    child.base_style = base
    _style_set_szcs_only(child, 28)  # szCs=28, no sz
    doc.add_paragraph("图题样式段落", style="CsChild")

    path = _save(doc, tmp_path)
    styles = parse_styles_attributes(str(path))
    assert styles.style_size_half_points("CsChild") == 24
    assert _features_for(path, 0).font_size_pt == 12.0


def test_style_chain_three_level_sz_track(tmp_path) -> None:
    """szCs never inserts itself into the sz-track walk: A(sz=20) ← B(szCs
    only) ← C(nothing) resolve to 20 at every level below A."""
    doc = Document()
    a = doc.styles.add_style("SzA", WD_STYLE_TYPE.PARAGRAPH)
    a.font.size = Pt(10)  # sz=20
    b = doc.styles.add_style("CsB", WD_STYLE_TYPE.PARAGRAPH)
    b.base_style = a
    _style_set_szcs_only(b, 36)
    c = doc.styles.add_style("EmptyC", WD_STYLE_TYPE.PARAGRAPH)
    c.base_style = b

    path = _save(doc, tmp_path)
    styles = parse_styles_attributes(str(path))
    assert styles.style_size_half_points("CsB") == 20
    assert styles.style_size_half_points("EmptyC") == 20


def test_style_chain_szcs_fallback_precedes_existing_docdefaults_fallback(
    tmp_path,
) -> None:
    """DELIBERATE compatibility behavior, not full OOXML cascading: a style
    chain with NO sz at any level still resolves to its szCs (F3, the
    CJK-only style-chain shape), and that synthesized style size outranks
    the docDefaults sz in the run fallback cascade."""
    doc = Document()
    _set_doc_default_size(doc, 21)  # docDefaults sz=21 (10.5pt)
    cjk = doc.styles.add_style("CjkOnly", WD_STYLE_TYPE.PARAGRAPH)
    _style_set_szcs_only(cjk, 28)  # whole chain has no sz
    doc.add_paragraph("中文字号样式", style="CjkOnly")

    path = _save(doc, tmp_path)
    styles = parse_styles_attributes(str(path))
    assert styles.style_size_half_points("CjkOnly") == 28
    assert _features_for(path, 0).font_size_pt == 14.0


def test_object_only_paragraph_zero_chars_and_chain_size(tmp_path) -> None:
    """The test11 offender shape end to end at the features level: a run with
    no rPr and only a ``w:object`` (embedded OLE image) yields zero visible
    chars, and its synthesized size comes from the paragraph style's sz
    TRACK (12pt), not the style's own szCs (14pt)."""
    doc = Document()
    base = doc.styles.add_style("SzBase", WD_STYLE_TYPE.PARAGRAPH)
    base.font.size = Pt(12)  # sz=24
    child = doc.styles.add_style("CsChild", WD_STYLE_TYPE.PARAGRAPH)
    child.base_style = base
    _style_set_szcs_only(child, 28)
    para = doc.add_paragraph(style="CsChild")
    run = para.add_run()
    run._r.append(OxmlElement("w:object"))

    path = _save(doc, tmp_path)
    feats = _features_for(path, 0)
    assert feats.visible_char_count == 0
    assert feats.font_size_pt == 12.0


# ---------------------------------------------------------------------------
# G1-3: whole-paragraph bold
# ---------------------------------------------------------------------------


def test_all_bold_ignores_whitespace_runs(tmp_path) -> None:
    doc = Document()
    p0 = doc.add_paragraph()
    p0.add_run("Bold head").bold = True
    p0.add_run("   ")  # non-bold whitespace run must not break all-bold
    p0.add_run("still bold").bold = True

    p1 = doc.add_paragraph()
    p1.add_run("Bold part").bold = True
    p1.add_run(" plain tail")

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).all_bold is True
    assert _features_for(path, 1).all_bold is False


# ---------------------------------------------------------------------------
# G1-4: alignment resolution
# ---------------------------------------------------------------------------


def test_alignment_explicit_and_style_chain(tmp_path) -> None:
    doc = Document()
    centered_style = doc.styles.add_style("CenteredStyle", WD_STYLE_TYPE.PARAGRAPH)
    ppr = centered_style.element.get_or_add_pPr()
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    ppr.append(jc)

    def _explicit(text: str, val: str):
        para = doc.add_paragraph(text)
        p_ppr = para._p.get_or_add_pPr()
        p_jc = OxmlElement("w:jc")
        p_jc.set(qn("w:val"), val)
        p_ppr.append(p_jc)

    _explicit("explicit center", "center")  # 0
    doc.add_paragraph("style chain center", style="CenteredStyle")  # 1
    _explicit("both aligned", "both")  # 2
    _explicit("distribute aligned", "distribute")  # 3
    doc.add_paragraph("no alignment")  # 4

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).alignment == "center"
    assert _features_for(path, 1).alignment == "center"
    assert _features_for(path, 2).alignment == "both"
    assert _features_for(path, 3).alignment == "distribute"
    assert _features_for(path, 4).alignment is None


# ---------------------------------------------------------------------------
# G1-5: char-weighted dominant size / first-line re-stat / markup exclusion
# ---------------------------------------------------------------------------


def test_dominant_size_char_weighted_not_first_run(tmp_path) -> None:
    doc = Document()
    para = doc.add_paragraph()
    para.add_run("short12pt!").font.size = Pt(12)  # 10 visible chars
    para.add_run("x" * 30).font.size = Pt(16)  # 30 visible chars dominate

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).font_size_pt == 16.0


def test_dominant_size_tie_prefers_larger(tmp_path) -> None:
    doc = Document()
    para = doc.add_paragraph()
    para.add_run("a" * 10).font.size = Pt(12)
    para.add_run("b" * 10).font.size = Pt(14)

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).font_size_pt == 14.0


def test_superscript_chars_counted_but_markup_not(tmp_path) -> None:
    """The visible sup/sub characters count; the synthesized ``<sup>``
    wrappers never exist in source XML so they cannot pollute the stats."""
    doc = Document()
    para = doc.add_paragraph()
    para.add_run("E=mc").font.size = Pt(12)
    sup = para.add_run("2")
    sup.font.superscript = True
    sup.font.size = Pt(12)

    path = _save(doc, tmp_path)
    feats = _features_for(path, 0)
    assert feats.font_size_pt == 12.0
    # 4 + 1 visible chars across two runs
    assert sum(len(rf.text) for rf in feats.run_features) == 5


def test_textbox_content_excluded_from_features(tmp_path) -> None:
    """Review F1 (§2.2.2 textbox exclusion): a run anchoring a drawing whose
    textbox holds large bold text must NOT pollute the host paragraph's
    font-size / bold / visible-char stats — the baseline treats the whole
    drawing as an opaque placeholder and so must the feature extractor."""
    from lxml import etree

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = (
        f'<w:p xmlns:w="{w}">'
        '<w:r><w:rPr><w:sz w:val="24"/></w:rPr><w:t>宿主标题</w:t></w:r>'
        '<w:r><w:rPr><w:sz w:val="44"/></w:rPr><w:drawing><wp:inline xmlns:wp="x">'
        '<w:txbxContent><w:p><w:r><w:rPr><w:sz w:val="44"/><w:b/></w:rPr>'
        "<w:t>文本框内的超大加粗装饰文字很长很长很长</w:t></w:r></w:p>"
        "</w:txbxContent></wp:inline></w:drawing></w:r></w:p>"
    )
    para = etree.fromstring(xml)
    feats = extract_paragraph_physical_features(para, StyleAttributes())
    assert feats.font_size_pt == 12.0  # host run only, not the 22pt textbox
    assert feats.all_bold is False  # host run is not bold
    assert feats.visible_char_count == 4  # 宿主标题, textbox text excluded


def test_visible_char_count_excludes_generated_text(tmp_path) -> None:
    """Review F14 (§2.2.2): visible_char_count counts only source w:t chars —
    auto-numbering labels (prepended at read time) and <sup>/<equation>/
    placeholder markup never enter run_features, so they cannot skew FS_base
    weighting."""
    doc = Document()
    para = doc.add_paragraph()
    para.add_run("正文内容").font.size = Pt(12)  # 4 visible source chars

    path = _save(doc, tmp_path)
    feats = _features_for(path, 0)
    assert feats.visible_char_count == 4


def test_first_line_size_restat_for_softbreak(tmp_path) -> None:
    doc = Document()
    para = doc.add_paragraph()
    head = para.add_run("Heading line")
    head.font.size = Pt(16)
    head.add_break(WD_BREAK.LINE)
    body = para.add_run("body remainder " * 5)
    body.font.size = Pt(10.5)

    path = _save(doc, tmp_path)
    doc2 = Document(str(path))
    styles = parse_styles_attributes(str(path))
    feats = extract_paragraph_physical_features(doc2.paragraphs[0]._p, styles)

    from lightrag.parser.docx.smart_heading.features import (
        first_line_size_half_points,
        half_points_to_pt,
    )

    # Whole-paragraph dominant follows the longer body run…
    assert feats.font_size_pt == 10.5
    # …but the first line re-stat sees only the heading run.
    assert half_points_to_pt(first_line_size_half_points(feats.run_features)) == 16.0


# ---------------------------------------------------------------------------
# page break + TOC evidence
# ---------------------------------------------------------------------------


def test_page_break_evidence(tmp_path) -> None:
    doc = Document()
    p0 = doc.add_paragraph()
    p0.add_run("after page break").add_break(WD_BREAK.PAGE)

    p1 = doc.add_paragraph("page break before")
    ppr = p1._p.get_or_add_pPr()
    ppr.append(OxmlElement("w:pageBreakBefore"))

    doc.add_paragraph("plain")

    path = _save(doc, tmp_path)
    # A8: the signals stay apart — a TRAILING page-break run means the NEXT
    # paragraph starts the new page; only a LEADING run (before any visible
    # text) is equivalent to pageBreakBefore for this paragraph.
    p0 = _features_for(path, 0)  # text, then the break at the run's end
    assert p0.has_page_break_run is True
    assert p0.has_leading_page_break_run is False
    assert p0.page_break_before is False
    p1 = _features_for(path, 1)
    assert p1.page_break_before is True
    assert p1.has_page_break_run is False
    p2 = _features_for(path, 2)
    assert p2.page_break_before is False
    assert p2.has_page_break_run is False


def test_toc_field_and_link_evidence(tmp_path) -> None:
    doc = Document()

    p0 = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), ' TOC \\o "1-3" \\h ')
    p0._p.append(fld)

    p1 = doc.add_paragraph()
    run = p1.add_run()
    instr = OxmlElement("w:instrText")
    instr.text = ' TOC \\o "1-3" '
    run._r.append(instr)

    p2 = doc.add_paragraph()
    link = OxmlElement("w:hyperlink")
    link.set(qn("w:anchor"), "_Toc123456")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "Chapter One\t3"
    r.append(t)
    link.append(r)
    p2._p.append(link)

    doc.add_paragraph("ordinary text")

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).is_toc_field is True
    assert _features_for(path, 1).is_toc_field is True
    assert _features_for(path, 2).is_toc_link is True
    p3 = _features_for(path, 3)
    assert p3.is_toc_field is False and p3.is_toc_link is False


def test_sdt_docpart_gallery_toc_evidence() -> None:
    """Review §2.2.2/§3.4: an in-paragraph SDT whose docPartObj gallery is
    'Table of Contents' is structural TOC evidence."""
    from lxml import etree

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = (
        f'<w:p xmlns:w="{w}"><w:sdt><w:sdtPr><w:docPartObj>'
        '<w:docPartGallery w:val="Table of Contents"/>'
        "</w:docPartObj></w:sdtPr><w:sdtContent>"
        "<w:r><w:t>第一章 绪论\t3</w:t></w:r>"
        "</w:sdtContent></w:sdt></w:p>"
    )
    feats = extract_paragraph_physical_features(
        etree.fromstring(xml), StyleAttributes()
    )
    assert feats.is_toc_field is True


# ---------------------------------------------------------------------------
# record-level wiring through the read pass
# ---------------------------------------------------------------------------


def test_read_pass_populates_smart_features(tmp_path) -> None:
    from lightrag.parser.docx.numbering_resolver import NumberingResolver
    from lightrag.parser.docx.parse_document import (
        _read_document_records,
        parse_styles_outline_levels,
    )

    doc = Document()
    para = doc.add_paragraph("Centered big text")
    para.runs[0].font.size = Pt(16)
    ppr = para._p.get_or_add_pPr()
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    ppr.append(jc)
    doc.add_paragraph("")  # empty paragraph → empty_para record
    doc.add_paragraph("List entry", style="List Number")

    path = _save(doc, tmp_path)

    doc2 = Document(str(path))
    resolver = NumberingResolver(str(path))
    styles_outline = parse_styles_outline_levels(str(path))
    style_attributes = parse_styles_attributes(str(path))

    records = _read_document_records(
        doc2,
        resolver,
        styles_outline,
        None,
        {},
        style_attributes=style_attributes,
    )

    kinds = [r.kind for r in records]
    # Trailing section_break: the document-level sectPr at body end.
    assert kinds == ["para", "empty_para", "para", "section_break"]

    first = records[0]
    assert first.font_size_pt == 16.0
    assert first.alignment == "center"
    assert first.full_text_raw == "Centered big text"
    assert first.label == ""

    numbered = records[2]
    assert numbered.label == "1."
    assert numbered.text == "1. List entry"
    # The label is resolver-synthesized — visible XML text excludes it, so
    # full_text_raw keeps it (assembly input) while char stats never saw it.
    assert numbered.full_text_raw == "1. List entry"


def test_read_pass_smart_off_skips_features(tmp_path) -> None:
    from lightrag.parser.docx.numbering_resolver import NumberingResolver
    from lightrag.parser.docx.parse_document import (
        _read_document_records,
        parse_styles_outline_levels,
    )

    doc = Document()
    para = doc.add_paragraph("plain")
    para.runs[0].font.size = Pt(16)
    path = _save(doc, tmp_path)

    doc2 = Document(str(path))
    records = _read_document_records(
        doc2,
        NumberingResolver(str(path)),
        parse_styles_outline_levels(str(path)),
        None,
        {},
        style_attributes=None,
    )
    assert records[0].font_size_pt is None
    assert records[0].full_text_raw is None


def test_fractional_size_values_round_to_nearest_half_point() -> None:
    """A11 (§2.2.1): theme-derived fractional half-point values snap to the
    NEAREST grid step — truncation would bias 21.75pt down to 21.5pt."""
    from xml.etree import ElementTree as ET

    from lightrag.parser.docx.smart_heading.features import (
        _RawStyle,
        _element_direct_size,
        _read_rpr,
    )

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    rpr = ET.fromstring(f'<w:rPr xmlns:w="{w}"><w:sz w:val="43.5"/></w:rPr>')
    assert _element_direct_size(rpr) == 44  # int(float(...)) gave 43

    raw = _RawStyle()
    _read_rpr(rpr, raw)
    assert raw.sz_half_points == 44

    exact = ET.fromstring(f'<w:rPr xmlns:w="{w}"><w:sz w:val="43"/></w:rPr>')
    assert _element_direct_size(exact) == 43  # integer values untouched


def test_bare_sz_does_not_mask_valid_szcs() -> None:
    """Review F3: a valueless ``<w:sz/>`` must not shadow a valid ``<w:szCs>``
    — the effective size falls back to the complex-script size."""
    from xml.etree import ElementTree as ET

    from lightrag.parser.docx.smart_heading.features import _element_direct_size

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    rpr = ET.fromstring(f'<w:rPr xmlns:w="{w}"><w:sz/><w:szCs w:val="28"/></w:rPr>')
    assert _element_direct_size(rpr) == 28  # 14pt, not None


def test_table_cell_features_captured_on_table_records(tmp_path) -> None:
    """§2.2.4 table channel wiring: the read pass captures per-PHYSICAL-cell
    (text, effective size, has_outline) on ``kind=="table"`` records — a
    gridSpan full-width merge is ONE cell, sizes resolve through the style
    cascade, and an ``outlineLvl`` cell paragraph flags ``has_outline``."""
    from lightrag.parser.docx.numbering_resolver import NumberingResolver
    from lightrag.parser.docx.parse_document import (
        _read_document_records,
        parse_styles_outline_levels,
    )

    doc = Document()
    _set_doc_default_size(doc, 21)  # docDefaults 10.5pt — inherited by cells
    table = doc.add_table(rows=3, cols=2)
    # Row 0: full-width merged title cell at 22pt (direct run size).
    merged = table.cell(0, 0).merge(table.cell(0, 1))
    run = merged.paragraphs[0].add_run("产品标准化大纲")
    run.font.size = Pt(22)
    # Row 1: two plain cells with NO direct size — must inherit 10.5pt.
    table.cell(1, 0).paragraphs[0].add_run("档 号")
    table.cell(1, 1).paragraphs[0].add_run("1V1.0.0")
    # Row 2: a cell paragraph carrying a direct outlineLvl.
    outline_para = table.cell(2, 0).paragraphs[0]
    outline_para.add_run("带大纲的格")
    ppr = outline_para._p.get_or_add_pPr()
    lvl = OxmlElement("w:outlineLvl")
    lvl.set(qn("w:val"), "0")
    ppr.append(lvl)
    doc.add_paragraph("正文段落，以句号结尾。")

    path = _save(doc, tmp_path)
    resolver = NumberingResolver(str(path))
    styles_outline = parse_styles_outline_levels(str(path))
    styles = parse_styles_attributes(str(path))
    records = _read_document_records(
        Document(str(path)),
        resolver,
        styles_outline,
        None,
        {},
        style_attributes=styles,
    )

    tables = [r for r in records if r.kind == "table"]
    assert len(tables) == 1
    cf = tables[0].table_cell_features
    assert cf is not None and len(cf) == 3
    assert len(cf[0]) == 1  # gridSpan merge → ONE physical cell
    assert cf[0][0][0] == "产品标准化大纲" and cf[0][0][1] == 22.0
    assert len(cf[1]) == 2  # plain row → two physical cells
    assert cf[1][0][:2] == ("档 号", 10.5)  # inherited docDefaults size
    assert cf[1][1][:2] == ("1V1.0.0", 10.5)
    assert cf[2][0][2] is True  # outlineLvl cell → has_outline

    # Smart off: no style_attributes → the field stays None (legacy parity).
    records_off = _read_document_records(
        Document(str(path)),
        NumberingResolver(str(path)),
        styles_outline,
        None,
        {},
        style_attributes=None,
    )
    assert [r.table_cell_features for r in records_off if r.kind == "table"] == [None]


def test_empty_para_records_carry_page_and_section_evidence(tmp_path) -> None:
    """0-A: empty paragraphs are boundary carriers — a w:br page inside an
    empty para (always LEADING: no text), an empty para with pageBreakBefore,
    and an empty para holding a sectPr must all surface on the record; the
    sectPr one must also reset numbering tracking (the same semantics a
    non-empty sectPr paragraph gets), observable as the auto-number sequence
    restarting after the blank section break."""
    from lightrag.parser.docx.numbering_resolver import NumberingResolver
    from lightrag.parser.docx.parse_document import (
        _read_document_records,
        parse_styles_outline_levels,
    )

    doc = Document()
    doc.add_paragraph("第一段正文，以句号结尾。")
    # (1) empty para whose run holds a page break
    p_br = doc.add_paragraph()
    br_run = p_br.add_run()
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    br_run._r.append(br)
    # (2) empty para with w:pageBreakBefore
    p_pbb = doc.add_paragraph()
    pbb = OxmlElement("w:pageBreakBefore")
    p_pbb._p.get_or_add_pPr().append(pbb)
    # (3) empty para carrying a paragraph-level sectPr
    p_sect = doc.add_paragraph()
    p_sect._p.get_or_add_pPr().append(OxmlElement("w:sectPr"))
    doc.add_paragraph("末尾正文，以句号结尾。")
    path = _save(doc, tmp_path)

    doc2 = Document(str(path))
    records = _read_document_records(
        doc2,
        NumberingResolver(str(path)),
        parse_styles_outline_levels(str(path)),
        None,
        {},
        style_attributes=parse_styles_attributes(str(path)),
    )
    empties = [r for r in records if r.kind == "empty_para"]
    assert len(empties) == 3
    r_br, r_pbb, r_sect = empties
    assert r_br.has_page_break_run and r_br.has_leading_page_break_run
    assert not r_br.has_nonleading_page_break_run  # no text → always leading
    assert r_pbb.page_break_before
    assert r_sect.ends_section


def test_empty_para_sectpr_reset_parity_with_nonempty(tmp_path) -> None:
    """0-A numbering-reset consistency: an EMPTY sectPr paragraph must drive
    the resolver exactly like a NON-EMPTY sectPr paragraph does (the reset
    clears continuity tracking, not the numId counters — Word numbering
    continues across sections by numId; what must not differ is the reset
    call itself). Asserted as label parity between the two shapes."""
    from lightrag.parser.docx.numbering_resolver import NumberingResolver
    from lightrag.parser.docx.parse_document import (
        _read_document_records,
        parse_styles_outline_levels,
    )

    def _labels(sect_para_text: str, name: str) -> list[str]:
        doc = Document()
        for text in ("甲项条目", "乙项条目"):
            doc.add_paragraph(text, style="List Number")
        p_sect = doc.add_paragraph(sect_para_text)
        p_sect._p.get_or_add_pPr().append(OxmlElement("w:sectPr"))
        doc.add_paragraph("丙项条目", style="List Number")
        path = tmp_path / f"{name}.docx"
        doc.save(str(path))
        doc2 = Document(str(path))
        records = _read_document_records(
            doc2,
            NumberingResolver(str(path)),
            parse_styles_outline_levels(str(path)),
            None,
            {},
            style_attributes=parse_styles_attributes(str(path)),
        )
        sect_recs = [r for r in records if r.ends_section]
        assert len(sect_recs) == 1  # the sectPr carrier, empty or not
        return [r.label for r in records if r.text.endswith("条目")]

    assert _labels("", "empty_sect") == _labels("分节说明文字", "nonempty_sect")


def test_paragraph_mark_size_does_not_feed_text_run(tmp_path) -> None:
    """Regression (test11 外购、外协价格明细表): a caption whose TEXT run is
    unsized while its ¶-mark rPr AND a text-less page-break run carry a large
    size must resolve to the paragraph STYLE size, not the ¶-mark size —
    matching what WPS renders. The old cascade read the ¶-mark's 16pt and
    made the line the document's largest text (a phantom level-0 heading)."""
    doc = Document()
    _set_doc_default_size(doc, 21)  # docDefaults 10.5pt (never reached here)
    # Normal is 12pt so the run resolves there.
    doc.styles["Normal"].font.size = Pt(12)  # sz=24

    para = doc.add_paragraph()
    ppr = para._p.get_or_add_pPr()
    mark_rpr = OxmlElement("w:rPr")
    mark_sz = OxmlElement("w:sz")
    mark_sz.set(qn("w:val"), "32")  # ¶-mark 16pt — must be ignored for text
    mark_rpr.append(mark_sz)
    ppr.append(mark_rpr)
    # A text-LESS page-break run at 18pt (weight 0, must not drive dominant).
    br_run = para.add_run()
    br_rpr = br_run._r.get_or_add_rPr()
    br_sz = OxmlElement("w:sz")
    br_sz.set(qn("w:val"), "36")
    br_rpr.append(br_sz)
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    br_run._r.append(br)
    # The actual text run: NO direct size.
    para.add_run("外购、外协价格明细表")

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).font_size_pt == 12.0


def test_toc_entry_field_reference_flags_toc(tmp_path) -> None:
    """A field-based TOC entry (auto TOC: HYPERLINK/PAGEREF to a _Toc
    bookmark, tab + page number — no dot leader, no <w:hyperlink> element,
    no 'TOC' instruction) must be flagged is_toc_field, else it evades TOC
    removal and — once its unsized runs resolve to the heading-sized TOC
    style — is mis-promoted to a heading (test2 目次 regression)."""
    from lightrag.parser.docx.smart_heading.features import _is_toc_instr

    assert _is_toc_instr('HYPERLINK \\L "_TOC114570378"')
    assert _is_toc_instr("PAGEREF _TOC114570378 \\H")
    assert _is_toc_instr("TOC \\O 1-3 \\H")
    # A body cross-reference to a non-TOC bookmark is NOT a TOC entry.
    assert not _is_toc_instr("PAGEREF _REF456 \\H")
    assert not _is_toc_instr('HYPERLINK "HTTP://EXAMPLE.COM"')

    doc = Document()
    para = doc.add_paragraph()
    r1 = para.add_run()
    instr = OxmlElement("w:instrText")
    instr.text = ' HYPERLINK \\l "_Toc114570378" '
    r1._r.append(instr)
    para.add_run("1.1")
    para.add_run(" 任务来源")
    r2 = para.add_run()
    instr2 = OxmlElement("w:instrText")
    instr2.text = " PAGEREF _Toc114570378 \\h "
    r2._r.append(instr2)
    para.add_run("1")

    path = _save(doc, tmp_path)
    assert _features_for(path, 0).is_toc_field is True
