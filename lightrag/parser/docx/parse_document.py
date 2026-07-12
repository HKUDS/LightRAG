#!/usr/bin/env python3
"""
ABOUTME: Parses DOCX documents into text blocks using python-docx
ABOUTME: Extracts automatic numbering, splits by headings, converts tables to JSON
"""

import json
import sys
from dataclasses import dataclass

try:
    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError
except ImportError as exc:
    # Raise instead of sys.exit: this module is imported in-process by the
    # gunicorn/uvicorn worker, where a SystemExit would tear down the whole
    # worker rather than surfacing a normal, catchable error.
    raise ImportError(
        "python-docx not installed. Run: pip install python-docx"
    ) from exc

from lightrag.constants import MAX_HEADING_LENGTH
from lightrag.parser._markdown import (
    render_heading_line,
    strip_heading_markdown_prefix,
)
from lightrag.utils import logger
from .numbering_resolver import NumberingResolver
from .table_extractor import TableExtractor
from .drawing_image_extractor import (
    DrawingExtractionContext,
    extract_drawing_placeholder_from_element,
    extract_vml_image_placeholder_from_element,
)


# MAX_HEADING_LENGTH is imported from lightrag.constants above (a raw-char UI
# ceiling shared with the smart-heading synthesis path); the module-level import
# re-exports it so existing ``from ...parse_document import MAX_HEADING_LENGTH``
# call sites (e.g. tests) keep working.

# OOXML tracked-change/comment tags whose subtree must be dropped so we only
# surface the *final* revised text. w:ins / w:moveTo are kept via default
# recursion so inserted/moved-in content survives.
_SKIP_REVISION_TAGS = frozenset({"del", "moveFrom"})
_SKIP_COMMENT_TAGS = frozenset(
    {"commentRangeStart", "commentRangeEnd", "commentReference", "annotationRef"}
)
_SKIP_PARAGRAPH_TAGS = _SKIP_REVISION_TAGS | _SKIP_COMMENT_TAGS


class DocxContentError(ValueError):
    """DOCX content violates a parsing constraint (heading/table/anchor limits).

    Raised instead of calling ``sys.exit`` so the pipeline's per-document
    ``except Exception`` handler marks just that document FAILED while the
    gunicorn/uvicorn worker process keeps running. Subclasses ``ValueError``
    (i.e. an ``Exception``, not ``BaseException``) so the existing pipeline
    handlers catch it.
    """


def format_error(title: str, details: str, solution: str) -> str:
    """
    Build a friendly, formatted error message (title / details / SOLUTION).

    Args:
        title: Error title
        details: Detailed error information
        solution: Suggested solution steps

    Returns:
        str: The formatted multi-line message.
    """
    return (
        "\n"
        + "=" * 68
        + f"\nERROR: {title}\n"
        + "=" * 68
        + f"\n\n{details}"
        + "\n\nSOLUTION:\n"
        + solution
        + "\n\n"
        + "=" * 68
        + "\n"
    )


def print_error(title: str, details: str, solution: str):
    """Print a friendly, formatted error message to stderr."""
    print(format_error(title, details, solution), file=sys.stderr)


def _diagnose_invalid_docx(file_path: str) -> tuple[str, str]:
    """Diagnose why a ``.docx`` file is not a valid OOXML/ZIP package.

    python-docx raises ``PackageNotFoundError("Package not found at '...'")``
    both when the path is missing AND when the file exists but is not a valid
    zip. By the time this runs the file has already been confirmed to exist
    (the native worker validates ``p.exists()`` first), so the real cause is a
    corrupt file or a non-DOCX payload wearing a ``.docx`` extension. Sniff the
    magic bytes to name the actual format so the error message reflects the
    real problem instead of an empty "not found".

    Returns a ``(details, solution)`` tuple for :func:`format_error`. Reads only
    the file header and never raises — any IO failure degrades to a generic
    "cannot read" diagnosis.
    """
    import zipfile

    convert_solution = (
        "  1. Open the file in Microsoft Word or WPS\n"
        '  2. Use "Save As" and choose "Word Document (*.docx)"\n'
        "  3. Re-upload the converted .docx to LightRAG"
    )

    try:
        with open(file_path, "rb") as f:
            head = f.read(8)
    except OSError as exc:
        return (
            f"The file at '{file_path}' could not be read: {exc}",
            "  1. Verify the file exists and is readable\n"
            "  2. Re-upload it to LightRAG",
        )

    if not head:
        return (
            f"The file at '{file_path}' is empty (0 bytes). The upload was "
            "likely truncated or the source file is corrupt.",
            "  1. Check the original document opens correctly\n"
            "  2. Re-upload a complete copy to LightRAG",
        )

    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        # OLE2 Compound File — the legacy binary Word 97-2003 .doc format.
        return (
            f"The file at '{file_path}' is a legacy Word 97-2003 (.doc) "
            "document saved with a .docx extension. The .doc binary format is "
            "not a ZIP/OOXML package and cannot be parsed by the native engine.",
            convert_solution,
        )

    if head.startswith(b"{\\rtf"):
        return (
            f"The file at '{file_path}' is an RTF document saved with a .docx "
            "extension. RTF is not a ZIP/OOXML package.",
            convert_solution,
        )

    if head.startswith(b"%PDF"):
        return (
            f"The file at '{file_path}' is a PDF saved with a .docx extension. "
            "It is not a ZIP/OOXML package.",
            "  1. Convert the PDF to .docx, or upload it through a PDF-capable "
            "parser engine (e.g. mineru/docling)\n"
            "  2. Re-upload to LightRAG",
        )

    stripped = head.lstrip()
    if stripped.startswith(b"<"):
        # <?xml ...>, <html ...>, or Word 2003 "<w:wordDocument>" flat XML.
        return (
            f"The file at '{file_path}' is an HTML or XML document saved with a "
            ".docx extension, not a ZIP/OOXML package.",
            convert_solution,
        )

    if head.startswith(b"PK\x03\x04") and not zipfile.is_zipfile(file_path):
        # Has the ZIP local-file-header magic but the archive is unreadable.
        return (
            f"The file at '{file_path}' starts like a ZIP archive but is "
            "truncated or corrupt, so it cannot be opened as a DOCX package.",
            "  1. Check the original document opens correctly\n"
            "  2. Re-upload a complete, uncorrupted copy to LightRAG",
        )

    return (
        f"The file at '{file_path}' is not a valid DOCX (ZIP/OOXML) package. "
        "It is either corrupt or a different file format saved with a .docx "
        "extension.",
        convert_solution,
    )


def truncate_heading(heading_text: str, para_id: str = None) -> str:
    """
    Truncate heading if it exceeds MAX_HEADING_LENGTH.

    Args:
        heading_text: The heading text to check
        para_id: Optional paragraph ID for warning message

    Returns:
        str: Original heading if within limit, truncated heading with "..." if too long
    """
    if len(heading_text) > MAX_HEADING_LENGTH:
        truncated = heading_text[: MAX_HEADING_LENGTH - 3] + "..."
        location = f" (para_id: {para_id})" if para_id else ""
        print(
            f"Warning: Heading truncated (length {len(heading_text)} > max {MAX_HEADING_LENGTH}){location}: "
            f'"{truncated}"',
            file=sys.stderr,
        )
        return truncated
    return heading_text


def validate_heading_length(heading_text: str, para_id: str):
    """
    Validate that heading length does not exceed MAX_HEADING_LENGTH.

    Args:
        heading_text: The heading text to validate
        para_id: The paragraph ID for error reporting

    Raises:
        DocxContentError: if heading exceeds maximum length
    """
    if len(heading_text) > MAX_HEADING_LENGTH:
        preview = (
            heading_text[:100] + "..." if len(heading_text) > 100 else heading_text
        )
        raise DocxContentError(
            format_error(
                f"Heading too long ({len(heading_text)} characters, max {MAX_HEADING_LENGTH})",
                f"The following heading exceeds the maximum allowed length:\n\n{preview}\n\n"
                f"Location(para_id): {para_id}\n"
                f"Actual length: {len(heading_text)} characters",
                "  1. Open the document in Microsoft Word\n"
                f"  2. Shorten this heading to {MAX_HEADING_LENGTH} characters or less\n"
                "  3. Re-upload it to LightRAG",
            )
        )


def find_first_valid_para_id(para_ids: list) -> str | None:
    """
    Find the first valid paraId in a 2D array of paraIds.

    Args:
        para_ids: 2D list of paraIds from table cells

    Returns:
        First non-None paraId found, or None when every cell lacks a paraId.
        Callers must tolerate ``None`` and treat it as a tracking gap rather
        than a fatal error (legacy / non-Word docx authors omit ``w14:paraId``
        attributes and we want to keep parsing).
    """
    for row in para_ids:
        for para_id in row:
            if para_id:
                return para_id
    return None


def find_last_valid_para_id(para_ids: list) -> str | None:
    """
    Find the last valid paraId in a 2D array of paraIds.

    Returns the last non-None paraId, falling back to the first valid one
    when reverse-iteration does not yield anything (single-paraId tables),
    and finally ``None`` when every cell lacks a paraId.
    """
    for row in reversed(para_ids):
        for para_id in reversed(row):
            if para_id:
                return para_id

    return find_first_valid_para_id(para_ids)


def _table_has_any_paraid(para_ids: list) -> bool:
    """True when at least one cell in the 2D paraId grid carries an id."""
    return find_first_valid_para_id(para_ids) is not None


def extract_para_id(para_element) -> str:
    """
    Extract w14:paraId attribute from paragraph element.

    Args:
        para_element: lxml paragraph element

    Returns:
        8-character hex paraId, or ``None`` when the paragraph carries no
        ``w14:paraId`` attribute (legacy / non-Word docx authors). Callers
        propagate the ``None`` upward — the LightRAG adapter counts these
        and surfaces a single warning per document.
    """
    return para_element.get(
        "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
    )


def parse_styles_outline_levels(docx_path: str) -> dict:
    """
    Parse styles.xml to extract outlineLvl definitions for each style,
    following style inheritance chain (basedOn).

    Args:
        docx_path: Path to DOCX file

    Returns:
        dict: styleId -> outlineLvl (0-8 for headings, 9 for body text)
    """
    import zipfile

    try:
        from defusedxml import ElementTree as ET
    except ImportError:
        from xml.etree import ElementTree as ET

    styles_outline = {}  # styleId -> outlineLvl (directly defined)
    style_based_on = {}  # styleId -> parent styleId

    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            if "word/styles.xml" not in zf.namelist():
                return styles_outline

            tree = ET.parse(zf.open("word/styles.xml"))
            root = tree.getroot()

            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

            # First pass: collect outlineLvl and basedOn for all styles
            for style in root.findall(f".//{{{ns}}}style"):
                style_id = style.get(f"{{{ns}}}styleId")
                if not style_id:
                    continue

                # Check for basedOn (style inheritance)
                based_on = style.find(f"{{{ns}}}basedOn")
                if based_on is not None:
                    parent_id = based_on.get(f"{{{ns}}}val")
                    if parent_id:
                        style_based_on[style_id] = parent_id

                # Check for outlineLvl in style's pPr
                pPr = style.find(f"{{{ns}}}pPr")
                if pPr is not None:
                    outline_lvl_elem = pPr.find(f"{{{ns}}}outlineLvl")
                    if outline_lvl_elem is not None:
                        level = int(outline_lvl_elem.get(f"{{{ns}}}val"))
                        styles_outline[style_id] = level

            # Second pass: resolve inheritance chain for styles without direct outlineLvl
            def get_outline_level(style_id: str, visited: set = None) -> int:
                if visited is None:
                    visited = set()
                if style_id in visited:
                    return None  # Prevent circular references
                visited.add(style_id)

                # If this style directly defines outlineLvl, return it
                if style_id in styles_outline:
                    return styles_outline[style_id]

                # Otherwise check parent style
                if style_id in style_based_on:
                    parent_id = style_based_on[style_id]
                    return get_outline_level(parent_id, visited)

                return None

            # Fill in missing outlineLvl from inheritance chain
            all_style_ids = set(styles_outline.keys()) | set(style_based_on.keys())
            for style_id in all_style_ids:
                if style_id not in styles_outline:
                    level = get_outline_level(style_id)
                    if level is not None:
                        styles_outline[style_id] = level
    except Exception:
        # Silently ignore parsing errors
        pass

    return styles_outline


def get_heading_level(para_element, styles_outline_map: dict) -> int:
    """
    Get heading level from paragraph, checking both direct format and style.

    Priority: paragraph outlineLvl > style outlineLvl

    Args:
        para_element: lxml paragraph element
        styles_outline_map: dict of styleId -> outlineLvl from styles.xml

    Returns:
        int: 0-8 for heading levels (0=level 1, 1=level 2, etc.), None for non-heading
    """
    # 1. Check paragraph direct format
    pPr = para_element.find(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr"
    )
    if pPr is not None:
        outline_elem = pPr.find(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}outlineLvl"
        )
        if outline_elem is not None:
            level = int(
                outline_elem.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                )
            )
            # Only 0-8 are true heading levels (9 is body text)
            if level < 9:
                return level
            else:
                return None  # Level 9 is body text

    # 2. Check style definition's outlineLvl
    if pPr is not None:
        pStyle_elem = pPr.find(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pStyle"
        )
        if pStyle_elem is not None:
            style_id = pStyle_elem.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
            )
            if style_id and style_id in styles_outline_map:
                level = styles_outline_map[style_id]
                if level < 9:
                    return level
                else:
                    return None

    return None


def extract_text_from_run(
    run,
    ns: dict,
    drawing_context: DrawingExtractionContext = None,
) -> str:
    """
    Extract text from a run element, preserving superscript/subscript with markup.

    Converts Word formatting to HTML-like tags:
    - Superscript: <sup>text</sup>
    - Subscript: <sub>text</sub>
    - Normal text: unchanged

    Args:
        run: lxml run element (w:r)
        ns: XML namespace dictionary

    Returns:
        Text string with <sup>/<sub> markup for formatted portions
    """
    text = ""

    # Check for vertAlign in rPr (superscript/subscript)
    vert_align = None
    rPr = run.find("w:rPr", ns)
    if rPr is not None:
        vert_elem = rPr.find("w:vertAlign", ns)
        if vert_elem is not None:
            vert_align = vert_elem.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
            )

    # Extract text content from run children
    for child in run:
        tag = child.tag.split("}")[-1]  # Remove namespace
        if tag == "t" and child.text:
            text += child.text
        elif tag == "tab":
            text += "\t"
        elif tag == "br":
            # Handle line breaks - textWrapping or no type = soft line break
            br_type = child.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type"
            )
            if br_type in (None, "textWrapping"):
                text += "\n"
            # Skip page and column breaks (layout elements)
        elif tag == "drawing":
            text += extract_drawing_placeholder_from_element(
                child,
                context=drawing_context,
                include_extended_attrs=True,
            )
        elif tag in ("pict", "object"):
            text += extract_vml_image_placeholder_from_element(
                child,
                context=drawing_context,
                include_extended_attrs=True,
            )

    # Apply superscript/subscript markup if needed
    if text and vert_align == "superscript":
        return f"<sup>{text}</sup>"
    elif text and vert_align == "subscript":
        return f"<sub>{text}</sub>"

    return text


def extract_paragraph_content(
    element,
    ns,
    drawing_context: DrawingExtractionContext = None,
) -> str:
    """
    Extract text and equations from a paragraph element in document order.

    Handles w:r (text runs), m:oMath (inline equations), and m:oMathPara
    (block equations). Recurses into container elements (e.g., w:hyperlink,
    w:ins, w:sdt, w:fldSimple, w:smartTag) to avoid dropping content.

    Args:
        element: lxml paragraph element (w:p)
        ns: XML namespace dictionary

    Returns:
        Text string with equations wrapped in <equation> tags
    """
    parts = []

    def append_from(node) -> None:
        tag = node.tag.split("}")[-1]
        # Drop tracked-change deletions (w:del/w:moveFrom) and comment markers
        # (w:commentRangeStart/End, w:commentReference, w:annotationRef) so the
        # output only contains the final revised text without annotation glyphs.
        if tag in _SKIP_PARAGRAPH_TAGS:
            return
        if tag == "r":
            parts.append(
                extract_text_from_run(node, ns, drawing_context=drawing_context)
            )
            return
        if tag == "oMath":
            from .omml import convert_omml_to_latex

            latex = convert_omml_to_latex(node)
            if latex:
                parts.append(f"<equation>{latex}</equation>")
            return
        if tag == "oMathPara":
            from .omml import convert_omml_to_latex

            for omath in node:
                if omath.tag.split("}")[-1] == "oMath":
                    latex = convert_omml_to_latex(omath)
                    if latex:
                        parts.append(f"<equation>{latex}</equation>")
            return
        for child in node:
            append_from(child)

    for child in element:
        append_from(child)

    return "".join(parts)


def _is_table_empty(rows: list) -> bool:
    """Return True iff every cell in ``rows`` is whitespace-only."""
    return all(not (cell or "").strip() for row in rows for cell in row)


def _collect_table_headers(paragraphs: list) -> list:
    """Collect per-table cross-page header rows from ``is_table`` paragraphs.

    The returned list is aligned 1:1 with the order of ``<table>`` placeholder
    tags emitted into the block's content; entries are either the list of
    header rows captured from ``w:tblHeader`` or ``None`` when the table has
    no cross-page repeating header.
    """
    return [p.get("_table_header") for p in paragraphs if p.get("is_table")]


def _build_unsplit_block(
    heading: str,
    paragraphs: list,
    parent_headings: list,
    level: int,
    is_title_block: bool = False,
) -> dict:
    """Build a single block from paragraphs without size-based splitting."""
    last_para = paragraphs[-1]
    block = {
        "uuid": paragraphs[0]["para_id"],
        "uuid_end": last_para.get("para_id_end") or last_para.get("para_id"),
        "heading": heading,
        "content": "\n".join(p["text"] for p in paragraphs),
        "type": "text",
        "parent_headings": parent_headings,
        "level": level,
    }
    if is_title_block:
        block["is_title_block"] = True
    table_headers = _collect_table_headers(paragraphs)
    if table_headers:
        block["table_headers"] = table_headers
    return block


def _flush_current_block(
    blocks: list,
    heading: str,
    paragraphs: list,
    parent_headings: list,
    level: int,
    is_title_block: bool = False,
) -> None:
    """Flush accumulated paragraphs into a single heading-scoped block.

    The native parser performs only heading-driven structural splitting; block
    sizing (long-block anchor splitting, table row splitting, small-block
    merging) is the downstream paragraph-semantic chunker's responsibility.
    """
    if not paragraphs:
        return

    blocks.append(
        _build_unsplit_block(
            heading, paragraphs, parent_headings, level, is_title_block
        )
    )


@dataclass
class ParagraphRecord:
    """Engine-neutral snapshot of one body element (spec §3.1 pass1).

    The single read pass emits these; the assembly passes (legacy or smart)
    consume them. Records hold only normalized feature fields — never lxml
    node references — and are transient inside ``extract_docx_blocks``: they
    are not persisted and never enter the IR.

    The ``kind`` values ``empty_para`` / ``empty_table`` / ``section_break``
    exist so the smart pass can see document-shape evidence (blank-line and
    section boundaries) that the legacy assembler simply skips.
    """

    kind: str  # "para" | "table" | "empty_para" | "empty_table" | "section_break"
    # --- fields the legacy (smart-off) assembler consumes -------------------
    text: str = ""  # para: post-policy label+text; table: "<table>…</table>"
    para_id: str | None = None
    para_id_end: str | None = None  # table only
    outline_level: int | None = None  # post-policy, 0-based
    demoted_body_text: str | None = None  # oversize soft-break remainder
    table_header_rows: list | None = None
    ends_section: bool = False  # paragraph-level w:pPr/w:sectPr
    # --- smart-heading features (populated only when smart is enabled) ------
    full_text_raw: str | None = None  # pre-policy label+text (keeps "\n")
    label: str = ""  # auto-numbering label, if any
    outline_level_raw: int | None = None  # pre-policy outline level
    style_id: str | None = None
    font_size_pt: float | None = None  # char-weighted dominant (0.5pt grid)
    # Visible source-text char count (w:t only; excludes auto-numbering labels
    # and <sup>/<equation>/<drawing>/<table> placeholder markup). Feeds FS_base
    # char weighting so generated text cannot skew the body-size mode (§2.2.2).
    # None means "not computed" (smart-off / manually built records).
    visible_char_count: int | None = None
    first_line_font_size_pt: float | None = None  # before first soft break
    all_bold: bool = False
    alignment: str | None = None  # resolved w:jc or None
    page_break_before: bool = False  # w:pPr/w:pageBreakBefore only
    has_page_break_run: bool = False  # w:br type="page" run inside this para
    has_leading_page_break_run: bool = False  # page-break run before any text
    has_nonleading_page_break_run: bool = False  # page-break run AFTER text
    is_toc_field: bool = False  # TOC field instruction inside the paragraph
    is_toc_link: bool = False  # hyperlink to a _Toc bookmark
    size_trace_failed: bool = False  # font-size cascade found nothing (CB5)
    # Table only (smart-only): per-row PHYSICAL cells as
    # (text, effective_size_pt, has_outline) tuples — one entry per w:tc, so a
    # gridSpan-merged full-width row is ONE cell. Feeds the §2.2.4 title-block
    # table channel; None when smart is off.
    table_cell_features: list | None = None


def _read_document_records(
    doc,
    resolver: NumberingResolver,
    styles_outline: dict,
    drawing_context: DrawingExtractionContext,
    parse_warnings: dict | None,
    *,
    style_attributes=None,
) -> list:
    """Single read pass over the document body (spec §3.1 read step).

    Shared by BOTH assembly modes, and physically unrepeatable: the numbering
    resolver is stateful and the drawing/table extractors export asset bytes,
    so this pass must run exactly once per parse. Every resolver/extractor
    side effect below keeps the exact order and trigger conditions of the
    pre-split loop (characterization-locked):

    - empty paragraphs never call ``resolver.get_label`` (a numbered empty
      paragraph must not advance counters) and never check the paragraph-level
      sectPr;
    - tables reset numbering tracking before extraction, again when skipped
      as empty, and after emission;
    - the oversize-heading policy (soft-break split / body demotion) applies
      here so its warning counters and stderr notices fire per document order.

    ``style_attributes`` (a ``smart_heading.features.StyleAttributes``) turns
    on collection of the smart-only physical features; ``None`` skips them.
    """
    records: list[ParagraphRecord] = []
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    }
    if style_attributes is not None:
        from lightrag.parser.docx.smart_heading.features import (
            extract_paragraph_physical_features,
            first_line_size_half_points,
            half_points_to_pt,
        )

    body = doc._element.body

    for element in body:
        tag = element.tag.split("}")[-1]  # Remove namespace

        if tag == "sectPr":  # Document-level section break
            resolver.reset_tracking_state()
            records.append(ParagraphRecord(kind="section_break"))
            continue

        if tag == "p":  # Paragraph
            para_text = extract_paragraph_content(
                element,
                ns,
                drawing_context=drawing_context,
            )
            para_text = para_text.strip()
            if not para_text:
                # A blank line still carries page/section-boundary evidence
                # the smart title-block pass reads (page breaks often live on
                # empty paragraphs). No visible text, so any w:br page here
                # is a LEADING break (never nonleading).
                rec = ParagraphRecord(kind="empty_para")
                w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                for br in element.iter(f"{w_ns}br"):
                    if br.get(f"{w_ns}type") == "page":
                        rec.has_page_break_run = True
                        rec.has_leading_page_break_run = True
                        break
                pPr = element.find(f"{w_ns}pPr")
                if pPr is not None:
                    pbb = pPr.find(f"{w_ns}pageBreakBefore")
                    # OOXML on/off: absent val means on.
                    if pbb is not None and pbb.get(f"{w_ns}val") not in (
                        "0",
                        "false",
                        "none",
                    ):
                        rec.page_break_before = True
                    if pPr.find(f"{w_ns}sectPr") is not None:
                        # Section boundary on a blank paragraph: same
                        # numbering-reset semantics as a non-empty one.
                        resolver.reset_tracking_state()
                        rec.ends_section = True
                records.append(rec)
                continue

            # Get numbering label using our resolver
            label = resolver.get_label(element)
            full_text = f"{label} {para_text}".strip() if label else para_text

            outline_level = get_heading_level(element, styles_outline)
            outline_level_raw = outline_level
            full_text_raw = full_text

            # A "heading" longer than MAX_HEADING_LENGTH is not a real heading.
            # The common cause (WPS/Word): the author set an outline level on a
            # paragraph but typed the body with soft line breaks (Shift+Enter →
            # <w:br/> → '\n') instead of starting a new paragraph, so heading
            # text + body live in one <w:p>. Split at the first soft break: the
            # first line stays the heading, the remainder becomes body text. If
            # there is no usable soft break (a genuine single-line over-long
            # heading), demote the whole paragraph to body text. Either way we
            # avoid crashing via validate_heading_length() and never drop content.
            demoted_body_text = None
            if outline_level is not None and len(full_text) > MAX_HEADING_LENGTH:
                head, sep, rest = full_text.partition("\n")
                if sep and len(head) <= MAX_HEADING_LENGTH:
                    full_text = head
                    demoted_body_text = rest.strip() or None
                    if parse_warnings is not None:
                        parse_warnings["heading_softbreak_split_count"] = (
                            parse_warnings.get("heading_softbreak_split_count", 0) + 1
                        )
                    print(
                        f"Warning: heading paragraph exceeded {MAX_HEADING_LENGTH} "
                        "chars; split at soft line break — kept first line as "
                        "heading, rest as body.",
                        file=sys.stderr,
                    )
                else:
                    outline_level = None
                    if parse_warnings is not None:
                        parse_warnings["demoted_oversize_heading_count"] = (
                            parse_warnings.get("demoted_oversize_heading_count", 0) + 1
                        )
                    print(
                        f"Warning: paragraph has outline level but is "
                        f"{len(full_text)} chars (> {MAX_HEADING_LENGTH}); treating "
                        "as body text, not a heading.",
                        file=sys.stderr,
                    )

            para_id = extract_para_id(element)
            if parse_warnings is not None and not para_id:
                parse_warnings["missing_paraid_count"] = (
                    parse_warnings.get("missing_paraid_count", 0) + 1
                )

            rec = ParagraphRecord(
                kind="para",
                text=full_text,
                para_id=para_id,
                outline_level=outline_level,
                demoted_body_text=demoted_body_text,
            )

            if style_attributes is not None:
                phys = extract_paragraph_physical_features(element, style_attributes)
                rec.full_text_raw = full_text_raw
                rec.label = label or ""
                rec.outline_level_raw = outline_level_raw
                rec.style_id = phys.style_id
                rec.font_size_pt = phys.font_size_pt
                rec.visible_char_count = phys.visible_char_count
                rec.first_line_font_size_pt = half_points_to_pt(
                    first_line_size_half_points(phys.run_features)
                )
                rec.all_bold = phys.all_bold
                rec.alignment = phys.alignment
                rec.page_break_before = phys.page_break_before
                rec.has_page_break_run = phys.has_page_break_run
                rec.has_leading_page_break_run = phys.has_leading_page_break_run
                rec.has_nonleading_page_break_run = phys.has_nonleading_page_break_run
                rec.is_toc_field = phys.is_toc_field
                rec.is_toc_link = phys.is_toc_link
                rec.size_trace_failed = phys.size_trace_failed

            # Check for paragraph-level section break (after processing paragraph)
            # sectPr in pPr means this paragraph ends a section
            pPr = element.find(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr"
            )
            if pPr is not None:
                sectPr = pPr.find(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr"
                )
                if sectPr is not None:
                    # Section break after this paragraph - reset tracking
                    resolver.reset_tracking_state()
                    rec.ends_section = True

            records.append(rec)

        elif tag == "tbl":  # Table
            # Reset numbering tracking before table (table start boundary)
            resolver.reset_tracking_state()

            # Directly create Table object from XML element to avoid index mismatch
            # (doc.tables may have different order due to nested tables)
            from docx.table import Table

            table = Table(element, doc)
            table_metadata = TableExtractor.extract_with_metadata(
                table,
                numbering_resolver=resolver,
                drawing_context=drawing_context,
                style_attributes=style_attributes,
                styles_outline=styles_outline,
            )

            table_rows = table_metadata["rows"]
            para_ids = table_metadata["para_ids"]
            para_ids_end = table_metadata["para_ids_end"]  # Last paraId in each cell
            header_indices = table_metadata["header_indices"]

            # Skip tables whose every cell is whitespace-only — otherwise an
            # empty `<table>[[""]]</table>` placeholder would leak into block
            # content and a useless IRTable would appear in tables.json.
            if _is_table_empty(table_rows):
                resolver.reset_tracking_state()
                records.append(ParagraphRecord(kind="empty_table"))
                continue

            # Count tables whose cells carry no w14:paraId. Legacy / non-Word
            # docx authors omit these attributes; we no longer fail-fast, but
            # the adapter surfaces a single warning so the user knows the edit
            # range hints will be missing for these tables.
            if parse_warnings is not None and not _table_has_any_paraid(para_ids):
                parse_warnings["missing_paraid_count"] = (
                    parse_warnings.get("missing_paraid_count", 0) + 1
                )

            # Convert table to JSON
            table_json = json.dumps(table_rows, ensure_ascii=False)

            # Extract cross-page repeating header rows (w:tblHeader) once per
            # table so we can surface them to the sidecar via the block-level
            # ``table_headers`` list.
            header_rows = []
            if header_indices:
                header_rows = [
                    table_rows[idx] for idx in header_indices if idx < len(table_rows)
                ]
            header_rows_or_none = header_rows if header_rows else None

            # Emit the whole table as a single <table> placeholder. Token-based
            # table row splitting is the downstream chunker's responsibility.
            # Use first valid paraId from table, and last valid paraId (from
            # para_ids_end) for uuid_end.
            records.append(
                ParagraphRecord(
                    kind="table",
                    text=f"<table>{table_json}</table>",
                    para_id=find_first_valid_para_id(para_ids),
                    para_id_end=find_last_valid_para_id(para_ids_end),
                    table_header_rows=header_rows_or_none,
                    table_cell_features=table_metadata.get("cell_features"),
                )
            )

            # Reset numbering tracking after table (table end boundary)
            resolver.reset_tracking_state()

    return records


def _assemble_blocks_legacy(
    records: list,
    parse_warnings: dict | None,
    parse_metadata: dict | None,
) -> list:
    """Assemble blocks from records exactly as the pre-split loop did.

    This is the smart-off (baseline) assembly step: the block-building half of
    the original single loop, moved verbatim with lxml locals replaced by
    record fields. Behavior is characterization-locked — do not "improve" it.
    """
    blocks = []
    current_heading = "Preface/Uncategorized"
    current_heading_level = 1  # Default level for "Preface/Uncategorized"
    current_heading_stack = {}  # {level: heading_text} - Use dict to correctly track heading hierarchy
    current_parent_headings = []  # Parent headings for current block
    current_paragraphs = []  # Track paragraphs with metadata for splitting
    first_heading_recorded = (
        False  # Track whether the document's first heading has been captured
    )

    for rec in records:
        if rec.kind in ("empty_para", "empty_table", "section_break"):
            continue

        if rec.kind == "table":
            current_paragraphs.append(
                {
                    "text": rec.text,
                    "para_id": rec.para_id,
                    "para_id_end": rec.para_id_end,  # Store end paraId for uuid_end calculation
                    "is_table": True,
                    "_table_header": rec.table_header_rows,
                }
            )
            continue

        # rec.kind == "para"
        if rec.outline_level is not None:
            # This is a heading (outline level 0-8)
            # Convert 0-based to 1-based level
            level = rec.outline_level + 1
            heading_para_id = rec.para_id

            # Validate heading length
            validate_heading_length(rec.text, heading_para_id)

            # Truncate heading if needed before storing
            truncated_text = truncate_heading(rec.text, heading_para_id)
            clean_heading_text = strip_heading_markdown_prefix(truncated_text)

            # Record the document's first heading (any level) for meta.doc_title.
            if not first_heading_recorded:
                if parse_metadata is not None:
                    parse_metadata["first_heading"] = clean_heading_text
                first_heading_recorded = True

            # Every recognized heading starts its own block. Always flush the
            # accumulated paragraphs so a heading with no body becomes a
            # standalone block whose content is just the heading text,
            # instead of being folded into the next heading's block.
            if current_paragraphs:
                _flush_current_block(
                    blocks,
                    current_heading,
                    current_paragraphs,
                    current_parent_headings,
                    current_heading_level,
                )

                # Reset for new block
                current_paragraphs = []

            # Add heading to current_paragraphs. The content line gets
            # a markdown ``#`` prefix (capped at 6) via
            # render_heading_line; ``clean_heading_text`` is kept
            # for the heading field / stack / parent_headings below.
            current_paragraphs.append(
                {
                    "text": render_heading_line(level, truncated_text),
                    "para_id": heading_para_id,
                    "is_table": False,
                }
            )

            # Update current_heading and parent_headings for the FIRST heading in a block
            # (when current_paragraphs just had this heading added as its first element)
            if len(current_paragraphs) == 1:
                current_heading = clean_heading_text
                current_heading_level = level  # Only set level when setting heading
                # Parent headings = all headings from levels strictly less than current level
                # Sort by level to maintain hierarchy order
                current_parent_headings = [
                    current_heading_stack[lvl]
                    for lvl in sorted(current_heading_stack.keys())
                    if lvl < level
                ]

            # Update heading stack: remove current level and all lower levels, then add current
            current_heading_stack = {
                k: v for k, v in current_heading_stack.items() if k < level
            }
            current_heading_stack[level] = clean_heading_text

            # Carry the body text that followed a soft break in an over-long
            # heading paragraph as a regular body paragraph in the same block.
            if rec.demoted_body_text:
                current_paragraphs.append(
                    {
                        "text": rec.demoted_body_text,
                        "para_id": heading_para_id,
                        "is_table": False,
                    }
                )
        else:
            # Regular paragraph content
            current_paragraphs.append(
                {"text": rec.text, "para_id": rec.para_id, "is_table": False}
            )

    # Save final block
    _flush_current_block(
        blocks,
        current_heading,
        current_paragraphs,
        current_parent_headings,
        current_heading_level,
    )

    return blocks


def extract_docx_blocks(
    file_path: str,
    drawing_context: DrawingExtractionContext = None,
    parse_warnings: dict | None = None,
    parse_metadata: dict | None = None,
    *,
    smart_heading_runtime=None,
) -> list:
    """
    Extract heading-scoped text blocks from a DOCX file.

    Uses python-docx with a custom numbering resolver to:
    1. Capture automatic numbering (list labels)
    2. Split the document into one block per heading (structural splitting)
    3. Convert tables to JSON (2D array) and emit them as <table> placeholders
    4. Preserve superscript/subscript formatting with <sup>/<sub> markup

    Block sizing — long-block anchor splitting, table row splitting, and
    small-block merging — is intentionally NOT done here; it is the downstream
    paragraph-semantic chunker's responsibility. Blocks emitted here may
    therefore be arbitrarily large.

    Structured as a single read pass (``_read_document_records``) shared by
    both assembly modes, then a mode-specific assembly step; smart-off runs
    ``_assemble_blocks_legacy`` whose output is characterization-locked to the
    pre-split behavior.

    Args:
        file_path: Path to the DOCX file
        parse_warnings: Optional out-dict that this function mutates with
            non-fatal warnings observed during parsing. Currently used for
            ``missing_paraid_count`` — incremented once per body-level
            paragraph (heading or text) that lacks a ``w14:paraId`` and once
            per table whose every cell lacks one. Callers (the LightRAG
            adapter / debug CLI) read this to surface a one-line warning per
            document instead of crashing.
        parse_metadata: Optional out-dict that this function mutates with
            document-level metadata derived during parsing. Currently used
            for ``first_heading`` — the text of the first heading encountered
            in document order (regardless of level). Used by the LightRAG
            adapter to populate ``meta.doc_title`` in ``.blocks.jsonl``.
        smart_heading_runtime: Optional ``NativeExtractRuntime``. When its
            engine params enable ``smart_heading``, the read pass collects the
            physical features the smart algorithm consumes. (The smart
            assembly itself lands with the algorithm commits; until then the
            output is identical to baseline.)

    Returns:
        List of block dictionaries with heading, content, type, and metadata
    """
    try:
        doc = Document(file_path)
    except PackageNotFoundError as exc:
        # python-docx surfaces a misleading "Package not found at '...'" for any
        # file it cannot open as a ZIP/OOXML package — including files that
        # exist but are corrupt or a different format wearing a .docx extension.
        # Diagnose the real cause from the magic bytes and raise a DocxContentError
        # (a ValueError) so the pipeline's per-document handler marks just this
        # document FAILED with an accurate, actionable message.
        details, solution = _diagnose_invalid_docx(file_path)
        raise DocxContentError(
            format_error("File is not a valid DOCX document", details, solution)
        ) from exc
    resolver = NumberingResolver(file_path)
    styles_outline = parse_styles_outline_levels(file_path)

    smart_enabled = bool(
        smart_heading_runtime is not None
        and getattr(smart_heading_runtime, "engine_params", None)
        and smart_heading_runtime.engine_params.get("smart_heading")
    )

    style_attributes = None
    if smart_enabled:
        # Function-local import: the smart-off path must stay free of the
        # smart_heading package (and its optional spaCy dependency).
        from lightrag.parser.docx.smart_heading.features import (
            parse_styles_attributes,
        )

        style_attributes = parse_styles_attributes(file_path, warnings=parse_warnings)

    records = _read_document_records(
        doc,
        resolver,
        styles_outline,
        drawing_context,
        parse_warnings,
        style_attributes=style_attributes,
    )

    if not smart_enabled:
        return _assemble_blocks_legacy(records, parse_warnings, parse_metadata)

    # --- smart path (§2.2 pipeline + §2.3 landing guardrails) ---------------
    from lightrag.parser.docx.smart_heading import guardrails as _guards
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    warnings_sink = parse_warnings if parse_warnings is not None else {}
    llm_judge = getattr(smart_heading_runtime, "llm_invoke", None)

    smart_result = run_smart_heading(
        records, llm_judge=llm_judge, warnings=warnings_sink
    )
    if smart_result is None:
        # CB4 whole-document gate: too short for smart — baseline output,
        # zero LLM calls, TOC untouched.
        warnings_sink["smart_skipped_short_document"] = 1
        return _assemble_blocks_legacy(records, parse_warnings, parse_metadata)

    smart_blocks = _assemble_blocks_smart(
        records, smart_result, parse_warnings, parse_metadata
    )

    # Landing guardrails: any violation abandons the smart output for THIS
    # document and re-runs the baseline assembly (§2.3.2). The shadow
    # baseline is synthesized from the SAME pass1 records in memory — no
    # extra IO, no re-parse.
    shadow_meta: dict = {}
    baseline_blocks = _assemble_blocks_legacy(records, None, shadow_meta)
    all_decisions = list(smart_result.decisions.values())
    i1_missing = _guards.verify_content_preservation(
        records,
        smart_blocks,
        toc_indices=smart_result.toc_indices,
        # Retained TOC body + the ellipsis are output lines that come from no
        # record; subtract them so an injected copy can't mask a lost source
        # line (§2.3 TOC retention).
        ignored_output_texts=[
            *smart_result.toc_kept_text.values(),
            *(
                [_guards.TOC_ELLIPSIS]
                if smart_result.toc_ellipsis_anchor is not None
                else []
            ),
        ],
    )
    i2_violations = _guards.verify_baseline_heading_retention(records, all_decisions)
    i3_violations = _guards.verify_anchor_semantics(all_decisions)
    length_ok = _guards.smart_output_length_ok(smart_blocks, baseline_blocks)
    if i1_missing or i2_violations or i3_violations or not length_ok:
        # §2.3.2 mandates an ERROR *log* (not a bare stderr print) so the
        # event is visible to the repo's caplog-based assertions and any
        # structured log routing.
        logger.error(
            "[smart_heading] guardrail violation "
            "(I1 missing=%d, I2=%s, I3=%s, length_ok=%s); "
            "falling back to baseline output for this document.",
            len(i1_missing),
            i2_violations,
            i3_violations,
            length_ok,
        )
        warnings_sink["smart_fallback_baseline"] = 1
        if parse_metadata is not None:
            parse_metadata.pop("first_heading", None)
            # Baseline output ships baseline doc_title semantics too — the
            # smart-only explicit key must not survive the fallback.
            parse_metadata.pop("doc_title", None)
        return _assemble_blocks_legacy(records, parse_warnings, parse_metadata)

    # These are content claims, so they land only now that the smart output is
    # accepted — the CB4 skip and the guardrail fallback above both ship
    # baseline output with the TOC intact. Line-level counts are exact (the keep
    # budget is per visible line). ``smart_toc_removed_paragraphs`` keeps its
    # name to ease migration but its meaning NARROWS to "fully-dropped visible
    # TOC paragraphs" (a straddling, partly-kept record counts as kept, not
    # removed); the precise removal count is ``smart_toc_removed_lines``. All
    # three counts come from guardrails.plan_toc_output — the single visibility
    # / disposition source — never re-derived here.
    if smart_result.toc_removed_lines:
        warnings_sink["smart_toc_removed_lines"] = smart_result.toc_removed_lines
    if smart_result.toc_kept_lines:
        warnings_sink["smart_toc_kept_lines"] = smart_result.toc_kept_lines
    if smart_result.toc_fully_removed_paragraphs:
        warnings_sink["smart_toc_removed_paragraphs"] = (
            smart_result.toc_fully_removed_paragraphs
        )

    audit = dict(smart_result.audit)
    audit["shadow_diff"] = _guards.shadow_baseline_diff(smart_blocks, baseline_blocks)
    audit["fallback_sub_documents"] = smart_result.fallback_sub_docs
    if parse_metadata is not None:
        parse_metadata["smart_audit"] = audit
    return smart_blocks


def _assemble_blocks_smart(
    records: list,
    smart_result,
    parse_warnings: dict | None,
    parse_metadata: dict | None,
) -> list:
    """Smart (pass3) assembly: rebuild blocks from records + HeadingDecisions.

    Same state machine as the legacy assembler; the differences are exactly
    the smart products — a detected TOC keeps its first few visible lines as
    body and collapses the rest to a single ``……`` (§2.3 TOC retention), a
    title block seeds a level-0 block flagged ``is_title_block`` (heading = the
    plain main title) that then OWNS the following body up to the next heading
    / title / EOF (like an ordinary section heading owns its body),
    heading/level come from the decisions (merged texts, demotions to
    ``full_text_raw``), and ``parent_headings`` chains rebuild naturally from
    the final levels.
    """
    # Module-level function: ``_guards`` is a local of extract_docx_blocks and
    # out of scope here, so import the shared ellipsis constant locally (mirrors
    # the flatten_heading_line import below; keeps smart-off from loading it).
    from .smart_heading.guardrails import TOC_ELLIPSIS

    decisions = smart_result.decisions
    toc_indices = smart_result.toc_indices
    kept_toc_text = smart_result.toc_kept_text
    toc_ellipsis_anchor = smart_result.toc_ellipsis_anchor

    # Non-lead members of every title block are emitted as part of their
    # lead record's composite block; their standalone rows must be skipped.
    # (Their sentinel decisions are NOT ``is_title_block``, so checking the
    # decision flag on each record misses them and double-emits the text —
    # review C3.) Populated lazily when a title lead is SUCCESSFULLY seeded
    # (below): a title verdict rejected for empty members must not silently
    # skip its members. A title lead's own index is always < its non-lead
    # member indices (single = (start,), multi = ascending index_map, table =
    # ascending member_tables), so seeding the lead before its members are
    # reached keeps the skip set correct.
    title_member_skip: set[int] = set()

    blocks: list = []
    current_heading = "Preface/Uncategorized"
    current_heading_level = 1
    current_heading_stack: dict[int, str] = {}
    current_parent_headings: list = []
    current_paragraphs: list = []
    current_is_title_block = False
    first_heading_recorded = False

    if parse_metadata is not None:
        # Smart mode: the LLM title block is the ONLY doc_title source — no
        # title block means an explicitly EMPTY title ("前言"-style first
        # headings must not masquerade as the document title). The dedicated
        # key overrides ir_builder's first_heading/stem fallback chain;
        # first_heading below keeps its legacy any-heading semantics.
        parse_metadata["doc_title"] = smart_result.doc_title or ""
    if smart_result.doc_title and parse_metadata is not None:
        parse_metadata["first_heading"] = smart_result.doc_title
        first_heading_recorded = True

    def _flush():
        nonlocal current_paragraphs
        if current_paragraphs:
            _flush_current_block(
                blocks,
                current_heading,
                current_paragraphs,
                current_parent_headings,
                current_heading_level,
                current_is_title_block,
            )
            current_paragraphs = []

    for i, rec in enumerate(records):
        if i in toc_indices:
            # §2.3 TOC retention: a TOC line is KEPT (first visible lines as
            # body), the ellipsis anchor also emits one ``……``, and the rest
            # are DROPPED. Every branch ``continue``s before any heading /
            # title / table branch, so a retained line is never a heading.
            kept = kept_toc_text.get(i)
            if kept:
                current_paragraphs.append(
                    {"text": kept, "para_id": rec.para_id, "is_table": False}
                )
            if i == toc_ellipsis_anchor:
                current_paragraphs.append(
                    {"text": TOC_ELLIPSIS, "para_id": rec.para_id, "is_table": False}
                )
            continue
        if i in title_member_skip:
            continue  # non-lead title-block member, emitted with the lead
        if rec.kind in ("empty_para", "empty_table", "section_break"):
            continue

        d = decisions.get(i)
        if d is not None and d.absorbed:
            continue  # text already merged into a preceding heading
        # The title-block lead branch must run BEFORE the plain-table branch:
        # a §2.2.4 TABLE-channel lead is itself a table record — emitting it
        # as body content would discard the composed heading and drop every
        # non-lead member via title_member_skip.
        if d is not None and d.is_title_block and i == d.member_indices[0]:
            _flush()
            member_paras = []
            for m in d.member_indices:
                mrec = records[m]
                if mrec.kind == "para":
                    member_paras.append(
                        {
                            "text": mrec.text,
                            "para_id": mrec.para_id,
                            "is_table": False,
                        }
                    )
                elif mrec.kind == "table" and mrec.table_cell_features:
                    # Absorbed cover table: its cell texts ARE the title
                    # material (every cell passed the §2.2.4 membership gate:
                    # short, non-body, no outline), so the block carries them
                    # verbatim and the <table> placeholder is dropped. I1
                    # only audits kind=="para" records — this explicit
                    # carry-over is what keeps absorption lossless.
                    cell_texts = [
                        t.strip()
                        for row in mrec.table_cell_features
                        for (t, _size, _outline) in row
                        if t and t.strip()
                    ]
                    if cell_texts:
                        member_paras.append(
                            {
                                "text": "\n".join(cell_texts),
                                "para_id": mrec.para_id,
                                "para_id_end": mrec.para_id_end,
                                "is_table": False,
                            }
                        )
            if not member_paras:
                # Defensive: a title verdict with no emittable member content
                # is unreachable in practice — judge_title_block's locate-back
                # requires main_title to match a non-empty window — but we must
                # never crash (_build_unsplit_block indexes paragraphs[-1]) nor
                # silently mis-label. Warn, drop the degenerate lead, and do
                # NOT fall through to the heading branch (which would coerce
                # is_heading/level=0 into a level-1 heading). Members stay OUT
                # of title_member_skip so they are not silently dropped.
                logger.warning(
                    "[smart_heading] title-block lead at record %d has no "
                    "emittable member content; skipping title treatment",
                    i,
                )
                if parse_warnings is not None:
                    parse_warnings["title_block_empty_members"] = (
                        parse_warnings.get("title_block_empty_members", 0) + 1
                    )
                current_is_title_block = False
                continue
            main_title = (
                d.doc_title_heading
                or (d.title_parts[0] if d.title_parts else None)
                or d.composed_heading
                or d.text
            )
            if "\n" in main_title:
                # LLM verdicts are flattened at construction (_opt_str), but
                # the d.text fallback (hand-built decisions) still carries raw
                # soft breaks — same single-line rule as the plain branch.
                from .smart_heading.title_block import flatten_heading_line

                main_title = flatten_heading_line(main_title) or main_title
            title_heading = truncate_heading(main_title, rec.para_id)
            # Seed the block with the title cover; the following body joins the
            # SAME block so the title block OWNS its content up to the next
            # heading/title/EOF. (Earlier versions emitted the cover standalone
            # then let the body form a second same-heading block — the split
            # was the block-boundary bug, not a parent-heading one.)
            current_paragraphs = list(member_paras)
            current_heading = title_heading
            current_heading_level = 0
            current_parent_headings = []
            current_heading_stack = {0: title_heading}
            current_is_title_block = True
            # Skip the non-lead members' standalone rows (their content is
            # already carried above). Added HERE, on a successful seed only.
            title_member_skip.update(d.member_indices[1:])
            if not first_heading_recorded:
                if parse_metadata is not None:
                    parse_metadata["first_heading"] = title_heading
                first_heading_recorded = True
            continue

        if rec.kind == "table":
            current_paragraphs.append(
                {
                    "text": rec.text,
                    "para_id": rec.para_id,
                    "para_id_end": rec.para_id_end,
                    "is_table": True,
                    "_table_header": rec.table_header_rows,
                }
            )
            continue

        if d is not None and d.is_heading:
            level = max(1, int(d.level or 1))
            heading_para_id = rec.para_id
            heading_text = d.text
            if "\n" in heading_text:
                # §2.2.7: a heading that still carries soft-break lines is
                # ONE title — render it as a single line (CJK-aware join),
                # or the I1 source paragraph can never match any output row.
                from .smart_heading.title_block import flatten_heading_line

                heading_text = flatten_heading_line(heading_text) or heading_text
            truncated_text = truncate_heading(heading_text, heading_para_id)
            clean_heading_text = strip_heading_markdown_prefix(truncated_text)

            if not first_heading_recorded:
                if parse_metadata is not None:
                    parse_metadata["first_heading"] = clean_heading_text
                first_heading_recorded = True

            _flush()
            current_paragraphs.append(
                {
                    "text": render_heading_line(level, truncated_text),
                    "para_id": heading_para_id,
                    "is_table": False,
                }
            )
            current_heading = clean_heading_text
            current_heading_level = level
            current_is_title_block = False
            current_parent_headings = [
                current_heading_stack[lvl]
                for lvl in sorted(current_heading_stack.keys())
                if lvl < level
            ]
            current_heading_stack = {
                k: v for k, v in current_heading_stack.items() if k < level
            }
            current_heading_stack[level] = clean_heading_text

            if rec.demoted_body_text and not d.use_raw_text:
                current_paragraphs.append(
                    {
                        "text": rec.demoted_body_text,
                        "para_id": heading_para_id,
                        "is_table": False,
                    }
                )
        else:
            if d is not None and d.use_raw_text and rec.full_text_raw:
                current_paragraphs.append(
                    {
                        "text": rec.full_text_raw,
                        "para_id": rec.para_id,
                        "is_table": False,
                    }
                )
            else:
                current_paragraphs.append(
                    {"text": rec.text, "para_id": rec.para_id, "is_table": False}
                )
                if rec.demoted_body_text:
                    current_paragraphs.append(
                        {
                            "text": rec.demoted_body_text,
                            "para_id": rec.para_id,
                            "is_table": False,
                        }
                    )

    _flush()
    return blocks
