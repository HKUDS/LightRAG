#!/usr/bin/env python3
"""
ABOUTME: Parses DOCX documents into text blocks using python-docx
ABOUTME: Extracts automatic numbering, splits by headings, converts tables to JSON
"""

import json
import sys

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

from lightrag.parser._markdown import (
    render_heading_line,
    strip_heading_markdown_prefix,
)
from .numbering_resolver import NumberingResolver
from .table_extractor import TableExtractor
from .drawing_image_extractor import (
    DrawingExtractionContext,
    extract_drawing_placeholder_from_element,
    extract_vml_image_placeholder_from_element,
)


# Constants for content validation (character-based for UI/display)
MAX_HEADING_LENGTH = 200  # Maximum heading length in characters (UI constraint)

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
    heading: str, paragraphs: list, parent_headings: list, level: int
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
) -> None:
    """Flush accumulated paragraphs into a single heading-scoped block.

    The native parser performs only heading-driven structural splitting; block
    sizing (long-block anchor splitting, table row splitting, small-block
    merging) is the downstream paragraph-semantic chunker's responsibility.
    """
    if not paragraphs:
        return

    blocks.append(_build_unsplit_block(heading, paragraphs, parent_headings, level))


def extract_docx_blocks(
    file_path: str,
    drawing_context: DrawingExtractionContext = None,
    parse_warnings: dict | None = None,
    parse_metadata: dict | None = None,
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

    blocks = []
    current_heading = "Preface/Uncategorized"
    current_heading_level = 1  # Default level for "Preface/Uncategorized"
    current_heading_stack = {}  # {level: heading_text} - Use dict to correctly track heading hierarchy
    current_parent_headings = []  # Parent headings for current block
    current_paragraphs = []  # Track paragraphs with metadata for splitting
    first_heading_recorded = (
        False  # Track whether the document's first heading has been captured
    )

    # Iterate through document body elements (paragraphs and tables)
    body = doc._element.body

    for element in body:
        tag = element.tag.split("}")[-1]  # Remove namespace

        if tag == "sectPr":  # Document-level section break
            resolver.reset_tracking_state()
            continue

        if tag == "p":  # Paragraph
            # Get paragraph text with superscript/subscript markup and equations
            para_text = ""
            ns = {
                "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
                "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
                "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
            }
            para_text = extract_paragraph_content(
                element,
                ns,
                drawing_context=drawing_context,
            )

            para_text = para_text.strip()
            if not para_text:
                continue

            # Get numbering label using our resolver
            label = resolver.get_label(element)
            full_text = f"{label} {para_text}".strip() if label else para_text

            # Check if this is a heading using the new function
            outline_level = get_heading_level(element, styles_outline)

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

            if outline_level is not None:
                # This is a heading (outline level 0-8)
                # Convert 0-based to 1-based level
                level = outline_level + 1

                # Extract paraId for this heading
                heading_para_id = extract_para_id(element)
                if parse_warnings is not None and not heading_para_id:
                    parse_warnings["missing_paraid_count"] = (
                        parse_warnings.get("missing_paraid_count", 0) + 1
                    )

                # Validate heading length
                validate_heading_length(full_text, heading_para_id)

                # Truncate heading if needed before storing
                truncated_text = truncate_heading(full_text, heading_para_id)
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
                if demoted_body_text:
                    current_paragraphs.append(
                        {
                            "text": demoted_body_text,
                            "para_id": heading_para_id,
                            "is_table": False,
                        }
                    )
            else:
                # Regular paragraph content
                para_id = extract_para_id(element)
                if parse_warnings is not None and not para_id:
                    parse_warnings["missing_paraid_count"] = (
                        parse_warnings.get("missing_paraid_count", 0) + 1
                    )

                # Store paragraph with metadata for potential splitting
                current_paragraphs.append(
                    {"text": full_text, "para_id": para_id, "is_table": False}
                )

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
            table_para_id = find_first_valid_para_id(para_ids)
            table_para_id_end = find_last_valid_para_id(para_ids_end)
            current_paragraphs.append(
                {
                    "text": f"<table>{table_json}</table>",
                    "para_id": table_para_id,
                    "para_id_end": table_para_id_end,  # Store end paraId for uuid_end calculation
                    "is_table": True,
                    "_table_header": header_rows_or_none,
                }
            )

            # Reset numbering tracking after table (table end boundary)
            resolver.reset_tracking_state()

    # Save final block
    _flush_current_block(
        blocks,
        current_heading,
        current_paragraphs,
        current_parent_headings,
        current_heading_level,
    )

    return blocks
