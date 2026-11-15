#!/usr/bin/env python3
"""
ABOUTME: Extracts tables from DOCX with proper merged cell handling
ABOUTME: Vertically merged cells: content repeated in all rows with shared paraId
ABOUTME: Horizontally merged cells: content in first cell only
ABOUTME: Preserves superscript/subscript formatting with <sup>/<sub> markup
"""

from docx.table import Table
from docx.oxml.ns import qn
from typing import List

from .content_controls import is_hidden_inline_control, iter_blocks, iter_cells
from .drawing_image_extractor import (
    DrawingExtractionContext,
    extract_drawing_placeholder_from_element,
    extract_vml_image_placeholder_from_element,
)

# Keep in sync with parse_document._SKIP_PARAGRAPH_TAGS — duplicated here to
# avoid a circular import between parse_document and table_extractor.
_SKIP_PARAGRAPH_TAGS = frozenset(
    {
        "del",
        "moveFrom",
        "commentRangeStart",
        "commentRangeEnd",
        "commentReference",
        "annotationRef",
    }
)


def extract_text_from_run_table(
    run_elem,
    qn_func,
    drawing_context: DrawingExtractionContext = None,
) -> str:
    """
    Extract text from a run element in table cell, preserving superscript/subscript with markup.

    Converts Word formatting to HTML-like tags:
    - Superscript: <sup>text</sup>
    - Subscript: <sub>text</sub>
    - Normal text: unchanged

    Args:
        run_elem: lxml run element (w:r)
        qn_func: qn function for namespace handling

    Returns:
        Text string with <sup>/<sub> markup for formatted portions
    """
    text = ""

    # Check for vertAlign in rPr (superscript/subscript)
    vert_align = None
    rPr = run_elem.find(qn_func("w:rPr"))
    if rPr is not None:
        vert_elem = rPr.find(qn_func("w:vertAlign"))
        if vert_elem is not None:
            vert_align = vert_elem.get(qn_func("w:val"))

    # Extract text content from run children
    for child in run_elem:
        tag = child.tag.split("}")[-1]  # Remove namespace
        if tag == "t" and child.text:
            text += child.text
        elif tag == "tab":
            text += "\t"
        elif tag == "br":
            # Handle line breaks - textWrapping or no type = soft line break
            br_type = child.get(qn_func("w:type"))
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


def extract_paragraph_content_table(
    para_elem,
    qn_func,
    drawing_context: DrawingExtractionContext = None,
) -> str:
    """
    Extract text and equations from a table cell paragraph in document order.

    Handles w:r (text runs), m:oMath (inline equations), and m:oMathPara
    (block equations). Recurses into container elements (e.g., w:hyperlink,
    w:ins, w:sdt, w:fldSimple, w:smartTag) to avoid dropping content, except
    a content control still showing its placeholder (see content_controls).

    Args:
        para_elem: lxml paragraph element (w:p)
        qn_func: qn function for namespace handling

    Returns:
        Text string with equations wrapped in <equation> tags
    """
    parts = []

    def append_from(node) -> None:
        tag = node.tag.split("}")[-1]
        # Drop tracked-change deletions (w:del/w:moveFrom) and comment markers
        # (w:commentRangeStart/End, w:commentReference, w:annotationRef) so the
        # output only contains the final revised text without annotation glyphs.
        if tag in _SKIP_PARAGRAPH_TAGS or is_hidden_inline_control(node):
            return
        if tag == "r":
            parts.append(
                extract_text_from_run_table(
                    node,
                    qn_func,
                    drawing_context=drawing_context,
                )
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

    for child in para_elem:
        append_from(child)

    return "".join(parts)


def _extract_cell_features(tc, qn_func, style_attributes, styles_outline) -> tuple:
    """Physical-cell features for the smart title-block table channel.

    Returns ``(text, effective_size_pt, has_outline)`` computed from the
    cell's OWN ``w:p`` children (direct or inside a content control, see
    ``content_controls`` — nested tables are excluded; vMerge-continue cells
    report their own, usually empty, text):

    - ``text``: visible source characters (``w:t`` via ``run_features``), so
      drawing/equation placeholders and auto-numbering labels never inflate
      the title-length gate;
    - ``effective_size_pt``: max per-paragraph char-weighted dominant size,
      resolved through the full cascade (run rPr > paragraph mark > style
      chain > docDefaults) — a bare ``run.font.size`` would miss inherited
      sizes;
    - ``has_outline``: any paragraph with a direct ``w:outlineLvl`` or a
      ``pStyle`` mapped by ``styles_outline`` (physical-outline cells are
      structural headings, never title-block material).
    """
    # Function-local import: the smart-off path must stay free of the
    # smart_heading package (same convention as parse_document).
    from lightrag.parser.docx.smart_heading.features import (
        extract_paragraph_physical_features,
    )

    texts = []
    max_size = None
    has_outline = False
    for para_elem in iter_blocks(tc, _PARAGRAPH_TAGS):
        phys = extract_paragraph_physical_features(para_elem, style_attributes)
        text = "".join(rf.text for rf in phys.run_features).strip()
        if text:
            texts.append(text)
        if phys.font_size_pt is not None and (
            max_size is None or phys.font_size_pt > max_size
        ):
            max_size = phys.font_size_pt
        if not has_outline:
            ppr = para_elem.find(qn_func("w:pPr"))
            if ppr is not None and ppr.find(qn_func("w:outlineLvl")) is not None:
                has_outline = True
            elif phys.style_id and styles_outline and phys.style_id in styles_outline:
                has_outline = True
    return ("\n".join(texts), max_size, has_outline)


_ROW_TAGS = frozenset({qn("w:tr")})
_PARAGRAPH_TAGS = frozenset({qn("w:p")})


def _grid_before(tr) -> int:
    """Return how many leading grid columns a ``w:tr`` leaves without a cell."""
    tr_pr = tr.find(qn("w:trPr"))
    node = tr_pr.find(qn("w:gridBefore")) if tr_pr is not None else None
    if node is None:
        return 0
    try:
        return max(int(node.get(qn("w:val"), "0")), 0)
    except ValueError:
        return 0


class TableExtractor:
    """
    Extract table content handling merged cells correctly.

    Merged cells in DOCX:
    - Horizontal: w:gridSpan specifies how many columns cell spans
    - Vertical: w:vMerge with val="restart" starts merge, subsequent cells continue

    Output format:
    - 2D list of strings
    - Vertically merged cells: content repeated in all rows, all rows use the same paraId (from start cell)
    - Horizontally merged cells: content in left-most position only, other positions empty
    """

    @staticmethod
    def extract(
        table: Table,
        numbering_resolver=None,
        drawing_context: DrawingExtractionContext = None,
    ) -> List[List[str]]:
        """
        Extract table to 2D string array.

        Args:
            table: python-docx Table object
            numbering_resolver: Optional NumberingResolver for extracting numbering

        Returns:
            List of rows, each row is list of cell text strings
        """
        result = TableExtractor.extract_with_metadata(
            table,
            numbering_resolver=numbering_resolver,
            drawing_context=drawing_context,
        )
        return result["rows"]

    @staticmethod
    def extract_with_metadata(
        table: Table,
        numbering_resolver=None,
        drawing_context: DrawingExtractionContext = None,
        style_attributes=None,
        styles_outline=None,
    ) -> dict:
        """
        Extract table to 2D string array with metadata (paraIds, header info).

        Vertical merge behavior:
        - All rows in a vertically merged region share the same content
        - All rows use the paraId from the merge start cell (for precise edit targeting)
        - A merge continues only from the row directly above: a row with no
          restart or continuation cell at that grid column ends it. A
          continuation below such a row repeats nothing; Word draws it as a
          separate empty cell, whether a normal cell or no cell (gridBefore)
          broke the merge. It keeps its own paraIds

        Grid alignment:
        - A row that starts late (``w:gridBefore``) leaves its first grid
          columns empty, so its cells stay under the columns they are drawn in
        - Rows and cells inside a content control or custom XML markup count
          like direct ones, and so do paragraphs a cell wraps in one (see
          ``content_controls``). A cell inside a table of contents or
          placeholder control keeps its grid columns and paraIds but no text
          (nothing in it is extracted, so its images are not exported); a row
          inside one is left out

        Args:
            table: python-docx Table object
            numbering_resolver: Optional NumberingResolver for extracting numbering
            style_attributes: Optional smart_heading ``StyleAttributes``; when
                given, per-physical-cell features are collected for the
                title-block table channel (smart-only; legacy passes None)
            styles_outline: Optional style-id -> outline-level map used for
                the cell ``has_outline`` feature

        Returns:
            Dict with:
            - rows: 2D list of cell text strings
            - para_ids: 2D list of paraIds (first paraId in each cell, or None)
                        For vertically merged cells, all rows share the start cell's paraId
            - para_ids_end: 2D list of paraIds (last paraId in each cell, or None)
                            Continuations use the last paraId of their own cell
            - header_indices: List of row indices marked as table headers
            - cell_features: per-row lists of PHYSICAL cells as
              ``(text, effective_size_pt, has_outline)`` tuples (one entry per
              ``w:tc`` — a gridSpan-merged full-width row is ONE cell), or
              None when ``style_attributes`` was not provided
        """
        tbl = table._tbl
        collect_features = style_attributes is not None

        # Get number of columns from tblGrid
        tbl_grid = tbl.find(qn("w:tblGrid"))
        num_cols = 0
        if tbl_grid is not None:
            num_cols = len(tbl_grid.findall(qn("w:gridCol")))

        if num_cols == 0:
            return {
                "rows": [],
                "para_ids": [],
                "para_ids_end": [],
                "header_indices": [],
                "cell_features": [] if collect_features else None,
            }

        # Detect header rows using w:tblHeader attribute
        header_indices = []
        for idx, tr in enumerate(iter_blocks(tbl, _ROW_TAGS)):
            trPr = tr.find(qn("w:trPr"))
            if trPr is not None:
                tbl_header = trPr.find(qn("w:tblHeader"))
                if tbl_header is not None:
                    header_indices.append(idx)

        # Process each row by directly iterating <w:tr> elements
        grid = []
        para_ids_grid = []
        para_ids_end_grid = []  # Track last paraId in each cell
        cell_features_grid = [] if collect_features else None
        vmerge_content = {}  # Track vertical merge by column: {col: {'text': str, 'para_id': str, 'para_id_end': str}}

        for tr in iter_blocks(tbl, _ROW_TAGS):
            row_data = [""] * num_cols  # Pre-fill with empty strings
            row_para_ids = [None] * num_cols  # Pre-fill with None
            row_para_ids_end = [None] * num_cols  # Pre-fill with None for last paraId
            row_cell_features = [] if collect_features else None
            grid_col = _grid_before(tr)
            merged_cols = set()  # columns whose vertical merge reaches this row

            # Iterate actual <w:tc> elements (each physical cell appears once)
            for tc, hidden in iter_cells(tr):
                if collect_features:
                    row_cell_features.append(
                        ("", None, False)
                        if hidden
                        else _extract_cell_features(
                            tc, qn, style_attributes, styles_outline
                        )
                    )
                # Reset numbering state when cell changes to prevent incorrect continuation
                if numbering_resolver is not None:
                    numbering_resolver.reset_tracking_state()

                tcPr = tc.find(qn("w:tcPr"))

                # Check gridSpan (horizontal merge)
                grid_span = 1
                if tcPr is not None:
                    gs = tcPr.find(qn("w:gridSpan"))
                    if gs is not None:
                        grid_span = int(gs.get(qn("w:val")))

                # Check vMerge (vertical merge)
                vmerge_elem = None
                vmerge_val = None
                if tcPr is not None:
                    vmerge_elem = tcPr.find(qn("w:vMerge"))
                    if vmerge_elem is not None:
                        vmerge_val = vmerge_elem.get(
                            qn("w:val")
                        )  # 'restart' or None (means 'continue')

                # Determine vMerge status
                is_vmerge_restart = vmerge_elem is not None and vmerge_val == "restart"
                is_vmerge_continue = vmerge_elem is not None and vmerge_val in (
                    None,
                    "continue",
                )
                is_normal_cell = vmerge_elem is None

                cell_text = ""
                cell_para_id = None
                cell_para_id_end = None  # Track last paraId in cell

                # Handle different vMerge cases
                if is_vmerge_restart or is_normal_cell:
                    # Extract content for restart or normal cells
                    # Get cell text with numbering support and format preservation
                    if hidden:
                        # Only the paraIds: extracting would still export its
                        # images (against the shared image budget) and count
                        # its list items.
                        for para_elem in iter_blocks(tc, _PARAGRAPH_TAGS):
                            para_id_attr = para_elem.get(
                                "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
                            )
                            if para_id_attr:
                                if cell_para_id is None:
                                    cell_para_id = para_id_attr
                                cell_para_id_end = para_id_attr
                    elif numbering_resolver is not None:
                        # Extract text with numbering labels and superscript/subscript markup
                        cell_paragraphs = []
                        for para_elem in iter_blocks(tc, _PARAGRAPH_TAGS):
                            # Capture paraId from each paragraph
                            para_id_attr = para_elem.get(
                                "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
                            )
                            if para_id_attr:
                                if cell_para_id is None:
                                    cell_para_id = para_id_attr  # First paraId
                                cell_para_id_end = (
                                    para_id_attr  # Always update to get last
                                )

                            # Get text content with format preservation (superscript/subscript/equations)
                            para_text = extract_paragraph_content_table(
                                para_elem,
                                qn,
                                drawing_context=drawing_context,
                            )

                            # Get numbering label
                            label = numbering_resolver.get_label(para_elem)

                            # Combine label and text
                            if label:
                                full_text = f"{label} {para_text}".strip()
                            else:
                                full_text = para_text.strip()

                            if full_text:
                                cell_paragraphs.append(full_text)

                        cell_text = "\n".join(cell_paragraphs).replace("\x07", "")
                    else:
                        # Fallback to simple text extraction with format preservation
                        # Cannot use cell.text here, must extract from XML
                        para_texts = []
                        for para_elem in iter_blocks(tc, _PARAGRAPH_TAGS):
                            # Capture paraId from each paragraph
                            para_id_attr = para_elem.get(
                                "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
                            )
                            if para_id_attr:
                                if cell_para_id is None:
                                    cell_para_id = para_id_attr  # First paraId
                                cell_para_id_end = (
                                    para_id_attr  # Always update to get last
                                )

                            # Extract text with format preservation (superscript/subscript/equations)
                            para_text = extract_paragraph_content_table(
                                para_elem,
                                qn,
                                drawing_context=drawing_context,
                            )

                            if para_text:
                                para_texts.append(para_text.strip())
                        cell_text = "\n".join(para_texts).replace("\x07", "")

                    # Store content and paraIds for vMerge restart
                    if is_vmerge_restart:
                        vmerge_content[grid_col] = {
                            "text": cell_text,
                            "para_id": cell_para_id,
                            "para_id_end": cell_para_id_end,
                        }
                        merged_cols.add(grid_col)

                elif is_vmerge_continue:
                    # The continuation's own paraIds bound the range (uuid_end),
                    # whether or not a merge still reaches it.
                    for para_elem in iter_blocks(tc, _PARAGRAPH_TAGS):
                        para_id_attr = para_elem.get(
                            "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
                        )
                        if para_id_attr:
                            if cell_para_id is None:
                                cell_para_id = para_id_attr
                            cell_para_id_end = para_id_attr
                    if grid_col in vmerge_content:
                        # Copy content and the restart's paraId (edit target)
                        merged_cols.add(grid_col)
                        cell_text = vmerge_content[grid_col]["text"]
                        cell_para_id = vmerge_content[grid_col]["para_id"]
                    # Otherwise no merge reaches it: Word draws a separate
                    # empty cell, so it keeps no text and its own paraIds.

                if hidden:
                    cell_text = ""

                # Place content at starting grid position only
                if grid_col < num_cols:
                    row_data[grid_col] = cell_text
                    row_para_ids[grid_col] = cell_para_id
                    row_para_ids_end[grid_col] = cell_para_id_end

                # Move grid position by gridSpan
                grid_col += grid_span

            # A normal cell, or no cell at all (gridBefore / gridAfter, a
            # short row), ends the merge in its column, even when empty.
            for col in [col for col in vmerge_content if col not in merged_cols]:
                del vmerge_content[col]

            grid.append(row_data)
            para_ids_grid.append(row_para_ids)
            para_ids_end_grid.append(row_para_ids_end)
            if collect_features:
                cell_features_grid.append(row_cell_features)

        return {
            "rows": grid,
            "para_ids": para_ids_grid,
            "para_ids_end": para_ids_end_grid,
            "header_indices": header_indices,
            "cell_features": cell_features_grid,
        }
