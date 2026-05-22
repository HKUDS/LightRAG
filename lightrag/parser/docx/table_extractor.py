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
    w:ins, w:sdt, w:fldSimple, w:smartTag) to avoid dropping content.

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
        if tag in _SKIP_PARAGRAPH_TAGS:
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
            from omml import convert_omml_to_latex

            latex = convert_omml_to_latex(node)
            if latex:
                parts.append(f"<equation>{latex}</equation>")
            return
        if tag == "oMathPara":
            from omml import convert_omml_to_latex

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
    ) -> dict:
        """
        Extract table to 2D string array with metadata (paraIds, header info).

        Vertical merge behavior:
        - All rows in a vertically merged region share the same content
        - All rows use the paraId from the merge start cell (for precise edit targeting)

        Args:
            table: python-docx Table object
            numbering_resolver: Optional NumberingResolver for extracting numbering

        Returns:
            Dict with:
            - rows: 2D list of cell text strings
            - para_ids: 2D list of paraIds (first paraId in each cell, or None)
                        For vertically merged cells, all rows share the start cell's paraId
            - para_ids_end: 2D list of paraIds (last paraId in each cell, or None)
                            For vertically merged cells, all rows share the start cell's paraId
            - header_indices: List of row indices marked as table headers
        """
        tbl = table._tbl

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
            }

        # Detect header rows using w:tblHeader attribute
        header_indices = []
        for idx, tr in enumerate(tbl.findall(qn("w:tr"))):
            trPr = tr.find(qn("w:trPr"))
            if trPr is not None:
                tbl_header = trPr.find(qn("w:tblHeader"))
                if tbl_header is not None:
                    header_indices.append(idx)

        # Process each row by directly iterating <w:tr> elements
        grid = []
        para_ids_grid = []
        para_ids_end_grid = []  # Track last paraId in each cell
        vmerge_content = {}  # Track vertical merge by column: {col: {'text': str, 'para_id': str, 'para_id_end': str}}

        for tr in tbl.findall(qn("w:tr")):
            row_data = [""] * num_cols  # Pre-fill with empty strings
            row_para_ids = [None] * num_cols  # Pre-fill with None
            row_para_ids_end = [None] * num_cols  # Pre-fill with None for last paraId
            grid_col = 0

            # Iterate actual <w:tc> elements (each physical cell appears once)
            for tc in tr.findall(qn("w:tc")):
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
                    if numbering_resolver is not None:
                        # Extract text with numbering labels and superscript/subscript markup
                        cell_paragraphs = []
                        for para_elem in tc.findall(qn("w:p")):
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
                        for para_elem in tc.findall(qn("w:p")):
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
                    elif is_normal_cell:
                        # For normal cells: if empty and we have active vMerge, copy all from start
                        # If non-empty, this ends the vMerge region
                        if not cell_text and grid_col in vmerge_content:
                            # Empty cell in vMerge region - copy content and paraIds from start
                            cell_text = vmerge_content[grid_col]["text"]
                            cell_para_id = vmerge_content[grid_col]["para_id"]
                            cell_para_id_end = vmerge_content[grid_col]["para_id_end"]
                        elif cell_text:
                            # Non-empty cell - this ends the vMerge for this column
                            vmerge_content.pop(grid_col, None)

                elif is_vmerge_continue:
                    # Copy content and para_id from previous merge start
                    # But extract actual para_id_end from this continue cell for range boundary
                    if grid_col in vmerge_content:
                        cell_text = vmerge_content[grid_col]["text"]
                        cell_para_id = vmerge_content[grid_col][
                            "para_id"
                        ]  # Use restart's paraId for edit targeting

                        # Extract actual paraId from this continue cell for uuid_end (range boundary)
                        for para_elem in tc.findall(qn("w:p")):
                            para_id_attr = para_elem.get(
                                "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
                            )
                            if para_id_attr:
                                cell_para_id_end = (
                                    para_id_attr  # Use actual paraId for range boundary
                                )

                # Place content at starting grid position only
                if grid_col < num_cols:
                    row_data[grid_col] = cell_text
                    row_para_ids[grid_col] = cell_para_id
                    row_para_ids_end[grid_col] = cell_para_id_end

                # Move grid position by gridSpan
                grid_col += grid_span

            grid.append(row_data)
            para_ids_grid.append(row_para_ids)
            para_ids_end_grid.append(row_para_ids_end)

        return {
            "rows": grid,
            "para_ids": para_ids_grid,
            "para_ids_end": para_ids_end_grid,
            "header_indices": header_indices,
        }
