"""Legacy text extractors (moved from the API layer).

``extract_text`` dispatches on file suffix: binary office/pdf formats use the
dedicated ``_extract_*`` helpers; everything else is decoded as UTF-8 text
with the same validation (empty / binary-looking / non-UTF-8) the API upload
path used to enforce — now raised as :class:`LegacyExtractionError` so a bad
file fails the parse stage instead of silently yielding an empty document.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO


class LegacyExtractionError(ValueError):
    """Raised when legacy extraction cannot produce usable text."""


def _extract_pdf_pypdf(file_bytes: bytes, password: str | None = None) -> str:
    """Extract PDF content using pypdf (synchronous)."""
    from pypdf import PdfReader  # type: ignore

    pdf_file = BytesIO(file_bytes)
    reader = PdfReader(pdf_file)

    if reader.is_encrypted:
        # Try empty password first (covers permission-only encrypted PDFs)
        decrypt_result = reader.decrypt(password or "")
        if decrypt_result == 0:
            if password:
                raise Exception("Incorrect PDF password")
            else:
                raise Exception("PDF is encrypted but no password provided")

    content = ""
    for page in reader.pages:
        content += page.extract_text() + "\n"
    return content


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_BLOCK_TAGS = frozenset({_W + "p", _W + "tbl"})
# A content control keeps its paragraphs, tables, rows or cells in
# w:sdtContent, and custom XML markup wraps them directly. Cover pages, the
# fields of a form template, whole template sections and repeating table rows
# are such controls.
_DOCX_BLOCK_WRAPPERS = frozenset({_W + "sdt", _W + "sdtContent", _W + "customXml"})
_MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
# Content below these is not text of its paragraph: deleted and moved-away
# revisions, the pronunciation guide of ruby text, and text boxes.
_DOCX_SKIPPED_INLINE = frozenset(
    {_W + "del", _W + "moveFrom", _W + "rt", _W + "txbxContent"}
)
# The children of a w:r that python-docx's run text is made of.
_DOCX_RUN_TEXT_TAGS = frozenset(
    _W + tag for tag in ("t", "tab", "br", "cr", "noBreakHyphen", "ptab")
)


def _is_skipped_docx_control(sdt) -> bool:
    """True for a ``w:sdt`` whose content is not document text.

    That is the control Word puts its built-in table of contents in, whose
    lines repeat the headings with page numbers, and a control still showing
    its placeholder ("Click or tap here to enter text."). Neither was ever
    read by this extractor.
    """
    sdt_pr = sdt.find(_W + "sdtPr")
    if sdt_pr is None:
        return False
    gallery = sdt_pr.find(f"{_W}docPartObj/{_W}docPartGallery")
    if gallery is not None and gallery.get(_W + "val") == "Table of Contents":
        return True
    placeholder = sdt_pr.find(_W + "showingPlcHdr")
    return placeholder is not None and placeholder.get(_W + "val") not in (
        "0",
        "false",
        "off",
    )


def _iter_docx_children(container, tags=_DOCX_BLOCK_TAGS):
    """Yield the children of ``container`` whose tag is in ``tags``, in order.

    Unlike a loop over the container's own children, this includes the ones
    inside content controls and custom XML markup, except the controls
    :func:`_is_skipped_docx_control` names.
    """
    for child in container:
        if child.tag in tags:
            yield child
        elif child.tag in _DOCX_BLOCK_WRAPPERS and not (
            child.tag == _W + "sdt" and _is_skipped_docx_control(child)
        ):
            yield from _iter_docx_children(child, tags)


def _iter_docx_row_cells(container, hidden: bool = False):
    """Yield ``(w:tc, hidden)`` for each cell of a ``w:tr``, in order.

    Looks through the same wrappers as :func:`_iter_docx_children`, but a cell
    inside a control that :func:`_is_skipped_docx_control` names is still
    yielded, with ``hidden`` set: it takes its grid columns, so the cells after
    it keep theirs, and only its text is left out.
    """
    for child in container:
        if child.tag == _W + "tc":
            yield child, hidden
        elif child.tag in _DOCX_BLOCK_WRAPPERS:
            yield from _iter_docx_row_cells(
                child,
                hidden or (child.tag == _W + "sdt" and _is_skipped_docx_control(child)),
            )


def _docx_paragraph_text(paragraph) -> str:
    """Text of a ``w:p`` element, including runs nested below the paragraph.

    ``Paragraph.text`` reads only direct runs and hyperlinks, so the text of an
    inline content control, a tracked insertion, a simple field's result or a
    smart tag is missing from it. Deleted text, and the controls
    :func:`_is_skipped_docx_control` names, stay out.

    Text comes out in document order, also when a run holds ruby text between
    its own pieces. Of an ``mc:AlternateContent``, only the branch a reader
    shows is read: the first ``mc:Choice``, or the ``mc:Fallback`` without one.
    """
    parts: list[str] = []
    _collect_docx_inline_text(paragraph, parts)
    return "".join(parts)


def _collect_docx_inline_text(element, parts: list[str]) -> None:
    for child in element:
        tag = child.tag
        if tag in _DOCX_RUN_TEXT_TAGS:
            # A w:tab outside a run is a tab stop of the paragraph properties.
            if element.tag == _W + "r":
                parts.append(str(child))
        elif tag == _MC + "AlternateContent":
            branch = child.find(_MC + "Choice")
            if branch is None:
                branch = child.find(_MC + "Fallback")
            if branch is not None:
                _collect_docx_inline_text(branch, parts)
        elif tag in _DOCX_SKIPPED_INLINE or (
            tag == _W + "sdt" and _is_skipped_docx_control(child)
        ):
            continue
        else:
            _collect_docx_inline_text(child, parts)


def _docx_cell_text(tc) -> str:
    """Text of a ``w:tc`` element, one line per paragraph, like ``_Cell.text``."""
    return "\n".join(
        _docx_paragraph_text(paragraph)
        for paragraph in _iter_docx_children(tc, frozenset({_W + "p"}))
    )


def _extract_docx(file_bytes: bytes) -> str:
    """Extract DOCX content including tables in document order (synchronous)."""
    from docx import Document  # type: ignore
    from docx.table import Table  # type: ignore

    docx_file = BytesIO(file_bytes)
    doc = Document(docx_file)

    def escape_cell(cell_value: str | None) -> str:
        if cell_value is None:
            return ""
        text = str(cell_value)
        return (
            text.replace("\\", "\\\\")
            .replace("\t", "&emsp;&emsp;")
            .replace("\r\n", "<br>")
            .replace("\r", "<br>")
            .replace("\n", "<br>")
        )

    content_parts = []
    in_table = False
    for element in _iter_docx_children(doc.element.body):
        if element.tag.endswith("p"):
            if in_table:
                content_parts.append("")
                in_table = False
            content_parts.append(_docx_paragraph_text(element))
        elif element.tag.endswith("tbl"):
            if content_parts and not in_table:
                content_parts.append("")
            in_table = True
            for row in _docx_table_grid_rows(Table(element, doc)):
                row_text = [escape_cell(cell) for cell in row]
                if any(cell for cell in row_text):
                    content_parts.append("\t".join(row_text))
    return "\n".join(content_parts)


def _docx_grid_skip(tr, tag: str) -> int:
    """Return the ``w:gridBefore`` / ``w:gridAfter`` count of a ``w:tr``."""
    from docx.oxml.ns import qn  # type: ignore

    tr_pr = tr.find(qn("w:trPr"))
    node = tr_pr.find(qn(tag)) if tr_pr is not None else None
    if node is None:
        return 0
    try:
        return max(int(node.get(qn("w:val"), "0")), 0)
    except ValueError:
        return 0


def _docx_table_grid_rows(table) -> Iterator[list[str]]:
    """Yield each row of a DOCX table as one text slot per layout-grid column.

    - A row that starts or ends late (``w:gridBefore`` / ``w:gridAfter``)
      gets an empty slot for each skipped grid column, so its cells stay
      under the columns they are drawn in.
    - A horizontally merged cell (``w:gridSpan``) puts its text in its first
      column and leaves the other spanned columns empty.
    - A vertically merged cell repeats the text of the row above at the same
      grid column; with no cell there, the slot is empty.
    - Rows and cells inside a content control or custom XML markup (a
      repeating section) count like direct ones. A cell inside a table of
      contents or placeholder control keeps its grid columns but no text.

    The two merge rules are the native table extractor's.

    Walks the ``w:tr`` / ``w:tc`` XML instead of ``python-docx``'s
    ``row.cells``: that omits the ``w:gridBefore`` / ``w:gridAfter`` columns,
    and it raises ``ValueError`` on a vertical-merge continuation with no
    cell at the same grid column above, which would fail the whole document.
    """
    from docx.oxml.ns import qn  # type: ignore

    above: dict[int, str] = {}
    for tr in _iter_docx_children(table._tbl, frozenset({qn("w:tr")})):
        row = [""] * _docx_grid_skip(tr, "w:gridBefore")
        current: dict[int, str] = {}
        for tc, hidden in _iter_docx_row_cells(tr):
            column = len(row)
            if hidden:
                text = ""
            elif tc.vMerge == "continue":
                text = above.get(column, "")
            else:
                text = _docx_cell_text(tc)
            span = max(tc.grid_span, 1)
            row.append(text)
            row.extend([""] * (span - 1))
            current[column] = text
        row.extend([""] * _docx_grid_skip(tr, "w:gridAfter"))
        above = current
        yield row


def _extract_pptx(file_bytes: bytes) -> str:
    """Extract PPTX text, including nested groups, in shape order (synchronous)."""
    from pptx import Presentation  # type: ignore
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.shapes.shapetree import GroupShapes, SlideShapes

    pptx_file = BytesIO(file_bytes)
    prs = Presentation(pptx_file)

    def iter_text(shapes: SlideShapes | GroupShapes) -> Iterator[str]:
        for shape in shapes:
            # Groups have no text of their own; their children can include groups.
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from iter_text(shape.shapes)
            elif hasattr(shape, "text"):
                yield shape.text + "\n"

    return "".join(text for slide in prs.slides for text in iter_text(slide.shapes))


def _extract_xlsx(file_bytes: bytes) -> str:
    """Extract XLSX content in tab-delimited format with sheet separators.

    Formula cells are indexed by their cached calculated value when the workbook
    carries one, falling back to the formula text when it does not. openpyxl
    fixes ``data_only`` at load time, so the workbook is deliberately loaded
    twice -- ``data_only=True`` for values, ``data_only=False`` for formulas.
    This doubles peak memory/parse cost but is required to have both views;
    ``read_only=True`` is not a safe optimization here because it can report
    ``max_row``/``max_column`` as ``None`` and collapse the iteration range.
    """
    from openpyxl import load_workbook  # type: ignore

    wb_values = load_workbook(BytesIO(file_bytes), data_only=True)
    wb_formulas = load_workbook(BytesIO(file_bytes), data_only=False)

    def escape_cell(cell_value: str | int | float | None) -> str:
        if cell_value is None:
            return ""
        text = str(cell_value)
        return (
            text.replace("\\", "\\\\")
            .replace("\t", "\\t")
            .replace("\r\n", "\\n")
            .replace("\r", "\\n")
            .replace("\n", "\\n")
        )

    def escape_sheet_title(title: str) -> str:
        return str(title).replace("\n", " ").replace("\t", " ").replace("\r", " ")

    def formula_fallback(formula_value: object) -> str | None:
        """Formula text to index for a cell that carries no cached value.

        Only reached when the values view is empty, so ``formula_value`` is
        ``None``, a formula string, or an openpyxl formula object. openpyxl
        >= 3.1 reads array formulas as ``ArrayFormula`` (text in ``.text``) and
        data-table formulas as ``DataTableFormula`` (no text at all); older
        versions return strings, a data-table master being a bare ``"="``.
        Anything that is not formula text becomes an empty cell, so an object
        repr -- which embeds a memory address and would make the dedup content
        hash differ on every parse -- never reaches the index.
        """
        text = getattr(formula_value, "text", formula_value)
        if not isinstance(text, str) or text == "=":
            return None
        return text

    content_parts: list[str] = []
    sheet_separator = "=" * 20

    for idx, sheet in enumerate(wb_values):
        if idx > 0:
            content_parts.append("")
        safe_title = escape_sheet_title(sheet.title)
        content_parts.append(f"{sheet_separator} Sheet: {safe_title} {sheet_separator}")
        formula_sheet = wb_formulas[sheet.title]
        max_rows = max(sheet.max_row or 0, formula_sheet.max_row or 0)
        max_columns = max(sheet.max_column or 0, formula_sheet.max_column or 0)
        value_rows = sheet.iter_rows(
            min_row=1, max_row=max_rows, max_col=max_columns, values_only=True
        )
        formula_rows = formula_sheet.iter_rows(
            min_row=1, max_row=max_rows, max_col=max_columns, values_only=True
        )
        for value_row, formula_row in zip(value_rows, formula_rows):
            row_parts = []
            row_has_content = False
            for cell_value, formula_value in zip(value_row, formula_row):
                if cell_value is None:
                    cell_value = formula_fallback(formula_value)
                cell_text = escape_cell(cell_value)
                row_parts.append(cell_text)
                if cell_text:
                    row_has_content = True
            if not row_has_content:
                content_parts.append("")
            else:
                content_parts.append("\t".join(row_parts))
    content_parts.append(sheet_separator)
    return "\n".join(content_parts)


# Suffixes (without dot) routed to dedicated binary extractors.
_BINARY_EXTRACTORS = {
    "pdf": _extract_pdf_pypdf,
    "docx": _extract_docx,
    "pptx": _extract_pptx,
    "xlsx": _extract_xlsx,
}

# OPC/OOXML formats — all ZIP archives, all parsed locally by a library that
# materializes the uncompressed parts (python-docx / python-pptx / openpyxl).
# They share the decompression budget: guarding only .docx leaves .pptx and
# .xlsx (registered above, reachable by default routing) exposed to the same
# zip-bomb class (GHSA-2wpj-ffvv-2pq8) — .xlsx worse, since _extract_xlsx
# deliberately load_workbook()s the bytes twice.
_ZIP_OFFICE_SUFFIXES = frozenset({"docx", "pptx", "xlsx"})


def _decode_text(file_bytes: bytes) -> str:
    """UTF-8 decode with the upload-path validation, raised on failure."""
    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise LegacyExtractionError(
            "File is not valid UTF-8 encoded text. Please convert it to "
            f"UTF-8 before processing: {e}"
        ) from e
    if not content or len(content.strip()) == 0:
        raise LegacyExtractionError("File contains no content or only whitespace")
    if content.startswith("b'") or content.startswith('b"'):
        raise LegacyExtractionError(
            "File appears to contain binary data representation instead of text"
        )
    return content


def extract_text(
    file_bytes: bytes,
    suffix: str,
    *,
    pdf_password: str | None = None,
    file_path: str | None = None,
) -> str:
    """Extract plain text from ``file_bytes`` based on ``suffix`` (no dot).

    Synchronous; callers run it in a thread.  Raises
    :class:`LegacyExtractionError` (or the extractor's own exception) on
    failure.

    ``file_path`` names the source in the decompression-budget error message;
    it falls back to the suffix so a direct caller that omits it still gets a
    usable message.
    """
    suffix = suffix.lower().lstrip(".")

    # Decompression budget for the OPC/OOXML formats, enforced BEFORE the
    # extractor library opens the archive (GHSA-2wpj-ffvv-2pq8). The native
    # engine guards .docx; the default routing (``*:native-teP,*:legacy-R``)
    # falls back here and reaches .pptx/.xlsx too, which native never handles,
    # so the guard belongs at this dispatcher rather than in one per-format
    # branch. Reads the ZIP central directory once and decompresses nothing.
    if suffix in _ZIP_OFFICE_SUFFIXES:
        from lightrag.parser.docx.zip_budget import enforce_docx_decompression_budget

        enforce_docx_decompression_budget(file_bytes, file_path or suffix)

    extractor = _BINARY_EXTRACTORS.get(suffix)
    if extractor is _extract_pdf_pypdf:
        return _extract_pdf_pypdf(file_bytes, pdf_password)
    if extractor is not None:
        return extractor(file_bytes)
    return _decode_text(file_bytes)
