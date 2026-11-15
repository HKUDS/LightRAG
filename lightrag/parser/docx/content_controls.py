"""Content controls (``w:sdt``) and custom XML markup in the native DOCX parser.

Both wrap ordinary document content: filled template and form fields,
generated sections and clauses, repeating sections, cover pages. What they
wrap is read like unwrapped content, at every level: body blocks, table
rows, table cells, the paragraphs of a cell and the runs of a paragraph.

Two kinds of control hold no document text and are left out:

- a table of contents (``docPartGallery`` "Table of Contents") at block,
  row or cell level. A TOC control inside a paragraph is still read: the
  smart-heading pass takes it as TOC evidence and decides what to keep.
- a control still showing its placeholder (``w:showingPlcHdr``, "Click or
  tap here to enter text."), at every level.

The legacy extractor (``lightrag/parser/legacy/extractors.py``) applies the
same rules, except that it also drops a TOC control inside a paragraph.

A bibliography control (``w:bibliography``, or ``docPartGallery``
"Bibliographies") is read too, but its blocks are marked
``session_type="references"`` so the chunker's ``drop_references`` option
can drop them whatever their heading says. A table is marked only when all
its content lies in bibliography controls (:func:`is_bibliography_table`): a
table that mixes a bibliography cell with other content stays body, keeping
the bibliography text rather than dropping the rest.
"""

from docx.oxml.ns import qn

# A content control keeps its content in w:sdtContent; custom XML markup
# wraps it directly.
WRAPPERS = frozenset({qn("w:sdt"), qn("w:sdtContent"), qn("w:customXml")})

_SDT = qn("w:sdt")
_P = qn("w:p")
# Paragraph children that hold no content of their own.
_NON_CONTENT = frozenset(
    qn(f"w:{tag}")
    for tag in (
        "pPr",
        "bookmarkStart",
        "bookmarkEnd",
        "proofErr",
        "permStart",
        "permEnd",
    )
)


def is_toc_control(sdt) -> bool:
    """True for a ``w:sdt`` holding a table of contents."""
    sdt_pr = sdt.find(qn("w:sdtPr"))
    if sdt_pr is None:
        return False
    gallery = sdt_pr.find(f"{qn('w:docPartObj')}/{qn('w:docPartGallery')}")
    return gallery is not None and gallery.get(qn("w:val")) == "Table of Contents"


def is_placeholder_control(sdt) -> bool:
    """True for a ``w:sdt`` still showing its placeholder text."""
    sdt_pr = sdt.find(qn("w:sdtPr"))
    if sdt_pr is None:
        return False
    placeholder = sdt_pr.find(qn("w:showingPlcHdr"))
    return placeholder is not None and placeholder.get(qn("w:val")) not in (
        "0",
        "false",
        "off",
    )


def is_bibliography_control(sdt) -> bool:
    """True for a ``w:sdt`` holding a bibliography (Word's References tab)."""
    sdt_pr = sdt.find(qn("w:sdtPr"))
    if sdt_pr is None:
        return False
    if sdt_pr.find(qn("w:bibliography")) is not None:
        return True
    gallery = sdt_pr.find(f"{qn('w:docPartObj')}/{qn('w:docPartGallery')}")
    return gallery is not None and gallery.get(qn("w:val")) == "Bibliographies"


def in_bibliography(element) -> bool:
    """True when ``element`` sits inside a bibliography control."""
    parent = element.getparent()
    while parent is not None:
        if parent.tag == _SDT and is_bibliography_control(parent):
            return True
        parent = parent.getparent()
    return False


def is_bibliography_table(tbl) -> bool:
    """True for a ``w:tbl`` whose every paragraph with content is bibliography.

    Any child other than paragraph properties and position markers counts as
    content (text, equations, drawings, even an empty run), so a doubtful
    table stays body: misjudging it keeps a bibliography, never drops a cell.
    """
    found = False
    for paragraph in tbl.iter(_P):
        if all(child.tag in _NON_CONTENT for child in paragraph):
            continue
        if not in_bibliography(paragraph):
            return False
        found = True
    return found


def is_hidden_control(sdt) -> bool:
    """True for a block-, row- or cell-level ``w:sdt`` whose content is not read."""
    return is_toc_control(sdt) or is_placeholder_control(sdt)


def is_hidden_inline_control(node) -> bool:
    """True for an element inside a paragraph whose text is not read."""
    return node.tag == _SDT and is_placeholder_control(node)


def iter_blocks(container, tags: frozenset):
    """Yield the children of ``container`` whose tag is in ``tags``, in order.

    Includes the ones inside wrappers, except inside a control
    :func:`is_hidden_control` names. Use it for body blocks, table rows and
    the paragraphs of a cell. Never for cells: a hidden cell still takes its
    grid columns, so use :func:`iter_cells`.
    """
    for child in container:
        if child.tag in tags:
            yield child
        elif child.tag in WRAPPERS and not (
            child.tag == _SDT and is_hidden_control(child)
        ):
            yield from iter_blocks(child, tags)


def iter_cells(tr, hidden: bool = False):
    """Yield ``(w:tc, hidden)`` for each cell of a ``w:tr``, including wrapped ones.

    A cell inside a control :func:`is_hidden_control` names is still
    yielded, with ``hidden`` set: it takes its grid columns, so the cells
    after it keep theirs, and only its paraIds are read.
    """
    for child in tr:
        if child.tag == qn("w:tc"):
            yield child, hidden
        elif child.tag in WRAPPERS:
            yield from iter_cells(
                child,
                hidden or (child.tag == _SDT and is_hidden_control(child)),
            )
