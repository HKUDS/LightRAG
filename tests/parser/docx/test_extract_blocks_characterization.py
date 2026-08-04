"""Characterization snapshots for ``extract_docx_blocks``.

Locks the CURRENT ``(blocks, warnings, metadata)`` output of the docx
extraction loop before the smart-heading read/assemble split refactor.
Each scenario is stored as a committed binary ``.docx`` (decoupled from the
installed python-docx version) plus a JSON snapshot of the extractor output;
the test replays the .docx through ``extract_docx_blocks`` and requires
strict equality.

The snapshots are one-shot: they were generated on the pre-refactor code and
must NOT be regenerated as part of a refactor — a diff here means the
smart-off path changed behavior.

Regenerate (only for a deliberate baseline change):

    python tests/parser/docx/test_extract_blocks_characterization.py --regen
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from lightrag.parser.docx.parse_document import extract_docx_blocks

pytestmark = pytest.mark.offline

FIXTURE_ROOT = Path(__file__).resolve().parent / "golden" / "extract_characterization"

_W14_PARA_ID = "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"


# ---------------------------------------------------------------------------
# scenario builders (used only by --regen; tests replay the committed .docx)
# ---------------------------------------------------------------------------


def _add_heading(doc: Document, text: str, level: int) -> None:
    """Paragraph with an explicit ``w:outlineLvl`` (1-based level)."""
    para = doc.add_paragraph(text)
    p_pr = para._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level - 1))
    p_pr.append(outline)


def _set_para_id(para, hex_id: str) -> None:
    para._p.set(_W14_PARA_ID, hex_id)


def _add_para_sectpr(para) -> None:
    """Paragraph-level section break (``w:pPr/w:sectPr``)."""
    p_pr = para._p.get_or_add_pPr()
    p_pr.append(OxmlElement("w:sectPr"))


def _mark_header_row(table, row_idx: int) -> None:
    tr = table.rows[row_idx]._tr
    tr_pr = tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:tblHeader"))


def _scenario_heading_hierarchy() -> Document:
    doc = Document()
    doc.add_paragraph("Preface body before any heading.")
    _add_heading(doc, "Chapter One", 1)
    doc.add_paragraph("Body under chapter one.")
    _add_heading(doc, "Section 1.1", 2)
    doc.add_paragraph("Body under section 1.1.")
    _add_heading(doc, "Subsection 1.1.1", 3)
    doc.add_paragraph("Deep body.")
    _add_heading(doc, "Section 1.2", 2)
    doc.add_paragraph("Body under section 1.2.")
    _add_heading(doc, "Chapter Two", 1)
    doc.add_paragraph("Body under chapter two.")
    return doc


def _scenario_numbered_list_basic() -> Document:
    doc = Document()
    _add_heading(doc, "Items", 1)
    doc.add_paragraph("First item", style="List Number")
    doc.add_paragraph("Second item", style="List Number")
    doc.add_paragraph("plain body between items")
    doc.add_paragraph("Third item", style="List Number")
    return doc


def _scenario_numbering_reset_around_table() -> Document:
    doc = Document()
    doc.add_paragraph("Alpha", style="List Number")
    doc.add_paragraph("Beta", style="List Number")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "cell A"
    table.cell(0, 1).text = "cell B"
    doc.add_paragraph("Gamma", style="List Number")
    doc.add_paragraph("Delta", style="List Number")
    return doc


def _scenario_numbered_list_with_empty_para() -> Document:
    doc = Document()
    doc.add_paragraph("One", style="List Number")
    doc.add_paragraph("", style="List Number")  # empty numbered para
    doc.add_paragraph("   ")  # whitespace-only plain para
    doc.add_paragraph("Two", style="List Number")
    return doc


def _scenario_sectpr_resets_numbering() -> Document:
    doc = Document()
    doc.add_paragraph("First", style="List Number")
    tail = doc.add_paragraph("Second ends the section", style="List Number")
    _add_para_sectpr(tail)
    doc.add_paragraph("After the break", style="List Number")
    return doc


def _scenario_oversize_heading_softbreak() -> Document:
    doc = Document()
    para = doc.add_paragraph()
    run = para.add_run("Short heading line kept as the heading")
    run.add_break(WD_BREAK.LINE)
    para.add_run("body remainder " * 20)
    p_pr = para._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "0")
    p_pr.append(outline)
    doc.add_paragraph("Following body paragraph.")
    return doc


def _scenario_oversize_heading_no_break() -> Document:
    doc = Document()
    _add_heading(doc, "An over-long single-line heading " * 10, 1)
    doc.add_paragraph("Following body paragraph.")
    return doc


def _scenario_paraid_mixed() -> Document:
    doc = Document()
    with_id = doc.add_paragraph("Heading with paraId")
    p_pr = with_id._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "0")
    p_pr.append(outline)
    _set_para_id(with_id, "1A2B3C4D")
    body = doc.add_paragraph("Body with paraId")
    _set_para_id(body, "5E6F7A8B")
    doc.add_paragraph("Body without paraId")
    return doc


def _scenario_empty_table_skipped() -> Document:
    doc = Document()
    doc.add_paragraph("Before the empty table.")
    doc.add_table(rows=2, cols=2)  # all cells whitespace-only
    doc.add_paragraph("After the empty table.")
    return doc


def _scenario_table_with_headers() -> Document:
    doc = Document()
    _add_heading(doc, "Data", 1)
    table = doc.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "alpha"
    table.cell(1, 1).text = "1"
    table.cell(2, 0).text = "beta"
    table.cell(2, 1).text = "2"
    _mark_header_row(table, 0)
    doc.add_paragraph("After the table.")
    return doc


def _scenario_supsub_and_soft_break() -> Document:
    doc = Document()
    para = doc.add_paragraph()
    para.add_run("E = mc")
    sup = para.add_run("2")
    sup.font.superscript = True
    para.add_run(" and H")
    sub = para.add_run("2")
    sub.font.subscript = True
    para.add_run("O")
    multi = doc.add_paragraph()
    multi.add_run("line one")
    multi.runs[0].add_break(WD_BREAK.LINE)
    multi.add_run("line two")
    return doc


def _scenario_heading_only_tail() -> Document:
    doc = Document()
    doc.add_paragraph("Some body.")
    _add_heading(doc, "Trailing heading with no body", 1)
    return doc


SCENARIOS = {
    "heading_hierarchy": _scenario_heading_hierarchy,
    "numbered_list_basic": _scenario_numbered_list_basic,
    "numbering_reset_around_table": _scenario_numbering_reset_around_table,
    "numbered_list_with_empty_para": _scenario_numbered_list_with_empty_para,
    "sectpr_resets_numbering": _scenario_sectpr_resets_numbering,
    "oversize_heading_softbreak": _scenario_oversize_heading_softbreak,
    "oversize_heading_no_break": _scenario_oversize_heading_no_break,
    "paraid_mixed": _scenario_paraid_mixed,
    "empty_table_skipped": _scenario_empty_table_skipped,
    "table_with_headers": _scenario_table_with_headers,
    "supsub_and_soft_break": _scenario_supsub_and_soft_break,
    "heading_only_tail": _scenario_heading_only_tail,
}


# ---------------------------------------------------------------------------
# replay + snapshot plumbing
# ---------------------------------------------------------------------------


def _run_extractor(docx_path: Path) -> dict:
    warnings: dict = {}
    metadata: dict = {}
    blocks = extract_docx_blocks(
        str(docx_path), parse_warnings=warnings, parse_metadata=metadata
    )
    return {"blocks": blocks, "warnings": warnings, "metadata": metadata}


def _snapshot_paths(name: str) -> tuple[Path, Path]:
    return FIXTURE_ROOT / f"{name}.docx", FIXTURE_ROOT / f"{name}.json"


@pytest.mark.parametrize("name", sorted(SCENARIOS), ids=sorted(SCENARIOS))
def test_extract_blocks_matches_snapshot(name: str) -> None:
    docx_path, json_path = _snapshot_paths(name)
    assert docx_path.is_file() and json_path.is_file(), (
        f"missing characterization fixture for {name!r}; run "
        "python tests/parser/docx/test_extract_blocks_characterization.py --regen"
    )
    expected = json.loads(json_path.read_text(encoding="utf-8"))
    actual = _run_extractor(docx_path)
    # Normalize through JSON so tuple/list and int/float representations
    # cannot cause spurious mismatches.
    actual = json.loads(json.dumps(actual, ensure_ascii=False))
    assert actual == expected


def _regen() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    for name, builder in sorted(SCENARIOS.items()):
        docx_path, json_path = _snapshot_paths(name)
        buf = io.BytesIO()
        builder().save(buf)
        docx_path.write_bytes(buf.getvalue())
        snapshot = _run_extractor(docx_path)
        json_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"regenerated {name}")


if __name__ == "__main__":
    import sys

    if "--regen" in sys.argv:
        _regen()
    else:
        print(__doc__)
