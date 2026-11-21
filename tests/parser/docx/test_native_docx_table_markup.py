"""A table cell holding markup text must survive the native docx parse."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from docx import Document

from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE, PARSED_DIR_NAME
from lightrag.parser.base import ParseContext
from lightrag.parser.debug import build_debug_rag
from lightrag.parser.registry import get_parser

pytestmark = pytest.mark.offline


def _parse_native(source: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import lightrag.pipeline as pipeline_module

    async def _noop_archive(_p: str) -> None:
        return None

    monkeypatch.setenv("INPUT_DIR", str(source.parent))
    monkeypatch.setattr(
        pipeline_module, "archive_docx_source_after_full_docs_sync", _noop_archive
    )
    asyncio.run(
        get_parser("native").parse(
            ParseContext(
                build_debug_rag(),
                "doc-table-markup",
                str(source),
                {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
            )
        )
    )
    return source.parent / PARSED_DIR_NAME / f"{source.name}.parsed"


def test_cell_text_with_closing_table_tag_keeps_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [["Tag", "Meaning"], ["</table>", "closes an HTML table"], ["<tr>", "row"]]
    document = Document()
    table = document.add_table(rows=len(rows), cols=2)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            table.cell(r, c).text = text
    source = tmp_path / "markup.docx"
    document.save(source)

    parsed_dir = _parse_native(source, monkeypatch)

    tables = json.loads(
        (parsed_dir / "markup.tables.json").read_text(encoding="utf-8")
    )["tables"]
    assert len(tables) == 1
    entry = next(iter(tables.values()))
    assert entry["format"] == "json"
    assert entry["dimension"] == [3, 2]
    assert json.loads(entry["content"]) == rows


def test_cell_inner_markup_is_written_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the text ``</table>`` is escaped in a cell.

    A superscript run becomes ``<sup>…</sup>`` in the cell text; its closing
    tag must reach the block as written, like every other inner tag.
    """
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Room"
    table.cell(0, 1).text = "Area"
    table.cell(1, 0).text = "Lab"
    paragraph = table.cell(1, 1).paragraphs[0]
    paragraph.add_run("25 m")
    paragraph.add_run("2").font.superscript = True
    source = tmp_path / "markup.docx"
    document.save(source)

    parsed_dir = _parse_native(source, monkeypatch)

    lines = (parsed_dir / "markup.blocks.jsonl").read_text(encoding="utf-8")
    contents = [
        row["content"]
        for row in map(json.loads, lines.splitlines())
        if row.get("type") == "content"
    ]
    assert contents == [
        '<table id="tb-table-markup-0001" format="json">'
        '[["Room", "Area"], ["Lab", "25 m<sup>2</sup>"]]</table>'
    ]
