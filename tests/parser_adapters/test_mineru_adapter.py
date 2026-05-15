"""MinerU adapter tests: content_list.json → IR translation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.parser_adapters import MinerUAdapter


def _write_bundle(tmp_path: Path, content_list: list[dict]) -> Path:
    """Build a minimal *.mineru_raw/ directory."""
    raw = tmp_path / "doc.mineru_raw"
    raw.mkdir()
    (raw / "content_list.json").write_text(
        json.dumps(content_list, ensure_ascii=False)
    )
    return raw


@pytest.mark.offline
def test_adapter_simple_text_and_heading(tmp_path: Path) -> None:
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "1 Introduction", "text_level": 1},
            {"type": "text", "text": "Body paragraph."},
            {"type": "text", "text": "1.1 Sub", "text_level": 2},
            {"type": "text", "text": "Sub body."},
        ],
    )
    ir = MinerUAdapter().normalize_from_workdir(raw, document_name="x.pdf")

    assert ir.doc_title == "1 Introduction"
    assert ir.document_format == "pdf"
    assert len(ir.blocks) == 4
    assert ir.blocks[0].heading == "1 Introduction"
    assert ir.blocks[0].level == 1
    # Body inherits the current heading + level.
    assert ir.blocks[1].heading == "1 Introduction"
    assert ir.blocks[1].level == 1
    # Sub-heading updates stack and records parent.
    assert ir.blocks[2].heading == "1.1 Sub"
    assert ir.blocks[2].level == 2
    assert ir.blocks[2].parent_headings == ["1 Introduction"]


@pytest.mark.offline
def test_adapter_table_and_drawing_and_equation(tmp_path: Path) -> None:
    raw = _write_bundle(
        tmp_path,
        [
            {
                "type": "table",
                "table_body": [["a", "b"], ["1", "2"]],
                "num_rows": 2,
                "num_cols": 2,
                "table_caption": ["Tbl"],
                "header": [["a", "b"]],
            },
            {
                "type": "image",
                "img_path": "images/img_001.jpg",
                "image_caption": ["Fig 1"],
                "page_idx": 1,
                "bbox": [10, 20, 30, 40],
            },
            {"type": "equation", "text": "$E = mc^2$", "caption": "Eq 1"},
        ],
    )
    # The drawing references images/img_001.jpg — adapter accepts missing
    # files and produces an AssetSpec with source=None.
    ir = MinerUAdapter().normalize_from_workdir(raw, document_name="d.pdf")

    table_block = next(b for b in ir.blocks if b.tables)
    table = table_block.tables[0]
    assert table.rows == [["a", "b"], ["1", "2"]]
    assert table.num_rows == 2 and table.num_cols == 2
    assert table.caption == "Tbl"
    assert table.table_header == [["a", "b"]]

    drawing_block = next(b for b in ir.blocks if b.drawings)
    drawing = drawing_block.drawings[0]
    assert drawing.fmt == "jpg"
    assert drawing.caption == "Fig 1"
    # Position carried through.
    assert drawing_block.positions[0].type == "bbox"
    assert drawing_block.positions[0].anchor == 2  # page_idx+1
    assert drawing_block.positions[0].range == [10.0, 20.0, 30.0, 40.0]

    # Asset is declared with the relative path as ref.
    assert any(a.ref == "images/img_001.jpg" for a in ir.assets)

    equation_block = next(b for b in ir.blocks if b.equations)
    eq = equation_block.equations[0]
    assert eq.latex == "E = mc^2"  # $..$ stripped
    assert eq.is_block is True
    assert eq.caption == "Eq 1"


@pytest.mark.offline
def test_adapter_empty_equation_dropped(tmp_path: Path) -> None:
    """Fix 2: equation items with empty text MUST NOT enter the IR (and
    consequently not the sidecar). They previously left dangling sidecar
    entries."""
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "equation", "text": "", "caption": "ghost"},
            {"type": "equation", "text": "   ", "caption": "ghost"},
            {"type": "text", "text": "kept"},
        ],
    )
    ir = MinerUAdapter().normalize_from_workdir(raw, document_name="g.pdf")
    eq_count = sum(len(b.equations) for b in ir.blocks)
    assert eq_count == 0
    assert any(b.content_template == "kept" for b in ir.blocks)


@pytest.mark.offline
def test_adapter_bbox_attributes_default_and_override(tmp_path: Path) -> None:
    raw = _write_bundle(tmp_path, [{"type": "text", "text": "x"}])
    adapter = MinerUAdapter()
    ir = adapter.normalize_from_workdir(raw, document_name="x.pdf")
    assert ir.bbox_attributes == {"origin": "LEFTTOP", "max": 1000}


@pytest.mark.offline
def test_adapter_bbox_attributes_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "MINERU_BBOX_ATTRIBUTES",
        '{"origin": "LEFTBOTTOM", "max": 612}',
    )
    raw = _write_bundle(tmp_path, [{"type": "text", "text": "x"}])
    adapter = MinerUAdapter()
    ir = adapter.normalize_from_workdir(raw, document_name="x.pdf")
    assert ir.bbox_attributes == {"origin": "LEFTBOTTOM", "max": 612}


@pytest.mark.offline
def test_adapter_engine_version_recorded_in_split_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINERU_ENGINE_VERSION", "magic-pdf 1.5.4")
    raw = _write_bundle(tmp_path, [{"type": "text", "text": "x"}])
    ir = MinerUAdapter().normalize_from_workdir(raw, document_name="x.pdf")
    assert ir.split_option == {"engine_version": "magic-pdf 1.5.4"}


@pytest.mark.offline
def test_adapter_missing_content_list_raises(tmp_path: Path) -> None:
    raw_dir = tmp_path / "bad.mineru_raw"
    raw_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        MinerUAdapter().normalize_from_workdir(raw_dir, document_name="x.pdf")


@pytest.mark.offline
def test_adapter_html_table_fallback(tmp_path: Path) -> None:
    """If table_body is a string that is not JSON, treat as HTML and keep
    on IRTable.html so the writer emits format="html"."""
    raw = _write_bundle(
        tmp_path,
        [
            {
                "type": "table",
                "table_body": "<table><tr><td>a</td></tr></table>",
                "num_rows": 1,
                "num_cols": 1,
            }
        ],
    )
    ir = MinerUAdapter().normalize_from_workdir(raw, document_name="h.pdf")
    table = ir.blocks[0].tables[0]
    assert table.rows is None
    assert table.html and "<td>a</td>" in table.html


@pytest.mark.offline
def test_adapter_list_items_joined_with_newline(tmp_path: Path) -> None:
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "list", "list_items": ["one", "two", "three"]},
        ],
    )
    ir = MinerUAdapter().normalize_from_workdir(raw, document_name="l.pdf")
    assert ir.blocks[0].content_template == "one\ntwo\nthree"


@pytest.mark.offline
def test_adapter_drawing_asset_source_only_when_file_exists(
    tmp_path: Path,
) -> None:
    """The adapter should declare an AssetSpec for the drawing in both
    cases, but ``source`` is set only when the bytes are on disk; the
    writer then warns and skips a missing-source asset."""
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "image", "img_path": "images/exists.png"},
            {"type": "image", "img_path": "images/missing.png"},
        ],
    )
    (raw / "images").mkdir()
    (raw / "images" / "exists.png").write_bytes(b"\x89PNG")

    ir = MinerUAdapter().normalize_from_workdir(raw, document_name="a.pdf")
    by_ref = {a.ref: a for a in ir.assets}
    assert by_ref["images/exists.png"].source is not None
    assert by_ref["images/missing.png"].source is None
