"""PaddleOCR-VL IR builder tests: official JSON output -> IRDoc."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.parser.external.paddleocr_vl import PaddleOCRVLIRBuilder


SAMPLE = (
    Path(__file__).resolve().parents[4]
    / "lightrag"
    / "parser"
    / "external"
    / "paddleocr_vl"
    / "2410.05779v3.pdf_by_PaddleOCR-VL-1.6.json"
)
_HAS_SAMPLE = SAMPLE.is_file()


def _write_bundle(tmp_path: Path, payload: object) -> Path:
    raw_dir = tmp_path / "demo.paddleocr_vl_raw"
    raw_dir.mkdir()
    (raw_dir / "content_list.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return raw_dir


@pytest.mark.skipif(not _HAS_SAMPLE, reason="sample JSON file not present")
@pytest.mark.offline
def test_sample_json_builds_structured_ir(tmp_path: Path) -> None:
    raw_dir = _write_bundle(tmp_path, json.loads(SAMPLE.read_text(encoding="utf-8")))

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="2410.05779v3.pdf"
    )

    assert ir.document_format == "pdf"
    assert ir.doc_title == "LIGHTRAG: SIMPLE AND FAST RETRIEVAL-AUGMENTED GENERATION"
    assert ir.split_option == {"engine_version": "PaddleOCR-VL"}
    assert ir.bbox_attributes == {"origin": "LEFTTOP"}
    assert len(ir.blocks) > 8

    first = ir.blocks[0]
    assert first.heading == "LIGHTRAG: SIMPLE AND FAST RETRIEVAL-AUGMENTED GENERATION"
    assert first.level == 1
    assert first.content_template.splitlines()[0] == (
        "# LIGHTRAG: SIMPLE AND FAST RETRIEVAL-AUGMENTED GENERATION"
    )

    abstract = next(block for block in ir.blocks if block.heading == "ABSTRACT")
    assert abstract.parent_headings == [
        "LIGHTRAG: SIMPLE AND FAST RETRIEVAL-AUGMENTED GENERATION"
    ]
    assert "Retrieval-Augmented Generation" in abstract.content_template
    assert any(pos.type == "bbox" and pos.anchor == "1" for pos in abstract.positions)

    table_block = next(block for block in ir.blocks if block.tables)
    table = table_block.tables[0]
    assert table.html and "<table" in table.html
    assert table.self_ref.startswith("result.json#/")
    assert f"{{{{TBL:{table.placeholder_key}}}}}" in table_block.content_template

    drawing_block = next(block for block in ir.blocks if block.drawings)
    drawing = drawing_block.drawings[0]
    assert drawing.asset_ref.startswith("imgs/")
    assert drawing.caption or drawing.src
    assert f"{{{{IMG:{drawing.placeholder_key}}}}}" in drawing_block.content_template

    equation_block = next(block for block in ir.blocks if block.equations)
    equation = equation_block.equations[0]
    assert equation.is_block is True
    assert "\\mathcal{M}" in equation.latex
    assert f"{{{{EQ:{equation.placeholder_key}}}}}" in equation_block.content_template


@pytest.mark.offline
def test_inline_payload_merges_blocks_and_skips_layout_noise(tmp_path: Path) -> None:
    raw_dir = _write_bundle(
        tmp_path,
        [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "number",
                            "block_content": "1",
                            "block_bbox": [1, 2, 3, 4],
                        },
                        {
                            "block_label": "doc_title",
                            "block_content": "# Title",
                            "block_bbox": [10, 20, 30, 40],
                        },
                        {
                            "block_label": "text",
                            "block_content": "Intro body.",
                            "block_bbox": [11, 21, 31, 41],
                        },
                        {
                            "block_label": "paragraph_title",
                            "block_content": "## Section",
                            "block_bbox": [12, 22, 32, 42],
                        },
                        {
                            "block_label": "display_formula",
                            "block_content": "$$x=1$$",
                            "block_bbox": [13, 23, 33, 43],
                        },
                    ]
                }
            }
        ],
    )

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    assert [block.heading for block in ir.blocks] == ["Title", "Section"]
    assert ir.blocks[0].content_template == "# Title\nIntro body."
    assert all(
        line != "1"
        for block in ir.blocks
        for line in block.content_template.splitlines()
    )
    assert ir.blocks[1].parent_headings == ["Title"]
    assert len(ir.blocks[1].equations) == 1


@pytest.mark.offline
def test_adjacent_figure_title_becomes_table_or_image_caption(tmp_path: Path) -> None:
    raw_dir = _write_bundle(
        tmp_path,
        [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "doc_title",
                            "block_content": "Title",
                            "block_bbox": [10, 20, 30, 40],
                        },
                        {
                            "block_label": "figure_title",
                            "block_content": "Table 1: Results.",
                            "block_bbox": [10, 50, 90, 60],
                        },
                        {
                            "block_label": "table",
                            "block_content": "<table><tr><td>A</td></tr></table>",
                            "block_bbox": [10, 70, 90, 110],
                        },
                        {
                            "block_label": "image",
                            "block_content": "",
                            "block_bbox": [10, 120, 90, 170],
                        },
                        {
                            "block_label": "figure_title",
                            "block_content": "Figure 1: Architecture.",
                            "block_bbox": [10, 180, 90, 190],
                        },
                    ]
                },
                "markdown": {
                    "images": {
                        "imgs/img_in_image_box_10_120_90_170.jpg": "ignored-url"
                    }
                },
            }
        ],
    )

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    block = ir.blocks[0]
    assert block.tables[0].caption == "Table 1: Results."
    assert block.drawings[0].caption == "Figure 1: Architecture."
    assert "Table 1: Results." not in block.content_template
    assert "Figure 1: Architecture." not in block.content_template
