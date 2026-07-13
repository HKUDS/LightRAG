"""PaddleOCR-VL IR builder tests: official JSON output -> IRDoc."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.parser.external._manifest import Manifest, ManifestFile, write_manifest
from lightrag.parser.external.paddleocr_vl import PaddleOCRVLIRBuilder


def _write_bundle(tmp_path: Path, payload: object) -> Path:
    raw_dir = tmp_path / "demo.paddleocr_vl_raw"
    raw_dir.mkdir()
    (raw_dir / "content_list.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return raw_dir


def _write_manifest(raw_dir: Path, *, page_ranges: str) -> None:
    result = raw_dir / "content_list.json"
    write_manifest(
        raw_dir,
        Manifest(
            engine="paddleocr_vl",
            source_content_hash="sha256:test",
            source_size_bytes=1,
            source_filename_at_parse="demo.pdf",
            critical_file=ManifestFile("content_list.json", result.stat().st_size),
            files=[],
            total_size_bytes=result.stat().st_size,
            extras={"page_ranges": page_ranges},
        ),
    )


def _sample_payload() -> list[dict]:
    return [
        {
            "prunedResult": {
                "parsing_res_list": [
                    {
                        "block_label": "doc_title",
                        "block_content": "Demo Paper",
                        "block_bbox": [10, 20, 30, 40],
                    },
                    {
                        "block_label": "text",
                        "block_content": "Intro body.",
                        "block_bbox": [10, 50, 100, 70],
                    },
                    {
                        "block_label": "table",
                        "block_content": "<table><tr><td>A</td></tr></table>",
                        "block_bbox": [10, 80, 100, 120],
                    },
                    {
                        "block_label": "image",
                        "block_content": "",
                        "block_bbox": [10, 130, 100, 170],
                    },
                    {
                        "block_label": "display_formula",
                        "block_content": "$$x=1$$",
                        "block_bbox": [10, 180, 100, 200],
                    },
                ]
            },
            "markdown": {
                "images": {"imgs/img_in_image_box_10_130_100_170.jpg": "ignored"}
            },
        }
    ]


@pytest.mark.offline
def test_sample_payload_builds_structured_ir(tmp_path: Path) -> None:
    raw_dir = _write_bundle(tmp_path, _sample_payload())

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="sample.pdf"
    )

    assert ir.document_format == "pdf"
    assert ir.doc_title
    assert ir.split_option.get("engine_version")
    assert ir.bbox_attributes == {"origin": "LEFTTOP"}
    assert ir.blocks

    first = ir.blocks[0]
    assert first.heading == ir.doc_title
    assert first.level == 1
    assert first.content_template.splitlines()[0] == f"# {ir.doc_title}"
    assert any(pos.type == "bbox" and pos.anchor == "1" for pos in first.positions)

    table_block = next(block for block in ir.blocks if block.tables)
    table = table_block.tables[0]
    assert table.html and "<table" in table.html
    assert table.self_ref.startswith("content_list.json#/")
    assert f"{{{{TBL:{table.placeholder_key}}}}}" in table_block.content_template

    drawing_block = next(block for block in ir.blocks if block.drawings)
    drawing = drawing_block.drawings[0]
    assert drawing.asset_ref.startswith("imgs/")
    assert drawing.caption or drawing.src
    assert f"{{{{IMG:{drawing.placeholder_key}}}}}" in drawing_block.content_template


@pytest.mark.offline
def test_page_anchors_follow_returned_order_not_source_page_ranges(
    tmp_path: Path,
) -> None:
    raw_dir = _write_bundle(
        tmp_path,
        [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "text",
                            "block_content": "Source page 5",
                            "block_bbox": [10, 20, 30, 40],
                        }
                    ]
                }
            },
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "text",
                            "block_content": "Source page 6",
                            "block_bbox": [10, 20, 30, 40],
                        }
                    ]
                }
            },
        ],
    )
    _write_manifest(raw_dir, page_ranges="5-6")

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    assert [position.anchor for position in ir.blocks[0].positions] == ["1", "2"]


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
                    "images": {"imgs/img_in_image_box_10_120_90_170.jpg": "ignored-url"}
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


@pytest.mark.offline
def test_caption_matches_across_ignorable_layout_noise(tmp_path: Path) -> None:
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
                            "block_label": "number",
                            "block_content": "1",
                            "block_bbox": [92, 50, 96, 60],
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
                            "block_label": "header",
                            "block_content": "demo.pdf",
                            "block_bbox": [10, 5, 90, 15],
                        },
                        {
                            "block_label": "figure_title",
                            "block_content": "Figure 1: Architecture.",
                            "block_bbox": [10, 180, 90, 190],
                        },
                    ]
                },
                "markdown": {
                    "images": {"imgs/img_in_image_box_10_120_90_170.jpg": "ignored-url"}
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


@pytest.mark.offline
def test_caption_matches_across_multiple_skip_labels(tmp_path: Path) -> None:
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
                            "block_label": "number",
                            "block_content": "1",
                            "block_bbox": [92, 50, 96, 60],
                        },
                        {
                            "block_label": "header",
                            "block_content": "demo.pdf",
                            "block_bbox": [10, 5, 90, 15],
                        },
                        {
                            "block_label": "footer",
                            "block_content": "footer",
                            "block_bbox": [10, 185, 90, 195],
                        },
                        {
                            "block_label": "formula_number",
                            "block_content": "(1)",
                            "block_bbox": [92, 70, 96, 80],
                        },
                        {
                            "block_label": "table",
                            "block_content": "<table><tr><td>A</td></tr></table>",
                            "block_bbox": [10, 70, 90, 110],
                        },
                    ]
                },
            }
        ],
    )

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    block = ir.blocks[0]
    assert block.tables[0].caption == "Table 1: Results."
    assert "Table 1: Results." not in block.content_template


@pytest.mark.offline
def test_caption_does_not_match_across_non_skip_content(tmp_path: Path) -> None:
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
                            "block_label": "number",
                            "block_content": "1",
                            "block_bbox": [92, 50, 96, 60],
                        },
                        {
                            "block_label": "text",
                            "block_content": "Intervening body text.",
                            "block_bbox": [10, 62, 90, 68],
                        },
                        {
                            "block_label": "table",
                            "block_content": "<table><tr><td>A</td></tr></table>",
                            "block_bbox": [10, 70, 90, 110],
                        },
                    ]
                },
            }
        ],
    )

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    block = ir.blocks[0]
    assert block.tables[0].caption == ""
    assert "Table 1: Results." in block.content_template


@pytest.mark.offline
def test_paddleocr_vl_known_labels_map_to_ir_actions(tmp_path: Path) -> None:
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
                            "block_label": "algorithm",
                            "block_content": "for i in range(3): pass",
                            "block_bbox": [10, 50, 90, 60],
                        },
                        {
                            "block_label": "content",
                            "block_content": "Content block.",
                            "block_bbox": [10, 62, 90, 68],
                        },
                        {
                            "block_label": "reference",
                            "block_content": "References",
                            "block_bbox": [10, 70, 90, 80],
                        },
                        {
                            "block_label": "vision_footnote",
                            "block_content": "Visual footnote.",
                            "block_bbox": [10, 82, 90, 90],
                        },
                        {
                            "block_label": "inline_formula",
                            "block_content": "x+y",
                            "block_bbox": [10, 92, 90, 100],
                        },
                        {
                            "block_label": "chart",
                            "block_content": "",
                            "block_bbox": [10, 110, 90, 150],
                        },
                        {
                            "block_label": "seal",
                            "block_content": "",
                            "block_bbox": [10, 160, 90, 190],
                        },
                    ]
                },
                "markdown": {
                    "images": {
                        "imgs/img_in_image_box_10_110_90_150.jpg": "ignored",
                        "imgs/img_in_image_box_10_160_90_190.jpg": "ignored",
                    }
                },
            }
        ],
    )

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    block = ir.blocks[0]
    assert "for i in range(3): pass" in block.content_template
    assert "Content block." in block.content_template
    assert "References" in block.content_template
    assert "Visual footnote." in block.content_template
    assert len(block.equations) == 1
    assert block.equations[0].latex == "x+y"
    assert block.equations[0].is_block is False
    assert f"{{{{EQI:{block.equations[0].placeholder_key}}}}}" in block.content_template
    assert [drawing.asset_ref for drawing in block.drawings] == [
        "imgs/img_in_image_box_10_110_90_150.jpg",
        "imgs/img_in_image_box_10_160_90_190.jpg",
    ]


@pytest.mark.offline
def test_poly_bbox_is_normalized_to_enclosing_rectangle(tmp_path: Path) -> None:
    raw_dir = _write_bundle(
        tmp_path,
        [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "doc_title",
                            "block_content": "Skewed",
                            "block_bbox": [[10, 20], [30, 18], [42, 55], [8, 60]],
                        }
                    ]
                }
            }
        ],
    )

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    assert ir.blocks[0].positions[0].range == [8, 18, 42, 60]


@pytest.mark.offline
def test_poly_bbox_image_uses_enclosing_rectangle_for_markdown_asset(
    tmp_path: Path,
) -> None:
    raw_dir = _write_bundle(
        tmp_path,
        [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "doc_title",
                            "block_content": "Skewed image",
                            "block_bbox": [10, 20, 30, 40],
                        },
                        {
                            "block_label": "image",
                            "block_content": "",
                            "block_bbox": [
                                [10, 20],
                                [30, 18],
                                [42, 55],
                                [8, 60],
                            ],
                        },
                    ]
                },
                "markdown": {
                    "images": {
                        "imgs/img_in_image_box_8_18_42_60.jpg": "ignored-url",
                    }
                },
            }
        ],
    )

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == "imgs/img_in_image_box_8_18_42_60.jpg"
    assert ir.blocks[0].positions[-1].range == [8, 18, 42, 60]


@pytest.mark.offline
def test_page_indexed_image_name_preserves_bbox_mapping_and_asset(
    tmp_path: Path,
) -> None:
    image_ref = "imgs/img_in_image_box_10_20_30_40_0.jpg"
    raw_dir = _write_bundle(
        tmp_path,
        [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "image",
                            "block_content": "",
                            "block_bbox": [10, 20, 30, 40],
                        }
                    ]
                },
                "markdown": {"images": {image_ref: "ignored-url"}},
            }
        ],
    )
    image_path = raw_dir / image_ref
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"page-zero")

    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == image_ref
    asset = next(asset for asset in ir.assets if asset.ref == image_ref)
    assert asset.source == image_path


@pytest.mark.offline
def test_paragraph_title_without_hash_defaults_to_level_2(tmp_path: Path) -> None:
    # A paragraph_title with no markdown '#' prefix falls back to level 2.
    raw_dir = _write_bundle(
        tmp_path,
        [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "doc_title",
                            "block_content": "Doc Title",
                            "block_bbox": [1, 2, 3, 4],
                        },
                        {
                            "block_label": "paragraph_title",
                            "block_content": "Bare Section Name",
                            "block_bbox": [5, 6, 7, 8],
                        },
                    ]
                }
            }
        ],
    )
    ir = PaddleOCRVLIRBuilder().normalize_from_workdir(
        raw_dir, document_name="demo.pdf"
    )
    assert len(ir.blocks) == 2
    assert ir.blocks[1].level == 2
    assert ir.blocks[1].parent_headings == ["Doc Title"]
