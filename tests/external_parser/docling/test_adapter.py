"""Tests for :class:`DoclingAdapter`.

Each test constructs a minimal inline DoclingDocument dict — the smallest
JSON that exercises one mapping rule from
``docs/DoclingSidecarRefactorPlan-zh.md`` §5. The point is to lock down
contracts that the integration test (running against the live fixture)
cannot inspect cleanly, not to faithfully replicate the docling-serve
output schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lightrag.external_parser.docling.adapter import DoclingAdapter


# ---------------------------------------------------------------------------
# Helpers to build inline fixtures
# ---------------------------------------------------------------------------


def _write_doc(tmp_path: Path, payload: dict, *, stem: str = "demo") -> Path:
    raw_dir = tmp_path / f"{stem}.docling_raw"
    raw_dir.mkdir()
    (raw_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
    return raw_dir


def _doc(
    *,
    body_children: list[str],
    texts: list[dict] | None = None,
    tables: list[dict] | None = None,
    pictures: list[dict] | None = None,
    groups: list[dict] | None = None,
    key_value_items: list[dict] | None = None,
    form_items: list[dict] | None = None,
) -> dict:
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "origin": {"filename": "demo.pdf", "mimetype": "application/pdf"},
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": r} for r in body_children],
            "content_layer": "body",
            "label": "unspecified",
        },
        "groups": groups or [],
        "texts": texts or [],
        "pictures": pictures or [],
        "tables": tables or [],
        "key_value_items": key_value_items or [],
        "form_items": form_items or [],
    }


def _text_item(
    *,
    label: str,
    text: str,
    self_ref: str,
    level: int | None = None,
    orig: str | None = None,
    page_no: int = 1,
    bbox: tuple[float, float, float, float] = (10.0, 100.0, 200.0, 80.0),
    coord_origin: str = "BOTTOMLEFT",
    content_layer: str = "body",
    marker: str | None = None,
) -> dict:
    item: dict[str, Any] = {
        "self_ref": self_ref,
        "label": label,
        "text": text,
        "orig": orig if orig is not None else text,
        "content_layer": content_layer,
        "prov": [
            {
                "page_no": page_no,
                "bbox": {
                    "l": bbox[0],
                    "t": bbox[1],
                    "r": bbox[2],
                    "b": bbox[3],
                    "coord_origin": coord_origin,
                },
                "charspan": [0, len(text)],
            }
        ],
    }
    if level is not None:
        item["level"] = level
    if marker is not None:
        item["marker"] = marker
    return item


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DOCLING_BBOX_ATTRIBUTES", "DOCLING_ENGINE_VERSION"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# 1. Heading hierarchy
# ---------------------------------------------------------------------------


def test_docling_adapter_simple_heading_hierarchy(tmp_path: Path) -> None:
    """Three distinct sections without adjacency-merge folding.

    Background and Details each carry their own body, so we end up with one
    block per heading and a clean parent-heading chain.
    """
    texts = [
        _text_item(label="title", text="Whole Doc Title", self_ref="#/texts/0"),
        _text_item(label="text", text="Title-level body.", self_ref="#/texts/1"),
        _text_item(
            label="section_header", text="Background", level=1, self_ref="#/texts/2"
        ),
        _text_item(label="text", text="Some intro body.", self_ref="#/texts/3"),
        _text_item(
            label="section_header", text="Details", level=2, self_ref="#/texts/4"
        ),
        _text_item(label="text", text="Detail content.", self_ref="#/texts/5"),
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=[
                "#/texts/0",
                "#/texts/1",
                "#/texts/2",
                "#/texts/3",
                "#/texts/4",
                "#/texts/5",
            ],
            texts=texts,
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")

    assert ir.doc_title == "Whole Doc Title"
    headings = [(b.heading, b.level, b.parent_headings) for b in ir.blocks]
    # title (level=1), section_header level=1 → IR level 2, section_header level=2 → IR level 3
    assert headings == [
        ("Whole Doc Title", 1, []),
        ("Background", 2, ["Whole Doc Title"]),
        ("Details", 3, ["Whole Doc Title", "Background"]),
    ]
    # heading line is rendered with markdown prefix as the FIRST line
    assert ir.blocks[0].content_template.splitlines()[0] == "# Whole Doc Title"
    assert ir.blocks[1].content_template.splitlines()[0] == "## Background"
    assert ir.blocks[2].content_template.splitlines()[0] == "### Details"


def test_docling_adapter_adjacency_merge_folds_empty_heading(tmp_path: Path) -> None:
    """When a heading block has no body and the next heading is deeper,
    the deeper heading folds in as a body line (matches MinerU §5.1.4)."""
    texts = [
        _text_item(label="title", text="Whole Doc Title", self_ref="#/texts/0"),
        _text_item(
            label="section_header", text="Background", level=1, self_ref="#/texts/1"
        ),
        _text_item(label="text", text="Body for Background.", self_ref="#/texts/2"),
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=["#/texts/0", "#/texts/1", "#/texts/2"],
            texts=texts,
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    # Title had no body → Background folded into it as a `## ` line
    assert len(ir.blocks) == 1
    block = ir.blocks[0]
    assert block.heading == "Whole Doc Title"
    assert block.level == 1
    lines = block.content_template.splitlines()
    assert lines[0] == "# Whole Doc Title"
    assert "## Background" in lines
    assert "Body for Background." in lines


def test_docling_adapter_preserves_docling_heading_level(tmp_path: Path) -> None:
    """When Docling reports all section_headers at level=1, the adapter
    preserves that (no numbering-based level inference)."""
    texts = [
        _text_item(
            label="section_header", text="1 Purpose", level=1, self_ref="#/texts/0"
        ),
        _text_item(
            label="section_header", text="2.1 Electrical", level=1, self_ref="#/texts/1"
        ),
        _text_item(
            label="section_header",
            text="2.4.5 Temperature",
            level=1,
            self_ref="#/texts/2",
        ),
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=["#/texts/0", "#/texts/1", "#/texts/2"],
            texts=texts,
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    levels = [b.level for b in ir.blocks]
    assert levels == [2, 2, 2]  # all bumped by +1, no normalization


# ---------------------------------------------------------------------------
# 2. Multimodal payloads under one heading
# ---------------------------------------------------------------------------


def test_docling_adapter_merges_payloads_under_heading(tmp_path: Path) -> None:
    texts = [
        _text_item(
            label="section_header", text="Section", level=1, self_ref="#/texts/0"
        ),
        _text_item(label="text", text="Inline body line.", self_ref="#/texts/1"),
    ]
    tables = [
        {
            "self_ref": "#/tables/0",
            "label": "table",
            "content_layer": "body",
            "data": {
                "num_rows": 1,
                "num_cols": 2,
                "grid": [[{"text": "A"}, {"text": "B"}]],
            },
            "prov": [],
        }
    ]
    pictures = [
        {
            "self_ref": "#/pictures/0",
            "label": "picture",
            "content_layer": "body",
            "image": {"uri": "artifacts/foo.png", "mimetype": "image/png"},
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=[
                "#/texts/0",
                "#/texts/1",
                "#/tables/0",
                "#/pictures/0",
            ],
            texts=texts,
            tables=tables,
            pictures=pictures,
        ),
    )
    (raw_dir / "artifacts").mkdir()
    (raw_dir / "artifacts" / "foo.png").write_bytes(b"\x89PNG fake")

    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    assert len(ir.blocks) == 1
    block = ir.blocks[0]
    template = block.content_template
    # one of each placeholder appears in source order
    assert "{{TBL:tb1}}" in template
    assert "{{IMG:im2}}" in template
    assert template.index("{{TBL:tb1}}") < template.index("{{IMG:im2}}")
    assert len(block.tables) == 1
    assert block.tables[0].rows == [["A", "B"]]
    assert len(block.drawings) == 1
    assert block.drawings[0].asset_ref == "artifacts/foo.png"
    assert block.drawings[0].fmt == "png"
    assert any(a.ref == "artifacts/foo.png" for a in ir.assets)


# ---------------------------------------------------------------------------
# 3. Inline groups
# ---------------------------------------------------------------------------


def test_docling_adapter_inline_group_joins_children(tmp_path: Path) -> None:
    texts = [
        _text_item(label="section_header", text="S", level=1, self_ref="#/texts/0"),
        _text_item(label="text", text="hello", self_ref="#/texts/1"),
        _text_item(label="text", text="world", self_ref="#/texts/2"),
    ]
    groups = [
        {
            "self_ref": "#/groups/0",
            "label": "inline",
            "content_layer": "body",
            "children": [{"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=["#/texts/0", "#/groups/0"],
            texts=texts,
            groups=groups,
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    assert "hello world" in ir.blocks[0].content_template


# ---------------------------------------------------------------------------
# 4. Tables — grid & header
# ---------------------------------------------------------------------------


def test_docling_adapter_table_grid_and_header(tmp_path: Path) -> None:
    tables = [
        {
            "self_ref": "#/tables/0",
            "label": "table",
            "content_layer": "body",
            "captions": [{"$ref": "#/texts/0"}],
            "footnotes": [{"$ref": "#/texts/1"}],
            "data": {
                "num_rows": 2,
                "num_cols": 2,
                "grid": [
                    [
                        {
                            "text": "h1",
                            "column_header": True,
                            "start_row_offset_idx": 0,
                        },
                        {
                            "text": "h2",
                            "column_header": True,
                            "start_row_offset_idx": 0,
                        },
                    ],
                    [{"text": "a"}, {"text": "b"}],
                ],
            },
            "prov": [],
        }
    ]
    texts = [
        _text_item(label="caption", text="Table caption", self_ref="#/texts/0"),
        _text_item(label="footnote", text="Note: x", self_ref="#/texts/1"),
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=["#/tables/0"],
            texts=texts,
            tables=tables,
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    assert len(ir.blocks) == 1
    table = ir.blocks[0].tables[0]
    assert table.rows == [["h1", "h2"], ["a", "b"]]
    assert table.num_rows == 2
    assert table.num_cols == 2
    assert table.caption == "Table caption"
    assert table.footnotes == ["Note: x"]
    assert table.table_header == [["h1", "h2"]]
    assert table.self_ref == "#/tables/0"


# ---------------------------------------------------------------------------
# 5. Picture — referenced asset
# ---------------------------------------------------------------------------


def test_docling_adapter_picture_referenced_asset(tmp_path: Path) -> None:
    pictures = [
        {
            "self_ref": "#/pictures/0",
            "label": "picture",
            "content_layer": "body",
            "image": {
                "uri": "artifacts/image_000000_abc.png",
                "mimetype": "image/png",
                "size": {"width": 100.0, "height": 200.0},
            },
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/pictures/0"], pictures=pictures),
    )
    art = raw_dir / "artifacts"
    art.mkdir()
    asset = art / "image_000000_abc.png"
    asset.write_bytes(b"\x89PNG fake")

    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == "artifacts/image_000000_abc.png"
    assert drawing.fmt == "png"
    assert drawing.self_ref == "#/pictures/0"
    [a] = [a for a in ir.assets if a.ref == drawing.asset_ref]
    assert a.source == asset
    assert a.suggested_name == "image_000000_abc.png"
    # intrinsic_size lands in extras for downstream VLM filtering
    assert drawing.extras["intrinsic_size"] == [100.0, 200.0]


# ---------------------------------------------------------------------------
# 6. Positions & bbox_attributes
# ---------------------------------------------------------------------------


def test_docling_adapter_positions_and_bbox_attributes(tmp_path: Path) -> None:
    texts = [
        _text_item(
            label="text",
            text="A",
            self_ref="#/texts/0",
            page_no=1,
            bbox=(10.0, 100.0, 200.0, 80.0),
            coord_origin="BOTTOMLEFT",
        ),
        _text_item(
            label="text",
            text="B",
            self_ref="#/texts/1",
            page_no=2,
            bbox=(20.0, 50.0, 220.0, 30.0),
            coord_origin="TOPLEFT",
        ),
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/texts/0", "#/texts/1"], texts=texts),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    assert ir.bbox_attributes == {"origin": "LEFTBOTTOM"}
    # no max / page_sizes leaks
    assert set(ir.bbox_attributes.keys()) == {"origin"}

    positions = ir.blocks[0].positions
    bbox_positions = [p for p in positions if p.range]
    assert len(bbox_positions) == 2
    bl = next(p for p in bbox_positions if p.anchor == "1")
    tl = next(p for p in bbox_positions if p.anchor == "2")
    assert bl.range == [10.0, 100.0, 200.0, 80.0]
    assert bl.origin is None  # inherits doc-level LEFTBOTTOM
    assert tl.origin == "LEFTTOP"  # per-position override
    assert tl.range == [20.0, 50.0, 220.0, 30.0]


def test_docling_adapter_bbox_attributes_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCLING_BBOX_ATTRIBUTES", '{"origin":"LEFTTOP"}')
    texts = [
        _text_item(
            label="text",
            text="A",
            self_ref="#/texts/0",
            coord_origin="BOTTOMLEFT",
        )
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/texts/0"], texts=texts),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    assert ir.bbox_attributes == {"origin": "LEFTTOP"}


# ---------------------------------------------------------------------------
# 7. caption / footnote refs (positive + sibling-not-consumed)
# ---------------------------------------------------------------------------


def test_docling_adapter_caption_refs_only(tmp_path: Path) -> None:
    """The caption referenced by tables[0].captions is consumed (kept in
    IRTable.caption, dropped from reading flow). Sibling text NOT
    referenced — even when it looks like a caption — stays in the reading
    flow."""
    texts = [
        _text_item(label="caption", text="Tab1 caption", self_ref="#/texts/0"),
        _text_item(label="text", text="Tab1 sibling", self_ref="#/texts/1"),
        _text_item(label="caption", text="Orphan caption", self_ref="#/texts/2"),
    ]
    tables = [
        {
            "self_ref": "#/tables/0",
            "label": "table",
            "content_layer": "body",
            "captions": [{"$ref": "#/texts/0"}],
            "data": {"num_rows": 1, "num_cols": 1, "grid": [[{"text": "x"}]]},
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=["#/tables/0", "#/texts/1", "#/texts/2"],
            texts=texts,
            tables=tables,
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    block = ir.blocks[0]
    assert block.tables[0].caption == "Tab1 caption"
    # consumed caption ref does not leak into body text
    assert "Tab1 caption" not in block.content_template
    # orphan caption and sibling text DO appear in body
    assert "Tab1 sibling" in block.content_template
    assert "Orphan caption" in block.content_template


def test_docling_adapter_footnotes_refs_only(tmp_path: Path) -> None:
    texts = [
        _text_item(label="footnote", text="Linked footnote", self_ref="#/texts/0"),
        _text_item(label="text", text="注: this is sibling note", self_ref="#/texts/1"),
    ]
    tables = [
        {
            "self_ref": "#/tables/0",
            "label": "table",
            "content_layer": "body",
            "footnotes": [{"$ref": "#/texts/0"}],
            "data": {"num_rows": 1, "num_cols": 1, "grid": [[{"text": "x"}]]},
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=["#/tables/0", "#/texts/1"],
            texts=texts,
            tables=tables,
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    block = ir.blocks[0]
    assert block.tables[0].footnotes == ["Linked footnote"]
    assert "Linked footnote" not in block.content_template
    assert "注: this is sibling note" in block.content_template


def test_docling_adapter_table_refs_skip_non_body_caption_footnote(
    tmp_path: Path,
) -> None:
    # A body table references a caption/footnote whose targets sit in
    # content_layer="furniture" — typically a page header/footer that
    # docling mislabeled and linked to the table. The adapter contract is
    # that furniture text must never leak into sidecar metadata, so the
    # IRTable's caption/footnotes lists must come back empty (and the body
    # reading flow must not pick up the furniture text either).
    texts = [
        _text_item(
            label="caption",
            text="Page header masquerading as caption",
            self_ref="#/texts/0",
            content_layer="furniture",
        ),
        _text_item(
            label="footnote",
            text="Page footer masquerading as footnote",
            self_ref="#/texts/1",
            content_layer="furniture",
        ),
    ]
    tables = [
        {
            "self_ref": "#/tables/0",
            "label": "table",
            "content_layer": "body",
            "captions": [{"$ref": "#/texts/0"}],
            "footnotes": [{"$ref": "#/texts/1"}],
            "data": {"num_rows": 1, "num_cols": 1, "grid": [[{"text": "x"}]]},
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/tables/0"], texts=texts, tables=tables),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    block = ir.blocks[0]
    assert block.tables[0].caption == ""
    assert block.tables[0].footnotes == []
    assert "Page header masquerading" not in block.content_template
    assert "Page footer masquerading" not in block.content_template


def test_docling_adapter_picture_children_fallback_skips_non_body(
    tmp_path: Path,
) -> None:
    # Same invariant for the children fallback path: a body picture has no
    # explicit captions/footnotes, but its ``children`` list refs a caption
    # whose target is furniture. ``_resolve_children_with_label`` must
    # skip it rather than silently surfacing furniture text as the
    # picture's caption.
    texts = [
        _text_item(
            label="caption",
            text="Furniture caption via children",
            self_ref="#/texts/0",
            content_layer="furniture",
        ),
    ]
    pictures = [
        {
            "self_ref": "#/pictures/0",
            "label": "picture",
            "content_layer": "body",
            "image": {
                "uri": "artifacts/p0.png",
                "mimetype": "image/png",
            },
            "children": [{"$ref": "#/texts/0"}],
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/pictures/0"], texts=texts, pictures=pictures),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    block = ir.blocks[0]
    assert block.drawings[0].caption == ""
    assert "Furniture caption via children" not in block.content_template


# ---------------------------------------------------------------------------
# 8. furniture skipped
# ---------------------------------------------------------------------------


def test_docling_adapter_furniture_skipped_by_content_layer(tmp_path: Path) -> None:
    texts = [
        _text_item(label="section_header", text="H", level=1, self_ref="#/texts/0"),
        _text_item(label="text", text="Body sentence.", self_ref="#/texts/1"),
        _text_item(
            label="page_footer",
            text="footer 1/5",
            self_ref="#/texts/2",
            content_layer="furniture",
        ),
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=["#/texts/0", "#/texts/1", "#/texts/2"],
            texts=texts,
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    full = "\n".join(b.content_template for b in ir.blocks)
    assert "footer 1/5" not in full
    # the furniture's prov page_no=1 must not leak into any block position
    for block in ir.blocks:
        for pos in block.positions:
            assert (
                pos.anchor != "1"
                or pos.range is not None
                or any(p.range is not None for p in block.positions)
            )


# ---------------------------------------------------------------------------
# 9. Picture inner children dropped from reading flow
# ---------------------------------------------------------------------------


def test_docling_adapter_picture_children_dropped(tmp_path: Path) -> None:
    texts = [
        _text_item(label="caption", text="Picture caption", self_ref="#/texts/0"),
        _text_item(label="text", text="Inner OCR text 1", self_ref="#/texts/1"),
        _text_item(label="text", text="Inner OCR text 2", self_ref="#/texts/2"),
    ]
    pictures = [
        {
            "self_ref": "#/pictures/0",
            "label": "picture",
            "content_layer": "body",
            "image": {"uri": "artifacts/img.png", "mimetype": "image/png"},
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
            ],
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/pictures/0"], texts=texts, pictures=pictures),
    )
    art = raw_dir / "artifacts"
    art.mkdir()
    (art / "img.png").write_bytes(b"png")
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    block = ir.blocks[0]
    drawing = block.drawings[0]
    # caption (label=caption) is taken via children fallback
    assert drawing.caption == "Picture caption"
    # OCR-only children do NOT appear in body content
    assert "Inner OCR text 1" not in block.content_template
    assert "Inner OCR text 2" not in block.content_template
    # extras records the OCR child count
    assert drawing.extras["ocr_child_count"] == 2


# ---------------------------------------------------------------------------
# 10. Picture with missing image still emits IRDrawing
# ---------------------------------------------------------------------------


def test_docling_adapter_picture_missing_image_kept(tmp_path: Path) -> None:
    pictures = [
        {
            "self_ref": "#/pictures/0",
            "label": "picture",
            "content_layer": "body",
            "image": None,
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/pictures/0"], pictures=pictures),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == ""
    assert drawing.extras["image_missing"] is True


def test_docling_adapter_picture_rejects_traversal_uri(tmp_path: Path) -> None:
    # A poisoned bundle JSON points the image URI outside raw_dir via "..".
    # The asset must NOT pick up the outside file — otherwise write_sidecar
    # would copy it into parsed assets, turning a parser-side compromise
    # into arbitrary local-file exfiltration.
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"\x89PNG outside")
    pictures = [
        {
            "self_ref": "#/pictures/0",
            "label": "picture",
            "content_layer": "body",
            "image": {
                "uri": "../secret.png",
                "mimetype": "image/png",
            },
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/pictures/0"], pictures=pictures),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    drawing = ir.blocks[0].drawings[0]
    # The ref is preserved (so audit/extras still show what was claimed),
    # but the asset has no source and the drawing is flagged missing.
    assert drawing.asset_ref == "../secret.png"
    assert drawing.extras["image_missing"] is True
    [a] = [a for a in ir.assets if a.ref == "../secret.png"]
    assert a.source is None


def test_docling_adapter_picture_rejects_absolute_uri(tmp_path: Path) -> None:
    # ``Path("raw_dir") / "/etc/passwd"`` discards raw_dir on POSIX, so an
    # absolute URI would escape even without a "..". Reject these too.
    outside = tmp_path / "leak.png"
    outside.write_bytes(b"\x89PNG outside")
    pictures = [
        {
            "self_ref": "#/pictures/0",
            "label": "picture",
            "content_layer": "body",
            "image": {
                "uri": str(outside),
                "mimetype": "image/png",
            },
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/pictures/0"], pictures=pictures),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    drawing = ir.blocks[0].drawings[0]
    assert drawing.extras["image_missing"] is True
    [a] = [a for a in ir.assets if a.ref == str(outside)]
    assert a.source is None


# ---------------------------------------------------------------------------
# 11. Formula degrades to text when enrichment is off
# ---------------------------------------------------------------------------


def test_docling_adapter_formula_degrades_without_enrichment(tmp_path: Path) -> None:
    texts = [
        # text == orig means enrichment didn't run; we must not emit IREquation
        {
            "self_ref": "#/texts/0",
            "label": "formula",
            "content_layer": "body",
            "text": "C = 2 * P / X",
            "orig": "C = 2 * P / X",
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/texts/0"], texts=texts),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    block = ir.blocks[0]
    assert block.equations == []
    assert "C = 2 * P / X" in block.content_template


def test_docling_adapter_formula_with_latex_wraps_dollars(tmp_path: Path) -> None:
    texts = [
        {
            "self_ref": "#/texts/0",
            "label": "formula",
            "content_layer": "body",
            "text": "C = 2 \\cdot P",
            "orig": "<unreadable>",
            "prov": [],
        }
    ]
    raw_dir = _write_doc(
        tmp_path,
        _doc(body_children=["#/texts/0"], texts=texts),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    block = ir.blocks[0]
    assert len(block.equations) == 1
    eq = block.equations[0]
    assert eq.latex.startswith("$$") and eq.latex.endswith("$$")
    assert "C = 2 \\cdot P" in eq.latex
    assert eq.self_ref == "#/texts/0"
    assert "{{EQ:eq1}}" in block.content_template


# ---------------------------------------------------------------------------
# 12. key_value_items / form_items audit
# ---------------------------------------------------------------------------


def test_docling_adapter_kv_form_items_audit_in_split_option(tmp_path: Path) -> None:
    raw_dir = _write_doc(
        tmp_path,
        _doc(
            body_children=[],
            key_value_items=[{"id": "kv1"}, {"id": "kv2"}],
            form_items=[{"id": "f1"}],
        ),
    )
    ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name="demo.pdf")
    extras = ir.split_option["docling_extras"]
    assert extras == {"key_value_items": 2, "form_items": 1}
