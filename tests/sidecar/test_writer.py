"""Spec-compliance tests for :func:`lightrag.sidecar.write_sidecar`.

These assertions are deliberately structural: they encode the contract in
``docs/LightRAGSidecarFormat-zh.md`` so accidental regressions in
``writer.py`` show up before downstream chunker / multimodal consumers see
malformed sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.sidecar import (
    AssetSpec,
    IRBlock,
    IRDoc,
    IRDrawing,
    IREquation,
    IRPosition,
    IRTable,
    write_sidecar,
)


def _load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    meta: dict = {}
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            obj = json.loads(line)
            if i == 0:
                meta = obj
            else:
                rows.append(obj)
    return meta, rows


@pytest.mark.offline
def test_writer_empty_doc_emits_only_blocks_jsonl(tmp_path: Path) -> None:
    """Document with no blocks: only the meta line, no per-modality JSONs,
    no assets dir."""
    parsed = tmp_path / "empty.parsed"
    ir = IRDoc(
        document_name="empty.docx",
        document_format="docx",
        doc_title="empty",
        split_option={},
        blocks=[],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-0001", engine="native")

    files = {p.name for p in parsed.iterdir()}
    assert files == {"empty.blocks.jsonl"}

    meta, rows = _load_jsonl(parsed / "empty.blocks.jsonl")
    assert meta["type"] == "meta"
    assert meta["blocks"] == 0
    assert meta["asset_dir"] is False
    assert meta["table_file"] is False
    assert meta["drawing_file"] is False
    assert meta["equation_file"] is False
    assert rows == []


@pytest.mark.offline
def test_writer_renders_table_with_inline_body(tmp_path: Path) -> None:
    """Spec §3.3 / fix 1: <table id="..." format="json">rows</table>; NOT
    <cite type="table">. Also verifies the table's JSON content appears in
    blocks.jsonl content so doc_hash and F/R/V chunkers see it."""
    parsed = tmp_path / "t.parsed"
    ir = IRDoc(
        document_name="t.pdf",
        document_format="pdf",
        doc_title="t",
        split_option={},
        blocks=[
            IRBlock(
                content_template="prefix {{TBL:t1}} suffix",
                tables=[
                    IRTable(
                        placeholder_key="t1",
                        rows=[["a", "b"], ["1", "2"]],
                        num_rows=2,
                        num_cols=2,
                        caption="cap",
                    )
                ],
            )
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-cafebabe", engine="mineru")

    _, rows = _load_jsonl(parsed / "t.blocks.jsonl")
    assert len(rows) == 1
    body = rows[0]["content"]
    assert '<table id="tb-cafebabe-0001" format="json">' in body
    assert '[["a", "b"], ["1", "2"]]' in body
    assert "</table>" in body
    # Negative: no <cite type="table"> placeholder anywhere.
    assert "<cite" not in body


@pytest.mark.offline
def test_writer_drawing_path_points_into_assets_dir(tmp_path: Path) -> None:
    """Spec §四 / fix 5: drawing path always points inside *.blocks.assets/.

    Asset must be materialized on disk; meta.asset_dir must reflect it.
    """
    parsed = tmp_path / "d.parsed"
    ir = IRDoc(
        document_name="d.pdf",
        document_format="pdf",
        doc_title="d",
        split_option={},
        blocks=[
            IRBlock(
                content_template="see {{IMG:i1}}",
                drawings=[
                    IRDrawing(
                        placeholder_key="i1",
                        asset_ref="img1",
                        fmt="png",
                        caption="figure 1",
                    )
                ],
            )
        ],
        assets=[AssetSpec(ref="img1", suggested_name="x.png", source=b"\x89PNG")],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-cafebabe", engine="mineru")

    meta, rows = _load_jsonl(parsed / "d.blocks.jsonl")
    assert meta["asset_dir"] is True
    assert meta["drawing_file"] is True

    body = rows[0]["content"]
    assert 'path="d.blocks.assets/x.png"' in body
    assert (parsed / "d.blocks.assets" / "x.png").read_bytes() == b"\x89PNG"

    drawings = json.loads((parsed / "d.drawings.json").read_text())["drawings"]
    item = drawings["im-cafebabe-0001"]
    assert item["path"] == "d.blocks.assets/x.png"
    assert item["caption"] == "figure 1"
    assert item["format"] == "png"


@pytest.mark.offline
def test_writer_equation_strips_dollar_wrappers_for_equations_json(
    tmp_path: Path,
) -> None:
    """When IREquation.latex carries MinerU's raw ``$$...$$``/``$..$``
    wrappers (preserved so blocks.jsonl shows the source verbatim), the
    writer must strip them when persisting equations.json content — that
    file holds clean latex by contract."""
    parsed = tmp_path / "d.parsed"
    ir = IRDoc(
        document_name="d.pdf",
        document_format="pdf",
        doc_title="d",
        split_option={},
        blocks=[
            IRBlock(
                content_template="see {{EQ:b1}}",
                equations=[
                    IREquation(
                        placeholder_key="b1",
                        latex="$$\nE = mc^2\n$$",
                        is_block=True,
                    ),
                ],
            )
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-deadbeef", engine="mineru")

    # blocks.jsonl: <equation> body preserves the parser's raw form.
    body = _load_jsonl(parsed / "d.blocks.jsonl")[1][0]["content"]
    assert (
        '<equation id="eq-deadbeef-0001" format="latex">$$\nE = mc^2\n$$</equation>'
        in body
    )

    # equations.json: dollar wrappers removed.
    equations = json.loads((parsed / "d.equations.json").read_text())["equations"]
    assert equations["eq-deadbeef-0001"]["content"] == "E = mc^2"


@pytest.mark.offline
def test_writer_equation_caption_preserved_block_and_inline(
    tmp_path: Path,
) -> None:
    """Fix 3 + design decision: <equation caption="..."> on both block and
    inline forms; inline does NOT receive an id and does NOT enter
    equations.json (spec §6 / §3.3)."""
    parsed = tmp_path / "e.parsed"
    ir = IRDoc(
        document_name="e.pdf",
        document_format="pdf",
        doc_title="e",
        split_option={},
        blocks=[
            IRBlock(
                content_template="block {{EQ:b1}} inline {{EQI:i1}}",
                equations=[
                    IREquation(
                        placeholder_key="b1",
                        latex="x^2",
                        is_block=True,
                        caption="Eq 1",
                    ),
                    IREquation(
                        placeholder_key="i1",
                        latex="y_n",
                        is_block=False,
                        caption="Inline",
                    ),
                ],
            )
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-cafebabe", engine="mineru")
    body = _load_jsonl(parsed / "e.blocks.jsonl")[1][0]["content"]

    assert (
        '<equation id="eq-cafebabe-0001" format="latex" caption="Eq 1">x^2</equation>'
        in body
    )
    # Inline: no id; caption preserved.
    assert '<equation format="latex" caption="Inline">y_n</equation>' in body

    equations = json.loads((parsed / "e.equations.json").read_text())["equations"]
    # Inline equation should NOT have produced a sidecar entry.
    assert list(equations.keys()) == ["eq-cafebabe-0001"]
    assert equations["eq-cafebabe-0001"]["caption"] == "Eq 1"


@pytest.mark.offline
def test_writer_propagates_parent_headings_to_sidecar_items(
    tmp_path: Path,
) -> None:
    """Spec §4/§5/§6: tables/drawings/equations items carry the owning
    block's ``parent_headings`` (the top-down ancestor chain), mirroring the
    block's ``parent_headings`` in blocks.jsonl so multimodal analysis sees
    the full section context, not just the nearest ``heading``."""
    parsed = tmp_path / "ph.parsed"
    parents = ["2 Product Description", "2.4 Environmental Adaptability"]
    ir = IRDoc(
        document_name="ph.pdf",
        document_format="pdf",
        doc_title="ph",
        split_option={},
        blocks=[
            IRBlock(
                content_template="see {{TBL:t1}} {{IMG:i1}} {{EQ:b1}}",
                heading="2.4.4 Combined",
                parent_headings=parents,
                tables=[
                    IRTable(
                        placeholder_key="t1",
                        rows=[["a", "b"]],
                        num_rows=1,
                        num_cols=2,
                    )
                ],
                drawings=[IRDrawing(placeholder_key="i1", asset_ref="img1", fmt="png")],
                equations=[
                    IREquation(placeholder_key="b1", latex="x^2", is_block=True)
                ],
            )
        ],
        assets=[AssetSpec(ref="img1", suggested_name="x.png", source=b"\x89PNG")],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-cafebabe", engine="mineru")

    # blocks.jsonl already carries parent_headings; sidecar items must match.
    _, rows = _load_jsonl(parsed / "ph.blocks.jsonl")
    assert rows[0]["parent_headings"] == parents

    tables = json.loads((parsed / "ph.tables.json").read_text())["tables"]
    drawings = json.loads((parsed / "ph.drawings.json").read_text())["drawings"]
    equations = json.loads((parsed / "ph.equations.json").read_text())["equations"]
    assert tables["tb-cafebabe-0001"]["parent_headings"] == parents
    assert drawings["im-cafebabe-0001"]["parent_headings"] == parents
    assert equations["eq-cafebabe-0001"]["parent_headings"] == parents


@pytest.mark.offline
def test_writer_positions_round_trip_bbox(tmp_path: Path) -> None:
    """Fix 4: positions go through unchanged. bbox type is the mineru path."""
    parsed = tmp_path / "p.parsed"
    ir = IRDoc(
        document_name="p.pdf",
        document_format="pdf",
        doc_title="p",
        split_option={},
        blocks=[
            IRBlock(
                content_template="text",
                positions=[
                    IRPosition(type="bbox", anchor=2, range=[10.0, 20.0, 100.0, 200.0])
                ],
            )
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-aaaa", engine="mineru")
    rows = _load_jsonl(parsed / "p.blocks.jsonl")[1]
    assert rows[0]["positions"] == [
        {"type": "bbox", "anchor": 2, "range": [10.0, 20.0, 100.0, 200.0]}
    ]


@pytest.mark.offline
def test_position_origin_to_jsonable_omits_when_none() -> None:
    """Spec §八 per-position origin: ``None`` ⇒ field absent (inherit from
    meta ``bbox_attributes.origin``)."""
    pos = IRPosition(type="bbox", anchor=1, range=[1.0, 2.0, 3.0, 4.0])
    assert "origin" not in pos.to_jsonable()


@pytest.mark.offline
def test_position_origin_to_jsonable_emits_when_set() -> None:
    """Spec §八 per-position origin: explicit value ⇒ override field in JSON."""
    pos = IRPosition(
        type="bbox", anchor=1, range=[1.0, 2.0, 3.0, 4.0], origin="LEFTTOP"
    )
    out = pos.to_jsonable()
    assert out["origin"] == "LEFTTOP"


@pytest.mark.offline
def test_writer_position_origin_mixed_per_block(tmp_path: Path) -> None:
    """Docling mixed coord_origin scenario: doc-level origin in meta,
    per-position override on the minority. Coordinates land verbatim."""
    parsed = tmp_path / "mixed.parsed"
    ir = IRDoc(
        document_name="mixed.pdf",
        document_format="pdf",
        doc_title="mixed",
        split_option={},
        blocks=[
            IRBlock(
                content_template="text",
                positions=[
                    IRPosition(type="bbox", anchor=1, range=[10.0, 20.0, 30.0, 40.0]),
                    IRPosition(
                        type="bbox",
                        anchor=1,
                        range=[50.0, 60.0, 70.0, 80.0],
                        origin="LEFTTOP",
                    ),
                ],
            )
        ],
        bbox_attributes={"origin": "LEFTBOTTOM"},
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-bbb1", engine="docling")
    meta, rows = _load_jsonl(parsed / "mixed.blocks.jsonl")
    assert meta["bbox_attributes"] == {"origin": "LEFTBOTTOM"}
    positions = rows[0]["positions"]
    assert positions[0] == {
        "type": "bbox",
        "anchor": 1,
        "range": [10.0, 20.0, 30.0, 40.0],
    }
    assert positions[1] == {
        "type": "bbox",
        "anchor": 1,
        "range": [50.0, 60.0, 70.0, 80.0],
        "origin": "LEFTTOP",
    }


@pytest.mark.offline
def test_writer_drawing_self_ref_emitted_only_when_nonempty(tmp_path: Path) -> None:
    """Spec §四 ``self_ref``: empty string ⇒ field absent; non-empty ⇒
    written verbatim. Keeps MinerU/native sidecars byte-compatible."""
    parsed = tmp_path / "sref.parsed"
    ir = IRDoc(
        document_name="sref.pdf",
        document_format="pdf",
        doc_title="sref",
        split_option={},
        blocks=[
            IRBlock(
                content_template="{{IMG:a}} {{IMG:b}}",
                drawings=[
                    IRDrawing(placeholder_key="a", asset_ref="img_a", fmt="png"),
                    IRDrawing(
                        placeholder_key="b",
                        asset_ref="img_b",
                        fmt="png",
                        self_ref="#/pictures/3",
                    ),
                ],
            )
        ],
        assets=[
            AssetSpec(ref="img_a", suggested_name="a.png", source=b"\x89PNG"),
            AssetSpec(ref="img_b", suggested_name="b.png", source=b"\x89PNG"),
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-ccc1", engine="docling")
    drawings = json.loads((parsed / "sref.drawings.json").read_text("utf-8"))[
        "drawings"
    ]
    items = list(drawings.values())
    assert "self_ref" not in items[0]
    assert items[1]["self_ref"] == "#/pictures/3"


@pytest.mark.offline
def test_writer_table_self_ref_emitted_only_when_nonempty(tmp_path: Path) -> None:
    """Spec §五 ``self_ref``: same omit-when-empty semantics as drawings."""
    parsed = tmp_path / "tsref.parsed"
    ir = IRDoc(
        document_name="tsref.pdf",
        document_format="pdf",
        doc_title="tsref",
        split_option={},
        blocks=[
            IRBlock(
                content_template="{{TBL:a}} {{TBL:b}}",
                tables=[
                    IRTable(placeholder_key="a", rows=[["x"]], num_rows=1, num_cols=1),
                    IRTable(
                        placeholder_key="b",
                        rows=[["y"]],
                        num_rows=1,
                        num_cols=1,
                        self_ref="#/tables/0",
                    ),
                ],
            )
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-ddd1", engine="docling")
    tables = json.loads((parsed / "tsref.tables.json").read_text("utf-8"))["tables"]
    items = list(tables.values())
    assert "self_ref" not in items[0]
    assert items[1]["self_ref"] == "#/tables/0"


@pytest.mark.offline
def test_writer_table_header_serialized_by_format(tmp_path: Path) -> None:
    """``table_header`` is stored in the table's own format: a JSON 2-D array
    for JSON tables, a raw ``<thead>`` (verbatim) for HTML tables — and a grid
    supplied for an HTML table is rendered to a span-less ``<thead>``."""
    parsed = tmp_path / "th.parsed"
    html_thead = '<thead><tr><th colspan="2">Group</th></tr></thead>'
    ir = IRDoc(
        document_name="th.pdf",
        document_format="pdf",
        doc_title="th",
        split_option={},
        blocks=[
            IRBlock(
                content_template="{{TBL:j}} {{TBL:h}} {{TBL:g}}",
                tables=[
                    # JSON table: grid header → JSON 2-D array string.
                    IRTable(
                        placeholder_key="j",
                        rows=[["a", "b"]],
                        num_rows=1,
                        num_cols=2,
                        table_header=[["H1", "H2"]],
                    ),
                    # HTML table: raw <thead> string → stored verbatim.
                    IRTable(
                        placeholder_key="h",
                        html="<table><tbody><tr><td>a</td><td>b</td></tr></tbody></table>",
                        num_rows=1,
                        num_cols=2,
                        table_header=html_thead,
                    ),
                    # HTML table: grid header → rendered to a span-less <thead>.
                    IRTable(
                        placeholder_key="g",
                        html="<table><tbody><tr><td>x</td><td>y</td></tr></tbody></table>",
                        num_rows=1,
                        num_cols=2,
                        table_header=[["P", "Q"]],
                    ),
                ],
            )
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-thh1", engine="mineru")
    tables = json.loads((parsed / "th.tables.json").read_text("utf-8"))["tables"]
    items = list(tables.values())

    json_item = items[0]
    assert json_item["format"] == "json"
    assert json.loads(json_item["table_header"]) == [["H1", "H2"]]

    html_item = items[1]
    assert html_item["format"] == "html"
    assert html_item["table_header"] == html_thead  # verbatim, spans preserved

    grid_html_item = items[2]
    assert grid_html_item["format"] == "html"
    assert (
        grid_html_item["table_header"] == "<thead><tr><th>P</th><th>Q</th></tr></thead>"
    )


@pytest.mark.offline
def test_writer_equation_self_ref_emitted_only_when_nonempty(tmp_path: Path) -> None:
    """Spec §六 ``self_ref``: block equations carry it; inline equations
    never reach equations.json so the field is moot there."""
    parsed = tmp_path / "esref.parsed"
    ir = IRDoc(
        document_name="esref.pdf",
        document_format="pdf",
        doc_title="esref",
        split_option={},
        blocks=[
            IRBlock(
                content_template="{{EQ:a}} {{EQ:b}}",
                equations=[
                    IREquation(placeholder_key="a", latex="a+b", is_block=True),
                    IREquation(
                        placeholder_key="b",
                        latex="c+d",
                        is_block=True,
                        self_ref="#/texts/15",
                    ),
                ],
            )
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-eee1", engine="docling")
    equations = json.loads((parsed / "esref.equations.json").read_text("utf-8"))[
        "equations"
    ]
    items = list(equations.values())
    assert "self_ref" not in items[0]
    assert items[1]["self_ref"] == "#/texts/15"


@pytest.mark.offline
def test_writer_id_sequence_is_global_per_kind(tmp_path: Path) -> None:
    """IDs increment across blocks within their own kind: tables ↑,
    drawings ↑, equations ↑ — three independent sequences."""
    parsed = tmp_path / "s.parsed"
    blocks = [
        IRBlock(
            content_template="a {{TBL:t}} b {{IMG:i}} c",
            tables=[IRTable(placeholder_key="t", rows=[["x"]], num_rows=1, num_cols=1)],
            drawings=[IRDrawing(placeholder_key="i", asset_ref="a1", fmt="png")],
        ),
        IRBlock(
            content_template="d {{EQ:e}} {{TBL:t}}",
            tables=[IRTable(placeholder_key="t", rows=[["y"]], num_rows=1, num_cols=1)],
            equations=[IREquation(placeholder_key="e", latex="z", is_block=True)],
        ),
    ]
    ir = IRDoc(
        document_name="s.pdf",
        document_format="pdf",
        doc_title="s",
        split_option={},
        blocks=blocks,
        assets=[AssetSpec(ref="a1", suggested_name="img.png", source=b"x")],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-bbbb", engine="mineru")
    tables = json.loads((parsed / "s.tables.json").read_text())["tables"]
    assert sorted(tables.keys()) == ["tb-bbbb-0001", "tb-bbbb-0002"]
    drawings = json.loads((parsed / "s.drawings.json").read_text())["drawings"]
    assert list(drawings.keys()) == ["im-bbbb-0001"]
    equations = json.loads((parsed / "s.equations.json").read_text())["equations"]
    assert list(equations.keys()) == ["eq-bbbb-0001"]


@pytest.mark.offline
def test_writer_empty_block_dropped(tmp_path: Path) -> None:
    """An IRBlock that strips to empty after placeholder expansion produces
    no blocks.jsonl row AND no sidecar items (its in-flight placeholders
    are stillborn)."""
    parsed = tmp_path / "empty_block.parsed"
    ir = IRDoc(
        document_name="x.pdf",
        document_format="pdf",
        doc_title="x",
        split_option={},
        blocks=[
            IRBlock(
                content_template="   \n  ",
                tables=[
                    IRTable(
                        placeholder_key="orphan",
                        rows=[["a"]],
                        num_rows=1,
                        num_cols=1,
                    )
                ],
            ),
            IRBlock(content_template="real content"),
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-eee", engine="mineru")
    meta, rows = _load_jsonl(parsed / "x.blocks.jsonl")
    assert meta["blocks"] == 1
    assert len(rows) == 1
    assert rows[0]["content"] == "real content"
    # No tables.json because the orphan placeholder is dropped.
    assert not (parsed / "x.tables.json").exists()


@pytest.mark.offline
def test_writer_asset_name_collision_suffixed(tmp_path: Path) -> None:
    """Two assets with identical suggested_name → second gets ``-2`` stem
    suffix; drawings.json paths reflect the actual on-disk names."""
    parsed = tmp_path / "c.parsed"
    ir = IRDoc(
        document_name="c.pdf",
        document_format="pdf",
        doc_title="c",
        split_option={},
        blocks=[
            IRBlock(
                content_template="{{IMG:a}} and {{IMG:b}}",
                drawings=[
                    IRDrawing(placeholder_key="a", asset_ref="r1", fmt="png"),
                    IRDrawing(placeholder_key="b", asset_ref="r2", fmt="png"),
                ],
            )
        ],
        assets=[
            AssetSpec(ref="r1", suggested_name="img.png", source=b"a"),
            AssetSpec(ref="r2", suggested_name="img.png", source=b"b"),
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-1111", engine="mineru")
    assets = sorted(p.name for p in (parsed / "c.blocks.assets").iterdir())
    assert assets == ["img-2.png", "img.png"]
    body = _load_jsonl(parsed / "c.blocks.jsonl")[1][0]["content"]
    assert 'path="c.blocks.assets/img.png"' in body
    assert 'path="c.blocks.assets/img-2.png"' in body


@pytest.mark.offline
def test_writer_meta_has_required_spec_fields(tmp_path: Path) -> None:
    """Spec §3.1: meta line contains every required field at fixed names."""
    parsed = tmp_path / "m.parsed"
    ir = IRDoc(
        document_name="m.pdf",
        document_format="pdf",
        doc_title="title",
        split_option={"engine_version": "magic-pdf 1.5.4"},
        blocks=[IRBlock(content_template="hello")],
        bbox_attributes={"origin": "LEFTTOP", "max": 1000},
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-deadbeef", engine="mineru")
    meta, _ = _load_jsonl(parsed / "m.blocks.jsonl")
    for k in (
        "type",
        "format",
        "version",
        "document_name",
        "document_format",
        "document_hash",
        "table_file",
        "equation_file",
        "drawing_file",
        "asset_dir",
        "split_option",
        "blocks",
        "doc_id",
        "parse_engine",
        "parse_time",
        "doc_title",
    ):
        assert k in meta, f"meta missing field: {k}"
    assert meta["document_hash"].startswith("sha256:")
    assert meta["parse_engine"] == "mineru"
    assert meta["bbox_attributes"] == {"origin": "LEFTTOP", "max": 1000}
    assert meta["split_option"] == {"engine_version": "magic-pdf 1.5.4"}


@pytest.mark.offline
def test_writer_sidecar_files_only_when_nonempty(tmp_path: Path) -> None:
    """tables.json / drawings.json / equations.json are NOT written when
    the corresponding maps are empty (spec §一 table)."""
    parsed = tmp_path / "n.parsed"
    ir = IRDoc(
        document_name="n.docx",
        document_format="docx",
        doc_title="n",
        split_option={},
        blocks=[
            IRBlock(
                content_template="{{IMG:i}}",
                drawings=[IRDrawing(placeholder_key="i", asset_ref="r", fmt="png")],
            )
        ],
        assets=[AssetSpec(ref="r", suggested_name="i.png", source=b"x")],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-aaaa", engine="native")
    files = {p.name for p in parsed.iterdir() if p.is_file()}
    assert "n.drawings.json" in files
    assert "n.tables.json" not in files
    assert "n.equations.json" not in files


@pytest.mark.offline
def test_writer_blockid_formula_stable(tmp_path: Path) -> None:
    """blockid = md5(doc_id:block_index:heading:content). Same content +
    metadata → same blockid."""
    parsed_a = tmp_path / "a.parsed"
    parsed_b = tmp_path / "b.parsed"
    ir = IRDoc(
        document_name="x.pdf",
        document_format="pdf",
        doc_title="x",
        split_option={},
        blocks=[IRBlock(content_template="abc", heading="H", level=1)],
    )
    write_sidecar(ir, parsed_dir=parsed_a, doc_id="doc-fixed", engine="mineru")
    write_sidecar(ir, parsed_dir=parsed_b, doc_id="doc-fixed", engine="mineru")
    rows_a = _load_jsonl(parsed_a / "x.blocks.jsonl")[1]
    rows_b = _load_jsonl(parsed_b / "x.blocks.jsonl")[1]
    assert rows_a[0]["blockid"] == rows_b[0]["blockid"]
