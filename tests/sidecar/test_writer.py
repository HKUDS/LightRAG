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
from lightrag.sidecar import writer as _writer
from lightrag.sidecar.writer import (
    _allocate_unique_name,
    _materialize_assets,
    _safe_asset_filename,
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


# --- Path-traversal boundary for sidecar asset materialization (PR #3316) ---


@pytest.mark.offline
@pytest.mark.parametrize(
    "suggested, expected",
    [
        ("evil.png", "evil.png"),
        ("../evil.png", "evil.png"),
        ("../../etc/passwd", "passwd"),
        ("..\\evil.png", "evil.png"),
        ("dir\\sub\\evil.png", "evil.png"),
        ("a/b/c/evil.png", "evil.png"),
        ("/abs/evil.png", "evil.png"),
        ("C:\\Windows\\evil.png", "evil.png"),
        ("  spaced.png  ", "spaced.png"),
    ],
)
def test_safe_asset_filename_collapses_to_basename(
    suggested: str, expected: str
) -> None:
    """Parser-suggested names are reduced to a bare basename regardless of the
    path separators they carry, so they can never steer the output path out of
    the assets dir. ``\\`` is normalised first so Windows-style payloads on a
    POSIX host still collapse rather than surviving as a single filename."""
    assert _safe_asset_filename(suggested) == expected


@pytest.mark.offline
@pytest.mark.parametrize("suggested", ["", "   ", "..", "...", "/", "\\", "\x00\x1f"])
def test_safe_asset_filename_falls_back_when_empty(suggested: str) -> None:
    """Names that strip down to nothing (pure dots/separators/control chars)
    yield the ``asset`` fallback rather than an empty or dot-only filename."""
    assert _safe_asset_filename(suggested) == "asset"


@pytest.mark.offline
def test_safe_asset_filename_drops_control_chars() -> None:
    """C0 control chars and DEL are stripped out of the basename so they can't
    smuggle separators or break the on-disk name."""
    assert _safe_asset_filename("ev\x00i\x7fl\x1f.png") == "evil.png"


@pytest.mark.offline
def test_allocate_unique_name_sanitises_before_suffixing() -> None:
    """``_allocate_unique_name`` sanitises first, then applies collision
    suffixes — a traversal payload and a plain basename that collapse to the
    same name still get distinct, contained filenames."""
    used: set[str] = set()
    first = _allocate_unique_name("../evil.png", used)
    used.add(first)
    second = _allocate_unique_name("dir/evil.png", used)
    assert first == "evil.png"
    assert second == "evil-2.png"


@pytest.mark.offline
def test_materialize_assets_keeps_traversal_payload_inside_assets_dir(
    tmp_path: Path,
) -> None:
    """End-to-end: a parser-controlled ``../`` asset name must not write bytes
    outside the assets dir. The byte payload lands at the collapsed basename
    inside ``assets_dir`` and the sibling traversal target stays absent."""
    assets_dir = tmp_path / "doc.blocks.assets"
    out = _materialize_assets(
        [AssetSpec(ref="r1", suggested_name="../evil.png", source=b"\x89PNG")],
        assets_dir,
    )
    assert out == {"r1": "evil.png"}
    assert (assets_dir / "evil.png").read_bytes() == b"\x89PNG"
    # The traversal target one level up must never be created.
    assert not (tmp_path / "evil.png").exists()


@pytest.mark.offline
def test_writer_traversal_asset_name_stays_in_assets_dir(tmp_path: Path) -> None:
    """Through the public ``write_sidecar`` path: a drawing whose asset uses a
    traversal ``suggested_name`` is materialised inside ``*.blocks.assets/`` and
    the rendered/drawing path is the contained basename, with nothing written
    to the parent ``parsed`` dir."""
    parsed = tmp_path / "trav.parsed"
    ir = IRDoc(
        document_name="trav.pdf",
        document_format="pdf",
        doc_title="trav",
        split_option={},
        blocks=[
            IRBlock(
                content_template="see {{IMG:i1}}",
                drawings=[IRDrawing(placeholder_key="i1", asset_ref="img1", fmt="png")],
            )
        ],
        assets=[
            AssetSpec(ref="img1", suggested_name="../../evil.png", source=b"\x89PNG")
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-trav", engine="mineru")

    assert (parsed / "trav.blocks.assets" / "evil.png").read_bytes() == b"\x89PNG"
    assert not (tmp_path / "evil.png").exists()
    assert not (parsed.parent / "evil.png").exists()

    body = _load_jsonl(parsed / "trav.blocks.jsonl")[1][0]["content"]
    assert 'path="trav.blocks.assets/evil.png"' in body
    drawings = json.loads((parsed / "trav.drawings.json").read_text())["drawings"]
    assert drawings["im-trav-0001"]["path"] == "trav.blocks.assets/evil.png"


@pytest.mark.offline
def test_materialize_assets_containment_check_skips_escaping_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Defense-in-depth: even if name sanitisation were bypassed and a target
    resolved outside the assets dir, the ``relative_to`` containment guard skips
    the write (no bytes escape) and the ref is dropped from the output map."""
    assets_dir = tmp_path / "doc.blocks.assets"

    # Force the allocator to hand back an escaping relative name, simulating a
    # sanitisation gap so the second-layer containment check is exercised.
    monkeypatch.setattr(_writer, "_allocate_unique_name", lambda *_: "../escape.png")

    import logging

    monkeypatch.setattr(_writer.logger, "propagate", True)
    with caplog.at_level(logging.WARNING, logger=_writer.logger.name):
        out = _materialize_assets(
            [AssetSpec(ref="r1", suggested_name="ok.png", source=b"\x89PNG")],
            assets_dir,
        )

    assert out == {}
    assert not (tmp_path / "escape.png").exists()
    assert "unsafe asset target" in caplog.text


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


@pytest.mark.offline
def test_writer_strips_control_separators_from_block_content(
    tmp_path: Path,
) -> None:
    """C0 control/separator chars (\\x1c-\\x1f FS/GS/RS/US, plus NUL/DEL) in a
    block must not survive into blocks.jsonl content (the chunk source) or the
    document_hash/merged_text. \\t/\\n inside the block are preserved."""
    parsed = tmp_path / "ctrl.parsed"
    ir = IRDoc(
        document_name="ctrl.pdf",
        document_format="pdf",
        doc_title="ctrl",
        split_option={},
        blocks=[
            IRBlock(
                content_template="a\x1cb\x1dc\x1ed\x1fe\x00\x7f\tkeep\nline",
                heading="H",
                level=1,
            )
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-ctrl", engine="mineru")

    meta, rows = _load_jsonl(parsed / "ctrl.blocks.jsonl")
    body = rows[0]["content"]
    assert not any(c in body for c in "\x1c\x1d\x1e\x1f\x00\x7f")
    # separators removed (no spurious split), whitespace controls kept.
    assert body == "abcde\tkeep\nline"
    # document_hash is derived from the cleaned merged_text.
    assert meta["document_hash"].startswith("sha256:")


@pytest.mark.offline
def test_writer_blockid_unchanged_when_no_control_chars(tmp_path: Path) -> None:
    """The control-char strip is a no-op for clean input: blockid/document_hash
    for a control-char-free block stay identical to the pre-change baseline
    (guards golden/byte-equivalence snapshots)."""
    parsed_clean = tmp_path / "clean.parsed"
    parsed_dirty = tmp_path / "dirty.parsed"
    clean = IRDoc(
        document_name="d.pdf",
        document_format="pdf",
        doc_title="d",
        split_option={},
        blocks=[IRBlock(content_template="hello world", heading="H", level=1)],
    )
    # Same logical content but with separators interspersed; after cleaning the
    # rendered text collapses to the clean form, so blockid must match.
    dirty = IRDoc(
        document_name="d.pdf",
        document_format="pdf",
        doc_title="d",
        split_option={},
        blocks=[IRBlock(content_template="hello\x1f world\x1c", heading="H", level=1)],
    )
    write_sidecar(clean, parsed_dir=parsed_clean, doc_id="doc-x", engine="mineru")
    write_sidecar(dirty, parsed_dir=parsed_dirty, doc_id="doc-x", engine="mineru")
    meta_c, rows_c = _load_jsonl(parsed_clean / "d.blocks.jsonl")
    meta_d, rows_d = _load_jsonl(parsed_dirty / "d.blocks.jsonl")
    assert rows_c[0]["content"] == rows_d[0]["content"] == "hello world"
    assert rows_c[0]["blockid"] == rows_d[0]["blockid"]
    assert meta_c["document_hash"] == meta_d["document_hash"]


@pytest.mark.offline
def test_writer_drops_block_that_is_only_control_chars(tmp_path: Path) -> None:
    """A block whose entire body is control chars + whitespace collapses to
    empty after cleaning and is dropped (not emitted as a blank row)."""
    parsed = tmp_path / "empty.parsed"
    ir = IRDoc(
        document_name="e.pdf",
        document_format="pdf",
        doc_title="e",
        split_option={},
        blocks=[
            IRBlock(content_template="\x1c\x1d  \x1f\x1e", heading="H", level=1),
            IRBlock(content_template="real body", heading="H2", level=1),
        ],
    )
    write_sidecar(ir, parsed_dir=parsed, doc_id="doc-drop", engine="mineru")
    _, rows = _load_jsonl(parsed / "e.blocks.jsonl")
    assert len(rows) == 1
    assert rows[0]["content"] == "real body"
