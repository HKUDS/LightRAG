"""MinerU IR builder tests: content_list.json → IR translation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.external_parser.mineru import MinerUIRBuilder


def _write_bundle(tmp_path: Path, content_list: list[dict]) -> Path:
    """Build a minimal *.mineru_raw/ directory."""
    raw = tmp_path / "doc.mineru_raw"
    raw.mkdir()
    (raw / "content_list.json").write_text(json.dumps(content_list, ensure_ascii=False))
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
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="x.pdf")

    assert ir.doc_title == "1 Introduction"
    assert ir.document_format == "pdf"
    # Heading + body merge into a single block per heading.
    assert len(ir.blocks) == 2
    assert ir.blocks[0].heading == "1 Introduction"
    assert ir.blocks[0].level == 1
    # Heading line is rendered with markdown ``#`` prefix matching the level.
    assert ir.blocks[0].content_template == "# 1 Introduction\nBody paragraph."
    # Sub-heading updates stack and records parent.
    assert ir.blocks[1].heading == "1.1 Sub"
    assert ir.blocks[1].level == 2
    assert ir.blocks[1].parent_headings == ["1 Introduction"]
    assert ir.blocks[1].content_template == "## 1.1 Sub\nSub body."


@pytest.mark.offline
def test_adapter_preface_block_for_pre_heading_content(tmp_path: Path) -> None:
    """Items emitted before the first heading land in a synthetic
    ``Preface/Uncategorized`` block at level 0."""
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "Floating intro line."},
            {"type": "list", "list_items": ["a", "b"]},
            {"type": "text", "text": "Section A", "text_level": 1},
            {"type": "text", "text": "A body."},
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="p.pdf")

    assert len(ir.blocks) == 2
    preface = ir.blocks[0]
    assert preface.heading == "Preface/Uncategorized"
    assert preface.level == 0
    assert preface.parent_headings == []
    assert preface.content_template == "Floating intro line.\na\nb"

    section = ir.blocks[1]
    assert section.heading == "Section A"
    assert section.level == 1
    assert section.content_template == "# Section A\nA body."


@pytest.mark.offline
def test_adapter_merges_mixed_payloads_under_heading(tmp_path: Path) -> None:
    """Tables / images / equations / code under the same heading merge into
    one block; their placeholders appear in document order."""
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "Methods", "text_level": 1},
            {"type": "text", "text": "We did stuff."},
            {
                "type": "table",
                "table_body": [["a", "b"], ["1", "2"]],
                "num_rows": 2,
                "num_cols": 2,
            },
            {"type": "image", "img_path": "images/fig1.png"},
            {"type": "equation", "text": "$$E = mc^2$$"},
            {"type": "code", "code_body": "print('ok')"},
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="m.pdf")
    assert len(ir.blocks) == 1
    block = ir.blocks[0]
    assert block.heading == "Methods"
    assert block.level == 1
    assert len(block.tables) == 1
    assert len(block.drawings) == 1
    assert len(block.equations) == 1
    # Lines are joined in source order; the heading carries its ``#`` prefix.
    expected_lines = [
        "# Methods",
        "We did stuff.",
        f"{{{{TBL:{block.tables[0].placeholder_key}}}}}",
        f"{{{{IMG:{block.drawings[0].placeholder_key}}}}}",
        f"{{{{EQ:{block.equations[0].placeholder_key}}}}}",
        "print('ok')",
    ]
    assert block.content_template == "\n".join(expected_lines)


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
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="d.pdf")

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
    # Position carried through. The bbox-bearing item produces exactly one
    # fine-grained position (anchor + range) and is NOT also rolled into the
    # page-only summary channel — so the block has a single position entry,
    # not a duplicate summary + bbox pair.
    assert len(drawing_block.positions) == 1
    assert drawing_block.positions[0].type == "bbox"
    # Anchor is always serialized as a string (uniform on-disk format,
    # accommodates book pagination labels like Roman "ii").
    assert drawing_block.positions[0].anchor == "2"  # page_idx+1
    assert drawing_block.positions[0].range == [10.0, 20.0, 30.0, 40.0]

    # Asset is declared with the relative path as ref.
    assert any(a.ref == "images/img_001.jpg" for a in ir.assets)

    equation_block = next(b for b in ir.blocks if b.equations)
    eq = equation_block.equations[0]
    # IREquation.latex preserves MinerU's raw form so blocks.jsonl shows it
    # verbatim; equations.json strips the ``$`` wrappers downstream (writer).
    assert eq.latex == "$E = mc^2$"
    assert eq.is_block is True
    assert eq.caption == "Eq 1"


@pytest.mark.offline
def test_adapter_page_idx_aggregated_and_deduped_when_no_bbox(
    tmp_path: Path,
) -> None:
    """Real MinerU output carries ``page_idx`` on every item but rarely a
    ``bbox``. Each unique page contributing to a merged block must surface as
    one anchor-only ``{type:"bbox", anchor:<page+1>}`` entry, sorted, no
    duplicates, no ``range``.
    """
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "Section", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "line A", "page_idx": 0},
            {"type": "text", "text": "line B", "page_idx": 1},
            {"type": "text", "text": "line C", "page_idx": 1},
            {"type": "text", "text": "line D", "page_idx": 2},
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="p.pdf")

    assert len(ir.blocks) == 1
    block = ir.blocks[0]
    # Pages 0, 1, 2 → anchors "1", "2", "3" — one entry per unique page.
    # Anchors are persisted as strings for on-disk uniformity.
    assert len(block.positions) == 3
    anchors = [p.anchor for p in block.positions]
    assert anchors == ["1", "2", "3"]
    for pos in block.positions:
        assert pos.type == "bbox"
        # Page-only summary entries have no range; ``to_jsonable`` must omit
        # the key entirely.
        assert pos.range is None
        assert "range" not in pos.to_jsonable()


@pytest.mark.offline
def test_adapter_bbox_items_and_page_only_items_coexist(tmp_path: Path) -> None:
    """When a block merges both bbox-bearing and bbox-less items, the bbox
    items are emitted per-item (no dedupe, with ``range``) and only the
    bbox-less items contribute to the page-only summary. Ordering: summary
    first (sorted by anchor), bbox entries after (source order).
    """
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "Mixed", "text_level": 1, "page_idx": 1},
            {
                "type": "image",
                "img_path": "images/fig.png",
                "page_idx": 1,
                "bbox": [10, 20, 30, 40],
            },
            {"type": "text", "text": "tail line", "page_idx": 2},
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="m.pdf")

    assert len(ir.blocks) == 1
    positions = ir.blocks[0].positions
    # One page-only summary for page 3 (the bbox-less tail line) and one
    # bbox entry for page 2 (the image). The heading item has page_idx=1
    # but no bbox, so it adds anchor 2 to the page set — combined with the
    # tail item's anchor 3 the summary section has TWO anchors (1+1, 2+1).
    assert [(p.anchor, p.range) for p in positions] == [
        ("2", None),
        ("3", None),
        ("2", [10.0, 20.0, 30.0, 40.0]),
    ]


@pytest.mark.offline
def test_adapter_page_sort_books_convention_with_mixed_anchors(
    tmp_path: Path,
) -> None:
    """Block merges items with Roman preface labels and Arabic numerals.

    Two guarantees:

    1. The adapter must not crash when sorting heterogeneous anchors — a
       previous bug surfaced ``TypeError: '<' not supported between
       instances of 'str' and 'int'`` whenever ``page_idx`` mixed types.
    2. Output order follows book pagination convention: Roman / letter
       labels first (lexical), then numeric pages by integer value, so
       ``"2"`` precedes ``"10"`` (not ``"10"`` before ``"2"`` as a naive
       lexical sort would do).
    """
    raw = _write_bundle(
        tmp_path,
        [
            {
                "type": "text",
                "text": "Mixed Pagination",
                "text_level": 1,
                "page_idx": "i",
            },
            {"type": "text", "text": "preface intro", "page_idx": "i"},
            {"type": "text", "text": "preface tail", "page_idx": "ii"},
            {"type": "text", "text": "chapter line A", "page_idx": 1},  # → "2"
            {"type": "text", "text": "chapter line B", "page_idx": 9},  # → "10"
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="mix.pdf")

    assert len(ir.blocks) == 1
    anchors = [p.anchor for p in ir.blocks[0].positions]
    # Roman labels first (lex order), then numerics by int value.
    assert anchors == ["i", "ii", "2", "10"]


@pytest.mark.offline
def test_adapter_empty_text_item_does_not_leak_page_to_block(
    tmp_path: Path,
) -> None:
    """An item whose body is empty must NOT contribute its ``page_idx`` to
    the current block's positions — otherwise spurious pages from
    content-less items poison provenance.

    Regression: empty text on page 99 sits between two real headings; its
    page must not appear under either block.
    """
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "Section A", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "real body", "page_idx": 0},
            # Empty body — should be silently dropped, page_idx not recorded.
            {"type": "text", "text": "", "page_idx": 98},
            {"type": "text", "text": "Section B", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "next body", "page_idx": 1},
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="leak.pdf")

    assert len(ir.blocks) == 2
    a_anchors = [p.anchor for p in ir.blocks[0].positions]
    b_anchors = [p.anchor for p in ir.blocks[1].positions]
    # Section A only mentions page 1 (page_idx 0 + 1) — NOT 99 from the
    # dropped empty item.
    assert a_anchors == ["1"]
    assert "99" not in a_anchors and "99" not in b_anchors
    # Section B only mentions page 2 (page_idx 1 + 1).
    assert b_anchors == ["2"]


@pytest.mark.offline
def test_adapter_adjacent_deeper_heading_merged_as_body(tmp_path: Path) -> None:
    """Two headings in a row with no body between them: when the second is
    strictly deeper (level number larger), it folds into the first heading's
    block as a body line. Mirrors the native docx parser's behaviour.
    """
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "1 Top", "text_level": 1},
            {"type": "text", "text": "1.1 Mid", "text_level": 2},
            {"type": "text", "text": "1.1.1 Deep", "text_level": 3},
            {"type": "text", "text": "Body for deep."},
            {"type": "text", "text": "2 Top Again", "text_level": 1},
            {"type": "text", "text": "More body."},
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="m.pdf")

    # First "1 Top" absorbs the immediately-following deeper headings;
    # body lands inside the same block. Then a new top-level heading
    # opens a fresh block.
    assert len(ir.blocks) == 2

    merged = ir.blocks[0]
    assert merged.heading == "1 Top"
    assert merged.level == 1
    assert merged.parent_headings == []
    assert merged.content_template == (
        "# 1 Top\n## 1.1 Mid\n### 1.1.1 Deep\nBody for deep."
    )

    fresh = ir.blocks[1]
    assert fresh.heading == "2 Top Again"
    assert fresh.level == 1
    # Heading stack reset cleanly — no stale deep parents leak.
    assert fresh.parent_headings == []
    assert fresh.content_template == "# 2 Top Again\nMore body."


@pytest.mark.offline
def test_adapter_adjacent_shallower_heading_starts_new_block(
    tmp_path: Path,
) -> None:
    """Inverse case: when the second adjacent heading is shallower (level
    number smaller or equal), it must NOT merge — it starts a new block.
    """
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "1.1 Mid first", "text_level": 2},
            {"type": "text", "text": "2 Top after", "text_level": 1},
            {"type": "text", "text": "body"},
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="m.pdf")

    # The first block is heading-only; the writer downstream will keep it
    # (the merged-heading rule only forwards DEEPER headings).
    assert len(ir.blocks) == 2
    assert ir.blocks[0].heading == "1.1 Mid first"
    assert ir.blocks[0].level == 2
    assert ir.blocks[0].content_template == "## 1.1 Mid first"

    assert ir.blocks[1].heading == "2 Top after"
    assert ir.blocks[1].level == 1
    assert ir.blocks[1].content_template == "# 2 Top after\nbody"


@pytest.mark.offline
def test_adapter_body_breaks_adjacent_heading_merge(tmp_path: Path) -> None:
    """Once any body content lands in the current block, the next heading —
    even a deeper one — must flush and open a fresh block (no merge)."""
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "text", "text": "1 Top", "text_level": 1},
            {"type": "text", "text": "Intro line under 1."},
            {"type": "text", "text": "1.1 Mid", "text_level": 2},
            {"type": "text", "text": "Mid body."},
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="m.pdf")

    assert len(ir.blocks) == 2
    assert ir.blocks[0].content_template == "# 1 Top\nIntro line under 1."
    assert ir.blocks[1].heading == "1.1 Mid"
    assert ir.blocks[1].parent_headings == ["1 Top"]
    assert ir.blocks[1].content_template == "## 1.1 Mid\nMid body."


@pytest.mark.offline
def test_adapter_block_equation_preserves_dollar_wrappers(tmp_path: Path) -> None:
    """Block equations keep the ``$$`` markers verbatim on IREquation.latex
    so the writer renders blocks.jsonl's ``<equation>`` body byte-identical
    to MinerU's source. The downstream writer is responsible for stripping
    them when generating equations.json."""
    raw = _write_bundle(
        tmp_path,
        [
            {
                "type": "equation",
                "text": "$$\n\\int_0^1 x dx = \\tfrac{1}{2}\n$$",
                "text_format": "block",
                "caption": "Eq A",
            },
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="b.pdf")
    eq = ir.blocks[0].equations[0]
    assert eq.is_block is True
    # No stripping in the adapter; whitespace.strip() only.
    assert eq.latex == "$$\n\\int_0^1 x dx = \\tfrac{1}{2}\n$$"


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
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="g.pdf")
    eq_count = sum(len(b.equations) for b in ir.blocks)
    assert eq_count == 0
    assert any(b.content_template == "kept" for b in ir.blocks)


@pytest.mark.offline
def test_adapter_bbox_attributes_default_and_override(tmp_path: Path) -> None:
    raw = _write_bundle(tmp_path, [{"type": "text", "text": "x"}])
    adapter = MinerUIRBuilder()
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
    adapter = MinerUIRBuilder()
    ir = adapter.normalize_from_workdir(raw, document_name="x.pdf")
    assert ir.bbox_attributes == {"origin": "LEFTBOTTOM", "max": 612}


@pytest.mark.offline
def test_adapter_engine_version_recorded_in_split_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINERU_ENGINE_VERSION", "magic-pdf 1.5.4")
    raw = _write_bundle(tmp_path, [{"type": "text", "text": "x"}])
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="x.pdf")
    assert ir.split_option == {"engine_version": "magic-pdf 1.5.4"}


@pytest.mark.offline
def test_adapter_missing_content_list_raises(tmp_path: Path) -> None:
    raw_dir = tmp_path / "bad.mineru_raw"
    raw_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        MinerUIRBuilder().normalize_from_workdir(raw_dir, document_name="x.pdf")


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
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="h.pdf")
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
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="l.pdf")
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

    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="a.pdf")
    by_ref = {a.ref: a for a in ir.assets}
    assert by_ref["images/exists.png"].source is not None
    assert by_ref["images/missing.png"].source is None


@pytest.mark.offline
def test_adapter_refuses_path_traversal_img_path(tmp_path: Path) -> None:
    """Untrusted img_path with ``..`` or absolute filesystem segments must
    not be allowed to point ``AssetSpec.source`` outside ``raw_dir``.

    Otherwise the writer would copy attacker-named files from the host into
    the sidecar's ``*.blocks.assets/`` directory (file-disclosure path).
    """
    # Place a "secret" file outside the raw bundle that should never be
    # selectable as an asset source.
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"private")
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "image", "img_path": "../secret.txt"},
            {"type": "image", "img_path": str(secret)},  # absolute path
        ],
    )
    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="x.pdf")
    by_ref = {a.ref: a for a in ir.assets}

    # Relative ``..`` escape is rejected outright.
    assert by_ref["../secret.txt"].source is None

    # Absolute filesystem path is reinterpreted as ``images/<basename>``
    # inside raw_dir. Since no such file exists, source must remain None
    # (and crucially must not point at the original secret file).
    abs_asset = by_ref[str(secret)]
    assert abs_asset.source is None


@pytest.mark.offline
def test_adapter_absolute_url_img_path_resolves_to_images_basename(
    tmp_path: Path,
) -> None:
    """When MinerU emits an absolute URL in img_path, the downloader saves
    it as ``images/<basename>``; the adapter must look there too."""
    raw = _write_bundle(
        tmp_path,
        [
            {
                "type": "image",
                "img_path": "https://cdn.example.com/imgs/figure_42.png",
            },
        ],
    )
    (raw / "images").mkdir()
    (raw / "images" / "figure_42.png").write_bytes(b"\x89PNGfake")

    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="u.pdf")
    asset = ir.assets[0]
    assert asset.ref == "https://cdn.example.com/imgs/figure_42.png"
    assert asset.suggested_name == "figure_42.png"
    assert asset.source is not None
    assert asset.source.read_bytes() == b"\x89PNGfake"


@pytest.mark.offline
def test_adapter_image_url_template_mode_maps_relative_to_images_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When MINERU_IMAGE_URL_TEMPLATE is set, MinerURawClient stores every
    image reference — including relative ones — at ``images/<basename>``.
    The adapter must mirror that lookup so the asset is wired up, otherwise
    the downloaded bytes are silently dropped from the sidecar."""
    monkeypatch.setenv(
        "MINERU_IMAGE_URL_TEMPLATE",
        "http://mineru.internal/assets/{name}",
    )
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "image", "img_path": "page/img.png"},
        ],
    )
    # Downloader's actual landing spot in template mode.
    (raw / "images").mkdir()
    (raw / "images" / "img.png").write_bytes(b"\x89PNGtemplate")
    # The "naive" location (raw_dir/page/img.png) does NOT exist; in
    # template mode the downloader does not write there.
    assert not (raw / "page" / "img.png").exists()

    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="t.pdf")
    asset = ir.assets[0]
    assert asset.source is not None
    assert asset.source.read_bytes() == b"\x89PNGtemplate"


@pytest.mark.offline
def test_adapter_no_template_keeps_relative_path_lookup(
    tmp_path: Path,
) -> None:
    """Sanity: without MINERU_IMAGE_URL_TEMPLATE, a relative img_path still
    resolves under raw_dir at its original location (regression guard for
    the template-mode change above)."""
    raw = _write_bundle(
        tmp_path,
        [
            {"type": "image", "img_path": "page/img.png"},
        ],
    )
    (raw / "page").mkdir()
    (raw / "page" / "img.png").write_bytes(b"\x89PNGrel")

    ir = MinerUIRBuilder().normalize_from_workdir(raw, document_name="r.pdf")
    asset = ir.assets[0]
    assert asset.source is not None
    assert asset.source.read_bytes() == b"\x89PNGrel"
