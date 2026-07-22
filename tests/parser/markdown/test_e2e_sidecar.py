"""End-to-end: markdown → IRDoc → sidecar files, and re-parse stability.

Drives the parser's ``extract`` + ``build_ir`` hooks through the real
``write_sidecar`` (the same path ``NativeParserBase.parse`` uses) and asserts
the produced sidecar shape, then re-runs to confirm byte-stable content blocks
(no runtime stamping leaks into the per-block content).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightrag.parser.markdown.parser as md_parser
from lightrag.parser.markdown.parser import NativeMarkdownParser
from lightrag.sidecar import write_sidecar
from lightrag.utils_pipeline import compute_text_content_hash

_MD = """# Doc Title

Intro ![x](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==).

## Section A

| Name | Age |
|------|-----|
| Bob  | 30  |

$$
E = mc^2
$$

<table><thead><tr><th>K</th></tr></thead><tbody><tr><td>a</td></tr></tbody></table>
"""


def _build_sidecar(tmp_path: Path, md_text: str = _MD) -> Path:
    parser = NativeMarkdownParser()
    parsed_dir = tmp_path / "doc.md.parsed"
    asset_dir = parsed_dir / "doc.blocks.assets"
    parsed_dir.mkdir(parents=True)
    asset_dir.mkdir()
    blocks, _, meta = parser._extract_text(md_text, bundle_root=None)
    ir = parser.build_ir(
        blocks,
        document_name="doc.md",
        asset_dir_name=asset_dir.name,
        metadata=meta,
    )
    write_sidecar(
        ir,
        parsed_dir=parsed_dir,
        doc_id="doc-test",
        engine="native",
        clean_parsed_dir=False,
        block_drawing_path_style=parser.sidecar_path_style,
    )
    return parsed_dir


def _content_lines(parsed_dir: Path) -> list[str]:
    lines = (parsed_dir / "doc.blocks.jsonl").read_text().splitlines()
    return [line for line in lines if json.loads(line).get("type") == "content"]


def test_sidecar_files_and_payload(tmp_path: Path):
    parsed_dir = _build_sidecar(tmp_path)
    assert (parsed_dir / "doc.blocks.jsonl").is_file()
    assert (parsed_dir / "doc.tables.json").is_file()
    assert (parsed_dir / "doc.equations.json").is_file()
    assert (parsed_dir / "doc.drawings.json").is_file()
    assets = list((parsed_dir / "doc.blocks.assets").iterdir())
    assert len(assets) == 1 and assets[0].read_bytes().startswith(b"\x89PNG")

    meta_line = json.loads(
        (parsed_dir / "doc.blocks.jsonl").read_text().splitlines()[0]
    )
    assert meta_line["document_format"] == "md"
    assert meta_line["doc_title"] == "Doc Title"
    assert meta_line["parse_engine"] == "native"

    tables = json.loads((parsed_dir / "doc.tables.json").read_text())["tables"]
    fmts = sorted(t["format"] for t in tables.values())
    assert fmts == ["html", "json"]
    # Pipe-table header survives as a JSON grid; HTML header as a <thead>.
    json_table = next(t for t in tables.values() if t["format"] == "json")
    assert json.loads(json_table["table_header"]) == [["Name", "Age"]]
    html_table = next(t for t in tables.values() if t["format"] == "html")
    assert "<thead>" in html_table["table_header"]

    equations = json.loads((parsed_dir / "doc.equations.json").read_text())["equations"]
    assert any(e["content"] == "E = mc^2" for e in equations.values())

    drawings = json.loads((parsed_dir / "doc.drawings.json").read_text())["drawings"]
    (drawing,) = drawings.values()
    assert drawing["path"].startswith("doc.blocks.assets/")
    assert drawing["format"] == "png"


def test_reparse_content_is_byte_stable(tmp_path: Path):
    first = _content_lines(_build_sidecar(tmp_path / "a"))
    second = _content_lines(_build_sidecar(tmp_path / "b"))
    assert first == second


_MD_EXTERNAL = """# T

See ![x](http://host/y.png) inline.
"""


def test_external_image_download_failure_renders_empty_path_with_src(
    tmp_path: Path, monkeypatch
):
    # A failed download falls back to an external-link drawing: nothing is
    # materialized, so ``path`` renders empty and the URL survives only in
    # ``src`` — never duplicated into ``path``.
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.delenv("NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED", raising=False)

    def _refuse(self, src):
        raise ValueError("download refused (test)")

    monkeypatch.setattr(md_parser._MarkdownImageResolver, "_download", _refuse)

    parsed_dir = _build_sidecar(tmp_path / "a", _MD_EXTERNAL)

    (tag_content,) = [
        json.loads(line)["content"]
        for line in _content_lines(parsed_dir)
        if "<drawing" in line
    ]
    assert 'path="" src="http://host/y.png"' in tag_content
    # Nothing materialized: the writer leaves no asset files behind (it
    # removes the pre-created empty assets dir entirely).
    asset_dir = parsed_dir / "doc.blocks.assets"
    assert not asset_dir.exists() or list(asset_dir.iterdir()) == []

    drawings = json.loads((parsed_dir / "doc.drawings.json").read_text())["drawings"]
    (drawing,) = drawings.values()
    assert drawing["path"] == ""
    assert drawing["src"] == "http://host/y.png"
    assert drawing["format"] == "png"

    # Identity anchoring: the new rendering is deterministic — blockids and
    # the dedup content hash reproduce across rebuilds (the migration off
    # the old path==src representation is a one-time, controlled shift).
    second_dir = _build_sidecar(tmp_path / "b", _MD_EXTERNAL)
    first_lines = _content_lines(parsed_dir)
    second_lines = _content_lines(second_dir)
    assert first_lines == second_lines
    assert [json.loads(line)["blockid"] for line in first_lines] == [
        json.loads(line)["blockid"] for line in second_lines
    ]
    merged = "\n\n".join(json.loads(line)["content"] for line in first_lines)
    merged_second = "\n\n".join(json.loads(line)["content"] for line in second_lines)
    assert compute_text_content_hash(merged) == compute_text_content_hash(merged_second)
