"""End-to-end: markdown → IRDoc → sidecar files, and re-parse stability.

Drives the parser's ``extract`` + ``build_ir`` hooks through the real
``write_sidecar`` (the same path ``NativeParserBase.parse`` uses) and asserts
the produced sidecar shape, then re-runs to confirm byte-stable content blocks
(no runtime stamping leaks into the per-block content).
"""

from __future__ import annotations

import json
from pathlib import Path

from lightrag.parser.markdown.parser import NativeMarkdownParser
from lightrag.sidecar import write_sidecar

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


def _build_sidecar(tmp_path: Path) -> Path:
    parser = NativeMarkdownParser()
    parsed_dir = tmp_path / "doc.md.parsed"
    asset_dir = parsed_dir / "doc.blocks.assets"
    parsed_dir.mkdir(parents=True)
    asset_dir.mkdir()
    blocks, _, meta = parser._extract_text(_MD, bundle_root=None)
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
