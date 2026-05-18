"""Tests for the unified parser debug CLI (``lightrag/parser_cli.py``).

The CLI behaviour under test is engine-agnostic: argument parsing, the
flat sidecar layout (no ``__parsed__/`` middle layer), the lenient raw
cache strategy (non-empty raw_dir reused without manifest checks), and
the no-archive guarantee on the source file.

We drive these checks via the **docling** engine path because docling's
raw bundle is the easiest to construct as static fixture (a single JSON
file) with zero external service or fixture-file dependency. The other
two engines exercise the same CLI code path:

- ``native`` would need a real ``.docx`` byte stream end-to-end (golden
  fixtures live under ``tests/native_parser/docx/golden/`` and have
  their own coverage via ``test_native_docx_golden.py``).
- ``mineru`` would need to mock ``MinerURawClient.download_into`` on
  the cache-miss path, or seed a mineru raw bundle layout (more files
  than docling's). Cache-hit reuses the same CLI orchestration as
  docling, so coverage here implicitly validates mineru's CLI wiring
  too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lightrag.parser_cli import main


def _make_main_json(
    *,
    origin_filename: str = "demo.pdf",
    with_table: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "origin": {"filename": origin_filename, "mimetype": "application/pdf"},
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}],
            "content_layer": "body",
            "label": "unspecified",
        },
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "title",
                "text": "Hello Title",
                "orig": "Hello Title",
                "content_layer": "body",
                "prov": [],
            },
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "Body line.",
                "orig": "Body line.",
                "content_layer": "body",
                "prov": [],
            },
        ],
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "form_items": [],
    }
    if with_table:
        payload["body"]["children"].append({"$ref": "#/tables/0"})
        payload["tables"].append(
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
        )
    return payload


def _seed_raw_dir(raw_dir: Path, *, with_table: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "demo.json").write_text(
        json.dumps(_make_main_json(with_table=with_table)),
        encoding="utf-8",
    )


def _read_meta(blocks_path: Path) -> dict[str, Any]:
    return json.loads(blocks_path.read_text(encoding="utf-8").splitlines()[0])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DOCLING_BBOX_ATTRIBUTES",
        "DOCLING_ENGINE_VERSION",
        "LIGHTRAG_FORCE_REPARSE_DOCLING",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cli_writes_sidecar_from_existing_raw_dir(tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.4\n")  # never read; presence is the only check
    _seed_raw_dir(tmp_path / "demo.pdf.docling_raw", with_table=True)

    rc = main([str(source), "--engine", "docling"])
    assert rc == 0

    parsed_dir = tmp_path / "demo.pdf.parsed"
    blocks_path = parsed_dir / "demo.blocks.jsonl"
    assert blocks_path.is_file()
    assert (parsed_dir / "demo.tables.json").is_file()

    meta = _read_meta(blocks_path)
    assert meta["parse_engine"] == "docling"
    assert meta["document_name"] == "demo.pdf"
    assert meta["table_file"] is True
    # Source file stays where it was — the CLI mocks the archive step.
    assert source.is_file()


def test_cli_doc_id_default_is_stable_across_runs(tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    _seed_raw_dir(tmp_path / "demo.pdf.docling_raw")

    blocks_path = tmp_path / "demo.pdf.parsed" / "demo.blocks.jsonl"

    assert main([str(source), "--engine", "docling"]) == 0
    first_lines = blocks_path.read_text(encoding="utf-8").splitlines()
    first_meta = json.loads(first_lines[0])
    first_block_ids = [json.loads(line)["blockid"] for line in first_lines[1:]]

    assert main([str(source), "--engine", "docling"]) == 0
    second_lines = blocks_path.read_text(encoding="utf-8").splitlines()
    second_meta = json.loads(second_lines[0])
    second_block_ids = [json.loads(line)["blockid"] for line in second_lines[1:]]

    assert first_meta["doc_id"].startswith("doc-")
    assert first_meta["doc_id"] == second_meta["doc_id"]
    assert first_block_ids and first_block_ids == second_block_ids


def test_cli_doc_id_override(tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    _seed_raw_dir(tmp_path / "demo.pdf.docling_raw", with_table=True)

    override = "doc-" + "a" * 32
    rc = main([str(source), "--engine", "docling", "--doc-id", override])
    assert rc == 0

    parsed_dir = tmp_path / "demo.pdf.parsed"
    meta = _read_meta(parsed_dir / "demo.blocks.jsonl")
    assert meta["doc_id"] == override

    tables = json.loads((parsed_dir / "demo.tables.json").read_text(encoding="utf-8"))[
        "tables"
    ]
    assert tables
    assert all(tid.startswith("tb-" + "a" * 32 + "-") for tid in tables)


def test_cli_custom_sidecar_parent_dir(tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    custom_parent = tmp_path / "elsewhere"
    custom_parent.mkdir()
    _seed_raw_dir(custom_parent / "demo.pdf.docling_raw")

    rc = main([str(source), "--engine", "docling", "-o", str(custom_parent)])
    assert rc == 0
    assert (custom_parent / "demo.pdf.parsed" / "demo.blocks.jsonl").is_file()
    # Nothing should land in the source's parent directory.
    assert not (tmp_path / "demo.pdf.parsed").exists()
    # Source file is preserved in place.
    assert source.is_file()


def test_cli_missing_input_file_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.pdf"
    rc = main([str(missing), "--engine", "docling"])
    assert rc == 1
    err = capsys.readouterr().err
    assert str(missing.resolve()) in err
