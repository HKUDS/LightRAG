"""Tests for the docling raw → sidecar debug CLI.

Covers the ``python -m lightrag.external_parser.docling`` entry point —
the same callable wired to the ``lightrag-docling-sidecar`` console script.
The CLI is debug-only: it must run the adapter + writer against an existing
``*.docling_raw/`` bundle without touching cache / storage / docling-serve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lightrag.external_parser.docling.__main__ import main


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


def _read_meta(blocks_path: Path) -> dict[str, Any]:
    first_line = blocks_path.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(first_line)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DOCLING_BBOX_ATTRIBUTES", "DOCLING_ENGINE_VERSION"):
        monkeypatch.delenv(name, raising=False)


def test_cli_standard_raw_dir_writes_sidecar(tmp_path: Path) -> None:
    raw_dir = tmp_path / "demo.pdf.docling_raw"
    raw_dir.mkdir()
    (raw_dir / "demo.json").write_text(
        json.dumps(_make_main_json(with_table=True)), encoding="utf-8"
    )

    rc = main([str(raw_dir)])
    assert rc == 0

    parsed_dir = tmp_path / "demo.pdf.parsed"
    blocks_path = parsed_dir / "demo.blocks.jsonl"
    tables_path = parsed_dir / "demo.tables.json"
    assert blocks_path.is_file()
    assert tables_path.is_file()

    meta = _read_meta(blocks_path)
    assert meta["parse_engine"] == "docling"
    assert meta["document_name"] == "demo.pdf"
    assert meta["table_file"] is True
    assert meta["drawing_file"] is False


def test_cli_document_name_override_for_non_standard_raw_dir(tmp_path: Path) -> None:
    raw_dir = tmp_path / "weird-name"
    raw_dir.mkdir()
    (raw_dir / "bundle.json").write_text(
        json.dumps(_make_main_json()), encoding="utf-8"
    )

    rc = main([str(raw_dir), "--document-name", "report.pdf"])
    assert rc == 0

    parsed_dir = tmp_path / "report.pdf.parsed"
    blocks_path = parsed_dir / "report.blocks.jsonl"
    assert blocks_path.is_file()
    meta = _read_meta(blocks_path)
    assert meta["document_name"] == "report.pdf"


def test_cli_infers_document_name_from_origin_filename(tmp_path: Path) -> None:
    raw_dir = tmp_path / "weird-name"
    raw_dir.mkdir()
    (raw_dir / "bundle.json").write_text(
        json.dumps(_make_main_json(origin_filename="auto.pdf")),
        encoding="utf-8",
    )

    rc = main([str(raw_dir)])
    assert rc == 0

    parsed_dir = tmp_path / "auto.pdf.parsed"
    blocks_path = parsed_dir / "auto.blocks.jsonl"
    assert blocks_path.is_file()
    meta = _read_meta(blocks_path)
    assert meta["document_name"] == "auto.pdf"


def test_cli_default_doc_id_is_stable_across_runs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "demo.pdf.docling_raw"
    raw_dir.mkdir()
    (raw_dir / "demo.json").write_text(json.dumps(_make_main_json()), encoding="utf-8")
    blocks_path = tmp_path / "demo.pdf.parsed" / "demo.blocks.jsonl"

    assert main([str(raw_dir)]) == 0
    first_lines = blocks_path.read_text(encoding="utf-8").splitlines()
    first_meta = json.loads(first_lines[0])
    first_blockids = [json.loads(line)["blockid"] for line in first_lines[1:]]

    assert main([str(raw_dir)]) == 0
    second_lines = blocks_path.read_text(encoding="utf-8").splitlines()
    second_meta = json.loads(second_lines[0])
    second_blockids = [json.loads(line)["blockid"] for line in second_lines[1:]]

    assert first_meta["doc_id"].startswith("doc-")
    assert first_meta["doc_id"] == second_meta["doc_id"]
    assert first_blockids and first_blockids == second_blockids


def test_cli_doc_id_override(tmp_path: Path) -> None:
    raw_dir = tmp_path / "demo.pdf.docling_raw"
    raw_dir.mkdir()
    (raw_dir / "demo.json").write_text(
        json.dumps(_make_main_json(with_table=True)), encoding="utf-8"
    )

    override_id = "doc-" + "a" * 32
    rc = main([str(raw_dir), "--doc-id", override_id])
    assert rc == 0

    parsed_dir = tmp_path / "demo.pdf.parsed"
    meta = _read_meta(parsed_dir / "demo.blocks.jsonl")
    assert meta["doc_id"] == override_id

    tables = json.loads((parsed_dir / "demo.tables.json").read_text(encoding="utf-8"))[
        "tables"
    ]
    table_ids = list(tables.keys())
    assert table_ids
    assert all(tid.startswith("tb-" + "a" * 32 + "-") for tid in table_ids)


def test_cli_missing_raw_dir_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nonexistent.docling_raw"
    rc = main([str(missing)])
    assert rc == 1
    err = capsys.readouterr().err
    assert str(missing) in err


def test_cli_fails_when_document_name_cannot_be_inferred(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir = tmp_path / "weird-name"
    raw_dir.mkdir()
    payload = _make_main_json()
    payload.pop("origin")
    (raw_dir / "bundle.json").write_text(json.dumps(payload), encoding="utf-8")

    rc = main([str(raw_dir)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--document-name" in err


def test_cli_custom_output_dir(tmp_path: Path) -> None:
    raw_dir = tmp_path / "demo.pdf.docling_raw"
    raw_dir.mkdir()
    (raw_dir / "demo.json").write_text(json.dumps(_make_main_json()), encoding="utf-8")
    custom_out = tmp_path / "custom_out"

    rc = main([str(raw_dir), "-o", str(custom_out)])
    assert rc == 0
    assert (custom_out / "demo.blocks.jsonl").is_file()
    # Default sibling location must NOT exist when -o is supplied.
    assert not (tmp_path / "demo.pdf.parsed").exists()
