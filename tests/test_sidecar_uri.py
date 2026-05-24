"""Unit tests for the sidecar URI helpers and document-path canonicalization
introduced when ``full_docs`` collapsed its four path fields to
``file_path`` + ``sidecar_location``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.utils_pipeline import (
    SIDECAR_LOCATION_UNKNOWN,
    normalize_document_file_path,
    parsed_artifact_dir_for,
    resolve_sidecar_uri,
    sidecar_assets_dir_for_uri,
    sidecar_blocks_path,
    sidecar_modality_path,
    sidecar_uri_for,
)


@pytest.mark.offline
def test_normalize_strips_hint_and_directory():
    assert normalize_document_file_path("abc.[native-iet].docx") == "abc.docx"
    assert normalize_document_file_path("/tmp/sub/abc.docx") == "abc.docx"
    assert normalize_document_file_path("abc.docx") == "abc.docx"


@pytest.mark.offline
def test_normalize_idempotent():
    once = normalize_document_file_path("/tmp/abc.[native].docx")
    twice = normalize_document_file_path(once)
    assert once == twice == "abc.docx"


@pytest.mark.offline
@pytest.mark.parametrize(
    "value",
    ["", None, "no-file-path", "unknown_source", "  "],
)
def test_normalize_maps_placeholders_to_unknown(value):
    assert normalize_document_file_path(value) == "unknown_source"


@pytest.mark.offline
def test_sidecar_uri_round_trip_ascii(tmp_path):
    sidecar_dir = tmp_path / "abc.docx.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    assert uri.startswith("file://")
    assert uri.endswith("/")
    resolved = resolve_sidecar_uri(uri)
    assert resolved == sidecar_dir.resolve()


@pytest.mark.offline
def test_sidecar_uri_round_trip_unicode_and_spaces(tmp_path):
    sidecar_dir = tmp_path / "中文 报告.docx.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    assert uri.startswith("file://")
    assert " " not in uri  # spaces are percent-encoded
    resolved = resolve_sidecar_uri(uri)
    assert resolved == sidecar_dir.resolve()


@pytest.mark.offline
def test_resolve_sidecar_uri_tolerates_missing_trailing_slash(tmp_path):
    sidecar_dir = tmp_path / "demo.parsed"
    sidecar_dir.mkdir()
    uri_no_slash = sidecar_uri_for(sidecar_dir).rstrip("/")
    assert resolve_sidecar_uri(uri_no_slash) == sidecar_dir.resolve()


@pytest.mark.offline
@pytest.mark.parametrize(
    "uri",
    [None, "", SIDECAR_LOCATION_UNKNOWN, "s3://bucket/path/"],
)
def test_resolve_sidecar_uri_returns_none_for_unsupported(uri):
    assert resolve_sidecar_uri(uri) is None


@pytest.mark.offline
def test_sidecar_blocks_path_locates_jsonl(tmp_path):
    sidecar_dir = tmp_path / "demo.docx.parsed"
    sidecar_dir.mkdir()
    blocks = sidecar_dir / "demo.blocks.jsonl"
    blocks.write_text("", encoding="utf-8")
    uri = sidecar_uri_for(sidecar_dir)

    assert sidecar_blocks_path(uri) == str(blocks)
    assert sidecar_modality_path(uri, "tables") == str(sidecar_dir / "demo.tables.json")
    assert sidecar_assets_dir_for_uri(uri) == Path(sidecar_dir / "demo.blocks.assets")


@pytest.mark.offline
def test_sidecar_blocks_path_returns_none_when_missing(tmp_path):
    empty = tmp_path / "empty.parsed"
    empty.mkdir()
    uri = sidecar_uri_for(empty)
    assert sidecar_blocks_path(uri) is None
    assert sidecar_modality_path(uri, "drawings") is None
    assert sidecar_assets_dir_for_uri(uri) is None


@pytest.mark.offline
def test_parsed_artifact_dir_for_uses_input_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    result = parsed_artifact_dir_for("demo.docx")
    assert result == tmp_path / "__parsed__" / "demo.docx.parsed"


@pytest.mark.offline
def test_parsed_artifact_dir_for_strips_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    result = parsed_artifact_dir_for("abc.[native-iet].docx")
    assert result == tmp_path / "__parsed__" / "abc.docx.parsed"
