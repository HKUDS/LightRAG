"""Unit tests for the sidecar URI helpers and document-path canonicalization
introduced when ``full_docs`` collapsed its four path fields to
``file_path`` + ``sidecar_location``.
"""

from __future__ import annotations

import os
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
    assert "file:////" not in uri
    resolved = resolve_sidecar_uri(uri)
    assert resolved == sidecar_dir.resolve()


@pytest.mark.offline
def test_sidecar_uri_round_trip_unicode_and_spaces(tmp_path):
    sidecar_dir = tmp_path / "中文 报告.docx.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    assert uri.startswith("file://")
    assert " " not in uri  # spaces are percent-encoded
    assert "file:////" not in uri
    resolved = resolve_sidecar_uri(uri)
    assert resolved == sidecar_dir.resolve()


@pytest.mark.offline
def test_sidecar_uri_has_no_extra_slash_after_authority(tmp_path):
    """Path.as_uri() must not emit file://// on POSIX absolute paths."""
    sidecar_dir = tmp_path / "abc.docx.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    assert "file:////" not in uri
    assert uri.endswith("/")


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
def test_resolve_sidecar_uri_strips_windows_drive_leading_slash(monkeypatch):
    """``file:///C:/...`` → ``C:/...`` on Windows (CI is Linux — fake os.name)."""
    monkeypatch.setattr(os, "name", "nt")
    uri = "file:///E:/hot100/contributor/LightRAG/parsed/doc.parsed/"
    resolved = resolve_sidecar_uri(uri)
    assert resolved == Path("E:/hot100/contributor/LightRAG/parsed/doc.parsed")


@pytest.mark.offline
def test_resolve_sidecar_uri_windows_drive_without_nt_keeps_leading_slash(
    monkeypatch,
):
    """On non-Windows (CI Linux), ``/E:/...`` is not rewritten (nt-only branch)."""
    monkeypatch.setattr(os, "name", "posix")
    uri = "file:///E:/hot100/parsed/doc.parsed/"
    resolved = resolve_sidecar_uri(uri)
    assert resolved == Path("/E:/hot100/parsed/doc.parsed")


@pytest.mark.offline
def test_resolve_sidecar_uri_preserves_unc_authority(monkeypatch):
    """``file://server/share/...`` keeps the host (Path.as_uri UNC round trip)."""
    monkeypatch.setattr(os, "name", "nt")
    uri = "file://fileserver/docs/report.docx.parsed/"
    resolved = resolve_sidecar_uri(uri)
    assert resolved == Path("//fileserver/docs/report.docx.parsed")
    # Drive-letter strip must not apply to UNC (still starts with //).
    assert str(resolved).replace("\\", "/").startswith("//fileserver/")


@pytest.mark.offline
def test_resolve_sidecar_uri_unc_with_percent_encoding(monkeypatch):
    """UNC share segments remain unquoted after authority+path join."""
    monkeypatch.setattr(os, "name", "nt")
    uri = "file://fileserver/share/my%20doc.parsed/"
    resolved = resolve_sidecar_uri(uri)
    assert resolved == Path("//fileserver/share/my doc.parsed")


@pytest.mark.offline
def test_resolve_sidecar_uri_legacy_netloc_only_drive(monkeypatch):
    """Empty path + netloc still uses the legacy drive-in-netloc branch."""
    monkeypatch.setattr(os, "name", "nt")
    uri = "file://E:"
    resolved = resolve_sidecar_uri(uri)
    assert resolved == Path("E:")


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
