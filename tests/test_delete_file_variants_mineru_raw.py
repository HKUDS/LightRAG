"""``delete_file_variants_by_file_path`` now also clears the
MinerU raw bundle (``*.mineru_raw/``) alongside the sidecar
(``*.parsed/``) and source file when ``delete_file=True`` is selected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# document_routes pulls in api.auth → api.config.parse_args(); blank argv
# to keep argparse from rejecting pytest's flags at import time.
sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import (  # noqa: E402
    _file_path_for_parsed_artifact_dir,
    delete_file_variants_by_file_path,
)
from lightrag.constants import PARSED_DIR_NAME  # noqa: E402


@pytest.mark.offline
def test_canonical_basename_recognizes_both_suffixes() -> None:
    assert _file_path_for_parsed_artifact_dir("foo.pdf.parsed") == "foo.pdf"
    assert (
        _file_path_for_parsed_artifact_dir("foo.pdf.mineru_raw") == "foo.pdf"
    )
    # archive variants (parsed_001, mineru_raw_002, ...) handled
    assert (
        _file_path_for_parsed_artifact_dir("foo.pdf.parsed_001") == "foo.pdf"
    )
    assert (
        _file_path_for_parsed_artifact_dir("foo.pdf.mineru_raw_002")
        == "foo.pdf"
    )
    # unrelated names don't match
    assert _file_path_for_parsed_artifact_dir("foo.parsed.bak") is None
    assert _file_path_for_parsed_artifact_dir("notes.txt") is None


@pytest.mark.offline
def test_delete_file_variants_removes_parsed_and_mineru_raw(
    tmp_path: Path,
) -> None:
    """Both sidecar dir and raw bundle dir should disappear in one call."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    parsed_root = input_dir / PARSED_DIR_NAME
    parsed_root.mkdir()

    # Original source file (already moved into __parsed__/ post-archive).
    archived_source = parsed_root / "demo.pdf"
    archived_source.write_bytes(b"PDFCONTENT")

    parsed_dir = parsed_root / "demo.pdf.parsed"
    parsed_dir.mkdir()
    (parsed_dir / "demo.blocks.jsonl").write_text("{}\n")

    raw_dir = parsed_root / "demo.pdf.mineru_raw"
    raw_dir.mkdir()
    (raw_dir / "_manifest.json").write_text("{}")
    (raw_dir / "content_list.json").write_text("[]")
    (raw_dir / "images").mkdir()
    (raw_dir / "images" / "img.png").write_bytes(b"png")

    deleted, errors = delete_file_variants_by_file_path(
        input_dir, file_path="demo.pdf"
    )

    # Both directories and the archived source file were deleted.
    assert errors == []
    deleted_names = {Path(p).name for p in deleted}
    assert "demo.pdf.parsed" in deleted_names
    assert "demo.pdf.mineru_raw" in deleted_names
    assert "demo.pdf" in deleted_names

    assert not parsed_dir.exists()
    assert not raw_dir.exists()
    assert not archived_source.exists()


@pytest.mark.offline
def test_delete_file_variants_handles_only_raw_dir(tmp_path: Path) -> None:
    """The raw bundle may exist without a corresponding sidecar (parse
    aborted between download and adapter). Delete should still pick it up.
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    parsed_root = input_dir / PARSED_DIR_NAME
    parsed_root.mkdir()

    raw_dir = parsed_root / "demo.pdf.mineru_raw"
    raw_dir.mkdir()
    (raw_dir / "_manifest.json").write_text("{}")

    deleted, errors = delete_file_variants_by_file_path(
        input_dir, file_path="demo.pdf"
    )
    assert errors == []
    assert any("demo.pdf.mineru_raw" in p for p in deleted)
    assert not raw_dir.exists()


@pytest.mark.offline
def test_delete_file_variants_leaves_unrelated_dirs(tmp_path: Path) -> None:
    """Directories that don't match the canonical filename are untouched."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    parsed_root = input_dir / PARSED_DIR_NAME
    parsed_root.mkdir()

    target_raw = parsed_root / "demo.pdf.mineru_raw"
    target_raw.mkdir()
    other_raw = parsed_root / "other.pdf.mineru_raw"
    other_raw.mkdir()
    other_parsed = parsed_root / "other.pdf.parsed"
    other_parsed.mkdir()

    deleted, errors = delete_file_variants_by_file_path(
        input_dir, file_path="demo.pdf"
    )
    assert errors == []
    assert not target_raw.exists()
    assert other_raw.exists(), "siblings for a different basename must survive"
    assert other_parsed.exists()
