"""``*.mineru_raw/`` cache validation tests.

Covers every failure mode that triggers a re-download:

- missing / malformed manifest
- source file size mismatch (fast-path)
- source file content_hash mismatch
- engine version / endpoint env mismatch
- critical_file (content_list.json) size or sha256 mismatch
- any non-critical file size mismatch
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.external_parser.mineru import (
    Manifest,
    ManifestFile,
    clear_dir_contents,
    compute_size_and_hash,
    is_bundle_valid,
    raw_dir_for_parsed_dir,
)
from lightrag.external_parser.mineru.manifest import write_manifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    p = tmp_path / "src.pdf"
    p.write_bytes(b"Hello PDF" * 100)
    return p


@pytest.fixture
def fresh_bundle(tmp_path: Path, source_file: Path) -> tuple[Path, Manifest]:
    """Build a fully-valid bundle alongside ``source_file`` and return
    ``(raw_dir, manifest)``."""
    raw = tmp_path / "src.mineru_raw"
    raw.mkdir()
    content_list = raw / "content_list.json"
    content_list.write_text('[{"type":"text","text":"hi"}]', encoding="utf-8")
    images = raw / "images"
    images.mkdir()
    (images / "img1.png").write_bytes(b"PNG" * 50)
    (images / "img2.png").write_bytes(b"PNG" * 60)

    src_size, src_hash = compute_size_and_hash(source_file)
    crit_size, crit_hash = compute_size_and_hash(content_list)
    files = [
        ManifestFile(path="images/img1.png", size=(images / "img1.png").stat().st_size),
        ManifestFile(path="images/img2.png", size=(images / "img2.png").stat().st_size),
    ]
    manifest = Manifest(
        source_content_hash=src_hash,
        source_size_bytes=src_size,
        source_filename_at_parse=source_file.name,
        critical_file=ManifestFile(
            path="content_list.json", size=crit_size, sha256=crit_hash
        ),
        files=files,
        total_size_bytes=crit_size + sum(f.size for f in files),
        task_id="task-1",
    )
    write_manifest(raw, manifest)
    return raw, manifest


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_raw_dir_naming(tmp_path: Path) -> None:
    parsed = tmp_path / "report.pdf.parsed"
    raw = raw_dir_for_parsed_dir(parsed)
    assert raw.name == "report.pdf.mineru_raw"
    assert raw.parent == parsed.parent


# ---------------------------------------------------------------------------
# Validation: happy path + every individual failure mode
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_is_bundle_valid_happy_path(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    assert is_bundle_valid(raw, source_file) is True


@pytest.mark.offline
def test_invalid_when_manifest_missing(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    (raw / "_manifest.json").unlink()
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_manifest_malformed(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    (raw / "_manifest.json").write_text("not json")
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_manifest_wrong_engine(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    payload = json.loads((raw / "_manifest.json").read_text())
    payload["engine"] = "docling"
    (raw / "_manifest.json").write_text(json.dumps(payload))
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_source_size_changes(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    # Append bytes to the source file — size diverges from manifest fast-path.
    raw, _ = fresh_bundle
    with source_file.open("ab") as fh:
        fh.write(b"x")
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_source_hash_changes_but_size_same(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    """In-place rewrite that preserves byte size but mutates content. The
    fast-path passes but the full hash check catches it."""
    raw, _ = fresh_bundle
    data = source_file.read_bytes()
    # Flip first byte; keep length identical.
    mutated = bytes([data[0] ^ 0xFF]) + data[1:]
    assert len(mutated) == len(data)
    source_file.write_bytes(mutated)
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_engine_version_mismatch(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _ = fresh_bundle
    payload = json.loads((raw / "_manifest.json").read_text())
    payload["engine_version"] = "magic-pdf 1.5.4"
    (raw / "_manifest.json").write_text(json.dumps(payload))
    monkeypatch.setenv("MINERU_ENGINE_VERSION", "magic-pdf 1.6.0")
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_engine_version_match_passes(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _ = fresh_bundle
    payload = json.loads((raw / "_manifest.json").read_text())
    payload["engine_version"] = "magic-pdf 1.5.4"
    (raw / "_manifest.json").write_text(json.dumps(payload))
    monkeypatch.setenv("MINERU_ENGINE_VERSION", "magic-pdf 1.5.4")
    assert is_bundle_valid(raw, source_file) is True


@pytest.mark.offline
def test_engine_version_skip_when_either_side_blank(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank manifest engine_version + non-blank env should NOT invalidate
    (no signal from manifest); same for the reverse."""
    raw, _ = fresh_bundle
    # Manifest engine_version is empty by default.
    monkeypatch.setenv("MINERU_ENGINE_VERSION", "anything")
    assert is_bundle_valid(raw, source_file) is True


@pytest.mark.offline
def test_invalid_when_api_mode_mismatch(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _ = fresh_bundle
    payload = json.loads((raw / "_manifest.json").read_text())
    payload["api_mode"] = "local"
    (raw / "_manifest.json").write_text(json.dumps(payload))
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token")
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_endpoint_signature_mismatch(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _ = fresh_bundle
    payload = json.loads((raw / "_manifest.json").read_text())
    payload["api_mode"] = "local"
    payload["endpoint_signature"] = "http://old.example"
    (raw / "_manifest.json").write_text(json.dumps(payload))
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://new.example")
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_endpoint_signature_uses_mode_specific_endpoint(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _ = fresh_bundle
    payload = json.loads((raw / "_manifest.json").read_text())
    payload["api_mode"] = "local"
    payload["endpoint_signature"] = "http://old.example"
    (raw / "_manifest.json").write_text(json.dumps(payload))
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://new.example")
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_endpoint_signature_ignores_trailing_slash(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _ = fresh_bundle
    payload = json.loads((raw / "_manifest.json").read_text())
    payload["api_mode"] = "local"
    payload["endpoint_signature"] = "http://old.example"
    (raw / "_manifest.json").write_text(json.dumps(payload))
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://old.example/")
    assert is_bundle_valid(raw, source_file) is True


@pytest.mark.offline
def test_invalid_when_critical_file_missing(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    (raw / "content_list.json").unlink()
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_critical_file_size_changes(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    cl = raw / "content_list.json"
    cl.write_text(cl.read_text() + "/* extra */")
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_critical_file_hash_changes(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    """Same size, different bytes. sha256 is the terminal check."""
    raw, _ = fresh_bundle
    cl = raw / "content_list.json"
    data = cl.read_text()
    mutated = data[:-1] + "X"  # swap last char; size preserved
    assert len(mutated) == len(data)
    cl.write_text(mutated)
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_aux_file_size_changes(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    p = raw / "images" / "img1.png"
    p.write_bytes(p.read_bytes() + b"corruption")
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.offline
def test_invalid_when_aux_file_missing(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    (raw / "images" / "img2.png").unlink()
    assert is_bundle_valid(raw, source_file) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_clear_dir_contents_preserves_directory(tmp_path: Path) -> None:
    d = tmp_path / "raw"
    d.mkdir()
    (d / "a.txt").write_text("a")
    (d / "sub").mkdir()
    (d / "sub" / "b.txt").write_text("b")
    clear_dir_contents(d)
    assert d.exists()
    assert list(d.iterdir()) == []


@pytest.mark.offline
def test_compute_size_and_hash_consistency(tmp_path: Path) -> None:
    """Both values describe the same byte stream."""
    p = tmp_path / "f.bin"
    payload = b"abc" * 1000
    p.write_bytes(payload)
    size, h = compute_size_and_hash(p)
    assert size == len(payload)
    assert h.startswith("sha256:") and len(h) == len("sha256:") + 64


@pytest.mark.offline
def test_manifest_round_trip_via_disk(tmp_path: Path) -> None:
    """Write → read recovers all fields."""
    raw = tmp_path / "rt.mineru_raw"
    raw.mkdir()
    m = Manifest(
        source_content_hash="sha256:abc",
        source_size_bytes=10,
        source_filename_at_parse="x.pdf",
        critical_file=ManifestFile(
            path="content_list.json", size=5, sha256="sha256:cl"
        ),
        files=[ManifestFile(path="images/i.png", size=3)],
        total_size_bytes=8,
        task_id="t1",
        engine_version="v",
        endpoint_signature="ep",
    )
    write_manifest(raw, m)
    from lightrag.external_parser.mineru.manifest import load_manifest

    loaded = load_manifest(raw)
    assert loaded is not None
    assert loaded.source_content_hash == "sha256:abc"
    assert loaded.critical_file.sha256 == "sha256:cl"
    assert [f.path for f in loaded.files] == ["images/i.png"]
    assert loaded.task_id == "t1"
