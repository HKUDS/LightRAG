"""Tests for ``lightrag/parser/external/docling/cache.py``.

Covers the cache-miss conditions enumerated in the module docstring:

- missing / malformed / wrong-engine manifest
- source size or hash mismatch
- engine_version / endpoint_signature env mismatch
- options_signature env mismatch
- critical-file size / sha256 mismatch
- non-critical file size mismatch
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.parser.external import Manifest, ManifestFile, write_manifest
from lightrag.parser.external._common import compute_size_and_hash
from lightrag.parser.external.docling.cache import (
    compute_options_signature,
    is_bundle_valid,
    snapshot_tunable_env,
)
from lightrag.parser.external.docling.client import FIXED_CONSTANTS


@pytest.fixture(autouse=True)
def _clear_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DOCLING_DO_OCR",
        "DOCLING_FORCE_OCR",
        "DOCLING_OCR_ENGINE",
        "DOCLING_OCR_PRESET",
        "DOCLING_OCR_LANG",
        "DOCLING_DO_FORMULA_ENRICHMENT",
        "DOCLING_ENGINE_VERSION",
        "DOCLING_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    p = tmp_path / "src.pdf"
    p.write_bytes(b"hello pdf payload" * 64)
    return p


def test_snapshot_tunable_env_uses_effective_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unset = snapshot_tunable_env()

    monkeypatch.setenv("DOCLING_DO_OCR", "true")
    monkeypatch.setenv("DOCLING_FORCE_OCR", "true")
    monkeypatch.setenv("DOCLING_OCR_ENGINE", "auto")
    monkeypatch.setenv("DOCLING_OCR_PRESET", "auto")
    monkeypatch.setenv("DOCLING_OCR_LANG", "")
    monkeypatch.setenv("DOCLING_DO_FORMULA_ENRICHMENT", "false")

    assert snapshot_tunable_env() == unset


def _build_valid_bundle(
    tmp_path: Path,
    source_file: Path,
    *,
    options_signature: str | None = None,
) -> Path:
    raw_dir = tmp_path / "src.docling_raw"
    raw_dir.mkdir()
    main_json = raw_dir / "src.json"
    main_json.write_text('{"schema_name": "DoclingDocument"}', encoding="utf-8")
    md = raw_dir / "src.md"
    md.write_text("# title", encoding="utf-8")

    src_size, src_hash = compute_size_and_hash(source_file)
    crit_size, crit_hash = compute_size_and_hash(main_json)
    sig = options_signature
    if sig is None:
        sig = compute_options_signature(
            tunable_env=snapshot_tunable_env(),
            fixed_constants=FIXED_CONSTANTS,
        )
    manifest = Manifest(
        engine="docling",
        source_content_hash=src_hash,
        source_size_bytes=src_size,
        source_filename_at_parse=source_file.name,
        critical_file=ManifestFile(path="src.json", size=crit_size, sha256=crit_hash),
        files=[ManifestFile(path="src.md", size=md.stat().st_size)],
        total_size_bytes=crit_size + md.stat().st_size,
        task_id="task-1",
        endpoint_signature="http://docling.test",
        engine_version="",
        options_signature=sig,
        extras={"fixed_constants": dict(FIXED_CONSTANTS)},
    )
    write_manifest(raw_dir, manifest)
    return raw_dir


def test_is_bundle_valid_happy_path(tmp_path: Path, source_file: Path) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    assert is_bundle_valid(raw, source_file) is True


def test_is_bundle_valid_missing_dir(tmp_path: Path, source_file: Path) -> None:
    assert is_bundle_valid(tmp_path / "ghost", source_file) is False


def test_is_bundle_valid_missing_manifest(tmp_path: Path, source_file: Path) -> None:
    raw = tmp_path / "src.docling_raw"
    raw.mkdir()
    (raw / "src.json").write_text("{}")
    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_wrong_engine(tmp_path: Path, source_file: Path) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    manifest_path = raw / "_manifest.json"
    data = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(data.replace('"docling"', '"mineru"'), encoding="utf-8")
    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_source_size_mismatch(
    tmp_path: Path, source_file: Path
) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    source_file.write_bytes(source_file.read_bytes() + b"!")
    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_source_hash_mismatch(
    tmp_path: Path, source_file: Path
) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    # Replace contents with same length but different bytes
    new = b"Y" * source_file.stat().st_size
    source_file.write_bytes(new)
    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_endpoint_change(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://other:5001")
    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_options_signature_change(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    monkeypatch.setenv("DOCLING_FORCE_OCR", "false")
    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_legacy_bundle_misses_with_overrides(
    tmp_path: Path, source_file: Path
) -> None:
    # A legacy bundle predating signature recording (empty options_signature)
    # is leniently accepted with no overrides, but a per-file override
    # (docling(force_ocr=...)) must force a miss — we cannot prove the bundle
    # was produced with that override, and silently reusing it would drop the
    # user's explicit param. Mirrors MinerU, which misses on any absent sig.
    raw = _build_valid_bundle(tmp_path, source_file, options_signature="")
    assert is_bundle_valid(raw, source_file) is True
    assert is_bundle_valid(raw, source_file, overrides={"force_ocr": False}) is False
    assert is_bundle_valid(raw, source_file, overrides={"force_ocr": True}) is False


def test_is_bundle_valid_fixed_constants_code_change(
    tmp_path: Path, source_file: Path
) -> None:
    # Simulate a code-only change to one of the fixed pipeline constants
    # (e.g. image_export_mode flipped from "referenced" to "embedded"
    # between parse time and validation time). The manifest stores both
    # the stale constants and a signature computed from them; validation
    # must compare against current FIXED_CONSTANTS and miss, not against
    # the manifest's own copy (which would always match).
    stale_constants = {**FIXED_CONSTANTS, "image_export_mode": "embedded"}
    stale_signature = compute_options_signature(
        tunable_env=snapshot_tunable_env(),
        fixed_constants=stale_constants,
    )
    raw = _build_valid_bundle(tmp_path, source_file, options_signature=stale_signature)
    # Overwrite the manifest's extras to record the stale constants too —
    # this is the bug surface: if validation rehydrated from extras, it
    # would reproduce stale_signature and falsely accept the bundle.
    import json as _json

    mp = raw / "_manifest.json"
    data = _json.loads(mp.read_text(encoding="utf-8"))
    data["extras"] = {"fixed_constants": stale_constants}
    mp.write_text(_json.dumps(data), encoding="utf-8")

    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_critical_file_corrupt(
    tmp_path: Path, source_file: Path
) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    # Corrupt the JSON: same length, different bytes — defeats size check,
    # so the sha256 path must catch it.
    current = (raw / "src.json").read_bytes()
    (raw / "src.json").write_bytes(b"X" * len(current))
    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_other_file_size_mismatch(
    tmp_path: Path, source_file: Path
) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    (raw / "src.md").write_text("totally different content here that is longer")
    assert is_bundle_valid(raw, source_file) is False
