"""Tests for shared helpers in ``lightrag/external_parser/``.

These cover the pure functions reused across engine integrations:

- ``compute_size_and_hash`` — single-read (size, hash) pair
- ``clear_dir_contents`` — empty a directory while keeping it
- ``raw_dir_for_parsed_dir`` — suffix-bound raw dir naming
- ``safe_extract_zip`` — refuses path traversal and absolute paths
- ``env_bool`` / ``env_int`` / ``env_json`` — env parsing
- ``Manifest`` round-trip via ``write_manifest`` / ``load_manifest``
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from lightrag.external_parser import (
    Manifest,
    ManifestFile,
    clear_dir_contents,
    compute_size_and_hash,
    env_bool,
    env_int,
    env_json,
    load_manifest,
    raw_dir_for_parsed_dir,
    safe_extract_zip,
    write_manifest,
)


# ---------------------------------------------------------------------------
# compute_size_and_hash
# ---------------------------------------------------------------------------


def test_compute_size_and_hash_stable(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    payload = b"hello-external-parser" * 1024
    p.write_bytes(payload)

    size_a, hash_a = compute_size_and_hash(p)
    size_b, hash_b = compute_size_and_hash(p)

    assert size_a == len(payload) == size_b
    assert hash_a == hash_b
    assert hash_a.startswith("sha256:") and len(hash_a) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# clear_dir_contents
# ---------------------------------------------------------------------------


def test_clear_dir_contents_keeps_dir_and_removes_children(tmp_path: Path) -> None:
    d = tmp_path / "raw"
    d.mkdir()
    (d / "a.txt").write_text("hi")
    sub = d / "nested"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"x" * 10)

    clear_dir_contents(d)

    assert d.is_dir()
    assert list(d.iterdir()) == []


def test_clear_dir_contents_noop_when_missing(tmp_path: Path) -> None:
    clear_dir_contents(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# raw_dir_for_parsed_dir
# ---------------------------------------------------------------------------


def test_raw_dir_for_parsed_dir_with_suffix(tmp_path: Path) -> None:
    parsed = tmp_path / "demo.pdf.parsed"
    raw = raw_dir_for_parsed_dir(parsed, suffix=".docling_raw")
    assert raw == tmp_path / "demo.pdf.docling_raw"


def test_raw_dir_for_parsed_dir_without_parsed_suffix(tmp_path: Path) -> None:
    parsed = tmp_path / "other_dir"
    raw = raw_dir_for_parsed_dir(parsed, suffix=".docling_raw")
    assert raw == tmp_path / "other_dir.docling_raw"


def test_raw_dir_for_parsed_dir_rejects_bad_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        raw_dir_for_parsed_dir(tmp_path / "x.parsed", suffix="docling_raw")


# ---------------------------------------------------------------------------
# safe_extract_zip
# ---------------------------------------------------------------------------


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def test_safe_extract_zip_extracts_flat_bundle(tmp_path: Path) -> None:
    payload = _make_zip(
        {
            "demo.json": b'{"schema_name": "DoclingDocument"}',
            "demo.md": b"# demo",
            "artifacts/image_000000.png": b"\x89PNG fake",
        }
    )
    dest = tmp_path / "raw"
    names = safe_extract_zip(payload, dest)

    assert (dest / "demo.json").read_bytes().startswith(b'{"schema_name"')
    assert (dest / "demo.md").read_text(encoding="utf-8") == "# demo"
    assert (dest / "artifacts" / "image_000000.png").is_file()
    assert sorted(names) == sorted(
        ["demo.json", "demo.md", "artifacts/image_000000.png"]
    )


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    payload = _make_zip({"../evil.txt": b"oops"})
    with pytest.raises(RuntimeError, match="unsafe path"):
        safe_extract_zip(payload, tmp_path / "raw")


def test_safe_extract_zip_rejects_absolute_path(tmp_path: Path) -> None:
    payload = _make_zip({"/etc/passwd": b"oops"})
    with pytest.raises(RuntimeError, match="unsafe path"):
        safe_extract_zip(payload, tmp_path / "raw")


# ---------------------------------------------------------------------------
# env coercion
# ---------------------------------------------------------------------------


def test_env_bool_truthy_falsy(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw in ("1", "true", "yes", "ON"):
        monkeypatch.setenv("X", raw)
        assert env_bool("X", False) is True
    for raw in ("0", "false", "no", "off"):
        monkeypatch.setenv("X", raw)
        assert env_bool("X", True) is False


def test_env_bool_falls_back_on_unrecognized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", "maybe")
    assert env_bool("X", True) is True
    assert env_bool("X", False) is False


def test_env_int_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", "not-an-int")
    assert env_int("X", 7) == 7


def test_env_json_returns_default_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", "{bad json")
    assert env_json("X", {"origin": "LEFTBOTTOM"}) == {"origin": "LEFTBOTTOM"}


def test_env_json_parses_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", '{"a": 1, "b": [2, 3]}')
    assert env_json("X", None) == {"a": 1, "b": [2, 3]}


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    raw = tmp_path / "demo.docling_raw"
    raw.mkdir()
    crit = ManifestFile(path="demo.json", size=42, sha256="sha256:" + "a" * 64)
    other = ManifestFile(path="demo.md", size=10)
    manifest = Manifest(
        engine="docling",
        source_content_hash="sha256:" + "b" * 64,
        source_size_bytes=100,
        source_filename_at_parse="demo.pdf",
        critical_file=crit,
        files=[other],
        total_size_bytes=52,
        task_id="task-xyz",
        endpoint_signature="http://l4ai:5001",
        engine_version="1.18.0",
        options_signature="sha256:" + "c" * 64,
        downloaded_at="2026-05-18T00:00:00Z",
        extras={"to_formats": ["json", "md"]},
    )
    write_manifest(raw, manifest)

    payload = json.loads((raw / "_manifest.json").read_text(encoding="utf-8"))
    assert payload["engine"] == "docling"
    assert payload["options_signature"] == "sha256:" + "c" * 64
    assert payload["extras"] == {"to_formats": ["json", "md"]}

    loaded = load_manifest(raw, expected_engine="docling")
    assert loaded is not None
    assert loaded.task_id == "task-xyz"
    assert loaded.critical_file.size == 42
    assert loaded.files[0].path == "demo.md"


def test_manifest_load_rejects_wrong_engine(tmp_path: Path) -> None:
    raw = tmp_path / "demo.docling_raw"
    raw.mkdir()
    manifest = Manifest(
        engine="mineru",
        source_content_hash="sha256:" + "0" * 64,
        source_size_bytes=1,
        source_filename_at_parse="x",
        critical_file=ManifestFile(path="c", size=1, sha256="sha256:" + "1" * 64),
        files=[],
        total_size_bytes=1,
    )
    write_manifest(raw, manifest)
    assert load_manifest(raw, expected_engine="docling") is None


def test_manifest_load_handles_missing_file(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "no-such-dir", expected_engine="docling") is None
