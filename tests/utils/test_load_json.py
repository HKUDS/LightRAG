"""load_json: empty or invalid store files must match the missing-file contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.utils import load_json

pytestmark = pytest.mark.offline


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_json(str(tmp_path / "no-such.json")) is None


@pytest.mark.parametrize("payload", [b"", b"   \n\t  "])
def test_empty_or_whitespace_file_returns_none(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "kv.json"
    path.write_bytes(payload)
    assert load_json(str(path)) is None
    assert (load_json(str(path)) or {}) == {}


def test_invalid_json_file_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "kv.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_json(str(path)) is None
    assert (load_json(str(path)) or {}) == {}


def test_valid_json_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "kv.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert load_json(str(path)) == {"a": 1}


def test_empty_object_json_is_not_none(tmp_path: Path) -> None:
    path = tmp_path / "kv.json"
    path.write_text("{}", encoding="utf-8")
    assert load_json(str(path)) == {}
