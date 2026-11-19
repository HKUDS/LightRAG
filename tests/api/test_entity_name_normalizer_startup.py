"""Validate the environment selection at each actual app startup entry point."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("factory", ["create_app", "get_application"])
def test_invalid_selection_fails_server_startup(monkeypatch, tmp_path, factory):
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", "missing-normalizer")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("UI_TEMPLATES_DIR", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lightrag.api.config import parse_args; "
            f"from lightrag.api.lightrag_server import {factory}; "
            f"{factory}(parse_args())",
            "--working-dir",
            str(tmp_path / "rag"),
            "--input-dir",
            str(tmp_path / "inputs"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "unknown normalizer 'missing-normalizer'" in result.stderr


def test_installed_entry_point_is_discovered_without_registration(
    monkeypatch, tmp_path
):
    dist_info = tmp_path / "example_names-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: example-names\nVersion: 1.0\n", encoding="utf-8"
    )
    (dist_info / "entry_points.txt").write_text(
        "[lightrag.entity_name_normalizers]\nexample = example_names:normalize\n",
        encoding="utf-8",
    )
    (tmp_path / "example_names.py").write_text(
        "def normalize(name):\n    return name.lower()\n", encoding="utf-8"
    )
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", "example")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from lightrag.utils import normalize_entity_name; "
            "assert normalize_entity_name('ACME') == 'acme'",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
