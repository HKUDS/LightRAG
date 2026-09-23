"""The setup wizard and the configuration storage anchor.

``make env-validate`` refuses an ``.env`` whose configuration backend resolves
to a TYPE other than the one the anchor binds -- the server refuses that start
too -- and ``make env-storage`` reports the anchor and warns on a mismatch.
The wizard only ever reads the anchor. See *The anchor and the container
identity* in docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.setup._helpers import REPO_ROOT, bash_bin, parse_lines, write_text_lines

pytestmark = pytest.mark.offline

UUID = "3f2b8c1e-6a4d-4e2f-9b7a-1c2d3e4f5a6b"

BASE_ENV = [
    "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
    "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
]


def _write_anchor(working_dir: Path, backend: str, *, raw: str | None = None) -> Path:
    path = working_dir / "_lightrag_config" / "storage_anchor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        raw = json.dumps(
            {"backend": backend, "schema_version": 1, "storage_uuid": UUID},
            indent=2,
            sort_keys=True,
        )
    path.write_text(raw)
    return path


def _run(case_dir: Path, body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            bash_bin(),
            "--norc",
            "--noprofile",
            "-c",
            f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{case_dir}"
reset_state
{body}
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _validate(case_dir: Path, env_lines: list[str]) -> subprocess.CompletedProcess:
    write_text_lines(case_dir / ".env", [*BASE_ENV, *env_lines])
    write_text_lines(case_dir / "env.example", ["LLM_BINDING=openai"])
    return _run(
        case_dir,
        """
if validate_env_file; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""",
    )


def test_validate_refuses_a_kv_change_the_anchor_does_not_bind(tmp_path: Path) -> None:
    """First bound on JSON; later only the KV backend moved to Mongo with the
    configuration unset -- the server would refuse, so validation does."""
    _write_anchor(tmp_path / "rag_storage", "JsonKVStorage")
    result = _validate(
        tmp_path,
        [
            "LIGHTRAG_KV_STORAGE=MongoKVStorage",
            "MONGO_URI=mongodb://localhost:27017/",
            "MONGO_DATABASE=LightRAG",
        ],
    )
    assert parse_lines(result.stdout)["VALID"] == "no"
    assert "binds this deployment to JsonKVStorage" in result.stderr
    assert "LIGHTRAG_CONFIG_STORAGE=JsonKVStorage" in result.stderr
    assert "lightrag-migrate-config --target-backend MongoKVStorage" in result.stderr


def test_validate_accepts_an_explicit_selection_of_the_anchored_backend(
    tmp_path: Path,
) -> None:
    _write_anchor(tmp_path / "rag_storage", "JsonKVStorage")
    result = _validate(
        tmp_path,
        [
            "LIGHTRAG_KV_STORAGE=MongoKVStorage",
            "LIGHTRAG_CONFIG_STORAGE=JsonKVStorage",
            "MONGO_URI=mongodb://localhost:27017/",
            "MONGO_DATABASE=LightRAG",
        ],
    )
    assert parse_lines(result.stdout)["VALID"] == "yes", result.stderr
    assert "storage_anchor" not in result.stderr


def test_validate_is_unchanged_without_an_anchor(tmp_path: Path) -> None:
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    assert parse_lines(result.stdout)["VALID"] == "yes", result.stderr


def test_validate_reads_the_anchor_under_the_configured_working_dir(
    tmp_path: Path,
) -> None:
    _write_anchor(tmp_path / "elsewhere", "PGKVStorage")
    result = _validate(
        tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage", "WORKING_DIR=./elsewhere"]
    )
    assert parse_lines(result.stdout)["VALID"] == "no"
    assert "binds this deployment to PGKVStorage" in result.stderr


def test_validate_reads_the_compose_mount_for_a_compose_runtime(
    tmp_path: Path,
) -> None:
    _write_anchor(tmp_path / "data" / "rag_storage", "PGKVStorage")
    result = _validate(
        tmp_path,
        [
            "LIGHTRAG_KV_STORAGE=JsonKVStorage",
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "WORKING_DIR=/app/data/rag_storage",
        ],
    )
    assert "binds this deployment to PGKVStorage" in result.stderr


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        '{"schema_version": 2, "backend": "JsonKVStorage", "storage_uuid": "%s"}'
        % UUID,
        '{"schema_version": 1, "backend": "RedisKVStorage", "storage_uuid": "%s"}'
        % UUID,
        '{"schema_version": 1, "backend": "JsonKVStorage", "storage_uuid": "x"}',
    ],
)
def test_an_anchor_the_wizard_cannot_confirm_is_a_warning_not_a_pass(
    tmp_path: Path, raw: str
) -> None:
    _write_anchor(tmp_path / "rag_storage", "", raw=raw)
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    assert parse_lines(result.stdout)["VALID"] == "yes", result.stderr
    assert "could not be read" in result.stderr


@pytest.mark.parametrize(
    "anchored, candidate, expected",
    [
        (
            "JsonKVStorage",
            "JsonKVStorage",
            "Configuration storage anchor: JsonKVStorage",
        ),
        ("PGKVStorage", "JsonKVStorage", "server will REFUSE to start"),
    ],
)
def test_env_storage_reports_the_anchor_and_warns_on_a_mismatch(
    tmp_path: Path, anchored: str, candidate: str, expected: str
) -> None:
    anchor = _write_anchor(tmp_path / "rag_storage", anchored)
    before = anchor.read_bytes()
    result = _run(tmp_path, f'report_config_anchor "{candidate}"')
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert anchor.read_bytes() == before


def test_env_storage_says_nothing_without_an_anchor(tmp_path: Path) -> None:
    result = _run(tmp_path, 'report_config_anchor "JsonKVStorage"')
    assert result.returncode == 0
    assert "anchor" not in result.stdout
    assert not (tmp_path / "rag_storage").exists()
