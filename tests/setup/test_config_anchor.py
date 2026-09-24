"""The setup wizard and the configuration storage anchor.

``make env-validate`` refuses an ``.env`` whose configuration backend resolves
to a TYPE other than the one the anchor binds -- the server refuses that start
too -- and ``make env-storage`` reports the anchor and warns on a mismatch.
The wizard only ever reads the anchor. See *The anchor and the container
identity* in docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import json
import os
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


# The wizard's parser must never say "readable" about a file the server
# refuses, nor read a different backend out of it than the server would.
@pytest.mark.parametrize(
    "raw",
    [
        # An extra member: the server requires exactly the three fields.
        '{"schema_version": 1, "backend": "JsonKVStorage", "storage_uuid": "%s",'
        ' "note": "x"}' % UUID,
        # A duplicate key alongside an extra one.
        '{"schema_version": 1, "backend": "JsonKVStorage", "backend":'
        ' "JsonKVStorage", "storage_uuid": "%s", "x": 1}' % UUID,
        # An empty object, and a trailing comma.
        "{}",
        '{"schema_version": 1, "backend": "JsonKVStorage", "storage_uuid": "%s",}'
        % UUID,
        # A raw newline inside a string is not JSON.
        '{"schema_version": 1, "backend": "JsonKV\nStorage", "storage_uuid": "%s"}'
        % UUID,
        # Not a JSON integer 1.
        '{"schema_version": 1.0, "backend": "JsonKVStorage", "storage_uuid": "%s"}'
        % UUID,
        # Trailing content after the object.
        '{"schema_version": 1, "backend": "JsonKVStorage", "storage_uuid": "%s"} x'
        % UUID,
        # A byte-order mark, which the server's UTF-8 decode keeps.
        '﻿{"schema_version": 1, "backend": "JsonKVStorage", "storage_uuid":'
        ' "%s"}' % UUID,
        # A NUL byte, which bash would silently drop.
        '{"schema_version": 1, "backend": "JsonKVStorage", "storage_uuid":'
        ' "%s"}\x00' % UUID,
    ],
)
def test_the_wizard_accepts_only_what_the_server_accepts(
    tmp_path: Path, raw: str
) -> None:
    from lightrag.config_anchor import read_anchor
    from lightrag.exceptions import ConfigurationIdentityError

    working_dir = tmp_path / "rag_storage"
    _write_anchor(working_dir, "", raw=raw)
    with pytest.raises(ConfigurationIdentityError):
        read_anchor(str(working_dir))

    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    assert parse_lines(result.stdout)["VALID"] == "yes", result.stderr
    assert "could not be read" in result.stderr
    assert "binds this deployment" not in result.stderr


@pytest.mark.parametrize(
    "raw",
    [
        '{"storage_uuid":"%s","backend":"PGKVStorage","schema_version":1}' % UUID,
        '\n\t{ "backend" : "PGKVStorage" ,\r\n "schema_version" : 1 ,'
        ' "storage_uuid" : "%s" }\n\n' % UUID,
        # A repeated key keeps its LAST value in Python's json, and so here: a
        # first-match reading would report JsonKVStorage.
        '{"schema_version": 1, "backend": "JsonKVStorage", "backend":'
        ' "PGKVStorage", "storage_uuid": "%s"}' % UUID,
    ],
)
def test_the_wizard_reads_the_backend_the_server_reads(
    tmp_path: Path, raw: str
) -> None:
    from lightrag.config_anchor import read_anchor

    working_dir = tmp_path / "rag_storage"
    _write_anchor(working_dir, "", raw=raw)
    assert read_anchor(str(working_dir)).backend == "PGKVStorage"

    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    assert parse_lines(result.stdout)["VALID"] == "no"
    assert "binds this deployment to PGKVStorage" in result.stderr


def test_an_interpolated_working_dir_is_reported_as_unchecked(
    tmp_path: Path,
) -> None:
    """python-dotenv expands ``${HOME}``; the wizard does not, so it must not
    look below a literal ``<repo>/${HOME}`` and call what it finds there (or
    fails to find) the deployment's anchor."""
    _write_anchor(tmp_path / "${HOME}" / "rag_storage", "PGKVStorage")
    result = _validate(
        tmp_path,
        ["LIGHTRAG_KV_STORAGE=JsonKVStorage", "WORKING_DIR=${HOME}/rag_storage"],
    )
    assert parse_lines(result.stdout)["VALID"] == "yes", result.stderr
    assert "interpolation" in result.stderr
    assert "binds this deployment" not in result.stderr

    report = _run(
        tmp_path,
        "ENV_VALUES[WORKING_DIR]='${HOME}/rag_storage'\n"
        'report_config_anchor "JsonKVStorage"',
    )
    assert "interpolation" in report.stdout
    assert "REFUSE" not in report.stdout


@pytest.mark.parametrize(
    "finalizer",
    ["finalize_storage_setup", "finalize_base_setup", "finalize_server_setup"],
)
def test_every_setup_flow_reads_the_anchor_of_the_runtime_it_is_switching_to(
    tmp_path: Path, finalizer: str
) -> None:
    """A host .env that ``env-storage``, ``env-base`` or ``env-server``
    switches to Compose: the next server reads the Compose mount, so that is
    the anchor to report on -- not the host WORKING_DIR the old .env named."""
    _write_anchor(tmp_path / "rag_storage", "JsonKVStorage")
    _write_anchor(tmp_path / "data" / "rag_storage", "PGKVStorage")
    write_text_lines(tmp_path / "env.example", ["LLM_BINDING=openai"])
    result = _run(
        tmp_path,
        """
ENV_VALUES[LIGHTRAG_KV_STORAGE]=JsonKVStorage
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]=NanoVectorDBStorage
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]=NetworkXStorage
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]=JsonDocStatusStorage
ENV_VALUES[LIGHTRAG_RUNTIME_TARGET]=host
show_summary() { :; }
confirm_required_yes_no() { return 0; }
resolve_compose_output_action() {
  local -n action_ref="$2" target_ref="$3" hint_ref="$4"
  action_ref="write_env_only"; target_ref="compose"; hint_ref="no"
}
backup_env_file() { :; }
generate_env_file() { :; }
%s
"""
        % finalizer,
    )
    assert result.returncode == 0, result.stderr
    assert "binds it to PGKVStorage" in result.stdout
    assert "server will REFUSE to start" in result.stdout


def test_an_empty_working_dir_is_the_directory_the_server_starts_in(
    tmp_path: Path,
) -> None:
    """``WORKING_DIR=`` is the empty string to the server, and
    ``os.path.abspath("")`` is its start directory -- not ``./rag_storage``."""
    _write_anchor(tmp_path, "PGKVStorage")
    _write_anchor(tmp_path / "rag_storage", "JsonKVStorage")
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage", "WORKING_DIR="])
    assert parse_lines(result.stdout)["VALID"] == "no"
    assert "binds this deployment to PGKVStorage" in result.stderr
    assert f"{tmp_path}/_lightrag_config/storage_anchor.json" in result.stderr


@pytest.mark.parametrize(
    "line",
    [
        "export WORKING_DIR=./actual",
        "WORKING_DIR = ./actual",
        "  WORKING_DIR=./actual",
    ],
)
def test_a_working_dir_in_a_form_the_wizard_does_not_read_is_unchecked(
    tmp_path: Path, line: str
) -> None:
    """python-dotenv accepts these forms, so the server reads the anchor under
    ``./actual``; the wizard does not parse them, and must not fall back to
    ``./rag_storage`` and pass what the server refuses."""
    _write_anchor(tmp_path / "actual", "PGKVStorage")
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage", line])
    assert parse_lines(result.stdout)["VALID"] == "yes", result.stderr
    assert "WORKING_DIR is assigned in a form this wizard does not read" in (
        result.stderr
    )
    assert "anchor was not checked" in result.stderr


def test_a_configuration_backend_the_wizard_does_not_read_is_unchecked(
    tmp_path: Path,
) -> None:
    """The comparison side too: the server would resolve PGKVStorage and
    match the anchor, the wizard would resolve JsonKVStorage and refuse."""
    _write_anchor(tmp_path / "rag_storage", "PGKVStorage")
    result = _validate(
        tmp_path,
        [
            "LIGHTRAG_KV_STORAGE=JsonKVStorage",
            "export LIGHTRAG_CONFIG_STORAGE=PGKVStorage",
        ],
    )
    assert "binds this deployment" not in result.stderr
    assert "LIGHTRAG_CONFIG_STORAGE is assigned in a form" in result.stderr


@pytest.mark.parametrize(
    "line",
    [
        "WORKING_DIR=./actual # local path",
        'WORKING_DIR="./actual" # local path',
        "WORKING_DIR=./actual ",
        "WORKING_DIR= ./actual",
        "WORKING_DIR=./actual\r",
    ],
)
def test_a_working_dir_dotenv_reads_differently_is_unchecked(
    tmp_path: Path, line: str
) -> None:
    """python-dotenv strips the comment and the whitespace and reads
    ``./actual``; the plain parse would keep them and look elsewhere."""
    _write_anchor(tmp_path / "actual", "PGKVStorage")
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage", line])
    assert "WORKING_DIR is assigned in a form this wizard does not read" in (
        result.stderr
    )
    assert "binds this deployment" not in result.stderr


def test_a_hash_inside_an_unquoted_value_is_not_a_comment(tmp_path: Path) -> None:
    _write_anchor(tmp_path / "a#b", "PGKVStorage")
    result = _validate(
        tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage", "WORKING_DIR=./a#b"]
    )
    assert "binds this deployment to PGKVStorage" in result.stderr


def test_an_anchor_behind_a_non_directory_is_not_absent(tmp_path: Path) -> None:
    """``_lightrag_config`` is a file: the server's open() fails with
    ENOTDIR and refuses, so this is not the absence that lets it bootstrap."""
    (tmp_path / "rag_storage").mkdir()
    (tmp_path / "rag_storage" / "_lightrag_config").write_text("x")
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    assert "could not be read" in result.stderr


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root bypasses directory search permission",
)
def test_an_anchor_behind_an_unsearchable_directory_is_not_absent(
    tmp_path: Path,
) -> None:
    anchor = _write_anchor(tmp_path / "rag_storage", "PGKVStorage")
    anchor.parent.chmod(0o600)
    try:
        result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    finally:
        anchor.parent.chmod(0o700)
    assert "could not be read" in result.stderr


def test_a_missing_anchor_below_a_searchable_directory_is_absent(
    tmp_path: Path,
) -> None:
    (tmp_path / "rag_storage").mkdir()
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    assert parse_lines(result.stdout)["VALID"] == "yes", result.stderr
    assert "anchor" not in result.stderr


@pytest.mark.parametrize(
    "line",
    [
        'WORKING_DIR="./a\\tactual"',
        'WORKING_DIR="./a\\$actual"',
        "WORKING_DIR='./a\\'actual'",
    ],
)
def test_a_working_dir_with_an_escape_dotenv_decodes_differently_is_unchecked(
    tmp_path: Path, line: str
) -> None:
    """python-dotenv decodes ``\\t`` and ``\\'`` and keeps ``\\$``; the plain
    parse does the opposite, so its path is not the server's."""
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage", line])
    assert "WORKING_DIR is assigned in a form this wizard does not read" in (
        result.stderr
    )


def test_escapes_both_parsers_decode_alike_are_read(tmp_path: Path) -> None:
    _write_anchor(tmp_path / 'a"b', "PGKVStorage")
    result = _validate(
        tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage", 'WORKING_DIR="./a\\"b"']
    )
    assert "binds this deployment to PGKVStorage" in result.stderr


def test_an_anchor_behind_a_symlink_loop_is_not_absent(tmp_path: Path) -> None:
    """ELOOP makes the server refuse; a broken-looking link is not absence."""
    (tmp_path / "rag_storage").symlink_to(tmp_path / "rag_storage")
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    assert "could not be read" in result.stderr


def _compose(case_dir: Path, mount: str) -> None:
    write_text_lines(
        case_dir / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: lightrag",
            "    volumes:",
            f"      - {mount}",
            "      - ./data/inputs:/app/data/inputs",
            "    environment:",
            '      WORKING_DIR: "/app/data/rag_storage"',
        ],
    )


COMPOSE_ENV = ["LIGHTRAG_KV_STORAGE=JsonKVStorage", "LIGHTRAG_RUNTIME_TARGET=compose"]


@pytest.mark.parametrize(
    "mount",
    ["{src}:/app/data/rag_storage", '"{src}:/app/data/rag_storage:rw"'],
)
def test_a_customized_compose_mount_is_where_the_anchor_is_read(
    tmp_path: Path, mount: str
) -> None:
    """The generator keeps a customized mount, so the anchor the Compose
    server reads is under ITS source -- not ./data/rag_storage."""
    source = tmp_path / "srv"
    _write_anchor(source, "PGKVStorage")
    _write_anchor(tmp_path / "data" / "rag_storage", "JsonKVStorage")
    _compose(tmp_path, mount.format(src=source))
    result = _validate(tmp_path, COMPOSE_ENV)
    assert parse_lines(result.stdout)["VALID"] == "no"
    assert f"{source}/_lightrag_config/storage_anchor.json" in result.stderr


@pytest.mark.parametrize(
    "mount",
    ["lightrag_data:/app/data/rag_storage", "${DATA}:/app/data/rag_storage"],
)
def test_a_compose_mount_the_wizard_cannot_resolve_is_unchecked(
    tmp_path: Path, mount: str
) -> None:
    _write_anchor(tmp_path / "data" / "rag_storage", "PGKVStorage")
    _compose(tmp_path, mount)
    result = _validate(tmp_path, COMPOSE_ENV)
    assert "does not resolve to a host path" in result.stderr
    assert "binds this deployment" not in result.stderr


def test_a_compose_file_without_the_mount_is_unchecked(tmp_path: Path) -> None:
    _write_anchor(tmp_path / "data" / "rag_storage", "PGKVStorage")
    _compose(tmp_path, "./data/inputs2:/app/data/inputs2")
    result = _validate(tmp_path, COMPOSE_ENV)
    assert "no short-form lightrag bind mount" in result.stderr


def test_compose_migration_advice_names_the_mounted_working_dir(
    tmp_path: Path,
) -> None:
    """Run on the host, the tool resolves WORKING_DIR from the host .env,
    which a Compose deployment does not point at its mount."""
    _write_anchor(tmp_path / "data" / "rag_storage", "PGKVStorage")
    result = _validate(tmp_path, COMPOSE_ENV)
    expected = (
        f"WORKING_DIR={tmp_path}/data/rag_storage lightrag-migrate-config"
        " --target-backend JsonKVStorage"
    )
    assert expected in result.stderr
    report = _run(
        tmp_path,
        "ENV_VALUES[LIGHTRAG_RUNTIME_TARGET]=compose\n"
        'report_config_anchor "JsonKVStorage"',
    )
    assert expected in report.stdout


def test_host_migration_advice_needs_no_working_dir(tmp_path: Path) -> None:
    _write_anchor(tmp_path / "rag_storage", "PGKVStorage")
    result = _validate(tmp_path, ["LIGHTRAG_KV_STORAGE=JsonKVStorage"])
    assert "'lightrag-migrate-config --target-backend JsonKVStorage'" in (result.stderr)
