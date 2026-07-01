"""Offline tests for scripts/setup/apple-container.sh.

These exercise only the pure, daemon-free parts of the orchestrator: argument
parsing, the generated env-file mapping, and the image tags. Anything that needs
a live `container` daemon (it cannot run on Linux CI) is left to manual testing
on a macOS 26 Apple Silicon box.
"""

from __future__ import annotations

import pytest

from tests.setup._helpers import REPO_ROOT, parse_lines, run_bash, run_bash_process

pytestmark = pytest.mark.offline

SCRIPT = REPO_ROOT / "scripts" / "setup" / "apple-container.sh"


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111, "apple-container.sh should be executable"


def test_script_syntax_is_valid() -> None:
    result = run_bash_process(f'bash -n "{SCRIPT}"')
    assert result.returncode == 0, result.stderr


def test_help_lists_commands_and_services() -> None:
    out = run_bash(f'bash "{SCRIPT}" help')
    for command in ("up", "down", "status", "logs", "restart", "pull"):
        assert command in out
    for service in (
        "postgres",
        "neo4j",
        "milvus",
        "milvus-etcd",
        "milvus-minio",
        "lightrag",
    ):
        assert service in out


def test_unknown_command_fails() -> None:
    result = run_bash_process(f'bash "{SCRIPT}" bogus-command')
    assert result.returncode != 0
    assert "Unknown command" in result.stderr


def test_unknown_option_fails() -> None:
    # Flags are parsed before dispatch, so this stays daemon-free.
    result = run_bash_process(f'bash "{SCRIPT}" up --bogus-flag')
    assert result.returncode != 0
    assert "Unknown option" in result.stderr


def test_generated_env_file_maps_storage_to_in_network_ips(tmp_path) -> None:
    src = tmp_path / "src.env"
    src.write_text(
        "LLM_BINDING=openai\n"
        "OPENAI_API_KEY=sk-test\n"
        "EMBEDDING_DIM=3072\n"
        "LIGHTRAG_KV_STORAGE=JsonKVStorage\n"  # must be overridden
        "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage\n"
    )
    src_before = src.read_text()
    gen = tmp_path / "out.env"
    snippet = f"""
        source "{SCRIPT}"
        ENV_SOURCE="{src}"
        ENV_GENERATED="{gen}"
        PG_ADDR=10.0.0.1
        NEO4J_ADDR=10.0.0.2
        MILVUS_ADDR=10.0.0.3
        generate_env_file
    """
    run_bash(snippet)

    # The user's own .env must never be mutated.
    assert src.read_text() == src_before

    values = parse_lines(gen.read_text())

    # Storage backends forced to the external services.
    assert values["LIGHTRAG_KV_STORAGE"] == "PGKVStorage"
    assert values["LIGHTRAG_DOC_STATUS_STORAGE"] == "PGDocStatusStorage"
    assert values["LIGHTRAG_GRAPH_STORAGE"] == "Neo4JStorage"
    assert values["LIGHTRAG_VECTOR_STORAGE"] == "MilvusVectorDBStorage"

    # Connection endpoints wired to the discovered container IPs.
    assert values["POSTGRES_HOST"] == "10.0.0.1"
    assert values["NEO4J_URI"] == "neo4j://10.0.0.2:7687"
    assert values["MILVUS_URI"] == "http://10.0.0.3:19530"

    # Server binds inside the container.
    assert values["HOST"] == "0.0.0.0"
    assert values["PORT"] == "9621"

    # Unrelated user settings are inherited unchanged.
    assert values["LLM_BINDING"] == "openai"
    assert values["OPENAI_API_KEY"] == "sk-test"
    assert values["EMBEDDING_DIM"] == "3072"


def test_set_kv_replaces_existing_key(tmp_path) -> None:
    target = tmp_path / "f.env"
    target.write_text("FOO=old\nBAR=keep\n")
    snippet = f"""
        source "{SCRIPT}"
        _set_kv "{target}" FOO new
    """
    run_bash(snippet)
    values = parse_lines(target.read_text())
    assert values["FOO"] == "new"
    assert values["BAR"] == "keep"
    # Exactly one FOO line remains.
    assert target.read_text().count("FOO=") == 1


def test_container_and_volume_names_derive_from_prefix() -> None:
    # A custom prefix must namespace BOTH container and volume names, so two
    # stacks never share storage.
    out = run_bash(
        f'export LIGHTRAG_AC_PREFIX=proj2-; source "{SCRIPT}"; '
        f'echo "container=$(cname postgres)"; echo "volume=$(vname pg)"'
    )
    values = parse_lines(out)
    assert values["container"] == "proj2-postgres"
    assert values["volume"] == "proj2_pg"


def test_reads_db_credentials_and_milvus_db_from_env_source(tmp_path) -> None:
    # Credentials and MILVUS_DB_NAME are read from the source .env so the created
    # database matches what the lightrag container connects with.
    envf = tmp_path / "custom.env"
    envf.write_text(
        "POSTGRES_PASSWORD='s3cret'\nNEO4J_PASSWORD=graphpass\nMILVUS_DB_NAME=mydb\n"
    )
    out = run_bash(
        "unset POSTGRES_PASSWORD NEO4J_PASSWORD MILVUS_DB_NAME; "
        f'export LIGHTRAG_AC_ENV_FILE="{envf}"; source "{SCRIPT}"; '
        f'echo "pg=$PG_PASSWORD"; echo "neo=$NEO4J_PASS"; echo "milvus=$MILVUS_DB"'
    )
    values = parse_lines(out)
    assert values["pg"] == "s3cret"  # surrounding quotes stripped
    assert values["neo"] == "graphpass"
    assert values["milvus"] == "mydb"


def test_credential_and_milvus_defaults_when_env_source_lacks_values(tmp_path) -> None:
    envf = tmp_path / "minimal.env"
    envf.write_text("LLM_BINDING=openai\n")
    out = run_bash(
        "unset POSTGRES_PASSWORD MILVUS_DB_NAME; "
        f'export LIGHTRAG_AC_ENV_FILE="{envf}"; source "{SCRIPT}"; '
        f'echo "pg=$PG_PASSWORD"; echo "milvus=$MILVUS_DB"'
    )
    values = parse_lines(out)
    assert values["pg"] == "rag"
    assert values["milvus"] == "lightrag"


def test_image_tags_stay_in_sync_with_repo() -> None:
    """The script must mirror the repo's image tags so the two do not drift.

    Postgres deliberately differs: the setup template's AGE image is amd64-only,
    so the arm64 stack uses pgvector/pgvector:pg18 (which docker-compose-full.yml
    also uses).
    """
    script_text = SCRIPT.read_text()
    milvus_tpl = (
        REPO_ROOT / "scripts" / "setup" / "templates" / "milvus.yml"
    ).read_text()

    # Tags shared verbatim with scripts/setup/templates/milvus.yml.
    assert "milvusdb/milvus:v2.6.11" in script_text
    assert "milvusdb/milvus:v2.6.11" in milvus_tpl
    assert "quay.io/coreos/etcd:v3.5.25" in script_text
    assert "minio/minio:RELEASE.2025-09-07T16-13-09Z" in script_text
    assert "neo4j:5-community" in script_text

    # Never the GPU Milvus tag (amd64 + CUDA, unusable on Apple Silicon).
    assert "milvus:v2.6.11-gpu" not in script_text

    # Postgres uses the multi-arch pgvector image, not the amd64-only AGE image
    # (the AGE image may still be named in comments explaining the deviation).
    assert 'IMG_PG="${LIGHTRAG_AC_IMG_PG:-pgvector/pgvector:pg18}"' in script_text
