"""Regression tests for interactive setup host vs. compose configuration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_bash(script: str, cwd: Path | None = None) -> str:
    """Run a bash snippet and return stdout."""

    result = subprocess.run(
        ["bash", "--norc", "--noprofile", "-c", script],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"bash script failed with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def parse_lines(output: str) -> dict[str, str]:
    """Parse KEY=value lines into a dictionary."""

    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def run_bash_lines(script: str, cwd: Path | None = None) -> dict[str, str]:
    """Run a bash snippet and parse KEY=value lines from stdout."""

    return parse_lines(run_bash(script, cwd=cwd))


def write_text_lines(path: Path, lines: list[str]) -> Path:
    """Write lines to a fixture file with a trailing newline."""

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_collect_postgres_config_keeps_host_reachable_env_values() -> None:
    """Bundled PostgreSQL should keep `.env` host-oriented and use compose overrides."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

confirm_default_yes() {{ return 0; }}
prompt_with_default() {{
  case "$1" in
    "PostgreSQL host") printf 'localhost' ;;
    "PostgreSQL user") printf 'lightrag' ;;
    "PostgreSQL database") printf 'lightrag' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_until_valid() {{
  if [[ "$1" == "PostgreSQL host port" ]]; then
    printf '15432'
  else
    printf '%s' "$2"
  fi
}}
mask_sensitive_input() {{ printf 'supersecret'; }}

collect_postgres_config yes

printf 'POSTGRES_HOST=%s\\n' "${{ENV_VALUES[POSTGRES_HOST]}}"
printf 'POSTGRES_PORT=%s\\n' "${{ENV_VALUES[POSTGRES_PORT]}}"
printf 'POSTGRES_HOST_PORT=%s\\n' "${{ENV_VALUES[POSTGRES_HOST_PORT]}}"
printf 'COMPOSE_POSTGRES_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[POSTGRES_HOST]}}"
printf 'COMPOSE_POSTGRES_PORT=%s\\n' "${{COMPOSE_ENV_OVERRIDES[POSTGRES_PORT]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}"
"""
    )
    values = parse_lines(output)

    assert values["POSTGRES_HOST"] == "localhost"
    assert values["POSTGRES_PORT"] == "15432"
    assert values["POSTGRES_HOST_PORT"] == "15432"
    assert values["COMPOSE_POSTGRES_HOST"] == "postgres"
    assert values["COMPOSE_POSTGRES_PORT"] == "5432"
    assert values["DOCKER_SERVICE"] == "postgres"


@pytest.mark.parametrize(
    ("setup_lines", "collector_call", "env_key", "expected_value"),
    [
        (
            [
                'ENV_VALUES[POSTGRES_HOST]="db.example.com"',
                'ENV_VALUES[POSTGRES_PORT]="6543"',
            ],
            "collect_postgres_config yes",
            "POSTGRES_HOST",
            "localhost",
        ),
        (
            [
                'ENV_VALUES[POSTGRES_HOST]="db.example.com"',
                'ENV_VALUES[POSTGRES_PORT]="6543"',
            ],
            "collect_postgres_config yes",
            "POSTGRES_PORT",
            "6543",
        ),
        (
            ['ENV_VALUES[NEO4J_URI]="neo4j+s://graph.example.com"'],
            "collect_neo4j_config yes",
            "NEO4J_URI",
            "neo4j://localhost:7687",
        ),
        (
            ['ENV_VALUES[MONGO_URI]="mongodb://mongo.example.com:27018/"'],
            "collect_mongodb_config yes",
            "MONGO_URI",
            "mongodb://localhost:27017/",
        ),
        (
            ['ENV_VALUES[REDIS_URI]="redis://cache.example.com:6380/1"'],
            "collect_redis_config yes",
            "REDIS_URI",
            "redis://localhost:6379/",
        ),
        (
            ['ENV_VALUES[MILVUS_URI]="http://milvus.example.com:19530"'],
            "collect_milvus_config yes",
            "MILVUS_URI",
            "http://localhost:19530",
        ),
        (
            ['ENV_VALUES[QDRANT_URL]="http://qdrant.example.com:6333"'],
            "collect_qdrant_config yes",
            "QDRANT_URL",
            "http://localhost:6333",
        ),
        (
            ['ENV_VALUES[MEMGRAPH_URI]="bolt://memgraph.example.com:7687"'],
            "collect_memgraph_config yes",
            "MEMGRAPH_URI",
            "bolt://localhost:7687",
        ),
        (
            ['ENV_VALUES[NEO4J_URI]="neo4j://localhost:7777"'],
            "collect_neo4j_config yes",
            "NEO4J_URI",
            "neo4j://localhost:7687",
        ),
        (
            ['ENV_VALUES[MILVUS_URI]="http://localhost:29530"'],
            "collect_milvus_config yes",
            "MILVUS_URI",
            "http://localhost:19530",
        ),
        (
            ['ENV_VALUES[QDRANT_URL]="http://localhost:16333"'],
            "collect_qdrant_config yes",
            "QDRANT_URL",
            "http://localhost:6333",
        ),
        (
            ['ENV_VALUES[MEMGRAPH_URI]="bolt://localhost:17687"'],
            "collect_memgraph_config yes",
            "MEMGRAPH_URI",
            "bolt://localhost:7687",
        ),
    ],
    ids=[
        "postgres-remote-host",
        "postgres-port-preserved",
        "neo4j-remote-uri",
        "mongodb-remote-uri",
        "redis-remote-uri",
        "milvus-remote-uri",
        "qdrant-remote-uri",
        "memgraph-remote-uri",
        "neo4j-local-port",
        "milvus-local-port",
        "qdrant-local-port",
        "memgraph-local-port",
    ],
)
def test_collect_local_service_configs_normalize_stale_values(
    setup_lines: list[str],
    collector_call: str,
    env_key: str,
    expected_value: str,
) -> None:
    """Bundled services should normalize stale remote or localhost endpoints on rerun."""

    setup_block = "\n".join(setup_lines)
    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

{setup_block}

confirm_default_yes() {{ return 0; }}
prompt_with_default() {{
  case "$1" in
    "PostgreSQL user") printf 'lightrag' ;;
    "PostgreSQL database") printf 'lightrag' ;;
    "Neo4j database") printf 'neo4j' ;;
    "MongoDB database") printf 'LightRAG' ;;
    "Milvus database name") printf 'lightrag' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}

{collector_call}

printf '{env_key}=%s\\n' "${{ENV_VALUES[{env_key}]}}"
"""
    )

    assert values[env_key] == expected_value


def test_prepare_compose_runtime_overrides_keeps_env_unchanged() -> None:
    """Loopback endpoints should be rewritten only for compose overrides."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LLM_BINDING_HOST]="http://localhost:11434"
ENV_VALUES[EMBEDDING_BINDING_HOST]="http://127.0.0.1:11434"
ENV_VALUES[RERANK_BINDING_HOST]="http://localhost:8000/rerank"

prepare_compose_runtime_overrides

printf 'ENV_LLM=%s\\n' "${{ENV_VALUES[LLM_BINDING_HOST]}}"
printf 'ENV_EMBEDDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
printf 'ENV_RERANK=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
printf 'COMPOSE_LLM=%s\\n' "${{COMPOSE_ENV_OVERRIDES[LLM_BINDING_HOST]}}"
printf 'COMPOSE_EMBEDDING=%s\\n' "${{COMPOSE_ENV_OVERRIDES[EMBEDDING_BINDING_HOST]}}"
printf 'COMPOSE_RERANK=%s\\n' "${{COMPOSE_ENV_OVERRIDES[RERANK_BINDING_HOST]}}"
"""
    )
    values = parse_lines(output)

    assert values["ENV_LLM"] == "http://localhost:11434"
    assert values["ENV_EMBEDDING"] == "http://127.0.0.1:11434"
    assert values["ENV_RERANK"] == "http://localhost:8000/rerank"
    assert values["COMPOSE_LLM"] == "http://host.docker.internal:11434"
    assert values["COMPOSE_EMBEDDING"] == "http://host.docker.internal:11434"
    assert values["COMPOSE_RERANK"] == "http://host.docker.internal:8000/rerank"


def test_generate_files_keep_host_env_values_and_inject_compose_overrides(
    tmp_path: Path,
) -> None:
    """This generation path keeps host-style values in `.env` and injects compose-only overrides separately."""

    env_example = tmp_path / "env.example"
    env_example.write_text(
        "\n".join(
            [
                "SSL_CERTFILE=/placeholder/cert.pem",
                "SSL_KEYFILE=/placeholder/key.pem",
                "LLM_BINDING_HOST=https://api.example.com/v1",
                "EMBEDDING_BINDING_HOST=https://api.example.com/v1",
                "RERANK_BINDING_HOST=https://api.example.com/v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    env_file:",
                "      - .env",
                "    volumes:",
                "      - ./.env:/app/.env",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cert_path = tmp_path / "cert.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path = tmp_path / "key.pem"
    key_path.write_text("key", encoding="utf-8")

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[SSL_CERTFILE]="{cert_path}"
ENV_VALUES[SSL_KEYFILE]="{key_path}"
ENV_VALUES[LLM_BINDING_HOST]="http://localhost:11434"
ENV_VALUES[EMBEDDING_BINDING_HOST]="http://127.0.0.1:11434"
ENV_VALUES[RERANK_BINDING_HOST]="http://localhost:8000/rerank"
SSL_CERT_SOURCE_PATH="{cert_path}"
SSL_KEY_SOURCE_PATH="{key_path}"

prepare_compose_env_overrides
stage_ssl_assets "$SSL_CERT_SOURCE_PATH" "$SSL_KEY_SOURCE_PATH"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )

    assert f"SSL_CERTFILE={cert_path}" in generated_env
    assert f"SSL_KEYFILE={key_path}" in generated_env
    assert "LLM_BINDING_HOST=http://localhost:11434" in generated_env
    assert "EMBEDDING_BINDING_HOST=http://127.0.0.1:11434" in generated_env
    assert "RERANK_BINDING_HOST=http://localhost:8000/rerank" in generated_env

    assert 'SSL_CERTFILE: "/app/data/certs/cert.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key.pem"' in generated_compose
    assert 'LLM_BINDING_HOST: "http://host.docker.internal:11434"' in generated_compose
    assert (
        'EMBEDDING_BINDING_HOST: "http://host.docker.internal:11434"'
        in generated_compose
    )
    assert (
        'RERANK_BINDING_HOST: "http://host.docker.internal:8000/rerank"'
        in generated_compose
    )
    assert "./data/certs/cert.pem:/app/data/certs/cert.pem:ro" in generated_compose
    assert "./data/certs/key.pem:/app/data/certs/key.pem:ro" in generated_compose
    assert "env_file:" not in generated_compose


def test_generate_docker_compose_removes_lightrag_env_file_to_preserve_dollar_values(
    tmp_path: Path,
) -> None:
    """Generated compose should remove `env_file` and skip empty environment blocks."""

    write_text_lines(
        tmp_path / "docker-compose.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    env_file:",
            "      - .env",
            "    volumes:",
            "      - ./.env:/app/.env",
        ],
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )

    assert "env_file:" not in generated_compose
    assert "environment:" not in generated_compose
    assert "- ./.env:/app/.env" in generated_compose


def test_generate_docker_compose_preserves_list_style_lightrag_environment(
    tmp_path: Path,
) -> None:
    """Compose regeneration should not mix mapping entries into list-style environments."""

    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    environment:",
            "      - PORT=9621",
            "      - FOO=bar",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
set_compose_override "PORT" "1234"
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )

    assert '      - "PORT=1234"' in generated_compose
    assert "      - FOO=bar" in generated_compose
    assert "      PORT:" not in generated_compose


def test_existing_ssl_env_keeps_compose_mount_overrides(tmp_path: Path) -> None:
    """Compose regeneration should preserve working SSL mounts without implying `.env` is permanently dual-purpose."""

    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    env_file:",
                "      - .env",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cert_path = tmp_path / "cert.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path = tmp_path / "key.pem"
    key_path.write_text("key", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SSL=true",
                f"SSL_CERTFILE={cert_path}",
                f"SSL_KEYFILE={key_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
prepare_compose_env_overrides
stage_ssl_assets "$SSL_CERT_SOURCE_PATH" "$SSL_KEY_SOURCE_PATH"
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )

    assert 'SSL_CERTFILE: "/app/data/certs/cert.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key.pem"' in generated_compose
    assert "./data/certs/cert.pem:/app/data/certs/cert.pem:ro" in generated_compose
    assert "./data/certs/key.pem:/app/data/certs/key.pem:ro" in generated_compose


def test_finalize_base_setup_rewrites_ssl_env_to_preserved_compose_paths(
    tmp_path: Path,
) -> None:
    """Compose-target reruns should rewrite broken SSL source paths to preserved staged compose paths."""

    staged_dir = tmp_path / "data" / "certs"
    staged_dir.mkdir(parents=True)
    (staged_dir / "server.pem").write_text("cert", encoding="utf-8")
    (staged_dir / "server.key").write_text("key", encoding="utf-8")

    write_text_lines(
        tmp_path / ".env",
        [
            "SSL=true",
            "SSL_CERTFILE=/missing/original-cert.pem",
            "SSL_KEYFILE=/missing/original-key.pem",
            "LIGHTRAG_KV_STORAGE=JsonKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    volumes:",
            "      - ./.env:/app/.env",
            "      - ./data/certs/server.pem:/app/data/certs/server.pem:ro",
            "      - ./data/certs/server.key:/app/data/certs/server.key:ro",
            "    environment:",
            "      SSL_CERTFILE: /app/data/certs/server.pem",
            "      SSL_KEYFILE: /app/data/certs/server.key",
        ],
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
initialize_default_storage_backends

show_summary() {{ :; }}
confirm_default_yes() {{ return 0; }}

finalize_base_setup

if validate_env_file; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )
    values = parse_lines(output)
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "SSL_CERTFILE=/app/data/certs/server.pem" in generated_env
    assert "SSL_KEYFILE=/app/data/certs/server.key" in generated_env
    assert values["VALID"] == "yes"


def test_removing_ssl_strips_wizard_bind_mounts_from_compose(tmp_path: Path) -> None:
    """Re-running setup without SSL must remove only wizard-managed SSL mounts."""

    # A previously generated compose file that has SSL mounts.
    compose_file = tmp_path / "docker-compose.final.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    volumes:",
                '      - "./data/certs/cert.pem:/app/data/certs/cert.pem:ro"',
                '      - "./data/certs/key.pem:/app/data/certs/key.pem:ro"',
                '      - "./data/rag_storage:/app/data/rag_storage"',
                '      - "./data/inputs:/app/data/inputs"',
                '      - "./custom-data:/app/data/custom"',
                "    environment:",
                '      SSL_CERTFILE: "/app/data/certs/cert.pem"',
                '      SSL_KEYFILE: "/app/data/certs/key.pem"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "env.example").write_text(
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Re-run without SSL: COMPOSE_ENV_OVERRIDES has no SSL_CERTFILE/SSL_KEYFILE.
    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
generate_docker_compose "{tmp_path}/docker-compose.final.yml"
"""
    )

    result = compose_file.read_text(encoding="utf-8")

    # SSL bind mounts must be gone.
    assert "/app/data/certs/cert.pem" not in result
    assert "/app/data/certs/key.pem" not in result
    # Default persistent mounts and user-added non-wizard mounts must be preserved.
    assert "./data/rag_storage:/app/data/rag_storage" in result
    assert "./data/inputs:/app/data/inputs" in result
    assert "./custom-data:/app/data/custom" in result


def test_generate_docker_compose_preserves_non_managed_named_volumes(
    tmp_path: Path,
) -> None:
    """Retained services should keep their referenced top-level named volumes."""

    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    volumes:",
            "      - my_cache:/app/cache",
            "  sidecar:",
            "    image: busybox",
            '    command: ["sleep", "infinity"]',
            "    volumes:",
            "      - sidecar_data:/data",
            "  postgres:",
            "    image: old/postgres:image",
            "    volumes:",
            "      - postgres_data:/var/lib/postgresql/data",
            "volumes:",
            "  my_cache:",
            "    driver: local",
            "  sidecar_data:",
            "    driver: local",
            "  postgres_data:",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml"
"""
    )

    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")

    assert "  sidecar:" in result
    assert "my_cache:/app/cache" in result
    assert "sidecar_data:/data" in result
    assert "  my_cache:" in result
    assert "    driver: local" in result
    assert "  sidecar_data:" in result
    assert "postgres_data:" not in result


def test_generate_docker_compose_inserts_managed_services_before_top_level_sections(
    tmp_path: Path,
) -> None:
    """Managed services should stay inside services: even when custom top-level sections exist."""

    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    volumes:",
            "      - ./.env:/app/.env",
            "  worker:",
            "    image: example/worker:test",
            "    networks:",
            "      - appnet",
            "networks:",
            "  appnet:",
            "    driver: bridge",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[POSTGRES_USER]="lightrag"
ENV_VALUES[POSTGRES_PASSWORD]="secret"
ENV_VALUES[POSTGRES_DATABASE]="lightrag"
add_docker_service "postgres"

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml"
"""
    )

    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")

    assert "  postgres:" in result
    assert "\nnetworks:\n" in result
    assert result.index("\n  postgres:") < result.index("\nnetworks:\n")
    assert "  appnet:" in result


def test_find_generated_compose_file_prefers_legacy_profile_match(
    tmp_path: Path,
) -> None:
    """Legacy setup profile metadata should steer compose migration when available."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_SETUP_PROFILE=production",
            "HOST=0.0.0.0",
        ],
    )
    write_text_lines(
        tmp_path / "docker-compose.development.yml",
        [
            "services:",
            "  lightrag:",
            "    image: dev/lightrag",
        ],
    )
    write_text_lines(
        tmp_path / "docker-compose.production.yml",
        [
            "services:",
            "  lightrag:",
            "    image: prod/lightrag",
        ],
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

printf 'COMPOSE=%s\\n' "$(find_generated_compose_file)"
"""
    )
    values = parse_lines(output)

    assert values["COMPOSE"] == str(tmp_path / "docker-compose.production.yml")


def test_find_generated_compose_file_falls_back_to_order_without_profile(
    tmp_path: Path,
) -> None:
    """Without legacy profile metadata, compose migration should use the default order."""

    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0"])
    write_text_lines(
        tmp_path / "docker-compose.development.yml",
        [
            "services:",
            "  lightrag:",
            "    image: dev/lightrag",
        ],
    )
    write_text_lines(
        tmp_path / "docker-compose.production.yml",
        [
            "services:",
            "  lightrag:",
            "    image: prod/lightrag",
        ],
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

printf 'COMPOSE=%s\\n' "$(find_generated_compose_file)"
"""
    )
    values = parse_lines(output)

    assert values["COMPOSE"] == str(tmp_path / "docker-compose.development.yml")


def test_collect_ssl_config_can_disable_loaded_ssl_values(tmp_path: Path) -> None:
    """Declining SSL should clear previously loaded cert paths and staged sources."""

    cert_path = tmp_path / "cert.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path = tmp_path / "key.pem"
    key_path.write_text("key", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SSL=true",
                f"SSL_CERTFILE={cert_path}",
                f"SSL_KEYFILE={key_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

confirm_default_yes() {{ return 1; }}

collect_ssl_config

printf 'SSL_IS_SET=%s\\n' "${{ENV_VALUES[SSL]+set}}"
printf 'SSL_CERTFILE_IS_SET=%s\\n' "${{ENV_VALUES[SSL_CERTFILE]+set}}"
printf 'SSL_KEYFILE_IS_SET=%s\\n' "${{ENV_VALUES[SSL_KEYFILE]+set}}"
printf 'SSL_CERT_SOURCE_PATH=%s\\n' "$SSL_CERT_SOURCE_PATH"
printf 'SSL_KEY_SOURCE_PATH=%s\\n' "$SSL_KEY_SOURCE_PATH"
"""
    )
    values = parse_lines(output)

    assert values["SSL_IS_SET"] == ""
    assert values["SSL_CERTFILE_IS_SET"] == ""
    assert values["SSL_KEYFILE_IS_SET"] == ""
    assert values["SSL_CERT_SOURCE_PATH"] == ""
    assert values["SSL_KEY_SOURCE_PATH"] == ""


def test_validate_env_file_rejects_missing_ssl_files(tmp_path: Path) -> None:
    """Validation should fail when SSL is enabled with missing cert/key paths."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SSL=true",
                "SSL_CERTFILE=/missing/cert.pem",
                "SSL_KEYFILE=/missing/key.pem",
                "LIGHTRAG_KV_STORAGE=JsonKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
validate_env_file
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Invalid SSL_CERTFILE" in result.stderr
    assert "Invalid SSL_KEYFILE" in result.stderr


def test_generate_env_file_comments_out_later_duplicate_active_keys(
    tmp_path: Path,
) -> None:
    """Commented example keys should not be overridden by later active defaults."""

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[EMBEDDING_BINDING]="ollama"
ENV_VALUES[EMBEDDING_MODEL]="bge-m3:latest"
ENV_VALUES[EMBEDDING_DIM]="1024"
ENV_VALUES[EMBEDDING_BINDING_HOST]="http://localhost:11434"

generate_env_file "{REPO_ROOT}/env.example" "$REPO_ROOT/.env"
"""
    )

    generated_env = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    active_embedding_lines = [
        line for line in generated_env if line.startswith("EMBEDDING_BINDING=")
    ]
    active_model_lines = [
        line for line in generated_env if line.startswith("EMBEDDING_MODEL=")
    ]
    active_host_lines = [
        line for line in generated_env if line.startswith("EMBEDDING_BINDING_HOST=")
    ]

    assert active_embedding_lines == ["EMBEDDING_BINDING=ollama"]
    assert active_model_lines == ["EMBEDDING_MODEL=bge-m3:latest"]
    assert active_host_lines == ["EMBEDDING_BINDING_HOST=http://localhost:11434"]
    assert "# EMBEDDING_BINDING=openai" in generated_env


def test_generate_env_file_round_trips_dollar_signs_in_quoted_values(
    tmp_path: Path,
) -> None:
    """Quoted values containing `$` should survive generate/load cycles unchanged."""

    env_example = tmp_path / "env.example"
    env_example.write_text(
        "\n".join(
            [
                "TOKEN_SECRET=placeholder",
                "LIGHTRAG_API_KEY=placeholder",
                "WEBUI_DESCRIPTION=placeholder",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[TOKEN_SECRET]='abc$HOME'
ENV_VALUES[LIGHTRAG_API_KEY]='plain$token'
ENV_VALUES[WEBUI_DESCRIPTION]='value with "$PATH" and $HOME'

generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
reset_state
load_env_file "$REPO_ROOT/.env"

printf 'TOKEN_SECRET=%s\\n' "${{ENV_VALUES[TOKEN_SECRET]}}"
printf 'LIGHTRAG_API_KEY=%s\\n' "${{ENV_VALUES[LIGHTRAG_API_KEY]}}"
printf 'WEBUI_DESCRIPTION=%s\\n' "${{ENV_VALUES[WEBUI_DESCRIPTION]}}"
"""
    )
    values = parse_lines(output)
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")

    assert 'TOKEN_SECRET="abc$HOME"' in generated_env
    assert 'LIGHTRAG_API_KEY="plain$token"' in generated_env
    assert 'WEBUI_DESCRIPTION="value with \\"$PATH\\" and $HOME"' in generated_env
    assert values["TOKEN_SECRET"] == "abc$HOME"
    assert values["LIGHTRAG_API_KEY"] == "plain$token"
    assert values["WEBUI_DESCRIPTION"] == 'value with "$PATH" and $HOME'


def test_validate_sensitive_env_literals_rejects_interpolation_syntax() -> None:
    """Sensitive values should reject `${...}` so default dotenv interpolation stays safe."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[TOKEN_SECRET]='${{JWT_SECRET}}'
ENV_VALUES[LIGHTRAG_API_KEY]='plain$token'
ENV_VALUES[WEBUI_DESCRIPTION]='${{ALLOWED_MACRO}}'

if validate_sensitive_env_literals; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["VALID"] == "no"


@pytest.mark.parametrize(
    ("collector_name", "binding_prefix", "env_lines"),
    [
        (
            "collect_llm_config",
            "LLM",
            [
                "LLM_BINDING=openai",
                "LLM_MODEL=gpt-4o",
                "LLM_BINDING_HOST=https://api.openai.com/v1",
                "LLM_BINDING_API_KEY=${OPENAI_API_KEY}",
            ],
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            [
                "EMBEDDING_BINDING=openai",
                "EMBEDDING_MODEL=text-embedding-3-large",
                "EMBEDDING_DIM=3072",
                "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
                "EMBEDDING_BINDING_API_KEY=${OPENAI_API_KEY}",
            ],
        ),
    ],
    ids=["llm-bedrock-clears-api-key", "embedding-bedrock-clears-api-key"],
)
def test_collect_provider_config_clears_stale_api_key_for_bedrock(
    tmp_path: Path,
    collector_name: str,
    binding_prefix: str,
    env_lines: list[str],
) -> None:
    """Switching a provider to Bedrock should remove stale API-key settings."""

    write_text_lines(tmp_path / ".env", env_lines)

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

prompt_choice() {{ printf 'aws_bedrock'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_required_secret() {{ printf 'dummy-secret'; }}
mask_sensitive_input() {{ printf ''; }}
confirm_default_yes() {{ return 0; }}

{collector_name}

printf 'BINDING=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING]}}"
printf 'API_KEY_SET=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_API_KEY]+set}}"
if validate_sensitive_env_literals; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )

    assert values["BINDING"] == "aws_bedrock"
    assert values["API_KEY_SET"] == ""
    assert values["VALID"] == "yes"


@pytest.mark.parametrize(
    (
        "collector_name",
        "binding_prefix",
        "provider_choice",
        "secret_stub",
        "expected_binding",
        "expected_model",
        "expected_host",
        "expected_dim",
        "expected_api_key_set",
    ),
    [
        (
            "collect_llm_config",
            "LLM",
            "ollama",
            "",
            "ollama",
            "mistral-nemo:latest",
            "http://localhost:11434",
            "",
            "",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            "jina",
            "prompt_secret_until_valid_with_default() { printf 'jina-secret-key'; }",
            "jina",
            "jina-embeddings-v4",
            "https://api.jina.ai/v1/embeddings",
            "2048",
            "set",
        ),
    ],
    ids=["llm-provider-defaults", "embedding-provider-defaults"],
)
def test_collect_provider_config_uses_provider_specific_defaults(
    collector_name: str,
    binding_prefix: str,
    provider_choice: str,
    secret_stub: str,
    expected_binding: str,
    expected_model: str,
    expected_host: str,
    expected_dim: str,
    expected_api_key_set: str,
) -> None:
    """Fresh provider selection should pick provider-specific defaults."""

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_choice() {{ printf '{provider_choice}'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
{secret_stub}

{collector_name}

printf 'BINDING=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING]}}"
printf 'MODEL=%s\\n' "${{ENV_VALUES[{binding_prefix}_MODEL]}}"
printf 'HOST=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_HOST]}}"
printf 'DIM=%s\\n' "${{ENV_VALUES[{binding_prefix}_DIM]:-}}"
printf 'API_KEY_SET=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_API_KEY]+set}}"
"""
    )

    assert values["BINDING"] == expected_binding
    assert values["MODEL"] == expected_model
    assert values["HOST"] == expected_host
    assert values["DIM"] == expected_dim
    assert values["API_KEY_SET"] == expected_api_key_set


@pytest.mark.parametrize(
    (
        "collector_name",
        "binding_prefix",
        "env_lines",
        "prompt_stubs",
        "expected_binding",
        "expected_model",
        "expected_host",
        "expected_dim",
        "expected_api_key",
    ),
    [
        (
            "collect_llm_config",
            "LLM",
            [
                "LLM_BINDING=openai-ollama",
                "LLM_MODEL=llama3.1:8b",
                "LLM_BINDING_HOST=http://localhost:11434/v1",
                "LLM_BINDING_API_KEY=sk-local-test-key",
            ],
            """
prompt_with_default() { printf '%s' "$2"; }
prompt_secret_until_valid_with_default() { printf '%s' "$2"; }
""",
            "openai-ollama",
            "llama3.1:8b",
            "http://localhost:11434/v1",
            "",
            "sk-local-test-key",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            [
                "EMBEDDING_BINDING=lollms",
                "EMBEDDING_MODEL=lollms_embedding_model",
                "EMBEDDING_DIM=1024",
                "EMBEDDING_BINDING_HOST=http://localhost:9600",
            ],
            """prompt_with_default() { printf '%s' "$2"; }""",
            "lollms",
            "lollms_embedding_model",
            "http://localhost:9600",
            "1024",
            "",
        ),
    ],
    ids=["llm-rerun-preserves-openai-ollama", "embedding-rerun-preserves-lollms"],
)
def test_collect_provider_config_preserves_supported_binding_on_rerun(
    tmp_path: Path,
    collector_name: str,
    binding_prefix: str,
    env_lines: list[str],
    prompt_stubs: str,
    expected_binding: str,
    expected_model: str,
    expected_host: str,
    expected_dim: str,
    expected_api_key: str,
) -> None:
    """Reruns should preserve supported provider bindings and their saved settings."""

    write_text_lines(tmp_path / ".env", env_lines)

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

{prompt_stubs}

{collector_name}

printf 'BINDING=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING]}}"
printf 'MODEL=%s\\n' "${{ENV_VALUES[{binding_prefix}_MODEL]}}"
printf 'HOST=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_HOST]}}"
printf 'DIM=%s\\n' "${{ENV_VALUES[{binding_prefix}_DIM]:-}}"
printf 'API_KEY=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_API_KEY]:-}}"
"""
    )

    assert values["BINDING"] == expected_binding
    assert values["MODEL"] == expected_model
    assert values["HOST"] == expected_host
    assert values["DIM"] == expected_dim
    assert values["API_KEY"] == expected_api_key


def test_collect_embedding_config_forces_ollama_for_openai_ollama_llm(
    tmp_path: Path,
) -> None:
    """`openai-ollama` should not preserve a conflicting embedding provider."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai-ollama",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=text-embedding-3-large",
            "EMBEDDING_DIM=3072",
            "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
            "EMBEDDING_BINDING_API_KEY=local-key",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_embedding_config

printf 'EMBEDDING_BINDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING]}}"
printf 'EMBEDDING_MODEL=%s\\n' "${{ENV_VALUES[EMBEDDING_MODEL]}}"
printf 'EMBEDDING_DIM=%s\\n' "${{ENV_VALUES[EMBEDDING_DIM]}}"
printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
printf 'EMBEDDING_BINDING_API_KEY_SET=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_API_KEY]+set}}"
"""
    )

    assert values["EMBEDDING_BINDING"] == "ollama"
    assert values["EMBEDDING_MODEL"] == "bge-m3:latest"
    assert values["EMBEDDING_DIM"] == "1024"
    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:11434"
    assert values["EMBEDDING_BINDING_API_KEY_SET"] == ""


def test_collect_llm_config_allows_bedrock_ambient_credential_chain() -> None:
    """Bedrock setup should allow IAM roles, AWS profiles, or SSO without saved keys."""

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_choice() {{ printf 'aws_bedrock'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_clearable_with_default() {{ printf ''; }}
prompt_required_secret() {{ return 1; }}
confirm_default_yes() {{ return 1; }}

collect_llm_config

printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
printf 'AWS_ACCESS_KEY_ID_SET=%s\\n' "${{ENV_VALUES[AWS_ACCESS_KEY_ID]+set}}"
printf 'AWS_SECRET_ACCESS_KEY_SET=%s\\n' "${{ENV_VALUES[AWS_SECRET_ACCESS_KEY]+set}}"
printf 'AWS_SESSION_TOKEN_SET=%s\\n' "${{ENV_VALUES[AWS_SESSION_TOKEN]+set}}"
printf 'AWS_REGION_SET=%s\\n' "${{ENV_VALUES[AWS_REGION]+set}}"
"""
    )

    assert values["LLM_BINDING"] == "aws_bedrock"
    assert values["AWS_ACCESS_KEY_ID_SET"] == ""
    assert values["AWS_SECRET_ACCESS_KEY_SET"] == ""
    assert values["AWS_SESSION_TOKEN_SET"] == ""
    assert values["AWS_REGION_SET"] == ""


def test_switching_both_providers_off_bedrock_clears_saved_aws_credentials(
    tmp_path: Path,
) -> None:
    """Reruns should not keep stale AWS Bedrock secrets in regenerated `.env` files."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=aws_bedrock",
            "LLM_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0",
            "LLM_BINDING_HOST=https://bedrock.amazonaws.com",
            "EMBEDDING_BINDING=aws_bedrock",
            "EMBEDDING_MODEL=amazon.titan-embed-text-v2:0",
            "EMBEDDING_DIM=1024",
            "EMBEDDING_BINDING_HOST=https://bedrock.amazonaws.com",
            "AWS_ACCESS_KEY_ID=AKIAOLDKEY",
            "AWS_SECRET_ACCESS_KEY=oldsecretvalue",
            "AWS_SESSION_TOKEN=oldsess",
            "AWS_REGION=us-east-1",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        [
            "# AWS_ACCESS_KEY_ID=your_aws_access_key_id",
            "# AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key",
            "# AWS_SESSION_TOKEN=your_optional_aws_session_token",
            "# AWS_REGION=us-east-1",
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=your_api_key",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=text-embedding-3-large",
            "EMBEDDING_DIM=3072",
            "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
            "EMBEDDING_BINDING_API_KEY=your_api_key",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

prompt_choice() {{ printf 'openai'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf 'fresh-key'; }}

collect_llm_config
collect_embedding_config
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env.generated"

printf 'AWS_ACCESS_KEY_ID_SET=%s\\n' "${{ENV_VALUES[AWS_ACCESS_KEY_ID]+set}}"
printf 'AWS_SECRET_ACCESS_KEY_SET=%s\\n' "${{ENV_VALUES[AWS_SECRET_ACCESS_KEY]+set}}"
printf 'AWS_SESSION_TOKEN_SET=%s\\n' "${{ENV_VALUES[AWS_SESSION_TOKEN]+set}}"
printf 'AWS_REGION_SET=%s\\n' "${{ENV_VALUES[AWS_REGION]+set}}"
"""
    )
    generated_lines = (
        (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()
    )

    assert values["AWS_ACCESS_KEY_ID_SET"] == ""
    assert values["AWS_SECRET_ACCESS_KEY_SET"] == ""
    assert values["AWS_SESSION_TOKEN_SET"] == ""
    assert values["AWS_REGION_SET"] == ""
    assert not any(line.startswith("AWS_ACCESS_KEY_ID=") for line in generated_lines)
    assert not any(
        line.startswith("AWS_SECRET_ACCESS_KEY=") for line in generated_lines
    )
    assert not any(line.startswith("AWS_SESSION_TOKEN=") for line in generated_lines)
    assert not any(line.startswith("AWS_REGION=") for line in generated_lines)


def test_collect_rerank_config_preserves_api_key_when_disabled(
    tmp_path: Path,
) -> None:
    """Disabling reranking should preserve credentials so they survive re-enable."""

    write_text_lines(
        tmp_path / ".env",
        [
            "RERANK_BINDING=cohere",
            "RERANK_MODEL=rerank-v3.5",
            "RERANK_BINDING_HOST=https://api.cohere.com/v1/rerank",
            "RERANK_BINDING_API_KEY=test-api-key-literal",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

confirm_default_yes() {{ return 1; }}

collect_rerank_config

printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}"
printf 'RERANK_BINDING_API_KEY_SET=%s\\n' "${{ENV_VALUES[RERANK_BINDING_API_KEY]+set}}"
if validate_sensitive_env_literals; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )

    assert values["RERANK_BINDING"] == "null"
    assert values["RERANK_BINDING_API_KEY_SET"] == "set"
    assert values["VALID"] == "yes"


@pytest.mark.parametrize(
    ("setup_lines", "prompt_choice_impl", "expected_model", "expected_docker_service"),
    [
        (
            [
                'ENV_VALUES[RERANK_BINDING]="cohere"',
                'ENV_VALUES[RERANK_MODEL]="rerank-v3.5"',
                'ENV_VALUES[RERANK_BINDING_HOST]="https://api.cohere.com/v1/rerank"',
            ],
            """
prompt_choice() {
  case "$1" in
    "Rerank provider") printf 'vllm' ;;
    "vLLM device") printf 'cpu' ;;
    *) printf '%s' "$2" ;;
  esac
}
""",
            "BAAI/bge-reranker-v2-m3",
            "vllm-rerank",
        ),
        (
            [
                'ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]="vllm"',
                'ENV_VALUES[RERANK_BINDING]="cohere"',
                'ENV_VALUES[RERANK_MODEL]="BAAI/bge-reranker-v2-m3"',
                'ENV_VALUES[RERANK_BINDING_HOST]="http://localhost:8000/rerank"',
                'ENV_VALUES[VLLM_RERANK_MODEL]="BAAI/bge-reranker-v2-m3"',
                'ENV_VALUES[VLLM_RERANK_PORT]="8000"',
                'ENV_VALUES[VLLM_RERANK_DEVICE]="cpu"',
            ],
            """prompt_choice() { printf '%s' "$2"; }""",
            "BAAI/bge-reranker-v2-m3",
            "vllm-rerank",
        ),
    ],
    ids=["switch-to-vllm", "rerun-vllm"],
)
def test_collect_rerank_config_uses_vllm_defaults(
    setup_lines: list[str],
    prompt_choice_impl: str,
    expected_model: str,
    expected_docker_service: str,
) -> None:
    """Selecting or reusing local vLLM should converge on the vLLM rerank defaults."""

    setup_block = "\n".join(setup_lines)
    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

{setup_block}

confirm_default_no() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
{prompt_choice_impl}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}

collect_rerank_config

printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}"
printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]}}"
printf 'RERANK_MODEL=%s\\n' "${{ENV_VALUES[RERANK_MODEL]}}"
printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]:-}}"
printf 'COMPOSE_RERANK_BINDING_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[RERANK_BINDING_HOST]}}"
"""
    )

    assert values["RERANK_BINDING"] == "cohere"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == "vllm"
    assert values["RERANK_MODEL"] == expected_model
    assert values["RERANK_BINDING_HOST"] == "http://localhost:8000/rerank"
    assert values["DOCKER_SERVICE"] == expected_docker_service
    assert values["COMPOSE_RERANK_BINDING_HOST"] == "http://vllm-rerank:8000/rerank"


def test_collect_rerank_config_switching_from_vllm_clears_local_defaults() -> None:
    """Switching from local vLLM to hosted rerank should replace stale vLLM values with provider defaults."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]="vllm"
ENV_VALUES[RERANK_BINDING]="cohere"
ENV_VALUES[RERANK_MODEL]="BAAI/bge-reranker-v2-m3"
ENV_VALUES[RERANK_BINDING_HOST]="http://localhost:8000/rerank"

confirm_default_no() {{ return 0; }}
prompt_choice() {{
  case "$1" in
    "Rerank provider") printf 'cohere' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf 'cohere-secret-123'; }}

collect_rerank_config

printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}"
printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]}}"
printf 'RERANK_MODEL=%s\\n' "${{ENV_VALUES[RERANK_MODEL]:-}}"
printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]:-}}"
"""
    )
    values = parse_lines(output)

    assert values["RERANK_BINDING"] == "cohere"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == "cohere"
    # Stale vLLM model should be replaced by the cohere provider default
    assert values["RERANK_MODEL"] != "BAAI/bge-reranker-v2-m3"
    assert values["RERANK_MODEL"] == "rerank-v3.5"
    # Stale vLLM localhost endpoint should be replaced by the cohere provider default
    assert "localhost:8000" not in values["RERANK_BINDING_HOST"]
    assert "cohere" in values["RERANK_BINDING_HOST"]


@pytest.mark.parametrize(
    (
        "setup_lines",
        "confirm_default_yes_impl",
        "prompt_choice_impl",
        "expected_device",
        "expected_cuda_set",
        "expected_nvidia_set",
        "expected_cpu_set",
    ),
    [
        (
            [
                'ENV_VALUES[CUDA_VISIBLE_DEVICES]="-1"',
                'ENV_VALUES[NVIDIA_VISIBLE_DEVICES]="-1"',
                'ENV_VALUES[VLLM_USE_CPU]="1"',
            ],
            """
confirm_default_yes() {
  if [[ "$1" == "Use CPU instead?" ]]; then
    return 1
  fi
  return 0
}
""",
            """
prompt_choice() {
  case "$1" in
    "Rerank provider") printf 'vllm' ;;
    "vLLM device") printf 'cuda' ;;
    *) printf '%s' "$2" ;;
  esac
}
""",
            "cuda",
            "",
            "",
            "",
        ),
        (
            [
                'ENV_VALUES[VLLM_RERANK_DEVICE]="cuda"',
            ],
            "confirm_default_yes() { return 0; }",
            """
prompt_choice() {
  case "$1" in
    "Rerank provider") printf 'vllm' ;;
    "vLLM device") printf 'cpu' ;;
    *) printf '%s' "$2" ;;
  esac
}
""",
            "cpu",
            "",
            "",
            "",
        ),
    ],
    ids=["cuda-clears-disabled-masks", "cpu-clears-gpu-flags"],
)
def test_collect_rerank_config_normalizes_vllm_device_state(
    setup_lines: list[str],
    confirm_default_yes_impl: str,
    prompt_choice_impl: str,
    expected_device: str,
    expected_cuda_set: str,
    expected_nvidia_set: str,
    expected_cpu_set: str,
) -> None:
    """Changing vLLM device modes should normalize the related environment state."""

    setup_block = "\n".join(setup_lines)
    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

{setup_block}

confirm_default_no() {{ return 0; }}
{confirm_default_yes_impl}
{prompt_choice_impl}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}

collect_rerank_config

printf 'VLLM_RERANK_DEVICE=%s\\n' "${{ENV_VALUES[VLLM_RERANK_DEVICE]}}"
printf 'CUDA_VISIBLE_DEVICES_SET=%s\\n' "${{ENV_VALUES[CUDA_VISIBLE_DEVICES]+set}}"
printf 'NVIDIA_VISIBLE_DEVICES_SET=%s\\n' "${{ENV_VALUES[NVIDIA_VISIBLE_DEVICES]+set}}"
printf 'VLLM_USE_CPU_SET=%s\\n' "${{ENV_VALUES[VLLM_USE_CPU]+set}}"
"""
    )

    assert values["VLLM_RERANK_DEVICE"] == expected_device
    assert values["CUDA_VISIBLE_DEVICES_SET"] == expected_cuda_set
    assert values["NVIDIA_VISIBLE_DEVICES_SET"] == expected_nvidia_set
    assert values["VLLM_USE_CPU_SET"] == expected_cpu_set


def test_generate_docker_compose_escapes_dollar_signs_in_overrides_and_service_secrets(
    tmp_path: Path,
) -> None:
    """Compose generation should keep `$` literals in runtime overrides and bundled secrets."""

    write_text_lines(
        tmp_path / "docker-compose.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    env_file:",
            "      - .env",
        ],
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[MONGO_URI]='mongodb://user:p$HOME@localhost:27017/'
ENV_VALUES[POSTGRES_USER]='user$ID'
ENV_VALUES[POSTGRES_PASSWORD]='pass$HOME'
ENV_VALUES[POSTGRES_DATABASE]='db$NAME'
ENV_VALUES[NEO4J_PASSWORD]='neo$PASS'
ENV_VALUES[NEO4J_DATABASE]='graph$DB'
ENV_VALUES[MINIO_ACCESS_KEY_ID]='minio$USER'
ENV_VALUES[MINIO_SECRET_ACCESS_KEY]='minio$SECRET'

prepare_compose_runtime_overrides
add_docker_service postgres
add_docker_service neo4j
add_docker_service milvus

generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )

    assert (
        'MONGO_URI: "mongodb://user:p$$HOME@host.docker.internal:27017/"'
        in generated_compose
    )
    assert 'POSTGRES_USER: "user$$ID"' in generated_compose
    assert 'POSTGRES_PASSWORD: "pass$$HOME"' in generated_compose
    assert 'POSTGRES_DB: "db$$NAME"' in generated_compose
    assert 'NEO4J_AUTH: "neo4j/neo$$PASS"' in generated_compose
    assert 'NEO4J_dbms_default__database: "graph$$DB"' in generated_compose
    assert 'MINIO_ACCESS_KEY_ID: "minio$$USER"' in generated_compose
    assert 'MINIO_SECRET_ACCESS_KEY: "minio$$SECRET"' in generated_compose
    assert 'MINIO_ROOT_USER: "minio$$USER"' in generated_compose
    assert 'MINIO_ROOT_PASSWORD: "minio$$SECRET"' in generated_compose
    assert "milvus-etcd" in generated_compose
    assert "milvus-minio" in generated_compose


def test_env_base_flow_preserves_non_inference_env_values(
    tmp_path: Path,
) -> None:
    """env-base wizard should leave server, security, and observability values untouched."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HOST=127.0.0.1",
                "PORT=9999",
                "WEBUI_TITLE=Existing Title",
                "WEBUI_DESCRIPTION=Existing Description",
                "SSL=true",
                "SSL_CERTFILE=/some/cert.pem",
                "SSL_KEYFILE=/some/key.pem",
                "AUTH_ACCOUNTS=admin:secret",
                "TOKEN_SECRET=jwt-secret",
                "LIGHTRAG_API_KEY=api-key",
                "LANGFUSE_ENABLE_TRACE=true",
                "LANGFUSE_SECRET_KEY=langfuse-secret",
                "LLM_BINDING_API_KEY=sk-existing",
                "EMBEDDING_BINDING_API_KEY=sk-existing",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{ return 1; }}

finalize_base_setup() {{
  printf 'HOST=%s\\n' "${{ENV_VALUES[HOST]}}"
  printf 'PORT=%s\\n' "${{ENV_VALUES[PORT]}}"
  printf 'WEBUI_TITLE=%s\\n' "${{ENV_VALUES[WEBUI_TITLE]}}"
  printf 'WEBUI_DESCRIPTION=%s\\n' "${{ENV_VALUES[WEBUI_DESCRIPTION]}}"
  printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
  printf 'LLM_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[LLM_BINDING_API_KEY]}}"
  printf 'EMBEDDING_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_API_KEY]}}"
  printf 'SSL_SET=%s\\n' "${{ENV_VALUES[SSL]+set}}"
  printf 'AUTH_ACCOUNTS_SET=%s\\n' "${{ENV_VALUES[AUTH_ACCOUNTS]+set}}"
  printf 'TOKEN_SECRET_SET=%s\\n' "${{ENV_VALUES[TOKEN_SECRET]+set}}"
  printf 'LIGHTRAG_API_KEY_SET=%s\\n' "${{ENV_VALUES[LIGHTRAG_API_KEY]+set}}"
  printf 'LANGFUSE_ENABLE_TRACE_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_ENABLE_TRACE]+set}}"
  printf 'LANGFUSE_SECRET_KEY_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_SECRET_KEY]+set}}"
}}

env_base_flow
"""
    )
    values = parse_lines(output)

    assert values["HOST"] == "127.0.0.1"
    assert values["PORT"] == "9999"
    assert values["WEBUI_TITLE"] == "Existing Title"
    assert values["WEBUI_DESCRIPTION"] == "Existing Description"
    assert values["LLM_BINDING"] == "openai"
    assert values["LLM_BINDING_API_KEY"] == "sk-existing"
    assert values["EMBEDDING_BINDING_API_KEY"] == "sk-existing"
    # env-base does not touch server / security / observability values
    assert values["SSL_SET"] == "set"
    assert values["AUTH_ACCOUNTS_SET"] == "set"
    assert values["TOKEN_SECRET_SET"] == "set"
    assert values["LIGHTRAG_API_KEY_SET"] == "set"
    assert values["LANGFUSE_ENABLE_TRACE_SET"] == "set"
    assert values["LANGFUSE_SECRET_KEY_SET"] == "set"


def test_env_base_flow_preserves_existing_provider_bindings_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning env-base should keep prior LLM and embedding provider settings."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_BINDING=ollama",
                "LLM_MODEL=llama3.2:latest",
                "LLM_BINDING_HOST=http://localhost:11434",
                "EMBEDDING_BINDING=ollama",
                "EMBEDDING_MODEL=nomic-embed-text:latest",
                "EMBEDDING_DIM=768",
                "EMBEDDING_BINDING_HOST=http://localhost:11434",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{ return 1; }}

finalize_base_setup() {{
  printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
  printf 'LLM_MODEL=%s\\n' "${{ENV_VALUES[LLM_MODEL]}}"
  printf 'LLM_BINDING_HOST=%s\\n' "${{ENV_VALUES[LLM_BINDING_HOST]}}"
  printf 'EMBEDDING_BINDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING]}}"
  printf 'EMBEDDING_MODEL=%s\\n' "${{ENV_VALUES[EMBEDDING_MODEL]}}"
  printf 'EMBEDDING_DIM=%s\\n' "${{ENV_VALUES[EMBEDDING_DIM]}}"
  printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
}}

env_base_flow
"""
    )
    values = parse_lines(output)

    assert values["LLM_BINDING"] == "ollama"
    assert values["LLM_MODEL"] == "llama3.2:latest"
    assert values["LLM_BINDING_HOST"] == "http://localhost:11434"
    assert values["EMBEDDING_BINDING"] == "ollama"
    assert values["EMBEDDING_MODEL"] == "nomic-embed-text:latest"
    assert values["EMBEDDING_DIM"] == "768"
    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:11434"


def test_env_base_flow_preserves_existing_vllm_embedding_settings_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning env-base should keep saved local vLLM embedding model and port."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=BAAI/custom-embed",
            "EMBEDDING_DIM=1024",
            "EMBEDDING_BINDING_HOST=http://localhost:9101/v1",
            "EMBEDDING_BINDING_API_KEY=embed-key",
            "LIGHTRAG_SETUP_EMBEDDING_PROVIDER=vllm",
            "VLLM_EMBED_MODEL=BAAI/custom-embed",
            "VLLM_EMBED_PORT=9101",
            "VLLM_EMBED_DEVICE=cpu",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 0 ;;
    *) return 1 ;;
  esac
}}

finalize_base_setup() {{
  printf 'EMBEDDING_MODEL=%s\\n' "${{ENV_VALUES[EMBEDDING_MODEL]}}"
  printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
  printf 'VLLM_EMBED_MODEL=%s\\n' "${{ENV_VALUES[VLLM_EMBED_MODEL]}}"
  printf 'VLLM_EMBED_PORT=%s\\n' "${{ENV_VALUES[VLLM_EMBED_PORT]}}"
}}

env_base_flow
"""
    )

    assert values["EMBEDDING_MODEL"] == "BAAI/custom-embed"
    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:9101/v1"
    assert values["VLLM_EMBED_MODEL"] == "BAAI/custom-embed"
    assert values["VLLM_EMBED_PORT"] == "9101"


def test_env_base_flow_resets_remote_embedding_host_when_switching_to_vllm(
    tmp_path: Path,
) -> None:
    """Switching a remote embedding provider to local vLLM should restore localhost."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "EMBEDDING_BINDING=jina",
            "EMBEDDING_MODEL=jina-embeddings-v4",
            "EMBEDDING_DIM=2048",
            "EMBEDDING_BINDING_HOST=https://api.jina.ai/v1/embeddings",
            "EMBEDDING_BINDING_API_KEY=jina-key",
            "VLLM_EMBED_PORT=9101",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 0 ;;
    "Enable reranking?") return 1 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{ return 1; }}

finalize_base_setup() {{
  printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
  printf 'LIGHTRAG_SETUP_EMBEDDING_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_EMBEDDING_PROVIDER]}}"
}}

env_base_flow
"""
    )

    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:9101/v1"
    assert values["LIGHTRAG_SETUP_EMBEDDING_PROVIDER"] == "vllm"


def test_env_base_flow_preserves_existing_vllm_embedding_device_on_gpu_host(
    tmp_path: Path,
) -> None:
    """Saved vLLM embedding CPU/GPU mode should win over auto-detected GPU defaults."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=BAAI/custom-embed",
            "EMBEDDING_DIM=1024",
            "EMBEDDING_BINDING_HOST=http://localhost:9101/v1",
            "EMBEDDING_BINDING_API_KEY=embed-key",
            "LIGHTRAG_SETUP_EMBEDDING_PROVIDER=vllm",
            "VLLM_EMBED_MODEL=BAAI/custom-embed",
            "VLLM_EMBED_PORT=9101",
            "VLLM_EMBED_DEVICE=cpu",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

nvidia-smi() {{ return 0; }}
prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 0 ;;
    *) return 1 ;;
  esac
}}

finalize_base_setup() {{
  printf 'VLLM_EMBED_DEVICE=%s\\n' "${{ENV_VALUES[VLLM_EMBED_DEVICE]}}"
}}

env_base_flow
"""
    )

    assert values["VLLM_EMBED_DEVICE"] == "cpu"


def test_env_base_flow_preserves_ssl_config_on_rerun(tmp_path: Path) -> None:
    """env-base should preserve SSL config on rerun, even when old paths are stale."""

    cases = {
        "stale-paths": [
            "SSL=true",
            "SSL_CERTFILE=/missing/cert.pem",
            "SSL_KEYFILE=/missing/key.pem",
            "LLM_BINDING_API_KEY=sk-existing",
            "EMBEDDING_BINDING_API_KEY=sk-existing",
        ],
        "existing-paths": [
            "SSL=true",
            "SSL_CERTFILE=/some/cert.pem",
            "SSL_KEYFILE=/some/key.pem",
        ],
    }

    for case_name, env_lines in cases.items():
        case_dir = tmp_path / case_name
        case_dir.mkdir()
        write_text_lines(
            case_dir / "env.example",
            (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
        )
        write_text_lines(case_dir / ".env", env_lines)

        run_bash(
            f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{case_dir}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{
  case "$1" in
    "Ready to proceed and write .env?") return 0 ;;
    *) return 1 ;;
  esac
}}

env_base_flow
"""
        )

        generated_lines = (case_dir / ".env").read_text(encoding="utf-8").splitlines()
        for line in env_lines:
            assert line in generated_lines


def test_env_base_flow_preserves_existing_compose_ssl_when_env_paths_are_stale(
    tmp_path: Path,
) -> None:
    """env-base should keep compose SSL wiring when inherited source paths no longer exist."""

    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / ".env",
        [
            "SSL=true",
            "SSL_CERTFILE=/missing/cert.pem",
            "SSL_KEYFILE=/missing/key.pem",
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=text-embedding-3-small",
            "EMBEDDING_DIM=1536",
            "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
            "EMBEDDING_BINDING_API_KEY=sk-existing",
        ],
    )
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    environment:",
            '      SSL_CERTFILE: "/app/data/certs/cert.pem"',
            '      SSL_KEYFILE: "/app/data/certs/key.pem"',
            "    volumes:",
            '      - "./data/certs/cert.pem:/app/data/certs/cert.pem:ro"',
            '      - "./data/certs/key.pem:/app/data/certs/key.pem:ro"',
        ],
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{
  case "$1" in
    "Ready to proceed and write .env?") return 0 ;;
    *) return 1 ;;
  esac
}}

env_base_flow
"""
    )

    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )

    assert 'SSL_CERTFILE: "/app/data/certs/cert.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key.pem"' in generated_compose
    assert "./data/certs/cert.pem:/app/data/certs/cert.pem:ro" in generated_compose
    assert "./data/certs/key.pem:/app/data/certs/key.pem:ro" in generated_compose


def test_env_base_flow_generates_env_and_compose_files(tmp_path: Path) -> None:
    """env-base should generate `.env` and docker-compose output for hosted and local providers."""

    cases = {
        "openai": {
            "prompt_choice": "prompt_choice() { printf '%s' \"$2\"; }",
            "prompt_secret": """
prompt_secret_until_valid_with_default() {
  case "$1" in
    "LLM API key: "|"Embedding API key: ") printf 'sk-test-key' ;;
    *) printf '%s' "$2" ;;
  esac
}
""",
            "env_assertions": [
                "LLM_BINDING=openai",
                "LLM_BINDING_API_KEY=sk-test-key",
                "EMBEDDING_BINDING_API_KEY=sk-test-key",
            ],
        },
        "ollama": {
            "prompt_choice": """
prompt_choice() {
  case "$1" in
    "LLM provider") printf 'ollama' ;;
    "Embedding provider") printf 'ollama' ;;
    *) printf '%s' "$2" ;;
  esac
}
""",
            "prompt_secret": "prompt_secret_until_valid_with_default() { printf '%s' \"$2\"; }",
            "env_assertions": [
                "LLM_BINDING=ollama",
                "EMBEDDING_BINDING=ollama",
            ],
        },
    }

    for case_name, case in cases.items():
        case_dir = tmp_path / case_name
        case_dir.mkdir()
        write_text_lines(
            case_dir / "env.example",
            (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
        )
        write_text_lines(
            case_dir / "docker-compose.yml",
            (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8").splitlines(),
        )

        run_bash(
            f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{case_dir}"

{case["prompt_choice"]}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
{case["prompt_secret"]}
confirm_default_no() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 1 ;;
    "Enable reranking?") return 1 ;;
    *"for LightRAG only?"*) return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{
  case "$1" in
    "Ready to proceed and write .env?") return 0 ;;
    *) return 1 ;;
  esac
}}

env_base_flow
"""
        )

        generated_env = (case_dir / ".env").read_text(encoding="utf-8")
        generated_compose = (case_dir / "docker-compose.final.yml").read_text(
            encoding="utf-8"
        )

        assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env
        assert "LIGHTRAG_KV_STORAGE=JsonKVStorage" in generated_env
        assert "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage" in generated_env
        assert "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage" in generated_env
        assert "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage" in generated_env
        for expected_line in case["env_assertions"]:
            assert expected_line in generated_env
        assert "services:" in generated_compose
        assert "  lightrag:" in generated_compose
        assert "env_file:" not in generated_compose


def test_env_base_flow_generates_validatable_env_on_clean_checkout(
    tmp_path: Path,
) -> None:
    """Fresh env-base output should include default storage selections and pass validation."""

    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{
  case "$1" in
    "LLM API key: "|"Embedding API key: ") printf 'sk-test-key' ;;
    *) printf '%s' "$2" ;;
  esac
}}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{
  case "$1" in
    "Ready to proceed and write .env?") return 0 ;;
    *) return 1 ;;
  esac
}}

env_base_flow
validate_env_file
"""
    )

    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LIGHTRAG_KV_STORAGE=JsonKVStorage" in generated_env
    assert "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage" in generated_env
    assert "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage" in generated_env
    assert "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage" in generated_env
    assert "LIGHTRAG_RUNTIME_TARGET=host" in generated_env
    assert "LIGHTRAG_SETUP_PROFILE=" not in generated_env


def test_env_storage_flow_drops_legacy_setup_profile_on_write(tmp_path: Path) -> None:
    """Modular flows should not persist LIGHTRAG_SETUP_PROFILE into regenerated .env files."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_SETUP_PROFILE=production",
            "LIGHTRAG_KV_STORAGE=JsonKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="JsonKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="NanoVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="NetworkXStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="JsonDocStatusStorage"
}}
collect_database_config() {{ :; }}
validate_required_variables() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}

env_storage_flow
"""
    )

    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LIGHTRAG_RUNTIME_TARGET=host" in generated_env
    assert "LIGHTRAG_SETUP_PROFILE=" not in generated_env


def test_env_base_flow_registers_vllm_rerank_service_for_docker_deployment(
    tmp_path: Path,
) -> None:
    """Choosing docker rerank in env-base should add vllm-rerank to DOCKER_SERVICE_SET."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

collect_llm_config() {{ :; }}
collect_embedding_config() {{ :; }}
prompt_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 1 ;;
    "Enable reranking?") return 0 ;;
    "Run rerank service locally via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{ return 1; }}

finalize_base_setup() {{
  if [[ -n "${{DOCKER_SERVICE_SET[vllm-rerank]+set}}" ]]; then
    printf 'HAS_VLLM_SERVICE=yes\\n'
  else
    printf 'HAS_VLLM_SERVICE=no\\n'
  fi
}}

env_base_flow
"""
    )
    values = parse_lines(output)

    assert values["HAS_VLLM_SERVICE"] == "yes"


def test_env_base_flow_preserves_existing_vllm_rerank_settings_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning env-base should keep saved local vLLM rerank model and port."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "RERANK_BINDING=cohere",
            "RERANK_MODEL=BAAI/custom-rerank",
            "RERANK_BINDING_HOST=http://localhost:9200/rerank",
            "RERANK_BINDING_API_KEY=rerank-key",
            "LIGHTRAG_SETUP_RERANK_PROVIDER=vllm",
            "VLLM_RERANK_MODEL=BAAI/custom-rerank",
            "VLLM_RERANK_PORT=9200",
            "VLLM_RERANK_DEVICE=cpu",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{
  case "$1" in
    "Enable reranking?") return 0 ;;
    "Run rerank service locally via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{ return 1; }}
collect_embedding_config() {{ :; }}

finalize_base_setup() {{
  printf 'RERANK_MODEL=%s\\n' "${{ENV_VALUES[RERANK_MODEL]}}"
  printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
  printf 'VLLM_RERANK_MODEL=%s\\n' "${{ENV_VALUES[VLLM_RERANK_MODEL]}}"
  printf 'VLLM_RERANK_PORT=%s\\n' "${{ENV_VALUES[VLLM_RERANK_PORT]}}"
}}

env_base_flow
"""
    )

    assert values["RERANK_MODEL"] == "BAAI/custom-rerank"
    assert values["RERANK_BINDING_HOST"] == "http://localhost:9200/rerank"
    assert values["VLLM_RERANK_MODEL"] == "BAAI/custom-rerank"
    assert values["VLLM_RERANK_PORT"] == "9200"


def test_env_base_flow_resets_remote_rerank_host_when_switching_to_vllm(
    tmp_path: Path,
) -> None:
    """Switching a remote reranker to local vLLM should restore localhost."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "RERANK_BINDING=jina",
            "RERANK_MODEL=jina-reranker-v2-base-multilingual",
            "RERANK_BINDING_HOST=https://api.jina.ai/v1/rerank",
            "RERANK_BINDING_API_KEY=jina-key",
            "VLLM_RERANK_PORT=9200",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{
  case "$1" in
    "Run rerank service locally via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{
  case "$1" in
    "Enable reranking?") return 0 ;;
    *) return 1 ;;
  esac
}}
collect_embedding_config() {{ :; }}

finalize_base_setup() {{
  printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
  printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]}}"
}}

env_base_flow
"""
    )

    assert values["RERANK_BINDING_HOST"] == "http://localhost:9200/rerank"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == "vllm"


def test_env_base_flow_preserves_existing_vllm_rerank_device_on_gpu_host(
    tmp_path: Path,
) -> None:
    """Saved vLLM rerank CPU/GPU mode should win over auto-detected GPU defaults."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "RERANK_BINDING=cohere",
            "RERANK_MODEL=BAAI/custom-rerank",
            "RERANK_BINDING_HOST=http://localhost:9200/rerank",
            "RERANK_BINDING_API_KEY=rerank-key",
            "LIGHTRAG_SETUP_RERANK_PROVIDER=vllm",
            "VLLM_RERANK_MODEL=BAAI/custom-rerank",
            "VLLM_RERANK_PORT=9200",
            "VLLM_RERANK_DEVICE=cpu",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

nvidia-smi() {{ return 0; }}
prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{
  case "$1" in
    "Enable reranking?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{
  case "$1" in
    "Run rerank service locally via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
collect_embedding_config() {{ :; }}

finalize_base_setup() {{
  printf 'VLLM_RERANK_DEVICE=%s\\n' "${{ENV_VALUES[VLLM_RERANK_DEVICE]}}"
}}

env_base_flow
"""
    )

    assert values["VLLM_RERANK_DEVICE"] == "cpu"


def test_env_storage_flow_applies_selected_storage_backends(
    tmp_path: Path,
) -> None:
    """env-storage should honor the selected backends without auto-applying a preset."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_KV_STORAGE=JsonKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
            "LLM_BINDING=ollama",
            "LLM_MODEL=llama3.2:latest",
            "LLM_BINDING_HOST=http://localhost:11434",
            "EMBEDDING_BINDING=ollama",
            "EMBEDDING_MODEL=nomic-embed-text:latest",
            "EMBEDDING_DIM=768",
            "EMBEDDING_BINDING_HOST=http://localhost:11434",
        ],
    )

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="RedisKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="MilvusVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="Neo4JStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="RedisDocStatusStorage"
}}
collect_database_config() {{ :; }}
collect_docker_image_tags() {{ :; }}
finalize_storage_setup() {{
  printf 'LIGHTRAG_KV_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_KV_STORAGE]}}"
  printf 'LIGHTRAG_VECTOR_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]}}"
  printf 'LIGHTRAG_GRAPH_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]}}"
  printf 'LIGHTRAG_DOC_STATUS_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]}}"
  printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
  printf 'EMBEDDING_BINDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING]}}"
}}

env_storage_flow
"""
    )

    assert values["LIGHTRAG_KV_STORAGE"] == "RedisKVStorage"
    assert values["LIGHTRAG_VECTOR_STORAGE"] == "MilvusVectorDBStorage"
    assert values["LIGHTRAG_GRAPH_STORAGE"] == "Neo4JStorage"
    assert values["LIGHTRAG_DOC_STATUS_STORAGE"] == "RedisDocStatusStorage"
    # LLM and embedding settings from existing .env are preserved
    assert values["LLM_BINDING"] == "ollama"
    assert values["EMBEDDING_BINDING"] == "ollama"


def test_env_storage_flow_generates_env_and_compose_files(tmp_path: Path) -> None:
    """env-storage should write updated .env and a docker-compose.final.yml."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_BINDING=ollama",
                "EMBEDDING_BINDING=ollama",
                "AUTH_ACCOUNTS=admin:secret",
                "TOKEN_SECRET=jwt-secret",
                "WHITELIST_PATHS=/health",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "env.example").write_text(
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "docker-compose.yml").write_text(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="MilvusVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="Neo4JStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
  add_docker_service "postgres"
  add_docker_service "neo4j"
}}
collect_database_config() {{ :; }}
collect_docker_image_tags() {{ :; }}
validate_required_variables() {{ return 0; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_yes() {{
  case "$1" in
    "Ready to proceed and write .env?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_no() {{ return 1; }}

env_storage_flow
"""
    )

    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )

    assert "LIGHTRAG_KV_STORAGE=PGKVStorage" in generated_env
    assert "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage" in generated_env
    assert "LLM_BINDING=ollama" in generated_env
    assert "services:" in generated_compose
    assert "  lightrag:" in generated_compose
    assert "env_file:" not in generated_compose


def test_env_storage_flow_preserves_existing_compose_ssl_when_env_paths_are_stale(
    tmp_path: Path,
) -> None:
    """env-storage should keep compose SSL wiring when inherited source paths no longer exist."""

    write_text_lines(
        tmp_path / ".env",
        [
            "SSL=true",
            "SSL_CERTFILE=/missing/cert.pem",
            "SSL_KEYFILE=/missing/key.pem",
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "LIGHTRAG_KV_STORAGE=JsonKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    environment:",
            '      SSL_CERTFILE: "/app/data/certs/cert.pem"',
            '      SSL_KEYFILE: "/app/data/certs/key.pem"',
            "    volumes:",
            '      - "./data/certs/cert.pem:/app/data/certs/cert.pem:ro"',
            '      - "./data/certs/key.pem:/app/data/certs/key.pem:ro"',
        ],
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="JsonKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="NanoVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="NetworkXStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="JsonDocStatusStorage"
}}
collect_database_config() {{ :; }}
validate_required_variables() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}

env_storage_flow
"""
    )

    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )

    assert 'SSL_CERTFILE: "/app/data/certs/cert.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key.pem"' in generated_compose
    assert "./data/certs/cert.pem:/app/data/certs/cert.pem:ro" in generated_compose
    assert "./data/certs/key.pem:/app/data/certs/key.pem:ro" in generated_compose


def test_env_server_flow_preserves_existing_compose_ssl_when_env_paths_are_stale(
    tmp_path: Path,
) -> None:
    """env-server should keep compose SSL wiring when inherited source paths no longer exist."""

    write_text_lines(
        tmp_path / ".env",
        [
            "SSL=true",
            "SSL_CERTFILE=/missing/cert.pem",
            "SSL_KEYFILE=/missing/key.pem",
            "HOST=0.0.0.0",
            "PORT=9621",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    environment:",
            '      SSL_CERTFILE: "/app/data/certs/cert.pem"',
            '      SSL_KEYFILE: "/app/data/certs/key.pem"',
            "    volumes:",
            '      - "./data/certs/cert.pem:/app/data/certs/cert.pem:ro"',
            '      - "./data/certs/key.pem:/app/data/certs/key.pem:ro"',
        ],
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

collect_server_config() {{
  ENV_VALUES[HOST]="0.0.0.0"
  ENV_VALUES[PORT]="8080"
}}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
confirm_default_yes() {{ return 0; }}

env_server_flow
"""
    )

    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )

    assert 'SSL_CERTFILE: "/app/data/certs/cert.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key.pem"' in generated_compose
    assert "./data/certs/cert.pem:/app/data/certs/cert.pem:ro" in generated_compose
    assert "./data/certs/key.pem:/app/data/certs/key.pem:ro" in generated_compose
    assert 'PORT: "9621"' in generated_compose


def test_switching_to_non_docker_storage_removes_stale_services_from_compose(
    tmp_path: Path,
) -> None:
    """env-storage must strip managed storage services while preserving user sidecars."""

    # Existing compose with postgres and neo4j Docker services.
    compose_file = tmp_path / "docker-compose.final.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "  postgres:",
                "    image: gzdaniel/postgres-for-rag:16.6",
                "  neo4j:",
                "    image: neo4j:5.26.21-community",
                "  sidecar:",
                "    image: busybox",
                '    command: ["sleep", "infinity"]',
                "    volumes:",
                "      - sidecar_data:/data",
                "volumes:",
                "  postgres_data:",
                "  neo4j_data:",
                "  sidecar_data:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BINDING=openai\n", encoding="utf-8")
    (tmp_path / "env.example").write_text(
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "docker-compose.yml").write_text(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # User switches to non-Docker backends: DOCKER_SERVICES stays empty.
    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="JsonKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="NanoVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="NetworkXStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="JsonDocStatusStorage"
}}
collect_database_config() {{ :; }}
collect_docker_image_tags() {{ :; }}
validate_required_variables() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}

env_storage_flow
"""
    )

    result = compose_file.read_text(encoding="utf-8")
    # Stale storage services must be gone.
    assert "postgres:" not in result
    assert "neo4j:" not in result
    assert "postgres_data:" not in result
    assert "neo4j_data:" not in result
    # lightrag and user services must be preserved.
    assert "  lightrag:" in result
    assert "  sidecar:" in result
    assert "sidecar_data:" in result


def test_generate_docker_compose_uses_template_images_even_with_old_env_overrides(
    tmp_path: Path,
) -> None:
    """Managed services should be regenerated from templates instead of legacy image overrides."""

    write_text_lines(
        tmp_path / ".env",
        [
            "POSTGRES_IMAGE=registry.example.com/postgres-for-rag:patched",
            "VLLM_EMBED_IMAGE_TAG=patched",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
add_docker_service postgres
add_docker_service vllm-embed
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml"
"""
    )

    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")

    assert "image: gzdaniel/postgres-for-rag:16.6" in result
    assert "image: vllm/vllm-openai-cpu:latest" in result
    assert "registry.example.com/postgres-for-rag:patched" not in result
    assert "vllm/vllm-openai-cpu:patched" not in result


def test_generate_docker_compose_preserves_long_form_named_sidecar_volumes(
    tmp_path: Path,
) -> None:
    """Managed-service regeneration must not misparse preserved long-form named volumes."""

    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
        ],
    )
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  sidecar:",
            "    image: busybox",
            '    command: ["sleep", "infinity"]',
            "    volumes:",
            "      - source: sidecar_data",
            "        target: /data",
            "        type: volume",
            "volumes:",
            "  sidecar_data:",
        ],
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
add_docker_service postgres
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml"
"""
    )

    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")

    assert "  sidecar_data:" in result
    assert "\n  source:\n" not in result


def test_collect_milvus_config_defaults_to_existing_database_name() -> None:
    """Milvus database prompt should preserve the documented default database."""

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

confirm_default_yes() {{ return 1; }}
prompt_with_default() {{
  printf '%s' "$2"
}}
prompt_until_valid() {{
  printf '%s' "$2"
}}

collect_milvus_config no

printf 'MILVUS_DB_NAME=%s\\n' "${{ENV_VALUES[MILVUS_DB_NAME]}}"
"""
    )

    assert values["MILVUS_DB_NAME"] == "lightrag"


@pytest.mark.parametrize(
    ("env_key", "env_value", "expected_value"),
    [
        ("POSTGRES_HOST", "127.0.0.1", "host.docker.internal"),
        ("REDIS_URI", "redis://localhost:6379", "redis://host.docker.internal:6379"),
        (
            "MONGO_URI",
            "mongodb://127.0.0.1:27017/",
            "mongodb://host.docker.internal:27017/",
        ),
        (
            "MONGO_URI",
            "mongodb://root:root@localhost:27017/",
            "mongodb://root:root@host.docker.internal:27017/",
        ),
        ("NEO4J_URI", "neo4j://localhost:7687", "neo4j://host.docker.internal:7687"),
        ("MILVUS_URI", "http://localhost:19530", "http://host.docker.internal:19530"),
        ("QDRANT_URL", "http://127.0.0.1:6333", "http://host.docker.internal:6333"),
        ("MEMGRAPH_URI", "bolt://localhost:7687", "bolt://host.docker.internal:7687"),
        ("POSTGRES_HOST", "0.0.0.0", "host.docker.internal"),
        (
            "LLM_BINDING_HOST",
            "http://0.0.0.0:11434",
            "http://host.docker.internal:11434",
        ),
        (
            "RERANK_BINDING_HOST",
            "http://0.0.0.0:8000/rerank",
            "http://host.docker.internal:8000/rerank",
        ),
    ],
    ids=[
        "postgres-loopback-host",
        "redis-loopback-uri",
        "mongo-loopback-uri",
        "mongo-authenticated-loopback-uri",
        "neo4j-loopback-uri",
        "milvus-loopback-uri",
        "qdrant-loopback-uri",
        "memgraph-loopback-uri",
        "postgres-zero-host",
        "llm-zero-host",
        "rerank-zero-host",
    ],
)
def test_prepare_compose_runtime_overrides_rewrites_container_endpoints(
    env_key: str, env_value: str, expected_value: str
) -> None:
    """Loopback and 0.0.0.0 endpoints should be rewritten for container reachability."""

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[{env_key}]="{env_value}"

prepare_compose_runtime_overrides

printf '{env_key}=%s\\n' "${{COMPOSE_ENV_OVERRIDES[{env_key}]}}"
"""
    )

    assert values[env_key] == expected_value


def test_wait_for_port_uses_explicit_timeout_argument() -> None:
    """`wait_for_port` should honor arg4 instead of always falling back to WAIT_TIMEOUT."""

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state
WAIT_TIMEOUT=1

start=$SECONDS
if wait_for_port 127.0.0.1 65535 probe 0 >/dev/null 2>&1; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
elapsed=$((SECONDS - start))
printf 'ELAPSED=%s\\n' "$elapsed"
"""
    )

    assert values["RESULT"] == "failure"
    assert int(values["ELAPSED"]) < 2


def test_collect_mongodb_config_local_service_strips_stale_credentials_on_rerun() -> (
    None
):
    """Bundled MongoDB should keep host `.env` aligned with the unauthenticated template."""

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[MONGO_URI]="mongodb://root:secret@localhost:27018/"

confirm_default_yes() {{ return 0; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{
  if [[ "$1" == "MongoDB database" ]]; then
    printf 'LightRAG'
  else
    printf '%s' "$2"
  fi
}}

collect_mongodb_config yes

printf 'MONGO_URI=%s\\n' "${{ENV_VALUES[MONGO_URI]}}"
printf 'COMPOSE_MONGO_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[MONGO_URI]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}"
"""
    )

    assert values["MONGO_URI"] == "mongodb://localhost:27017/"
    assert values["COMPOSE_MONGO_URI"] == "mongodb://mongodb:27017/"
    assert values["DOCKER_SERVICE"] == "mongodb"


def test_collect_redis_config_local_service_normalizes_custom_host_port() -> None:
    """Bundled Redis should keep host `.env` aligned with the published local port."""

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[REDIS_URI]="redis://localhost:6380/1"

confirm_default_yes() {{ return 0; }}
prompt_until_valid() {{ printf '%s' "$2"; }}

collect_redis_config yes

printf 'REDIS_URI=%s\\n' "${{ENV_VALUES[REDIS_URI]}}"
printf 'COMPOSE_REDIS_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[REDIS_URI]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}"
"""
    )

    assert values["REDIS_URI"] == "redis://localhost:6379/1"
    assert values["COMPOSE_REDIS_URI"] == "redis://redis:6379"
    assert values["DOCKER_SERVICE"] == "redis"


@pytest.mark.parametrize(
    "host_value",
    ["127.0.0.1", "192.168.1.10"],
    ids=["loopback-bind", "lan-bind"],
)
def test_prepare_compose_runtime_overrides_normalizes_server_binding(
    host_value: str,
) -> None:
    """Compose runtime should always bind the API to the container-facing host/port."""

    values = run_bash_lines(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[HOST]="{host_value}"
ENV_VALUES[PORT]="8080"

prepare_compose_runtime_overrides

printf 'HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[HOST]}}"
printf 'PORT=%s\\n' "${{COMPOSE_ENV_OVERRIDES[PORT]}}"
"""
    )

    assert values["HOST"] == "0.0.0.0"
    assert values["PORT"] == "9621"


def test_generate_docker_compose_injects_server_host_and_port_overrides(
    tmp_path: Path,
) -> None:
    """Generated compose should keep the published host port while fixing container bind values."""

    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    env_file:",
                "      - .env",
                "    ports:",
                '      - "${PORT:-9621}:9621"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[HOST]="localhost"
ENV_VALUES[PORT]="8080"

prepare_compose_runtime_overrides
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )

    assert 'HOST: "0.0.0.0"' in generated_compose
    assert 'PORT: "9621"' in generated_compose
    assert "${PORT:-9621}:9621" in generated_compose


def test_finalize_server_setup_skips_embedded_milvus_sub_services(
    tmp_path: Path,
) -> None:
    """finalize_server_setup must keep prefixed Milvus child services on rerun."""

    compose_file = tmp_path / "docker-compose.final.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "  milvus:",
                "    image: milvusdb/milvus:v2.6.11",
                "  milvus-etcd:",
                "    image: quay.io/coreos/etcd:v3.5.16",
                "  milvus-minio:",
                "    image: minio/minio:RELEASE.2024-12-13T22-19-12Z",
                "volumes:",
                "  milvus_data:",
                "  milvus-etcd_data:",
                "  milvus-minio_data:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "env.example").write_text(
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Should complete without error; Milvus child services are managed via the
    # Milvus template, not as independent root services.
    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
collect_server_config() {{ :; }}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
finalize_server_setup
"""
    )

    result = compose_file.read_text(encoding="utf-8")
    # The Milvus template and its prefixed child services must still be present.
    assert "milvus" in result
    assert "milvus-etcd" in result
    assert "milvus-minio" in result


def test_finalize_server_setup_allows_risky_security_config_and_security_check_reports_it(
    tmp_path: Path,
) -> None:
    """Wizard writes `.env` without blocking, while security-check reports risky settings."""

    write_text_lines(
        tmp_path / ".env",
        [
            "AUTH_ACCOUNTS=admin:secret",
            "TOKEN_SECRET=jwt-secret",
            "WHITELIST_PATHS=/health,/api/*",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

show_summary() {{ :; }}
confirm_default_yes() {{ return 0; }}

if finalize_server_setup; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["RESULT"] == "success"

    result = subprocess.run(
        [
            "bash",
            "--norc",
            "--noprofile",
            "-c",
            f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
security_check_env_file
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "WHITELIST_PATHS exposes /api routes" in result.stdout


def test_validate_uri_accepts_neo4j_self_signed_tls_scheme() -> None:
    """Neo4j self-signed TLS URIs should pass validation."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"

if validate_uri "neo4j+ssc://db.example.com:7687" neo4j; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["VALID"] == "yes"


def test_ssl_staging_uses_distinct_names_for_same_basename_inputs(
    tmp_path: Path,
) -> None:
    """Cert/key files with the same basename should stage to distinct paths."""

    env_example = tmp_path / "env.example"
    env_example.write_text(
        "\n".join(
            [
                "SSL_CERTFILE=/placeholder/cert.pem",
                "SSL_KEYFILE=/placeholder/key.pem",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    env_file:",
                "      - .env",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cert_dir = tmp_path / "certs"
    key_dir = tmp_path / "keys"
    cert_dir.mkdir()
    key_dir.mkdir()
    cert_path = cert_dir / "server.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path = key_dir / "server.pem"
    key_path.write_text("key", encoding="utf-8")

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[SSL_CERTFILE]="{cert_path}"
ENV_VALUES[SSL_KEYFILE]="{key_path}"
SSL_CERT_SOURCE_PATH="{cert_path}"
SSL_KEY_SOURCE_PATH="{key_path}"

prepare_compose_env_overrides
stage_ssl_assets "$SSL_CERT_SOURCE_PATH" "$SSL_KEY_SOURCE_PATH"
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )
    staged_cert = tmp_path / "data" / "certs" / "cert-server.pem"
    staged_key = tmp_path / "data" / "certs" / "key-server.pem"

    assert staged_cert.read_text(encoding="utf-8") == "cert"
    assert staged_key.read_text(encoding="utf-8") == "key"
    assert 'SSL_CERTFILE: "/app/data/certs/cert-server.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key-server.pem"' in generated_compose
    assert (
        "./data/certs/cert-server.pem:/app/data/certs/cert-server.pem:ro"
        in generated_compose
    )
    assert (
        "./data/certs/key-server.pem:/app/data/certs/key-server.pem:ro"
        in generated_compose
    )


def test_ssl_staging_skips_copy_for_already_staged_relative_paths(
    tmp_path: Path,
) -> None:
    """Re-running setup with already-staged certs should not fail on identical copies."""

    staged_dir = tmp_path / "data" / "certs"
    staged_dir.mkdir(parents=True)
    cert_path = staged_dir / "server.pem"
    key_path = staged_dir / "server.key"
    cert_path.write_text("cert", encoding="utf-8")
    key_path.write_text("key", encoding="utf-8")

    run_bash(
        f"""
set -euo pipefail
cd "{tmp_path}"
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

stage_ssl_assets "./data/certs/server.pem" "./data/certs/server.key"
"""
    )

    assert cert_path.read_text(encoding="utf-8") == "cert"
    assert key_path.read_text(encoding="utf-8") == "key"


@pytest.mark.parametrize(
    ("name", "env_lines", "setup_snippet", "finalize_call"),
    [
        (
            "base",
            [],
            "\n".join(
                [
                    'ENV_VALUES[VLLM_EMBED_DEVICE]="cpu"',
                    'ENV_VALUES[VLLM_EMBED_MODEL]="BAAI/bge-m3"',
                    'ENV_VALUES[VLLM_EMBED_PORT]="8001"',
                    'ENV_VALUES[VLLM_EMBED_API_KEY]="local-key"',
                    'add_docker_service "vllm-embed"',
                    "confirm_default_no() { return 1; }",
                ]
            ),
            "finalize_base_setup",
        ),
        (
            "storage",
            [
                "LIGHTRAG_KV_STORAGE=PGKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
                "POSTGRES_USER=lightrag",
                "POSTGRES_PASSWORD=secret",
                "POSTGRES_DATABASE=lightrag",
            ],
            'add_docker_service "postgres"',
            "finalize_storage_setup",
        ),
    ],
    ids=["base", "storage"],
)
def test_finalize_flows_stage_inherited_ssl_assets_for_compose(
    tmp_path: Path,
    name: str,
    env_lines: list[str],
    setup_snippet: str,
    finalize_call: str,
) -> None:
    """Compose-writing finalize flows should stage inherited SSL assets before mounting them."""

    cert_path = tmp_path / f"{name}-source-cert.pem"
    key_path = tmp_path / f"{name}-source-key.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path.write_text("key", encoding="utf-8")
    write_text_lines(
        tmp_path / ".env",
        [
            *env_lines,
            "SSL=true",
            f"SSL_CERTFILE={cert_path}",
            f"SSL_KEYFILE={key_path}",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.yml").write_text(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

{setup_snippet}

show_summary() {{ :; }}
confirm_default_yes() {{ return 0; }}

{finalize_call}
"""
    )

    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )

    staged_cert = tmp_path / "data" / "certs" / f"{name}-source-cert.pem"
    staged_key = tmp_path / "data" / "certs" / f"{name}-source-key.pem"

    assert staged_cert.read_text(encoding="utf-8") == "cert"
    assert staged_key.read_text(encoding="utf-8") == "key"
    assert (
        f"./data/certs/{name}-source-cert.pem:/app/data/certs/{name}-source-cert.pem:ro"
        in generated_compose
    )
    assert (
        f"./data/certs/{name}-source-key.pem:/app/data/certs/{name}-source-key.pem:ro"
        in generated_compose
    )


def test_generate_docker_compose_vllm_gpu_honors_documented_gpu_selector(
    tmp_path: Path,
) -> None:
    """GPU vLLM compose should honor the documented CUDA selector variables."""

    env_example = tmp_path / "env.example"
    env_example.write_text(
        "\n".join(
            [
                "# VLLM_RERANK_DEVICE=cuda",
                "# CUDA_VISIBLE_DEVICES=-1",
                "# NVIDIA_VISIBLE_DEVICES=all",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    env_file:",
                "      - .env",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[VLLM_RERANK_DEVICE]="cuda"
ENV_VALUES[CUDA_VISIBLE_DEVICES]="0"
add_docker_service "vllm-rerank"

generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )

    assert "CUDA_VISIBLE_DEVICES=0" in generated_env
    assert "NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-all}" in generated_compose


def test_collect_security_config_can_clear_existing_values_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning security setup should be able to remove previously saved values."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AUTH_ACCOUNTS=admin:secret",
                "TOKEN_SECRET=jwt-secret",
                "TOKEN_EXPIRE_HOURS=72",
                "LIGHTRAG_API_KEY=api-key",
                "WHITELIST_PATHS=/health,/api/*,/docs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

confirm_default_no() {{ return 0; }}
prompt_clearable_with_default() {{ printf '%s' "$CLEAR_INPUT_SENTINEL"; }}
prompt_clearable_secret_with_default() {{ printf '%s' "$CLEAR_INPUT_SENTINEL"; }}

collect_security_config yes no
generate_env_file "{REPO_ROOT}/env.example" "$REPO_ROOT/.env.generated"

printf 'AUTH_ACCOUNTS_SET=%s\\n' "${{ENV_VALUES[AUTH_ACCOUNTS]+set}}"
printf 'TOKEN_SECRET_SET=%s\\n' "${{ENV_VALUES[TOKEN_SECRET]+set}}"
printf 'TOKEN_EXPIRE_HOURS_SET=%s\\n' "${{ENV_VALUES[TOKEN_EXPIRE_HOURS]+set}}"
printf 'LIGHTRAG_API_KEY_SET=%s\\n' "${{ENV_VALUES[LIGHTRAG_API_KEY]+set}}"
printf 'WHITELIST_PATHS_SET=%s\\n' "${{ENV_VALUES[WHITELIST_PATHS]+set}}"
"""
    )
    values = parse_lines(output)
    generated_lines = (
        (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()
    )

    assert values["AUTH_ACCOUNTS_SET"] == ""
    assert values["TOKEN_SECRET_SET"] == ""
    assert values["TOKEN_EXPIRE_HOURS_SET"] == ""
    assert values["LIGHTRAG_API_KEY_SET"] == ""
    assert values["WHITELIST_PATHS_SET"] == "set"
    assert not any(line.startswith("AUTH_ACCOUNTS=") for line in generated_lines)
    assert not any(line.startswith("TOKEN_SECRET=") for line in generated_lines)
    assert not any(line.startswith("TOKEN_EXPIRE_HOURS=") for line in generated_lines)
    assert not any(line.startswith("LIGHTRAG_API_KEY=") for line in generated_lines)
    assert "WHITELIST_PATHS=" in generated_lines


def test_collect_security_config_preserves_explicit_empty_whitelist_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning security setup should keep an explicitly empty whitelist unchanged."""

    env_file = tmp_path / ".env"
    env_file.write_text("WHITELIST_PATHS=\n", encoding="utf-8")

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

prompt_clearable_with_default() {{ printf '%s' "$2"; }}
prompt_clearable_secret_with_default() {{ printf '%s' "$2"; }}

collect_security_config no no

printf 'WHITELIST_PATHS_SET=%s\\n' "${{ENV_VALUES[WHITELIST_PATHS]+set}}"
printf 'WHITELIST_PATHS=%s\\n' "${{ENV_VALUES[WHITELIST_PATHS]}}"
"""
    )
    values = parse_lines(output)

    assert values["WHITELIST_PATHS_SET"] == "set"
    assert values["WHITELIST_PATHS"] == ""


def test_collect_observability_config_clears_existing_values_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning setup should remove saved Langfuse settings when observability is declined."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LANGFUSE_ENABLE_TRACE=true",
                "LANGFUSE_SECRET_KEY=old-secret",
                "LANGFUSE_PUBLIC_KEY=old-public",
                "LANGFUSE_HOST=https://langfuse.example",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present


collect_observability_config
generate_env_file "{REPO_ROOT}/env.example" "$REPO_ROOT/.env.generated"

printf 'LANGFUSE_ENABLE_TRACE_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_ENABLE_TRACE]+set}}"
printf 'LANGFUSE_SECRET_KEY_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_SECRET_KEY]+set}}"
printf 'LANGFUSE_PUBLIC_KEY_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_PUBLIC_KEY]+set}}"
printf 'LANGFUSE_HOST_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_HOST]+set}}"
"""
    )
    values = parse_lines(output)
    generated_lines = (
        (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()
    )

    assert values["LANGFUSE_ENABLE_TRACE_SET"] == ""
    assert values["LANGFUSE_SECRET_KEY_SET"] == ""
    assert values["LANGFUSE_PUBLIC_KEY_SET"] == ""
    assert values["LANGFUSE_HOST_SET"] == ""
    assert not any(
        line.startswith("LANGFUSE_ENABLE_TRACE=") for line in generated_lines
    )
    assert not any(line.startswith("LANGFUSE_SECRET_KEY=") for line in generated_lines)
    assert not any(line.startswith("LANGFUSE_PUBLIC_KEY=") for line in generated_lines)
    assert not any(line.startswith("LANGFUSE_HOST=") for line in generated_lines)


def test_collect_neo4j_config_bundled_service_pins_username_to_default(
    tmp_path: Path,
) -> None:
    """Bundled Neo4j should keep the bootstrap username aligned with the image defaults."""

    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    env_file:",
                "      - .env",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[NEO4J_USERNAME]="custom-user"
ENV_VALUES[NEO4J_PASSWORD]="existing-password"

confirm_default_yes() {{ return 0; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{
  if [[ "$1" == "Neo4j database" ]]; then
    printf 'neo4j'
  else
    printf 'custom-user'
  fi
}}
prompt_secret_with_default() {{ printf 'test-password'; }}

collect_neo4j_config yes
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"

printf 'NEO4J_USERNAME=%s\\n' "${{ENV_VALUES[NEO4J_USERNAME]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}"
"""
    )
    values = parse_lines(output)
    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )

    assert values["NEO4J_USERNAME"] == "neo4j"
    assert values["DOCKER_SERVICE"] == "neo4j"
    assert 'NEO4J_AUTH: "neo4j/test-password"' in generated_compose


def test_validate_security_config_rejects_malformed_auth_accounts() -> None:
    """Security validation should reject auth entries the API cannot parse."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

if validate_security_config "admin" "token-secret" "" no "/health"; then
  printf 'MISSING_COLON=yes\\n'
else
  printf 'MISSING_COLON=no\\n'
fi

if validate_security_config "admin:secret," "token-secret" "" no "/health"; then
  printf 'TRAILING_COMMA=yes\\n'
else
  printf 'TRAILING_COMMA=no\\n'
fi

if validate_security_config "admin:secret,reader:hunter2" "token-secret" "" no "/health"; then
  printf 'VALID_FORMAT=yes\\n'
else
  printf 'VALID_FORMAT=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["MISSING_COLON"] == "no"
    assert values["TRAILING_COMMA"] == "no"
    assert values["VALID_FORMAT"] == "yes"


def test_security_check_reports_missing_authentication(tmp_path: Path) -> None:
    """Security audit should flag unauthenticated API exposure."""

    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0"])

    result = subprocess.run(
        [
            "bash",
            "--norc",
            "--noprofile",
            "-c",
            f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
security_check_env_file
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "No API protection is configured." in result.stdout


def test_security_check_passes_for_authenticated_minimal_config(tmp_path: Path) -> None:
    """Security audit should pass for a minimally hardened config."""

    write_text_lines(
        tmp_path / ".env",
        [
            "AUTH_ACCOUNTS=admin:secret",
            "TOKEN_SECRET=jwt-secret",
            "WHITELIST_PATHS=/health",
        ],
    )

    result = subprocess.run(
        [
            "bash",
            "--norc",
            "--noprofile",
            "-c",
            f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
security_check_env_file
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "No obvious security issues found" in result.stdout


def test_show_summary_masks_auth_accounts() -> None:
    """Configuration summaries should not print auth account passwords."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[AUTH_ACCOUNTS]="admin:secret,reader:hunter2"
ENV_VALUES[TOKEN_SECRET]="jwt-secret"
ENV_VALUES[HOST]="0.0.0.0"

show_summary
"""
    )

    assert "AUTH_ACCOUNTS=***" in output
    assert "TOKEN_SECRET=***" in output
    assert "admin:secret" not in output
    assert "reader:hunter2" not in output


def test_finalize_setup_validates_inherited_ssl_assets_before_staging(
    tmp_path: Path,
) -> None:
    """Finalize should fail with a clear validation error before copying missing SSL files."""

    (tmp_path / "env.example").write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=JsonKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[LIGHTRAG_KV_STORAGE]="JsonKVStorage"
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="NanoVectorDBStorage"
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="NetworkXStorage"
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="JsonDocStatusStorage"
SSL_CERT_SOURCE_PATH="/missing/cert.pem"
SSL_KEY_SOURCE_PATH="/missing/key.pem"


finalize_setup
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Invalid SSL_CERTFILE" in result.stderr
    assert "Invalid SSL_KEYFILE" not in result.stderr
    assert "No such file or directory" not in result.stderr


def test_validate_env_file_handles_supported_and_unsupported_uri_schemes(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject malformed schemes and allow supported TLS variants."""

    cases = {
        "invalid-neo4j-scheme": (
            ["NEO4J_URI=http://localhost:7687"],
            "no",
            "Invalid NEO4J_URI",
        ),
        "invalid-redis-scheme": (
            ["REDIS_URI=tcp://localhost:6379"],
            "no",
            "Invalid REDIS_URI",
        ),
        "valid-rediss-scheme": (
            ["REDIS_URI=rediss://localhost:6380"],
            "yes",
            "",
        ),
    }

    for case_name, (extra_lines, expected_valid, expected_stderr) in cases.items():
        case_dir = tmp_path / case_name
        case_dir.mkdir()
        write_text_lines(
            case_dir / ".env",
            [
                "LIGHTRAG_KV_STORAGE=JsonKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
                *extra_lines,
            ],
        )
        write_text_lines(case_dir / "env.example", ["LLM_BINDING=openai"])

        result = subprocess.run(
            [
                "bash",
                "--norc",
                "--noprofile",
                "-c",
                f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{case_dir}"
reset_state
if validate_env_file; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        values = parse_lines(result.stdout)
        assert values["VALID"] == expected_valid
        if expected_stderr:
            assert expected_stderr in result.stderr


def test_validate_env_file_rejects_invalid_runtime_target(tmp_path: Path) -> None:
    """validate_env_file should reject unsupported LIGHTRAG_RUNTIME_TARGET values."""

    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=laptop",
            "LIGHTRAG_KV_STORAGE=JsonKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
        ],
    )
    write_text_lines(tmp_path / "env.example", ["LLM_BINDING=openai"])

    result = subprocess.run(
        [
            "bash",
            "--norc",
            "--noprofile",
            "-c",
            f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
if validate_env_file; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    values = parse_lines(result.stdout)
    assert values["VALID"] == "no"
    assert "Invalid LIGHTRAG_RUNTIME_TARGET" in result.stderr


def test_validate_env_file_rejects_mongo_vector_storage_without_atlas_uri(
    tmp_path: Path,
) -> None:
    """validate_env_file must reject MongoVectorDBStorage when MONGO_URI is not Atlas (mongodb+srv://)."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=JsonKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
                "MONGO_URI=mongodb://localhost:27017",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "env.example").write_text("LLM_BINDING=openai\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "--norc",
            "--noprofile",
            "-c",
            f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
if validate_env_file; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    values = parse_lines(result.stdout)
    assert values["VALID"] == "no"
    assert "MongoVectorDBStorage requires a MongoDB Atlas URI" in result.stderr
