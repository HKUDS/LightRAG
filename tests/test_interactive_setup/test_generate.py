# Regression tests for interactive setup wizard.
# Classification: keep tests here when they cover generate_* helpers that render or rewrite .env and docker-compose file contents.

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_interactive_setup._helpers import (
    PRESERVED_HEADER,
    PRESERVED_NOTICE,
    REPO_ROOT,
    parse_lines,
    run_bash,
    write_text_lines,
)

pytestmark = pytest.mark.offline


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
    run_bash(f"""
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
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml\"
""")
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
            "    container_name: lightrag",
            "    image: example/lightrag:test",
            "    env_file:",
            "      - .env",
            "    volumes:",
            "      - ./.env:/app/.env",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )
    assert "env_file:" not in generated_compose
    assert "environment:" not in generated_compose
    assert "container_name:" not in generated_compose
    assert "- ./.env:/app/.env" in generated_compose


def test_generate_docker_compose_removes_lightrag_container_name_from_existing_output(
    tmp_path: Path,
) -> None:
    """Compose regeneration should strip fixed lightrag container names from prior output."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    container_name: lightrag",
            "    image: example/lightrag:test",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert "container_name:" not in generated_compose


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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
set_compose_override "PORT" "1234"
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert '      - "PORT=1234"' in generated_compose
    assert "      - FOO=bar" in generated_compose
    assert "      PORT:" not in generated_compose


def test_generate_docker_compose_injects_healthchecks_and_lightrag_depends_on(
    tmp_path: Path,
) -> None:
    """Generated compose should gate LightRAG on all managed dependencies becoming healthy."""
    write_text_lines(
        tmp_path / "docker-compose.yml",
        ["services:", "  lightrag:", "    image: example/lightrag:test"],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

add_docker_service postgres
add_docker_service neo4j
add_docker_service mongodb
add_docker_service redis
add_docker_service milvus
add_docker_service qdrant
add_docker_service memgraph
add_docker_service vllm-embed
add_docker_service vllm-rerank

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    lightrag_start = generated_compose.index("  lightrag:\n")
    embed_start = generated_compose.index("\n  vllm-embed:\n")
    lightrag_block = generated_compose[lightrag_start:embed_start]
    assert "    depends_on:" in generated_compose
    assert "    depends_on:" in lightrag_block
    for service_name in (
        "postgres",
        "neo4j",
        "mongodb",
        "redis",
        "milvus",
        "qdrant",
        "memgraph",
        "vllm-embed",
        "vllm-rerank",
    ):
        assert (
            f"""      {service_name}:
        condition: service_healthy"""
            in lightrag_block
        )
    assert generated_compose.count("    healthcheck:") == 10
    assert "  milvus-etcd:" in generated_compose
    assert "  milvus-minio:" in generated_compose
    assert (
        """      milvus-etcd:
        condition: service_healthy"""
        in generated_compose
    )
    assert (
        """      milvus-minio:
        condition: service_healthy"""
        in generated_compose
    )


def test_generate_docker_compose_preserves_user_depends_on_and_removes_stale_managed_entries(
    tmp_path: Path,
) -> None:
    """Compose regeneration should preserve user dependencies while refreshing wizard-managed ones."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    depends_on:",
            "      sidecar:",
            "        condition: service_started",
            "      postgres:",
            "        condition: service_started",
            "      vllm-embed:",
            "        condition: service_healthy",
            "  sidecar:",
            "    image: busybox",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

add_docker_service postgres
add_docker_service redis

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert (
        """      sidecar:
        condition: service_started"""
        in generated_compose
    )
    assert (
        """      postgres:
        condition: service_healthy"""
        in generated_compose
    )
    assert (
        """      redis:
        condition: service_healthy"""
        in generated_compose
    )
    assert (
        """      vllm-embed:
        condition: service_healthy"""
        not in generated_compose
    )


def test_generate_docker_compose_repairs_misplaced_lightrag_depends_on_from_existing_output(
    tmp_path: Path,
) -> None:
    """Regeneration should move stale lightrag depends_on content back onto the lightrag service."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    environment:",
            "  vllm-rerank:",
            "    image: example/vllm:test",
            "    restart: unless-stopped",
            "    depends_on:",
            "      my-service:",
            "        condition: service_healthy",
            "volumes:",
            "  vllm_rerank_cache:",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

add_docker_service vllm-rerank

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    lightrag_start = generated_compose.index("  lightrag:\n")
    rerank_start = generated_compose.index("\n  vllm-rerank:\n")
    lightrag_block = generated_compose[lightrag_start:rerank_start]
    rerank_block = generated_compose[rerank_start:]
    assert "    depends_on:" in lightrag_block
    assert (
        """      my-service:
        condition: service_healthy"""
        in lightrag_block
    )
    assert (
        """      vllm-rerank:
        condition: service_healthy"""
        in lightrag_block
    )
    assert "    depends_on:" not in rerank_block
    assert generated_compose.count("\n  vllm-rerank:\n") == 1


def test_generate_docker_compose_normalizes_lightrag_restart_policy_from_existing_output(
    tmp_path: Path,
) -> None:
    """Regeneration should replace legacy lightrag restart with deploy.restart_policy."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    restart: unless-stopped",
            "    extra_hosts:",
            '      - "host.docker.internal:host-gateway"',
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    lightrag_start = generated_compose.index("  lightrag:\n")
    lightrag_block = generated_compose[lightrag_start:]
    assert "    restart: unless-stopped" not in lightrag_block
    assert "    deploy:\n" in lightrag_block
    assert "      restart_policy:\n" in lightrag_block
    assert "        condition: on-failure\n" in lightrag_block
    assert "        max_attempts: 10\n" in lightrag_block


def test_generate_docker_compose_normalizes_lightrag_restart_policy_without_blank_line_before_deploy(
    tmp_path: Path,
) -> None:
    """Regeneration should move the separator blank line after deploy, not before it."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    restart: unless-stopped",
            "",
            "  sidecar:",
            "    image: busybox",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert (
        """    image: example/lightrag:test

    deploy:
"""
        not in generated_compose
    )
    assert (
        """    image: example/lightrag:test
    deploy:
"""
        in generated_compose
    )
    assert "        max_attempts: 10\n\n  sidecar:\n" in generated_compose


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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[POSTGRES_USER]="lightrag"
ENV_VALUES[POSTGRES_PASSWORD]="secret"
ENV_VALUES[POSTGRES_DATABASE]="lightrag"
add_docker_service "postgres"

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "  postgres:" in result
    assert "\n\nnetworks:\n" in result
    assert result.index("\n  postgres:") < result.index("\nnetworks:\n")
    assert "  appnet:" in result


def test_generate_docker_compose_cleans_marker_and_blank_lines_when_only_lightrag_remains(
    tmp_path: Path,
) -> None:
    """Regeneration should not leave a managed-services marker or stacked blank lines behind."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    depends_on:",
            "      vllm-embed:",
            "        condition: service_healthy",
            "      vllm-rerank:",
            "        condition: service_healthy",
            "",
            "  vllm-embed:",
            "    image: example/vllm:embed",
            "",
            "  vllm-rerank:",
            "    image: example/vllm:rerank",
            "",
            "",
            "",
            "# __WIZARD_MANAGED_SERVICES__",
            "networks:",
            "  appnet:",
            "    driver: bridge",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "  vllm-embed:" not in result
    assert "  vllm-rerank:" not in result
    assert "__WIZARD_MANAGED_SERVICES__" not in result
    assert "depends_on:" not in result
    assert "        max_attempts: 10\n\nnetworks:\n" in result


def test_generate_docker_compose_keeps_blank_line_between_managed_service_and_top_level_sections(
    tmp_path: Path,
) -> None:
    """Managed service blocks should stay visually separated from following top-level sections."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "networks:",
            "  web_network:",
            "    driver: bridge",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

add_docker_service "vllm-embed"

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "  vllm-embed:" in result
    assert (
        """        max_attempts: 10
    depends_on:
"""
        in result
    )
    assert "    restart: unless-stopped\n\nnetworks:\n" in result


def test_generate_docker_compose_keeps_single_blank_line_before_generated_volumes(
    tmp_path: Path,
) -> None:
    """Generated top-level volumes should be separated from prior sections by one blank line."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "networks:",
            "  web_network:",
            "    driver: bridge",
            "",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

add_docker_service "vllm-embed"

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "\n\nvolumes:\n" in result
    assert "\n\n\nvolumes:\n" not in result


def test_generate_env_file_comments_out_later_duplicate_active_keys(
    tmp_path: Path,
) -> None:
    """Commented example keys should not be overridden by later active defaults."""
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[EMBEDDING_BINDING]="ollama"
ENV_VALUES[EMBEDDING_MODEL]="bge-m3:latest"
ENV_VALUES[EMBEDDING_DIM]="1024"
ENV_VALUES[EMBEDDING_BINDING_HOST]="http://localhost:11434"

generate_env_file "{REPO_ROOT}/env.example" "$REPO_ROOT/.env\"
""")
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


def test_generate_env_file_preserves_custom_variables_not_declared_in_template(
    tmp_path: Path,
) -> None:
    """Reruns should keep custom `.env` variables that are not declared in env.example."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            "",
            "# Custom integration settings",
            "EXTRA_API_BASE='https://example.com/api'",
            "# EXTRA_API_TOKEN=secret",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"

generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "HOST=0.0.0.0" in generated_env
    assert "PORT=9621" in generated_env
    assert PRESERVED_HEADER in generated_env
    assert "# Custom integration settings" not in generated_env
    assert "EXTRA_API_BASE='https://example.com/api'" in generated_env
    assert "# EXTRA_API_TOKEN=secret" in generated_env


def test_generate_env_file_keeps_preserved_section_idempotent_across_reruns(
    tmp_path: Path,
) -> None:
    """Repeated reruns should keep a single preserved marker and its leading blank line."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            "",
            PRESERVED_HEADER,
            "",
            "# Custom integration settings",
            "EXTRA_API_BASE='https://example.com/api'",
            "# EXTRA_API_TOKEN=secret",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    marker = PRESERVED_HEADER
    notice = PRESERVED_NOTICE
    marker_indexes = [idx for idx, line in enumerate(generated_lines) if line == marker]
    assert marker_indexes == [3]
    assert generated_lines[2] == ""
    assert generated_lines[4] == notice
    assert generated_lines[5] == ""
    assert generated_lines[6] == "# Custom integration settings"
    assert generated_lines[7] == "EXTRA_API_BASE='https://example.com/api'"
    assert generated_lines[8] == "# EXTRA_API_TOKEN=secret"


def test_generate_env_file_preserves_multi_line_comments_inside_preserved_section(
    tmp_path: Path,
) -> None:
    """Only comments already inside the preserved section should survive reruns."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            "",
            "# External note that should not migrate",
            PRESERVED_HEADER,
            "",
            "# Group A",
            "# Shared settings",
            "EXTRA_API_BASE='https://example.com/api'",
            "",
            "# Group B",
            "EXTRA_API_TOKEN=secret",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    marker = PRESERVED_HEADER
    notice = PRESERVED_NOTICE
    marker_index = generated_lines.index(marker)
    assert generated_lines.count(marker) == 1
    assert generated_lines.count(notice) == 1
    assert "# External note that should not migrate" not in generated_lines
    assert generated_lines[marker_index + 1] == notice
    assert generated_lines[marker_index + 2] == ""
    assert generated_lines[marker_index + 3] == "# Group A"
    assert generated_lines[marker_index + 4] == "# Shared settings"
    assert (
        generated_lines[marker_index + 5] == "EXTRA_API_BASE='https://example.com/api'"
    )
    assert generated_lines[marker_index + 6] == ""
    assert generated_lines[marker_index + 7] == "# Group B"
    assert generated_lines[marker_index + 8] == "EXTRA_API_TOKEN=secret"


def test_generate_env_file_preserves_trailing_comments_at_end_of_preserved_section(
    tmp_path: Path,
) -> None:
    """Free-form comments after the last preserved variable should survive reruns."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            "",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "",
            "EXTRA_API_BASE='https://example.com/api'",
            "# Free-form note",
            "# This should stay at EOF",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert generated_lines[-3] == "EXTRA_API_BASE='https://example.com/api'"
    assert generated_lines[-2] == "# Free-form note"
    assert generated_lines[-1] == "# This should stay at EOF"


def test_generate_env_file_appends_new_external_entries_after_existing_preserved_block(
    tmp_path: Path,
) -> None:
    """New template-external entries should be appended after the existing preserved payload."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            "EXTRA_EARLY=alpha",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "",
            "# Existing note",
            "EXTRA_EXISTING=omega",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    marker_index = generated_lines.index(PRESERVED_HEADER)
    assert generated_lines[marker_index + 1] == PRESERVED_NOTICE
    assert generated_lines[marker_index + 2] == ""
    assert generated_lines[marker_index + 3] == "# Existing note"
    assert generated_lines[marker_index + 4] == "EXTRA_EXISTING=omega"
    assert generated_lines[marker_index + 5] == "EXTRA_EARLY=alpha"


def test_generate_env_file_appends_multiple_new_external_entries_in_discovery_order(
    tmp_path: Path,
) -> None:
    """Multiple new external entries should append after preserved payload in source order."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            "EXTRA_FIRST=one",
            "# Outside comment should not migrate",
            "EXTRA_SECOND=two",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "",
            "EXTRA_EXISTING=existing",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    marker_index = generated_lines.index(PRESERVED_HEADER)
    assert "# Outside comment should not migrate" not in generated_lines
    assert generated_lines[marker_index + 3] == "EXTRA_EXISTING=existing"
    assert generated_lines[marker_index + 4] == "EXTRA_FIRST=one"
    assert generated_lines[marker_index + 5] == "EXTRA_SECOND=two"


def test_generate_env_file_keeps_commented_template_keys_inside_preserved_section(
    tmp_path: Path,
) -> None:
    """Commented env vars already placed in preserved should survive even if the template declares them."""
    write_text_lines(
        tmp_path / "env.example",
        ["HOST=0.0.0.0", "# PORT=9621", "# ENTITY_EXTRACTION_USE_JSON=true"],
    )
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "",
            "# ENTITY_EXTRACTION_USE_JSON=true",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    marker_index = generated_lines.index(PRESERVED_HEADER)
    assert generated_lines.count("# ENTITY_EXTRACTION_USE_JSON=true") == 2
    assert generated_lines[marker_index + 3] == "# ENTITY_EXTRACTION_USE_JSON=true"


def test_generate_env_file_recognizes_lowercase_extra_variables(tmp_path: Path) -> None:
    """Lowercase template-external variables should be preserved like uppercase ones."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(tmp_path / ".env", ["HOST=127.0.0.1", "workspace_name=demo"])
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert PRESERVED_HEADER in generated_lines
    assert "workspace_name=demo" in generated_lines


def test_generate_env_file_recognizes_lowercase_commented_extra_variables(
    tmp_path: Path,
) -> None:
    """Lowercase commented env vars should create and survive in the preserved section."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(tmp_path / ".env", ["HOST=127.0.0.1", "# workspace_name=demo"])
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert PRESERVED_HEADER in generated_lines
    assert "# workspace_name=demo" in generated_lines


def test_generate_env_file_uses_template_preserved_block_when_env_missing(
    tmp_path: Path,
) -> None:
    """Missing `.env` should still produce the preserved block from env.example."""
    write_text_lines(
        tmp_path / "env.example",
        [
            "HOST=0.0.0.0",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "### Template preserved comment",
            "# template_example=true",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[HOST]="0.0.0.0"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert PRESERVED_HEADER in generated_lines
    assert PRESERVED_NOTICE in generated_lines
    assert generated_lines.count(PRESERVED_HEADER) == 1
    assert generated_lines.count(PRESERVED_NOTICE) == 1
    assert "### Template preserved comment" in generated_lines
    assert "# template_example=true" in generated_lines


def test_generate_env_file_keeps_template_separator_adjacent_to_preserved_header(
    tmp_path: Path,
) -> None:
    """Injected template preserved blocks should not add a blank line after the copied separator."""
    write_text_lines(
        tmp_path / "env.example",
        [
            "HOST=0.0.0.0",
            "##########################################################################",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "### Template preserved comment",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[HOST]="0.0.0.0"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    header_index = generated_lines.index(PRESERVED_HEADER)
    assert (
        generated_lines[header_index - 1]
        == "##########################################################################"
    )


def test_generate_env_file_does_not_inject_template_payload_when_old_preserved_exists(
    tmp_path: Path,
) -> None:
    """Existing preserved blocks should stay authoritative over template preserved payload."""
    write_text_lines(
        tmp_path / "env.example",
        [
            "HOST=0.0.0.0",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "### Template preserved comment",
            "# template_example=true",
        ],
    )
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            "",
            PRESERVED_HEADER,
            "",
            "# Existing preserved comment",
            "EXTRA_OLD=1",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert PRESERVED_HEADER in generated_lines
    assert PRESERVED_NOTICE in generated_lines
    assert "### Template preserved comment" not in generated_lines
    assert "# template_example=true" not in generated_lines
    assert "# Existing preserved comment" in generated_lines
    assert "EXTRA_OLD=1" in generated_lines


def test_generate_env_file_keeps_old_preserved_lines_even_when_they_match_template(
    tmp_path: Path,
) -> None:
    """Old preserved content should not be removed just because it matches env.example."""
    write_text_lines(
        tmp_path / "env.example",
        [
            "HOST=0.0.0.0",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "### Template preserved comment",
            "# template_example=true",
        ],
    )
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            "",
            PRESERVED_HEADER,
            "### Template preserved comment",
            "# template_example=true",
            "EXTRA_OLD=1",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert PRESERVED_HEADER in generated_lines
    assert PRESERVED_NOTICE in generated_lines
    assert "### Template preserved comment" in generated_lines
    assert "# template_example=true" in generated_lines
    assert "EXTRA_OLD=1" in generated_lines


def test_generate_env_file_preserves_comments_before_active_template_keys_in_preserved(
    tmp_path: Path,
) -> None:
    """Comments in preserved should survive even when followed by active template-managed keys."""
    write_text_lines(tmp_path / "env.example", ["HOST=0.0.0.0", "# PORT=9621"])
    write_text_lines(
        tmp_path / ".env",
        [
            "HOST=127.0.0.1",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "",
            "# Preserved note before active template key",
            "# Another note",
            "PORT=9999",
            "EXTRA_AFTER=1",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
ENV_VALUES[PORT]="9621"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    marker_index = generated_lines.index(PRESERVED_HEADER)
    assert (
        generated_lines[marker_index + 3]
        == "# Preserved note before active template key"
    )
    assert generated_lines[marker_index + 4] == "# Another note"
    assert "PORT=9999" not in generated_lines[marker_index + 1 :]
    assert "EXTRA_AFTER=1" in generated_lines


def test_generate_env_file_appends_extra_variables_after_template_preserved_block(
    tmp_path: Path,
) -> None:
    """Extras from old `.env` should append after the template preserved block when none existed before."""
    write_text_lines(
        tmp_path / "env.example",
        [
            "HOST=0.0.0.0",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "### Template preserved comment",
            "# template_example=true",
        ],
    )
    write_text_lines(tmp_path / ".env", ["HOST=127.0.0.1", "EXTRA_NEW=1"])
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert generated_lines[-3] == "### Template preserved comment"
    assert generated_lines[-2] == "# template_example=true"
    assert generated_lines[-1] == "EXTRA_NEW=1"


def test_generate_env_file_appends_commented_env_vars_after_template_preserved_block(
    tmp_path: Path,
) -> None:
    """Commented env vars from old `.env` should append after the template preserved block when none existed before."""
    write_text_lines(
        tmp_path / "env.example",
        [
            "HOST=0.0.0.0",
            PRESERVED_HEADER,
            PRESERVED_NOTICE,
            "### Template preserved comment",
            "# template_example=true",
        ],
    )
    write_text_lines(tmp_path / ".env", ["HOST=127.0.0.1", "# EXTRA_COMMENTED=1"])
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

load_env_file "$REPO_ROOT/.env"
ENV_VALUES[HOST]="0.0.0.0"
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert generated_lines[-3] == "### Template preserved comment"
    assert generated_lines[-2] == "# template_example=true"
    assert generated_lines[-1] == "# EXTRA_COMMENTED=1"


def test_generate_env_file_round_trips_dollar_signs_in_single_quoted_values(
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
    output = run_bash(f"""
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
printf 'WEBUI_DESCRIPTION=%s\\n' "${{ENV_VALUES[WEBUI_DESCRIPTION]}}\"
""")
    values = parse_lines(output)
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "TOKEN_SECRET='abc$HOME'" in generated_env
    assert "LIGHTRAG_API_KEY='plain$token'" in generated_env
    assert "WEBUI_DESCRIPTION='value with \"$PATH\" and $HOME'" in generated_env
    assert values["TOKEN_SECRET"] == "abc$HOME"
    assert values["LIGHTRAG_API_KEY"] == "plain$token"
    assert values["WEBUI_DESCRIPTION"] == 'value with "$PATH" and $HOME'


def test_generate_env_file_avoids_double_quotes_for_compose_sensitive_strings(
    tmp_path: Path,
) -> None:
    """Setup output should avoid double quotes for affected string variables."""
    env_example = tmp_path / "env.example"
    env_example.write_text(
        "\n".join(
            [
                "WEBUI_TITLE='My Graph KB'",
                "WEBUI_DESCRIPTION='Simple and Fast Graph Based RAG System'",
                "# AUTH_ACCOUNTS='admin:admin123,user1:{bcrypt}$2b$12$hash'",
                "# LANGFUSE_SECRET_KEY=''",
                "# LANGFUSE_PUBLIC_KEY=''",
                "# LANGFUSE_HOST='https://cloud.langfuse.com'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[WEBUI_TITLE]='My Graph KB'
ENV_VALUES[WEBUI_DESCRIPTION]='Simple and Fast Graph Based RAG System'
ENV_VALUES[AUTH_ACCOUNTS]='admin:admin123,user1:pa$$word'
ENV_VALUES[LANGFUSE_SECRET_KEY]='sk-lf-secret'
ENV_VALUES[LANGFUSE_PUBLIC_KEY]='pk-lf-public'
ENV_VALUES[LANGFUSE_HOST]='https://langfuse.example'

generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""")
    generated_lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert "WEBUI_TITLE='My Graph KB'" in generated_lines
    assert (
        "WEBUI_DESCRIPTION='Simple and Fast Graph Based RAG System'" in generated_lines
    )
    assert "AUTH_ACCOUNTS='admin:admin123,user1:pa$$word'" in generated_lines
    assert "LANGFUSE_SECRET_KEY=sk-lf-secret" in generated_lines
    assert "LANGFUSE_PUBLIC_KEY=pk-lf-public" in generated_lines
    assert "LANGFUSE_HOST=https://langfuse.example" in generated_lines
    assert not any(
        line.startswith('WEBUI_TITLE="')
        or line.startswith('WEBUI_DESCRIPTION="')
        or line.startswith('AUTH_ACCOUNTS="')
        or line.startswith('LANGFUSE_SECRET_KEY="')
        or line.startswith('LANGFUSE_PUBLIC_KEY="')
        or line.startswith('LANGFUSE_HOST="')
        for line in generated_lines
    )


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
    run_bash(f"""
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

generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml\"
""")
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
    assert (
        "NEO4J_AUTH: ${NEO4J_USERNAME:?missing}/${NEO4J_PASSWORD:?missing}"
        in generated_compose
    )
    assert 'NEO4J_dbms_default__database: "graph$$DB"' in generated_compose
    assert 'MINIO_ACCESS_KEY_ID: "${MINIO_ACCESS_KEY_ID:?missing}"' in generated_compose
    assert (
        'MINIO_SECRET_ACCESS_KEY: "${MINIO_SECRET_ACCESS_KEY:?missing}"'
        in generated_compose
    )
    assert 'MINIO_ROOT_USER: "${MINIO_ACCESS_KEY_ID:?missing}"' in generated_compose
    assert (
        'MINIO_ROOT_PASSWORD: "${MINIO_SECRET_ACCESS_KEY:?missing}"'
        in generated_compose
    )
    assert "milvus-etcd" in generated_compose
    assert "milvus-minio" in generated_compose


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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
add_docker_service postgres
add_docker_service vllm-embed
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
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
        tmp_path / ".env", ["LLM_BINDING=openai", "EMBEDDING_BINDING=openai"]
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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
add_docker_service postgres
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "  sidecar_data:" in result
    assert "\n  source:\n" not in result


def test_generate_docker_compose_includes_all_atlas_local_mongodb_volumes(
    tmp_path: Path,
) -> None:
    """MongoDB Atlas Local should emit data, config, and mongot named volumes."""
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
add_docker_service mongodb
generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "hostname: mongodb" in result
    assert "image: mongodb/mongodb-atlas-local:" in result
    assert "mongo_data:/data/db" in result
    assert "mongo_config_data:/data/configdb" in result
    assert "mongo_mongot_data:/data/mongot" in result
    assert "healthcheck:" not in result
    assert (
        "\nvolumes:\n  mongo_data:\n  mongo_config_data:\n  mongo_mongot_data:\n"
        in result
    )


def test_generate_docker_compose_injects_server_host_and_port_overrides(
    tmp_path: Path,
) -> None:
    """Generated compose should preserve variable-based host publishing and fix container bind values."""
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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[HOST]="localhost"
ENV_VALUES[PORT]="8080"

prepare_compose_runtime_overrides
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )
    assert 'HOST: "0.0.0.0"' in generated_compose
    assert 'PORT: "9621"' in generated_compose
    assert '      - "${HOST:-0.0.0.0}:${PORT:-9621}:9621"' in generated_compose


def test_generate_docker_compose_injects_env_overrides_into_lightrag_not_after_managed_services(
    tmp_path: Path,
) -> None:
    """Env overrides must appear inside the lightrag environment block, not after managed services.

    When the base compose has a top-level volumes: section, the strip pass inserts a
    __WIZARD_MANAGED_SERVICES__ marker at the point where volumes: begins.  Before the
    fix the environment injector would miss that marker (column-0 comment) as an
    end-of-environment boundary and append overrides after it — which placed them outside
    the lightrag service once postgres/neo4j were merged in.
    """
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    environment:",
                "      EXISTING_KEY: existing_value",
                "    volumes:",
                "      - ./.env:/app/.env",
                "volumes:",
                "  some_volume:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[POSTGRES_USER]="lightrag"
ENV_VALUES[POSTGRES_PASSWORD]="secret"
ENV_VALUES[POSTGRES_DATABASE]="lightrag"
add_docker_service "postgres"
set_compose_override "LLM_BINDING_HOST" "http://host.docker.internal:11434"

generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml\"
""")
    result = (tmp_path / "docker-compose.generated.yml").read_text(encoding="utf-8")
    lightrag_pos = result.index("  lightrag:")
    postgres_pos = result.index("  postgres:")
    override_pos = result.index('LLM_BINDING_HOST: "http://host.docker.internal:11434"')
    assert lightrag_pos < override_pos < postgres_pos


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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[VLLM_RERANK_DEVICE]="cuda"
ENV_VALUES[CUDA_VISIBLE_DEVICES]="0"
add_docker_service "vllm-rerank"

generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml\"
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )
    assert "CUDA_VISIBLE_DEVICES=0" in generated_env
    assert "NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-all}" in generated_compose
    assert (
        """      vllm-rerank:
        condition: service_healthy"""
        in generated_compose
    )
    assert "    healthcheck:" in generated_compose
    assert "VLLM_RERANK_PORT:-8000" in generated_compose
    assert 'grep -q ":$${PORT_HEX} "' in generated_compose


@pytest.mark.parametrize(
    ("device", "expected_image"),
    [
        ("cpu", "image: milvusdb/milvus:v2.6.11"),
        ("cuda", "image: milvusdb/milvus:v2.6.11-gpu"),
    ],
)
def test_generate_docker_compose_selects_milvus_template_from_device(
    tmp_path: Path, device: str, expected_image: str
) -> None:
    """Milvus compose generation should switch templates based on MILVUS_DEVICE."""
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / "docker-compose.yml",
        ["services:", "  lightrag:", "    image: example/lightrag:test"],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[MILVUS_DEVICE]="{device}"
add_docker_service milvus

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml\"
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert expected_image in generated_compose


def test_generate_docker_compose_omits_config_ini_mount_from_base_template(
    tmp_path: Path,
) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    volumes:",
                "      - ./data/rag_storage:/app/data/rag_storage",
                "      - ./data/inputs:/app/data/inputs",
                "      - ./.env:/app/.env",
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

generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )

    assert "./config.ini:/app/config.ini" not in generated_compose
    assert "./data/rag_storage:/app/data/rag_storage" in generated_compose
    assert "./data/inputs:/app/data/inputs" in generated_compose
    assert "./.env:/app/.env" in generated_compose


def test_generate_docker_compose_preserves_existing_config_ini_mount(
    tmp_path: Path,
) -> None:
    compose_file = tmp_path / "docker-compose.final.yml"
    compose_file.write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    volumes:",
                "      - ./data/rag_storage:/app/data/rag_storage",
                "      - ./data/inputs:/app/data/inputs",
                "      - ./config.ini:/app/config.ini",
                "      - ./.env:/app/.env",
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

generate_docker_compose "$REPO_ROOT/docker-compose.final.yml"
"""
    )

    generated_compose = compose_file.read_text(encoding="utf-8")

    assert "./config.ini:/app/config.ini" in generated_compose
    assert "./data/rag_storage:/app/data/rag_storage" in generated_compose
    assert "./data/inputs:/app/data/inputs" in generated_compose
    assert "./.env:/app/.env" in generated_compose
