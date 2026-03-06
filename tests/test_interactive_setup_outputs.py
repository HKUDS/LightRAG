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
        ["bash", "-lc", script],
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


def test_prepare_compose_runtime_overrides_keeps_env_unchanged() -> None:
    """Loopback endpoints should be rewritten only for compose overrides."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LLM_BINDING_HOST]="http://localhost:11434"
ENV_VALUES[EMBEDDING_BINDING_HOST]="http://127.0.0.1:11434"
ENV_VALUES[RERANK_BINDING_HOST]="http://localhost:8000/v1/rerank"

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
    assert values["ENV_RERANK"] == "http://localhost:8000/v1/rerank"
    assert values["COMPOSE_LLM"] == "http://host.docker.internal:11434"
    assert values["COMPOSE_EMBEDDING"] == "http://host.docker.internal:11434"
    assert values["COMPOSE_RERANK"] == "http://host.docker.internal:8000/v1/rerank"


def test_generate_files_keep_host_env_values_and_inject_compose_overrides(
    tmp_path: Path,
) -> None:
    """Generated `.env` should keep host paths while compose gets container overrides."""

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
ENV_VALUES[RERANK_BINDING_HOST]="http://localhost:8000/v1/rerank"
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
    assert "RERANK_BINDING_HOST=http://localhost:8000/v1/rerank" in generated_env

    assert 'SSL_CERTFILE: "/app/data/certs/cert.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key.pem"' in generated_compose
    assert 'LLM_BINDING_HOST: "http://host.docker.internal:11434"' in generated_compose
    assert (
        'EMBEDDING_BINDING_HOST: "http://host.docker.internal:11434"'
        in generated_compose
    )
    assert (
        'RERANK_BINDING_HOST: "http://host.docker.internal:8000/v1/rerank"'
        in generated_compose
    )
    assert './data/certs/cert.pem:/app/data/certs/cert.pem:ro' in generated_compose
    assert './data/certs/key.pem:/app/data/certs/key.pem:ro' in generated_compose


def test_existing_ssl_env_keeps_compose_mount_overrides(tmp_path: Path) -> None:
    """Existing SSL-enabled `.env` files should keep compose cert mounts working."""

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
    assert './data/certs/cert.pem:/app/data/certs/cert.pem:ro' in generated_compose
    assert './data/certs/key.pem:/app/data/certs/key.pem:ro' in generated_compose


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


def test_generate_docker_compose_skips_environment_block_without_overrides(
    tmp_path: Path,
) -> None:
    """Compose generation should not add a `lightrag.environment` block when unused."""

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
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )
    assert "environment:" not in generated_compose


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
ENV_VALUES[LIGHTRAG_API_KEY]='${{TOKEN}}'
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

    assert 'TOKEN_SECRET="abc\\$HOME"' in generated_env
    assert 'LIGHTRAG_API_KEY="\\${TOKEN}"' in generated_env
    assert 'WEBUI_DESCRIPTION="value with \\"\\$PATH\\" and \\$HOME"' in generated_env
    assert values["TOKEN_SECRET"] == "abc$HOME"
    assert values["LIGHTRAG_API_KEY"] == "${TOKEN}"
    assert values["WEBUI_DESCRIPTION"] == 'value with "$PATH" and $HOME'


def test_generate_docker_compose_escapes_dollar_signs_in_overrides(
    tmp_path: Path,
) -> None:
    """Compose overrides should escape `$` so Docker keeps literal credentials."""

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

ENV_VALUES[MONGO_URI]='mongodb://user:p$HOME@localhost:27017/'

prepare_compose_runtime_overrides
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


def test_load_preset_preserves_existing_env_values() -> None:
    """Preset defaults should fill missing keys without clobbering current config."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LIGHTRAG_KV_STORAGE]="RedisKVStorage"
ENV_VALUES[LLM_BINDING]="ollama"
ENV_VALUES[POSTGRES_IMAGE]="custom/postgres:17"

load_preset development

printf 'LIGHTRAG_KV_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_KV_STORAGE]}}"
printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
printf 'POSTGRES_IMAGE=%s\\n' "${{ENV_VALUES[POSTGRES_IMAGE]}}"
printf 'EMBEDDING_MODEL=%s\\n' "${{ENV_VALUES[EMBEDDING_MODEL]}}"
"""
    )
    values = parse_lines(output)

    assert values["LIGHTRAG_KV_STORAGE"] == "RedisKVStorage"
    assert values["LLM_BINDING"] == "ollama"
    assert values["POSTGRES_IMAGE"] == "custom/postgres:17"
    assert values["EMBEDDING_MODEL"] == "text-embedding-3-large"


def test_select_storage_backends_prefers_existing_env_values() -> None:
    """Storage selection defaults should reuse values loaded from an existing `.env`."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LIGHTRAG_KV_STORAGE]="RedisKVStorage"
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="QdrantVectorDBStorage"
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="Neo4JStorage"
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="RedisDocStatusStorage"

prompt_choice() {{
  printf '%s' "$2"
}}

select_storage_backends production

printf 'LIGHTRAG_KV_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_KV_STORAGE]}}"
printf 'LIGHTRAG_VECTOR_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]}}"
printf 'LIGHTRAG_GRAPH_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]}}"
printf 'LIGHTRAG_DOC_STATUS_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]}}"
"""
    )
    values = parse_lines(output)

    assert values["LIGHTRAG_KV_STORAGE"] == "RedisKVStorage"
    assert values["LIGHTRAG_VECTOR_STORAGE"] == "QdrantVectorDBStorage"
    assert values["LIGHTRAG_GRAPH_STORAGE"] == "Neo4JStorage"
    assert values["LIGHTRAG_DOC_STATUS_STORAGE"] == "RedisDocStatusStorage"


def test_quick_start_flow_preserves_existing_non_preset_values(
    tmp_path: Path,
) -> None:
    """Quick start should load `.env` defaults without overwriting unrelated settings."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HOST=127.0.0.1",
                "PORT=9999",
                "WEBUI_TITLE=Existing Title",
                "WEBUI_DESCRIPTION=Existing Description",
                "RERANK_BINDING=jina",
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

prompt_secret_until_valid_with_default() {{
  printf '%s' "$2"
}}

finalize_setup() {{
  printf 'HOST=%s\\n' "${{ENV_VALUES[HOST]}}"
  printf 'PORT=%s\\n' "${{ENV_VALUES[PORT]}}"
  printf 'WEBUI_TITLE=%s\\n' "${{ENV_VALUES[WEBUI_TITLE]}}"
  printf 'WEBUI_DESCRIPTION=%s\\n' "${{ENV_VALUES[WEBUI_DESCRIPTION]}}"
  printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}"
  printf 'LIGHTRAG_KV_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_KV_STORAGE]}}"
  printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
  printf 'LLM_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[LLM_BINDING_API_KEY]}}"
  printf 'EMBEDDING_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_API_KEY]}}"
}}

quick_start_flow
"""
    )
    values = parse_lines(output)

    assert values["HOST"] == "127.0.0.1"
    assert values["PORT"] == "9999"
    assert values["WEBUI_TITLE"] == "Existing Title"
    assert values["WEBUI_DESCRIPTION"] == "Existing Description"
    assert values["RERANK_BINDING"] == "jina"
    assert values["LIGHTRAG_KV_STORAGE"] == "JsonKVStorage"
    assert values["LLM_BINDING"] == "openai"
    assert values["LLM_BINDING_API_KEY"] == "sk-existing"
    assert values["EMBEDDING_BINDING_API_KEY"] == "sk-existing"


def test_quick_start_flow_resets_existing_provider_settings_to_development_preset(
    tmp_path: Path,
) -> None:
    """Quick start should replace prior provider bindings with the OpenAI preset."""

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
                "LLM_BINDING_API_KEY=sk-existing",
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

prompt_secret_until_valid_with_default() {{
  printf '%s' "$2"
}}

finalize_setup() {{
  printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
  printf 'LLM_MODEL=%s\\n' "${{ENV_VALUES[LLM_MODEL]}}"
  printf 'LLM_BINDING_HOST=%s\\n' "${{ENV_VALUES[LLM_BINDING_HOST]}}"
  printf 'EMBEDDING_BINDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING]}}"
  printf 'EMBEDDING_MODEL=%s\\n' "${{ENV_VALUES[EMBEDDING_MODEL]}}"
  printf 'EMBEDDING_DIM=%s\\n' "${{ENV_VALUES[EMBEDDING_DIM]}}"
  printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
  printf 'LLM_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[LLM_BINDING_API_KEY]}}"
  printf 'EMBEDDING_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_API_KEY]}}"
}}

quick_start_flow
"""
    )
    values = parse_lines(output)

    assert values["LLM_BINDING"] == "openai"
    assert values["LLM_MODEL"] == "gpt-4o"
    assert values["LLM_BINDING_HOST"] == "https://api.openai.com/v1"
    assert values["EMBEDDING_BINDING"] == "openai"
    assert values["EMBEDDING_MODEL"] == "text-embedding-3-large"
    assert values["EMBEDDING_DIM"] == "3072"
    assert values["EMBEDDING_BINDING_HOST"] == "https://api.openai.com/v1"
    assert values["LLM_BINDING_API_KEY"] == "sk-existing"
    assert values["EMBEDDING_BINDING_API_KEY"] == "sk-existing"


def test_production_flow_resets_existing_storage_settings_to_production_preset(
    tmp_path: Path,
) -> None:
    """Production flow should preselect production storage backends on reruns."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
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

prompt_choice() {{
  printf '%s' "$2"
}}
confirm_default_yes() {{
  return 1
}}
confirm() {{
  return 1
}}
prompt_with_default() {{
  printf '%s' "$2"
}}
prompt_until_valid() {{
  printf '%s' "$2"
}}
prompt_secret_with_default() {{
  printf '%s' "$2"
}}
prompt_secret_until_valid_with_default() {{
  printf '%s' "$2"
}}
finalize_setup() {{
  printf 'LIGHTRAG_KV_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_KV_STORAGE]}}"
  printf 'LIGHTRAG_VECTOR_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]}}"
  printf 'LIGHTRAG_GRAPH_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]}}"
  printf 'LIGHTRAG_DOC_STATUS_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]}}"
  printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
  printf 'EMBEDDING_BINDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING]}}"
}}

production_flow
"""
    )
    values = parse_lines(output)

    assert values["LIGHTRAG_KV_STORAGE"] == "PGKVStorage"
    assert values["LIGHTRAG_VECTOR_STORAGE"] == "MilvusVectorDBStorage"
    assert values["LIGHTRAG_GRAPH_STORAGE"] == "Neo4JStorage"
    assert values["LIGHTRAG_DOC_STATUS_STORAGE"] == "PGDocStatusStorage"
    assert values["LLM_BINDING"] == "ollama"
    assert values["EMBEDDING_BINDING"] == "ollama"


def test_collect_milvus_config_defaults_to_existing_database_name() -> None:
    """Milvus database prompt should preserve the documented default database."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

confirm() {{ return 1; }}
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
    values = parse_lines(output)

    assert values["MILVUS_DB_NAME"] == "lightrag"


def test_prepare_compose_runtime_overrides_rewrites_host_database_loopback() -> None:
    """Host-run databases on loopback should stay reachable from the container."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[POSTGRES_HOST]="127.0.0.1"
ENV_VALUES[REDIS_URI]="redis://localhost:6379"
ENV_VALUES[MONGO_URI]="mongodb://127.0.0.1:27017/"
ENV_VALUES[NEO4J_URI]="neo4j://localhost:7687"
ENV_VALUES[MILVUS_URI]="http://localhost:19530"
ENV_VALUES[QDRANT_URL]="http://127.0.0.1:6333"
ENV_VALUES[MEMGRAPH_URI]="bolt://localhost:7687"

prepare_compose_runtime_overrides

printf 'POSTGRES_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[POSTGRES_HOST]}}"
printf 'REDIS_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[REDIS_URI]}}"
printf 'MONGO_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[MONGO_URI]}}"
printf 'NEO4J_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[NEO4J_URI]}}"
printf 'MILVUS_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[MILVUS_URI]}}"
printf 'QDRANT_URL=%s\\n' "${{COMPOSE_ENV_OVERRIDES[QDRANT_URL]}}"
printf 'MEMGRAPH_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[MEMGRAPH_URI]}}"
"""
    )
    values = parse_lines(output)

    assert values["POSTGRES_HOST"] == "host.docker.internal"
    assert values["REDIS_URI"] == "redis://host.docker.internal:6379"
    assert values["MONGO_URI"] == "mongodb://host.docker.internal:27017/"
    assert values["NEO4J_URI"] == "neo4j://host.docker.internal:7687"
    assert values["MILVUS_URI"] == "http://host.docker.internal:19530"
    assert values["QDRANT_URL"] == "http://host.docker.internal:6333"
    assert values["MEMGRAPH_URI"] == "bolt://host.docker.internal:7687"


def test_prepare_compose_runtime_overrides_rewrites_authenticated_loopback_uri() -> None:
    """Loopback URIs with credentials should still be rewritten for the container."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[MONGO_URI]="mongodb://root:root@localhost:27017/"

prepare_compose_runtime_overrides

printf 'MONGO_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[MONGO_URI]}}"
"""
    )
    values = parse_lines(output)

    assert values["MONGO_URI"] == "mongodb://root:root@host.docker.internal:27017/"


def test_prepare_compose_runtime_overrides_rewrites_zero_host_loopback() -> None:
    """Host-bound 0.0.0.0 endpoints should be rewritten for the container."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[POSTGRES_HOST]="0.0.0.0"
ENV_VALUES[LLM_BINDING_HOST]="http://0.0.0.0:11434"
ENV_VALUES[RERANK_BINDING_HOST]="http://0.0.0.0:8000/v1/rerank"

prepare_compose_runtime_overrides

printf 'POSTGRES_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[POSTGRES_HOST]}}"
printf 'LLM_BINDING_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[LLM_BINDING_HOST]}}"
printf 'RERANK_BINDING_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[RERANK_BINDING_HOST]}}"
"""
    )
    values = parse_lines(output)

    assert values["POSTGRES_HOST"] == "host.docker.internal"
    assert values["LLM_BINDING_HOST"] == "http://host.docker.internal:11434"
    assert values["RERANK_BINDING_HOST"] == "http://host.docker.internal:8000/v1/rerank"


def test_prepare_compose_runtime_overrides_aligns_server_binding_for_container() -> None:
    """Container runtime should bind the API to a reachable host/port combination."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[HOST]="127.0.0.1"
ENV_VALUES[PORT]="8080"

prepare_compose_runtime_overrides

printf 'HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[HOST]}}"
printf 'PORT=%s\\n' "${{COMPOSE_ENV_OVERRIDES[PORT]}}"
"""
    )
    values = parse_lines(output)

    assert values["HOST"] == "0.0.0.0"
    assert values["PORT"] == "9621"


def test_prepare_compose_runtime_overrides_rewrites_non_loopback_server_host() -> None:
    """Container runtime should not inherit a host-only LAN bind address."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[HOST]="192.168.1.10"
ENV_VALUES[PORT]="8080"

prepare_compose_runtime_overrides

printf 'HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[HOST]}}"
printf 'PORT=%s\\n' "${{COMPOSE_ENV_OVERRIDES[PORT]}}"
"""
    )
    values = parse_lines(output)

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
    assert '${PORT:-9621}:9621' in generated_compose


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
    assert "./data/certs/cert-server.pem:/app/data/certs/cert-server.pem:ro" in generated_compose
    assert "./data/certs/key-server.pem:/app/data/certs/key-server.pem:ro" in generated_compose


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
    assert (
        "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-${NVIDIA_VISIBLE_DEVICES:-all}}"
        in generated_compose
    )
    assert (
        "NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-all}}"
        in generated_compose
    )


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

confirm() {{ return 0; }}
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
    generated_lines = (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()

    assert values["AUTH_ACCOUNTS_SET"] == ""
    assert values["TOKEN_SECRET_SET"] == ""
    assert values["TOKEN_EXPIRE_HOURS_SET"] == ""
    assert values["LIGHTRAG_API_KEY_SET"] == ""
    assert values["WHITELIST_PATHS_SET"] == ""
    assert not any(line.startswith("AUTH_ACCOUNTS=") for line in generated_lines)
    assert not any(line.startswith("TOKEN_SECRET=") for line in generated_lines)
    assert not any(line.startswith("TOKEN_EXPIRE_HOURS=") for line in generated_lines)
    assert not any(line.startswith("LIGHTRAG_API_KEY=") for line in generated_lines)
    assert not any(line.startswith("WHITELIST_PATHS=") for line in generated_lines)
