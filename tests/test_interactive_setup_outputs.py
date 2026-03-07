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


def test_setup_script_rejects_bash_3_with_clear_message() -> None:
    """The setup entrypoint should fail cleanly on Bash versions without associative arrays."""

    bash_major = subprocess.run(
        ["/bin/bash", "-lc", 'printf "%s" "${BASH_VERSINFO[0]}"'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if bash_major.returncode != 0 or int(bash_major.stdout or "0") >= 4:
        pytest.skip("/bin/bash already provides Bash 4+ on this runner")

    result = subprocess.run(
        ["/bin/bash", str(REPO_ROOT / "scripts/setup/setup.sh"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires Bash 4 or newer" in result.stderr


def test_makefile_setup_targets_prefer_common_bash4_locations() -> None:
    """Setup targets should prefer a modern Bash even when PATH still resolves to Bash 3."""

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "SETUP_BASH ?=" in makefile
    assert "/opt/homebrew/bin/bash" in makefile
    assert "/usr/local/bin/bash" in makefile
    assert "@$(SETUP_BASH) $(SETUP_SCRIPT)" in makefile
    assert "@bash $(SETUP_SCRIPT)" not in makefile


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


def test_collect_local_service_configs_reset_remote_endpoints_on_rerun() -> None:
    """Bundled services should default back to host-local endpoints on reruns."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[POSTGRES_HOST]="db.example.com"
ENV_VALUES[POSTGRES_PORT]="6543"
ENV_VALUES[NEO4J_URI]="neo4j+s://graph.example.com"
ENV_VALUES[MONGO_URI]="mongodb://mongo.example.com:27018/"
ENV_VALUES[REDIS_URI]="redis://cache.example.com:6380/1"
ENV_VALUES[MILVUS_URI]="http://milvus.example.com:19530"
ENV_VALUES[QDRANT_URL]="http://qdrant.example.com:6333"
ENV_VALUES[MEMGRAPH_URI]="bolt://memgraph.example.com:7687"

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

collect_postgres_config yes
collect_neo4j_config yes
collect_mongodb_config yes
collect_redis_config yes
collect_milvus_config yes
collect_qdrant_config yes
collect_memgraph_config yes

printf 'POSTGRES_HOST=%s\\n' "${{ENV_VALUES[POSTGRES_HOST]}}"
printf 'POSTGRES_PORT=%s\\n' "${{ENV_VALUES[POSTGRES_PORT]}}"
printf 'NEO4J_URI=%s\\n' "${{ENV_VALUES[NEO4J_URI]}}"
printf 'MONGO_URI=%s\\n' "${{ENV_VALUES[MONGO_URI]}}"
printf 'REDIS_URI=%s\\n' "${{ENV_VALUES[REDIS_URI]}}"
printf 'MILVUS_URI=%s\\n' "${{ENV_VALUES[MILVUS_URI]}}"
printf 'QDRANT_URL=%s\\n' "${{ENV_VALUES[QDRANT_URL]}}"
printf 'MEMGRAPH_URI=%s\\n' "${{ENV_VALUES[MEMGRAPH_URI]}}"
"""
    )
    values = parse_lines(output)

    assert values["POSTGRES_HOST"] == "localhost"
    assert values["POSTGRES_PORT"] == "6543"
    assert values["NEO4J_URI"] == "neo4j://localhost:7687"
    assert values["MONGO_URI"] == "mongodb://localhost:27017/"
    assert values["REDIS_URI"] == "redis://localhost:6379/"
    assert values["MILVUS_URI"] == "http://localhost:19530"
    assert values["QDRANT_URL"] == "http://localhost:6333"
    assert values["MEMGRAPH_URI"] == "bolt://localhost:7687"


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
    assert "env_file:" not in generated_compose


def test_generate_docker_compose_removes_lightrag_env_file_to_preserve_dollar_values(
    tmp_path: Path,
) -> None:
    """Generated compose should not re-parse `.env` values through Docker's env_file."""

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
    assert "- ./.env:/app/.env" in generated_compose


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


def test_collect_llm_config_clears_stale_api_key_for_bedrock(
    tmp_path: Path,
) -> None:
    """Switching to Bedrock should remove stale OpenAI API key settings."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_BINDING=openai",
                "LLM_MODEL=gpt-4o",
                "LLM_BINDING_HOST=https://api.openai.com/v1",
                "LLM_BINDING_API_KEY=${OPENAI_API_KEY}",
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

prompt_choice() {{ printf 'aws_bedrock'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_required_secret() {{ printf 'dummy-secret'; }}
mask_sensitive_input() {{ printf ''; }}
confirm() {{ return 0; }}
confirm_default_yes() {{ return 0; }}

collect_llm_config

printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
printf 'LLM_BINDING_API_KEY_SET=%s\\n' "${{ENV_VALUES[LLM_BINDING_API_KEY]+set}}"
if validate_sensitive_env_literals; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["LLM_BINDING"] == "aws_bedrock"
    assert values["LLM_BINDING_API_KEY_SET"] == ""
    assert values["VALID"] == "yes"


def test_collect_llm_config_uses_provider_specific_defaults() -> None:
    """Fresh provider selection should not pin the OpenAI model for local backends."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_choice() {{ printf 'ollama'; }}
prompt_with_default() {{ printf '%s' "$2"; }}

collect_llm_config

printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
printf 'LLM_MODEL=%s\\n' "${{ENV_VALUES[LLM_MODEL]}}"
printf 'LLM_BINDING_HOST=%s\\n' "${{ENV_VALUES[LLM_BINDING_HOST]}}"
printf 'LLM_BINDING_API_KEY_SET=%s\\n' "${{ENV_VALUES[LLM_BINDING_API_KEY]+set}}"
"""
    )
    values = parse_lines(output)

    assert values["LLM_BINDING"] == "ollama"
    assert values["LLM_MODEL"] == "mistral-nemo:latest"
    assert values["LLM_BINDING_HOST"] == "http://localhost:11434"
    assert values["LLM_BINDING_API_KEY_SET"] == ""


def test_collect_llm_config_preserves_supported_openai_ollama_binding_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning setup should keep an existing openai-ollama selection valid."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_BINDING=openai-ollama",
                "LLM_MODEL=llama3.1:8b",
                "LLM_BINDING_HOST=http://localhost:11434/v1",
                "LLM_BINDING_API_KEY=sk-local-test-key",
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

prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_llm_config

printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
printf 'LLM_MODEL=%s\\n' "${{ENV_VALUES[LLM_MODEL]}}"
printf 'LLM_BINDING_HOST=%s\\n' "${{ENV_VALUES[LLM_BINDING_HOST]}}"
printf 'LLM_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[LLM_BINDING_API_KEY]}}"
"""
    )
    values = parse_lines(output)

    assert values["LLM_BINDING"] == "openai-ollama"
    assert values["LLM_MODEL"] == "llama3.1:8b"
    assert values["LLM_BINDING_HOST"] == "http://localhost:11434/v1"
    assert values["LLM_BINDING_API_KEY"] == "sk-local-test-key"


def test_collect_embedding_config_forces_ollama_for_openai_ollama_llm(
    tmp_path: Path,
) -> None:
    """`openai-ollama` should not preserve a conflicting embedding provider."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_BINDING=openai-ollama",
                "EMBEDDING_BINDING=openai",
                "EMBEDDING_MODEL=text-embedding-3-large",
                "EMBEDDING_DIM=3072",
                "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
                "EMBEDDING_BINDING_API_KEY=local-key",
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
    values = parse_lines(output)

    assert values["EMBEDDING_BINDING"] == "ollama"
    assert values["EMBEDDING_MODEL"] == "bge-m3:latest"
    assert values["EMBEDDING_DIM"] == "1024"
    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:11434"
    assert values["EMBEDDING_BINDING_API_KEY_SET"] == ""


def test_collect_embedding_config_clears_stale_api_key_for_bedrock(
    tmp_path: Path,
) -> None:
    """Switching embedding to Bedrock should remove stale provider API keys."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "EMBEDDING_BINDING=openai",
                "EMBEDDING_MODEL=text-embedding-3-large",
                "EMBEDDING_DIM=3072",
                "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
                "EMBEDDING_BINDING_API_KEY=${OPENAI_API_KEY}",
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

prompt_choice() {{ printf 'aws_bedrock'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_required_secret() {{ printf 'dummy-secret'; }}
mask_sensitive_input() {{ printf ''; }}
confirm() {{ return 0; }}
confirm_default_yes() {{ return 0; }}

collect_embedding_config

printf 'EMBEDDING_BINDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING]}}"
printf 'EMBEDDING_BINDING_API_KEY_SET=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_API_KEY]+set}}"
if validate_sensitive_env_literals; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["EMBEDDING_BINDING"] == "aws_bedrock"
    assert values["EMBEDDING_BINDING_API_KEY_SET"] == ""
    assert values["VALID"] == "yes"


def test_collect_llm_config_allows_bedrock_ambient_credential_chain() -> None:
    """Bedrock setup should allow IAM roles, AWS profiles, or SSO without saved keys."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_choice() {{ printf 'aws_bedrock'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_clearable_with_default() {{ printf ''; }}
prompt_required_secret() {{ return 1; }}
confirm() {{ return 1; }}
confirm_default_yes() {{ return 1; }}

collect_llm_config

printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
printf 'AWS_ACCESS_KEY_ID_SET=%s\\n' "${{ENV_VALUES[AWS_ACCESS_KEY_ID]+set}}"
printf 'AWS_SECRET_ACCESS_KEY_SET=%s\\n' "${{ENV_VALUES[AWS_SECRET_ACCESS_KEY]+set}}"
printf 'AWS_SESSION_TOKEN_SET=%s\\n' "${{ENV_VALUES[AWS_SESSION_TOKEN]+set}}"
printf 'AWS_REGION_SET=%s\\n' "${{ENV_VALUES[AWS_REGION]+set}}"
"""
    )
    values = parse_lines(output)

    assert values["LLM_BINDING"] == "aws_bedrock"
    assert values["AWS_ACCESS_KEY_ID_SET"] == ""
    assert values["AWS_SECRET_ACCESS_KEY_SET"] == ""
    assert values["AWS_SESSION_TOKEN_SET"] == ""
    assert values["AWS_REGION_SET"] == ""


def test_collect_embedding_config_uses_provider_specific_defaults() -> None:
    """Fresh embedding provider selection should use that provider's model and dimension."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_choice() {{ printf 'jina'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf 'jina-secret-key'; }}

collect_embedding_config

printf 'EMBEDDING_BINDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING]}}"
printf 'EMBEDDING_MODEL=%s\\n' "${{ENV_VALUES[EMBEDDING_MODEL]}}"
printf 'EMBEDDING_DIM=%s\\n' "${{ENV_VALUES[EMBEDDING_DIM]}}"
printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
"""
    )
    values = parse_lines(output)

    assert values["EMBEDDING_BINDING"] == "jina"
    assert values["EMBEDDING_MODEL"] == "jina-embeddings-v4"
    assert values["EMBEDDING_DIM"] == "2048"
    assert values["EMBEDDING_BINDING_HOST"] == "https://api.jina.ai/v1/embeddings"


def test_switching_both_providers_off_bedrock_clears_saved_aws_credentials(
    tmp_path: Path,
) -> None:
    """Reruns should not keep stale AWS Bedrock secrets in regenerated `.env` files."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
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
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_example = tmp_path / "env.example"
    env_example.write_text(
        "\n".join(
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
    values = parse_lines(output)
    generated_lines = (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()

    assert values["AWS_ACCESS_KEY_ID_SET"] == ""
    assert values["AWS_SECRET_ACCESS_KEY_SET"] == ""
    assert values["AWS_SESSION_TOKEN_SET"] == ""
    assert values["AWS_REGION_SET"] == ""
    assert not any(line.startswith("AWS_ACCESS_KEY_ID=") for line in generated_lines)
    assert not any(line.startswith("AWS_SECRET_ACCESS_KEY=") for line in generated_lines)
    assert not any(line.startswith("AWS_SESSION_TOKEN=") for line in generated_lines)
    assert not any(line.startswith("AWS_REGION=") for line in generated_lines)


def test_collect_embedding_config_preserves_supported_lollms_binding_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning setup should keep an existing lollms embedding binding valid."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "EMBEDDING_BINDING=lollms",
                "EMBEDDING_MODEL=lollms_embedding_model",
                "EMBEDDING_DIM=1024",
                "EMBEDDING_BINDING_HOST=http://localhost:9600",
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

prompt_with_default() {{ printf '%s' "$2"; }}

collect_embedding_config

printf 'EMBEDDING_BINDING=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING]}}"
printf 'EMBEDDING_MODEL=%s\\n' "${{ENV_VALUES[EMBEDDING_MODEL]}}"
printf 'EMBEDDING_DIM=%s\\n' "${{ENV_VALUES[EMBEDDING_DIM]}}"
printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
"""
    )
    values = parse_lines(output)

    assert values["EMBEDDING_BINDING"] == "lollms"
    assert values["EMBEDDING_MODEL"] == "lollms_embedding_model"
    assert values["EMBEDDING_DIM"] == "1024"
    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:9600"


def test_collect_rerank_config_clears_stale_api_key_when_disabled(
    tmp_path: Path,
) -> None:
    """Disabling reranking should remove stale credentials from a prior config."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "RERANK_BINDING=cohere",
                "RERANK_MODEL=rerank-v3.5",
                "RERANK_BINDING_HOST=https://api.cohere.com/v1/rerank",
                "RERANK_BINDING_API_KEY=${COHERE_API_KEY}",
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

confirm() {{ return 1; }}

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
    values = parse_lines(output)

    assert values["RERANK_BINDING"] == "null"
    assert values["RERANK_BINDING_API_KEY_SET"] == ""
    assert values["VALID"] == "yes"


def test_collect_rerank_config_vllm_resets_stale_hosted_defaults() -> None:
    """Selecting local vLLM should not keep a previously hosted rerank endpoint."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[RERANK_BINDING]="cohere"
ENV_VALUES[RERANK_MODEL]="rerank-v3.5"
ENV_VALUES[RERANK_BINDING_HOST]="https://api.cohere.com/v1/rerank"

confirm() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
prompt_choice() {{
  case "$1" in
    "Rerank provider") printf 'vllm' ;;
    "vLLM device") printf 'cpu' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}

collect_rerank_config

printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}"
printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]}}"
printf 'RERANK_MODEL=%s\\n' "${{ENV_VALUES[RERANK_MODEL]}}"
printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
printf 'COMPOSE_RERANK_BINDING_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[RERANK_BINDING_HOST]}}"
"""
    )
    values = parse_lines(output)

    assert values["RERANK_BINDING"] == "cohere"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == "vllm"
    assert values["RERANK_MODEL"] == "BAAI/bge-reranker-v2-m3"
    assert values["RERANK_BINDING_HOST"] == "http://localhost:8000/v1/rerank"
    assert (
        values["COMPOSE_RERANK_BINDING_HOST"]
        == "http://vllm-rerank:8000/v1/rerank"
    )


def test_collect_rerank_config_preserves_vllm_default_on_rerun() -> None:
    """A saved local vLLM choice should come back as the default provider on rerun."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]="vllm"
ENV_VALUES[RERANK_BINDING]="cohere"
ENV_VALUES[RERANK_MODEL]="BAAI/bge-reranker-v2-m3"
ENV_VALUES[RERANK_BINDING_HOST]="http://localhost:8000/v1/rerank"
ENV_VALUES[VLLM_RERANK_MODEL]="BAAI/bge-reranker-v2-m3"
ENV_VALUES[VLLM_RERANK_PORT]="8000"
ENV_VALUES[VLLM_RERANK_DEVICE]="cpu"
ENV_VALUES[VLLM_RERANK_DTYPE]="float32"

confirm() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}

collect_rerank_config

printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}"
printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]}}"
printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}"
printf 'COMPOSE_RERANK_BINDING_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[RERANK_BINDING_HOST]}}"
"""
    )
    values = parse_lines(output)

    assert values["RERANK_BINDING"] == "cohere"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == "vllm"
    assert values["RERANK_BINDING_HOST"] == "http://localhost:8000/v1/rerank"
    assert values["DOCKER_SERVICE"] == "vllm-rerank"
    assert (
        values["COMPOSE_RERANK_BINDING_HOST"]
        == "http://vllm-rerank:8000/v1/rerank"
    )


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


def test_generate_docker_compose_escapes_bundled_service_secrets(
    tmp_path: Path,
) -> None:
    """Bundled dependency services should keep `$`-bearing secrets literal in compose."""

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

ENV_VALUES[POSTGRES_USER]='user$ID'
ENV_VALUES[POSTGRES_PASSWORD]='pass$HOME'
ENV_VALUES[POSTGRES_DATABASE]='db$NAME'
ENV_VALUES[NEO4J_PASSWORD]='neo$PASS'
ENV_VALUES[NEO4J_DATABASE]='graph$DB'
ENV_VALUES[MINIO_ACCESS_KEY_ID]='minio$USER'
ENV_VALUES[MINIO_SECRET_ACCESS_KEY]='minio$SECRET'

add_docker_service postgres
add_docker_service neo4j
add_docker_service milvus

generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
"""
    )

    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
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


def test_select_storage_backends_allows_development_defaults_with_warnings() -> None:
    """Development defaults should be selectable even when they emit advisory warnings."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_choice() {{
  printf '%s' "$2"
}}

select_storage_backends development

printf 'LIGHTRAG_KV_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_KV_STORAGE]}}"
printf 'LIGHTRAG_VECTOR_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]}}"
printf 'LIGHTRAG_GRAPH_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]}}"
printf 'LIGHTRAG_DOC_STATUS_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]}}"
"""
    )
    values = parse_lines(output)

    assert values["LIGHTRAG_KV_STORAGE"] == "JsonKVStorage"
    assert values["LIGHTRAG_VECTOR_STORAGE"] == "NanoVectorDBStorage"
    assert values["LIGHTRAG_GRAPH_STORAGE"] == "NetworkXStorage"
    assert values["LIGHTRAG_DOC_STATUS_STORAGE"] == "JsonDocStatusStorage"


def test_quick_start_flow_preserves_safe_values_and_clears_inherited_state(
    tmp_path: Path,
) -> None:
    """Quick start should keep safe UI values while clearing deployment-shaping state."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HOST=127.0.0.1",
                "PORT=9999",
                "WEBUI_TITLE=Existing Title",
                "WEBUI_DESCRIPTION=Existing Description",
                "SSL=true",
                "SSL_CERTFILE=/missing/cert.pem",
                "SSL_KEYFILE=/missing/key.pem",
                "RERANK_BINDING=jina",
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

prompt_secret_until_valid_with_default() {{
  printf '%s' "$2"
}}

finalize_setup() {{
  printf 'HOST=%s\\n' "${{ENV_VALUES[HOST]}}"
  printf 'PORT=%s\\n' "${{ENV_VALUES[PORT]}}"
  printf 'WEBUI_TITLE=%s\\n' "${{ENV_VALUES[WEBUI_TITLE]}}"
  printf 'WEBUI_DESCRIPTION=%s\\n' "${{ENV_VALUES[WEBUI_DESCRIPTION]}}"
  printf 'LIGHTRAG_KV_STORAGE=%s\\n' "${{ENV_VALUES[LIGHTRAG_KV_STORAGE]}}"
  printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
  printf 'LLM_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[LLM_BINDING_API_KEY]}}"
  printf 'EMBEDDING_BINDING_API_KEY=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_API_KEY]}}"
  printf 'SSL_SET=%s\\n' "${{ENV_VALUES[SSL]+set}}"
  printf 'SSL_CERTFILE_SET=%s\\n' "${{ENV_VALUES[SSL_CERTFILE]+set}}"
  printf 'SSL_KEYFILE_SET=%s\\n' "${{ENV_VALUES[SSL_KEYFILE]+set}}"
  printf 'RERANK_BINDING_SET=%s\\n' "${{ENV_VALUES[RERANK_BINDING]+set}}"
  printf 'AUTH_ACCOUNTS_SET=%s\\n' "${{ENV_VALUES[AUTH_ACCOUNTS]+set}}"
  printf 'TOKEN_SECRET_SET=%s\\n' "${{ENV_VALUES[TOKEN_SECRET]+set}}"
  printf 'LIGHTRAG_API_KEY_SET=%s\\n' "${{ENV_VALUES[LIGHTRAG_API_KEY]+set}}"
  printf 'LANGFUSE_ENABLE_TRACE_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_ENABLE_TRACE]+set}}"
  printf 'LANGFUSE_SECRET_KEY_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_SECRET_KEY]+set}}"
}}

quick_start_flow
"""
    )
    values = parse_lines(output)

    assert values["HOST"] == "127.0.0.1"
    assert values["PORT"] == "9999"
    assert values["WEBUI_TITLE"] == "Existing Title"
    assert values["WEBUI_DESCRIPTION"] == "Existing Description"
    assert values["LIGHTRAG_KV_STORAGE"] == "JsonKVStorage"
    assert values["LLM_BINDING"] == "openai"
    assert values["LLM_BINDING_API_KEY"] == "sk-existing"
    assert values["EMBEDDING_BINDING_API_KEY"] == "sk-existing"
    assert values["SSL_SET"] == ""
    assert values["SSL_CERTFILE_SET"] == ""
    assert values["SSL_KEYFILE_SET"] == ""
    assert values["RERANK_BINDING_SET"] == ""
    assert values["AUTH_ACCOUNTS_SET"] == ""
    assert values["TOKEN_SECRET_SET"] == ""
    assert values["LIGHTRAG_API_KEY_SET"] == ""
    assert values["LANGFUSE_ENABLE_TRACE_SET"] == ""
    assert values["LANGFUSE_SECRET_KEY_SET"] == ""


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


def test_quick_start_flow_clears_inherited_ssl_state_before_writing_env(
    tmp_path: Path,
) -> None:
    """Quick start should not fail on rerun because of stale SSL certificate paths."""

    env_example = tmp_path / "env.example"
    env_example.write_text((REPO_ROOT / "env.example").read_text(encoding="utf-8"))

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SSL=true",
                "SSL_CERTFILE=/missing/cert.pem",
                "SSL_KEYFILE=/missing/key.pem",
                "LLM_BINDING_API_KEY=sk-existing",
                "EMBEDDING_BINDING_API_KEY=sk-existing",
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

confirm_count=0
prompt_secret_until_valid_with_default() {{
  printf '%s' "$2"
}}
confirm() {{
  confirm_count=$((confirm_count + 1))
  if [[ "$confirm_count" -eq 1 ]]; then
    return 0
  fi
  return 1
}}

quick_start_flow
"""
    )

    generated_lines = env_file.read_text(encoding="utf-8").splitlines()

    assert not any(line == "SSL=true" for line in generated_lines)
    assert not any(line.startswith("SSL_CERTFILE=") for line in generated_lines)
    assert not any(line.startswith("SSL_KEYFILE=") for line in generated_lines)
    assert "LIGHTRAG_SETUP_PROFILE=development" in generated_lines
    assert "LLM_BINDING_API_KEY=sk-existing" in generated_lines
    assert "EMBEDDING_BINDING_API_KEY=sk-existing" in generated_lines


def test_interactive_flow_clears_inherited_ssl_state_for_non_production_reruns(
    tmp_path: Path,
) -> None:
    """Development/custom reruns should not keep hidden SSL values from old `.env`."""

    env_example = tmp_path / "env.example"
    env_example.write_text((REPO_ROOT / "env.example").read_text(encoding="utf-8"))

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SSL=true",
                "SSL_CERTFILE=/missing/cert.pem",
                "SSL_KEYFILE=/missing/key.pem",
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

confirm_count=0
select_deployment_type() {{ printf 'development'; }}
select_storage_backends() {{ :; }}
collect_docker_image_tags() {{ :; }}
collect_llm_config() {{ :; }}
collect_embedding_config() {{ :; }}
collect_rerank_config() {{ ENV_VALUES[RERANK_BINDING]="null"; }}
collect_server_config() {{ :; }}
collect_security_config() {{ :; }}
collect_observability_config() {{ :; }}
confirm() {{
  confirm_count=$((confirm_count + 1))
  if [[ "$confirm_count" -eq 1 ]]; then
    return 0
  fi
  return 1
}}

interactive_flow
"""
    )

    generated_lines = env_file.read_text(encoding="utf-8").splitlines()

    assert not any(line == "SSL=true" for line in generated_lines)
    assert not any(line.startswith("SSL_CERTFILE=") for line in generated_lines)
    assert not any(line.startswith("SSL_KEYFILE=") for line in generated_lines)
    assert "LIGHTRAG_SETUP_PROFILE=development" in generated_lines


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


def test_collect_mongodb_config_local_service_strips_stale_credentials_on_rerun() -> None:
    """Bundled MongoDB should keep host `.env` aligned with the unauthenticated template."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[MONGO_URI]="mongodb://root:secret@localhost:27017/"

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
    values = parse_lines(output)

    assert values["MONGO_URI"] == "mongodb://localhost:27017/"
    assert values["COMPOSE_MONGO_URI"] == "mongodb://mongodb:27017/"
    assert values["DOCKER_SERVICE"] == "mongodb"


def test_collect_redis_config_local_service_normalizes_custom_host_port() -> None:
    """Bundled Redis should keep host `.env` aligned with the published local port."""

    output = run_bash(
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
    values = parse_lines(output)

    assert values["REDIS_URI"] == "redis://localhost:6379/1"
    assert values["COMPOSE_REDIS_URI"] == "redis://redis:6379"
    assert values["DOCKER_SERVICE"] == "redis"


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
    assert values["WHITELIST_PATHS_SET"] == "set"
    assert not any(line.startswith("AUTH_ACCOUNTS=") for line in generated_lines)
    assert not any(line.startswith("TOKEN_SECRET=") for line in generated_lines)
    assert not any(line.startswith("TOKEN_EXPIRE_HOURS=") for line in generated_lines)
    assert not any(line.startswith("LIGHTRAG_API_KEY=") for line in generated_lines)
    assert "WHITELIST_PATHS=" in generated_lines


def test_collect_security_config_uses_safe_production_whitelist_default() -> None:
    """Production security prompts should not default to exposing `/api/*`."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[WHITELIST_PATHS]="/health,/api/*"

confirm_default_yes() {{ return 0; }}
prompt_clearable_with_default() {{ printf '%s' "$2"; }}
prompt_clearable_secret_with_default() {{ printf '%s' "$2"; }}

collect_security_config yes yes

printf 'WHITELIST_PATHS=%s\\n' "${{ENV_VALUES[WHITELIST_PATHS]}}"
"""
    )
    values = parse_lines(output)

    assert values["WHITELIST_PATHS"] == "/health"


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

confirm() {{ return 0; }}
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


def test_validate_env_file_allows_empty_production_whitelist_with_auth_accounts(
    tmp_path: Path,
) -> None:
    """Production validation should accept an explicitly empty whitelist."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=PGKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
                "POSTGRES_USER=lightrag",
                "POSTGRES_PASSWORD=secret",
                "POSTGRES_DATABASE=lightrag",
                "MILVUS_URI=http://localhost:19530",
                "MILVUS_DB_NAME=lightrag",
                "NEO4J_URI=neo4j://localhost:7687",
                "NEO4J_USERNAME=neo4j",
                "NEO4J_PASSWORD=secret",
                "AUTH_ACCOUNTS=admin:secret",
                "TOKEN_SECRET=jwt-secret",
                "WHITELIST_PATHS=",
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

    assert result.returncode == 0
    assert "Validation passed." in result.stdout


def test_validate_env_file_rejects_missing_production_whitelist_with_auth_accounts(
    tmp_path: Path,
) -> None:
    """Production validation should reject auth configs that omit the whitelist key."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=PGKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
                "POSTGRES_USER=lightrag",
                "POSTGRES_PASSWORD=secret",
                "POSTGRES_DATABASE=lightrag",
                "MILVUS_URI=http://localhost:19530",
                "MILVUS_DB_NAME=lightrag",
                "NEO4J_URI=neo4j://localhost:7687",
                "NEO4J_USERNAME=neo4j",
                "NEO4J_PASSWORD=secret",
                "AUTH_ACCOUNTS=admin:secret",
                "TOKEN_SECRET=jwt-secret",
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

    assert result.returncode != 0
    assert "must not whitelist /api routes" in result.stderr


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

confirm() {{ return 1; }}

collect_observability_config
generate_env_file "{REPO_ROOT}/env.example" "$REPO_ROOT/.env.generated"

printf 'LANGFUSE_ENABLE_TRACE_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_ENABLE_TRACE]+set}}"
printf 'LANGFUSE_SECRET_KEY_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_SECRET_KEY]+set}}"
printf 'LANGFUSE_PUBLIC_KEY_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_PUBLIC_KEY]+set}}"
printf 'LANGFUSE_HOST_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_HOST]+set}}"
"""
    )
    values = parse_lines(output)
    generated_lines = (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()

    assert values["LANGFUSE_ENABLE_TRACE_SET"] == ""
    assert values["LANGFUSE_SECRET_KEY_SET"] == ""
    assert values["LANGFUSE_PUBLIC_KEY_SET"] == ""
    assert values["LANGFUSE_HOST_SET"] == ""
    assert not any(line.startswith("LANGFUSE_ENABLE_TRACE=") for line in generated_lines)
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


def test_finalize_setup_rejects_production_config_without_auth_or_api_key() -> None:
    """Production setup should not generate an unauthenticated deployment."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state
DEPLOYMENT_TYPE="production"

ENV_VALUES[LIGHTRAG_KV_STORAGE]="JsonKVStorage"
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="NanoVectorDBStorage"
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="NetworkXStorage"
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="JsonDocStatusStorage"

if finalize_setup >/tmp/finalize.out 2>/tmp/finalize.err; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["RESULT"] == "failure"


def test_finalize_setup_fails_when_selected_services_never_become_ready(
    tmp_path: Path,
) -> None:
    """Setup should fail fast instead of reporting success on broken dependency startup."""

    env_example = tmp_path / "env.example"
    env_example.write_text(
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

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
DEPLOYMENT_TYPE="development"
DOCKER_SERVICES=("redis")

ENV_VALUES[LIGHTRAG_KV_STORAGE]="JsonKVStorage"
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="NanoVectorDBStorage"
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="NetworkXStorage"
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="JsonDocStatusStorage"

confirm() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
check_docker_availability() {{ return 0; }}
backup_env_file() {{ return 0; }}
generate_env_file() {{ :; }}
generate_docker_compose() {{ :; }}
wait_for_services() {{ return 1; }}
docker() {{
  printf '%s\\n' "$*" >> "$REPO_ROOT/docker_calls.log"
}}

if finalize_setup; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
"""
    )
    values = parse_lines(output)
    docker_calls = (tmp_path / "docker_calls.log").read_text(encoding="utf-8").splitlines()

    assert values["RESULT"] == "failure"
    assert docker_calls == [f"compose -f {tmp_path}/docker-compose.development.yml up -d redis"]


def test_validate_security_config_requires_auth_accounts_for_production() -> None:
    """Production validation should require account-based auth."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

if validate_security_config "" "" "" yes; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi

if validate_security_config "" "" "api-key" yes "/health"; then
  printf 'WITH_API_KEY=yes\\n'
else
  printf 'WITH_API_KEY=no\\n'
fi

if validate_security_config "admin:secret" "token-secret" "" yes "/health"; then
  printf 'WITH_AUTH_ACCOUNTS=yes\\n'
else
  printf 'WITH_AUTH_ACCOUNTS=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["VALID"] == "no"
    assert values["WITH_API_KEY"] == "no"
    assert values["WITH_AUTH_ACCOUNTS"] == "yes"


def test_validate_security_config_allows_empty_but_rejects_api_whitelist_for_production() -> None:
    """Production validation should allow an empty whitelist but still reject `/api/*`."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

if validate_security_config "admin:secret" "token-secret" "" yes ""; then
  printf 'EMPTY_WHITELIST=yes\\n'
else
  printf 'EMPTY_WHITELIST=no\\n'
fi

if validate_security_config "admin:secret" "token-secret" "" yes; then
  printf 'OMITTED_WHITELIST=yes\\n'
else
  printf 'OMITTED_WHITELIST=no\\n'
fi

if validate_security_config "admin:secret" "token-secret" "" yes "/health,/api/*"; then
  printf 'API_WHITELIST=yes\\n'
else
  printf 'API_WHITELIST=no\\n'
fi

if validate_security_config "admin:secret" "token-secret" "" yes "/health,/api/v1/*"; then
  printf 'API_PREFIX_WHITELIST=yes\\n'
else
  printf 'API_PREFIX_WHITELIST=no\\n'
fi

if validate_security_config "admin:secret" "token-secret" "" yes "/health,/docs"; then
  printf 'SAFE_WHITELIST=yes\\n'
else
  printf 'SAFE_WHITELIST=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["EMPTY_WHITELIST"] == "yes"
    assert values["OMITTED_WHITELIST"] == "no"
    assert values["API_WHITELIST"] == "no"
    assert values["API_PREFIX_WHITELIST"] == "no"
    assert values["SAFE_WHITELIST"] == "yes"


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


def test_validate_api_key_accepts_non_sk_keys_for_openai_compatible_providers() -> None:
    """OpenAI-compatible endpoints should accept any non-empty API key."""

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"

if validate_api_key "local-key" openai; then
  printf 'OPENAI_VALID=yes\\n'
else
  printf 'OPENAI_VALID=no\\n'
fi

if validate_api_key "gateway-token" openrouter; then
  printf 'OPENROUTER_VALID=yes\\n'
else
  printf 'OPENROUTER_VALID=no\\n'
fi

if validate_api_key "" openai; then
  printf 'EMPTY_VALID=yes\\n'
else
  printf 'EMPTY_VALID=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["OPENAI_VALID"] == "yes"
    assert values["OPENROUTER_VALID"] == "yes"
    assert values["EMPTY_VALID"] == "no"


def test_validate_env_file_requires_protection_for_production_storage_profile(
    tmp_path: Path,
) -> None:
    """`setup-validate` should reject the production storage profile without auth."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=PGKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
                "POSTGRES_USER=lightrag",
                "POSTGRES_PASSWORD=secret",
                "POSTGRES_DATABASE=lightrag",
                "MILVUS_URI=http://localhost:19530",
                "MILVUS_DB_NAME=lightrag",
                "NEO4J_URI=neo4j://localhost:7687",
                "NEO4J_USERNAME=neo4j",
                "NEO4J_PASSWORD=secret",
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

if validate_env_file; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["VALID"] == "no"


def test_validate_env_file_requires_protection_for_production_setup_profile(
    tmp_path: Path,
) -> None:
    """`setup-validate` should honor persisted production setup metadata."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_SETUP_PROFILE=production",
                "LIGHTRAG_KV_STORAGE=PGKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=QdrantVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
                "POSTGRES_USER=lightrag",
                "POSTGRES_PASSWORD=secret",
                "POSTGRES_DATABASE=lightrag",
                "QDRANT_URL=http://localhost:6333",
                "NEO4J_URI=neo4j://localhost:7687",
                "NEO4J_USERNAME=neo4j",
                "NEO4J_PASSWORD=secret",
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

if validate_env_file; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["VALID"] == "no"


def test_finalize_setup_requires_protection_for_custom_production_storage_profile(
    tmp_path: Path,
) -> None:
    """Custom mode should not allow the production storage profile without auth."""

    (tmp_path / "env.example").write_text("LLM_BINDING=openai\n", encoding="utf-8")

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
DEPLOYMENT_TYPE="custom"

ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="MilvusVectorDBStorage"
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="Neo4JStorage"
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
ENV_VALUES[POSTGRES_USER]="lightrag"
ENV_VALUES[POSTGRES_PASSWORD]="secret"
ENV_VALUES[POSTGRES_DATABASE]="lightrag"
ENV_VALUES[MILVUS_URI]="http://localhost:19530"
ENV_VALUES[MILVUS_DB_NAME]="lightrag"
ENV_VALUES[NEO4J_URI]="neo4j://localhost:7687"
ENV_VALUES[NEO4J_USERNAME]="neo4j"
ENV_VALUES[NEO4J_PASSWORD]="secret"

confirm() {{ return 0; }}
confirm_default_yes() {{ return 1; }}
backup_env_file() {{ return 0; }}
generate_env_file() {{ :; }}
generate_docker_compose() {{ :; }}
show_summary() {{ :; }}

if finalize_setup >/dev/null 2>&1; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["RESULT"] == "failure"


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


def test_finalize_setup_rejects_api_key_only_production_storage_profile(
    tmp_path: Path,
) -> None:
    """Production-like storage profiles should not pass with API-key-only protection."""

    (tmp_path / "env.example").write_text("LLM_BINDING=openai\n", encoding="utf-8")

    output = run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
DEPLOYMENT_TYPE="custom"

ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="MilvusVectorDBStorage"
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="Neo4JStorage"
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
ENV_VALUES[POSTGRES_USER]="lightrag"
ENV_VALUES[POSTGRES_PASSWORD]="secret"
ENV_VALUES[POSTGRES_DATABASE]="lightrag"
ENV_VALUES[MILVUS_URI]="http://localhost:19530"
ENV_VALUES[MILVUS_DB_NAME]="lightrag"
ENV_VALUES[NEO4J_URI]="neo4j://localhost:7687"
ENV_VALUES[NEO4J_USERNAME]="neo4j"
ENV_VALUES[NEO4J_PASSWORD]="secret"
ENV_VALUES[LIGHTRAG_API_KEY]="api-key"

confirm() {{ return 0; }}
confirm_default_yes() {{ return 1; }}
backup_env_file() {{ return 0; }}
generate_env_file() {{ :; }}
generate_docker_compose() {{ :; }}
show_summary() {{ :; }}

if finalize_setup >/dev/null 2>&1; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
"""
    )
    values = parse_lines(output)

    assert values["RESULT"] == "failure"


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

confirm() {{ return 0; }}

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
