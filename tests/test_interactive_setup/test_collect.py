# Regression tests for interactive setup wizard.
# Classification: keep tests here when they target collect_* prompt/normalization logic for one config area before env/compose files are finalized.

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_interactive_setup._helpers import (
    REPO_ROOT,
    parse_lines,
    run_bash,
    run_bash_lines,
    write_text_lines,
)

pytestmark = pytest.mark.offline


def test_collect_postgres_config_uses_fixed_bundled_port_and_compose_overrides() -> (
    None
):
    """Bundled PostgreSQL should use the fixed service port and compose overrides."""
    values = run_bash_lines(f"""
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
mask_sensitive_input() {{ printf 'supersecret'; }}

collect_postgres_config yes

printf 'POSTGRES_HOST=%s\\n' "${{ENV_VALUES[POSTGRES_HOST]}}"
printf 'POSTGRES_PORT=%s\\n' "${{ENV_VALUES[POSTGRES_PORT]}}"
printf 'COMPOSE_POSTGRES_HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[POSTGRES_HOST]}}"
printf 'COMPOSE_POSTGRES_PORT=%s\\n' "${{COMPOSE_ENV_OVERRIDES[POSTGRES_PORT]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}\"
""")
    assert values["POSTGRES_HOST"] == "localhost"
    assert values["POSTGRES_PORT"] == "5432"
    assert values["COMPOSE_POSTGRES_HOST"] == "postgres"
    assert values["COMPOSE_POSTGRES_PORT"] == "5432"
    assert values["DOCKER_SERVICE"] == "postgres"


def test_collect_postgres_config_uses_rag_defaults_without_prompt_for_empty_docker_credentials() -> (
    None
):
    """Docker PostgreSQL should auto-fill bundled credentials when old `.env` creds are empty."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

PROMPT_LOG_FILE="$(mktemp)"
: > "$PROMPT_LOG_FILE"

confirm_default_yes() {{ return 0; }}
prompt_with_default() {{
  printf '%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  case "$1" in
    "PostgreSQL host") printf 'localhost' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_secret_with_default() {{
  printf 'secret:%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}

ORIGINAL_ENV_VALUES[POSTGRES_USER]=""
ORIGINAL_ENV_VALUES[POSTGRES_PASSWORD]=""
ORIGINAL_ENV_VALUES[POSTGRES_DATABASE]=""

collect_postgres_config yes

printf 'POSTGRES_USER=%s\\n' "${{ENV_VALUES[POSTGRES_USER]}}"
printf 'POSTGRES_PASSWORD=%s\\n' "${{ENV_VALUES[POSTGRES_PASSWORD]}}"
printf 'POSTGRES_DATABASE=%s\\n' "${{ENV_VALUES[POSTGRES_DATABASE]}}"
printf 'PROMPT_LOG=%s\\n' "$(paste -sd '|' "$PROMPT_LOG_FILE")\"
""")
    assert values["POSTGRES_USER"] == "rag"
    assert values["POSTGRES_PASSWORD"] == "rag"
    assert values["POSTGRES_DATABASE"] == "rag"
    assert values["PROMPT_LOG"] == "PostgreSQL host"


def test_collect_postgres_config_prompts_for_existing_docker_credentials() -> None:
    """Docker PostgreSQL should preserve editability when old `.env` creds already exist."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

PROMPT_LOG_FILE="$(mktemp)"
: > "$PROMPT_LOG_FILE"

confirm_default_yes() {{ return 0; }}
prompt_with_default() {{
  printf '%s[%s]\\n' "$1" "$2" >> "$PROMPT_LOG_FILE"
  case "$1" in
    "PostgreSQL host") printf 'localhost' ;;
    "PostgreSQL user") printf 'updated-user' ;;
    "PostgreSQL database") printf 'updated-db' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_secret_with_default() {{
  printf '%s[%s]\\n' "$1" "$2" >> "$PROMPT_LOG_FILE"
  printf 'updated-password'
}}

ORIGINAL_ENV_VALUES[POSTGRES_USER]="existing-user"
ORIGINAL_ENV_VALUES[POSTGRES_PASSWORD]="existing-password"
ORIGINAL_ENV_VALUES[POSTGRES_DATABASE]="existing-db"

collect_postgres_config yes

printf 'POSTGRES_USER=%s\\n' "${{ENV_VALUES[POSTGRES_USER]}}"
printf 'POSTGRES_PASSWORD=%s\\n' "${{ENV_VALUES[POSTGRES_PASSWORD]}}"
printf 'POSTGRES_DATABASE=%s\\n' "${{ENV_VALUES[POSTGRES_DATABASE]}}"
printf 'PROMPT_LOG=%s\\n' "$(paste -sd '|' "$PROMPT_LOG_FILE")\"
""")
    assert values["POSTGRES_USER"] == "updated-user"
    assert values["POSTGRES_PASSWORD"] == "updated-password"
    assert values["POSTGRES_DATABASE"] == "updated-db"
    assert (
        values["PROMPT_LOG"]
        == "PostgreSQL host[localhost]|PostgreSQL user[existing-user]|PostgreSQL password: [existing-password]|PostgreSQL database[existing-db]"
    )


def test_collect_postgres_config_still_prompts_for_host_credentials() -> None:
    """Host PostgreSQL should keep prompting even when saved creds are empty."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

PROMPT_LOG_FILE="$(mktemp)"
: > "$PROMPT_LOG_FILE"

confirm_default_no() {{ return 1; }}
prompt_with_default() {{
  printf '%s[%s]\\n' "$1" "$2" >> "$PROMPT_LOG_FILE"
  case "$1" in
    "PostgreSQL host") printf 'db.internal' ;;
    "PostgreSQL user") printf 'host-user' ;;
    "PostgreSQL database") printf 'host-db' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_until_valid() {{
  printf '%s[%s]\\n' "$1" "$2" >> "$PROMPT_LOG_FILE"
  if [[ "$1" == "PostgreSQL port" ]]; then
    printf '6543'
  else
    printf '%s' "$2"
  fi
}}
prompt_secret_with_default() {{
  printf '%s[%s]\\n' "$1" "$2" >> "$PROMPT_LOG_FILE"
  printf 'host-password'
}}

ORIGINAL_ENV_VALUES[POSTGRES_USER]=""
ORIGINAL_ENV_VALUES[POSTGRES_PASSWORD]=""

collect_postgres_config no

printf 'POSTGRES_HOST=%s\\n' "${{ENV_VALUES[POSTGRES_HOST]}}"
printf 'POSTGRES_PORT=%s\\n' "${{ENV_VALUES[POSTGRES_PORT]}}"
printf 'POSTGRES_USER=%s\\n' "${{ENV_VALUES[POSTGRES_USER]}}"
printf 'POSTGRES_PASSWORD=%s\\n' "${{ENV_VALUES[POSTGRES_PASSWORD]}}"
printf 'PROMPT_LOG=%s\\n' "$(paste -sd '|' "$PROMPT_LOG_FILE")\"
""")
    assert values["POSTGRES_HOST"] == "db.internal"
    assert values["POSTGRES_PORT"] == "6543"
    assert values["POSTGRES_USER"] == "host-user"
    assert values["POSTGRES_PASSWORD"] == "host-password"
    assert (
        values["PROMPT_LOG"]
        == "PostgreSQL host[localhost]|PostgreSQL port[5432]|PostgreSQL user[rag]|PostgreSQL password: [rag]|PostgreSQL database[lightrag]"
    )


def test_collect_server_config_includes_summary_language_last() -> None:
    """Server config should prompt for summary language after the WebUI fields."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

PROMPT_LOG_FILE="$(mktemp)"
: > "$PROMPT_LOG_FILE"

prompt_with_default() {{
  printf '%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  case "$1" in
    "Server host") printf '127.0.0.1' ;;
    "WebUI title") printf 'Custom KB' ;;
    "WebUI description") printf 'Custom description' ;;
    "Summary language") printf 'Chinese' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_until_valid() {{
  printf '%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  if [[ "$1" == "Server port" ]]; then
    printf '9630'
  else
    printf '%s' "$2"
  fi
}}

collect_server_config

printf 'HOST=%s\\n' "${{ENV_VALUES[HOST]}}"
printf 'PORT=%s\\n' "${{ENV_VALUES[PORT]}}"
printf 'WEBUI_TITLE=%s\\n' "${{ENV_VALUES[WEBUI_TITLE]}}"
printf 'WEBUI_DESCRIPTION=%s\\n' "${{ENV_VALUES[WEBUI_DESCRIPTION]}}"
printf 'SUMMARY_LANGUAGE=%s\\n' "${{ENV_VALUES[SUMMARY_LANGUAGE]}}"
printf 'PROMPT_LOG=%s\\n' "$(paste -sd '|' "$PROMPT_LOG_FILE")\"
""")
    assert values["HOST"] == "127.0.0.1"
    assert values["PORT"] == "9630"
    assert values["WEBUI_TITLE"] == "Custom KB"
    assert values["WEBUI_DESCRIPTION"] == "Custom description"
    assert values["SUMMARY_LANGUAGE"] == "Chinese"
    assert (
        values["PROMPT_LOG"]
        == "Server host|Server port|WebUI title|WebUI description|Summary language"
    )


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
            "5432",
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
            "mongodb://localhost:27017/?directConnection=true",
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
        "postgres-port-reset-to-bundled-default",
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
    setup_lines: list[str], collector_call: str, env_key: str, expected_value: str
) -> None:
    """Bundled services should normalize stale remote or localhost endpoints on rerun."""
    setup_block = "\n".join(setup_lines)
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

{setup_block}

confirm_default_yes() {{ return 0; }}
prompt_choice() {{ printf '%s' "$2"; }}
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

printf '{env_key}=%s\\n' "${{ENV_VALUES[{env_key}]}}\"
""")
    assert values[env_key] == expected_value


def test_collect_ssl_config_can_disable_loaded_ssl_values(tmp_path: Path) -> None:
    """Declining SSL should clear previously loaded cert paths and staged sources."""
    cert_path = tmp_path / "cert.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path = tmp_path / "key.pem"
    key_path.write_text("key", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(["SSL=true", f"SSL_CERTFILE={cert_path}", f"SSL_KEYFILE={key_path}"])
        + "\n",
        encoding="utf-8",
    )
    output = run_bash(f"""
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
printf 'SSL_KEY_SOURCE_PATH=%s\\n' "$SSL_KEY_SOURCE_PATH\"
""")
    values = parse_lines(output)
    assert values["SSL_IS_SET"] == ""
    assert values["SSL_CERTFILE_IS_SET"] == ""
    assert values["SSL_KEYFILE_IS_SET"] == ""
    assert values["SSL_CERT_SOURCE_PATH"] == ""
    assert values["SSL_KEY_SOURCE_PATH"] == ""


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
    tmp_path: Path, collector_name: str, binding_prefix: str, env_lines: list[str]
) -> None:
    """Switching a provider to Bedrock should remove stale API-key settings."""
    write_text_lines(tmp_path / ".env", env_lines)
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

prompt_choice() {{ printf 'bedrock'; }}
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
""")
    values = parse_lines(output)
    assert values["BINDING"] == "bedrock"
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
        (
            "collect_llm_config",
            "LLM",
            "gemini",
            "prompt_secret_until_valid_with_default() { printf 'gemini-secret-key'; }",
            "gemini",
            "gemini-flash-latest",
            "DEFAULT_GEMINI_ENDPOINT",
            "",
            "set",
        ),
        (
            "collect_llm_config",
            "LLM",
            "bedrock",
            """
prompt_clearable_with_default() { printf ''; }
prompt_required_secret() { return 1; }
confirm_default_yes() { return 1; }
""",
            "bedrock",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "DEFAULT_BEDROCK_ENDPOINT",
            "",
            "",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            "gemini",
            "prompt_secret_until_valid_with_default() { printf 'gemini-secret-key'; }",
            "gemini",
            "gemini-embedding-001",
            "DEFAULT_GEMINI_ENDPOINT",
            "1536",
            "set",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            "bedrock",
            """
prompt_clearable_with_default() { printf ''; }
prompt_required_secret() { return 1; }
confirm_default_yes() { return 1; }
""",
            "bedrock",
            "amazon.titan-embed-text-v2:0",
            "DEFAULT_BEDROCK_ENDPOINT",
            "1024",
            "",
        ),
    ],
    ids=[
        "llm-provider-defaults",
        "embedding-provider-defaults",
        "llm-gemini-sentinel-default",
        "llm-bedrock-sentinel-default",
        "embedding-gemini-sentinel-default",
        "embedding-bedrock-sentinel-default",
    ],
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
    output = run_bash(f"""
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
printf 'API_KEY_SET=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_API_KEY]+set}}\"
""")
    values = parse_lines(output)
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
        "expected_host",
        "expected_api_key",
    ),
    [
        (
            "collect_llm_config",
            "LLM",
            [
                "LLM_BINDING=gemini",
                "LLM_MODEL=gemini-flash-latest",
                "LLM_BINDING_HOST=https://generativelanguage.googleapis.com",
                "LLM_BINDING_API_KEY=gemini-existing-key",
            ],
            "https://generativelanguage.googleapis.com",
            "gemini-existing-key",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            [
                "EMBEDDING_BINDING=bedrock",
                "EMBEDDING_MODEL=amazon.titan-embed-text-v2:0",
                "EMBEDDING_DIM=1024",
                "EMBEDDING_BINDING_HOST=https://bedrock.amazonaws.com",
            ],
            "https://bedrock.amazonaws.com",
            "",
        ),
    ],
    ids=[
        "llm-rerun-preserves-explicit-gemini-host",
        "embedding-rerun-preserves-explicit-bedrock-host",
    ],
)
def test_collect_provider_config_preserves_explicit_host_on_rerun(
    tmp_path: Path,
    collector_name: str,
    binding_prefix: str,
    env_lines: list[str],
    expected_host: str,
    expected_api_key: str,
) -> None:
    """Reruns should keep saved explicit provider hosts instead of swapping to sentinels."""
    write_text_lines(tmp_path / ".env", env_lines)
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
prompt_clearable_with_default() {{ printf ''; }}
prompt_required_secret() {{ return 1; }}
confirm_default_yes() {{ return 1; }}

{collector_name}

printf 'HOST=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_HOST]}}"
printf 'API_KEY=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_API_KEY]:-}}\"
""")
    values = parse_lines(output)
    assert values["HOST"] == expected_host
    assert values["API_KEY"] == expected_api_key


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
            "prompt_with_default() { printf '%s' \"$2\"; }",
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
    output = run_bash(f"""
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
printf 'API_KEY=%s\\n' "${{ENV_VALUES[{binding_prefix}_BINDING_API_KEY]:-}}\"
""")
    values = parse_lines(output)
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
    output = run_bash(f"""
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
printf 'EMBEDDING_BINDING_API_KEY_SET=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_API_KEY]+set}}\"
""")
    values = parse_lines(output)
    assert values["EMBEDDING_BINDING"] == "ollama"
    assert values["EMBEDDING_MODEL"] == "bge-m3:latest"
    assert values["EMBEDDING_DIM"] == "1024"
    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:11434"
    assert values["EMBEDDING_BINDING_API_KEY_SET"] == ""


def test_collect_llm_config_allows_bedrock_ambient_credential_chain() -> None:
    """Bedrock setup should allow IAM roles, AWS profiles, or SSO without saved keys."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_choice() {{ printf 'bedrock'; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_clearable_with_default() {{ printf ''; }}
prompt_required_secret() {{ return 1; }}
confirm_default_yes() {{ return 1; }}

collect_llm_config

printf 'LLM_BINDING=%s\\n' "${{ENV_VALUES[LLM_BINDING]}}"
printf 'AWS_ACCESS_KEY_ID_SET=%s\\n' "${{ENV_VALUES[AWS_ACCESS_KEY_ID]+set}}"
printf 'AWS_SECRET_ACCESS_KEY_SET=%s\\n' "${{ENV_VALUES[AWS_SECRET_ACCESS_KEY]+set}}"
printf 'AWS_SESSION_TOKEN_SET=%s\\n' "${{ENV_VALUES[AWS_SESSION_TOKEN]+set}}"
printf 'AWS_REGION_SET=%s\\n' "${{ENV_VALUES[AWS_REGION]+set}}\"
""")
    values = parse_lines(output)
    assert values["LLM_BINDING"] == "bedrock"
    assert values["AWS_ACCESS_KEY_ID_SET"] == ""
    assert values["AWS_SECRET_ACCESS_KEY_SET"] == ""
    assert values["AWS_SESSION_TOKEN_SET"] == ""
    assert values["AWS_REGION_SET"] == ""


def test_collect_rerank_config_preserves_api_key_when_disabled(tmp_path: Path) -> None:
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
    values = run_bash_lines(f"""
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
""")
    assert values["RERANK_BINDING"] == "null"
    assert values["RERANK_BINDING_API_KEY_SET"] == "set"
    assert values["VALID"] == "yes"


def test_collect_rerank_config_does_not_offer_vllm_provider_option() -> None:
    """The generic rerank provider prompt should only expose valid RERANK_BINDING values."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[RERANK_BINDING]="cohere"

confirm_default_no() {{ return 0; }}
prompt_choice() {{
  case "$1" in
    "Rerank provider")
      shift 2
      for option in "$@"; do
        if [[ "$option" == "vllm" ]]; then
          echo "unexpected vllm option" >&2
          return 91
        fi
      done
      printf 'cohere'
      ;;
    *)
      printf '%s' "$2"
      ;;
  esac
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf 'cohere-secret-123'; }}

collect_rerank_config

printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}\"
""")
    values = parse_lines(output)
    assert values["RERANK_BINDING"] == "cohere"


def test_collect_rerank_config_switching_from_vllm_clears_local_defaults() -> None:
    """Switching from local vLLM to hosted rerank should replace stale vLLM values with provider defaults."""
    output = run_bash(f"""
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
printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]:-}}"
printf 'RERANK_MODEL=%s\\n' "${{ENV_VALUES[RERANK_MODEL]:-}}"
printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]:-}}\"
""")
    values = parse_lines(output)
    assert values["RERANK_BINDING"] == "cohere"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == ""
    assert values["RERANK_MODEL"] != "BAAI/bge-reranker-v2-m3"
    assert values["RERANK_MODEL"] == "rerank-v3.5"
    assert "localhost:8000" not in values["RERANK_BINDING_HOST"]
    assert "cohere" in values["RERANK_BINDING_HOST"]


def test_collect_rerank_config_ignores_vllm_marker_when_docker_is_predeclined() -> None:
    """A predeclined Docker path should default the provider prompt to the binding, not the setup marker."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]="vllm"
ENV_VALUES[RERANK_BINDING]="cohere"

prompt_choice() {{
  case "$1" in
    "Rerank provider")
      if [[ "$2" != "cohere" ]]; then
        echo "unexpected rerank provider default: $2" >&2
        return 91
      fi
      printf 'cohere'
      ;;
    *)
      printf '%s' "$2"
      ;;
  esac
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf 'cohere-secret-123'; }}

collect_rerank_config "yes" "no"

printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}\"
""")
    values = parse_lines(output)
    assert values["RERANK_BINDING"] == "cohere"


def test_collect_milvus_config_defaults_to_existing_database_name() -> None:
    """Milvus database prompt should preserve the documented default database."""
    values = run_bash_lines(f"""
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

printf 'MILVUS_DB_NAME=%s\\n' "${{ENV_VALUES[MILVUS_DB_NAME]}}\"
""")
    assert values["MILVUS_DB_NAME"] == "lightrag"


def test_collect_milvus_config_initializes_minio_credentials_for_local_docker(
    tmp_path: Path,
) -> None:
    """Local Docker Milvus should write default MinIO credentials when none exist yet."""
    env_file = tmp_path / ".env"
    env_example = tmp_path / "env.example"
    env_example.write_text((REPO_ROOT / "env.example").read_text(encoding="utf-8"))
    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

confirm_default_yes() {{ return 0; }}
prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{
  printf '%s' "$2"
}}
prompt_until_valid() {{
  printf '%s' "$2"
}}

collect_milvus_config yes
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env\"
""",
        cwd=tmp_path,
    )
    env_text = env_file.read_text(encoding="utf-8")
    assert "MINIO_ACCESS_KEY_ID=minioadmin" in env_text
    assert "MINIO_SECRET_ACCESS_KEY=minioadmin" in env_text


@pytest.mark.parametrize(
    ("setup_lines", "nvidia_impl", "expected_device"),
    [
        (['ENV_VALUES[MILVUS_DEVICE]="cpu"'], "nvidia-smi() { return 0; }", "cpu"),
        (['ENV_VALUES[MILVUS_DEVICE]="cuda"'], "nvidia-smi() { return 1; }", "cuda"),
        ([], "nvidia-smi() { return 0; }", "cuda"),
    ],
    ids=["saved-cpu-wins", "saved-cuda-wins", "gpu-host-defaults-to-cuda"],
)
def test_collect_milvus_config_resolves_device_default_for_local_docker(
    setup_lines: list[str], nvidia_impl: str, expected_device: str
) -> None:
    """Milvus device defaults should prefer saved state and otherwise use host CUDA detection."""
    setup_block = "\n".join(setup_lines)
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

{setup_block}
{nvidia_impl}

confirm_default_yes() {{ return 0; }}
prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}

collect_milvus_config yes

printf 'MILVUS_DEVICE=%s\\n' "${{ENV_VALUES[MILVUS_DEVICE]}}\"
""")
    assert values["MILVUS_DEVICE"] == expected_device


@pytest.mark.parametrize(
    ("collector_call", "device_prompt", "endpoint_prompt"),
    [
        ("collect_milvus_config yes", "Milvus device", "Milvus URI"),
        ("collect_qdrant_config yes", "Qdrant device", "Qdrant URL"),
    ],
)
def test_local_vector_db_device_prompt_is_first_follow_up_after_docker_choice(
    collector_call: str, device_prompt: str, endpoint_prompt: str
) -> None:
    """GPU/CPU selection should be the first prompt after choosing local Docker deployment."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

PROMPT_LOG_FILE="$(mktemp)"
: > "$PROMPT_LOG_FILE"

confirm_default_yes() {{ return 0; }}
prompt_choice() {{
  printf '%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}
prompt_with_default() {{
  printf '%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}
prompt_until_valid() {{
  printf '%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}

{collector_call}

printf 'PROMPT_LOG=%s\\n' "$(paste -sd '|' "$PROMPT_LOG_FILE")"
""")
    assert values["PROMPT_LOG"].startswith(f"{device_prompt}|{endpoint_prompt}")


def test_collect_mongodb_config_local_service_strips_stale_credentials_on_rerun() -> (
    None
):
    """Bundled MongoDB should keep host `.env` aligned with the unauthenticated template."""
    values = run_bash_lines(f"""
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
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}\"
""")
    assert values["MONGO_URI"] == "mongodb://localhost:27017/?directConnection=true"
    assert (
        values["COMPOSE_MONGO_URI"] == "mongodb://mongodb:27017/?directConnection=true"
    )
    assert values["DOCKER_SERVICE"] == "mongodb"


def test_collect_mongodb_config_resets_wizard_managed_local_uri_when_switching_off_docker() -> (
    None
):
    """Switching off bundled MongoDB should not preserve the old wizard-managed local URI."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="MongoVectorDBStorage"
ENV_VALUES[MONGO_URI]="mongodb://localhost:27017/?directConnection=true"

confirm_default_yes() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}

collect_mongodb_config yes

printf 'MONGO_URI=%s\\n' "${{ENV_VALUES[MONGO_URI]}}"
printf 'COMPOSE_MONGO_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[MONGO_URI]-}}\"
""")
    assert values["MONGO_URI"] == "mongodb+srv://cluster.example.mongodb.net/"
    assert values["COMPOSE_MONGO_URI"] == ""


def test_collect_mongodb_config_preserves_external_atlas_local_uri_when_switching_off_docker() -> (
    None
):
    """Switching off bundled MongoDB should keep an explicitly configured external Atlas Local URI."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="MongoVectorDBStorage"
ENV_VALUES[MONGO_URI]="mongodb://atlas-local.example.com:27017/LightRAG?replicaSet=rs0&directConnection=true"

confirm_default_yes() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}

collect_mongodb_config yes

printf 'MONGO_URI=%s\\n' "${{ENV_VALUES[MONGO_URI]}}"
printf 'COMPOSE_MONGO_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[MONGO_URI]-}}\"
""")
    assert (
        values["MONGO_URI"]
        == "mongodb://atlas-local.example.com:27017/LightRAG?replicaSet=rs0&directConnection=true"
    )
    assert values["COMPOSE_MONGO_URI"] == ""


def test_collect_redis_config_local_service_normalizes_custom_host_port() -> None:
    """Bundled Redis should keep host `.env` aligned with the published local port."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[REDIS_URI]="redis://localhost:6380/1"

confirm_default_yes() {{ return 0; }}
prompt_until_valid() {{ printf '%s' "$2"; }}

collect_redis_config yes

printf 'REDIS_URI=%s\\n' "${{ENV_VALUES[REDIS_URI]}}"
printf 'COMPOSE_REDIS_URI=%s\\n' "${{COMPOSE_ENV_OVERRIDES[REDIS_URI]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}\"
""")
    assert values["REDIS_URI"] == "redis://localhost:6379/1"
    assert values["COMPOSE_REDIS_URI"] == "redis://redis:6379"
    assert values["DOCKER_SERVICE"] == "redis"


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
    output = run_bash(f"""
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
printf 'WHITELIST_PATHS_SET=%s\\n' "${{ENV_VALUES[WHITELIST_PATHS]+set}}\"
""")
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
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

prompt_clearable_with_default() {{ printf '%s' "$2"; }}
prompt_clearable_secret_with_default() {{ printf '%s' "$2"; }}

collect_security_config no no

printf 'WHITELIST_PATHS_SET=%s\\n' "${{ENV_VALUES[WHITELIST_PATHS]+set}}"
printf 'WHITELIST_PATHS=%s\\n' "${{ENV_VALUES[WHITELIST_PATHS]}}\"
""")
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
    output = run_bash(f"""
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
printf 'LANGFUSE_HOST_SET=%s\\n' "${{ENV_VALUES[LANGFUSE_HOST]+set}}\"
""")
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


def test_collect_neo4j_config_bundled_service_keeps_username_editable(
    tmp_path: Path,
) -> None:
    """Bundled Neo4j should preserve editable credentials and existing database overrides."""
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
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[NEO4J_USERNAME]="custom-user"
ENV_VALUES[NEO4J_PASSWORD]="existing-password"
ENV_VALUES[NEO4J_DATABASE]="custom-db"

confirm_default_yes() {{ return 0; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_log_file="$(mktemp)"
trap 'rm -f "$prompt_log_file"' EXIT
prompt_with_default() {{
  printf '%s\\n' "$1" >> "$prompt_log_file"
  if [[ "$1" == "Neo4j database" ]]; then
    printf 'custom-db-2'
  else
    printf '%s' "$2"
  fi
}}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_neo4j_config yes
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"

printf 'NEO4J_USERNAME=%s\\n' "${{ENV_VALUES[NEO4J_USERNAME]}}"
printf 'NEO4J_PASSWORD=%s\\n' "${{ENV_VALUES[NEO4J_PASSWORD]}}"
printf 'NEO4J_DATABASE=%s\\n' "${{ENV_VALUES[NEO4J_DATABASE]}}"
printf 'DOCKER_SERVICE=%s\\n' "${{DOCKER_SERVICES[0]}}"
printf 'DATABASE_PROMPTS=%s\\n' "$(grep -c '^Neo4j database$' "$prompt_log_file" || true)\"
""")
    values = parse_lines(output)
    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )
    assert values["NEO4J_USERNAME"] == "custom-user"
    assert values["NEO4J_PASSWORD"] == "existing-password"
    assert values["NEO4J_DATABASE"] == "custom-db-2"
    assert values["DOCKER_SERVICE"] == "neo4j"
    assert values["DATABASE_PROMPTS"] == "1"
    assert (
        "NEO4J_AUTH: ${NEO4J_USERNAME:?missing}/${NEO4J_PASSWORD:?missing}"
        in generated_compose
    )
    assert 'NEO4J_dbms_default__database: "custom-db-2"' in generated_compose


def test_collect_neo4j_config_bundled_service_defaults_database_when_unset() -> None:
    """Bundled Neo4j should pin the community default database when no prior value exists."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_log_file="$(mktemp)"
trap 'rm -f "$prompt_log_file"' EXIT

confirm_default_yes() {{ return 0; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{
  printf '%s\\n' "$1" >> "$prompt_log_file"
  printf '%s' "$2"
}}
prompt_secret_until_valid_with_default() {{ printf 'secure-password'; }}

collect_neo4j_config yes

printf 'DATABASE=%s\\n' "${{ENV_VALUES[NEO4J_DATABASE]}}"
printf 'DATABASE_PROMPTS=%s\\n' "$(grep -c '^Neo4j database$' "$prompt_log_file" || true)\"
""")
    values = parse_lines(output)
    assert values["DATABASE"] == "neo4j"
    assert values["DATABASE_PROMPTS"] == "0"


def test_collect_neo4j_config_uses_existing_password_as_default_in_docker_mode() -> (
    None
):
    """Bundled Neo4j should preserve the existing password when the default is accepted."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[NEO4J_PASSWORD]="from-env-password"

confirm_default_yes() {{ return 0; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_neo4j_config yes

printf 'PASSWORD=%s\\n' "${{ENV_VALUES[NEO4J_PASSWORD]}}\"
""")
    values = parse_lines(output)
    assert values["PASSWORD"] == "from-env-password"


def test_collect_neo4j_config_uses_existing_password_as_default_in_external_mode() -> (
    None
):
    """External Neo4j should preserve the existing password when the default is accepted."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[NEO4J_PASSWORD]="from-env-password"

confirm_default_no() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}

collect_neo4j_config no

printf 'PASSWORD=%s\\n' "${{ENV_VALUES[NEO4J_PASSWORD]}}\"
""")
    values = parse_lines(output)
    assert values["PASSWORD"] == "from-env-password"


def test_collect_neo4j_config_bundled_service_reprompts_for_empty_credentials() -> None:
    """Bundled Neo4j should reject empty username and password values."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_log_file="$(mktemp)"
trap 'rm -f "$prompt_log_file"' EXIT

confirm_default_yes() {{ return 0; }}
prompt_until_valid() {{
  local prompt="$1"
  local default="$2"
  local validator="$3"
  shift 3
  local value=""

  while true; do
    if [[ "$prompt" == "Neo4j URI" ]]; then
      value="$default"
    else
      printf 'username\\n' >> "$prompt_log_file"
      if [[ "$(grep -c '^username$' "$prompt_log_file")" -eq 1 ]]; then
        value=""
      else
        value="neo4j-user"
      fi
    fi

    if "$validator" "$value" "$@"; then
      printf '%s' "$value"
      return 0
    fi
  done
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{
  local prompt="$1"
  local default="$2"
  local validator="$3"
  shift 3
  local value=""

  while true; do
    printf 'password\\n' >> "$prompt_log_file"
    if [[ "$(grep -c '^password$' "$prompt_log_file")" -eq 1 ]]; then
      value=""
    else
      value="secure-password"
    fi

    if "$validator" "$value" "$@"; then
      printf '%s' "$value"
      return 0
    fi
  done
}}

collect_neo4j_config yes

printf 'USERNAME=%s\\n' "${{ENV_VALUES[NEO4J_USERNAME]}}"
printf 'PASSWORD=%s\\n' "${{ENV_VALUES[NEO4J_PASSWORD]}}"
printf 'USERNAME_CALLS=%s\\n' "$(grep -c '^username$' "$prompt_log_file")"
printf 'PASSWORD_CALLS=%s\\n' "$(grep -c '^password$' "$prompt_log_file")\"
""")
    values = parse_lines(output)
    assert values["USERNAME"] == "neo4j-user"
    assert values["PASSWORD"] == "secure-password"
    assert values["USERNAME_CALLS"] == "2"
    assert values["PASSWORD_CALLS"] == "2"


def test_collect_neo4j_config_external_service_still_uses_standard_prompts() -> None:
    """External Neo4j setup should keep the non-Docker prompt behavior unchanged."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

prompt_log_file="$(mktemp)"
trap 'rm -f "$prompt_log_file"' EXIT

confirm_default_no() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{
  printf 'with_default\\n' >> "$prompt_log_file"
  if [[ "$1" == "Neo4j username" ]]; then
    printf 'external-user'
  elif [[ "$1" == "Neo4j database" ]]; then
    printf 'external-db'
  else
    printf '%s' "$2"
  fi
}}
prompt_secret_with_default() {{
  printf 'secret_with_default\\n' >> "$prompt_log_file"
  printf 'external-password'
}}

collect_neo4j_config no

printf 'USERNAME=%s\\n' "${{ENV_VALUES[NEO4J_USERNAME]}}"
printf 'PASSWORD=%s\\n' "${{ENV_VALUES[NEO4J_PASSWORD]}}"
printf 'DATABASE=%s\\n' "${{ENV_VALUES[NEO4J_DATABASE]}}"
printf 'USERNAME_PROMPTS=%s\\n' "$(grep -c '^with_default$' "$prompt_log_file")"
printf 'PASSWORD_PROMPTS=%s\\n' "$(grep -c '^secret_with_default$' "$prompt_log_file")\"
""")
    values = parse_lines(output)
    assert values["USERNAME"] == "external-user"
    assert values["PASSWORD"] == "external-password"
    assert values["DATABASE"] == "external-db"
    assert values["USERNAME_PROMPTS"] == "2"
    assert values["PASSWORD_PROMPTS"] == "1"


def test_collect_opensearch_config_preserves_graphlookup_auto_detection() -> None:
    """collect_opensearch_config should leave PPL graphlookup unset unless explicitly configured."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_opensearch_config "yes"

if [[ -v 'ENV_VALUES[OPENSEARCH_USE_PPL_GRAPHLOOKUP]' ]]; then
  printf 'GRAPHLOOKUP_SET=yes\\n'
else
  printf 'GRAPHLOOKUP_SET=no\\n'
fi
""")
    assert values["GRAPHLOOKUP_SET"] == "no"


def test_collect_opensearch_config_preserves_explicit_graphlookup_override() -> None:
    """collect_opensearch_config should keep an existing PPL graphlookup override."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[OPENSEARCH_USE_PPL_GRAPHLOOKUP]="true"

confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_opensearch_config "yes"
printf 'GRAPHLOOKUP=%s\\n' "${{ENV_VALUES[OPENSEARCH_USE_PPL_GRAPHLOOKUP]}}\"
""")
    assert values["GRAPHLOOKUP"] == "true"


def test_collect_opensearch_config_forces_docker_verify_certs_false() -> None:
    """collect_opensearch_config should force OPENSEARCH_VERIFY_CERTS=false for Docker."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[OPENSEARCH_USE_SSL]="false"
ENV_VALUES[OPENSEARCH_VERIFY_CERTS]="true"

confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_opensearch_config "yes"
printf 'USE_SSL=%s\\n' "${{ENV_VALUES[OPENSEARCH_USE_SSL]}}"
printf 'VERIFY_CERTS=%s\\n' "${{ENV_VALUES[OPENSEARCH_VERIFY_CERTS]}}\"
""")
    assert values["USE_SSL"] == "false"
    assert values["VERIFY_CERTS"] == "false"


def test_collect_opensearch_config_defaults_docker_tls_flags_when_unset() -> None:
    """collect_opensearch_config should supply Docker TLS defaults when .env has no values."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_opensearch_config "yes"
printf 'USE_SSL=%s\\n' "${{ENV_VALUES[OPENSEARCH_USE_SSL]}}"
printf 'VERIFY_CERTS=%s\\n' "${{ENV_VALUES[OPENSEARCH_VERIFY_CERTS]}}\"
""")
    assert values["USE_SSL"] == "true"
    assert values["VERIFY_CERTS"] == "false"


def test_collect_opensearch_config_uses_original_index_settings_as_defaults() -> None:
    """collect_opensearch_config should prefer ORIGINAL_ENV_VALUES for shard/replica defaults."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

default_log="$(mktemp)"

ORIGINAL_ENV_VALUES[OPENSEARCH_NUMBER_OF_SHARDS]="3"
ORIGINAL_ENV_VALUES[OPENSEARCH_NUMBER_OF_REPLICAS]="2"
ENV_VALUES[OPENSEARCH_NUMBER_OF_SHARDS]="9"
ENV_VALUES[OPENSEARCH_NUMBER_OF_REPLICAS]="8"

confirm_default_yes() {{ return 1; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{
  case "$1" in
    "Number of index shards"|"Number of index replicas (use 2 for 3-AZ clusters)")
      printf '%s=%s\\n' "$1" "$2" >> "$default_log"
      ;;
  esac
  printf '%s' "$2"
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_opensearch_config "no"
printf 'SHARDS=%s\\n' "${{ENV_VALUES[OPENSEARCH_NUMBER_OF_SHARDS]}}"
printf 'REPLICAS=%s\\n' "${{ENV_VALUES[OPENSEARCH_NUMBER_OF_REPLICAS]}}"
printf 'DEFAULTS=%s\\n' "$(tr '\\n' ';' < "$default_log")\"
""")
    assert values["SHARDS"] == "3"
    assert values["REPLICAS"] == "2"
    assert "Number of index shards=3;" in values["DEFAULTS"]
    assert "Number of index replicas (use 2 for 3-AZ clusters)=2;" in values["DEFAULTS"]


def test_collect_opensearch_config_validates_index_settings_during_prompt() -> None:
    """collect_opensearch_config should validate shard and replica prompts."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

validator_file="$(mktemp)"

confirm_default_yes() {{ return 1; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{
  case "$1" in
    "Number of index shards"|"Number of index replicas (use 2 for 3-AZ clusters)")
      printf '%s=%s\\n' "$1" "$3" >> "$validator_file"
      ;;
  esac
  printf '%s' "$2"
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_opensearch_config "no"
printf 'VALIDATORS=%s\\n' "$(tr '\\n' ';' < "$validator_file")\"
""")
    assert "Number of index shards=validate_positive_integer;" in values["VALIDATORS"]
    assert (
        "Number of index replicas (use 2 for 3-AZ clusters)=validate_non_negative_integer;"
        in values["VALIDATORS"]
    )


def test_collect_opensearch_config_validates_hosts_during_prompt() -> None:
    """collect_opensearch_config should validate OPENSEARCH_HOSTS at prompt time."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

validator_file="$(mktemp)"

confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{
  printf '%s' "$3" > "$validator_file"
  printf '%s' "$2"
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}

collect_opensearch_config "yes"
printf 'HOST_VALIDATOR=%s\\n' "$(cat "$validator_file")\"
""")
    assert values["HOST_VALIDATOR"] == "validate_opensearch_hosts_format"


def test_collect_opensearch_config_validates_password_during_prompt() -> None:
    """collect_opensearch_config should validate OPENSEARCH_PASSWORD at prompt time."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

validator_file="$(mktemp)"

confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{
  printf '%s' "$3" > "$validator_file"
  printf '%s' "$2"
}}

collect_opensearch_config "yes"
printf 'PASSWORD_VALIDATOR=%s\\n' "$(cat "$validator_file")\"
""")
    assert values["PASSWORD_VALIDATOR"] == "validate_opensearch_password_strength"
