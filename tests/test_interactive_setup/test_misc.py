# Regression tests for interactive setup wizard.
# Classification: keep tests here when they cover cross-cutting setup helpers, finalization/migration, backup, or security-check behavior that does not fit collect_/env_/generate_/validate_ buckets.

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.test_interactive_setup._helpers import (
    REPO_ROOT,
    assert_single_compose_backup,
    parse_lines,
    run_bash,
    run_bash_process,
    run_bash_lines,
    write_text_lines,
)

pytestmark = pytest.mark.offline


def test_prepare_compose_runtime_overrides_keeps_env_unchanged() -> None:
    """Loopback endpoints should be rewritten only for compose overrides."""
    output = run_bash(f"""
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
printf 'COMPOSE_RERANK=%s\\n' "${{COMPOSE_ENV_OVERRIDES[RERANK_BINDING_HOST]}}\"
""")
    values = parse_lines(output)
    assert values["ENV_LLM"] == "http://localhost:11434"
    assert values["ENV_EMBEDDING"] == "http://127.0.0.1:11434"
    assert values["ENV_RERANK"] == "http://localhost:8000/rerank"
    assert values["COMPOSE_LLM"] == "http://host.docker.internal:11434"
    assert values["COMPOSE_EMBEDDING"] == "http://host.docker.internal:11434"
    assert values["COMPOSE_RERANK"] == "http://host.docker.internal:8000/rerank"


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
        "\n".join(["SSL=true", f"SSL_CERTFILE={cert_path}", f"SSL_KEYFILE={key_path}"])
        + "\n",
        encoding="utf-8",
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
prepare_compose_env_overrides
stage_ssl_assets "$SSL_CERT_SOURCE_PATH" "$SSL_KEY_SOURCE_PATH"
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml\"
""")
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
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
initialize_default_storage_backends

show_summary() {{ :; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

finalize_base_setup

if validate_env_file; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""")
    values = parse_lines(output)
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "SSL_CERTFILE=/app/data/certs/server.pem" in generated_env
    assert "SSL_KEYFILE=/app/data/certs/server.key" in generated_env
    assert values["VALID"] == "yes"


def test_removing_ssl_strips_wizard_bind_mounts_from_compose(tmp_path: Path) -> None:
    """Re-running setup without SSL must remove only wizard-managed SSL mounts."""
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
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
generate_docker_compose "{tmp_path}/docker-compose.final.yml\"
""")
    result = compose_file.read_text(encoding="utf-8")
    assert "/app/data/certs/cert.pem" not in result
    assert "/app/data/certs/key.pem" not in result
    assert "./data/rag_storage:/app/data/rag_storage" in result
    assert "./data/inputs:/app/data/inputs" in result
    assert "./custom-data:/app/data/custom" in result


def test_find_generated_compose_file_prefers_final_compose_file(tmp_path: Path) -> None:
    """Compose discovery should prefer docker-compose.final.yml over legacy files."""
    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0"])
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        ["services:", "  lightrag:", "    image: final/lightrag"],
    )
    write_text_lines(
        tmp_path / "docker-compose.development.yml",
        ["services:", "  lightrag:", "    image: dev/lightrag"],
    )
    write_text_lines(
        tmp_path / "docker-compose.production.yml",
        ["services:", "  lightrag:", "    image: prod/lightrag"],
    )
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

printf 'COMPOSE=%s\\n' "$(find_generated_compose_file)\"
""")
    values = parse_lines(output)
    assert values["COMPOSE"] == str(tmp_path / "docker-compose.final.yml")


def test_find_generated_compose_file_falls_back_to_order_without_profile(
    tmp_path: Path,
) -> None:
    """Without legacy profile metadata, compose migration should use the default order."""
    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0"])
    write_text_lines(
        tmp_path / "docker-compose.development.yml",
        ["services:", "  lightrag:", "    image: dev/lightrag"],
    )
    write_text_lines(
        tmp_path / "docker-compose.production.yml",
        ["services:", "  lightrag:", "    image: prod/lightrag"],
    )
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

printf 'COMPOSE=%s\\n' "$(find_generated_compose_file)\"
""")
    values = parse_lines(output)
    assert values["COMPOSE"] == str(tmp_path / "docker-compose.development.yml")


def test_switching_both_providers_off_bedrock_clears_saved_aws_credentials(
    tmp_path: Path,
) -> None:
    """Reruns should not keep stale AWS Bedrock secrets in regenerated `.env` files."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=bedrock",
            "LLM_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0",
            "LLM_BINDING_HOST=https://bedrock.amazonaws.com",
            "EMBEDDING_BINDING=bedrock",
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
    output = run_bash(f"""
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
printf 'AWS_REGION_SET=%s\\n' "${{ENV_VALUES[AWS_REGION]+set}}\"
""")
    values = parse_lines(output)
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


def test_load_existing_env_forces_cohere_binding_for_vllm_rerank(
    tmp_path: Path,
) -> None:
    """Loading a Docker-managed vLLM rerank config should normalize the binding to cohere."""
    write_text_lines(
        tmp_path / ".env",
        [
            "RERANK_BINDING=jina",
            "LIGHTRAG_SETUP_RERANK_PROVIDER=vllm",
            "RERANK_BINDING_HOST=http://localhost:8000/rerank",
        ],
    )
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}"
printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]}}\"
""")
    assert values["RERANK_BINDING"] == "cohere"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == "vllm"


@pytest.mark.parametrize(
    ("llm_binding", "embedding_binding", "expected_llm_host", "expected_embed_host"),
    [
        ("bedrock", "bedrock", "DEFAULT_BEDROCK_ENDPOINT", "DEFAULT_BEDROCK_ENDPOINT"),
        ("gemini", "gemini", "DEFAULT_GEMINI_ENDPOINT", "DEFAULT_GEMINI_ENDPOINT"),
        ("bedrock", "openai", "DEFAULT_BEDROCK_ENDPOINT", ""),
    ],
    ids=["bedrock-both", "gemini-both", "bedrock-llm-only"],
)
def test_load_existing_env_backfills_sentinel_hosts_for_bedrock_and_gemini(
    tmp_path: Path,
    llm_binding: str,
    embedding_binding: str,
    expected_llm_host: str,
    expected_embed_host: str,
) -> None:
    """Flows that skip collect_*_config (--server, --storage) must not let env.example's openai URL leak through for sentinel-based providers."""
    write_text_lines(
        tmp_path / ".env",
        [
            f"LLM_BINDING={llm_binding}",
            f"EMBEDDING_BINDING={embedding_binding}",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env.generated"

printf 'LOADED_LLM_HOST=%s\\n' "${{ENV_VALUES[LLM_BINDING_HOST]:-}}"
printf 'LOADED_EMBED_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]:-}}\"
""")
    assert values["LOADED_LLM_HOST"] == expected_llm_host
    generated_lines = (
        (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()
    )
    llm_host_line = next(
        line for line in generated_lines if line.startswith("LLM_BINDING_HOST=")
    )
    assert llm_host_line == f"LLM_BINDING_HOST={expected_llm_host}"
    if expected_embed_host:
        assert values["LOADED_EMBED_HOST"] == expected_embed_host
        embed_host_line = next(
            line
            for line in generated_lines
            if line.startswith("EMBEDDING_BINDING_HOST=")
        )
        assert embed_host_line == f"EMBEDDING_BINDING_HOST={expected_embed_host}"


def test_finalize_base_setup_uses_compose_native_storage_endpoints_on_rerun(
    tmp_path: Path,
) -> None:
    """Preserved managed storage services should inject compose-native endpoints on base reruns."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LIGHTRAG_SETUP_NEO4J_DEPLOYMENT=docker",
            "LIGHTRAG_SETUP_MILVUS_DEPLOYMENT=docker",
            "NEO4J_URI=neo4j://localhost:7687",
            "MILVUS_URI=http://localhost:19530",
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
            "  neo4j:",
            "    image: neo4j:latest",
            "  milvus:",
            "    image: milvusdb/milvus:v2.6.11",
            "  milvus-etcd:",
            "    image: quay.io/coreos/etcd:v3.5.16",
            "  milvus-minio:",
            "    image: minio/minio:latest",
            "volumes:",
            "  neo4j_data:",
            "  milvus_data:",
            "  milvus-etcd_data:",
            "  milvus-minio_data:",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
show_summary() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
finalize_base_setup
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert 'NEO4J_URI: "neo4j://neo4j:7687"' in result
    assert 'MILVUS_URI: "http://milvus:19530"' in result
    assert 'NEO4J_URI: "neo4j://host.docker.internal:7687"' not in result
    assert 'MILVUS_URI: "http://host.docker.internal:19530"' not in result
    assert (
        """      milvus:
        condition: service_healthy"""
        in result
    )
    assert (
        """      milvus-etcd:
        condition: service_healthy"""
        not in result
    )
    assert (
        """      milvus-minio:
        condition: service_healthy"""
        not in result
    )


def test_finalize_base_setup_migrates_mongodb_to_atlas_local_for_mongo_vector_storage(
    tmp_path: Path,
) -> None:
    """Base reruns should upgrade docker-managed MongoDB to Atlas Local when Mongo vector storage needs it."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
            "LIGHTRAG_KV_STORAGE=MongoKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
            "MONGO_URI=mongodb://localhost:27017/?directConnection=true",
            "MONGO_DATABASE=LightRAG",
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
            "  mongodb:",
            "    image: mongo:8.2.4",
            "    volumes:",
            "      - mongo_data:/data/db",
            "volumes:",
            "  mongo_data:",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
show_summary() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
finalize_base_setup
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: mongodb/mongodb-atlas-local:" in result
    assert "mongo_config_data:/data/configdb" in result
    assert "mongo_mongot_data:/data/mongot" in result
    assert "image: mongo:8.2.4" not in result


def test_finalize_base_setup_rejects_invalid_preserved_mongo_vector_config(
    tmp_path: Path,
) -> None:
    """Base reruns should fail before writing when preserved Mongo vector config is invalid."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
            "LIGHTRAG_KV_STORAGE=MongoKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
            "MONGO_URI=mongodb://mongo.example.com:27017/?directConnection=true",
            "MONGO_DATABASE=LightRAG",
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
            "  mongodb:",
            "    image: mongo:8.2.4",
            "    volumes:",
            "      - mongo_data:/data/db",
            "volumes:",
            "  mongo_data:",
        ],
    )
    result = run_bash_process(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
show_summary() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
finalize_base_setup
""",
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert (
        "MongoVectorDBStorage requires the bundled Atlas Local endpoint"
        in result.stderr
    )
    assert "image: mongo:8.2.4" in (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )


def test_finalize_base_setup_drops_stale_storage_services_missing_from_env_markers(
    tmp_path: Path,
) -> None:
    """env-base should treat storage Docker state in `.env` as authoritative."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=text-embedding-3-small",
            "EMBEDDING_DIM=1536",
            "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
            "EMBEDDING_BINDING_API_KEY=sk-existing",
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
            "  redis:",
            "    image: redis:latest",
            "  qdrant:",
            "    image: qdrant/qdrant:latest",
            "volumes:",
            "  redis_data:",
            "  qdrant_data:",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
show_summary() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
confirm_default_yes() {{ return 1; }}
confirm_default_no() {{ return 1; }}
validate_sensitive_env_literals() {{ return 0; }}
finalize_base_setup
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "  lightrag:" in result
    assert "  redis:" not in result
    assert "  qdrant:" not in result
    assert "redis_data:" not in result
    assert "qdrant_data:" not in result
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env


@pytest.mark.parametrize(
    ("changed_key", "changed_value", "expected_rewrite"),
    [
        ("NEO4J_PASSWORD", "updated-password", "no"),
        ("NEO4J_DATABASE", "updated-database", "yes"),
    ],
    ids=["neo4j-password-does-not-rewrite", "neo4j-database-rewrites"],
)
def test_configure_storage_compose_rewrites_only_rewrites_neo4j_on_database_change(
    changed_key: str, changed_value: str, expected_rewrite: str
) -> None:
    """Neo4j service rewrites should be driven by database changes, not credentials."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

EXISTING_MANAGED_ROOT_SERVICE_SET[neo4j]=1
DOCKER_SERVICE_SET[neo4j]=1
ORIGINAL_ENV_VALUES[NEO4J_PASSWORD]="original-password"
ORIGINAL_ENV_VALUES[NEO4J_DATABASE]="neo4j"
ENV_VALUES[NEO4J_PASSWORD]="original-password"
ENV_VALUES[NEO4J_DATABASE]="neo4j"
ENV_VALUES[{changed_key}]="{changed_value}"

configure_storage_compose_rewrites

if [[ -n "${{COMPOSE_REWRITE_SERVICE_SET[neo4j]+set}}" ]]; then
  printf 'REWRITE=yes\\n'
else
  printf 'REWRITE=no\\n'
fi
""")
    values = parse_lines(output)
    assert values["REWRITE"] == expected_rewrite


@pytest.mark.parametrize(
    ("changed_key", "changed_value", "expected_rewrite"),
    [
        ("POSTGRES_HOST", "db.example.com", "no"),
        ("POSTGRES_PORT", "6543", "no"),
        ("POSTGRES_USER", "updated-user", "yes"),
        ("POSTGRES_PASSWORD", "updated-password", "yes"),
        ("POSTGRES_DATABASE", "updated-database", "yes"),
    ],
    ids=[
        "postgres-host-does-not-rewrite",
        "postgres-port-does-not-rewrite",
        "postgres-user-rewrites",
        "postgres-password-rewrites",
        "postgres-database-rewrites",
    ],
)
def test_configure_storage_compose_rewrites_only_rewrites_postgres_for_service_env_changes(
    changed_key: str, changed_value: str, expected_rewrite: str
) -> None:
    """Postgres service rewrites should only follow changes emitted into the postgres block."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

EXISTING_MANAGED_ROOT_SERVICE_SET[postgres]=1
DOCKER_SERVICE_SET[postgres]=1
ORIGINAL_ENV_VALUES[POSTGRES_HOST]="localhost"
ORIGINAL_ENV_VALUES[POSTGRES_PORT]="5432"
ORIGINAL_ENV_VALUES[POSTGRES_USER]="rag"
ORIGINAL_ENV_VALUES[POSTGRES_PASSWORD]="rag"
ORIGINAL_ENV_VALUES[POSTGRES_DATABASE]="lightrag"
ENV_VALUES[POSTGRES_HOST]="localhost"
ENV_VALUES[POSTGRES_PORT]="5432"
ENV_VALUES[POSTGRES_USER]="rag"
ENV_VALUES[POSTGRES_PASSWORD]="rag"
ENV_VALUES[POSTGRES_DATABASE]="lightrag"
ENV_VALUES[{changed_key}]="{changed_value}"

configure_storage_compose_rewrites

if [[ -n "${{COMPOSE_REWRITE_SERVICE_SET[postgres]+set}}" ]]; then
  printf 'REWRITE=yes\\n'
else
  printf 'REWRITE=no\\n'
fi
""")
    values = parse_lines(output)
    assert values["REWRITE"] == expected_rewrite


@pytest.mark.parametrize(
    ("vector_storage", "deployment_marker", "expected_rewrite"),
    [
        ("MongoVectorDBStorage", "docker", "yes"),
        ("NanoVectorDBStorage", "docker", "no"),
        ("MongoVectorDBStorage", "", "no"),
    ],
    ids=[
        "mongo-vector-with-docker-rewrites",
        "non-mongo-vector-does-not-rewrite",
        "mongo-vector-without-docker-does-not-rewrite",
    ],
)
def test_configure_mongodb_compose_migration_rewrite_only_runs_for_atlas_local_vector_path(
    tmp_path: Path, vector_storage: str, deployment_marker: str, expected_rewrite: str
) -> None:
    """Atlas Local migration should only run for docker-managed MongoDB vector storage."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  mongodb:",
            "    image: mongo:8.2.4",
            "    volumes:",
            "      - mongo_data:/data/db",
            "volumes:",
            "  mongo_data:",
        ],
    )
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="{vector_storage}"
ENV_VALUES[LIGHTRAG_SETUP_MONGODB_DEPLOYMENT]="{deployment_marker}"
EXISTING_MANAGED_ROOT_SERVICE_SET[mongodb]=1
DOCKER_SERVICE_SET[mongodb]=1

configure_mongodb_compose_migration_rewrite "$REPO_ROOT/docker-compose.final.yml"

if [[ -n "${{COMPOSE_REWRITE_SERVICE_SET[mongodb]+set}}" ]]; then
  printf 'REWRITE=yes\\n'
else
  printf 'REWRITE=no\\n'
fi
""")
    assert values["REWRITE"] == expected_rewrite


def test_configure_mongodb_compose_migration_rewrite_repairs_missing_mongot_volume(
    tmp_path: Path,
) -> None:
    """Atlas Local compose rewrites should repair stale MongoDB services missing mongot persistence."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  mongodb:",
            "    image: mongodb/mongodb-atlas-local:8",
            "    volumes:",
            "      - mongo_data:/data/db",
            "      - mongo_config_data:/data/configdb",
            "volumes:",
            "  mongo_data:",
            "  mongo_config_data:",
        ],
    )
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="MongoVectorDBStorage"
ENV_VALUES[LIGHTRAG_SETUP_MONGODB_DEPLOYMENT]="docker"
EXISTING_MANAGED_ROOT_SERVICE_SET[mongodb]=1
DOCKER_SERVICE_SET[mongodb]=1

configure_mongodb_compose_migration_rewrite "$REPO_ROOT/docker-compose.final.yml"

if [[ -n "${{COMPOSE_REWRITE_SERVICE_SET[mongodb]+set}}" ]]; then
  printf 'REWRITE=yes\\n'
else
  printf 'REWRITE=no\\n'
fi
""")
    assert values["REWRITE"] == "yes"


def test_switching_to_non_docker_storage_removes_stale_services_from_compose(
    tmp_path: Path,
) -> None:
    """env-storage must strip managed storage services while preserving user sidecars."""
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
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "docker-compose.yml").write_text(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_bash(f"""
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
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = compose_file.read_text(encoding="utf-8")
    assert "postgres:" not in result
    assert "neo4j:" not in result
    assert "postgres_data:" not in result
    assert "neo4j_data:" not in result
    assert "  lightrag:" in result
    assert "  sidecar:" in result
    assert "sidecar_data:" in result


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
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[{env_key}]="{env_value}"

prepare_compose_runtime_overrides

printf '{env_key}=%s\\n' "${{COMPOSE_ENV_OVERRIDES[{env_key}]}}\"
""")
    assert values[env_key] == expected_value


@pytest.mark.parametrize(
    ("host_value", "expected_port_mapping"),
    [
        ("127.0.0.1", "${HOST:-0.0.0.0}:${PORT:-9621}:9621"),
        ("192.168.1.10", "${HOST:-0.0.0.0}:${PORT:-9621}:9621"),
    ],
    ids=["loopback-bind", "lan-bind"],
)
def test_prepare_compose_runtime_overrides_normalizes_server_binding(
    host_value: str, expected_port_mapping: str
) -> None:
    """Compose runtime should keep variable-based publishing while fixing container bind values."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[HOST]="{host_value}"
ENV_VALUES[PORT]="8080"

prepare_compose_runtime_overrides

printf 'HOST=%s\\n' "${{COMPOSE_ENV_OVERRIDES[HOST]}}"
printf 'PORT=%s\\n' "${{COMPOSE_ENV_OVERRIDES[PORT]}}"
printf 'PORT_MAPPING=%s\\n' "${{LIGHTRAG_COMPOSE_SERVER_PORT_MAPPING}}\"
""")
    assert values["HOST"] == "0.0.0.0"
    assert values["PORT"] == "9621"
    assert values["PORT_MAPPING"] == expected_port_mapping


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
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_text_lines(tmp_path / ".env", ["LIGHTRAG_SETUP_MILVUS_DEPLOYMENT=docker"])
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
collect_server_config() {{ :; }}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
finalize_server_setup
""")
    result = compose_file.read_text(encoding="utf-8")
    assert "milvus" in result
    assert "milvus-etcd" in result
    assert "milvus-minio" in result
    assert (
        """      milvus:
        condition: service_healthy"""
        in result
    )
    assert (
        """      milvus-etcd:
        condition: service_healthy"""
        not in result
    )
    assert (
        """      milvus-minio:
        condition: service_healthy"""
        not in result
    )


def test_finalize_server_setup_uses_compose_native_neo4j_endpoint_on_rerun(
    tmp_path: Path,
) -> None:
    """Preserved managed services should inject compose-native endpoints on server reruns."""
    write_text_lines(
        tmp_path / ".env",
        ["LIGHTRAG_SETUP_NEO4J_DEPLOYMENT=docker", "NEO4J_URI=neo4j://localhost:7687"],
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
            "  neo4j:",
            "    image: neo4j:latest",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
show_summary() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_security_config() {{ return 0; }}
finalize_server_setup
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert 'NEO4J_URI: "neo4j://neo4j:7687"' in result
    assert 'NEO4J_URI: "neo4j://host.docker.internal:7687"' not in result


def test_finalize_server_setup_migrates_mongodb_to_atlas_local_for_mongo_vector_storage(
    tmp_path: Path,
) -> None:
    """Server reruns should upgrade docker-managed MongoDB to Atlas Local when Mongo vector storage needs it."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
            "LIGHTRAG_KV_STORAGE=MongoKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
            "MONGO_URI=mongodb://localhost:27017/?directConnection=true",
            "MONGO_DATABASE=LightRAG",
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
            "  mongodb:",
            "    image: mongo:8.2.4",
            "    volumes:",
            "      - mongo_data:/data/db",
            "volumes:",
            "  mongo_data:",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
show_summary() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_security_config() {{ return 0; }}
finalize_server_setup
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: mongodb/mongodb-atlas-local:" in result
    assert "mongo_config_data:/data/configdb" in result
    assert "mongo_mongot_data:/data/mongot" in result
    assert "image: mongo:8.2.4" not in result


def test_finalize_server_setup_rejects_invalid_preserved_mongo_vector_config(
    tmp_path: Path,
) -> None:
    """Server reruns should fail before writing when preserved Mongo vector config is invalid."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
            "LIGHTRAG_KV_STORAGE=MongoKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
            "MONGO_URI=mongodb://mongo.example.com:27017/?directConnection=true",
            "MONGO_DATABASE=LightRAG",
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
            "  mongodb:",
            "    image: mongo:8.2.4",
            "    volumes:",
            "      - mongo_data:/data/db",
            "volumes:",
            "  mongo_data:",
        ],
    )
    result = run_bash_process(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
show_summary() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_security_config() {{ return 0; }}
finalize_server_setup
""",
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert (
        "MongoVectorDBStorage requires the bundled Atlas Local endpoint"
        in result.stderr
    )
    assert "image: mongo:8.2.4" in (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )


def test_finalize_server_setup_drops_stale_managed_services_missing_from_env_markers(
    tmp_path: Path,
) -> None:
    """env-server should remove stale wizard-managed services not marked in `.env`."""
    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0", "PORT=9621"])
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
            "  redis:",
            "    image: redis:latest",
            "  vllm-embed:",
            "    image: vllm/vllm-openai:latest",
            "volumes:",
            "  redis_data:",
            "  vllm_embed_cache:",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present
show_summary() {{ :; }}
collect_server_config() {{ :; }}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
validate_sensitive_env_literals() {{ return 0; }}
validate_security_config() {{ return 0; }}
finalize_server_setup
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "  redis:" not in result
    assert "  vllm-embed:" not in result
    assert "redis_data:" not in result
    assert "vllm_embed_cache:" not in result
    assert "  lightrag:" in result
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env


def test_detect_managed_root_services_deduplicates_embedded_milvus_children(
    tmp_path: Path,
) -> None:
    """Managed service discovery should collapse Milvus child services to the root service."""
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  milvus:",
            "    image: milvusdb/milvus:v2.6.11",
            "  milvus-etcd:",
            "    image: quay.io/coreos/etcd:v3.5.16",
            "  milvus-minio:",
            "    image: minio/minio:latest",
            "  neo4j:",
            "    image: neo4j:latest",
        ],
    )
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
detect_managed_root_services "{tmp_path}/docker-compose.final.yml\"
""")
    assert output.splitlines() == ["milvus", "neo4j"]


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
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

show_summary() {{ :; }}
confirm_default_yes() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

if finalize_server_setup; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
""")
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


def test_finalize_server_setup_allows_predictable_auth_passwords_and_security_check_reports_it(
    tmp_path: Path,
) -> None:
    """Server setup should not block on weak password prefixes that belong to security audit."""
    write_text_lines(
        tmp_path / ".env",
        [
            "AUTH_ACCOUNTS=admin:Passw0rd!",
            "TOKEN_SECRET=jwt-secret",
            "WHITELIST_PATHS=/health",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

show_summary() {{ :; }}
confirm_default_yes() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

if finalize_server_setup; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
""")
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
    assert "AUTH_ACCOUNTS uses a predictable password prefix." in result.stdout


def test_finalize_server_setup_rejects_malformed_auth_accounts(tmp_path: Path) -> None:
    """Server setup should fail fast instead of persisting invalid AUTH_ACCOUNTS syntax."""
    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0"])
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

collect_server_config() {{ :; }}
collect_ssl_config() {{ :; }}
ENV_VALUES[AUTH_ACCOUNTS]="admin"
ENV_VALUES[TOKEN_SECRET]="jwt-secret"
show_summary() {{ :; }}
confirm_default_yes() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

if finalize_server_setup; then
  printf 'RESULT=success\\n'
else
  printf 'RESULT=failure\\n'
fi
printf 'ENV=%s\\n' "$(cat "$REPO_ROOT/.env")\"
""",
        cwd=tmp_path,
    )
    values = parse_lines(output)
    assert values["RESULT"] == "failure"
    assert values["ENV"] == "HOST=0.0.0.0"


def test_ssl_staging_uses_distinct_names_for_same_basename_inputs(
    tmp_path: Path,
) -> None:
    """Cert/key files with the same basename should stage to distinct paths."""
    env_example = tmp_path / "env.example"
    env_example.write_text(
        "\n".join(
            ["SSL_CERTFILE=/placeholder/cert.pem", "SSL_KEYFILE=/placeholder/key.pem"]
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
    run_bash(f"""
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
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml\"
""")
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
    run_bash(f"""
set -euo pipefail
cd "{tmp_path}"
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state

stage_ssl_assets "./data/certs/server.pem" "./data/certs/server.key\"
""")
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
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
load_existing_env_if_present

{setup_snippet}

show_summary() {{ :; }}
confirm_default_yes() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

{finalize_call}
""")
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


def test_security_check_reports_predictable_auth_password_prefix(
    tmp_path: Path,
) -> None:
    """Security audit should flag AUTH_ACCOUNTS passwords with predictable prefixes."""
    write_text_lines(
        tmp_path / ".env",
        [
            "AUTH_ACCOUNTS=admin:admin123!",
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
    assert result.returncode == 1
    assert "AUTH_ACCOUNTS uses a predictable password prefix." in result.stdout


def test_security_check_reports_api_key_only_with_default_whitelist(
    tmp_path: Path,
) -> None:
    """API-key-only deployment with unset WHITELIST_PATHS inherits /api/* and must be flagged."""
    write_text_lines(tmp_path / ".env", ["LIGHTRAG_API_KEY=my-secret-key"])
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


def test_security_check_reports_api_key_only_with_explicit_api_wildcard_whitelist(
    tmp_path: Path,
) -> None:
    """API-key-only deployment with WHITELIST_PATHS=/health,/api/* must be flagged."""
    write_text_lines(
        tmp_path / ".env",
        ["LIGHTRAG_API_KEY=my-secret-key", "WHITELIST_PATHS=/health,/api/*"],
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
    assert result.returncode == 1
    assert "WHITELIST_PATHS exposes /api routes" in result.stdout


def test_security_check_passes_for_api_key_only_with_safe_whitelist(
    tmp_path: Path,
) -> None:
    """API-key-only deployment with a safe WHITELIST_PATHS should pass the security check."""
    write_text_lines(
        tmp_path / ".env", ["LIGHTRAG_API_KEY=my-secret-key", "WHITELIST_PATHS=/health"]
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


def test_security_check_ignores_default_opensearch_password_when_opensearch_unused(
    tmp_path: Path,
) -> None:
    """Security audit should ignore OpenSearch defaults when no OpenSearch storage is selected."""
    write_text_lines(
        tmp_path / ".env",
        [
            "AUTH_ACCOUNTS=admin:secret",
            "TOKEN_SECRET=jwt-secret",
            "WHITELIST_PATHS=/health",
            "LIGHTRAG_KV_STORAGE=JsonKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
            "OPENSEARCH_PASSWORD=LightRAG2026_!@",
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
    assert "OPENSEARCH_PASSWORD uses a well-known default value." not in result.stdout


def test_security_check_reports_default_opensearch_password_when_opensearch_selected(
    tmp_path: Path,
) -> None:
    """Security audit should flag the default OpenSearch password when OpenSearch is selected."""
    write_text_lines(
        tmp_path / ".env",
        [
            "AUTH_ACCOUNTS=admin:secret",
            "TOKEN_SECRET=jwt-secret",
            "WHITELIST_PATHS=/health",
            "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
            "OPENSEARCH_HOSTS=localhost:9200",
            "OPENSEARCH_USER=admin",
            "OPENSEARCH_PASSWORD=LightRAG2026_!@",
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
    assert result.returncode == 1
    assert "OPENSEARCH_PASSWORD uses a well-known default value." in result.stdout


def test_show_summary_masks_auth_accounts() -> None:
    """Configuration summaries should not print auth account passwords."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[AUTH_ACCOUNTS]="admin:secret,reader:hunter2"
ENV_VALUES[TOKEN_SECRET]="jwt-secret"
ENV_VALUES[HOST]="0.0.0.0"

show_summary
""")
    assert "AUTH_ACCOUNTS=***" in output
    assert "TOKEN_SECRET=***" in output
    assert "admin:secret" not in output
    assert "reader:hunter2" not in output


def test_opensearch_index_validators_accept_zero_padded_values() -> None:
    """OpenSearch shard and replica validators should accept zero-padded decimals."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"

if validate_positive_integer "08"; then
  printf 'SHARDS=valid\\n'
else
  printf 'SHARDS=invalid\\n'
fi

if validate_non_negative_integer "09"; then
  printf 'REPLICAS=valid\\n'
else
  printf 'REPLICAS=invalid\\n'
fi
""")
    assert values["SHARDS"] == "valid"
    assert values["REPLICAS"] == "valid"


def test_backup_only_backs_up_env_and_generated_compose(tmp_path: Path) -> None:
    """backup_only should back up both .env and the active generated compose file."""
    compose_content = (
        "\n".join(["services:", "  lightrag:", "    image: example/lightrag:test"])
        + "\n"
    )
    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0"])
    (tmp_path / "docker-compose.final.yml").write_text(
        compose_content, encoding="utf-8"
    )
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

backup_only
""")
    env_backups = sorted(tmp_path.glob(".env.backup.*"))
    assert len(env_backups) == 1
    assert env_backups[0].read_text(encoding="utf-8") == "HOST=0.0.0.0\n"
    assert "Backed up .env to" in output
    assert "Backed up compose file to" in output
    assert_single_compose_backup(tmp_path, compose_content)


def test_backup_only_skips_compose_backup_when_no_generated_compose_exists(
    tmp_path: Path,
) -> None:
    """backup_only should still succeed when only .env exists."""
    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0"])
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

backup_only
""")
    env_backups = sorted(tmp_path.glob(".env.backup.*"))
    assert len(env_backups) == 1
    assert "Backed up .env to" in output
    assert "Backed up compose file to" not in output
    assert list(tmp_path.glob("docker-compose.backup*.yml")) == []
