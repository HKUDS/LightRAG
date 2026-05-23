# Regression tests for interactive setup wizard.
# Classification: keep tests here when they verify validate_* and related checks that accept or reject final env/security/runtime configuration values.

from __future__ import annotations

import subprocess
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


def test_validate_env_file_rejects_container_ssl_paths_for_host_target(
    tmp_path: Path,
) -> None:
    """host-target .env must not accept /app/data/certs/* even when the staged file exists."""
    (tmp_path / "data" / "certs").mkdir(parents=True)
    (tmp_path / "data" / "certs" / "cert.pem").write_text("cert", encoding="utf-8")
    (tmp_path / "data" / "certs" / "key.pem").write_text("key", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SSL=true",
                "SSL_CERTFILE=/app/data/certs/cert.pem",
                "SSL_KEYFILE=/app/data/certs/key.pem",
                "LIGHTRAG_RUNTIME_TARGET=host",
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


def test_validate_env_file_rejects_container_ssl_paths_for_default_host_target(
    tmp_path: Path,
) -> None:
    """Omitting LIGHTRAG_RUNTIME_TARGET defaults to host; container paths must still be rejected."""
    (tmp_path / "data" / "certs").mkdir(parents=True)
    (tmp_path / "data" / "certs" / "cert.pem").write_text("cert", encoding="utf-8")
    (tmp_path / "data" / "certs" / "key.pem").write_text("key", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SSL=true",
                "SSL_CERTFILE=/app/data/certs/cert.pem",
                "SSL_KEYFILE=/app/data/certs/key.pem",
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


def test_validate_env_file_accepts_container_ssl_paths_for_compose_target(
    tmp_path: Path,
) -> None:
    """compose-target .env may use /app/data/certs/* when the staged files exist."""
    (tmp_path / "data" / "certs").mkdir(parents=True)
    (tmp_path / "data" / "certs" / "cert.pem").write_text("cert", encoding="utf-8")
    (tmp_path / "data" / "certs" / "key.pem").write_text("key", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SSL=true",
                "SSL_CERTFILE=/app/data/certs/cert.pem",
                "SSL_KEYFILE=/app/data/certs/key.pem",
                "LIGHTRAG_RUNTIME_TARGET=compose",
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
    assert result.returncode == 0


def test_validate_sensitive_env_literals_rejects_interpolation_syntax() -> None:
    """Sensitive values should reject `${...}` so default dotenv interpolation stays safe."""
    output = run_bash(f"""
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
""")
    values = parse_lines(output)
    assert values["VALID"] == "no"


def test_validate_env_file_allows_predictable_auth_passwords_and_leaves_them_to_audit(
    tmp_path: Path,
) -> None:
    """validate_env_file should allow risky-but-runnable auth settings."""
    write_text_lines(
        tmp_path / ".env",
        [
            "AUTH_ACCOUNTS=admin:Passw0rd!",
            "TOKEN_SECRET=jwt-secret",
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
    assert values["VALID"] == "yes"
    audit_result = subprocess.run(
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
    assert audit_result.returncode == 1
    assert "AUTH_ACCOUNTS uses a predictable password prefix." in audit_result.stdout


def test_validate_uri_accepts_neo4j_self_signed_tls_scheme() -> None:
    """Neo4j self-signed TLS URIs should pass validation."""
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"

if validate_uri "neo4j+ssc://db.example.com:7687" neo4j; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""")
    values = parse_lines(output)
    assert values["VALID"] == "yes"


def test_validate_security_config_rejects_malformed_auth_accounts() -> None:
    """Security validation should reject auth entries the API cannot parse."""
    output = run_bash(f"""
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

if validate_security_config 'admin:{{bcrypt}}$2b$12$abcdefghijklmnopqrstuuuuuuuuuuuuuuuuuuuuuuuuuuuu' "token-secret" "" no "/health"; then
  printf 'BCRYPT_FORMAT=yes\\n'
else
  printf 'BCRYPT_FORMAT=no\\n'
fi

if validate_security_config "admin:admin123!" "token-secret" "" no "/health"; then
  printf 'ADMIN_PREFIX=yes\\n'
else
  printf 'ADMIN_PREFIX=no\\n'
fi

if validate_security_config "admin:Passw0rd!" "token-secret" "" no "/health"; then
  printf 'PASS_PREFIX=yes\\n'
else
  printf 'PASS_PREFIX=no\\n'
fi
""")
    values = parse_lines(output)
    assert values["MISSING_COLON"] == "no"
    assert values["TRAILING_COMMA"] == "no"
    assert values["VALID_FORMAT"] == "yes"
    assert values["BCRYPT_FORMAT"] == "yes"
    assert values["ADMIN_PREFIX"] == "no"
    assert values["PASS_PREFIX"] == "no"


def test_validate_env_file_handles_supported_and_unsupported_uri_schemes(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject malformed schemes and allow supported TLS variants."""
    cases = {
        "invalid-neo4j-scheme": (
            [
                "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage",
                "NEO4J_URI=http://localhost:7687",
                "NEO4J_USERNAME=neo4j",
                "NEO4J_PASSWORD=secret",
            ],
            "no",
            "Invalid NEO4J_URI",
        ),
        "invalid-redis-scheme": (
            ["LIGHTRAG_KV_STORAGE=RedisKVStorage", "REDIS_URI=tcp://localhost:6379"],
            "no",
            "Invalid REDIS_URI",
        ),
        "valid-rediss-scheme": (
            ["LIGHTRAG_KV_STORAGE=RedisKVStorage", "REDIS_URI=rediss://localhost:6380"],
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


def test_validate_required_variables_requires_opensearch_basic_auth() -> None:
    """OpenSearch storages should require both OPENSEARCH_USER and OPENSEARCH_PASSWORD."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
reset_state

ENV_VALUES[LIGHTRAG_KV_STORAGE]="OpenSearchKVStorage"
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="OpenSearchVectorDBStorage"
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="OpenSearchGraphStorage"
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="OpenSearchDocStatusStorage"
ENV_VALUES[OPENSEARCH_HOSTS]="localhost:9200"

if validate_required_variables   "${{ENV_VALUES[LIGHTRAG_KV_STORAGE]}}"   "${{ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]}}"   "${{ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]}}"   "${{ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]}}"; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""")
    assert values["VALID"] == "no"


def test_validate_env_file_rejects_invalid_opensearch_index_settings(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject invalid OpenSearch shard and replica counts."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
            "OPENSEARCH_HOSTS=localhost:9200",
            "OPENSEARCH_USER=admin",
            "OPENSEARCH_PASSWORD=StrongPass1!",
            "OPENSEARCH_NUMBER_OF_SHARDS=abc",
            "OPENSEARCH_NUMBER_OF_REPLICAS=-1",
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
    assert "OPENSEARCH_NUMBER_OF_SHARDS must be a positive integer." in result.stderr


def test_validate_env_file_rejects_blank_opensearch_index_settings(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject blank OpenSearch shard and replica counts."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
            "OPENSEARCH_HOSTS=localhost:9200",
            "OPENSEARCH_USER=admin",
            "OPENSEARCH_PASSWORD=StrongPass1!",
            "OPENSEARCH_NUMBER_OF_SHARDS=",
            "OPENSEARCH_NUMBER_OF_REPLICAS=",
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
    assert "OPENSEARCH_NUMBER_OF_SHARDS must be a positive integer." in result.stderr


def test_validate_env_file_rejects_mongo_vector_storage_without_atlas_capable_uri(
    tmp_path: Path,
) -> None:
    """validate_env_file must reject MongoVectorDBStorage when the URI is not an Atlas cluster and no Atlas Local marker is set."""
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
    assert "MongoVectorDBStorage requires an Atlas-capable MongoDB URI" in result.stderr


def test_validate_env_file_allows_mongo_vector_storage_with_wizard_managed_atlas_local(
    tmp_path: Path,
) -> None:
    """validate_env_file should allow MongoVectorDBStorage with the bundled Atlas Local deployment."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
                "LIGHTRAG_KV_STORAGE=MongoKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
                "MONGO_URI=mongodb://localhost:27017/?directConnection=true",
                "MONGO_DATABASE=LightRAG",
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
    assert values["VALID"] == "yes"


def test_validate_env_file_allows_external_atlas_local_for_mongo_vector_storage(
    tmp_path: Path,
) -> None:
    """validate_env_file should allow Atlas Local URIs outside the wizard-managed docker path."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=MongoKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
                "MONGO_URI=mongodb://atlas-local.example.com:27017/LightRAG?replicaSet=rs0&directConnection=true",
                "MONGO_DATABASE=LightRAG",
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
    assert values["VALID"] == "yes"


def test_validate_env_file_rejects_remote_mongo_uri_with_docker_marker(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject remote mongodb:// URIs when the docker marker is set."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
                "LIGHTRAG_KV_STORAGE=MongoKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
                "MONGO_URI=mongodb://mongo.example.com:27017/?directConnection=true",
                "MONGO_DATABASE=LightRAG",
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
    assert (
        "MongoVectorDBStorage requires the bundled Atlas Local endpoint"
        in result.stderr
    )


def test_validate_env_file_rejects_stale_local_mongo_uri_without_direct_connection(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject the old local MongoDB URI format when the docker marker is set."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
                "LIGHTRAG_KV_STORAGE=MongoKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
                "MONGO_URI=mongodb://localhost:27017/",
                "MONGO_DATABASE=LightRAG",
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
    assert (
        "MongoVectorDBStorage requires the bundled Atlas Local endpoint"
        in result.stderr
    )


def test_validate_env_file_rejects_wrong_local_mongo_port_with_docker_marker(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject local MongoDB URIs that do not use the managed Atlas Local port."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
                "LIGHTRAG_KV_STORAGE=MongoKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
                "MONGO_URI=mongodb://localhost:9999/?directConnection=true",
                "MONGO_DATABASE=LightRAG",
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
    assert (
        "MongoVectorDBStorage requires the bundled Atlas Local endpoint"
        in result.stderr
    )


def test_validate_env_file_rejects_empty_opensearch_hosts(tmp_path: Path) -> None:
    """validate_env_file should reject an explicitly empty OPENSEARCH_HOSTS setting."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
                "OPENSEARCH_HOSTS=",
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
    assert "Empty OPENSEARCH_HOSTS" in result.stderr


def test_validate_env_file_rejects_whitespace_only_opensearch_hosts(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject OpenSearch host lists with only blank entries."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
                "OPENSEARCH_HOSTS=   ,   ",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=StrongPass1!",
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
    assert "OPENSEARCH_HOSTS must not contain empty host entries." in result.stderr


def test_validate_env_file_rejects_docker_opensearch_without_password(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject bundled OpenSearch when auth is incomplete."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
                "LIGHTRAG_SETUP_OPENSEARCH_DEPLOYMENT=docker",
                "OPENSEARCH_HOSTS=localhost:9200",
                "OPENSEARCH_USER=admin",
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
    assert (
        "Bundled OpenSearch requires OPENSEARCH_USER and OPENSEARCH_PASSWORD"
        in result.stderr
    )


def test_validate_env_file_rejects_weak_docker_opensearch_password(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject bundled OpenSearch passwords the image will refuse."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
                "LIGHTRAG_SETUP_OPENSEARCH_DEPLOYMENT=docker",
                "OPENSEARCH_HOSTS=localhost:9200",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=weakpass",
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
    assert "OpenSearch requires a strong OPENSEARCH_PASSWORD" in result.stderr


def test_validate_env_file_rejects_weak_host_opensearch_password(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject weak OpenSearch passwords even for host deployments."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
                "OPENSEARCH_HOSTS=localhost:9200",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=weakpass",
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
    assert "OpenSearch requires a strong OPENSEARCH_PASSWORD" in result.stderr


def test_validate_env_file_rejects_unauthenticated_host_opensearch(
    tmp_path: Path,
) -> None:
    """validate_env_file should reject host-mode OpenSearch with no auth fields."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
                "OPENSEARCH_HOSTS=localhost:9200",
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
    assert "OPENSEARCH_USER" in result.stderr
    assert "OPENSEARCH_PASSWORD" in result.stderr


def test_validate_env_file_rejects_partial_host_opensearch_auth(tmp_path: Path) -> None:
    """validate_env_file should reject host-mode OpenSearch when only one auth field is set."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
                "OPENSEARCH_HOSTS=localhost:9200",
                "OPENSEARCH_USER=admin",
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
    assert "OPENSEARCH_PASSWORD" in result.stderr


def test_validate_env_file_rejects_opensearch_hosts_with_uri_scheme(
    tmp_path: Path,
) -> None:
    """validate_env_file should require OPENSEARCH_HOSTS to stay as host:port entries."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
                "OPENSEARCH_HOSTS=https://localhost:9200",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=StrongPass1!",
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
    assert (
        "OPENSEARCH_HOSTS must use bare host:port entries, not URLs." in result.stderr
    )


def test_validate_env_file_ignores_invalid_unused_storage_settings(
    tmp_path: Path,
) -> None:
    """validate_env_file should ignore malformed settings for backends not selected by storage."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=JsonKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
                "NEO4J_URI=http://localhost:7687",
                "MONGO_URI=not-a-mongo-uri",
                "REDIS_URI=tcp://localhost:6379",
                "MILVUS_URI=tcp://localhost:19530",
                "QDRANT_URL=tcp://localhost:6333",
                "MEMGRAPH_URI=http://localhost:7687",
                "POSTGRES_PORT=99999",
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
    assert values["VALID"] == "yes"
    assert "Invalid NEO4J_URI" not in result.stderr
    assert "Invalid MONGO_URI" not in result.stderr
    assert "Invalid REDIS_URI" not in result.stderr
    assert "Invalid MILVUS_URI" not in result.stderr
    assert "Invalid QDRANT_URL" not in result.stderr
    assert "Invalid MEMGRAPH_URI" not in result.stderr
    assert "Invalid POSTGRES_PORT" not in result.stderr


def test_validate_env_file_allows_empty_opensearch_hosts_when_unused(
    tmp_path: Path,
) -> None:
    """validate_env_file should ignore blank OpenSearch hosts when no OpenSearch storage is selected."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LIGHTRAG_KV_STORAGE=JsonKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
                "OPENSEARCH_HOSTS=",
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
    assert values["VALID"] == "yes"
    assert "Empty OPENSEARCH_HOSTS" not in result.stderr
