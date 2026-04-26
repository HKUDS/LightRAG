# Regression tests for interactive setup wizard.
# Classification: keep tests here when they exercise env_* top-level wizard flows and their end-to-end env/compose rewrite outcomes.

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_interactive_setup._helpers import (
    REPO_ROOT,
    assert_single_compose_backup,
    parse_lines,
    run_bash,
    run_bash_process,
    run_bash_lines,
    write_storage_setup_files,
    write_text_lines,
)

pytestmark = pytest.mark.offline


def test_env_base_flow_preserves_non_inference_env_values(tmp_path: Path) -> None:
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
    output = run_bash(f"""
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
""")
    values = parse_lines(output)
    assert values["HOST"] == "127.0.0.1"
    assert values["PORT"] == "9999"
    assert values["WEBUI_TITLE"] == "Existing Title"
    assert values["WEBUI_DESCRIPTION"] == "Existing Description"
    assert values["LLM_BINDING"] == "openai"
    assert values["LLM_BINDING_API_KEY"] == "sk-existing"
    assert values["EMBEDDING_BINDING_API_KEY"] == "sk-existing"
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
    output = run_bash(f"""
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
""")
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
    """Rerunning env-base should keep saved local vLLM embedding settings."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=BAAI/custom-embed",
            "EMBEDDING_DIM=768",
            "EMBEDDING_BINDING_HOST=http://localhost:9101/v1",
            "EMBEDDING_BINDING_API_KEY=embed-key",
            "LIGHTRAG_SETUP_EMBEDDING_PROVIDER=vllm",
            "VLLM_EMBED_MODEL=BAAI/custom-embed",
            "VLLM_EMBED_PORT=9101",
            "VLLM_EMBED_DEVICE=cpu",
        ],
    )
    values = run_bash_lines(f"""
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
  printf 'EMBEDDING_DIM=%s\\n' "${{ENV_VALUES[EMBEDDING_DIM]}}"
  printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
  printf 'VLLM_EMBED_MODEL=%s\\n' "${{ENV_VALUES[VLLM_EMBED_MODEL]}}"
  printf 'VLLM_EMBED_PORT=%s\\n' "${{ENV_VALUES[VLLM_EMBED_PORT]}}"
}}

env_base_flow
""")
    assert values["EMBEDDING_MODEL"] == "BAAI/custom-embed"
    assert values["EMBEDDING_DIM"] == "768"
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
    values = run_bash_lines(f"""
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
""")
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
    values = run_bash_lines(f"""
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
""")
    assert values["VLLM_EMBED_DEVICE"] == "cpu"


def test_env_base_flow_preserves_existing_vllm_embedding_cuda_device_on_rerun(
    tmp_path: Path,
) -> None:
    """Saved vLLM embedding CUDA mode should survive env-base reruns."""
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
            "VLLM_EMBED_DEVICE=cuda",
        ],
    )
    values = run_bash_lines(f"""
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
""")
    assert values["VLLM_EMBED_DEVICE"] == "cuda"


def test_env_base_flow_defaults_new_vllm_embedding_to_cuda_on_gpu_host(
    tmp_path: Path,
) -> None:
    """Fresh local vLLM embedding setup should honor GPU auto-detection."""
    values = run_bash_lines(f"""
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
    "Run embedding model locally via Docker (vLLM)?") return 0 ;;
    "Enable reranking?") return 1 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{
  return 1
}}

finalize_base_setup() {{
  printf 'VLLM_EMBED_DEVICE=%s\\n' "${{ENV_VALUES[VLLM_EMBED_DEVICE]}}"
}}

env_base_flow
""")
    assert values["VLLM_EMBED_DEVICE"] == "cuda"


def test_env_base_flow_forced_vllm_cuda_selection_writes_cuda_devices_to_env(
    tmp_path: Path,
) -> None:
    """Forced CUDA selection should drive both .env devices and GPU compose templates."""
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / "docker-compose.yml",
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8").splitlines(),
    )

    result = run_bash_process(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

host_cuda_available() {{ return 1; }}
prompt_choice() {{
  case "$1" in
    "LLM provider") printf 'ollama' ;;
    "Embedding device"|"Rerank device") printf 'cuda' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 0 ;;
    "Enable reranking?") return 0 ;;
    "Run rerank service locally via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{
  case "$1" in
    *"The compose file will be created/updated. Continue?"*) return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_base_flow
""",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert "VLLM_EMBED_DEVICE=cuda" in generated_env
    assert "VLLM_RERANK_DEVICE=cuda" in generated_env
    assert generated_compose.count("capabilities: [gpu]") >= 2
    assert (
        "CUDA device selected for vLLM embedding but no NVIDIA driver detected on host."
        in result.stdout
    )
    assert (
        "CUDA device selected for vLLM rerank but no NVIDIA driver detected on host."
        in result.stdout
    )


def test_env_base_flow_vllm_defaults_prefer_original_env_values_on_rerun(
    tmp_path: Path,
) -> None:
    """vLLM prompt defaults should prefer the loaded `.env` snapshot over later mutations."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LLM_BINDING=openai",
            "LLM_MODEL=gpt-4o-mini",
            "LLM_BINDING_HOST=https://api.openai.com/v1",
            "LLM_BINDING_API_KEY=sk-existing",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=BAAI/original-embed",
            "EMBEDDING_DIM=1024",
            "EMBEDDING_BINDING_HOST=http://localhost:9101/v1",
            "EMBEDDING_BINDING_API_KEY=embed-key",
            "LIGHTRAG_SETUP_EMBEDDING_PROVIDER=vllm",
            "VLLM_EMBED_MODEL=BAAI/original-embed",
            "VLLM_EMBED_PORT=9101",
            "VLLM_EMBED_DEVICE=cpu",
            "RERANK_BINDING=cohere",
            "RERANK_MODEL=BAAI/original-rerank",
            "RERANK_BINDING_HOST=http://localhost:9200/rerank",
            "RERANK_BINDING_API_KEY=rerank-key",
            "LIGHTRAG_SETUP_RERANK_PROVIDER=vllm",
            "VLLM_RERANK_MODEL=BAAI/original-rerank",
            "VLLM_RERANK_PORT=9200",
            "VLLM_RERANK_DEVICE=cpu",
        ],
    )
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

nvidia-smi() {{ return 0; }}
collect_llm_config() {{
  ENV_VALUES[VLLM_EMBED_MODEL]="BAAI/mutated-embed"
  ENV_VALUES[EMBEDDING_DIM]="2048"
  ENV_VALUES[VLLM_EMBED_PORT]="9991"
  ENV_VALUES[EMBEDDING_BINDING_HOST]="http://localhost:9991/v1"
  ENV_VALUES[VLLM_EMBED_DEVICE]="cuda"
  ENV_VALUES[VLLM_RERANK_MODEL]="BAAI/mutated-rerank"
  ENV_VALUES[VLLM_RERANK_PORT]="9990"
  ENV_VALUES[RERANK_BINDING_HOST]="http://localhost:9990/rerank"
  ENV_VALUES[VLLM_RERANK_DEVICE]="cuda"
}}
prompt_choice() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{
  case "$1" in
    "Enable reranking?") return 0 ;;
    "Run embedding model locally via Docker (vLLM)?") return 0 ;;
    "Run rerank service locally via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}

finalize_base_setup() {{
  printf 'VLLM_EMBED_MODEL=%s\\n' "${{ENV_VALUES[VLLM_EMBED_MODEL]}}"
  printf 'EMBEDDING_DIM=%s\\n' "${{ENV_VALUES[EMBEDDING_DIM]}}"
  printf 'VLLM_EMBED_PORT=%s\\n' "${{ENV_VALUES[VLLM_EMBED_PORT]}}"
  printf 'EMBEDDING_BINDING_HOST=%s\\n' "${{ENV_VALUES[EMBEDDING_BINDING_HOST]}}"
  printf 'VLLM_EMBED_DEVICE=%s\\n' "${{ENV_VALUES[VLLM_EMBED_DEVICE]}}"
  printf 'VLLM_RERANK_MODEL=%s\\n' "${{ENV_VALUES[VLLM_RERANK_MODEL]}}"
  printf 'VLLM_RERANK_PORT=%s\\n' "${{ENV_VALUES[VLLM_RERANK_PORT]}}"
  printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
  printf 'VLLM_RERANK_DEVICE=%s\\n' "${{ENV_VALUES[VLLM_RERANK_DEVICE]}}"
}}

env_base_flow
""")
    assert values["VLLM_EMBED_MODEL"] == "BAAI/original-embed"
    assert values["EMBEDDING_DIM"] == "1024"
    assert values["VLLM_EMBED_PORT"] == "9101"
    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:9101/v1"
    assert values["VLLM_EMBED_DEVICE"] == "cpu"
    assert values["VLLM_RERANK_MODEL"] == "BAAI/original-rerank"
    assert values["VLLM_RERANK_PORT"] == "9200"
    assert values["RERANK_BINDING_HOST"] == "http://localhost:9200/rerank"
    assert values["VLLM_RERANK_DEVICE"] == "cpu"


def test_env_base_flow_vllm_device_prompt_is_first_after_docker_choice(
    tmp_path: Path,
) -> None:
    """vLLM should ask for device before model-specific prompts once Docker is selected."""
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

PROMPT_LOG_FILE="$(mktemp)"
: > "$PROMPT_LOG_FILE"

prompt_choice() {{
  printf '%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}
prompt_with_default() {{
  printf '%s\\n' "$1" >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 0 ;;
    "Enable reranking?") return 0 ;;
    "Run rerank service locally via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{
  return 1
}}
finalize_base_setup() {{ :; }}

env_base_flow

printf 'PROMPT_LOG=%s\\n' "$(paste -sd '|' "$PROMPT_LOG_FILE")"
""")
    prompt_log = values["PROMPT_LOG"]
    assert prompt_log.index("Embedding device") < prompt_log.index("Embedding model")
    assert prompt_log.index("Rerank device") < prompt_log.index("Rerank model")


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
        run_bash(f"""
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
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_base_flow
""")
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
    run_bash(f"""
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
    "Run LightRAG Server via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_base_flow
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert 'SSL_CERTFILE: "/app/data/certs/cert.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key.pem"' in generated_compose
    assert "./data/certs/cert.pem:/app/data/certs/cert.pem:ro" in generated_compose
    assert "./data/certs/key.pem:/app/data/certs/key.pem:ro" in generated_compose


def test_env_base_flow_preserves_existing_storage_images_on_rerun(
    tmp_path: Path,
) -> None:
    """env-base should preserve postgres and neo4j images from an existing compose rerun."""
    write_storage_setup_files(
        tmp_path,
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=docker",
            "LIGHTRAG_SETUP_NEO4J_DEPLOYMENT=docker",
        ],
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  postgres:",
            "    image: registry.example.com/postgres-for-rag:patched",
            "  neo4j:",
            "    image: registry.example.com/neo4j:custom",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

host_cuda_available() {{ return 1; }}
collect_llm_config() {{ :; }}
collect_embedding_config() {{ :; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{
  case "$1" in
    *"The compose file will be created/updated. Continue?"*) return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}

env_base_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: registry.example.com/postgres-for-rag:patched" in result
    assert "image: registry.example.com/neo4j:custom" in result


def test_env_base_flow_backs_up_legacy_generated_compose_before_rewrite(
    tmp_path: Path,
) -> None:
    """env-base should back up the active legacy compose file before regenerating final output."""
    legacy_compose = (
        "\n".join(["services:", "  lightrag:", "    image: prod/lightrag"]) + "\n"
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / ".env",
        [
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
    (tmp_path / "docker-compose.production.yml").write_text(
        legacy_compose, encoding="utf-8"
    )
    run_bash(f"""
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
    "Run LightRAG Server via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_base_flow
""")
    assert_single_compose_backup(tmp_path, legacy_compose)
    assert (tmp_path / "docker-compose.final.yml").exists()
    assert (tmp_path / "docker-compose.production.yml").read_text(
        encoding="utf-8"
    ) == legacy_compose


def test_env_base_flow_deletes_compose_when_switching_lightrag_to_host(
    tmp_path: Path,
) -> None:
    """env-base should back up and delete compose when no Docker services remain."""
    existing_compose = (
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "  redis:",
                "    image: redis:latest",
            ]
        )
        + "\n"
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
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
        ],
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        existing_compose, encoding="utf-8"
    )
    run_bash(f"""
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
confirm_default_no() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_base_flow
""")
    assert_single_compose_backup(tmp_path, existing_compose)
    assert not (tmp_path / "docker-compose.final.yml").exists()
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LIGHTRAG_RUNTIME_TARGET=host" in generated_env


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
            "env_assertions": ["LLM_BINDING=ollama", "EMBEDDING_BINDING=ollama"],
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
        run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{case_dir}"

{case['prompt_choice']}
prompt_with_default() {{ printf '%s' "$2"; }}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
{case['prompt_secret']}
confirm_default_no() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 1 ;;
    "Enable reranking?") return 1 ;;
    "Run LightRAG Server via Docker?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{
  case "$1" in
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_base_flow
""")
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
    run_bash(f"""
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
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_base_flow
validate_env_file
""")
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
    run_bash(f"""
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
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LIGHTRAG_RUNTIME_TARGET=host" in generated_env
    assert "LIGHTRAG_SETUP_PROFILE=" not in generated_env


def test_env_base_flow_registers_vllm_rerank_service_for_docker_deployment(
    tmp_path: Path,
) -> None:
    """Choosing docker rerank in env-base should add vllm-rerank to DOCKER_SERVICE_SET."""
    output = run_bash(f"""
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
""")
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
    values = run_bash_lines(f"""
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
""")
    assert values["RERANK_MODEL"] == "BAAI/custom-rerank"
    assert values["RERANK_BINDING_HOST"] == "http://localhost:9200/rerank"
    assert values["VLLM_RERANK_MODEL"] == "BAAI/custom-rerank"
    assert values["VLLM_RERANK_PORT"] == "9200"


def test_env_base_flow_does_not_repeat_rerank_docker_prompt_when_declined(
    tmp_path: Path,
) -> None:
    """Declining rerank Docker at the outer prompt should switch to endpoint-based config."""
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
    output = run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

DOCKER_PROMPT_COUNT=0
RERANK_MODEL_PROMPT_LOG="$REPO_ROOT/rerank-model-prompts.log"
: > "$RERANK_MODEL_PROMPT_LOG"

prompt_choice() {{
  case "$1" in
    "vLLM device")
      echo "unexpected vLLM device prompt" >&2
      return 91
      ;;
    *)
      printf '%s' "$2"
      ;;
  esac
}}
prompt_with_default() {{
  case "$1" in
    "vLLM rerank model")
      echo "unexpected vLLM rerank model prompt" >&2
      return 93
      ;;
    "Rerank model")
      printf 'hit
' >> "$RERANK_MODEL_PROMPT_LOG"
      printf '%s' "$2"
      return 0
      ;;
    "Rerank endpoint")
      printf '%s' "https://rerank.example.internal/rerank"
      return 0
      ;;
  esac
  printf '%s' "$2"
}}
prompt_until_valid() {{
  case "$1" in
    "vLLM rerank port")
      echo "unexpected vLLM rerank port prompt" >&2
      return 92
      ;;
  esac
  printf '%s' "$2"
}}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{ return 1; }}
confirm_default_yes() {{
  case "$1" in
    "Enable reranking?") return 0 ;;
    "Run rerank service locally via Docker?")
      DOCKER_PROMPT_COUNT=$((DOCKER_PROMPT_COUNT + 1))
      return 1
      ;;
    *) return 1 ;;
  esac
}}
collect_embedding_config() {{ :; }}

finalize_base_setup() {{
  local rerank_model_prompt_count
  rerank_model_prompt_count="$(wc -l < "$RERANK_MODEL_PROMPT_LOG" | tr -d '[:space:]')"
  printf 'DOCKER_PROMPT_COUNT=%s\\n' "$DOCKER_PROMPT_COUNT"
  printf 'RERANK_MODEL_PROMPT_COUNT=%s\\n' "$rerank_model_prompt_count"
  printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
  printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]:-}}"
}}

env_base_flow
""")
    values = parse_lines(output)
    assert values["DOCKER_PROMPT_COUNT"] == "1"
    assert values["RERANK_MODEL_PROMPT_COUNT"] == "1"
    assert values["RERANK_BINDING_HOST"] == "https://rerank.example.internal/rerank"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == ""
    assert "vLLM uses the Cohere-compatible rerank API." not in output


def test_env_base_flow_comments_rerank_setup_marker_when_switching_off_docker(
    tmp_path: Path,
) -> None:
    """Switching rerank from Docker to a non-Docker provider should drop the setup marker."""
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

prompt_choice() {{
  case "$1" in
    "Rerank provider") printf 'cohere' ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_with_default() {{
  case "$1" in
    "Rerank endpoint") printf '%s' "https://api.cohere.com/v2/rerank" ;;
    *) printf '%s' "$2" ;;
  esac
}}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_secret_with_default() {{ printf '%s' "$2"; }}
prompt_secret_until_valid_with_default() {{ printf '%s' "$2"; }}
confirm_default_no() {{
  case "$1" in
    "Run embedding model locally via Docker (vLLM)?") return 1 ;;
    "Run rerank service locally via Docker?") return 1 ;;
    "Run LightRAG Server via Docker?") return 1 ;;
    *) return 1 ;;
  esac
}}
confirm_default_yes() {{
  case "$1" in
    "Enable reranking?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_base_flow
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    active_marker_lines = [
        line
        for line in generated_env.splitlines()
        if line.startswith("LIGHTRAG_SETUP_RERANK_PROVIDER=")
    ]
    assert "RERANK_BINDING=cohere" in generated_env
    assert active_marker_lines == []


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
    values = run_bash_lines(f"""
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
  printf 'RERANK_BINDING=%s\\n' "${{ENV_VALUES[RERANK_BINDING]}}"
  printf 'RERANK_BINDING_HOST=%s\\n' "${{ENV_VALUES[RERANK_BINDING_HOST]}}"
  printf 'LIGHTRAG_SETUP_RERANK_PROVIDER=%s\\n' "${{ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]}}"
}}

env_base_flow
""")
    assert values["RERANK_BINDING"] == "cohere"
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
    values = run_bash_lines(f"""
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
""")
    assert values["VLLM_RERANK_DEVICE"] == "cpu"


def test_env_base_flow_preserves_existing_vllm_rerank_cuda_device_on_rerun(
    tmp_path: Path,
) -> None:
    """Saved vLLM rerank CUDA mode should survive env-base reruns."""
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
            "VLLM_RERANK_DEVICE=cuda",
        ],
    )
    values = run_bash_lines(f"""
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
""")
    assert values["VLLM_RERANK_DEVICE"] == "cuda"


def test_env_storage_flow_applies_selected_storage_backends(tmp_path: Path) -> None:
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
    values = run_bash_lines(f"""
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
""")
    assert values["LIGHTRAG_KV_STORAGE"] == "RedisKVStorage"
    assert values["LIGHTRAG_VECTOR_STORAGE"] == "MilvusVectorDBStorage"
    assert values["LIGHTRAG_GRAPH_STORAGE"] == "Neo4JStorage"
    assert values["LIGHTRAG_DOC_STATUS_STORAGE"] == "RedisDocStatusStorage"
    assert values["LLM_BINDING"] == "ollama"
    assert values["EMBEDDING_BINDING"] == "ollama"


def test_env_storage_flow_reuses_saved_storage_docker_default(tmp_path: Path) -> None:
    """Saved storage deployment metadata should drive the next Docker prompt default."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=docker",
            "LIGHTRAG_KV_STORAGE=PGKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=PGVectorStorage",
            "LIGHTRAG_GRAPH_STORAGE=PGGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
        ],
    )
    values = run_bash_lines(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="PGVectorStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="PGGraphStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
  REQUIRED_DB_TYPES[postgresql]=1
}}
collect_postgres_config() {{
  printf 'POSTGRES_DEFAULT_DOCKER=%s\\n' "$1"
}}
finalize_storage_setup() {{ :; }}

env_storage_flow
""")
    assert values["POSTGRES_DEFAULT_DOCKER"] == "yes"


def test_env_storage_flow_writes_storage_docker_marker_for_selected_service(
    tmp_path: Path,
) -> None:
    """Choosing a bundled storage service should persist its deployment marker in `.env`."""
    write_text_lines(
        tmp_path / ".env", ["LLM_BINDING=ollama", "EMBEDDING_BINDING=ollama"]
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

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="PGVectorStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="PGGraphStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
  REQUIRED_DB_TYPES[postgresql]=1
}}
collect_postgres_config() {{
  add_docker_service "postgres"
  ENV_VALUES[POSTGRES_HOST]="localhost"
  ENV_VALUES[POSTGRES_PORT]="5432"
  ENV_VALUES[POSTGRES_USER]="lightrag"
  ENV_VALUES[POSTGRES_PASSWORD]="secret"
  ENV_VALUES[POSTGRES_DATABASE]="lightrag"
}}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert any(
        line == "LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=docker"
        for line in generated_env.splitlines()
    )
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env


def test_env_storage_flow_writes_opensearch_docker_marker_for_selected_service(
    tmp_path: Path,
) -> None:
    """Choosing bundled OpenSearch should persist its deployment marker in `.env`."""
    write_text_lines(
        tmp_path / ".env", ["LLM_BINDING=ollama", "EMBEDDING_BINDING=ollama"]
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

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="OpenSearchKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="OpenSearchVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="OpenSearchGraphStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="OpenSearchDocStatusStorage"
  REQUIRED_DB_TYPES[opensearch]=1
}}
collect_opensearch_config() {{
  add_docker_service "opensearch"
  ENV_VALUES[OPENSEARCH_HOSTS]="localhost:9200"
  ENV_VALUES[OPENSEARCH_USER]="admin"
  ENV_VALUES[OPENSEARCH_PASSWORD]="secret"
  ENV_VALUES[OPENSEARCH_USE_SSL]="true"
  ENV_VALUES[OPENSEARCH_VERIFY_CERTS]="false"
}}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert any(
        line == "LIGHTRAG_SETUP_OPENSEARCH_DEPLOYMENT=docker"
        for line in generated_env.splitlines()
    )
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env


def test_env_storage_flow_removes_storage_docker_marker_when_switching_to_host(
    tmp_path: Path,
) -> None:
    """Choosing a host-managed storage backend should clear a previously saved Docker marker."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=docker",
            "LIGHTRAG_KV_STORAGE=PGKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=PGVectorStorage",
            "LIGHTRAG_GRAPH_STORAGE=PGGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
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

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="PGVectorStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="PGGraphStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
  REQUIRED_DB_TYPES[postgresql]=1
}}
collect_postgres_config() {{
  ENV_VALUES[POSTGRES_HOST]="localhost"
  ENV_VALUES[POSTGRES_PORT]="5432"
  ENV_VALUES[POSTGRES_USER]="lightrag"
  ENV_VALUES[POSTGRES_PASSWORD]="secret"
  ENV_VALUES[POSTGRES_DATABASE]="lightrag"
}}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert not any(
        line.startswith("LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=")
        for line in generated_env.splitlines()
    )
    assert "LIGHTRAG_RUNTIME_TARGET=host" in generated_env


def test_env_storage_flow_clears_unused_storage_docker_markers(tmp_path: Path) -> None:
    """Markers for databases no longer required by the selected backends should be removed."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=docker",
            "LIGHTRAG_KV_STORAGE=PGKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=PGVectorStorage",
            "LIGHTRAG_GRAPH_STORAGE=PGGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
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

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="JsonKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="NanoVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="NetworkXStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="JsonDocStatusStorage"
}}
collect_database_config() {{ :; }}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert not any(
        line.startswith("LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=")
        for line in generated_env.splitlines()
    )
    assert "LIGHTRAG_KV_STORAGE=JsonKVStorage" in generated_env


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
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "docker-compose.yml").write_text(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_bash(f"""
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
    *) return 1 ;;
  esac
}}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
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


def test_env_storage_flow_uses_rag_defaults_for_empty_postgres_docker_credentials(
    tmp_path: Path,
) -> None:
    """env-storage should write bundled postgres credentials when old `.env` creds are empty."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_BINDING=ollama",
                "EMBEDDING_BINDING=ollama",
                "AUTH_ACCOUNTS=admin:secret",
                "TOKEN_SECRET=jwt-secret",
                "WHITELIST_PATHS=/health",
                "POSTGRES_USER=",
                "POSTGRES_PASSWORD=",
                "POSTGRES_DATABASE=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "env.example").write_text(
        (REPO_ROOT / "env.example").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "docker-compose.yml").write_text(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_bash(
        f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

PROMPT_LOG_FILE="$(mktemp)"
: > "$PROMPT_LOG_FILE"

select_storage_backends() {{
  REQUIRED_DB_TYPES[postgresql]=1
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="PGVectorStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="PGGraphStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
}}
confirm_default_no() {{
  if [[ "$1" == "Run PostgreSQL locally via Docker?" ]]; then
    return 0
  fi
  return 1
}}
confirm_default_yes() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}
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

env_storage_flow

printf 'PROMPT_LOG=%s\\n' "$(paste -sd '|' "$PROMPT_LOG_FILE")\"
""",
        cwd=tmp_path,
    )
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert "POSTGRES_USER=rag" in generated_env
    assert "POSTGRES_PASSWORD=rag" in generated_env
    assert "POSTGRES_DATABASE=rag" in generated_env
    assert 'POSTGRES_USER: "rag"' in generated_compose
    assert 'POSTGRES_PASSWORD: "rag"' in generated_compose
    assert 'POSTGRES_DB: "rag"' in generated_compose


def test_env_storage_flow_preserves_existing_postgres_image_during_rewrite(
    tmp_path: Path,
) -> None:
    """Postgres env rewrites should keep an existing custom image."""
    write_storage_setup_files(
        tmp_path,
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "POSTGRES_USER=rag",
            "POSTGRES_PASSWORD=rag",
            "POSTGRES_DATABASE=rag",
        ],
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  postgres:",
            "    image: registry.example.com/postgres-for-rag:patched",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  REQUIRED_DB_TYPES[postgresql]=1
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="PGVectorStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="PGGraphStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
}}
collect_database_config() {{
  if [[ "$1" == "postgresql" ]]; then
    add_docker_service "postgres"
    ENV_VALUES[POSTGRES_USER]="updated-user"
  fi
}}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: registry.example.com/postgres-for-rag:patched" in result
    assert 'POSTGRES_USER: "updated-user"' in result
    assert 'POSTGRES_DB: "rag"' in result


def test_env_storage_flow_preserves_existing_neo4j_image_during_rewrite(
    tmp_path: Path,
) -> None:
    """Neo4j database rewrites should keep an existing custom image."""
    write_storage_setup_files(
        tmp_path,
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "NEO4J_USERNAME=neo4j",
            "NEO4J_PASSWORD=neo4j-password",
            "NEO4J_DATABASE=neo4j",
        ],
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  neo4j:",
            "    image: registry.example.com/neo4j:custom",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  REQUIRED_DB_TYPES[neo4j]=1
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="JsonKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="NanoVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="Neo4JStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="JsonDocStatusStorage"
}}
collect_database_config() {{
  if [[ "$1" == "neo4j" ]]; then
    add_docker_service "neo4j"
    ENV_VALUES[NEO4J_DATABASE]="updated-database"
  fi
}}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: registry.example.com/neo4j:custom" in result
    assert 'NEO4J_dbms_default__database: "updated-database"' in result


def test_env_storage_flow_preserves_existing_postgres_and_neo4j_images_on_rewrite(
    tmp_path: Path,
) -> None:
    """Concurrent postgres and neo4j rewrites should preserve both custom images."""
    write_storage_setup_files(
        tmp_path,
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "POSTGRES_USER=rag",
            "POSTGRES_PASSWORD=rag",
            "POSTGRES_DATABASE=rag",
            "NEO4J_USERNAME=neo4j",
            "NEO4J_PASSWORD=neo4j-password",
            "NEO4J_DATABASE=neo4j",
        ],
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  postgres:",
            "    image: registry.example.com/postgres-for-rag:patched",
            "  neo4j:",
            "    image: registry.example.com/neo4j:custom",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  REQUIRED_DB_TYPES[postgresql]=1
  REQUIRED_DB_TYPES[neo4j]=1
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="PGVectorStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="Neo4JStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
}}
collect_database_config() {{
  case "$1" in
    postgresql)
      add_docker_service "postgres"
      ENV_VALUES[POSTGRES_USER]="updated-user"
      ;;
    neo4j)
      add_docker_service "neo4j"
      ENV_VALUES[NEO4J_DATABASE]="updated-database"
      ;;
  esac
}}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: registry.example.com/postgres-for-rag:patched" in result
    assert "image: registry.example.com/neo4j:custom" in result
    assert 'POSTGRES_USER: "updated-user"' in result
    assert 'NEO4J_dbms_default__database: "updated-database"' in result


def test_env_storage_flow_uses_template_image_when_existing_service_has_no_image(
    tmp_path: Path,
) -> None:
    """A rewritten service without an existing image should fall back to the template image."""
    write_storage_setup_files(
        tmp_path,
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "POSTGRES_USER=rag",
            "POSTGRES_PASSWORD=rag",
            "POSTGRES_DATABASE=rag",
        ],
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  postgres:",
            "    environment:",
            '      LEGACY_SETTING: "1"',
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

select_storage_backends() {{
  REQUIRED_DB_TYPES[postgresql]=1
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="PGVectorStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="PGGraphStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
}}
collect_database_config() {{
  if [[ "$1" == "postgresql" ]]; then
    add_docker_service "postgres"
    ENV_VALUES[POSTGRES_USER]="updated-user"
  fi
}}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: gzdaniel/postgres-for-rag:16.6" in result
    assert 'POSTGRES_USER: "updated-user"' in result


def test_env_storage_flow_force_rewrite_drops_preserved_storage_images(
    tmp_path: Path,
) -> None:
    """FORCE_REWRITE_COMPOSE should bypass preserved postgres and neo4j images."""
    write_storage_setup_files(
        tmp_path,
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "POSTGRES_USER=rag",
            "POSTGRES_PASSWORD=rag",
            "POSTGRES_DATABASE=rag",
            "NEO4J_USERNAME=neo4j",
            "NEO4J_PASSWORD=neo4j-password",
            "NEO4J_DATABASE=neo4j",
        ],
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  postgres:",
            "    image: registry.example.com/postgres-for-rag:patched",
            "  neo4j:",
            "    image: registry.example.com/neo4j:custom",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
FORCE_REWRITE_COMPOSE="yes"

select_storage_backends() {{
  REQUIRED_DB_TYPES[postgresql]=1
  REQUIRED_DB_TYPES[neo4j]=1
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="PGKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="PGVectorStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="Neo4JStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="PGDocStatusStorage"
}}
collect_database_config() {{
  case "$1" in
    postgresql)
      add_docker_service "postgres"
      ENV_VALUES[POSTGRES_USER]="updated-user"
      ;;
    neo4j)
      add_docker_service "neo4j"
      ENV_VALUES[NEO4J_DATABASE]="updated-database"
      ;;
  esac
}}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: gzdaniel/postgres-for-rag:16.6" in result
    assert "image: neo4j:5-community" in result
    assert "registry.example.com/postgres-for-rag:patched" not in result
    assert "registry.example.com/neo4j:custom" not in result


def test_env_storage_flow_backs_up_existing_compose_before_rewrite(
    tmp_path: Path,
) -> None:
    """env-storage should back up the current compose file before rewriting it."""
    existing_compose = (
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    environment:",
                '      LEGACY_SETTING: "1"',
                "  postgres:",
                "    image: gzdaniel/postgres-for-rag:16.6",
            ]
        )
        + "\n"
    )
    write_text_lines(
        tmp_path / ".env", ["LLM_BINDING=openai", "EMBEDDING_BINDING=openai"]
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        existing_compose, encoding="utf-8"
    )
    run_bash(f"""
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
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    assert_single_compose_backup(tmp_path, existing_compose)
    assert (tmp_path / "docker-compose.final.yml").exists()


def test_env_storage_flow_keeps_compose_mode_for_user_sidecars(tmp_path: Path) -> None:
    """env-storage should keep LightRAG in Docker when user sidecars are present."""
    existing_compose = (
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    environment:",
                '      LEGACY_SETTING: "1"',
                "  sidecar:",
                "    image: busybox",
            ]
        )
        + "\n"
    )
    write_text_lines(
        tmp_path / ".env", ["LLM_BINDING=openai", "EMBEDDING_BINDING=openai"]
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        existing_compose, encoding="utf-8"
    )
    run_bash(f"""
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
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert_single_compose_backup(tmp_path, existing_compose)
    assert "  lightrag:" in result
    assert "  sidecar:" in result
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env


def test_env_storage_flow_preserves_mongodb_docker_marker_for_atlas_local_vector_storage(
    tmp_path: Path,
) -> None:
    """MongoDB Atlas Local vector storage should preserve the bundled Docker deployment marker."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
            "LIGHTRAG_KV_STORAGE=MongoKVStorage",
            "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
            "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
            "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
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

select_storage_backends() {{
  ENV_VALUES[LIGHTRAG_KV_STORAGE]="MongoKVStorage"
  ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="MongoVectorDBStorage"
  ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="MongoGraphStorage"
  ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="MongoDocStatusStorage"
  REQUIRED_DB_TYPES[mongodb]=1
}}
prompt_until_valid() {{ printf '%s' "$2"; }}
prompt_with_default() {{ printf '%s' "$2"; }}
validate_required_variables() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker" in generated_env
    assert "MONGO_URI=mongodb://localhost:27017/?directConnection=true" in generated_env


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
    run_bash(f"""
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
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
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
    """env-server should keep compose SSL wiring and variable-based port publishing."""
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
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

collect_server_config() {{
  ENV_VALUES[HOST]="0.0.0.0"
  ENV_VALUES[PORT]="8080"
}}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_server_flow
""")
    generated_compose = (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    )
    assert 'SSL_CERTFILE: "/app/data/certs/cert.pem"' in generated_compose
    assert 'SSL_KEYFILE: "/app/data/certs/key.pem"' in generated_compose
    assert "./data/certs/cert.pem:/app/data/certs/cert.pem:ro" in generated_compose
    assert "./data/certs/key.pem:/app/data/certs/key.pem:ro" in generated_compose
    assert 'PORT: "9621"' in generated_compose
    assert '      - "${HOST:-0.0.0.0}:${PORT:-9621}:9621"' in generated_compose


def test_env_server_flow_backs_up_existing_compose_before_rewrite(
    tmp_path: Path,
) -> None:
    """env-server should back up the current compose file before rewriting it."""
    existing_compose = (
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "    environment:",
                '      PORT: "9621"',
            ]
        )
        + "\n"
    )
    write_text_lines(tmp_path / ".env", ["HOST=0.0.0.0", "PORT=9621"])
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        existing_compose, encoding="utf-8"
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

collect_server_config() {{
  ENV_VALUES[HOST]="0.0.0.0"
  ENV_VALUES[PORT]="8080"
}}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_security_config() {{ return 0; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_server_flow
""")
    assert_single_compose_backup(tmp_path, existing_compose)
    assert (tmp_path / "docker-compose.final.yml").read_text(
        encoding="utf-8"
    ) != existing_compose


def test_env_storage_flow_drops_stale_vllm_services_missing_from_env_markers(
    tmp_path: Path,
) -> None:
    """env-storage should remove stale vLLM services unless `.env` still marks them as Docker-managed."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "RERANK_BINDING=cohere",
            "LIGHTRAG_SETUP_RERANK_PROVIDER=cohere",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "  vllm-embed:",
                "    image: vllm/vllm-openai:latest",
                "  vllm-rerank:",
                "    image: vllm/vllm-openai:latest",
                "volumes:",
                "  vllm_embed_cache:",
                "  vllm_rerank_cache:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_bash(f"""
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
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{ return 1; }}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "  vllm-embed:" not in result
    assert "  vllm-rerank:" not in result
    assert "vllm_embed_cache:" not in result
    assert "vllm_rerank_cache:" not in result
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env


def test_env_storage_flow_preserves_vllm_services_marked_in_env(tmp_path: Path) -> None:
    """env-storage should restore vLLM services from `.env` markers even without old compose entries."""
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_BINDING_HOST=http://localhost:8001/v1",
            "LIGHTRAG_SETUP_EMBEDDING_PROVIDER=vllm",
            "VLLM_EMBED_MODEL=BAAI/bge-m3",
            "VLLM_EMBED_PORT=8001",
            "VLLM_EMBED_DEVICE=cpu",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(
        tmp_path / "docker-compose.final.yml",
        ["services:", "  lightrag:", "    image: example/lightrag:test"],
    )
    run_bash(f"""
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
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{ return 1; }}
confirm_default_no() {{ return 1; }}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "  vllm-embed:" in result
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env


def test_env_storage_flow_deletes_compose_when_switching_lightrag_to_host(
    tmp_path: Path,
) -> None:
    """env-storage should back up and delete compose when no Docker services remain."""
    existing_compose = (
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "  redis:",
                "    image: redis:latest",
            ]
        )
        + "\n"
    )
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        existing_compose, encoding="utf-8"
    )
    run_bash(f"""
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
validate_mongo_vector_storage_config() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
confirm_default_yes() {{ return 1; }}
confirm_default_no() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_storage_flow
""")
    assert_single_compose_backup(tmp_path, existing_compose)
    assert not (tmp_path / "docker-compose.final.yml").exists()
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LIGHTRAG_RUNTIME_TARGET=host" in generated_env


def test_env_server_flow_preserves_existing_storage_images_on_compose_rewrite(
    tmp_path: Path,
) -> None:
    """env-server should preserve postgres and neo4j images when a compose rewrite is triggered."""
    original_compose_lines = [
        "services:",
        "  lightrag:",
        "    image: example/lightrag:test",
        "    environment:",
        '      PORT: "9621"',
        "  postgres:",
        "    image: registry.example.com/postgres-for-rag:patched",
        "  neo4j:",
        "    image: registry.example.com/neo4j:custom",
    ]
    original_compose_content = "\n".join(original_compose_lines) + "\n"
    write_storage_setup_files(
        tmp_path,
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "HOST=0.0.0.0",
            "PORT=9621",
            "LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=docker",
            "LIGHTRAG_SETUP_NEO4J_DEPLOYMENT=docker",
        ],
        original_compose_lines,
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

collect_server_config() {{
  ENV_VALUES[HOST]="0.0.0.0"
  ENV_VALUES[PORT]="8080"
}}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
confirm_default_yes() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 1 ;;
    *) return 0 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_auth_accounts_runtime_config() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}

env_server_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert_single_compose_backup(tmp_path, expected_content=original_compose_content)
    assert "image: registry.example.com/postgres-for-rag:patched" in result
    assert "image: registry.example.com/neo4j:custom" in result


def test_env_server_flow_preserves_existing_storage_images_on_env_only_rerun(
    tmp_path: Path,
) -> None:
    """env-server write_env_only path should leave custom storage images untouched."""
    write_storage_setup_files(
        tmp_path,
        [
            "LLM_BINDING=openai",
            "EMBEDDING_BINDING=openai",
            "LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT=docker",
            "LIGHTRAG_SETUP_NEO4J_DEPLOYMENT=docker",
        ],
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "  postgres:",
            "    image: registry.example.com/postgres-for-rag:patched",
            "  neo4j:",
            "    image: registry.example.com/neo4j:custom",
        ],
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

collect_server_config() {{ :; }}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
confirm_required_yes_no() {{ return 0; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_auth_accounts_runtime_config() {{ return 0; }}
validate_mongo_vector_storage_config() {{ return 0; }}

env_server_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "image: registry.example.com/postgres-for-rag:patched" in result
    assert "image: registry.example.com/neo4j:custom" in result


def test_env_server_flow_deletes_compose_when_switching_lightrag_to_host(
    tmp_path: Path,
) -> None:
    """env-server should back up and delete compose when no managed or sidecar services remain."""
    existing_compose = (
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "  redis:",
                "    image: redis:latest",
            ]
        )
        + "\n"
    )
    write_text_lines(
        tmp_path / ".env",
        ["LIGHTRAG_RUNTIME_TARGET=compose", "HOST=0.0.0.0", "PORT=9621"],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        existing_compose, encoding="utf-8"
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

collect_server_config() {{
  ENV_VALUES[HOST]="0.0.0.0"
  ENV_VALUES[PORT]="8080"
}}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_security_config() {{ return 0; }}
confirm_default_yes() {{ return 1; }}
confirm_default_no() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_server_flow
""")
    assert_single_compose_backup(tmp_path, existing_compose)
    assert not (tmp_path / "docker-compose.final.yml").exists()
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LIGHTRAG_RUNTIME_TARGET=host" in generated_env


def test_env_server_flow_keeps_compose_mode_for_user_sidecars(tmp_path: Path) -> None:
    """env-server should keep LightRAG in Docker when compose still carries user sidecars."""
    existing_compose = (
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "  sidecar:",
                "    image: busybox",
            ]
        )
        + "\n"
    )
    write_text_lines(
        tmp_path / ".env",
        ["LIGHTRAG_RUNTIME_TARGET=compose", "HOST=0.0.0.0", "PORT=9621"],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        existing_compose, encoding="utf-8"
    )
    run_bash(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

collect_server_config() {{
  ENV_VALUES[HOST]="0.0.0.0"
  ENV_VALUES[PORT]="8080"
}}
collect_security_config() {{ :; }}
collect_ssl_config() {{ :; }}
validate_sensitive_env_literals() {{ return 0; }}
validate_security_config() {{ return 0; }}
confirm_default_yes() {{ return 0; }}
confirm_required_yes_no() {{ return 0; }}

env_server_flow
""")
    result = (tmp_path / "docker-compose.final.yml").read_text(encoding="utf-8")
    generated_env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "  sidecar:" in result
    assert "  lightrag:" in result
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in generated_env


def test_env_server_flow_rejects_invalid_ssl_cert_when_switching_to_host(
    tmp_path: Path,
) -> None:
    """finalize_server_setup should reject a missing SSL cert even when switching to host mode."""
    existing_compose = (
        "\n".join(
            [
                "services:",
                "  lightrag:",
                "    image: example/lightrag:test",
                "  redis:",
                "    image: redis:latest",
            ]
        )
        + "\n"
    )
    write_text_lines(
        tmp_path / ".env",
        [
            "LIGHTRAG_RUNTIME_TARGET=compose",
            "HOST=0.0.0.0",
            "PORT=9621",
            "SSL=true",
            "SSL_CERTFILE=/nonexistent/cert.pem",
            "SSL_KEYFILE=/nonexistent/key.pem",
        ],
    )
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "docker-compose.final.yml").write_text(
        existing_compose, encoding="utf-8"
    )
    result = run_bash_process(f"""
set -euo pipefail
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"

collect_server_config() {{ :; }}
collect_security_config() {{ :; }}
collect_ssl_config() {{
  ENV_VALUES[SSL]="true"
  SSL_CERT_SOURCE_PATH="/nonexistent/cert.pem"
  SSL_KEY_SOURCE_PATH="/nonexistent/key.pem"
}}
validate_sensitive_env_literals() {{ return 0; }}
validate_security_config() {{ return 0; }}
confirm_default_yes() {{ return 1; }}
confirm_default_no() {{
  case "$1" in
    "All wizard-managed services have been removed. Remove LightRAG from Docker and switch to host mode?") return 0 ;;
    *) return 1 ;;
  esac
}}
confirm_required_yes_no() {{ return 0; }}

env_server_flow
""")
    assert result.returncode != 0
    assert (
        "Invalid SSL_CERTFILE" in result.stderr
        or "Invalid SSL_CERTFILE" in result.stdout
    )
    assert (tmp_path / "docker-compose.final.yml").exists()
    assert "LIGHTRAG_RUNTIME_TARGET=compose" in (tmp_path / ".env").read_text(
        encoding="utf-8"
    )
