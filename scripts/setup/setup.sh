#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSINFO+x}" || "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  echo "Error: scripts/setup/setup.sh requires Bash 4 or newer." >&2
  echo "Hint: install a newer bash and run it via 'bash scripts/setup/setup.sh ...'." >&2
  exit 1
fi

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
# shellcheck disable=SC2034
TEMPLATES_DIR="$SCRIPT_DIR/templates"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

declare -A ENV_VALUES
declare -A ORIGINAL_ENV_VALUES
declare -A COMPOSE_ENV_OVERRIDES
declare -A COMPOSE_REWRITE_SERVICE_SET
declare -A REQUIRED_DB_TYPES
declare -A DOCKER_SERVICE_SET
declare -A EXISTING_MANAGED_ROOT_SERVICE_SET
declare -a DOCKER_SERVICES
SSL_CERT_SOURCE_PATH=""
SSL_KEY_SOURCE_PATH=""
LIGHTRAG_COMPOSE_SERVER_PORT_MAPPING=""
NORMALIZED_SERVER_HOST_FOR_COMPOSE=""
FORCE_REWRITE_COMPOSE="no"
DEBUG="${DEBUG:-false}"

PRESET_VLLM_EMBEDDING=(
  "EMBEDDING_BINDING=openai"
  "EMBEDDING_BINDING_HOST=http://localhost:8001/v1"
  "EMBEDDING_MODEL=BAAI/bge-m3"
  "EMBEDDING_DIM=1024"
  "VLLM_EMBED_MODEL=BAAI/bge-m3"
  "VLLM_EMBED_PORT=8001"
  "VLLM_EMBED_DEVICE=cpu"
)

PRESET_VLLM_RERANKER=(
  "RERANK_BINDING=cohere"
  "LIGHTRAG_SETUP_RERANK_PROVIDER=vllm"
  "RERANK_MODEL=BAAI/bge-reranker-v2-m3"
  "RERANK_BINDING_HOST=http://localhost:8000/rerank"
  "VLLM_RERANK_MODEL=BAAI/bge-reranker-v2-m3"
  "VLLM_RERANK_PORT=8000"
  "VLLM_RERANK_DEVICE=cpu"
)
VLLM_SERVICES=(
  "vllm-embed"
  "vllm-rerank"
)

STORAGE_SERVICES=(
  "postgres"
  "neo4j"
  "mongodb"
  "redis"
  "milvus"
  "qdrant"
  "memgraph"
)
DEFAULT_RUNTIME_TARGET="host"
# shellcheck disable=SC2034
COLOR_RESET=""
COLOR_BOLD=""
COLOR_BLUE=""
COLOR_GREEN=""
COLOR_YELLOW=""
# shellcheck disable=SC2034
COLOR_RED=""

# shellcheck disable=SC1091
source "$LIB_DIR/storage_requirements.sh"
# shellcheck disable=SC1091
source "$LIB_DIR/validation.sh"
# shellcheck disable=SC1091
source "$LIB_DIR/prompts.sh"
# shellcheck disable=SC1091
source "$LIB_DIR/file_ops.sh"
# shellcheck disable=SC1091
source "$LIB_DIR/presets.sh"

init_colors() {
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    COLOR_RESET=$'\033[0m'
    COLOR_BOLD=$'\033[1m'
    COLOR_BLUE=$'\033[34m'
    COLOR_GREEN=$'\033[32m'
    COLOR_YELLOW=$'\033[33m'
    # shellcheck disable=SC2034
    COLOR_RED=$'\033[31m'
  fi
}

reset_state() {
  ENV_VALUES=()
  ORIGINAL_ENV_VALUES=()
  COMPOSE_ENV_OVERRIDES=()
  COMPOSE_REWRITE_SERVICE_SET=()
  REQUIRED_DB_TYPES=()
  DOCKER_SERVICE_SET=()
  EXISTING_MANAGED_ROOT_SERVICE_SET=()
  DOCKER_SERVICES=()
  SSL_CERT_SOURCE_PATH=""
  SSL_KEY_SOURCE_PATH=""
  LIGHTRAG_COMPOSE_SERVER_PORT_MAPPING=""
  NORMALIZED_SERVER_HOST_FOR_COMPOSE=""
}

validate_runtime_target() {
  local runtime_target="${1:-$DEFAULT_RUNTIME_TARGET}"

  case "$runtime_target" in
    host|compose)
      return 0
      ;;
    *)
      format_error \
        "Invalid LIGHTRAG_RUNTIME_TARGET: ${runtime_target}" \
        "Use 'host' or 'compose', or rerun the setup wizard to regenerate .env."
      return 1
      ;;
  esac
}

set_runtime_target() {
  local runtime_target="${1:-$DEFAULT_RUNTIME_TARGET}"

  if ! validate_runtime_target "$runtime_target"; then
    return 1
  fi

  ENV_VALUES["LIGHTRAG_RUNTIME_TARGET"]="$runtime_target"
}

clear_deprecated_vllm_dtype_state() {
  unset 'ENV_VALUES[VLLM_EMBED_DTYPE]'
  unset 'ENV_VALUES[VLLM_RERANK_DTYPE]'
}

load_existing_env_if_present() {
  local env_file="${REPO_ROOT}/.env"

  if [[ -f "$env_file" ]]; then
    log_debug "Loading existing .env defaults from $env_file"
    load_env_file "$env_file"
    clear_deprecated_vllm_dtype_state
    if [[ "${ENV_VALUES[SSL]:-false}" == "true" ]]; then
      SSL_CERT_SOURCE_PATH="${ENV_VALUES[SSL_CERTFILE]:-}"
      SSL_KEY_SOURCE_PATH="${ENV_VALUES[SSL_KEYFILE]:-}"
    fi

    snapshot_original_env_values
  fi
}

snapshot_original_env_values() {
  local key

  ORIGINAL_ENV_VALUES=()
  for key in "${!ENV_VALUES[@]}"; do
    ORIGINAL_ENV_VALUES["$key"]="${ENV_VALUES[$key]}"
  done
}

prepare_compose_output_from_existing() {
  local output_file="$1"
  local existing_file="$2"

  if [[ -z "$existing_file" || "$existing_file" == "$output_file" || -f "$output_file" ]]; then
    return 0
  fi

  if ! cp "$existing_file" "$output_file"; then
    format_error "Failed to prepare compose output at ${output_file}" \
      "Check file permissions and available disk space, then rerun setup."
    return 1
  fi

  log_success "Using ${existing_file} as merge input for ${output_file}"
}

log_debug() {
  if [[ "$DEBUG" == "true" ]]; then
    echo "${COLOR_YELLOW}[debug]${COLOR_RESET} $*"
  fi
}

log_info() {
  echo "${COLOR_BLUE}${COLOR_BOLD}$*${COLOR_RESET}"
}

log_warn() {
  echo "${COLOR_YELLOW}$*${COLOR_RESET}"
}

log_success() {
  echo "${COLOR_GREEN}$*${COLOR_RESET}"
}

log_step() {
  echo "${COLOR_BLUE}${COLOR_BOLD}$*${COLOR_RESET}"
}

normalize_loopback_uri_for_compose() {
  local uri="$1"

  if [[ "$uri" =~ ^([a-zA-Z][a-zA-Z0-9+.-]*://)([^/?#]+@)?(localhost|127\.0\.0\.1|0\.0\.0\.0)([/:?].*)?$ ]]; then
    printf '%s%shost.docker.internal%s' \
      "${BASH_REMATCH[1]}" \
      "${BASH_REMATCH[2]}" \
      "${BASH_REMATCH[4]}"
    return 0
  fi

  printf '%s' "$uri"
}

normalize_mongodb_uri_for_local_service() {
  local uri="$1"

  if [[ "$uri" =~ ^mongodb://([^/?#]+@)?(mongodb|localhost|127\.0\.0\.1|0\.0\.0\.0)(:[0-9]+)?([/?#].*)?$ ]]; then
    printf 'mongodb://localhost:27017%s' "${BASH_REMATCH[4]:-/}"
    return 0
  fi

  printf '%s' "$uri"
}

normalize_neo4j_uri_for_local_service() {
  local uri="$1"

  if [[ "$uri" =~ ^([a-zA-Z][a-zA-Z0-9+.-]*://)([^/?#]+@)?(neo4j|localhost|127\.0\.0\.1|0\.0\.0\.0)(:[0-9]+)?([/?#].*)?$ ]]; then
    printf '%s%slocalhost:7687%s' \
      "${BASH_REMATCH[1]}" \
      "${BASH_REMATCH[2]}" \
      "${BASH_REMATCH[5]}"
    return 0
  fi

  printf '%s' "$uri"
}

normalize_redis_uri_for_local_service() {
  local uri="$1"

  if [[ "$uri" =~ ^rediss?://([^/?#]+@)?(redis|localhost|127\.0\.0\.1|0\.0\.0\.0)(:([0-9]+))?(/.*)?$ ]]; then
    printf 'redis://localhost:6379%s' "${BASH_REMATCH[5]:-/}"
    return 0
  fi

  printf '%s' "$uri"
}

normalize_milvus_uri_for_local_service() {
  local uri="$1"

  if [[ "$uri" =~ ^(https?://)([^/?#]+@)?(milvus|localhost|127\.0\.0\.1|0\.0\.0\.0)(:[0-9]+)?([/?#].*)?$ ]]; then
    printf '%slocalhost:19530%s' \
      "${BASH_REMATCH[1]}" \
      "${BASH_REMATCH[5]}"
    return 0
  fi

  printf '%s' "$uri"
}

normalize_qdrant_uri_for_local_service() {
  local uri="$1"

  if [[ "$uri" =~ ^(https?://)([^/?#]+@)?(qdrant|localhost|127\.0\.0\.1|0\.0\.0\.0)(:[0-9]+)?([/?#].*)?$ ]]; then
    printf '%slocalhost:6333%s' \
      "${BASH_REMATCH[1]}" \
      "${BASH_REMATCH[5]}"
    return 0
  fi

  printf '%s' "$uri"
}

normalize_memgraph_uri_for_local_service() {
  local uri="$1"

  if [[ "$uri" =~ ^(bolt://)([^/?#]+@)?(memgraph|localhost|127\.0\.0\.1|0\.0\.0\.0)(:[0-9]+)?([/?#].*)?$ ]]; then
    printf 'bolt://localhost:7687%s' "${BASH_REMATCH[5]}"
    return 0
  fi

  printf '%s' "$uri"
}

normalize_loopback_host_for_compose() {
  local host="$1"

  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "0.0.0.0" ]]; then
    printf 'host.docker.internal'
    return 0
  fi

  printf '%s' "$host"
}

normalize_server_host_for_compose() {
  local host="${1:-}"
  local published_host="$host"
  local published_port="${ENV_VALUES[PORT]:-9621}"

  if [[ -z "$published_host" ]]; then
    published_host="0.0.0.0"
  elif [[ "$published_host" == "localhost" ]]; then
    published_host="127.0.0.1"
  fi

  if [[ -z "$published_port" ]]; then
    published_port="9621"
  fi

  LIGHTRAG_COMPOSE_SERVER_PORT_MAPPING="${published_host}:${published_port}:9621"

  if [[ -z "${COMPOSE_ENV_OVERRIDES[PORT]+set}" ]]; then
    if [[ "$published_port" != "9621" ]]; then
      set_compose_override "PORT" "9621"
    else
      set_compose_override "PORT" ""
    fi
  fi

  NORMALIZED_SERVER_HOST_FOR_COMPOSE="0.0.0.0"
}

host_cuda_available() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

resolve_local_device_default() {
  local configured_device="${1:-}"

  if [[ "$configured_device" == "cpu" || "$configured_device" == "cuda" ]]; then
    printf '%s' "$configured_device"
    return 0
  fi

  if host_cuda_available; then
    printf 'cuda'
  else
    printf 'cpu'
  fi
}

default_loopback_url() {
  local port="$1"
  local path="${2:-}"
  printf 'http://localhost:%s%s' "$port" "$path"
}

uri_points_to_host() {
  local uri="$1"
  shift
  local host=""
  local allowed_host

  if [[ "$uri" =~ ^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]+@)?(\[[^]]+\]|[^/:?#]+) ]]; then
    host="${BASH_REMATCH[2]}"
    for allowed_host in "$@"; do
      if [[ "$host" == "$allowed_host" ]]; then
        return 0
      fi
    done
  fi

  return 1
}

prefer_local_service_uri() {
  local current_uri="$1"
  local default_uri="$2"
  shift 2

  if [[ -z "$current_uri" ]]; then
    printf '%s' "$default_uri"
    return 0
  fi

  if uri_points_to_host "$current_uri" "$@"; then
    printf '%s' "$current_uri"
    return 0
  fi

  printf '%s' "$default_uri"
}

set_compose_override() {
  local key="$1"
  local value="${2:-}"

  if [[ -n "$value" ]]; then
    COMPOSE_ENV_OVERRIDES["$key"]="$value"
  else
    unset "COMPOSE_ENV_OVERRIDES[$key]"
  fi
}

set_managed_service_compose_overrides() {
  local root_service="$1"

  case "$root_service" in
    postgres)
      if [[ -z "${COMPOSE_ENV_OVERRIDES[POSTGRES_HOST]+set}" ]]; then
        set_compose_override "POSTGRES_HOST" "postgres"
      fi
      # The bundled postgres compose service always listens on 5432 internally.
      if [[ -z "${COMPOSE_ENV_OVERRIDES[POSTGRES_PORT]+set}" ]]; then
        set_compose_override "POSTGRES_PORT" "5432"
      fi
      ;;
    neo4j)
      if [[ -z "${COMPOSE_ENV_OVERRIDES[NEO4J_URI]+set}" ]]; then
        set_compose_override "NEO4J_URI" "neo4j://neo4j:7687"
      fi
      ;;
    mongodb)
      if [[ -z "${COMPOSE_ENV_OVERRIDES[MONGO_URI]+set}" ]]; then
        set_compose_override "MONGO_URI" "mongodb://mongodb:27017/"
      fi
      ;;
    redis)
      if [[ -z "${COMPOSE_ENV_OVERRIDES[REDIS_URI]+set}" ]]; then
        set_compose_override "REDIS_URI" "redis://redis:6379"
      fi
      ;;
    milvus)
      if [[ -z "${COMPOSE_ENV_OVERRIDES[MILVUS_URI]+set}" ]]; then
        set_compose_override "MILVUS_URI" "http://milvus:19530"
      fi
      ;;
    qdrant)
      if [[ -z "${COMPOSE_ENV_OVERRIDES[QDRANT_URL]+set}" ]]; then
        set_compose_override "QDRANT_URL" "http://qdrant:6333"
      fi
      ;;
    memgraph)
      if [[ -z "${COMPOSE_ENV_OVERRIDES[MEMGRAPH_URI]+set}" ]]; then
        set_compose_override "MEMGRAPH_URI" "bolt://memgraph:7687"
      fi
      ;;
  esac
}

prepare_compose_runtime_overrides() {
  local normalized_value
  local key
  local root_service

  # EMBEDDING_BINDING_HOST: when vllm-embed is part of this compose, the LightRAG
  # container must reach it by Docker service name, not by a loopback address.
  # This applies even when the wizard did not visit the embedding step (e.g.
  # env_server_flow), because vllm-embed is detected and added to DOCKER_SERVICE_SET
  # before prepare_compose_env_overrides is called.
  if [[ -z "${COMPOSE_ENV_OVERRIDES[EMBEDDING_BINDING_HOST]+set}" ]]; then
    if [[ -n "${DOCKER_SERVICE_SET[vllm-embed]+set}" ]]; then
      set_compose_override "EMBEDDING_BINDING_HOST" \
        "http://vllm-embed:${ENV_VALUES[VLLM_EMBED_PORT]:-8001}/v1"
    elif [[ -n "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" ]]; then
      normalized_value="$(normalize_loopback_uri_for_compose "${ENV_VALUES[EMBEDDING_BINDING_HOST]}")"
      if [[ "$normalized_value" != "${ENV_VALUES[EMBEDDING_BINDING_HOST]}" ]]; then
        set_compose_override "EMBEDDING_BINDING_HOST" "$normalized_value"
      fi
    fi
  fi

  # RERANK_BINDING_HOST: same pattern for vllm-rerank.
  if [[ -z "${COMPOSE_ENV_OVERRIDES[RERANK_BINDING_HOST]+set}" ]]; then
    if [[ -n "${DOCKER_SERVICE_SET[vllm-rerank]+set}" ]]; then
      set_compose_override "RERANK_BINDING_HOST" \
        "http://vllm-rerank:${ENV_VALUES[VLLM_RERANK_PORT]:-8000}/rerank"
    elif [[ -n "${ENV_VALUES[RERANK_BINDING_HOST]:-}" ]]; then
      normalized_value="$(normalize_loopback_uri_for_compose "${ENV_VALUES[RERANK_BINDING_HOST]}")"
      if [[ "$normalized_value" != "${ENV_VALUES[RERANK_BINDING_HOST]}" ]]; then
        set_compose_override "RERANK_BINDING_HOST" "$normalized_value"
      fi
    fi
  fi

  for root_service in postgres neo4j mongodb redis milvus qdrant memgraph; do
    if [[ -n "${DOCKER_SERVICE_SET[$root_service]+set}" ]]; then
      set_managed_service_compose_overrides "$root_service"
    fi
  done

  for key in \
    "LLM_BINDING_HOST" \
    "REDIS_URI" \
    "MONGO_URI" \
    "NEO4J_URI" \
    "MILVUS_URI" \
    "QDRANT_URL" \
    "MEMGRAPH_URI"; do
    if [[ -n "${COMPOSE_ENV_OVERRIDES[$key]+set}" ]]; then
      continue
    fi
    if [[ -n "${ENV_VALUES[$key]:-}" ]]; then
      normalized_value="$(normalize_loopback_uri_for_compose "${ENV_VALUES[$key]}")"
      if [[ "$normalized_value" != "${ENV_VALUES[$key]}" ]]; then
        set_compose_override "$key" "$normalized_value"
      fi
    fi
  done

  for key in "POSTGRES_HOST"; do
    if [[ -n "${COMPOSE_ENV_OVERRIDES[$key]+set}" ]]; then
      continue
    fi
    if [[ -n "${ENV_VALUES[$key]:-}" ]]; then
      normalized_value="$(normalize_loopback_host_for_compose "${ENV_VALUES[$key]}")"
      if [[ "$normalized_value" != "${ENV_VALUES[$key]}" ]]; then
        set_compose_override "$key" "$normalized_value"
      fi
    fi
  done

  if [[ -n "${ENV_VALUES[HOST]:-}" || -n "${ENV_VALUES[PORT]:-}" ]]; then
    normalize_server_host_for_compose "${ENV_VALUES[HOST]:-0.0.0.0}"
    normalized_value="$NORMALIZED_SERVER_HOST_FOR_COMPOSE"
    if [[ -z "${COMPOSE_ENV_OVERRIDES[HOST]+set}" && "$normalized_value" != "${ENV_VALUES[HOST]:-0.0.0.0}" ]]; then
      set_compose_override "HOST" "$normalized_value"
    fi
  fi
}

prepare_compose_ssl_overrides() {
  local cert_name=""
  local key_name=""

  if [[ -n "$SSL_CERT_SOURCE_PATH" ]]; then
    cert_name="$(resolve_staged_ssl_basename "cert" "$SSL_CERT_SOURCE_PATH" "$SSL_KEY_SOURCE_PATH")"
    set_compose_override "SSL_CERTFILE" "/app/data/certs/${cert_name}"
  fi

  if [[ -n "$SSL_KEY_SOURCE_PATH" ]]; then
    key_name="$(resolve_staged_ssl_basename "key" "$SSL_KEY_SOURCE_PATH" "$SSL_CERT_SOURCE_PATH")"
    set_compose_override "SSL_KEYFILE" "/app/data/certs/${key_name}"
  fi
}

prepare_compose_env_overrides() {
  prepare_compose_runtime_overrides
  prepare_compose_ssl_overrides
}

add_docker_service() {
  local service="$1"

  if [[ -z "${DOCKER_SERVICE_SET[$service]+set}" ]]; then
    DOCKER_SERVICE_SET["$service"]=1
    DOCKER_SERVICES+=("$service")
  fi
}

mark_compose_service_for_rewrite() {
  local service="$1"
  local root_service=""

  root_service="$(_managed_service_root_name "$service")"
  if [[ -n "$root_service" ]]; then
    COMPOSE_REWRITE_SERVICE_SET["$root_service"]=1
  fi
}

record_existing_managed_root_services() {
  local compose_file="$1"
  local service_name
  local root_service

  EXISTING_MANAGED_ROOT_SERVICE_SET=()

  if [[ -z "$compose_file" || ! -f "$compose_file" ]]; then
    return 0
  fi

  while IFS= read -r service_name; do
    root_service="$(_managed_service_root_name "$service_name")"
    if [[ -n "$root_service" ]]; then
      EXISTING_MANAGED_ROOT_SERVICE_SET["$root_service"]=1
    fi
  done < <(detect_managed_root_services "$compose_file")
}

backup_existing_compose_if_generating() {
  local generate_compose="${1:-no}"
  local existing_compose="${2:-}"
  local compose_backup_path=""

  if [[ "$generate_compose" != "yes" ]]; then
    return 0
  fi

  compose_backup_path="$(backup_compose_file "$existing_compose")" || return 1
  if [[ -n "$compose_backup_path" ]]; then
    log_success "Backed up existing compose file to $compose_backup_path"
  fi
}

existing_managed_root_service_present() {
  local root_service="$1"

  [[ -n "${EXISTING_MANAGED_ROOT_SERVICE_SET[$root_service]+set}" ]]
}

env_value_changed_from_original() {
  local key="$1"
  local missing_marker="__LIGHTRAG_MISSING__"
  local current_value="${ENV_VALUES[$key]-$missing_marker}"
  local original_value="${ORIGINAL_ENV_VALUES[$key]-$missing_marker}"

  [[ "$current_value" != "$original_value" ]]
}

any_env_value_changed_from_original() {
  local key

  for key in "$@"; do
    if env_value_changed_from_original "$key"; then
      return 0
    fi
  done

  return 1
}

compose_template_variant_for_service() {
  local service="$1"
  local snapshot="${2:-current}"
  local device=""

  case "$service" in
    milvus)
      if [[ "$snapshot" == "original" ]]; then
        device="${ORIGINAL_ENV_VALUES[MILVUS_DEVICE]:-cpu}"
      else
        device="${ENV_VALUES[MILVUS_DEVICE]:-cpu}"
      fi
      ;;
    qdrant)
      if [[ "$snapshot" == "original" ]]; then
        device="${ORIGINAL_ENV_VALUES[QDRANT_DEVICE]:-cpu}"
      else
        device="${ENV_VALUES[QDRANT_DEVICE]:-cpu}"
      fi
      ;;
    vllm-embed)
      if [[ "$snapshot" == "original" ]]; then
        device="${ORIGINAL_ENV_VALUES[VLLM_EMBED_DEVICE]:-cpu}"
      else
        device="${ENV_VALUES[VLLM_EMBED_DEVICE]:-cpu}"
      fi
      ;;
    vllm-rerank)
      if [[ "$snapshot" == "original" ]]; then
        device="${ORIGINAL_ENV_VALUES[VLLM_RERANK_DEVICE]:-cpu}"
      else
        device="${ENV_VALUES[VLLM_RERANK_DEVICE]:-cpu}"
      fi
      ;;
    *)
      printf 'default'
      return 0
      ;;
  esac

  if [[ "$device" == "cuda" ]]; then
    printf 'gpu'
  else
    printf 'cpu'
  fi
}

configure_base_compose_rewrites() {
  if [[ "$FORCE_REWRITE_COMPOSE" == "yes" ]]; then
    return 0
  fi

  if existing_managed_root_service_present "vllm-embed" && \
    [[ -n "${DOCKER_SERVICE_SET[vllm-embed]+set}" ]] && \
    [[ "$(compose_template_variant_for_service "vllm-embed" "current")" != \
      "$(compose_template_variant_for_service "vllm-embed" "original")" ]]; then
    mark_compose_service_for_rewrite "vllm-embed"
  fi

  if existing_managed_root_service_present "vllm-rerank" && \
    [[ -n "${DOCKER_SERVICE_SET[vllm-rerank]+set}" ]] && \
    [[ "$(compose_template_variant_for_service "vllm-rerank" "current")" != \
      "$(compose_template_variant_for_service "vllm-rerank" "original")" ]]; then
    mark_compose_service_for_rewrite "vllm-rerank"
  fi
}

configure_storage_compose_rewrites() {
  if [[ "$FORCE_REWRITE_COMPOSE" == "yes" ]]; then
    return 0
  fi

  if existing_managed_root_service_present "postgres" && \
    [[ -n "${DOCKER_SERVICE_SET[postgres]+set}" ]] && \
    any_env_value_changed_from_original "POSTGRES_USER" "POSTGRES_PASSWORD" "POSTGRES_DATABASE"; then
    mark_compose_service_for_rewrite "postgres"
  fi

  if existing_managed_root_service_present "neo4j" && \
    [[ -n "${DOCKER_SERVICE_SET[neo4j]+set}" ]] && \
    any_env_value_changed_from_original "NEO4J_DATABASE"; then
    mark_compose_service_for_rewrite "neo4j"
  fi

  if existing_managed_root_service_present "milvus" && \
    [[ -n "${DOCKER_SERVICE_SET[milvus]+set}" ]] && \
    [[ "$(compose_template_variant_for_service "milvus" "current")" != \
      "$(compose_template_variant_for_service "milvus" "original")" ]]; then
    mark_compose_service_for_rewrite "milvus"
  fi

  if existing_managed_root_service_present "qdrant" && \
    [[ -n "${DOCKER_SERVICE_SET[qdrant]+set}" ]] && \
    [[ "$(compose_template_variant_for_service "qdrant" "current")" != \
      "$(compose_template_variant_for_service "qdrant" "original")" ]]; then
    mark_compose_service_for_rewrite "qdrant"
  fi
}

select_storage_backends() {
  local deployment_type="$1"
  local kv_default="JsonKVStorage"
  local vector_default="NanoVectorDBStorage"
  local graph_default="NetworkXStorage"
  local doc_default="JsonDocStatusStorage"
  local kv_storage vector_storage graph_storage doc_storage

  if [[ "$deployment_type" == "production" ]]; then
    kv_default="PGKVStorage"
    vector_default="MilvusVectorDBStorage"
    graph_default="Neo4JStorage"
    doc_default="PGDocStatusStorage"
  fi

  kv_default="${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-$kv_default}"
  vector_default="${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-$vector_default}"
  graph_default="${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]:-$graph_default}"
  doc_default="${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]:-$doc_default}"

  while true; do
    kv_storage="$(prompt_choice "KV storage" "$kv_default" "${KV_STORAGE_OPTIONS[@]}")"
    vector_storage="$(prompt_choice "Vector storage" "$vector_default" "${VECTOR_STORAGE_OPTIONS[@]}")"
    graph_storage="$(prompt_choice "Graph storage" "$graph_default" "${GRAPH_STORAGE_OPTIONS[@]}")"
    doc_storage="$(prompt_choice "Doc status storage" "$doc_default" "${DOC_STATUS_STORAGE_OPTIONS[@]}")"

    if check_storage_compatibility "$kv_storage" "$vector_storage" "$graph_storage" "$doc_storage"; then
      break
    fi

    if confirm_default_no "Proceed with these storage selections anyway?"; then
      break
    fi
  done

  ENV_VALUES["LIGHTRAG_KV_STORAGE"]="$kv_storage"
  ENV_VALUES["LIGHTRAG_VECTOR_STORAGE"]="$vector_storage"
  ENV_VALUES["LIGHTRAG_GRAPH_STORAGE"]="$graph_storage"
  ENV_VALUES["LIGHTRAG_DOC_STATUS_STORAGE"]="$doc_storage"

  for storage in "$kv_storage" "$vector_storage" "$graph_storage" "$doc_storage"; do
    if [[ -n "${STORAGE_DB_TYPES[$storage]:-}" ]]; then
      REQUIRED_DB_TYPES["${STORAGE_DB_TYPES[$storage]}"]=1
    fi
  done
}

initialize_default_storage_backends() {
  # env-base does not prompt for storage, but its generated .env must remain
  # self-consistent for first-run users who have not run env-storage yet.
  ENV_VALUES["LIGHTRAG_KV_STORAGE"]="${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-JsonKVStorage}"
  ENV_VALUES["LIGHTRAG_VECTOR_STORAGE"]="${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-NanoVectorDBStorage}"
  ENV_VALUES["LIGHTRAG_GRAPH_STORAGE"]="${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]:-NetworkXStorage}"
  ENV_VALUES["LIGHTRAG_DOC_STATUS_STORAGE"]="${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]:-JsonDocStatusStorage}"
}

storage_service_name_for_db_type() {
  local db_type="$1"

  case "$db_type" in
    postgresql)
      printf 'postgres'
      ;;
    neo4j|mongodb|redis|milvus|qdrant|memgraph)
      printf '%s' "$db_type"
      ;;
    *)
      printf ''
      ;;
  esac
}

storage_deployment_marker_key() {
  local db_type="$1"

  case "$db_type" in
    postgresql)
      printf 'LIGHTRAG_SETUP_POSTGRES_DEPLOYMENT'
      ;;
    neo4j)
      printf 'LIGHTRAG_SETUP_NEO4J_DEPLOYMENT'
      ;;
    mongodb)
      printf 'LIGHTRAG_SETUP_MONGODB_DEPLOYMENT'
      ;;
    redis)
      printf 'LIGHTRAG_SETUP_REDIS_DEPLOYMENT'
      ;;
    milvus)
      printf 'LIGHTRAG_SETUP_MILVUS_DEPLOYMENT'
      ;;
    qdrant)
      printf 'LIGHTRAG_SETUP_QDRANT_DEPLOYMENT'
      ;;
    memgraph)
      printf 'LIGHTRAG_SETUP_MEMGRAPH_DEPLOYMENT'
      ;;
    *)
      printf ''
      ;;
  esac
}

storage_default_docker_for_db_type() {
  local db_type="$1"
  local marker_key

  marker_key="$(storage_deployment_marker_key "$db_type")"
  if [[ -n "$marker_key" && "${ENV_VALUES[$marker_key]:-}" == "docker" ]]; then
    printf 'yes'
  else
    printf 'no'
  fi
}

persist_storage_deployment_choice() {
  local db_type="$1"
  local deployment_mode="${2:-no}"
  local marker_key

  marker_key="$(storage_deployment_marker_key "$db_type")"
  if [[ -z "$marker_key" ]]; then
    return 0
  fi

  case "$deployment_mode" in
    yes|docker)
      ENV_VALUES["$marker_key"]="docker"
      ;;
    no|'')
      unset "ENV_VALUES[$marker_key]"
      ;;
    *)
      ENV_VALUES["$marker_key"]="$deployment_mode"
      ;;
  esac
}

clear_unused_storage_deployment_markers() {
  local db_type

  for db_type in postgresql neo4j mongodb redis milvus qdrant memgraph; do
    if [[ -z "${REQUIRED_DB_TYPES[$db_type]+set}" ]]; then
      persist_storage_deployment_choice "$db_type" "no"
    fi
  done
}

collect_database_config() {
  local db_type="$1"
  local default_docker="${2:-no}"
  local service_name=""
  local deployment_mode="no"

  case "$db_type" in
    postgresql)
      collect_postgres_config "$default_docker"
      ;;
    neo4j)
      collect_neo4j_config "$default_docker"
      ;;
    mongodb)
      collect_mongodb_config "$default_docker"
      ;;
    redis)
      collect_redis_config "$default_docker"
      ;;
    milvus)
      collect_milvus_config "$default_docker"
      ;;
    qdrant)
      collect_qdrant_config "$default_docker"
      ;;
    memgraph)
      collect_memgraph_config "$default_docker"
      ;;
    *)
      echo "Unknown database type: $db_type" >&2
      return 1
      ;;
  esac

  service_name="$(storage_service_name_for_db_type "$db_type")"
  if [[ -n "$service_name" && -n "${DOCKER_SERVICE_SET[$service_name]+set}" ]]; then
    deployment_mode="docker"
  fi
  persist_storage_deployment_choice "$db_type" "$deployment_mode"
}

collect_postgres_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local host port user password database
  local existing_user="" existing_password="" existing_database=""

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Run PostgreSQL locally via Docker?"; then
      use_docker="yes"
    fi
  else
    if confirm_default_no "Run PostgreSQL locally via Docker?"; then
      use_docker="yes"
    fi
  fi

  if [[ "$use_docker" == "yes" ]]; then
    add_docker_service "postgres"
    host="${ENV_VALUES[POSTGRES_HOST]:-localhost}"
    if [[ "$host" != "localhost" && "$host" != "127.0.0.1" && "$host" != "0.0.0.0" && "$host" != "postgres" ]]; then
      host="localhost"
    elif [[ "$host" == "postgres" ]]; then
      host="localhost"
    fi
  else
    host="${ENV_VALUES[POSTGRES_HOST]:-localhost}"
  fi

  host="$(prompt_with_default "PostgreSQL host" "$host")"
  if [[ "$use_docker" == "yes" ]]; then
    port="5432"
    set_compose_override "POSTGRES_HOST" "postgres"
    set_compose_override "POSTGRES_PORT" "5432"
  else
    port="$(prompt_until_valid "PostgreSQL port" "${ENV_VALUES[POSTGRES_PORT]:-5432}" validate_port)"
    set_compose_override "POSTGRES_HOST" ""
    set_compose_override "POSTGRES_PORT" ""
  fi

  existing_user="${ORIGINAL_ENV_VALUES[POSTGRES_USER]-${ENV_VALUES[POSTGRES_USER]:-}}"
  existing_password="${ORIGINAL_ENV_VALUES[POSTGRES_PASSWORD]-${ENV_VALUES[POSTGRES_PASSWORD]:-}}"
  existing_database="${ORIGINAL_ENV_VALUES[POSTGRES_DATABASE]-${ENV_VALUES[POSTGRES_DATABASE]:-}}"
  if [[ "$use_docker" == "yes" && -z "$existing_user" && -z "$existing_password" ]]; then
    user="rag"
    password="rag"
  else
    user="$(prompt_with_default "PostgreSQL user" "${existing_user:-rag}")"
    password="$(prompt_secret_with_default "PostgreSQL password: " "${existing_password:-rag}")"
  fi
  if [[ "$use_docker" == "yes" && -z "$existing_database" ]]; then
    database="rag"
  else
    database="$(prompt_with_default "PostgreSQL database" "${existing_database:-lightrag}")"
  fi

  ENV_VALUES["POSTGRES_HOST"]="$host"
  ENV_VALUES["POSTGRES_PORT"]="$port"
  ENV_VALUES["POSTGRES_USER"]="$user"
  ENV_VALUES["POSTGRES_PASSWORD"]="$password"
  ENV_VALUES["POSTGRES_DATABASE"]="$database"
}

collect_neo4j_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local uri username password database

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Run Neo4j locally via Docker?"; then
      use_docker="yes"
    fi
  else
    if confirm_default_no "Run Neo4j locally via Docker?"; then
      use_docker="yes"
    fi
  fi

  if [[ "$use_docker" == "yes" ]]; then
    add_docker_service "neo4j"
    uri="$(prefer_local_service_uri "${ENV_VALUES[NEO4J_URI]:-}" "neo4j://localhost:7687" "neo4j" "localhost" "127.0.0.1" "0.0.0.0")"
  else
    uri="${ENV_VALUES[NEO4J_URI]:-neo4j://localhost:7687}"
  fi

  uri="$(prompt_until_valid "Neo4j URI" "$uri" validate_uri neo4j)"
  if [[ "$use_docker" == "yes" ]]; then
    uri="$(normalize_neo4j_uri_for_local_service "$uri")"
  fi
  if [[ "$use_docker" == "yes" ]]; then
    username="$(prompt_until_valid "Neo4j username" "${ENV_VALUES[NEO4J_USERNAME]:-neo4j}" validate_non_empty)"
    password="$(prompt_secret_until_valid_with_default "Neo4j password: " "${ENV_VALUES[NEO4J_PASSWORD]:-neo4j_password}" validate_non_empty)"
    if [[ -n "${ENV_VALUES[NEO4J_DATABASE]:-}" ]]; then
      database="$(prompt_with_default "Neo4j database" "${ENV_VALUES[NEO4J_DATABASE]}")"
    else
      database="neo4j"
    fi
  else
    username="$(prompt_with_default "Neo4j username" "${ENV_VALUES[NEO4J_USERNAME]:-neo4j}")"
    password="$(prompt_secret_with_default "Neo4j password: " "${ENV_VALUES[NEO4J_PASSWORD]:-neo4j_password}")"
    database="$(prompt_with_default "Neo4j database" "${ENV_VALUES[NEO4J_DATABASE]:-neo4j}")"
  fi

  ENV_VALUES["NEO4J_URI"]="$uri"
  ENV_VALUES["NEO4J_USERNAME"]="$username"
  ENV_VALUES["NEO4J_PASSWORD"]="$password"
  ENV_VALUES["NEO4J_DATABASE"]="$database"
  if [[ "$use_docker" == "yes" ]]; then
    set_compose_override "NEO4J_URI" "neo4j://neo4j:7687"
  else
    set_compose_override "NEO4J_URI" ""
  fi
}

collect_mongodb_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local uri database
  local vector_search_required="no"

  if [[ "${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-}" == "MongoVectorDBStorage" ]]; then
    vector_search_required="yes"
  fi

  if [[ "$vector_search_required" == "yes" ]]; then
    log_warn "MongoVectorDBStorage cannot use the local Docker MongoDB service from this setup wizard."
    log_warn "Reason: the bundled local Docker MongoDB service is MongoDB Community Edition, but MongoVectorDBStorage requires Atlas Search / Vector Search support."
    log_warn "Provide a MongoDB endpoint that supports Atlas Search / Vector Search, such as MongoDB Atlas or Atlas local."
    uri="${ENV_VALUES[MONGO_URI]:-mongodb+srv://cluster.example.mongodb.net/}"
  else
    if [[ "$default_docker" == "yes" ]]; then
      if confirm_default_yes "Run MongoDB locally via Docker?"; then
        use_docker="yes"
      fi
    else
      if confirm_default_no "Run MongoDB locally via Docker?"; then
        use_docker="yes"
      fi
    fi

    if [[ "$use_docker" == "yes" ]]; then
      add_docker_service "mongodb"
      uri="$(prefer_local_service_uri "${ENV_VALUES[MONGO_URI]:-}" "mongodb://localhost:27017/" "mongodb" "localhost" "127.0.0.1" "0.0.0.0")"
    else
      uri="${ENV_VALUES[MONGO_URI]:-mongodb://localhost:27017/}"
    fi
  fi

  if [[ "$vector_search_required" == "yes" ]]; then
    uri="$(prompt_until_valid "MongoDB URI (must support Atlas Search / Vector Search)" "$uri" validate_uri mongodb)"
  else
    uri="$(prompt_until_valid "MongoDB URI" "$uri" validate_uri mongodb)"
  fi
  if [[ "$use_docker" == "yes" ]]; then
    uri="$(normalize_mongodb_uri_for_local_service "$uri")"
  fi
  database="$(prompt_with_default "MongoDB database" "${ENV_VALUES[MONGO_DATABASE]:-LightRAG}")"

  ENV_VALUES["MONGO_URI"]="$uri"
  ENV_VALUES["MONGO_DATABASE"]="$database"
  if [[ "$use_docker" == "yes" ]]; then
    set_compose_override "MONGO_URI" "mongodb://mongodb:27017/"
  else
    set_compose_override "MONGO_URI" ""
  fi
}

collect_redis_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local uri

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Run Redis locally via Docker?"; then
      use_docker="yes"
    fi
  else
    if confirm_default_no "Run Redis locally via Docker?"; then
      use_docker="yes"
    fi
  fi

  if [[ "$use_docker" == "yes" ]]; then
    add_docker_service "redis"
    uri="$(prefer_local_service_uri "${ENV_VALUES[REDIS_URI]:-}" "redis://localhost:6379" "redis" "localhost" "127.0.0.1" "0.0.0.0")"
  else
    uri="${ENV_VALUES[REDIS_URI]:-redis://localhost:6379}"
  fi

  uri="$(prompt_until_valid "Redis URI" "$uri" validate_uri redis)"
  if [[ "$use_docker" == "yes" ]]; then
    uri="$(normalize_redis_uri_for_local_service "$uri")"
  fi
  ENV_VALUES["REDIS_URI"]="$uri"
  if [[ "$use_docker" == "yes" ]]; then
    set_compose_override "REDIS_URI" "redis://redis:6379"
  else
    set_compose_override "REDIS_URI" ""
  fi
}

collect_milvus_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local uri db_name milvus_device=""

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Run Milvus locally via Docker?"; then
      use_docker="yes"
    fi
  else
    if confirm_default_no "Run Milvus locally via Docker?"; then
      use_docker="yes"
    fi
  fi

  if [[ "$use_docker" == "yes" ]]; then
    add_docker_service "milvus"
    uri="$(prefer_local_service_uri "${ENV_VALUES[MILVUS_URI]:-}" "http://localhost:19530" "milvus" "localhost" "127.0.0.1" "0.0.0.0")"
  else
    uri="${ENV_VALUES[MILVUS_URI]:-http://localhost:19530}"
  fi

  uri="$(prompt_until_valid "Milvus URI" "$uri" validate_uri milvus)"
  if [[ "$use_docker" == "yes" ]]; then
    milvus_device="$(resolve_local_device_default "${ENV_VALUES[MILVUS_DEVICE]:-}")"
    milvus_device="$(prompt_choice "Milvus device" "$milvus_device" "cpu" "cuda")"
    if [[ "$milvus_device" == "cuda" ]] && ! host_cuda_available; then
      log_warn "CUDA device selected for Milvus but no NVIDIA driver detected on host."
    fi
    uri="$(normalize_milvus_uri_for_local_service "$uri")"
    if [[ -z "${ENV_VALUES[MINIO_ACCESS_KEY_ID]:-}" ]]; then
      ENV_VALUES["MINIO_ACCESS_KEY_ID"]="minioadmin"
    fi
    if [[ -z "${ENV_VALUES[MINIO_SECRET_ACCESS_KEY]:-}" ]]; then
      ENV_VALUES["MINIO_SECRET_ACCESS_KEY"]="minioadmin"
    fi
  fi
  db_name="$(prompt_with_default "Milvus database name" "${ENV_VALUES[MILVUS_DB_NAME]:-lightrag}")"

  ENV_VALUES["MILVUS_URI"]="$uri"
  ENV_VALUES["MILVUS_DB_NAME"]="$db_name"
  if [[ -n "$milvus_device" ]]; then
    ENV_VALUES["MILVUS_DEVICE"]="$milvus_device"
  fi
  if [[ "$use_docker" == "yes" ]]; then
    set_compose_override "MILVUS_URI" "http://milvus:19530"
  else
    set_compose_override "MILVUS_URI" ""
  fi
}

collect_qdrant_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local url qdrant_device=""

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Run Qdrant locally via Docker?"; then
      use_docker="yes"
    fi
  else
    if confirm_default_no "Run Qdrant locally via Docker?"; then
      use_docker="yes"
    fi
  fi

  if [[ "$use_docker" == "yes" ]]; then
    add_docker_service "qdrant"
    url="$(prefer_local_service_uri "${ENV_VALUES[QDRANT_URL]:-}" "http://localhost:6333" "qdrant" "localhost" "127.0.0.1" "0.0.0.0")"
  else
    url="${ENV_VALUES[QDRANT_URL]:-http://localhost:6333}"
  fi

  url="$(prompt_until_valid "Qdrant URL" "$url" validate_uri qdrant)"
  if [[ "$use_docker" == "yes" ]]; then
    qdrant_device="$(resolve_local_device_default "${ENV_VALUES[QDRANT_DEVICE]:-}")"
    qdrant_device="$(prompt_choice "Qdrant device" "$qdrant_device" "cpu" "cuda")"
    if [[ "$qdrant_device" == "cuda" ]] && ! host_cuda_available; then
      log_warn "CUDA device selected for Qdrant but no NVIDIA driver detected on host."
    fi
    url="$(normalize_qdrant_uri_for_local_service "$url")"
  fi
  ENV_VALUES["QDRANT_URL"]="$url"
  if [[ -n "$qdrant_device" ]]; then
    ENV_VALUES["QDRANT_DEVICE"]="$qdrant_device"
  fi
  if [[ "$use_docker" == "yes" ]]; then
    set_compose_override "QDRANT_URL" "http://qdrant:6333"
  else
    set_compose_override "QDRANT_URL" ""
  fi
}

collect_memgraph_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local uri

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Run Memgraph locally via Docker?"; then
      use_docker="yes"
    fi
  else
    if confirm_default_no "Run Memgraph locally via Docker?"; then
      use_docker="yes"
    fi
  fi

  if [[ "$use_docker" == "yes" ]]; then
    add_docker_service "memgraph"
    uri="$(prefer_local_service_uri "${ENV_VALUES[MEMGRAPH_URI]:-}" "bolt://localhost:7687" "memgraph" "localhost" "127.0.0.1" "0.0.0.0")"
  else
    uri="${ENV_VALUES[MEMGRAPH_URI]:-bolt://localhost:7687}"
  fi

  uri="$(prompt_until_valid "Memgraph URI" "$uri" validate_uri memgraph)"
  if [[ "$use_docker" == "yes" ]]; then
    uri="$(normalize_memgraph_uri_for_local_service "$uri")"
  fi
  ENV_VALUES["MEMGRAPH_URI"]="$uri"
  if [[ "$use_docker" == "yes" ]]; then
    set_compose_override "MEMGRAPH_URI" "bolt://memgraph:7687"
  else
    set_compose_override "MEMGRAPH_URI" ""
  fi
}

clear_bedrock_credentials() {
  unset 'ENV_VALUES[AWS_ACCESS_KEY_ID]'
  unset 'ENV_VALUES[AWS_SECRET_ACCESS_KEY]'
  unset 'ENV_VALUES[AWS_SESSION_TOKEN]'
  unset 'ENV_VALUES[AWS_REGION]'
}

bedrock_binding_in_use() {
  [[ "${ENV_VALUES[LLM_BINDING]:-}" == "aws_bedrock" ||
    "${ENV_VALUES[EMBEDDING_BINDING]:-}" == "aws_bedrock" ]]
}

clear_bedrock_credentials_if_unused() {
  if ! bedrock_binding_in_use; then
    clear_bedrock_credentials
  fi
}

collect_bedrock_credentials() {
  local access_key secret_key session_token region

  log_info "Bedrock uses the AWS credential chain instead of LLM_BINDING_API_KEY/EMBEDDING_BINDING_API_KEY."
  if [[ -n "${ENV_VALUES[AWS_ACCESS_KEY_ID]:-}" && -n "${ENV_VALUES[AWS_SECRET_ACCESS_KEY]:-}" ]]; then
    if confirm_default_yes "Reuse existing AWS Bedrock credentials?"; then
      region="$(prompt_with_default "AWS region" "${ENV_VALUES[AWS_REGION]:-us-east-1}")"
      ENV_VALUES["AWS_REGION"]="$region"
      return 0
    fi
  fi

  if confirm_default_no "Store explicit AWS Bedrock credentials in .env?"; then
    access_key="$(prompt_required_secret "AWS access key ID: ")"
    secret_key="$(prompt_required_secret "AWS secret access key: ")"
    session_token="$(mask_sensitive_input "AWS session token (optional): ")"
    region="$(prompt_with_default "AWS region" "${ENV_VALUES[AWS_REGION]:-us-east-1}")"

    ENV_VALUES["AWS_ACCESS_KEY_ID"]="$access_key"
    ENV_VALUES["AWS_SECRET_ACCESS_KEY"]="$secret_key"
    ENV_VALUES["AWS_REGION"]="$region"
    if [[ -n "$session_token" ]]; then
      ENV_VALUES["AWS_SESSION_TOKEN"]="$session_token"
    else
      unset 'ENV_VALUES[AWS_SESSION_TOKEN]'
    fi
    return 0
  fi

  log_info "Using the ambient AWS credential chain (for example IAM roles, AWS profiles, or aws sso login)."
  clear_bedrock_credentials
  region="$(prompt_clearable_with_default "AWS region (optional)" "${ENV_VALUES[AWS_REGION]:-}")"
  apply_clearable_env_value "AWS_REGION" "$region"
}

store_optional_env_value() {
  local key="$1"
  local value="${2:-}"

  if [[ -n "$value" ]]; then
    ENV_VALUES["$key"]="$value"
  else
    unset "ENV_VALUES[$key]"
  fi
}

provider_default_or_existing() {
  local selected_binding="$1"
  local existing_binding="${2:-}"
  local existing_value="${3:-}"
  local default_value="${4:-}"

  if [[ "$selected_binding" == "$existing_binding" && -n "$existing_value" ]]; then
    printf '%s' "$existing_value"
    return 0
  fi

  printf '%s' "$default_value"
}

default_llm_model_for_binding() {
  local binding="$1"

  case "$binding" in
    openai|azure_openai)
      printf 'gpt-5-mini'
      ;;
    ollama|lollms|openai-ollama)
      printf 'mistral-nemo:latest'
      ;;
    gemini)
      printf 'gemini-flash-latest'
      ;;
    aws_bedrock)
      printf 'anthropic.claude-3-5-sonnet-20241022-v2:0'
      ;;
    *)
      printf 'gpt-5-mini'
      ;;
  esac
}

default_embedding_model_for_binding() {
  local binding="$1"

  case "$binding" in
    openai|azure_openai)
      printf 'text-embedding-3-large'
      ;;
    ollama)
      printf 'bge-m3:latest'
      ;;
    jina)
      printf 'jina-embeddings-v4'
      ;;
    gemini)
      printf 'gemini-embedding-001'
      ;;
    aws_bedrock)
      printf 'amazon.titan-embed-text-v2:0'
      ;;
    lollms)
      printf 'lollms_embedding_model'
      ;;
    *)
      printf 'text-embedding-3-large'
      ;;
  esac
}

default_embedding_dim_for_binding() {
  local binding="$1"

  case "$binding" in
    openai|azure_openai)
      printf '3072'
      ;;
    ollama|aws_bedrock|lollms)
      printf '1024'
      ;;
    jina)
      printf '2048'
      ;;
    gemini)
      printf '1536'
      ;;
    *)
      printf '3072'
      ;;
  esac
}

collect_llm_config() {
  local options=("openai" "azure_openai" "ollama" "openai-ollama" "lollms" "gemini" "aws_bedrock")
  local current_binding="${ENV_VALUES[LLM_BINDING]:-openai}"
  local binding model model_default host host_default api_key

  binding="$(prompt_choice "LLM provider" "$current_binding" "${options[@]}")"
  model_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[LLM_MODEL]:-}" "$(default_llm_model_for_binding "$binding")")"
  model="$(prompt_with_default "LLM model" "$model_default")"

  case "$binding" in
    ollama)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[LLM_BINDING_HOST]:-}" "$(default_loopback_url 11434)")"
      host="$(prompt_with_default "Ollama host" "$host_default")"
      api_key=""
      ;;
    openai-ollama)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[LLM_BINDING_HOST]:-}" "$(default_loopback_url 11434 "/v1")")"
      host="$(prompt_with_default "OpenAI-compatible Ollama endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "LLM API key: " "${ENV_VALUES[LLM_BINDING_API_KEY]:-}" validate_api_key openai)"
      ;;
    lollms)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[LLM_BINDING_HOST]:-}" "http://localhost:9600")"
      host="$(prompt_with_default "LoLLMs host" "$host_default")"
      api_key=""
      ;;
    azure_openai)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[LLM_BINDING_HOST]:-}" "https://example.openai.azure.com/")"
      host="$(prompt_with_default "Azure OpenAI endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Azure OpenAI API key: " "${ENV_VALUES[LLM_BINDING_API_KEY]:-}" validate_api_key azure_openai)"
      ;;
    gemini)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[LLM_BINDING_HOST]:-}" "https://generativelanguage.googleapis.com")"
      host="$(prompt_with_default "Gemini endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Gemini API key: " "${ENV_VALUES[LLM_BINDING_API_KEY]:-}" validate_api_key gemini)"
      ;;
    aws_bedrock)
      host="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[LLM_BINDING_HOST]:-}" "https://bedrock.amazonaws.com")"
      api_key=""
      collect_bedrock_credentials
      ;;
    *)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[LLM_BINDING_HOST]:-}" "https://api.openai.com/v1")"
      host="$(prompt_with_default "LLM endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "LLM API key: " "${ENV_VALUES[LLM_BINDING_API_KEY]:-}" validate_api_key "$binding")"
      ;;
  esac

  ENV_VALUES["LLM_BINDING"]="$binding"
  ENV_VALUES["LLM_MODEL"]="$model"
  ENV_VALUES["LLM_BINDING_HOST"]="$host"
  store_optional_env_value "LLM_BINDING_API_KEY" "$api_key"
  clear_bedrock_credentials_if_unused
}

collect_embedding_config() {
  local options=("openai" "azure_openai" "ollama" "jina" "lollms" "gemini" "aws_bedrock")
  local current_binding="${ENV_VALUES[EMBEDDING_BINDING]:-openai}"
  local binding model model_default host host_default api_key dim dim_default

  if [[ "${ENV_VALUES[LLM_BINDING]:-}" == "openai-ollama" ]]; then
    binding="ollama"
    if [[ "$current_binding" != "ollama" ]]; then
      log_info "openai-ollama uses Ollama embeddings. Forcing embedding provider to ollama."
    fi
  else
    binding="$(prompt_choice "Embedding provider" "$current_binding" "${options[@]}")"
  fi
  model_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_MODEL]:-}" "$(default_embedding_model_for_binding "$binding")")"
  dim_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_DIM]:-}" "$(default_embedding_dim_for_binding "$binding")")"
  model="$(prompt_with_default "Embedding model" "$model_default")"
  dim="$(prompt_with_default "Embedding dimension" "$dim_default")"

  local llm_host_fallback="" llm_api_key_default=""
  if [[ "$binding" == "${ENV_VALUES[LLM_BINDING]:-}" ]]; then
    llm_host_fallback="${ENV_VALUES[LLM_BINDING_HOST]:-}"
    llm_api_key_default="${ENV_VALUES[LLM_BINDING_API_KEY]:-}"
  fi

  case "$binding" in
    ollama)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "${llm_host_fallback:-$(default_loopback_url 11434)}")"
      host="$(prompt_with_default "Ollama embedding host" "$host_default")"
      api_key=""
      ;;
    lollms)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "${llm_host_fallback:-http://localhost:9600}")"
      host="$(prompt_with_default "LoLLMs embedding host" "$host_default")"
      api_key=""
      ;;
    azure_openai)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "${llm_host_fallback:-https://example.openai.azure.com/}")"
      host="$(prompt_with_default "Azure OpenAI endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Azure OpenAI API key: " "${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-$llm_api_key_default}" validate_api_key azure_openai)"
      ;;
    gemini)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "${llm_host_fallback:-https://generativelanguage.googleapis.com}")"
      host="$(prompt_with_default "Gemini endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Gemini API key: " "${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-$llm_api_key_default}" validate_api_key gemini)"
      ;;
    aws_bedrock)
      host="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "${llm_host_fallback:-https://bedrock.amazonaws.com}")"
      api_key=""
      collect_bedrock_credentials
      ;;
    jina)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "${llm_host_fallback:-https://api.jina.ai/v1/embeddings}")"
      host="$(prompt_with_default "Jina endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Jina API key: " "${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-$llm_api_key_default}" validate_api_key jina)"
      ;;
    *)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "${llm_host_fallback:-https://api.openai.com/v1}")"
      host="$(prompt_with_default "Embedding endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Embedding API key: " "${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-$llm_api_key_default}" validate_api_key "$binding")"
      ;;
  esac

  ENV_VALUES["EMBEDDING_BINDING"]="$binding"
  ENV_VALUES["EMBEDDING_MODEL"]="$model"
  ENV_VALUES["EMBEDDING_DIM"]="$dim"
  ENV_VALUES["EMBEDDING_BINDING_HOST"]="$host"
  store_optional_env_value "EMBEDDING_BINDING_API_KEY" "$api_key"
  clear_bedrock_credentials_if_unused
  # User chose a remote provider — clear the Docker deployment marker.
  unset 'ENV_VALUES[LIGHTRAG_SETUP_EMBEDDING_PROVIDER]'
}

collect_rerank_config() {
  # Pass "yes" to skip the "Enable reranking?" prompt (caller already asked it).
  # The optional second argument can force the Docker choice to "yes" or "no".
  local skip_enable_check="${1:-no}"
  local docker_choice_override="${2:-prompt}"
  local options=("cohere" "jina" "aliyun" "vllm")
  local binding_choice binding model host api_key
  local vllm_model vllm_port vllm_device vllm_extra
  local vllm_host_default=""
  local default_model="" default_host="" model_default="" host_default="" use_docker="no"
  local previous_provider="${ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]:-}"
  local reset_vllm_defaults="no"
  local rerank_default="${ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]:-${ENV_VALUES[RERANK_BINDING]:-cohere}}"

  unset 'ENV_VALUES[VLLM_RERANK_DTYPE]'

  if [[ "$skip_enable_check" != "yes" ]]; then
    local rerank_was_enabled="no"
    if [[ -n "${ENV_VALUES[RERANK_BINDING]:-}" && "${ENV_VALUES[RERANK_BINDING]}" != "null" ]]; then
      rerank_was_enabled="yes"
    fi

    local rerank_enabled="no"
    if [[ "$rerank_was_enabled" == "yes" ]]; then
      confirm_default_yes "Enable reranking?" && rerank_enabled="yes"
    else
      confirm_default_no "Enable reranking?" && rerank_enabled="yes"
    fi

    if [[ "$rerank_enabled" != "yes" ]]; then
      ENV_VALUES["RERANK_BINDING"]="null"
      unset 'ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]'
      return
    fi
  fi

  if [[ "$rerank_default" == "null" ]]; then
    rerank_default="cohere"
  fi

  binding_choice="$(prompt_choice "Rerank provider" "$rerank_default" "${options[@]}")"
  if [[ "$binding_choice" != "vllm" && "$previous_provider" == "vllm" ]]; then
    reset_vllm_defaults="yes"
  fi

  if [[ "$binding_choice" == "vllm" ]]; then
    if [[ "$docker_choice_override" == "yes" || "$docker_choice_override" == "no" ]]; then
      use_docker="$docker_choice_override"
    elif confirm_default_yes "Run rerank service locally via Docker?"; then
      use_docker="yes"
    fi
    if [[ "$use_docker" == "yes" ]]; then
      add_docker_service "vllm-rerank"
      vllm_model="$(prompt_with_default "vLLM rerank model" "${ENV_VALUES[VLLM_RERANK_MODEL]:-BAAI/bge-reranker-v2-m3}")"
      vllm_port="$(prompt_until_valid "vLLM rerank port" "${ENV_VALUES[VLLM_RERANK_PORT]:-8000}" validate_port)"
      vllm_device="$(resolve_local_device_default "${ENV_VALUES[VLLM_RERANK_DEVICE]:-}")"
      vllm_device="$(prompt_choice "vLLM device" "$vllm_device" "cpu" "cuda")"
      if [[ "$vllm_device" == "cuda" ]] && ! host_cuda_available; then
        log_warn "CUDA device selected but no NVIDIA driver detected on host."
        if confirm_default_yes "Use CPU instead?"; then
          vllm_device="cpu"
        fi
      fi
      vllm_extra="$(prompt_with_default "vLLM extra args" "${ENV_VALUES[VLLM_RERANK_EXTRA_ARGS]:-}")"
    fi

    if [[ "$use_docker" == "yes" && "$vllm_device" == "cuda" ]]; then
      if [[ "${ENV_VALUES[CUDA_VISIBLE_DEVICES]:-}" == "-1" ]]; then
        unset 'ENV_VALUES[CUDA_VISIBLE_DEVICES]'
      fi
      if [[ "${ENV_VALUES[NVIDIA_VISIBLE_DEVICES]:-}" == "-1" ]]; then
        unset 'ENV_VALUES[NVIDIA_VISIBLE_DEVICES]'
      fi
      unset 'ENV_VALUES[VLLM_USE_CPU]'
    fi

    if [[ "$use_docker" == "yes" ]]; then
      ENV_VALUES["VLLM_RERANK_MODEL"]="$vllm_model"
      ENV_VALUES["VLLM_RERANK_PORT"]="$vllm_port"
      ENV_VALUES["VLLM_RERANK_DEVICE"]="$vllm_device"
      if [[ -n "$vllm_extra" ]]; then
        ENV_VALUES["VLLM_RERANK_EXTRA_ARGS"]="$vllm_extra"
      fi
    fi

    if [[ "$use_docker" == "yes" ]]; then
      default_model="$vllm_model"
      default_host="$(default_loopback_url "$vllm_port" "/rerank")"
      set_compose_override "RERANK_BINDING_HOST" "http://vllm-rerank:${vllm_port}/rerank"
    else
      default_model="${ENV_VALUES[RERANK_MODEL]:-${ENV_VALUES[VLLM_RERANK_MODEL]:-BAAI/bge-reranker-v2-m3}}"
      vllm_host_default="$(default_loopback_url "${ENV_VALUES[VLLM_RERANK_PORT]:-8000}" "/rerank")"
      default_host="${ENV_VALUES[RERANK_BINDING_HOST]:-$vllm_host_default}"
      set_compose_override "RERANK_BINDING_HOST" ""
    fi
    binding="cohere"
  else
    binding="$binding_choice"
  fi

  if [[ "$binding_choice" == "vllm" ]]; then
    model_default="$default_model"
    host_default="$default_host"
  elif [[ "$reset_vllm_defaults" == "yes" ]]; then
    case "$binding_choice" in
      cohere)
        default_model="rerank-v3.5"
        default_host="https://api.cohere.com/v2/rerank"
        ;;
      jina)
        default_model="jina-reranker-v2-base-multilingual"
        default_host="https://api.jina.ai/v1/rerank"
        ;;
      aliyun)
        default_model="gte-rerank-v2"
        default_host="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        ;;
      *)
        default_model=""
        default_host=""
        ;;
    esac
    # Switching away from local vLLM should replace stale localhost/model values.
    model_default="$default_model"
    host_default="$default_host"
  else
    model_default="${ENV_VALUES[RERANK_MODEL]:-$default_model}"
    host_default="${ENV_VALUES[RERANK_BINDING_HOST]:-$default_host}"
  fi

  model="$(prompt_with_default "Rerank model" "$model_default")"
  host="$(prompt_with_default "Rerank endpoint" "$host_default")"
  if [[ "$binding_choice" == "vllm" ]]; then
    # Ensure a consistent key exists before prompting, generating one if needed.
    local vllm_rerank_api_key="${ENV_VALUES[VLLM_RERANK_API_KEY]:-${ENV_VALUES[RERANK_BINDING_API_KEY]:-}}"
    if [[ -z "$vllm_rerank_api_key" ]]; then
      vllm_rerank_api_key="$(openssl rand -hex 16 2>/dev/null || LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 32)"
    fi
    api_key="$(prompt_secret_with_default "Rerank API key (optional): " "$vllm_rerank_api_key")"
  else
    api_key="$(prompt_secret_until_valid_with_default "Rerank API key: " "${ENV_VALUES[RERANK_BINDING_API_KEY]:-}" validate_api_key "$binding")"
  fi

  ENV_VALUES["RERANK_BINDING"]="$binding"
  # Only keep the setup marker for wizard-managed local Docker vLLM rerank.
  # Host-managed or remote rerank endpoints should rely on RERANK_BINDING alone.
  if [[ "$binding_choice" == "vllm" && "$use_docker" == "yes" ]]; then
    ENV_VALUES["LIGHTRAG_SETUP_RERANK_PROVIDER"]="vllm"
  else
    unset 'ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]'
  fi
  if [[ -n "$model" ]]; then
    ENV_VALUES["RERANK_MODEL"]="$model"
  elif [[ "$reset_vllm_defaults" == "yes" ]]; then
    unset 'ENV_VALUES[RERANK_MODEL]'
  fi
  if [[ -n "$host" ]]; then
    ENV_VALUES["RERANK_BINDING_HOST"]="$host"
  elif [[ "$reset_vllm_defaults" == "yes" ]]; then
    unset 'ENV_VALUES[RERANK_BINDING_HOST]'
  fi
  if [[ "$binding_choice" == "vllm" ]]; then
    # Keep VLLM_RERANK_API_KEY and RERANK_BINDING_API_KEY in sync.
    ENV_VALUES["VLLM_RERANK_API_KEY"]="${api_key:-$vllm_rerank_api_key}"
    store_optional_env_value "RERANK_BINDING_API_KEY" "${api_key:-$vllm_rerank_api_key}"
  else
    store_optional_env_value "RERANK_BINDING_API_KEY" "$api_key"
  fi
}

collect_server_config() {
  local host port title description summary_language

  host="$(prompt_with_default "Server host" "${ENV_VALUES[HOST]:-0.0.0.0}")"
  port="$(prompt_until_valid "Server port" "${ENV_VALUES[PORT]:-9621}" validate_port)"
  title="$(prompt_with_default "WebUI title" "${ENV_VALUES[WEBUI_TITLE]:-My Graph KB}")"
  description="$(prompt_with_default "WebUI description" "${ENV_VALUES[WEBUI_DESCRIPTION]:-Simple and Fast Graph Based RAG System}")"
  summary_language="$(prompt_with_default "Summary language" "${ENV_VALUES[SUMMARY_LANGUAGE]:-English}")"

  ENV_VALUES["HOST"]="$host"
  ENV_VALUES["PORT"]="$port"
  ENV_VALUES["WEBUI_TITLE"]="$title"
  ENV_VALUES["WEBUI_DESCRIPTION"]="$description"
  ENV_VALUES["SUMMARY_LANGUAGE"]="$summary_language"
}

collect_ssl_config() {
  local cert key
  local ssl_enabled_default="no"

  case "${ENV_VALUES[SSL]:-}" in
    true|TRUE|True|1|yes|YES|Yes|y|Y|on|ON|On|t|T)
      ssl_enabled_default="yes"
      ;;
  esac

  if [[ "$ssl_enabled_default" == "yes" ]]; then
    if ! confirm_default_yes "Enable SSL/TLS for the API server?"; then
      unset 'ENV_VALUES[SSL]'
      unset 'ENV_VALUES[SSL_CERTFILE]'
      unset 'ENV_VALUES[SSL_KEYFILE]'
      SSL_CERT_SOURCE_PATH=""
      SSL_KEY_SOURCE_PATH=""
      return
    fi
  else
    if ! confirm_default_no "Enable SSL/TLS for the API server?"; then
      unset 'ENV_VALUES[SSL]'
      unset 'ENV_VALUES[SSL_CERTFILE]'
      unset 'ENV_VALUES[SSL_KEYFILE]'
      SSL_CERT_SOURCE_PATH=""
      SSL_KEY_SOURCE_PATH=""
      return
    fi
  fi

  cert="$(prompt_until_valid "SSL certificate file" "${ENV_VALUES[SSL_CERTFILE]:-}" validate_existing_file)"
  key="$(prompt_until_valid "SSL key file" "${ENV_VALUES[SSL_KEYFILE]:-}" validate_existing_file)"

  ENV_VALUES["SSL"]="true"
  ENV_VALUES["SSL_CERTFILE"]="$cert"
  ENV_VALUES["SSL_KEYFILE"]="$key"
  SSL_CERT_SOURCE_PATH="$cert"
  SSL_KEY_SOURCE_PATH="$key"
}

collect_security_config() {
  local required="${1:-no}"
  local default_yes="${2:-no}"
  local auth_accounts token_secret token_expire api_key whitelist
  local confirm_result=1
  local whitelist_default=""
  local whitelist_is_set="no"

  if [[ -n "${ENV_VALUES[WHITELIST_PATHS]+set}" ]]; then
    whitelist_default="${ENV_VALUES[WHITELIST_PATHS]}"
    whitelist_is_set="yes"
  fi

  if [[ "$default_yes" == "yes" ]]; then
    if confirm_default_yes "Configure authentication and API key settings?"; then
      confirm_result=0
    fi
  else
    if confirm_default_no "Configure authentication and API key settings?"; then
      confirm_result=0
    fi
  fi

  if ((confirm_result != 0)); then
    if [[ "$required" == "yes" ]]; then
      echo "Warning: production deployments should configure AUTH_ACCOUNTS; API keys are optional on top." >&2
    fi
    return
  fi

  echo "Press Enter to keep an existing value. Type 'clear' to remove it." >&2

  if [[ "$whitelist_is_set" == "no" ]]; then
    whitelist_default="/health"
  elif [[ "$required" == "yes" && "$whitelist_default" == "/health,/api/*" ]]; then
    whitelist_default="/health"
  fi

  auth_accounts="$(prompt_clearable_with_default "Auth accounts (user:pass,comma-separated)" "${ENV_VALUES[AUTH_ACCOUNTS]:-}")"
  token_secret="$(prompt_clearable_secret_with_default "JWT token secret: " "${ENV_VALUES[TOKEN_SECRET]:-}")"
  token_expire="$(prompt_clearable_with_default "Token expire hours" "${ENV_VALUES[TOKEN_EXPIRE_HOURS]:-48}")"
  api_key="$(prompt_clearable_secret_with_default "LightRAG API key: " "${ENV_VALUES[LIGHTRAG_API_KEY]:-}")"
  whitelist="$(prompt_clearable_with_default "Whitelist paths (comma-separated)" "$whitelist_default")"
  if [[ "$whitelist_is_set" == "yes" && -z "$whitelist_default" && -z "$whitelist" ]]; then
    whitelist="$CLEAR_INPUT_SENTINEL"
  fi

  if [[ -z "$token_secret" ]]; then
    token_secret="$(openssl rand -hex 32 2>/dev/null || LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 64)"
    log_info "Generated TOKEN_SECRET and saved to .env."
  fi

  apply_clearable_env_value "AUTH_ACCOUNTS" "$auth_accounts"
  apply_clearable_env_value "TOKEN_SECRET" "$token_secret"
  apply_clearable_env_value "TOKEN_EXPIRE_HOURS" "$token_expire"
  apply_clearable_env_value "LIGHTRAG_API_KEY" "$api_key"
  apply_clearable_env_value "WHITELIST_PATHS" "$whitelist" "empty"
}

apply_clearable_env_value() {
  local key="$1"
  local value="${2:-}"
  local clear_mode="${3:-unset}"

  if [[ "$clear_mode" == "empty" && "$value" == "$CLEAR_INPUT_SENTINEL" ]]; then
    ENV_VALUES["$key"]=""
    return 0
  fi

  if [[ "$value" == "$CLEAR_INPUT_SENTINEL" || -z "$value" ]]; then
    unset "ENV_VALUES[$key]"
    return 0
  fi

  ENV_VALUES["$key"]="$value"
}

collect_observability_config() {
  local secret_key public_key host

  if ! confirm_default_no "Enable Langfuse observability?"; then
    unset 'ENV_VALUES[LANGFUSE_ENABLE_TRACE]'
    unset 'ENV_VALUES[LANGFUSE_SECRET_KEY]'
    unset 'ENV_VALUES[LANGFUSE_PUBLIC_KEY]'
    unset 'ENV_VALUES[LANGFUSE_HOST]'
    return
  fi

  secret_key="$(prompt_secret_until_valid_with_default "Langfuse secret key: " "${ENV_VALUES[LANGFUSE_SECRET_KEY]:-}" validate_api_key langfuse)"
  public_key="$(prompt_secret_until_valid_with_default "Langfuse public key: " "${ENV_VALUES[LANGFUSE_PUBLIC_KEY]:-}" validate_api_key langfuse)"
  host="$(prompt_with_default "Langfuse host" "${ENV_VALUES[LANGFUSE_HOST]:-https://cloud.langfuse.com}")"

  if [[ -n "$secret_key" ]]; then
    ENV_VALUES["LANGFUSE_SECRET_KEY"]="$secret_key"
  fi
  if [[ -n "$public_key" ]]; then
    ENV_VALUES["LANGFUSE_PUBLIC_KEY"]="$public_key"
  fi
  if [[ -n "$host" ]]; then
    ENV_VALUES["LANGFUSE_HOST"]="$host"
  fi
  ENV_VALUES["LANGFUSE_ENABLE_TRACE"]="true"
}


show_summary() {
  local key
  local value

  echo
  log_info "Configuration summary:"
  if ((${#ENV_VALUES[@]} > 0)); then
    local -a sorted_keys
    mapfile -t sorted_keys < <(printf '%s\n' "${!ENV_VALUES[@]}" | sort)
    for key in "${sorted_keys[@]}"; do
      value="${ENV_VALUES[$key]}"
      if is_sensitive_env_key "$key"; then
        value="***"
      fi
      printf '  %s=%s\n' "$key" "$value"
    done
  fi

  if ((${#DOCKER_SERVICES[@]} > 0)); then
    echo
    log_info "Docker services to include:"
    for service in "${DOCKER_SERVICES[@]}"; do
      echo "  - $service"
    done
    echo "  Compose file: docker-compose.final.yml"
  fi
}

# Preserve already-staged SSL mounts when regenerating compose output. The
# setup wizards treat .env as the configuration for the current target runtime,
# not as a single file guaranteed to work for both host and Docker Compose at
# the same time. A later wizard run may rewrite .env again when the operator
# switches between host and compose workflows.
prepare_inherited_ssl_assets_for_compose() {
  local existing_compose="${1:-}"
  local staged_cert_source="$SSL_CERT_SOURCE_PATH"
  local staged_key_source="$SSL_KEY_SOURCE_PATH"
  local preserved_cert_path=""
  local preserved_key_path=""

  if [[ -n "$SSL_CERT_SOURCE_PATH" ]] && ! validate_existing_file "$SSL_CERT_SOURCE_PATH"; then
    if [[ -n "$existing_compose" ]]; then
      preserved_cert_path="$(read_service_environment_value "$existing_compose" "lightrag" "SSL_CERTFILE" || true)"
    fi
    if [[ "$preserved_cert_path" == /app/data/certs/* ]]; then
      log_warn "SSL_CERTFILE source is missing; preserving the existing compose SSL certificate mount."
      staged_cert_source=""
      ENV_VALUES["SSL_CERTFILE"]="$preserved_cert_path"
      set_compose_override "SSL_CERTFILE" "$preserved_cert_path"
    else
      format_error "Invalid SSL_CERTFILE" \
        "Set it to an existing certificate file, disable SSL, or rerun the wizard to choose a new certificate."
      return 1
    fi
  fi

  if [[ -n "$SSL_KEY_SOURCE_PATH" ]] && ! validate_existing_file "$SSL_KEY_SOURCE_PATH"; then
    if [[ -n "$existing_compose" ]]; then
      preserved_key_path="$(read_service_environment_value "$existing_compose" "lightrag" "SSL_KEYFILE" || true)"
    fi
    if [[ "$preserved_key_path" == /app/data/certs/* ]]; then
      log_warn "SSL_KEYFILE source is missing; preserving the existing compose SSL key mount."
      staged_key_source=""
      ENV_VALUES["SSL_KEYFILE"]="$preserved_key_path"
      set_compose_override "SSL_KEYFILE" "$preserved_key_path"
    else
      format_error "Invalid SSL_KEYFILE" \
        "Set it to an existing private key file, disable SSL, or rerun the wizard to choose a new key."
      return 1
    fi
  fi

  SSL_CERT_SOURCE_PATH="$staged_cert_source"
  SSL_KEY_SOURCE_PATH="$staged_key_source"

  if [[ -n "$SSL_CERT_SOURCE_PATH" || -n "$SSL_KEY_SOURCE_PATH" ]]; then
    stage_ssl_assets "$SSL_CERT_SOURCE_PATH" "$SSL_KEY_SOURCE_PATH"
  fi
}

prepare_managed_service_assets_for_compose() {
  local existing_compose="${1:-}"

  if ! prepare_inherited_ssl_assets_for_compose "$existing_compose"; then
    return 1
  fi

  if [[ -n "${DOCKER_SERVICE_SET[redis]:-}" ]]; then
    stage_redis_config_asset || return 1
  fi
}

env_base_flow() {
  local vllm_embed_api_key=""
  local vllm_rerank_api_key=""
  local existing_vllm_embed_model=""
  local existing_embedding_dim=""
  local existing_vllm_embed_port=""
  local existing_vllm_embed_host=""
  local existing_vllm_embed_device=""
  local previous_embedding_provider=""
  local existing_vllm_rerank_model=""
  local existing_vllm_rerank_port=""
  local existing_vllm_rerank_host=""
  local existing_vllm_rerank_device=""
  local previous_rerank_provider=""
  if host_cuda_available; then
    log_info "GPU detected: NVIDIA GPU found. New local vLLM services default to CUDA (GPU image + float16)."
  else
    log_info "GPU detection: no NVIDIA GPU found. New local vLLM services default to CPU image + float32."
  fi

  reset_state
  load_existing_env_if_present
  initialize_default_storage_backends

  log_info "Base configuration wizard (LLM / Embedding / Reranker)"
  echo "This wizard only modifies LLM, embedding, and reranker settings."
  echo "Storage, server, and security settings are preserved."
  echo ""

  log_step "LLM configuration"
  collect_llm_config
  echo ""

  # ── Embedding ────────────────────────────────────────────────────────────────
  log_step "Embedding configuration"
  local docker_embed_default="no"
  previous_embedding_provider="${ENV_VALUES[LIGHTRAG_SETUP_EMBEDDING_PROVIDER]:-}"
  if [[ "$previous_embedding_provider" == "vllm" ]]; then
    docker_embed_default="yes"
  fi

  local use_docker_embed="no"
  if [[ "$docker_embed_default" == "yes" ]]; then
    confirm_default_yes "Run embedding model locally via Docker (vLLM)?" && use_docker_embed="yes" || use_docker_embed="no"
  else
    confirm_default_no "Run embedding model locally via Docker (vLLM)?" && use_docker_embed="yes" || use_docker_embed="no"
  fi

  if [[ "$use_docker_embed" == "yes" ]]; then
    existing_vllm_embed_model="${ENV_VALUES[VLLM_EMBED_MODEL]:-}"
    existing_embedding_dim="${ENV_VALUES[EMBEDDING_DIM]:-}"
    existing_vllm_embed_port="${ENV_VALUES[VLLM_EMBED_PORT]:-}"
    existing_vllm_embed_host="${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}"
    existing_vllm_embed_device="${ENV_VALUES[VLLM_EMBED_DEVICE]:-}"
    apply_preset_overwrite "${PRESET_VLLM_EMBEDDING[@]}"
    if [[ -n "$existing_vllm_embed_port" ]]; then
      ENV_VALUES["VLLM_EMBED_PORT"]="$existing_vllm_embed_port"
    fi
    if [[ -n "$existing_embedding_dim" ]]; then
      ENV_VALUES["EMBEDDING_DIM"]="$existing_embedding_dim"
    fi
    if [[ "$previous_embedding_provider" == "vllm" && -n "$existing_vllm_embed_host" ]]; then
      ENV_VALUES["EMBEDDING_BINDING_HOST"]="$existing_vllm_embed_host"
    else
      ENV_VALUES["EMBEDDING_BINDING_HOST"]="http://localhost:${ENV_VALUES[VLLM_EMBED_PORT]:-8001}/v1"
    fi
    local embed_model
    embed_model="$(prompt_with_default "Embedding model" "${existing_vllm_embed_model:-${ENV_VALUES[VLLM_EMBED_MODEL]:-BAAI/bge-m3}}")"
    ENV_VALUES["VLLM_EMBED_MODEL"]="$embed_model"
    ENV_VALUES["EMBEDDING_MODEL"]="$embed_model"

    local vllm_embed_device
    vllm_embed_device="$(resolve_local_device_default "$existing_vllm_embed_device")"
    ENV_VALUES["VLLM_EMBED_DEVICE"]="$vllm_embed_device"
    ENV_VALUES["LIGHTRAG_SETUP_EMBEDDING_PROVIDER"]="vllm"

    vllm_embed_api_key="${ENV_VALUES[VLLM_EMBED_API_KEY]:-${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-}}"
    if [[ -z "$vllm_embed_api_key" ]]; then
      vllm_embed_api_key="$(openssl rand -hex 16 2>/dev/null || LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 32)"
    fi
    ENV_VALUES["VLLM_EMBED_API_KEY"]="$vllm_embed_api_key"
    ENV_VALUES["EMBEDDING_BINDING_API_KEY"]="$vllm_embed_api_key"
    add_docker_service "vllm-embed"
    set_compose_override "EMBEDDING_BINDING_HOST" \
      "http://vllm-embed:${ENV_VALUES[VLLM_EMBED_PORT]:-8001}/v1"
  else
    collect_embedding_config
  fi
  echo ""

  # ── Reranker ─────────────────────────────────────────────────────────────────
  log_step "Reranker configuration"
  local rerank_enabled_default="no"
  if [[ -n "${ENV_VALUES[RERANK_BINDING]:-}" && "${ENV_VALUES[RERANK_BINDING]}" != "null" ]]; then
    rerank_enabled_default="yes"
  fi
  previous_rerank_provider="${ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]:-}"

  local enable_reranking="no"
  if [[ "$rerank_enabled_default" == "yes" ]]; then
    confirm_default_yes "Enable reranking?" && enable_reranking="yes" || enable_reranking="no"
  else
    confirm_default_no "Enable reranking?" && enable_reranking="yes" || enable_reranking="no"
  fi

  if [[ "$enable_reranking" == "yes" ]]; then
    local docker_rerank_default="no"
    if [[ "$previous_rerank_provider" == "vllm" ]]; then
      docker_rerank_default="yes"
    fi

    local use_docker_rerank="no"
    if [[ "$docker_rerank_default" == "yes" ]]; then
      confirm_default_yes "Run rerank service locally via Docker?" && use_docker_rerank="yes" || use_docker_rerank="no"
    else
      confirm_default_no "Run rerank service locally via Docker?" && use_docker_rerank="yes" || use_docker_rerank="no"
    fi

    if [[ "$use_docker_rerank" == "yes" ]]; then
      existing_vllm_rerank_model="${ENV_VALUES[VLLM_RERANK_MODEL]:-}"
      existing_vllm_rerank_port="${ENV_VALUES[VLLM_RERANK_PORT]:-}"
      existing_vllm_rerank_host="${ENV_VALUES[RERANK_BINDING_HOST]:-}"
      existing_vllm_rerank_device="${ENV_VALUES[VLLM_RERANK_DEVICE]:-}"
      apply_preset_overwrite "${PRESET_VLLM_RERANKER[@]}"
      local rerank_model rerank_port
      if [[ -n "$existing_vllm_rerank_port" ]]; then
        ENV_VALUES["VLLM_RERANK_PORT"]="$existing_vllm_rerank_port"
      fi
      if [[ "$previous_rerank_provider" == "vllm" && -n "$existing_vllm_rerank_host" ]]; then
        ENV_VALUES["RERANK_BINDING_HOST"]="$existing_vllm_rerank_host"
      else
        ENV_VALUES["RERANK_BINDING_HOST"]="http://localhost:${ENV_VALUES[VLLM_RERANK_PORT]:-8000}/rerank"
      fi
      rerank_model="$(prompt_with_default "Rerank model" "${existing_vllm_rerank_model:-${ENV_VALUES[VLLM_RERANK_MODEL]:-BAAI/bge-reranker-v2-m3}}")"
      rerank_port="${ENV_VALUES[VLLM_RERANK_PORT]:-8000}"
      ENV_VALUES["VLLM_RERANK_MODEL"]="$rerank_model"
      ENV_VALUES["RERANK_MODEL"]="$rerank_model"
      ENV_VALUES["VLLM_RERANK_PORT"]="$rerank_port"

      local vllm_rerank_device
      vllm_rerank_device="$(resolve_local_device_default "$existing_vllm_rerank_device")"
      ENV_VALUES["VLLM_RERANK_DEVICE"]="$vllm_rerank_device"
      ENV_VALUES["LIGHTRAG_SETUP_RERANK_PROVIDER"]="vllm"

      vllm_rerank_api_key="${ENV_VALUES[VLLM_RERANK_API_KEY]:-${ENV_VALUES[RERANK_BINDING_API_KEY]:-}}"
      if [[ -z "$vllm_rerank_api_key" ]]; then
        vllm_rerank_api_key="$(openssl rand -hex 16 2>/dev/null || LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 32)"
      fi
      ENV_VALUES["VLLM_RERANK_API_KEY"]="$vllm_rerank_api_key"
      ENV_VALUES["RERANK_BINDING_API_KEY"]="$vllm_rerank_api_key"
      add_docker_service "vllm-rerank"
      set_compose_override "RERANK_BINDING_HOST" \
        "http://vllm-rerank:${rerank_port}/rerank"
    else
      # Reranking enabled but not via Docker — ask provider/host/model/api_key
      collect_rerank_config "yes" "no"
    fi
  else
    ENV_VALUES["RERANK_BINDING"]="null"
    unset 'ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]'
  fi
  echo ""

  finalize_base_setup
}

finalize_base_setup() {
  local backup_path
  local compose_file
  local existing_compose
  local generate_compose="no"
  local runtime_target="$DEFAULT_RUNTIME_TARGET"
  local show_host_start_hint="no"
  local svc

  if [[ ! -f "${REPO_ROOT}/env.example" ]]; then
    format_error "env.example is missing in $REPO_ROOT" "Restore env.example before running setup."
    return 1
  fi
  if [[ ! -w "$REPO_ROOT" ]]; then
    format_error "No write permission in $REPO_ROOT" "Run the setup from a writable directory."
    return 1
  fi

  if ! validate_sensitive_env_literals; then
    return 1
  fi

  show_summary

  if ! confirm_required_yes_no "${COLOR_YELLOW}Ready to proceed and write .env${COLOR_RESET}"; then
    log_warn "Setup cancelled."
    return 1
  fi

  existing_compose="$(find_generated_compose_file)"
  compose_file="${REPO_ROOT}/docker-compose.final.yml"
  record_existing_managed_root_services "$existing_compose"

  # Preserve storage services from any existing compose file.
  if [[ -n "$existing_compose" ]]; then
    while IFS= read -r svc; do
      local is_storage="no"
      for storage_svc in "${STORAGE_SERVICES[@]}"; do
        if [[ "$svc" == "$storage_svc" ]]; then
          is_storage="yes"
          break
        fi
      done
      if [[ "$is_storage" == "yes" ]]; then
        add_docker_service "$svc"
      fi
    done < <(detect_managed_root_services "$existing_compose")
  fi

  configure_base_compose_rewrites

  if ((${#DOCKER_SERVICES[@]} > 0)); then
    # LightRAG depends on managed Docker services; it must run via Docker.
    local svc_names
    svc_names="$(printf '%s ' "${DOCKER_SERVICES[@]}")"
    svc_names="${svc_names% }"
    echo "LightRAG requires Docker services: ${svc_names}"
    if ! confirm_default_yes "The compose file will be created/updated. Continue?"; then
      log_warn "Setup cancelled."
      return 1
    fi
    generate_compose="yes"
    runtime_target="compose"
  else
    # No managed service dependencies — ask whether to run LightRAG via Docker.
    local current_target="${ENV_VALUES[LIGHTRAG_RUNTIME_TARGET]:-$DEFAULT_RUNTIME_TARGET}"
    # If an existing compose file is present, default to keeping Docker mode.
    local effective_default="$current_target"
    if [[ -n "$existing_compose" ]]; then
      effective_default="compose"
    fi

    if [[ "$effective_default" == "compose" ]]; then
      if ! confirm_default_yes "Run LightRAG Server via Docker?"; then
        # User opts out: switch to host mode and remove the stale compose file.
        if [[ -n "$existing_compose" ]]; then
          rm "$existing_compose"
          log_success "Removed ${existing_compose}"
        fi
        show_host_start_hint="yes"
      else
        generate_compose="yes"
        runtime_target="compose"
      fi
    else
      if confirm_default_no "Run LightRAG Server via Docker?"; then
        generate_compose="yes"
        runtime_target="compose"
      fi
    fi
  fi

  if [[ "$generate_compose" == "yes" ]]; then
    backup_existing_compose_if_generating "$generate_compose" "$existing_compose" || return 1
    if ! prepare_managed_service_assets_for_compose "$existing_compose"; then
      return 1
    fi
    prepare_compose_env_overrides
  fi

  backup_path="$(backup_env_file)"
  if [[ -n "$backup_path" ]]; then
    log_success "Backed up existing .env to $backup_path"
  fi

  clear_deprecated_vllm_dtype_state
  set_runtime_target "$runtime_target" || return 1
  generate_env_file "${REPO_ROOT}/env.example" "${REPO_ROOT}/.env"
  log_success "Wrote .env"

  if [[ "$generate_compose" == "yes" ]]; then
    prepare_compose_output_from_existing "$compose_file" "$existing_compose" || return 1
    generate_docker_compose "$compose_file"
    log_success "Wrote ${compose_file}"
    if [[ -n "$existing_compose" ]]; then
      log_success "Storage services preserved; vLLM services updated."
    fi
    echo "  To start: docker compose -f ${compose_file} up -d"
  elif [[ "$show_host_start_hint" == "yes" ]]; then
    echo "  To start: lightrag-server"
  fi
}

env_storage_flow() {
  local env_file="${REPO_ROOT}/.env"
  local db_type
  local db_order=("postgresql" "neo4j" "mongodb" "redis" "milvus" "qdrant" "memgraph")

  if [[ ! -f "$env_file" ]]; then
    format_error "No .env file found." "Run 'make env-base' first to configure LLM and embedding."
    return 1
  fi

  reset_state
  load_existing_env_if_present

  log_info "Storage configuration wizard"
  echo "This wizard only modifies storage backend settings."
  echo "LLM, embedding, reranker, server, and security settings are preserved."
  echo ""

  log_step "Storage backend selection"
  select_storage_backends "custom"
  log_debug "Storage selections: kv=${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-} vector=${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-} graph=${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]:-} doc=${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]:-}"
  clear_unused_storage_deployment_markers

  log_step "Database configuration"
  for db_type in "${db_order[@]}"; do
    if [[ -n "${REQUIRED_DB_TYPES[$db_type]+set}" ]]; then
      collect_database_config "$db_type" "$(storage_default_docker_for_db_type "$db_type")"
      echo ""
    fi
  done

  finalize_storage_setup
}

finalize_storage_setup() {
  local backup_path
  local compose_file
  local existing_compose
  local generate_compose="no"
  local has_docker_storage="no"
  local runtime_target="$DEFAULT_RUNTIME_TARGET"
  local svc

  if [[ ! -f "${REPO_ROOT}/env.example" ]]; then
    format_error "env.example is missing in $REPO_ROOT" "Restore env.example before running setup."
    return 1
  fi
  if [[ ! -w "$REPO_ROOT" ]]; then
    format_error "No write permission in $REPO_ROOT" "Run the setup from a writable directory."
    return 1
  fi

  if [[ -n "${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-}" ]]; then
    if ! validate_required_variables \
      "${ENV_VALUES[LIGHTRAG_KV_STORAGE]}" \
      "${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]}" \
      "${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]}" \
      "${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]}"; then
      return 1
    fi
  fi

  if ! validate_mongo_vector_storage_config \
    "${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-}" \
    "${ENV_VALUES[MONGO_URI]:-}" \
    "${ENV_VALUES[LIGHTRAG_SETUP_MONGODB_DEPLOYMENT]:-}"; then
    return 1
  fi

  if ! validate_sensitive_env_literals; then
    return 1
  fi

  if ((${#DOCKER_SERVICES[@]} > 0)); then
    has_docker_storage="yes"
  fi

  show_summary

  if ! confirm_required_yes_no "${COLOR_YELLOW}Ready to proceed and write .env${COLOR_RESET}"; then
    log_warn "Setup cancelled."
    return 1
  fi

  existing_compose="$(find_generated_compose_file)"
  compose_file="${REPO_ROOT}/docker-compose.final.yml"
  record_existing_managed_root_services "$existing_compose"

  if [[ "$has_docker_storage" == "no" && ${#EXISTING_MANAGED_ROOT_SERVICE_SET[@]} -eq 0 ]]; then
    # No managed services are selected and no existing managed services need cleanup.
    backup_path="$(backup_env_file)"
    if [[ -n "$backup_path" ]]; then
      log_success "Backed up existing .env to $backup_path"
    fi
    clear_deprecated_vllm_dtype_state
    set_runtime_target "$runtime_target" || return 1
    generate_env_file "${REPO_ROOT}/env.example" "${REPO_ROOT}/.env"
    log_success "Wrote .env"
    return 0
  fi

  if [[ -n "$existing_compose" ]]; then
    # Detect and preserve existing vLLM services.
    while IFS= read -r svc; do
      local is_vllm="no"
      for vllm_svc in "${VLLM_SERVICES[@]}"; do
        if [[ "$svc" == "$vllm_svc" ]]; then
          is_vllm="yes"
          break
        fi
      done
      if [[ "$is_vllm" == "yes" ]]; then
        add_docker_service "$svc"
      fi
    done < <(detect_managed_root_services "$existing_compose")
  fi
  configure_storage_compose_rewrites
  generate_compose="yes"
  runtime_target="compose"

  backup_existing_compose_if_generating "$generate_compose" "$existing_compose" || return 1

  if ! prepare_managed_service_assets_for_compose "$existing_compose"; then
    return 1
  fi
  prepare_compose_env_overrides

  backup_path="$(backup_env_file)"
  if [[ -n "$backup_path" ]]; then
    log_success "Backed up existing .env to $backup_path"
  fi

  clear_deprecated_vllm_dtype_state
  set_runtime_target "$runtime_target" || return 1
  generate_env_file "${REPO_ROOT}/env.example" "${REPO_ROOT}/.env"
  log_success "Wrote .env"

  prepare_compose_output_from_existing "$compose_file" "$existing_compose" || return 1
  generate_docker_compose "$compose_file"
  log_success "Wrote ${compose_file}"
  if [[ -n "$existing_compose" ]]; then
    log_success "vLLM services preserved; storage services updated."
  fi
  echo "  To start: docker compose -f ${compose_file} up -d"
}

env_server_flow() {
  local env_file="${REPO_ROOT}/.env"

  if [[ ! -f "$env_file" ]]; then
    format_error "No .env file found." "Run 'make env-base' first to configure LLM and embedding."
    return 1
  fi

  reset_state
  load_existing_env_if_present

  log_info "Server configuration wizard"
  echo "This wizard only modifies server, security, and SSL settings."
  echo "LLM, embedding, reranker, and storage settings are preserved."
  echo ""

  log_step "Server configuration"
  collect_server_config
  echo ""
  log_step "Security configuration"
  collect_security_config "no" "no"
  echo ""
  log_step "SSL configuration"
  collect_ssl_config
  echo ""

  finalize_server_setup
}

finalize_server_setup() {
  local backup_path
  local compose_file
  local existing_compose
  local generate_compose="no"
  local runtime_target="$DEFAULT_RUNTIME_TARGET"
  local svc

  if [[ ! -f "${REPO_ROOT}/env.example" ]]; then
    format_error "env.example is missing in $REPO_ROOT" "Restore env.example before running setup."
    return 1
  fi
  if [[ ! -w "$REPO_ROOT" ]]; then
    format_error "No write permission in $REPO_ROOT" "Run the setup from a writable directory."
    return 1
  fi

  if ! validate_sensitive_env_literals; then
    return 1
  fi

  if ! validate_security_config \
    "${ENV_VALUES[AUTH_ACCOUNTS]:-}" \
    "${ENV_VALUES[TOKEN_SECRET]:-}" \
    "${ENV_VALUES[LIGHTRAG_API_KEY]:-}"; then
    return 1
  fi

  show_summary

  if ! confirm_required_yes_no "${COLOR_YELLOW}Ready to proceed and write .env${COLOR_RESET}"; then
    log_warn "Setup cancelled."
    return 1
  fi

  existing_compose="$(find_generated_compose_file)"
  compose_file="${REPO_ROOT}/docker-compose.final.yml"
  record_existing_managed_root_services "$existing_compose"

  if [[ -n "$existing_compose" ]]; then
    generate_compose="yes"
    runtime_target="compose"
    # Detect and preserve all existing wizard-managed root services.
    while IFS= read -r svc; do
      add_docker_service "$svc"
    done < <(detect_managed_root_services "$existing_compose")
  fi

  if [[ "$generate_compose" == "yes" ]]; then
    backup_existing_compose_if_generating "$generate_compose" "$existing_compose" || return 1
    if ! prepare_managed_service_assets_for_compose "$existing_compose"; then
      return 1
    fi
    prepare_compose_env_overrides
  else
    if [[ -n "$SSL_CERT_SOURCE_PATH" ]] && ! validate_existing_file "$SSL_CERT_SOURCE_PATH"; then
      format_error "Invalid SSL_CERTFILE" \
        "Set it to an existing certificate file, disable SSL, or rerun the wizard to choose a new certificate."
      return 1
    fi

    if [[ -n "$SSL_KEY_SOURCE_PATH" ]] && ! validate_existing_file "$SSL_KEY_SOURCE_PATH"; then
      format_error "Invalid SSL_KEYFILE" \
        "Set it to an existing private key file, disable SSL, or rerun the wizard to choose a new key."
      return 1
    fi
  fi

  backup_path="$(backup_env_file)"
  if [[ -n "$backup_path" ]]; then
    log_success "Backed up existing .env to $backup_path"
  fi

  clear_deprecated_vllm_dtype_state
  set_runtime_target "$runtime_target" || return 1
  generate_env_file "${REPO_ROOT}/env.example" "${REPO_ROOT}/.env"
  log_success "Wrote .env"

  if [[ "$generate_compose" == "yes" ]]; then
    prepare_compose_output_from_existing "$compose_file" "$existing_compose" || return 1
    generate_docker_compose "$compose_file"
    log_success "Wrote ${compose_file}"
    log_success "Server port and security settings updated in compose."
    echo "  To restart: docker compose -f ${compose_file} up -d --force-recreate lightrag"
  fi
}

load_env_file() {
  local env_file="$1"
  local line key value

  if [[ ! -f "$env_file" ]]; then
    format_error ".env file not found at $env_file" "Run make env-base to generate it."
    return 1
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[A-Z0-9_]+= ]]; then
      key="${line%%=*}"
      value="${line#*=}"
      if [[ "$value" =~ ^\".*\"$ ]]; then
        value="${value:1:${#value}-2}"
        value="${value//\\\$/\$}"
        value="${value//\\\"/\"}"
        value="${value//\\\\/\\}"
      elif [[ "$value" =~ ^\'.*\'$ ]]; then
        value="${value:1:${#value}-2}"
      fi
      ENV_VALUES["$key"]="$value"
    fi
  done < "$env_file"
}

validate_ssl_runtime_path() {
  local path="$1"
  local runtime_target="${ENV_VALUES[LIGHTRAG_RUNTIME_TARGET]:-$DEFAULT_RUNTIME_TARGET}"
  local staged_path=""

  if validate_existing_file "$path"; then
    return 0
  fi

  if [[ "$runtime_target" == "compose" && "$path" == /app/data/certs/* ]]; then
    staged_path="${REPO_ROOT}/data/certs/${path#/app/data/certs/}"
    validate_existing_file "$staged_path"
    return $?
  fi

  return 1
}

validate_env_file() {
  local env_file="${REPO_ROOT}/.env"
  local errors=0
  local kv vector graph doc_status
  local runtime_target

  reset_state

  if ! load_env_file "$env_file"; then
    return 1
  fi

  kv="${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-}"
  vector="${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-}"
  graph="${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]:-}"
  doc_status="${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]:-}"
  runtime_target="${ENV_VALUES[LIGHTRAG_RUNTIME_TARGET]:-$DEFAULT_RUNTIME_TARGET}"

  if ! validate_runtime_target "$runtime_target"; then
    errors=1
  fi

  if [[ -z "$kv" || -z "$vector" || -z "$graph" || -z "$doc_status" ]]; then
    format_error "Storage selections are missing in .env" "Set LIGHTRAG_*_STORAGE variables."
    return 1
  fi

  if ! validate_mongo_vector_storage_config \
    "$vector" \
    "${ENV_VALUES[MONGO_URI]:-}" \
    "${ENV_VALUES[LIGHTRAG_SETUP_MONGODB_DEPLOYMENT]:-}"; then
    errors=1
  fi

  if ! validate_required_variables "$kv" "$vector" "$graph" "$doc_status"; then
    errors=1
  fi

  if ! validate_security_config \
    "${ENV_VALUES[AUTH_ACCOUNTS]:-}" \
    "${ENV_VALUES[TOKEN_SECRET]:-}" \
    "${ENV_VALUES[LIGHTRAG_API_KEY]:-}"; then
    errors=1
  fi

  if ! validate_sensitive_env_literals; then
    errors=1
  fi

  if [[ "${ENV_VALUES[SSL]:-false}" == "true" ]]; then
    if ! validate_ssl_runtime_path "${ENV_VALUES[SSL_CERTFILE]:-}"; then
      format_error "Invalid SSL_CERTFILE" "Set it to an existing certificate file when SSL=true."
      errors=1
    fi
    if ! validate_ssl_runtime_path "${ENV_VALUES[SSL_KEYFILE]:-}"; then
      format_error "Invalid SSL_KEYFILE" "Set it to an existing private key file when SSL=true."
      errors=1
    fi
  fi

  if [[ -n "${ENV_VALUES[NEO4J_URI]:-}" ]] && ! validate_uri "${ENV_VALUES[NEO4J_URI]}" neo4j; then
    format_error "Invalid NEO4J_URI" "Use neo4j:// or bolt:// format."
    errors=1
  fi
  if [[ -n "${ENV_VALUES[MONGO_URI]:-}" ]] && ! validate_uri "${ENV_VALUES[MONGO_URI]}" mongodb; then
    format_error "Invalid MONGO_URI" "Use mongodb:// or mongodb+srv:// format."
    errors=1
  fi
  if [[ -n "${ENV_VALUES[REDIS_URI]:-}" ]] && ! validate_uri "${ENV_VALUES[REDIS_URI]}" redis; then
    format_error "Invalid REDIS_URI" "Use redis:// or rediss:// format."
    errors=1
  fi
  if [[ -n "${ENV_VALUES[MILVUS_URI]:-}" ]] && ! validate_uri "${ENV_VALUES[MILVUS_URI]}" milvus; then
    format_error "Invalid MILVUS_URI" "Use http://host:port format."
    errors=1
  fi
  if [[ -n "${ENV_VALUES[QDRANT_URL]:-}" ]] && ! validate_uri "${ENV_VALUES[QDRANT_URL]}" qdrant; then
    format_error "Invalid QDRANT_URL" "Use http://host:port format."
    errors=1
  fi
  if [[ -n "${ENV_VALUES[MEMGRAPH_URI]:-}" ]] && ! validate_uri "${ENV_VALUES[MEMGRAPH_URI]}" memgraph; then
    format_error "Invalid MEMGRAPH_URI" "Use bolt://host:port format."
    errors=1
  fi
  if [[ -n "${ENV_VALUES[POSTGRES_PORT]:-}" ]] && ! validate_port "${ENV_VALUES[POSTGRES_PORT]}"; then
    format_error "Invalid POSTGRES_PORT" "Use a port between 1 and 65535."
    errors=1
  fi
  if ((errors != 0)); then
    return 1
  fi

  log_success "Validation passed."
}

report_security_issue() {
  local message="$1"
  local suggestion="${2:-}"

  echo "${COLOR_YELLOW:-}Security issue:${COLOR_RESET:-} $message"
  if [[ -n "$suggestion" ]]; then
    echo "  Suggestion: $suggestion"
  fi
}

security_check_env_file() {
  local env_file="${REPO_ROOT}/.env"
  local findings=0
  local auth_accounts=""
  local token_secret=""
  local api_key=""
  local whitelist_paths=""
  local whitelist_is_set="no"
  local effective_whitelist=""
  local key value
  local invalid_sensitive_keys=()

  reset_state

  if ! load_env_file "$env_file"; then
    return 1
  fi

  auth_accounts="${ENV_VALUES[AUTH_ACCOUNTS]:-}"
  token_secret="${ENV_VALUES[TOKEN_SECRET]:-}"
  api_key="${ENV_VALUES[LIGHTRAG_API_KEY]:-}"
  if [[ -n "${ENV_VALUES[WHITELIST_PATHS]+set}" ]]; then
    whitelist_paths="${ENV_VALUES[WHITELIST_PATHS]}"
    whitelist_is_set="yes"
  fi

  for key in "${!ENV_VALUES[@]}"; do
    if ! is_sensitive_env_key "$key"; then
      continue
    fi
    value="${ENV_VALUES[$key]:-}"
    if [[ -n "$value" ]] && contains_env_interpolation_syntax "$value"; then
      invalid_sensitive_keys+=("$key")
    fi
  done

  if ((${#invalid_sensitive_keys[@]} > 0)); then
    report_security_issue \
      "Sensitive values still contain \${...} interpolation syntax: ${invalid_sensitive_keys[*]}" \
      "Replace them with literal values or inject those secrets at runtime."
    findings=$((findings + 1))
  fi

  if [[ -z "$auth_accounts" && -z "$api_key" ]]; then
    report_security_issue \
      "No API protection is configured." \
      "Set AUTH_ACCOUNTS and TOKEN_SECRET, add LIGHTRAG_API_KEY, or put the service behind a trusted reverse proxy."
    findings=$((findings + 1))
  fi

  if [[ -n "$auth_accounts" ]]; then
    if ! validate_auth_accounts_format "$auth_accounts"; then
      report_security_issue \
        "AUTH_ACCOUNTS is malformed." \
        "Use comma-separated user:password pairs such as admin:secret or admin:secret,reader:another-secret."
      findings=$((findings + 1))
    fi

    if [[ -z "$token_secret" ]]; then
      report_security_issue \
        "AUTH_ACCOUNTS is set but TOKEN_SECRET is missing." \
        "Set a non-empty JWT signing secret before enabling account-based authentication."
      findings=$((findings + 1))
    elif [[ "$token_secret" == "lightrag-jwt-default-secret" ]]; then
      report_security_issue \
        "TOKEN_SECRET still uses the built-in default value." \
        "Generate a unique JWT signing secret and update TOKEN_SECRET."
      findings=$((findings + 1))
    fi

    effective_whitelist="$whitelist_paths"
    if [[ "$whitelist_is_set" != "yes" ]]; then
      effective_whitelist="/health,/api/*"
    fi
    if whitelist_exposes_api_routes "$effective_whitelist"; then
      report_security_issue \
        "WHITELIST_PATHS exposes /api routes while AUTH_ACCOUNTS is enabled." \
        "Use a minimal whitelist such as /health,/docs and keep /api routes authenticated."
      findings=$((findings + 1))
    fi
  fi

  if [[ -z "$auth_accounts" && -n "$api_key" ]]; then
    effective_whitelist="$whitelist_paths"
    if [[ "$whitelist_is_set" != "yes" ]]; then
      effective_whitelist="/health,/api/*"
    fi
    if whitelist_exposes_api_routes "$effective_whitelist"; then
      report_security_issue \
        "WHITELIST_PATHS exposes /api routes while LIGHTRAG_API_KEY is the only active auth mechanism." \
        "Use a minimal whitelist such as /health,/docs and keep /api routes protected by the API key."
      findings=$((findings + 1))
    fi
  fi

  if ((findings == 0)); then
    log_success "No obvious security issues found in ${env_file}."
    return 0
  fi

  log_warn "Security check found ${findings} issue(s) in ${env_file}."
  return 1
}

backup_only() {
  local backup_path
  local compose_backup_path

  backup_path="$(backup_env_file)"
  if [[ -z "$backup_path" ]]; then
    format_error "No .env file found to back up." "Create one with make env-base first."
    return 1
  fi
  echo "Backed up .env to $backup_path"

  compose_backup_path="$(backup_compose_file)" || return 1
  if [[ -n "$compose_backup_path" ]]; then
    echo "Backed up compose file to $compose_backup_path"
  fi
}

print_help() {
  cat <<'HELP'
Usage: scripts/setup/setup.sh [--base|--storage|--server|--validate|--security-check|--backup] [--rewrite-compose]

Options:
  --base         Configure LLM, embedding, and reranker (run first)
  --storage      Configure storage backends and databases (requires .env)
  --server       Configure server, security, and SSL (requires .env)
  --validate     Validate an existing .env file
  --security-check  Audit an existing .env for security risks
  --backup       Backup the current .env and generated compose file when present
  --rewrite-compose  Force regeneration of all wizard-managed compose services
  --debug        Enable debug logging
  --help         Show this help message
HELP
}

_sigint_handler() {
  echo ""
  echo "Setup interrupted."
  exit 130
}

main() {
  trap '_sigint_handler' INT
  init_colors
  local mode="help"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --base)
        mode="base"
        ;;
      --storage)
        mode="storage"
        ;;
      --server)
        mode="server"
        ;;
      --validate)
        mode="validate"
        ;;
      --security-check)
        mode="security-check"
        ;;
      --backup)
        mode="backup"
        ;;
      --debug)
        DEBUG="true"
        ;;
      --rewrite-compose)
        FORCE_REWRITE_COMPOSE="yes"
        ;;
      --help|-h)
        mode="help"
        ;;
      *)
        echo "Unknown option: $1" >&2
        print_help
        return 1
        ;;
    esac
    shift
  done

  case "$mode" in
    base)
      env_base_flow
      ;;
    storage)
      env_storage_flow
      ;;
    server)
      env_server_flow
      ;;
    validate)
      validate_env_file
      ;;
    security-check)
      security_check_env_file
      ;;
    backup)
      backup_only
      ;;
    *)
      print_help
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
