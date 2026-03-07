#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSINFO+x}" || "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  echo "Error: scripts/setup/setup.sh requires Bash 4 or newer." >&2
  echo "Hint: install a newer bash and run it via 'bash scripts/setup/setup.sh ...'." >&2
  exit 1
fi

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
PRESETS_DIR="$SCRIPT_DIR/presets"
# shellcheck disable=SC2034
TEMPLATES_DIR="$SCRIPT_DIR/templates"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

declare -A ENV_VALUES
declare -A COMPOSE_ENV_OVERRIDES
declare -A REQUIRED_DB_TYPES
declare -A DOCKER_SERVICE_SET
declare -a DOCKER_SERVICES
SSL_CERT_SOURCE_PATH=""
SSL_KEY_SOURCE_PATH=""
declare -A DOCKER_IMAGE_DEFAULTS=(
  ["postgres"]="gzdaniel/postgres-for-rag:16.6"
  ["neo4j"]="5.26.19-community"
  ["mongodb"]="8.2.3"
  ["redis"]="8.4"
  ["milvus"]="2.6-20251227-44275071"
  ["etcd"]="v3.5.16"
  ["minio"]="RELEASE.2024-12-13T22-19-12Z"
  ["qdrant"]="v1.16.0"
  ["memgraph"]="3.7.2"
  ["vllm-rerank"]="latest"
)
declare -A DOCKER_IMAGE_ENV=(
  ["postgres"]="POSTGRES_IMAGE"
  ["neo4j"]="NEO4J_IMAGE_TAG"
  ["mongodb"]="MONGODB_IMAGE_TAG"
  ["redis"]="REDIS_IMAGE_TAG"
  ["milvus"]="MILVUS_IMAGE_TAG"
  ["etcd"]="ETCD_IMAGE_TAG"
  ["minio"]="MINIO_IMAGE_TAG"
  ["qdrant"]="QDRANT_IMAGE_TAG"
  ["memgraph"]="MEMGRAPH_IMAGE_TAG"
  ["vllm-rerank"]="VLLM_RERANK_IMAGE_TAG"
)
DEPLOYMENT_TYPE=""
DEBUG="${DEBUG:-false}"
WAIT_TIMEOUT="${SETUP_WAIT_TIMEOUT:-90}"
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

# shellcheck disable=SC1091
source "$PRESETS_DIR/development.sh"
# shellcheck disable=SC1091
source "$PRESETS_DIR/production.sh"
# shellcheck disable=SC1091
source "$PRESETS_DIR/local.sh"

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
  COMPOSE_ENV_OVERRIDES=()
  REQUIRED_DB_TYPES=()
  DOCKER_SERVICE_SET=()
  DOCKER_SERVICES=()
  SSL_CERT_SOURCE_PATH=""
  SSL_KEY_SOURCE_PATH=""
  DEPLOYMENT_TYPE=""
}

load_existing_env_if_present() {
  local env_file="${REPO_ROOT}/.env"

  if [[ -f "$env_file" ]]; then
    log_debug "Loading existing .env defaults from $env_file"
    load_env_file "$env_file"
    if [[ "${ENV_VALUES[SSL]:-false}" == "true" ]]; then
      SSL_CERT_SOURCE_PATH="${ENV_VALUES[SSL_CERTFILE]:-}"
      SSL_KEY_SOURCE_PATH="${ENV_VALUES[SSL_KEYFILE]:-}"
    fi
  fi
}

clear_inherited_ssl_state() {
  unset 'ENV_VALUES[SSL]'
  unset 'ENV_VALUES[SSL_CERTFILE]'
  unset 'ENV_VALUES[SSL_KEYFILE]'
  SSL_CERT_SOURCE_PATH=""
  SSL_KEY_SOURCE_PATH=""
}

reset_quick_start_inherited_state() {
  local key

  clear_inherited_ssl_state

  for key in "${!ENV_VALUES[@]}"; do
    case "$key" in
      HOST|PORT|WEBUI_TITLE|WEBUI_DESCRIPTION|LLM_BINDING_API_KEY|EMBEDDING_BINDING_API_KEY)
        ;;
      AUTH_ACCOUNTS|TOKEN_SECRET|TOKEN_EXPIRE_HOURS|GUEST_TOKEN_EXPIRE_HOURS|JWT_ALGORITHM|TOKEN_AUTO_RENEW|TOKEN_RENEW_THRESHOLD|LIGHTRAG_API_KEY|WHITELIST_PATHS|LANGFUSE_*|LIGHTRAG_SETUP_RERANK_PROVIDER|RERANK_*|VLLM_*|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|NVIDIA_DRIVER_CAPABILITIES)
        unset "ENV_VALUES[$key]"
        ;;
    esac
  done
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

wait_for_port() {
  local host="$1"
  local port="$2"
  local label="$3"
  local timeout="${4:-$WAIT_TIMEOUT}"
  local start_time=$SECONDS

  log_step "Waiting for ${label} on ${host}:${port} (timeout ${timeout}s)"
  while true; do
    if (echo > "/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
      log_success "${label} is ready."
      return 0
    fi
    if (( SECONDS - start_time >= timeout )); then
      log_warn "Timed out waiting for ${label} (${host}:${port})."
      return 1
    fi
    sleep 2
  done
}

wait_for_services() {
  local service
  local host="127.0.0.1"
  local port=""
  local failures=()

  for service in "${DOCKER_SERVICES[@]}"; do
    case "$service" in
      postgres)
        port="${ENV_VALUES[POSTGRES_HOST_PORT]:-${ENV_VALUES[POSTGRES_PORT]:-5432}}"
        ;;
      neo4j)
        port="7687"
        ;;
      mongodb)
        port="27017"
        ;;
      redis)
        port="6379"
        ;;
      milvus)
        port="19530"
        ;;
      qdrant)
        port="6333"
        ;;
      memgraph)
        port="7687"
        ;;
      vllm-rerank)
        port="${ENV_VALUES[VLLM_RERANK_PORT]:-8000}"
        ;;
      *)
        port=""
        ;;
    esac

    if [[ -n "$port" ]]; then
      if ! wait_for_port "$host" "$port" "$service"; then
        failures+=("$service")
      fi
    fi
  done

  if ((${#failures[@]} > 0)); then
    format_error \
      "Some docker services did not become ready: ${failures[*]}" \
      "Inspect 'docker compose ps' and service logs, then rerun setup after fixing the failing services."
    return 1
  fi
}

wait_for_lightrag_service() {
  local compose_file="$1"
  local port="${ENV_VALUES[PORT]:-9621}"

  if wait_for_port "127.0.0.1" "$port" "lightrag"; then
    return 0
  fi

  format_error \
    "LightRAG did not become ready on 127.0.0.1:${port}." \
    "Inspect 'docker compose -f ${compose_file} ps' and 'docker compose -f ${compose_file} logs lightrag' before retrying."
  return 1
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
  local host="$1"

  if [[ -n "$host" && "$host" != "0.0.0.0" ]]; then
    printf '0.0.0.0'
    return 0
  fi

  printf '%s' "$host"
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

prepare_compose_runtime_overrides() {
  local normalized_value
  local key

  for key in \
    "LLM_BINDING_HOST" \
    "EMBEDDING_BINDING_HOST" \
    "RERANK_BINDING_HOST" \
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

  for key in "HOST"; do
    if [[ -n "${COMPOSE_ENV_OVERRIDES[$key]+set}" ]]; then
      continue
    fi
    if [[ -n "${ENV_VALUES[$key]:-}" ]]; then
      normalized_value="$(normalize_server_host_for_compose "${ENV_VALUES[$key]}")"
      if [[ "$normalized_value" != "${ENV_VALUES[$key]}" ]]; then
        set_compose_override "$key" "$normalized_value"
      fi
    fi
  done

  if [[ -z "${COMPOSE_ENV_OVERRIDES[PORT]+set}" && -n "${ENV_VALUES[PORT]:-}" && "${ENV_VALUES[PORT]}" != "9621" ]]; then
    set_compose_override "PORT" "9621"
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

confirm_default_yes() {
  local prompt="$1"
  local response

  read -r -p "$prompt [Y/n]: " response
  case "${response,,}" in
    n|no)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

prompt_choice() {
  local prompt="$1"
  local default="$2"
  shift 2
  local options=("$@")
  local choice
  local found
  local index=1
  local default_index=""
  local prompt_default="$default"

  while true; do
    printf '%s\n' "${COLOR_BLUE}${prompt}${COLOR_RESET} options:" >&2
    index=1
    default_index=""
    for option in "${options[@]}"; do
      if [[ "$option" == "$default" ]]; then
        default_index="$index"
      fi
      printf '  %s) %s\n' "${COLOR_GREEN}${index}${COLOR_RESET}" "$option" >&2
      index=$((index + 1))
    done
    if [[ -n "$default_index" ]]; then
      prompt_default="$default_index"
    else
      prompt_default="$default"
    fi
    printf '%s\n' "Enter a number or name (default: $prompt_default)." >&2

    choice="$(prompt_with_default "$prompt" "$prompt_default")"
    found=""
    if [[ "$choice" =~ ^[0-9]+$ ]]; then
      if ((choice >= 1 && choice <= ${#options[@]})); then
        choice="${options[choice-1]}"
        found="yes"
      fi
    else
      for option in "${options[@]}"; do
        if [[ "$choice" == "$option" ]]; then
          found="yes"
          break
        fi
      done
    fi

    if [[ -n "$found" ]]; then
      printf '%s' "$choice"
      return 0
    fi

    echo "${COLOR_YELLOW}Invalid selection.${COLOR_RESET} Please choose one of the listed options." >&2
  done
}

select_deployment_type() {
  local options=("development" "production" "custom")
  prompt_choice "Deployment type" "development" "${options[@]}"
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

    if confirm "Proceed with these storage selections anyway?"; then
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

collect_database_config() {
  local db_type="$1"
  local default_docker="${2:-no}"

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
}

collect_postgres_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local host port user password database host_port=""

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Add PostgreSQL service to docker-compose.yml?"; then
      use_docker="yes"
    fi
  else
    if confirm "Add PostgreSQL service to docker-compose.yml?"; then
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
    host_port="$(prompt_until_valid "PostgreSQL host port" "${ENV_VALUES[POSTGRES_HOST_PORT]:-${ENV_VALUES[POSTGRES_PORT]:-5432}}" validate_port)"
    port="$host_port"
    ENV_VALUES["POSTGRES_HOST_PORT"]="$host_port"
    set_compose_override "POSTGRES_HOST" "postgres"
    set_compose_override "POSTGRES_PORT" "5432"
  else
    port="$(prompt_until_valid "PostgreSQL port" "${ENV_VALUES[POSTGRES_PORT]:-5432}" validate_port)"
    set_compose_override "POSTGRES_HOST" ""
    set_compose_override "POSTGRES_PORT" ""
  fi
  user="$(prompt_with_default "PostgreSQL user" "${ENV_VALUES[POSTGRES_USER]:-lightrag}")"
  password="$(prompt_secret_with_default "PostgreSQL password: " "${ENV_VALUES[POSTGRES_PASSWORD]:-}")"
  database="$(prompt_with_default "PostgreSQL database" "${ENV_VALUES[POSTGRES_DATABASE]:-lightrag}")"

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
    if confirm_default_yes "Add Neo4j service to docker-compose.yml?"; then
      use_docker="yes"
    fi
  else
    if confirm "Add Neo4j service to docker-compose.yml?"; then
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
    username="neo4j"
  else
    username="$(prompt_with_default "Neo4j username" "${ENV_VALUES[NEO4J_USERNAME]:-neo4j}")"
  fi
  password="$(prompt_secret_with_default "Neo4j password: " "${ENV_VALUES[NEO4J_PASSWORD]:-neo4j_password}")"
  database="$(prompt_with_default "Neo4j database" "${ENV_VALUES[NEO4J_DATABASE]:-neo4j}")"

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
  local atlas_required="no"

  if [[ "${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-}" == "MongoVectorDBStorage" ]]; then
    atlas_required="yes"
  fi

  if [[ "$atlas_required" == "yes" ]]; then
    log_warn "MongoVectorDBStorage requires MongoDB Atlas. Skipping local Docker MongoDB."
    uri="mongodb+srv://cluster.example.mongodb.net/"
  else
    if [[ "$default_docker" == "yes" ]]; then
      if confirm_default_yes "Add MongoDB service to docker-compose.yml?"; then
        use_docker="yes"
      fi
    else
      if confirm "Add MongoDB service to docker-compose.yml?"; then
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

  if [[ "$atlas_required" == "yes" ]]; then
    uri="$(prompt_until_valid "MongoDB Atlas URI" "${ENV_VALUES[MONGO_URI]:-$uri}" validate_mongodb_atlas_uri)"
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
    if confirm_default_yes "Add Redis service to docker-compose.yml?"; then
      use_docker="yes"
    fi
  else
    if confirm "Add Redis service to docker-compose.yml?"; then
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
  local uri db_name

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Add Milvus service to docker-compose.yml?"; then
      use_docker="yes"
    fi
  else
    if confirm "Add Milvus service to docker-compose.yml?"; then
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
    uri="$(normalize_milvus_uri_for_local_service "$uri")"
  fi
  db_name="$(prompt_with_default "Milvus database name" "${ENV_VALUES[MILVUS_DB_NAME]:-lightrag}")"

  ENV_VALUES["MILVUS_URI"]="$uri"
  ENV_VALUES["MILVUS_DB_NAME"]="$db_name"
  if [[ "$use_docker" == "yes" ]]; then
    set_compose_override "MILVUS_URI" "http://milvus:19530"
  else
    set_compose_override "MILVUS_URI" ""
  fi
}

collect_qdrant_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local url

  if [[ "$default_docker" == "yes" ]]; then
    if confirm_default_yes "Add Qdrant service to docker-compose.yml?"; then
      use_docker="yes"
    fi
  else
    if confirm "Add Qdrant service to docker-compose.yml?"; then
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
    url="$(normalize_qdrant_uri_for_local_service "$url")"
  fi
  ENV_VALUES["QDRANT_URL"]="$url"
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
    if confirm_default_yes "Add Memgraph service to docker-compose.yml?"; then
      use_docker="yes"
    fi
  else
    if confirm "Add Memgraph service to docker-compose.yml?"; then
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

  if confirm "Store explicit AWS Bedrock credentials in .env?"; then
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
      printf 'gpt-4o'
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
      printf 'gpt-4o'
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

  case "$binding" in
    ollama)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "$(default_loopback_url 11434)")"
      host="$(prompt_with_default "Ollama embedding host" "$host_default")"
      api_key=""
      ;;
    lollms)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "http://localhost:9600")"
      host="$(prompt_with_default "LoLLMs embedding host" "$host_default")"
      api_key=""
      ;;
    azure_openai)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "https://example.openai.azure.com/")"
      host="$(prompt_with_default "Azure OpenAI endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Azure OpenAI API key: " "${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-}" validate_api_key azure_openai)"
      ;;
    gemini)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "https://generativelanguage.googleapis.com")"
      host="$(prompt_with_default "Gemini endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Gemini API key: " "${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-}" validate_api_key gemini)"
      ;;
    aws_bedrock)
      host="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "https://bedrock.amazonaws.com")"
      api_key=""
      collect_bedrock_credentials
      ;;
    jina)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "https://api.jina.ai/v1/embeddings")"
      host="$(prompt_with_default "Jina endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Jina API key: " "${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-}" validate_api_key jina)"
      ;;
    *)
      host_default="$(provider_default_or_existing "$binding" "$current_binding" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-}" "https://api.openai.com/v1")"
      host="$(prompt_with_default "Embedding endpoint" "$host_default")"
      api_key="$(prompt_secret_until_valid_with_default "Embedding API key: " "${ENV_VALUES[EMBEDDING_BINDING_API_KEY]:-}" validate_api_key "$binding")"
      ;;
  esac

  ENV_VALUES["EMBEDDING_BINDING"]="$binding"
  ENV_VALUES["EMBEDDING_MODEL"]="$model"
  ENV_VALUES["EMBEDDING_DIM"]="$dim"
  ENV_VALUES["EMBEDDING_BINDING_HOST"]="$host"
  store_optional_env_value "EMBEDDING_BINDING_API_KEY" "$api_key"
  clear_bedrock_credentials_if_unused
}

collect_rerank_config() {
  local options=("cohere" "jina" "aliyun" "vllm")
  local binding_choice binding model host api_key
  local vllm_model vllm_port vllm_device vllm_dtype vllm_extra
  local default_dtype=""
  local existing_vllm_device=""
  local existing_vllm_dtype=""
  local default_model="" default_host="" model_default="" host_default="" use_docker="no"
  local previous_provider="${ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]:-}"
  local reset_vllm_defaults="no"
  local rerank_default="${ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]:-${ENV_VALUES[RERANK_BINDING]:-cohere}}"

  if ! confirm "Enable reranking?"; then
    ENV_VALUES["RERANK_BINDING"]="null"
    unset 'ENV_VALUES[LIGHTRAG_SETUP_RERANK_PROVIDER]'
    unset 'ENV_VALUES[RERANK_BINDING_API_KEY]'
    return
  fi

  if [[ "$rerank_default" == "null" ]]; then
    rerank_default="cohere"
  fi

  binding_choice="$(prompt_choice "Rerank provider" "$rerank_default" "${options[@]}")"
  if [[ "$binding_choice" != "vllm" && "$previous_provider" == "vllm" ]]; then
    reset_vllm_defaults="yes"
  fi

  if [[ "$binding_choice" == "vllm" ]]; then
    log_info "vLLM uses the Cohere-compatible rerank API."
    if confirm_default_yes "Add local vLLM rerank service to docker-compose.yml?"; then
      add_docker_service "vllm-rerank"
      use_docker="yes"
    fi
    vllm_model="$(prompt_with_default "vLLM rerank model" "${ENV_VALUES[VLLM_RERANK_MODEL]:-BAAI/bge-reranker-v2-m3}")"
    vllm_port="$(prompt_until_valid "vLLM rerank port" "${ENV_VALUES[VLLM_RERANK_PORT]:-8000}" validate_port)"
    vllm_device="$(prompt_choice "vLLM device" "${ENV_VALUES[VLLM_RERANK_DEVICE]:-cpu}" "cpu" "cuda")"
    if [[ "$vllm_device" == "cuda" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
      log_warn "CUDA device selected but no NVIDIA driver detected on host."
      if confirm_default_yes "Use CPU instead?"; then
        vllm_device="cpu"
      fi
    fi
    existing_vllm_device="${ENV_VALUES[VLLM_RERANK_DEVICE]:-}"
    existing_vllm_dtype="${ENV_VALUES[VLLM_RERANK_DTYPE]:-}"
    if [[ -n "$existing_vllm_dtype" && "$existing_vllm_device" == "$vllm_device" ]]; then
      default_dtype="$existing_vllm_dtype"
    fi
    if [[ -z "$default_dtype" ]]; then
      if [[ "$vllm_device" == "cpu" ]]; then
        default_dtype="float32"
      else
        default_dtype="float16"
      fi
    fi
    vllm_dtype="$(prompt_with_default "vLLM dtype" "$default_dtype")"
    vllm_extra="$(prompt_with_default "vLLM extra args" "${ENV_VALUES[VLLM_RERANK_EXTRA_ARGS]:-}")"

    if [[ "$vllm_device" == "cuda" ]]; then
      if [[ "${ENV_VALUES[CUDA_VISIBLE_DEVICES]:-}" == "-1" ]]; then
        unset 'ENV_VALUES[CUDA_VISIBLE_DEVICES]'
      fi
      if [[ "${ENV_VALUES[NVIDIA_VISIBLE_DEVICES]:-}" == "-1" ]]; then
        unset 'ENV_VALUES[NVIDIA_VISIBLE_DEVICES]'
      fi
      unset 'ENV_VALUES[VLLM_USE_CPU]'
    fi

    ENV_VALUES["VLLM_RERANK_MODEL"]="$vllm_model"
    ENV_VALUES["VLLM_RERANK_PORT"]="$vllm_port"
    ENV_VALUES["VLLM_RERANK_DEVICE"]="$vllm_device"
    ENV_VALUES["VLLM_RERANK_DTYPE"]="$vllm_dtype"
    if [[ -n "$vllm_extra" ]]; then
      ENV_VALUES["VLLM_RERANK_EXTRA_ARGS"]="$vllm_extra"
    fi

    default_model="$vllm_model"
    default_host="$(default_loopback_url "$vllm_port" "/v1/rerank")"
    if [[ "$use_docker" == "yes" ]]; then
      set_compose_override "RERANK_BINDING_HOST" "http://vllm-rerank:${vllm_port}/v1/rerank"
    else
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
    model_default="$default_model"
    host_default="$default_host"
  else
    model_default="${ENV_VALUES[RERANK_MODEL]:-$default_model}"
    host_default="${ENV_VALUES[RERANK_BINDING_HOST]:-$default_host}"
  fi

  model="$(prompt_with_default "Rerank model" "$model_default")"
  host="$(prompt_with_default "Rerank endpoint" "$host_default")"
  if [[ "$binding_choice" == "vllm" ]]; then
    api_key="$(prompt_secret_with_default "Rerank API key (optional): " "${ENV_VALUES[RERANK_BINDING_API_KEY]:-}")"
  else
    api_key="$(prompt_secret_until_valid_with_default "Rerank API key: " "${ENV_VALUES[RERANK_BINDING_API_KEY]:-}" validate_api_key "$binding")"
  fi

  ENV_VALUES["RERANK_BINDING"]="$binding"
  ENV_VALUES["LIGHTRAG_SETUP_RERANK_PROVIDER"]="$binding_choice"
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
  store_optional_env_value "RERANK_BINDING_API_KEY" "$api_key"
}

collect_server_config() {
  local host port title description

  host="$(prompt_with_default "Server host" "${ENV_VALUES[HOST]:-0.0.0.0}")"
  port="$(prompt_until_valid "Server port" "${ENV_VALUES[PORT]:-9621}" validate_port)"
  title="$(prompt_with_default "WebUI title" "${ENV_VALUES[WEBUI_TITLE]:-My Graph KB}")"
  description="$(prompt_with_default "WebUI description" "${ENV_VALUES[WEBUI_DESCRIPTION]:-Simple and Fast Graph Based RAG System}")"

  ENV_VALUES["HOST"]="$host"
  ENV_VALUES["PORT"]="$port"
  ENV_VALUES["WEBUI_TITLE"]="$title"
  ENV_VALUES["WEBUI_DESCRIPTION"]="$description"
}

collect_ssl_config() {
  local cert key

  if ! confirm_default_yes "Enable SSL/TLS for the API server?"; then
    unset 'ENV_VALUES[SSL]'
    unset 'ENV_VALUES[SSL_CERTFILE]'
    unset 'ENV_VALUES[SSL_KEYFILE]'
    SSL_CERT_SOURCE_PATH=""
    SSL_KEY_SOURCE_PATH=""
    return
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
    if confirm "Configure authentication and API key settings?"; then
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

  if ! confirm "Enable Langfuse observability?"; then
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

collect_docker_image_tags() {
  local service env_var default_tag selected_tag
  local -a tag_services=()

  if ((${#DOCKER_SERVICES[@]} == 0)); then
    return
  fi

  tag_services=("${DOCKER_SERVICES[@]}")
  if [[ -n "${DOCKER_SERVICE_SET[milvus]+set}" ]]; then
    tag_services+=("etcd" "minio")
  fi

  log_info "Docker image settings for selected services:"
  for service in "${tag_services[@]}"; do
    env_var="${DOCKER_IMAGE_ENV[$service]}"
    default_tag="${DOCKER_IMAGE_DEFAULTS[$service]}"
    if [[ -n "$env_var" && -n "$default_tag" ]]; then
      ENV_VALUES["$env_var"]="${ENV_VALUES[$env_var]:-$default_tag}"
      echo "  - $service: ${ENV_VALUES[$env_var]} ($env_var)"
    fi
  done

  if confirm "Keep these image settings?"; then
    return
  fi

  for service in "${tag_services[@]}"; do
    env_var="${DOCKER_IMAGE_ENV[$service]}"
    default_tag="${ENV_VALUES[$env_var]:-${DOCKER_IMAGE_DEFAULTS[$service]}}"
    if [[ -n "$env_var" ]]; then
      selected_tag="$(prompt_with_default "Image setting for $service ($env_var)" "$default_tag")"
      ENV_VALUES["$env_var"]="$selected_tag"
    fi
  done
}

show_summary() {
  local key
  local value
  local compose_suffix=""

  echo
  log_info "Configuration summary:"
  for key in "${!ENV_VALUES[@]}"; do
    value="${ENV_VALUES[$key]}"
    if is_sensitive_env_key "$key"; then
      value="***"
    fi
    printf '  %s=%s\n' "$key" "$value"
  done

  if ((${#DOCKER_SERVICES[@]} > 0)); then
    echo
    log_info "Docker services to include:"
    for service in "${DOCKER_SERVICES[@]}"; do
      echo "  - $service"
    done
    compose_suffix="${DEPLOYMENT_TYPE:-custom}"
    echo "  Compose file: docker-compose.${compose_suffix}.yml"
  fi
}

require_production_security_profile() {
  local setup_profile="${ENV_VALUES[LIGHTRAG_SETUP_PROFILE]:-${DEPLOYMENT_TYPE:-}}"

  if [[ "$setup_profile" == "production" ]]; then
    return 0
  fi

  is_production_storage_profile \
    "${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-}" \
    "${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-}" \
    "${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]:-}" \
    "${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]:-}"
}

finalize_setup() {
  local backup_path
  local compose_suffix
  local compose_file
  local generate_compose="no"
  local require_protection="no"

  if [[ ! -f "${REPO_ROOT}/env.example" ]]; then
    format_error "env.example is missing in $REPO_ROOT" "Restore env.example before running setup."
    return 1
  fi

  if [[ ! -w "$REPO_ROOT" ]]; then
    format_error "No write permission in $REPO_ROOT" "Run the setup from a writable directory."
    return 1
  fi

  if [[ -n "$DEPLOYMENT_TYPE" ]]; then
    ENV_VALUES["LIGHTRAG_SETUP_PROFILE"]="$DEPLOYMENT_TYPE"
  else
    unset 'ENV_VALUES[LIGHTRAG_SETUP_PROFILE]'
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
    "${ENV_VALUES[MONGO_URI]:-}"; then
    return 1
  fi

  if require_production_security_profile; then
    require_protection="yes"
  fi

  if ! validate_security_config \
    "${ENV_VALUES[AUTH_ACCOUNTS]:-}" \
    "${ENV_VALUES[TOKEN_SECRET]:-}" \
    "${ENV_VALUES[LIGHTRAG_API_KEY]:-}" \
    "$require_protection" \
    "${ENV_VALUES[WHITELIST_PATHS]:-}" \
    "${ENV_VALUES[WHITELIST_PATHS]+set}"; then
    return 1
  fi

  if ! validate_sensitive_env_literals; then
    return 1
  fi

  show_summary

  if ! confirm "Generate .env and docker-compose.yml now?"; then
    log_warn "Setup cancelled."
    return 1
  fi

  if ((${#DOCKER_SERVICES[@]} > 0)); then
    generate_compose="yes"
  else
    if confirm "Generate docker-compose for LightRAG only?"; then
      generate_compose="yes"
    fi
  fi

  if [[ "$generate_compose" == "yes" ]]; then
    prepare_compose_env_overrides
  fi

  backup_path="$(backup_env_file)"
  if [[ -n "$backup_path" ]]; then
    log_success "Backed up existing .env to $backup_path"
  fi

  if [[ -n "$SSL_CERT_SOURCE_PATH" ]] && ! validate_existing_file "$SSL_CERT_SOURCE_PATH"; then
    format_error \
      "Invalid SSL_CERTFILE" \
      "Set it to an existing certificate file, disable SSL, or rerun the full setup to choose a new certificate."
    return 1
  fi

  if [[ -n "$SSL_KEY_SOURCE_PATH" ]] && ! validate_existing_file "$SSL_KEY_SOURCE_PATH"; then
    format_error \
      "Invalid SSL_KEYFILE" \
      "Set it to an existing private key file, disable SSL, or rerun the full setup to choose a new key."
    return 1
  fi

  if [[ -n "$SSL_CERT_SOURCE_PATH" || -n "$SSL_KEY_SOURCE_PATH" ]]; then
    stage_ssl_assets "$SSL_CERT_SOURCE_PATH" "$SSL_KEY_SOURCE_PATH"
  fi

  log_debug "Writing .env to ${REPO_ROOT}/.env"
  generate_env_file "${REPO_ROOT}/env.example" "${REPO_ROOT}/.env"
  log_success "Wrote .env"

  if [[ "$generate_compose" == "yes" ]]; then
    compose_suffix="${DEPLOYMENT_TYPE:-custom}"
    compose_file="${REPO_ROOT}/docker-compose.${compose_suffix}.yml"
    if [[ -f "$compose_file" ]]; then
      if ! confirm "Overwrite existing ${compose_file}?"; then
        compose_file="${REPO_ROOT}/docker-compose.${compose_suffix}.$(date +%Y%m%d_%H%M%S).yml"
        log_warn "Using new compose file: $compose_file"
      fi
    fi
    generate_docker_compose "$compose_file"
    log_success "Wrote ${compose_file}"
    if confirm_default_yes "Start docker services now?"; then
      if ! check_docker_availability; then
        return 1
      fi
      log_step "Starting docker services with ${compose_file}"
      if ((${#DOCKER_SERVICES[@]} > 0)); then
        docker compose -f "$compose_file" up -d "${DOCKER_SERVICES[@]}"
        if ! wait_for_services; then
          return 1
        fi
      fi
      docker compose -f "$compose_file" up -d lightrag
      if ! wait_for_lightrag_service "$compose_file"; then
        return 1
      fi
      log_success "Docker services are up."
    fi
  else
    log_warn "No docker services selected."
  fi
}

interactive_flow() {
  local deployment_type
  local db_type
  local db_order=("postgresql" "neo4j" "mongodb" "redis" "milvus" "qdrant" "memgraph")

  reset_state
  load_existing_env_if_present

  log_info "Interactive setup wizard"
  log_step "Step 1: Deployment type"
  echo "  - development: local JSON/NetworkX defaults"
  echo "  - production: database-backed defaults with security prompts"
  echo "  - custom: pick each backend manually"
  deployment_type="$(select_deployment_type)"
  DEPLOYMENT_TYPE="$deployment_type"

  case "$deployment_type" in
    development)
      load_storage_preset_overwrite "development"
      load_preset "development"
      ;;
    production)
      load_storage_preset_overwrite "production"
      load_preset "production"
      ;;
    custom)
      ;;
  esac

  log_step "Step 2: Storage backends"
  select_storage_backends "$deployment_type"
  log_debug "Storage selections: kv=${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-} vector=${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-} graph=${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]:-} doc=${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]:-}"

  log_step "Step 3: Database configuration"
  for db_type in "${db_order[@]}"; do
    if [[ -n "${REQUIRED_DB_TYPES[$db_type]+set}" ]]; then
      collect_database_config "$db_type" "no"
    fi
  done

  log_step "Step 4: LLM configuration"
  collect_llm_config
  log_step "Step 5: Embedding configuration"
  collect_embedding_config
  log_step "Step 6: Reranking configuration"
  collect_rerank_config
  collect_docker_image_tags
  log_step "Step 7: Server configuration"
  collect_server_config
  if [[ "$deployment_type" == "production" ]]; then
    log_step "Step 8: SSL configuration"
    collect_ssl_config
    log_step "Step 9: Security configuration"
  else
    log_step "Step 8: Security configuration"
  fi
  if [[ "$deployment_type" == "production" ]]; then
    collect_security_config "yes" "yes"
  else
    collect_security_config "no" "no"
  fi
  if [[ "$deployment_type" == "production" ]]; then
    log_step "Step 10: Observability configuration"
  else
    log_step "Step 9: Observability configuration"
  fi
  collect_observability_config

  if [[ "$deployment_type" != "production" ]]; then
    clear_inherited_ssl_state
  fi

  finalize_setup
}

quick_start_flow() {
  local api_key

  reset_state
  load_existing_env_if_present
  reset_quick_start_inherited_state
  apply_preset_overwrite "${PRESET_DEVELOPMENT[@]}"
  DEPLOYMENT_TYPE="development"
  clear_bedrock_credentials_if_unused

  log_info "Quick start setup"
  echo "Using development preset. The wizard stores an OpenAI API key in .env for convenience, but you can override it with runtime environment variables later."

  api_key="$(prompt_secret_until_valid_with_default "OpenAI API key: " "${ENV_VALUES[LLM_BINDING_API_KEY]:-}" validate_api_key openai)"
  ENV_VALUES["LLM_BINDING_API_KEY"]="$api_key"
  ENV_VALUES["EMBEDDING_BINDING_API_KEY"]="$api_key"

  finalize_setup
}

production_flow() {
  local db_type
  local db_order=("postgresql" "neo4j" "mongodb" "redis" "milvus" "qdrant" "memgraph")

  reset_state
  load_existing_env_if_present
  load_storage_preset_overwrite "production"
  load_preset "production"
  DEPLOYMENT_TYPE="production"

  log_info "Production setup wizard"
  echo "Recommended defaults are preselected. Customize as needed."

  select_storage_backends "production"
  log_debug "Storage selections: kv=${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-} vector=${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-} graph=${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]:-} doc=${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]:-}"

  log_step "Configuring database services"
  for db_type in "${db_order[@]}"; do
    if [[ -n "${REQUIRED_DB_TYPES[$db_type]+set}" ]]; then
      collect_database_config "$db_type" "yes"
    fi
  done

  log_step "Configuring LLM and embedding providers"
  collect_llm_config
  collect_embedding_config
  collect_rerank_config
  collect_docker_image_tags
  log_step "Configuring server and security settings"
  collect_server_config
  collect_ssl_config
  collect_security_config "yes" "yes"
  collect_observability_config

  finalize_setup
}

load_env_file() {
  local env_file="$1"
  local line key value

  if [[ ! -f "$env_file" ]]; then
    format_error ".env file not found at $env_file" "Run make setup to generate it."
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

is_production_storage_profile() {
  local kv="$1"
  local vector="$2"
  local graph="$3"
  local doc_status="$4"

  [[ "$kv" == "PGKVStorage" &&
    "$vector" == "MilvusVectorDBStorage" &&
    "$graph" == "Neo4JStorage" &&
    "$doc_status" == "PGDocStatusStorage" ]]
}

validate_env_file() {
  local env_file="${REPO_ROOT}/.env"
  local errors=0
  local kv vector graph doc_status
  local require_protection="no"

  reset_state

  if ! load_env_file "$env_file"; then
    return 1
  fi

  kv="${ENV_VALUES[LIGHTRAG_KV_STORAGE]:-}"
  vector="${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]:-}"
  graph="${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]:-}"
  doc_status="${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]:-}"

  if [[ -z "$kv" || -z "$vector" || -z "$graph" || -z "$doc_status" ]]; then
    format_error "Storage selections are missing in .env" "Set LIGHTRAG_*_STORAGE variables."
    return 1
  fi

  if ! validate_required_variables "$kv" "$vector" "$graph" "$doc_status"; then
    errors=1
  fi

  if ! validate_mongo_vector_storage_config "$vector" "${ENV_VALUES[MONGO_URI]:-}"; then
    errors=1
  fi

  if require_production_security_profile; then
    require_protection="yes"
  fi

  if ! validate_security_config \
    "${ENV_VALUES[AUTH_ACCOUNTS]:-}" \
    "${ENV_VALUES[TOKEN_SECRET]:-}" \
    "${ENV_VALUES[LIGHTRAG_API_KEY]:-}" \
    "$require_protection" \
    "${ENV_VALUES[WHITELIST_PATHS]:-}" \
    "${ENV_VALUES[WHITELIST_PATHS]+set}"; then
    errors=1
  fi

  if ! validate_sensitive_env_literals; then
    errors=1
  fi

  if [[ "${ENV_VALUES[SSL]:-false}" == "true" ]]; then
    if ! validate_existing_file "${ENV_VALUES[SSL_CERTFILE]:-}"; then
      format_error "Invalid SSL_CERTFILE" "Set it to an existing certificate file when SSL=true."
      errors=1
    fi
    if ! validate_existing_file "${ENV_VALUES[SSL_KEYFILE]:-}"; then
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
  if [[ -n "${ENV_VALUES[POSTGRES_HOST_PORT]:-}" ]] && ! validate_port "${ENV_VALUES[POSTGRES_HOST_PORT]}"; then
    format_error "Invalid POSTGRES_HOST_PORT" "Use a port between 1 and 65535."
    errors=1
  fi

  if ((errors != 0)); then
    return 1
  fi

  log_success "Validation passed."
}

backup_only() {
  local backup_path

  backup_path="$(backup_env_file)"
  if [[ -z "$backup_path" ]]; then
    format_error "No .env file found to back up." "Create one with make setup first."
    return 1
  fi
  echo "Backed up .env to $backup_path"
}

print_help() {
  cat <<'HELP'
Usage: scripts/setup/setup.sh [--quick|--production|--validate|--backup]

Options:
  --quick        Run the quick start flow (development preset, minimal prompts)
  --production   Run the production preset flow (recommended defaults)
  --validate     Validate an existing .env file
  --backup       Backup the current .env file
  --debug        Enable debug logging
  --help         Show this help message
HELP
}

main() {
  init_colors
  local mode="interactive"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --quick)
        mode="quick"
        ;;
      --production)
        mode="production"
        ;;
      --validate)
        mode="validate"
        ;;
      --backup)
        mode="backup"
        ;;
      --debug)
        DEBUG="true"
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
    quick)
      quick_start_flow
      ;;
    production)
      production_flow
      ;;
    validate)
      validate_env_file
      ;;
    backup)
      backup_only
      ;;
    help)
      print_help
      ;;
    *)
      interactive_flow
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
