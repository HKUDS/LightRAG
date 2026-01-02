#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
PRESETS_DIR="$SCRIPT_DIR/presets"
# shellcheck disable=SC2034
TEMPLATES_DIR="$SCRIPT_DIR/templates"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

declare -A ENV_VALUES
declare -A REQUIRED_DB_TYPES
declare -A DOCKER_SERVICE_SET
declare -a DOCKER_SERVICES
declare -A DOCKER_IMAGE_TAG_DEFAULTS=(
  ["postgres"]="18"
  ["neo4j"]="5.26.19-community"
  ["mongodb"]="8.2.3"
  ["redis"]="8.4"
  ["milvus"]="2.6-20251227-44275071"
  ["etcd"]="v3.5.16"
  ["minio"]="RELEASE.2024-12-13T22-19-12Z"
  ["qdrant"]="v1.16-gpu-nvidia"
  ["memgraph"]="3.7.2"
  ["vllm-rerank"]="latest"
)
declare -A DOCKER_IMAGE_TAG_ENV=(
  ["postgres"]="POSTGRES_IMAGE_TAG"
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
  REQUIRED_DB_TYPES=()
  DOCKER_SERVICE_SET=()
  DOCKER_SERVICES=()
  DEPLOYMENT_TYPE=""
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

  for service in "${DOCKER_SERVICES[@]}"; do
    case "$service" in
      postgres)
        port="${ENV_VALUES[POSTGRES_PORT]:-5432}"
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
      wait_for_port "$host" "$port" "$service" || true
    fi
  done
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
  local host port user password database

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
    host="postgres"
  else
    host="localhost"
  fi

  host="$(prompt_with_default "PostgreSQL host" "$host")"
  port="$(prompt_until_valid "PostgreSQL port" "5432" validate_port)"
  user="$(prompt_with_default "PostgreSQL user" "lightrag")"
  password="$(mask_sensitive_input "PostgreSQL password: ")"
  database="$(prompt_with_default "PostgreSQL database" "lightrag")"

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
    uri="neo4j://neo4j:7687"
  else
    uri="neo4j://localhost:7687"
  fi

  uri="$(prompt_until_valid "Neo4j URI" "$uri" validate_uri neo4j)"
  username="$(prompt_with_default "Neo4j username" "neo4j")"
  password="$(prompt_with_default "Neo4j password" "neo4j_password")"
  database="$(prompt_with_default "Neo4j database" "neo4j")"

  ENV_VALUES["NEO4J_URI"]="$uri"
  ENV_VALUES["NEO4J_USERNAME"]="$username"
  ENV_VALUES["NEO4J_PASSWORD"]="$password"
  ENV_VALUES["NEO4J_DATABASE"]="$database"
}

collect_mongodb_config() {
  local default_docker="${1:-no}"
  local use_docker="no"
  local uri database

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
    uri="mongodb://mongodb:27017/"
  else
    uri="mongodb://localhost:27017/"
  fi

  uri="$(prompt_until_valid "MongoDB URI" "$uri" validate_uri mongodb)"
  database="$(prompt_with_default "MongoDB database" "LightRAG")"

  ENV_VALUES["MONGO_URI"]="$uri"
  ENV_VALUES["MONGO_DATABASE"]="$database"
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
    uri="redis://redis:6379"
  else
    uri="redis://localhost:6379"
  fi

  uri="$(prompt_until_valid "Redis URI" "$uri" validate_uri redis)"
  ENV_VALUES["REDIS_URI"]="$uri"
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
    uri="http://milvus:19530"
  else
    uri="http://localhost:19530"
  fi

  uri="$(prompt_until_valid "Milvus URI" "$uri" validate_uri milvus)"
  db_name="$(prompt_with_default "Milvus database name" "default")"

  ENV_VALUES["MILVUS_URI"]="$uri"
  ENV_VALUES["MILVUS_DB_NAME"]="$db_name"
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
    url="http://qdrant:6333"
  else
    url="http://localhost:6333"
  fi

  url="$(prompt_until_valid "Qdrant URL" "$url" validate_uri qdrant)"
  ENV_VALUES["QDRANT_URL"]="$url"
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
    uri="bolt://memgraph:7687"
  else
    uri="bolt://localhost:7687"
  fi

  uri="$(prompt_until_valid "Memgraph URI" "$uri" validate_uri memgraph)"
  ENV_VALUES["MEMGRAPH_URI"]="$uri"
}

collect_llm_config() {
  local options=("openai" "azure_openai" "ollama" "gemini" "aws_bedrock")
  local binding model host api_key

  binding="$(prompt_choice "LLM provider" "${ENV_VALUES[LLM_BINDING]:-openai}" "${options[@]}")"
  model="$(prompt_with_default "LLM model" "${ENV_VALUES[LLM_MODEL]:-gpt-4o}")"

  case "$binding" in
    ollama)
      host="$(prompt_with_default "Ollama host" "${ENV_VALUES[LLM_BINDING_HOST]:-http://localhost:11434}")"
      api_key=""
      ;;
    azure_openai)
      host="$(prompt_with_default "Azure OpenAI endpoint" "${ENV_VALUES[LLM_BINDING_HOST]:-https://example.openai.azure.com/}")"
      api_key="$(prompt_secret_until_valid "Azure OpenAI API key: " validate_api_key azure_openai)"
      ;;
    gemini)
      host="$(prompt_with_default "Gemini endpoint" "${ENV_VALUES[LLM_BINDING_HOST]:-https://generativelanguage.googleapis.com}")"
      api_key="$(prompt_secret_until_valid "Gemini API key: " validate_api_key gemini)"
      ;;
    aws_bedrock)
      host="$(prompt_with_default "Bedrock endpoint" "${ENV_VALUES[LLM_BINDING_HOST]:-https://bedrock.amazonaws.com}")"
      api_key="$(prompt_secret_until_valid "Bedrock API key: " validate_api_key aws_bedrock)"
      ;;
    *)
      host="$(prompt_with_default "LLM endpoint" "${ENV_VALUES[LLM_BINDING_HOST]:-https://api.openai.com/v1}")"
      api_key="$(prompt_secret_until_valid "LLM API key: " validate_api_key "$binding")"
      ;;
  esac

  ENV_VALUES["LLM_BINDING"]="$binding"
  ENV_VALUES["LLM_MODEL"]="$model"
  ENV_VALUES["LLM_BINDING_HOST"]="$host"
  if [[ -n "$api_key" ]]; then
    ENV_VALUES["LLM_BINDING_API_KEY"]="$api_key"
  fi
}

collect_embedding_config() {
  local options=("openai" "azure_openai" "ollama" "jina" "gemini" "aws_bedrock")
  local binding model host api_key dim

  binding="$(prompt_choice "Embedding provider" "${ENV_VALUES[EMBEDDING_BINDING]:-openai}" "${options[@]}")"
  model="$(prompt_with_default "Embedding model" "${ENV_VALUES[EMBEDDING_MODEL]:-text-embedding-3-large}")"
  dim="$(prompt_with_default "Embedding dimension" "${ENV_VALUES[EMBEDDING_DIM]:-3072}")"

  case "$binding" in
    ollama)
      host="$(prompt_with_default "Ollama embedding host" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-http://localhost:11434}")"
      api_key=""
      ;;
    azure_openai)
      host="$(prompt_with_default "Azure OpenAI endpoint" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-https://example.openai.azure.com/}")"
      api_key="$(prompt_secret_until_valid "Azure OpenAI API key: " validate_api_key azure_openai)"
      ;;
    gemini)
      host="$(prompt_with_default "Gemini endpoint" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-https://generativelanguage.googleapis.com}")"
      api_key="$(prompt_secret_until_valid "Gemini API key: " validate_api_key gemini)"
      ;;
    aws_bedrock)
      host="$(prompt_with_default "Bedrock endpoint" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-https://bedrock.amazonaws.com}")"
      api_key="$(prompt_secret_until_valid "Bedrock API key: " validate_api_key aws_bedrock)"
      ;;
    jina)
      host="$(prompt_with_default "Jina endpoint" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-https://api.jina.ai/v1/embeddings}")"
      api_key="$(prompt_secret_until_valid "Jina API key: " validate_api_key jina)"
      ;;
    *)
      host="$(prompt_with_default "Embedding endpoint" "${ENV_VALUES[EMBEDDING_BINDING_HOST]:-https://api.openai.com/v1}")"
      api_key="$(prompt_secret_until_valid "Embedding API key: " validate_api_key "$binding")"
      ;;
  esac

  ENV_VALUES["EMBEDDING_BINDING"]="$binding"
  ENV_VALUES["EMBEDDING_MODEL"]="$model"
  ENV_VALUES["EMBEDDING_DIM"]="$dim"
  ENV_VALUES["EMBEDDING_BINDING_HOST"]="$host"
  if [[ -n "$api_key" ]]; then
    ENV_VALUES["EMBEDDING_BINDING_API_KEY"]="$api_key"
  fi
}

collect_rerank_config() {
  local options=("cohere" "jina" "aliyun" "vllm")
  local binding_choice binding model host api_key
  local vllm_model vllm_port vllm_device vllm_dtype vllm_extra
  local default_dtype=""
  local default_model="" default_host="" use_docker="no"

  if ! confirm "Enable reranking?"; then
    ENV_VALUES["RERANK_BINDING"]="null"
    return
  fi

  binding_choice="$(prompt_choice "Rerank provider" "cohere" "${options[@]}")"

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
    default_dtype="${ENV_VALUES[VLLM_RERANK_DTYPE]:-}"
    if [[ -z "$default_dtype" ]]; then
      if [[ "$vllm_device" == "cpu" ]]; then
        default_dtype="float32"
      else
        default_dtype="float16"
      fi
    fi
    vllm_dtype="$(prompt_with_default "vLLM dtype" "$default_dtype")"
    vllm_extra="$(prompt_with_default "vLLM extra args" "${ENV_VALUES[VLLM_RERANK_EXTRA_ARGS]:-}")"

    ENV_VALUES["VLLM_RERANK_MODEL"]="$vllm_model"
    ENV_VALUES["VLLM_RERANK_PORT"]="$vllm_port"
    ENV_VALUES["VLLM_RERANK_DEVICE"]="$vllm_device"
    ENV_VALUES["VLLM_RERANK_DTYPE"]="$vllm_dtype"
    if [[ -n "$vllm_extra" ]]; then
      ENV_VALUES["VLLM_RERANK_EXTRA_ARGS"]="$vllm_extra"
    fi

    default_model="$vllm_model"
    if [[ "$use_docker" == "yes" ]]; then
      default_host="http://vllm-rerank:${vllm_port}/v1/rerank"
    else
      default_host="http://localhost:${vllm_port}/v1/rerank"
    fi
    binding="cohere"
  else
    binding="$binding_choice"
  fi

  model="$(prompt_with_default "Rerank model" "${ENV_VALUES[RERANK_MODEL]:-$default_model}")"
  host="$(prompt_with_default "Rerank endpoint" "${ENV_VALUES[RERANK_BINDING_HOST]:-$default_host}")"
  if [[ "$binding_choice" == "vllm" ]]; then
    api_key="$(prompt_with_default "Rerank API key (optional)" "${ENV_VALUES[RERANK_BINDING_API_KEY]:-}")"
  else
    api_key="$(prompt_secret_until_valid "Rerank API key: " validate_api_key "$binding")"
  fi

  ENV_VALUES["RERANK_BINDING"]="$binding"
  if [[ -n "$model" ]]; then
    ENV_VALUES["RERANK_MODEL"]="$model"
  fi
  if [[ -n "$host" ]]; then
    ENV_VALUES["RERANK_BINDING_HOST"]="$host"
  fi
  if [[ -n "$api_key" ]]; then
    ENV_VALUES["RERANK_BINDING_API_KEY"]="$api_key"
  fi
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
    return
  fi

  cert="$(prompt_with_default "SSL certificate file" "")"
  key="$(prompt_with_default "SSL key file" "")"

  ENV_VALUES["SSL"]="true"
  if [[ -n "$cert" ]]; then
    ENV_VALUES["SSL_CERTFILE"]="$cert"
  fi
  if [[ -n "$key" ]]; then
    ENV_VALUES["SSL_KEYFILE"]="$key"
  fi
}

collect_security_config() {
  local required="${1:-no}"
  local default_yes="${2:-no}"
  local auth_accounts token_secret token_expire api_key whitelist
  local confirm_result=1

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
      echo "Warning: production deployments should set authentication and API keys." >&2
    fi
    return
  fi

  auth_accounts="$(prompt_with_default "Auth accounts (user:pass,comma-separated)" "")"
  token_secret="$(mask_sensitive_input "JWT token secret: ")"
  token_expire="$(prompt_with_default "Token expire hours" "48")"
  api_key="$(mask_sensitive_input "LightRAG API key: ")"
  whitelist="$(prompt_with_default "Whitelist paths (comma-separated)" "/health,/api/*")"

  if [[ -n "$auth_accounts" ]]; then
    ENV_VALUES["AUTH_ACCOUNTS"]="$auth_accounts"
  fi
  if [[ -n "$token_secret" ]]; then
    ENV_VALUES["TOKEN_SECRET"]="$token_secret"
  fi
  if [[ -n "$token_expire" ]]; then
    ENV_VALUES["TOKEN_EXPIRE_HOURS"]="$token_expire"
  fi
  if [[ -n "$api_key" ]]; then
    ENV_VALUES["LIGHTRAG_API_KEY"]="$api_key"
  fi
  if [[ -n "$whitelist" ]]; then
    ENV_VALUES["WHITELIST_PATHS"]="$whitelist"
  fi
}

collect_observability_config() {
  local secret_key public_key host

  if ! confirm "Enable Langfuse observability?"; then
    return
  fi

  secret_key="$(prompt_secret_until_valid "Langfuse secret key: " validate_api_key langfuse)"
  public_key="$(prompt_secret_until_valid "Langfuse public key: " validate_api_key langfuse)"
  host="$(prompt_with_default "Langfuse host" "https://cloud.langfuse.com")"

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

  log_info "Docker image tags for selected services:"
  for service in "${tag_services[@]}"; do
    env_var="${DOCKER_IMAGE_TAG_ENV[$service]}"
    default_tag="${DOCKER_IMAGE_TAG_DEFAULTS[$service]}"
    if [[ -n "$env_var" && -n "$default_tag" ]]; then
      ENV_VALUES["$env_var"]="${ENV_VALUES[$env_var]:-$default_tag}"
      echo "  - $service: ${ENV_VALUES[$env_var]} ($env_var)"
    fi
  done

  if confirm "Keep these image tags?"; then
    return
  fi

  for service in "${tag_services[@]}"; do
    env_var="${DOCKER_IMAGE_TAG_ENV[$service]}"
    default_tag="${ENV_VALUES[$env_var]:-${DOCKER_IMAGE_TAG_DEFAULTS[$service]}}"
    if [[ -n "$env_var" ]]; then
      selected_tag="$(prompt_with_default "Tag for $service ($env_var)" "$default_tag")"
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
    if [[ "$key" =~ (KEY|PASSWORD|SECRET|TOKEN) ]]; then
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

finalize_setup() {
  local backup_path
  local compose_suffix
  local compose_file
  local generate_compose="no"

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

  show_summary

  if ! confirm "Generate .env and docker-compose.yml now?"; then
    log_warn "Setup cancelled."
    return 1
  fi

  backup_path="$(backup_env_file)"
  if [[ -n "$backup_path" ]]; then
    log_success "Backed up existing .env to $backup_path"
  fi

  log_debug "Writing .env to ${REPO_ROOT}/.env"
  generate_env_file "${REPO_ROOT}/env.example" "${REPO_ROOT}/.env"
  log_success "Wrote .env"

  if ((${#DOCKER_SERVICES[@]} > 0)); then
    generate_compose="yes"
  else
    if confirm "Generate docker-compose for LightRAG only?"; then
      generate_compose="yes"
    fi
  fi

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
        wait_for_services
      fi
      docker compose -f "$compose_file" up -d lightrag
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

  log_info "Interactive setup wizard"
  log_step "Step 1: Deployment type"
  echo "  - development: local JSON/NetworkX defaults"
  echo "  - production: database-backed defaults with security prompts"
  echo "  - custom: pick each backend manually"
  deployment_type="$(select_deployment_type)"
  DEPLOYMENT_TYPE="$deployment_type"

  case "$deployment_type" in
    development)
      load_preset "development"
      ;;
    production)
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

  collect_docker_image_tags

  log_step "Step 4: LLM configuration"
  collect_llm_config
  log_step "Step 5: Embedding configuration"
  collect_embedding_config
  log_step "Step 6: Reranking configuration"
  collect_rerank_config
  log_step "Step 7: Server configuration"
  collect_server_config
  log_step "Step 8: Security configuration"
  collect_security_config "no" "no"
  log_step "Step 9: Observability configuration"
  collect_observability_config

  finalize_setup
}

quick_start_flow() {
  local api_key

  reset_state
  load_preset "development"
  DEPLOYMENT_TYPE="development"

  log_info "Quick start setup"
  echo "Using development preset. Only OpenAI API key is required."

  ENV_VALUES["HOST"]="0.0.0.0"
  ENV_VALUES["PORT"]="9621"
  ENV_VALUES["WEBUI_TITLE"]="My Graph KB"
  ENV_VALUES["WEBUI_DESCRIPTION"]="Simple and Fast Graph Based RAG System"
  ENV_VALUES["RERANK_BINDING"]="null"

  api_key="$(prompt_secret_until_valid "OpenAI API key: " validate_api_key openai)"
  ENV_VALUES["LLM_BINDING_API_KEY"]="$api_key"
  ENV_VALUES["EMBEDDING_BINDING_API_KEY"]="$api_key"

  finalize_setup
}

production_flow() {
  local db_type
  local db_order=("postgresql" "neo4j" "mongodb" "redis" "milvus" "qdrant" "memgraph")

  reset_state
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

  collect_docker_image_tags

  log_step "Configuring LLM and embedding providers"
  collect_llm_config
  collect_embedding_config
  collect_rerank_config
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
      elif [[ "$value" =~ ^\'.*\'$ ]]; then
        value="${value:1:${#value}-2}"
      fi
      ENV_VALUES["$key"]="$value"
    fi
  done < "$env_file"
}

validate_env_file() {
  local env_file="${REPO_ROOT}/.env"
  local errors=0
  local kv vector graph doc_status

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

main "$@"
