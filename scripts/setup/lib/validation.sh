# Validation helpers for interactive setup.

validate_uri() {
  local uri="$1"
  local db_type="$2"

  if [[ -z "$uri" ]]; then
    return 1
  fi

  case "$db_type" in
    postgresql)
      [[ "$uri" =~ ^postgres(ql)?://.+ ]]
      return $?; ;;
    neo4j)
      [[ "$uri" =~ ^(neo4j(\+s)?|bolt)://.+ ]]
      return $?; ;;
    mongodb)
      [[ "$uri" =~ ^mongodb(\+srv)?://.+ ]]
      return $?; ;;
    redis)
      [[ "$uri" =~ ^rediss?://.+ ]]
      return $?; ;;
    milvus|qdrant)
      [[ "$uri" =~ ^https?://.+ ]]
      return $?; ;;
    memgraph)
      [[ "$uri" =~ ^bolt://.+ ]]
      return $?; ;;
    *)
      return 1
      ;;
  esac
}

validate_api_key() {
  local key="$1"
  local provider="$2"

  if [[ -z "$key" ]]; then
    return 1
  fi

  case "$provider" in
    openai|openrouter)
      [[ "$key" == sk-* ]]
      return $?; ;;
    *)
      [[ ${#key} -ge 8 ]]
      return $?; ;;
  esac
}

validate_port() {
  local port="$1"

  if [[ ! "$port" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  if (( port < 1 || port > 65535 )); then
    return 1
  fi

  return 0
}

check_storage_compatibility() {
  local kv_storage="$1"
  local vector_storage="$2"
  local graph_storage="$3"
  local doc_status_storage="$4"
  local warnings=()

  if [[ "$vector_storage" == "MongoVectorDBStorage" ]]; then
    warnings+=("MongoDB vector storage requires Atlas (mongodb+srv:// URI).")
  fi

  if [[ "$graph_storage" == "Neo4JStorage" && "$kv_storage" == "JsonKVStorage" ]]; then
    warnings+=("Neo4j graph with JSON KV storage is fine for dev, but not ideal for production.")
  fi

  if [[ "$graph_storage" == "NetworkXStorage" ]]; then
    warnings+=("NetworkX graph storage is memory-bound and suited for small datasets only.")
  fi

  if [[ "$vector_storage" == "FaissVectorDBStorage" ]]; then
    warnings+=("Faiss vector storage is local-only and requires manual persistence management.")
  fi

  if [[ "$kv_storage" == "JsonKVStorage" || "$doc_status_storage" == "JsonDocStatusStorage" ]]; then
    warnings+=("JSON-based KV/doc status storage is recommended only for local development.")
  fi

  if ((${#warnings[@]} > 0)); then
    echo "${COLOR_YELLOW:-}Storage compatibility/performance warnings:${COLOR_RESET:-}" >&2
    for warning in "${warnings[@]}"; do
      echo "  - $warning" >&2
    done
    return 1
  fi

  return 0
}

format_error() {
  local message="$1"
  local suggestion="${2:-}"

  echo "${COLOR_RED:-}Error:${COLOR_RESET:-} $message" >&2
  if [[ -n "$suggestion" ]]; then
    echo "${COLOR_YELLOW:-}Hint:${COLOR_RESET:-} $suggestion" >&2
  fi
}

validate_required_variables() {
  local storages=("$@")
  local missing=()
  local storage required var

  for storage in "${storages[@]}"; do
    if [[ -z "$storage" ]]; then
      continue
    fi
    required="${STORAGE_ENV_REQUIREMENTS[$storage]}"
    if [[ -z "$required" ]]; then
      continue
    fi
    for var in $required; do
      if [[ -z "${ENV_VALUES[$var]:-}" ]]; then
        missing+=("$var")
      fi
    done
  done

  if ((${#missing[@]} > 0)); then
    format_error "Missing required variables: ${missing[*]}" "Fill them in .env or re-run setup."
    return 1
  fi

  return 0
}

check_docker_availability() {
  if ! command -v docker >/dev/null 2>&1; then
    format_error "Docker is not installed or not in PATH." "Install Docker or disable docker service generation."
    return 1
  fi

  if ! docker compose version >/dev/null 2>&1; then
    format_error "Docker Compose is not available." "Install the Docker Compose plugin or use docker-compose."
    return 1
  fi

  return 0
}
