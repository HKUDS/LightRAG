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
      [[ "$uri" =~ ^(neo4j(\+s|\+ssc)?|bolt)://.+ ]]
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
      return 0; ;;
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

validate_existing_file() {
  local path="$1"

  [[ -n "$path" && -f "$path" ]]
}

validate_mongodb_atlas_uri() {
  local uri="$1"

  [[ "$uri" =~ ^mongodb\+srv://.+ ]]
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

contains_env_interpolation_syntax() {
  local value="$1"

  [[ "$value" == *'${'* ]]
}

is_sensitive_env_key() {
  local key="$1"

  case "$key" in
    AUTH_ACCOUNTS|*API_KEY*|*ACCESS_KEY*|*PUBLIC_KEY*|*SECRET*|*PASSWORD*|*TOKEN*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

validate_sensitive_env_literals() {
  local key value
  local invalid_keys=()

  for key in "${!ENV_VALUES[@]}"; do
    if ! is_sensitive_env_key "$key"; then
      continue
    fi

    value="${ENV_VALUES[$key]:-}"
    if [[ -n "$value" ]] && contains_env_interpolation_syntax "$value"; then
      invalid_keys+=("$key")
    fi
  done

  if ((${#invalid_keys[@]} > 0)); then
    format_error \
      "Sensitive values must not contain \${...} interpolation syntax: ${invalid_keys[*]}" \
      "Use literal values, plain \$ characters, or inject those secrets via runtime environment variables instead of .env."
    return 1
  fi

  return 0
}

validate_required_variables() {
  local storages=("$@")
  local missing=()
  local unknown=()
  local storage required var

  for storage in "${storages[@]}"; do
    if [[ -z "$storage" ]]; then
      continue
    fi
    if [[ ! -v "STORAGE_ENV_REQUIREMENTS[$storage]" ]]; then
      unknown+=("$storage")
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

  if ((${#unknown[@]} > 0)); then
    format_error \
      "Unsupported storage selections: ${unknown[*]}" \
      "Use a supported LightRAG storage class name or rerun setup to pick a valid backend."
    return 1
  fi

  if ((${#missing[@]} > 0)); then
    format_error "Missing required variables: ${missing[*]}" "Fill them in .env or re-run setup."
    return 1
  fi

  return 0
}

validate_mongo_vector_storage_config() {
  local vector_storage="$1"
  local mongo_uri="${2:-${ENV_VALUES[MONGO_URI]:-}}"

  if [[ "$vector_storage" != "MongoVectorDBStorage" ]]; then
    return 0
  fi

  if ! validate_mongodb_atlas_uri "$mongo_uri"; then
    format_error \
      "MongoVectorDBStorage requires a MongoDB Atlas URI." \
      "Set MONGO_URI to a mongodb+srv:// Atlas connection string or choose another vector backend."
    return 1
  fi

  return 0
}

validate_auth_accounts_format() {
  local auth_accounts="$1"
  local entry username password

  if [[ -z "$auth_accounts" ]]; then
    return 0
  fi

  if [[ "$auth_accounts" == ,* || "$auth_accounts" == *, || "$auth_accounts" == *",,"* ]]; then
    return 1
  fi

  IFS=',' read -r -a entries <<< "$auth_accounts"
  for entry in "${entries[@]}"; do
    if [[ -z "$entry" || "$entry" != *:* ]]; then
      return 1
    fi

    username="${entry%%:*}"
    password="${entry#*:}"
    if [[ -z "$username" || -z "$password" ]]; then
      return 1
    fi
  done

  return 0
}

production_whitelist_exposes_api_routes() {
  local whitelist_paths="$1"
  local entry trimmed_entry normalized_entry

  IFS=',' read -r -a entries <<< "$whitelist_paths"
  for entry in "${entries[@]}"; do
    trimmed_entry="${entry#"${entry%%[![:space:]]*}"}"
    trimmed_entry="${trimmed_entry%"${trimmed_entry##*[![:space:]]}"}"
    normalized_entry="$trimmed_entry"

    if [[ "$normalized_entry" == *"/*" ]]; then
      normalized_entry="${normalized_entry%/*}"
    fi
    if [[ "$normalized_entry" != "/" ]]; then
      normalized_entry="${normalized_entry%/}"
    fi

    if [[ "$normalized_entry" == "/api" || "$normalized_entry" == /api/* ]]; then
      return 0
    fi
  done

  return 1
}

validate_security_config() {
  local auth_accounts="${1:-${ENV_VALUES[AUTH_ACCOUNTS]:-}}"
  local token_secret="${2:-${ENV_VALUES[TOKEN_SECRET]:-}}"
  local api_key="${3:-${ENV_VALUES[LIGHTRAG_API_KEY]:-}}"
  local require_protection="${4:-no}"
  local whitelist_paths="${5:-${ENV_VALUES[WHITELIST_PATHS]:-}}"
  local whitelist_is_set="${6:-}"
  local effective_whitelist="$whitelist_paths"

  if [[ "${6+x}" == "x" ]]; then
    if [[ -n "$whitelist_is_set" ]]; then
      whitelist_is_set="yes"
    else
      whitelist_is_set="no"
    fi
  elif [[ -z "$whitelist_is_set" ]]; then
    if [[ "${5+x}" == "x" || -v "ENV_VALUES[WHITELIST_PATHS]" ]]; then
      whitelist_is_set="yes"
    else
      whitelist_is_set="no"
    fi
  fi

  if [[ "$require_protection" == "yes" && -z "$auth_accounts" ]]; then
    format_error \
      "Production setup requires AUTH_ACCOUNTS." \
      "Configure account-based auth, optionally add LIGHTRAG_API_KEY on top, or switch to a non-production deployment."
    return 1
  fi

  if [[ -n "$auth_accounts" && "$whitelist_is_set" != "yes" ]]; then
    effective_whitelist="/health,/api/*"
  fi

  if [[ "$require_protection" == "yes" ]] && production_whitelist_exposes_api_routes "$effective_whitelist"; then
    format_error \
      "Production setup must not whitelist /api routes when authentication is enabled." \
      "Set WHITELIST_PATHS to a minimal list such as /health before using AUTH_ACCOUNTS."
    return 1
  fi

  if [[ -z "$auth_accounts" ]]; then
    return 0
  fi

  if ! validate_auth_accounts_format "$auth_accounts"; then
    format_error \
      "AUTH_ACCOUNTS must use comma-separated user:password pairs." \
      "Use entries like admin:secret or admin:secret,reader:another-secret."
    return 1
  fi

  if [[ -z "$token_secret" ]]; then
    format_error \
      "AUTH_ACCOUNTS is set but TOKEN_SECRET is missing." \
      "Set a non-empty JWT signing secret before enabling account-based authentication."
    return 1
  fi

  if [[ "$token_secret" == "lightrag-jwt-default-secret" ]]; then
    format_error \
      "TOKEN_SECRET must not use the built-in default value when AUTH_ACCOUNTS is enabled." \
      "Generate a unique JWT signing secret and update TOKEN_SECRET."
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
