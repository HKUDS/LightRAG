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

validate_positive_integer() {
  local value="$1"

  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  (( 10#$value > 0 ))
}

validate_non_negative_integer() {
  local value="$1"

  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  (( 10#$value >= 0 ))
}

validate_non_empty() {
  local value="$1"

  [[ -n "$value" ]]
}

validate_existing_file() {
  local path="$1"

  [[ -n "$path" && -f "$path" ]]
}

check_storage_compatibility() {
  local kv_storage="$1"
  local vector_storage="$2"
  local graph_storage="$3"
  local doc_status_storage="$4"
  local warnings=()

  if [[ "$vector_storage" == "MongoVectorDBStorage" ]]; then
    warnings+=("MongoDB vector storage requires an Atlas-capable deployment with Atlas Search / Vector Search support.")
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

validate_opensearch_hosts_format() {
  local hosts="${1:-${ENV_VALUES[OPENSEARCH_HOSTS]:-}}"
  local entry=""
  local trimmed=""
  local has_host="no"
  local -a entries=()

  if [[ "$hosts" == *"://"* ]]; then
    format_error \
      "OPENSEARCH_HOSTS must use bare host:port entries, not URLs." \
      "Set comma-separated host:port values such as localhost:9200; control TLS with OPENSEARCH_USE_SSL."
    return 1
  fi

  IFS=',' read -r -a entries <<< "$hosts"
  for entry in "${entries[@]}"; do
    trimmed="${entry#"${entry%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
    if [[ -z "$trimmed" ]]; then
      format_error \
        "OPENSEARCH_HOSTS must not contain empty host entries." \
        "Use comma-separated host:port values such as localhost:9200 or host1:9200,host2:9200."
      return 1
    fi
    has_host="yes"
  done

  if [[ "$has_host" != "yes" ]]; then
    format_error \
      "OPENSEARCH_HOSTS must include at least one host:port entry." \
      "Set it to a value such as localhost:9200."
    return 1
  fi

  return 0
}

validate_opensearch_password_strength() {
  local password="${1:-${ENV_VALUES[OPENSEARCH_PASSWORD]:-}}"

  if [[ ${#password} -lt 8 || ! "$password" =~ [A-Z] || ! "$password" =~ [a-z] || ! "$password" =~ [0-9] || ! "$password" =~ [^A-Za-z0-9] ]]; then
    format_error \
      "OpenSearch requires a strong OPENSEARCH_PASSWORD." \
      "Use at least 8 characters with uppercase, lowercase, number, and special character."
    return 1
  fi

  return 0
}

validate_opensearch_config() {
  local deployment_mode="${1:-${ENV_VALUES[LIGHTRAG_SETUP_OPENSEARCH_DEPLOYMENT]:-}}"
  local hosts="${2:-${ENV_VALUES[OPENSEARCH_HOSTS]:-}}"
  local user="${3:-${ENV_VALUES[OPENSEARCH_USER]:-}}"
  local password="${4:-${ENV_VALUES[OPENSEARCH_PASSWORD]:-}}"
  local num_shards="${5-${ENV_VALUES[OPENSEARCH_NUMBER_OF_SHARDS]-1}}"
  local num_replicas="${6-${ENV_VALUES[OPENSEARCH_NUMBER_OF_REPLICAS]-0}}"

  if ! validate_opensearch_hosts_format "$hosts"; then
    return 1
  fi

  if [[ -z "$user" || -z "$password" ]]; then
    if [[ "$deployment_mode" == "docker" ]]; then
      format_error \
        "Bundled OpenSearch requires OPENSEARCH_USER and OPENSEARCH_PASSWORD." \
        "Set both variables or rerun setup; the managed Docker service starts with security enabled."
    else
      format_error \
        "OpenSearch requires both OPENSEARCH_USER and OPENSEARCH_PASSWORD." \
        "This setup wizard only supports authenticated OpenSearch clusters. Set both values or rerun setup."
    fi
    return 1
  fi

  if ! validate_opensearch_password_strength "$password"; then
    if [[ "$deployment_mode" == "docker" ]]; then
      echo "${COLOR_YELLOW:-}Hint:${COLOR_RESET:-} The managed Docker image also enforces this password strength at startup." >&2
    fi
    return 1
  fi

  if ! validate_positive_integer "$num_shards"; then
    format_error \
      "OPENSEARCH_NUMBER_OF_SHARDS must be a positive integer." \
      "Set it to 1 or greater, or rerun setup to regenerate the OpenSearch index settings."
    return 1
  fi

  if ! validate_non_negative_integer "$num_replicas"; then
    format_error \
      "OPENSEARCH_NUMBER_OF_REPLICAS must be a non-negative integer." \
      "Set it to 0 or greater, or rerun setup to regenerate the OpenSearch index settings."
    return 1
  fi

  return 0
}

validate_mongo_vector_storage_config() {
  local vector_storage="$1"
  local mongo_uri="${2:-${ENV_VALUES[MONGO_URI]:-}}"
  local mongo_deployment="${3:-${ENV_VALUES[LIGHTRAG_SETUP_MONGODB_DEPLOYMENT]:-}}"

  if [[ "$vector_storage" != "MongoVectorDBStorage" ]]; then
    return 0
  fi

  if [[ "$mongo_deployment" == "docker" ]]; then
    format_error \
      "MongoVectorDBStorage cannot use the local Docker MongoDB service managed by this setup wizard." \
      "That service is MongoDB Community Edition without Atlas Search / Vector Search support. Use an Atlas-capable MongoDB endpoint instead."
    return 1
  fi

  if ! validate_uri "$mongo_uri" mongodb; then
    format_error \
      "MongoVectorDBStorage requires a valid MongoDB URI." \
      "Set MONGO_URI to a mongodb:// or mongodb+srv:// endpoint that supports Atlas Search / Vector Search."
    return 1
  fi

  if [[ ! "$mongo_uri" =~ ^mongodb\+srv:// ]]; then
    format_error \
      "MongoVectorDBStorage requires a MongoDB Atlas URI." \
      "Set MONGO_URI to a mongodb+srv:// endpoint backed by Atlas Search / Vector Search."
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

validate_auth_accounts_password_safety() {
  local auth_accounts="$1"
  local entry password normalized_password

  if [[ -z "$auth_accounts" ]]; then
    return 0
  fi

  IFS=',' read -r -a entries <<< "$auth_accounts"
  for entry in "${entries[@]}"; do
    password="${entry#*:}"
    normalized_password="${password,,}"
    if [[ "$normalized_password" == admin* || "$normalized_password" == pass* ]]; then
      return 1
    fi
  done

  return 0
}

whitelist_exposes_api_routes() {
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
  local _api_key="${3:-${ENV_VALUES[LIGHTRAG_API_KEY]:-}}"
  local _unused_flag="${4:-no}"
  local _unused_whitelist="${5:-${ENV_VALUES[WHITELIST_PATHS]:-}}"
  local _unused_whitelist_is_set="${6:-}"

  if [[ -z "$auth_accounts" ]]; then
    return 0
  fi

  if ! validate_auth_accounts_format "$auth_accounts"; then
    format_error \
      "AUTH_ACCOUNTS must use comma-separated user:password pairs." \
      "Use entries like admin:{bcrypt}<hash> or admin:secret,reader:another-secret."
    return 1
  fi

  if ! validate_auth_accounts_password_safety "$auth_accounts"; then
    format_error \
      "AUTH_ACCOUNTS passwords must not start with 'admin' or 'pass'." \
      "Choose a less predictable password or use lightrag-hash-password to generate a {bcrypt} value."
    return 1
  fi

  if [[ -z "$token_secret" ]]; then
    format_error \
      "AUTH_ACCOUNTS is set but TOKEN_SECRET is missing." \
      "Set a non-empty JWT signing secret before enabling account-based authentication."
    return 1
  fi

  if [[ "$token_secret" == "lightrag-jwt-default-secret-key!" ]]; then
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
