# File operations for interactive setup.

# Registry of temp files created during this session; cleaned up on exit.
_FILE_OPS_CLEANUP_TMP=()
declare -A _FILE_OPS_VOLUME_BLOCKS=()
declare -a _FILE_OPS_VOLUME_ORDER=()
_WIZARD_MANAGED_SERVICES_MARKER="# __WIZARD_MANAGED_SERVICES__"
_file_ops_cleanup() {
  local f
  for f in "${_FILE_OPS_CLEANUP_TMP[@]:-}"; do
    rm -f "$f" 2>/dev/null || true
  done
}
trap '_file_ops_cleanup' EXIT INT TERM

format_env_value() {
  local value="$1"
  local escaped

  if [[ -z "$value" ]]; then
    printf ''
    return
  fi

  if [[ "$value" =~ [[:space:]] || "$value" == *"\""* || "$value" == *"$"* || "$value" == *"#"* ]]; then
    # Prefer single quotes when quoting is required so generated .env values
    # match env.example style and remain Compose-friendly. Fall back to
    # double quotes only when the value itself contains a single quote.
    if [[ "$value" != *"'"* ]]; then
      printf "'%s'" "$value"
      return
    fi

    # Double-quoted .env values only need escaping for backslash and double quote.
    # Do not escape '$': python-dotenv preserves plain '$' literally, while '\$'
    # changes the loaded value.
    # '#' in unquoted values is treated as a comment by python-dotenv, so any
    # value containing '#' must be quoted.
    escaped="${value//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    printf '"%s"' "$escaped"
    return
  fi

  printf '%s' "$value"
}

backup_env_file() {
  local env_file="${1:-${REPO_ROOT:-.}/.env}"
  local backup_file=""

  if [[ -f "$env_file" ]]; then
    backup_file="${env_file}.backup.$(date +%Y%m%d_%H%M%S)"
    if ! cp "$env_file" "$backup_file"; then
      format_error "Failed to back up ${env_file} to ${backup_file}." "Check disk space and file permissions."
      return 1
    fi
    printf '%s' "$backup_file"
  fi
}

backup_compose_file() {
  local compose_file="${1:-}"
  local repo_root="${REPO_ROOT:-.}"
  local backup_file=""

  if [[ -z "$compose_file" ]]; then
    compose_file="$(find_generated_compose_file)"
  fi

  if [[ -z "$compose_file" || ! -f "$compose_file" ]]; then
    return 0
  fi

  backup_file="${repo_root}/docker-compose.backup$(date +%Y%m%d_%H%M%S).yml"
  if ! cp "$compose_file" "$backup_file"; then
    format_error "Failed to back up ${compose_file} to ${backup_file}." \
      "Check disk space and file permissions."
    return 1
  fi

  printf '%s' "$backup_file"
}

stage_ssl_assets() {
  local cert_source="$1"
  local key_source="$2"
  local certs_dir="${REPO_ROOT:-.}/data/certs"
  local cert_target=""
  local key_target=""

  mkdir -p "$certs_dir"

  if [[ -n "$cert_source" ]]; then
    cert_target="${certs_dir}/$(resolve_staged_ssl_basename "cert" "$cert_source" "$key_source")"
    if [[ ! -e "$cert_target" || ! "$cert_source" -ef "$cert_target" ]]; then
      cp "$cert_source" "$cert_target"
    fi
  fi

  if [[ -n "$key_source" ]]; then
    key_target="${certs_dir}/$(resolve_staged_ssl_basename "key" "$key_source" "$cert_source")"
    if [[ ! -e "$key_target" || ! "$key_source" -ef "$key_target" ]]; then
      cp "$key_source" "$key_target"
    fi
  fi
}

stage_redis_config_asset() {
  local template_path="${TEMPLATES_DIR:-}/redis.conf.template"
  local config_dir="${REPO_ROOT:-.}/data/config"
  local config_target="${config_dir}/redis.conf"

  if [[ -z "$template_path" || ! -f "$template_path" ]]; then
    format_error "Missing Redis config template: ${template_path}" \
      "Restore scripts/setup/templates/redis.conf.template before rerunning setup."
    return 1
  fi

  mkdir -p "$config_dir"

  if [[ -e "$config_target" ]]; then
    log_info "Preserving existing Redis config at ${config_target}"
    return 0
  fi

  if ! cp "$template_path" "$config_target"; then
    format_error "Failed to stage Redis config at ${config_target}" \
      "Check file permissions and available disk space, then rerun setup."
    return 1
  fi

  log_success "Staged Redis config at ${config_target}"
}

resolve_staged_ssl_basename() {
  local asset_type="$1"
  local source_path="$2"
  local peer_path="${3:-}"
  local basename_value=""
  local peer_basename=""

  basename_value="$(basename "$source_path")"
  if [[ -n "$peer_path" ]]; then
    peer_basename="$(basename "$peer_path")"
    if [[ "$basename_value" == "$peer_basename" ]]; then
      printf '%s-%s' "$asset_type" "$basename_value"
      return 0
    fi
  fi

  printf '%s' "$basename_value"
}

append_preserved_non_template_env_lines() {
  local template_file="$1"
  local existing_env_file="$2"
  local output_file="$3"
  local line key
  local in_preserved_section="no"
  local line_is_commented_env="no"
  local preserved_header="### Preserved custom environment variables from previous .env"
  local preserved_notice="### Comments in this session will persist across regenerations"
  local -a pending_lines=()
  local -a preserved_payload=()
  local -a discovered_payload=()
  local -A ignored_keys=(
    ["LIGHTRAG_SETUP_PROFILE"]=1
  )
  local -A template_keys=()

  if [[ ! -f "$existing_env_file" ]]; then
    return 0
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[A-Za-z0-9_]+= ]]; then
      template_keys["${line%%=*}"]=1
    elif [[ "$line" =~ ^#[[:space:]]*([A-Za-z0-9_]+)=(.*)$ ]]; then
      template_keys["${BASH_REMATCH[1]}"]=1
    fi
  done < "$template_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "$preserved_header" ]]; then
      in_preserved_section="yes"
      pending_lines=()
      continue
    fi

    if [[ "$in_preserved_section" == "yes" && "$line" == "$preserved_notice" ]]; then
      continue
    fi

    key=""
    line_is_commented_env="no"

    if [[ "$line" =~ ^([A-Za-z0-9_]+)= ]]; then
      key="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^#[[:space:]]*([A-Za-z0-9_]+)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      line_is_commented_env="yes"
    fi

    if [[ -z "$key" ]]; then
      if [[ "$in_preserved_section" == "yes" ]]; then
        pending_lines+=("$line")
      fi
      continue
    fi

    if [[ -n "${ignored_keys[$key]+set}" ]]; then
      pending_lines=()
      continue
    fi

    if [[ -z "${template_keys[$key]+set}" || \
      ("$in_preserved_section" == "yes" && "$line_is_commented_env" == "yes") ]]; then
      if ((${#pending_lines[@]} > 0)); then
        if [[ "$in_preserved_section" == "yes" ]]; then
          preserved_payload+=("${pending_lines[@]}")
        fi
      fi

      if [[ "$in_preserved_section" == "yes" ]]; then
        preserved_payload+=("$line")
      else
        discovered_payload+=("$line")
      fi
    fi

    pending_lines=()
  done < "$existing_env_file"

  if ((${#pending_lines[@]} > 0)); then
    preserved_payload+=("${pending_lines[@]}")
  fi

  if ((${#preserved_payload[@]} == 0 && ${#discovered_payload[@]} == 0)); then
    return 0
  fi

  printf '\n%s\n%s\n' "$preserved_header" "$preserved_notice" >> "$output_file"

  if ((${#preserved_payload[@]} > 0)); then
    printf '%s\n' "${preserved_payload[@]}" >> "$output_file"
  fi

  if ((${#discovered_payload[@]} > 0)); then
    printf '%s\n' "${discovered_payload[@]}" >> "$output_file"
  fi
}

generate_env_file() {
  local template_file="${1:-${REPO_ROOT:-.}/env.example}"
  local output_file="${2:-${REPO_ROOT:-.}/.env}"
  local tmp_file="${output_file}.tmp"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line key value
  local -A written_keys=()
  local -A match_write_keys=()

  if [[ ! -f "$template_file" ]]; then
    echo "env.example not found at $template_file" >&2
    return 1
  fi

  # Pre-scan: identify commented keys whose value exactly matches the ENV_VALUE.
  # When a match exists, the active value is written only at that matching line,
  # leaving all other commented examples intact.
  local _prescan_key _prescan_val _prescan_env_val _prescan_fmt
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^#[[:space:]]*([A-Za-z0-9_]+)=(.*)$ ]]; then
      _prescan_key="${BASH_REMATCH[1]}"
      _prescan_val="${BASH_REMATCH[2]}"
      if [[ -z "${match_write_keys[$_prescan_key]+set}" && -n "${ENV_VALUES[$_prescan_key]+set}" ]]; then
        _prescan_env_val="${ENV_VALUES[$_prescan_key]}"
        _prescan_fmt="$(format_env_value "$_prescan_env_val" "$_prescan_key")"
        if [[ "$_prescan_val" == "$_prescan_env_val" || "$_prescan_val" == "$_prescan_fmt" ]]; then
          match_write_keys["$_prescan_key"]=1
        fi
      fi
    fi
  done < "$template_file"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[A-Za-z0-9_]+= ]]; then
      key="${line%%=*}"
      if [[ -z "${written_keys[$key]+set}" ]]; then
        if [[ -n "${ENV_VALUES[$key]+set}" ]]; then
          value="${ENV_VALUES[$key]}"
          local _fmt_active_val
          _fmt_active_val="$(format_env_value "$value" "$key")"
          printf '%s=%s\n' "$key" "$_fmt_active_val" >> "$tmp_file"
          local _orig_tmpl_val="${line#*=}"
          if [[ "$_orig_tmpl_val" != "$value" && "$_orig_tmpl_val" != "$_fmt_active_val" ]]; then
            printf '# %s\n' "$line" >> "$tmp_file"
          fi
        else
          printf '%s\n' "$line" >> "$tmp_file"
        fi
        written_keys["$key"]=1
      else
        if [[ -n "${ENV_VALUES[$key]+set}" ]]; then
          printf '# %s\n' "$line" >> "$tmp_file"
        else
          printf '%s\n' "$line" >> "$tmp_file"
        fi
      fi
    elif [[ "$line" =~ ^#[[:space:]]*([A-Za-z0-9_]+)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      local _commented_val="${BASH_REMATCH[2]}"
      if [[ -z "${written_keys[$key]+set}" && -n "${ENV_VALUES[$key]+set}" ]]; then
        value="${ENV_VALUES[$key]}"
        if [[ -n "${match_write_keys[$key]+set}" ]]; then
          # A commented line matching the ENV value exists; only activate at that line.
          local _fmt_val
          _fmt_val="$(format_env_value "$value" "$key")"
          if [[ "$_commented_val" == "$value" || "$_commented_val" == "$_fmt_val" ]]; then
            printf '%s=%s\n' "$key" "$_fmt_val" >> "$tmp_file"
            written_keys["$key"]=1
          else
            printf '%s\n' "$line" >> "$tmp_file"
          fi
        else
          # No matching commented line; fall back to activating at first occurrence.
          printf '%s=%s\n' "$key" "$(format_env_value "$value" "$key")" >> "$tmp_file"
          written_keys["$key"]=1
        fi
      else
        printf '%s\n' "$line" >> "$tmp_file"
      fi
    else
      printf '%s\n' "$line" >> "$tmp_file"
    fi
  done < "$template_file"

  append_preserved_non_template_env_lines "$template_file" "$output_file" "$tmp_file"

  mv "$tmp_file" "$output_file"
}

# All environment keys the wizard may inject into the lightrag service via
# COMPOSE_ENV_OVERRIDES.  Used to remove stale entries before re-injection so
# keys no longer needed are not left behind in the compose file.
_WIZARD_COMPOSE_LIGHTRAG_KEYS=(
  "EMBEDDING_BINDING_HOST" "RERANK_BINDING_HOST" "LLM_BINDING_HOST"
  "REDIS_URI" "MONGO_URI" "NEO4J_URI" "MILVUS_URI" "QDRANT_URL" "MEMGRAPH_URI" "OPENSEARCH_HOSTS"
  "POSTGRES_HOST" "POSTGRES_PORT" "PORT" "HOST" "SSL_CERTFILE" "SSL_KEYFILE"
  "WORKING_DIR" "INPUT_DIR"
)

_managed_service_root_name() {
  local service_name="$1"

  case "$service_name" in
    postgres|neo4j|mongodb|redis|qdrant|memgraph|opensearch|vllm-embed|vllm-rerank)
      printf '%s' "$service_name"
      ;;
    milvus|milvus-etcd|milvus-minio)
      printf 'milvus'
      ;;
    *)
      printf ''
      ;;
  esac
}

_managed_volume_root_name() {
  local volume_name="$1"

  case "$volume_name" in
    postgres_data)
      printf 'postgres'
      ;;
    neo4j_data)
      printf 'neo4j'
      ;;
    mongo_data)
      printf 'mongodb'
      ;;
    redis_data)
      printf 'redis'
      ;;
    milvus_data|milvus-etcd_data|milvus-minio_data)
      printf 'milvus'
      ;;
    qdrant_data)
      printf 'qdrant'
      ;;
    memgraph_data)
      printf 'memgraph'
      ;;
    opensearch_data)
      printf 'opensearch'
      ;;
    vllm_rerank_cache)
      printf 'vllm-rerank'
      ;;
    vllm_embed_cache)
      printf 'vllm-embed'
      ;;
    *)
      printf ''
      ;;
  esac
}

_should_rewrite_wizard_managed_root_service() {
  local root_service="$1"

  if [[ -z "$root_service" ]]; then
    return 1
  fi

  if [[ "${FORCE_REWRITE_COMPOSE:-no}" == "yes" ]]; then
    return 0
  fi

  if [[ -z "${DOCKER_SERVICE_SET[$root_service]+set}" ]]; then
    return 0
  fi

  if [[ -n "${COMPOSE_REWRITE_SERVICE_SET[$root_service]+set}" ]]; then
    return 0
  fi

  return 1
}

_should_preserve_wizard_managed_root_service() {
  local root_service="$1"

  if [[ -z "$root_service" || "${FORCE_REWRITE_COMPOSE:-no}" == "yes" ]]; then
    return 1
  fi

  if [[ -z "${DOCKER_SERVICE_SET[$root_service]+set}" ]]; then
    return 1
  fi

  if [[ -n "${COMPOSE_REWRITE_SERVICE_SET[$root_service]+set}" ]]; then
    return 1
  fi

  return 0
}

_existing_managed_root_service_present() {
  local root_service="$1"

  [[ -n "$root_service" && -n "${EXISTING_MANAGED_ROOT_SERVICE_SET[$root_service]+set}" ]]
}

_refresh_existing_managed_root_service_set_from_compose() {
  local compose_file="$1"
  local service_name

  EXISTING_MANAGED_ROOT_SERVICE_SET=()

  if [[ -z "$compose_file" || ! -f "$compose_file" ]]; then
    return 0
  fi

  while IFS= read -r service_name; do
    EXISTING_MANAGED_ROOT_SERVICE_SET["$service_name"]=1
  done < <(detect_managed_root_services "$compose_file")
}

_is_wizard_managed_root_service_name() {
  local service_name="$1"

  [[ -n "$(_managed_service_root_name "$service_name")" ]]
}

_is_wizard_managed_service_name() {
  local service_name="$1"

  [[ -n "$(_managed_service_root_name "$service_name")" ]]
}

_is_wizard_managed_volume_name() {
  local volume_name="$1"

  [[ -n "$(_managed_volume_root_name "$volume_name")" ]]
}

# Remove wizard-managed keys from the lightrag service's environment block,
# leaving any user-added keys intact.
_strip_lightrag_wizard_environment_keys() {
  local compose_file="$1"
  local tmp_file="${compose_file}.strip-wizard-keys"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line key wk list_entry
  local in_lightrag="no"
  local in_environment="no"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      in_environment="no"
    elif [[ "$in_lightrag" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  lightrag:" ]]; then
      in_lightrag="no"
      in_environment="no"
    fi

    if [[ "$in_lightrag" == "yes" && "$line" == "    environment:" ]]; then
      in_environment="yes"
      printf '%s\n' "$line" >> "$tmp_file"
      continue
    fi

    if [[ "$in_lightrag" == "yes" && "$in_environment" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{6}([A-Z0-9_]+): ]]; then
        key="${BASH_REMATCH[1]}"
        for wk in "${_WIZARD_COMPOSE_LIGHTRAG_KEYS[@]}"; do
          if [[ "$key" == "$wk" ]]; then
            continue 2  # skip this wizard-managed key
          fi
        done
      elif [[ "$line" =~ ^[[:space:]]{6}-[[:space:]](.+)$ ]]; then
        list_entry="$(_strip_wrapping_quotes "${BASH_REMATCH[1]}")"
        key="${list_entry%%=*}"
        if [[ "$key" =~ ^[A-Z0-9_]+$ ]]; then
          for wk in "${_WIZARD_COMPOSE_LIGHTRAG_KEYS[@]}"; do
            if [[ "$key" == "$wk" ]]; then
              continue 2  # skip this wizard-managed key
            fi
          done
        fi
      elif [[ -z "$line" ]]; then
        continue  # skip blank lines inside the environment block
      elif [[ ! "$line" =~ ^[[:space:]]{6} ]]; then
        in_environment="no"
      fi
    fi

    printf '%s\n' "$line" >> "$tmp_file"
  done < "$compose_file"

  mv "$tmp_file" "$compose_file"
}

_write_service_environment_entries() {
  local tmp_file="$1"
  local style="$2"
  shift 2
  local entries=("$@")
  local key value

  for entry in "${entries[@]}"; do
    key="${entry%%=*}"
    value="${entry#*=}"
    if [[ "$style" == "list" ]]; then
      if [[ "$value" == "${_COMPOSE_RAW_VALUE_PREFIX}"* ]]; then
        printf '      - %s=%s\n' "$key" "${value#${_COMPOSE_RAW_VALUE_PREFIX}}" >> "$tmp_file"
      else
        printf '      - %s\n' "$(format_yaml_value "${key}=${value}")" >> "$tmp_file"
      fi
    else
      printf '      %s: %s\n' "$key" "$(format_compose_environment_value "$value")" >> "$tmp_file"
    fi
  done
}

# Capture top-level named volume blocks so user-managed definitions can be
# re-emitted when still referenced by preserved services.
_collect_top_level_volume_blocks() {
  local compose_file="$1"
  local line
  local in_top_volumes="no"
  local current_volume=""
  local current_block=""

  _FILE_OPS_VOLUME_BLOCKS=()
  _FILE_OPS_VOLUME_ORDER=()

  if [[ ! -f "$compose_file" ]]; then
    return 0
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[A-Za-z] ]]; then
      if [[ "$in_top_volumes" == "yes" && -n "$current_volume" ]]; then
        _FILE_OPS_VOLUME_BLOCKS["$current_volume"]="$current_block"
        _FILE_OPS_VOLUME_ORDER+=("$current_volume")
      fi

      current_volume=""
      current_block=""
      if [[ "$line" == "volumes:" ]]; then
        in_top_volumes="yes"
      else
        in_top_volumes="no"
      fi
      continue
    fi

    if [[ "$in_top_volumes" != "yes" ]]; then
      continue
    fi

    if [[ "$line" =~ ^[[:space:]]{2}([A-Za-z0-9_.-]+):[[:space:]]*$ ]]; then
      if [[ -n "$current_volume" ]]; then
        _FILE_OPS_VOLUME_BLOCKS["$current_volume"]="$current_block"
        _FILE_OPS_VOLUME_ORDER+=("$current_volume")
      fi

      current_volume="${BASH_REMATCH[1]}"
      current_block="${line}"$'\n'
      continue
    fi

    if [[ -n "$current_volume" ]]; then
      current_block+="${line}"$'\n'
    fi
  done < "$compose_file"

  if [[ "$in_top_volumes" == "yes" && -n "$current_volume" ]]; then
    _FILE_OPS_VOLUME_BLOCKS["$current_volume"]="$current_block"
    _FILE_OPS_VOLUME_ORDER+=("$current_volume")
  fi
}

# Remove wizard-managed services and the top-level volumes block from a compose
# file. Non-managed services are preserved verbatim.
_strip_wizard_managed_services_and_top_level_volumes() {
  local compose_file="$1"
  local tmp_file="${compose_file}.strip-svc"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line current_service="" current_root_service=""
  local in_services="no"
  local in_top_volumes="no"
  local inserted_marker="no"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "$_WIZARD_MANAGED_SERVICES_MARKER" ]]; then
      continue
    fi

    # Detect top-level (non-indented) keys.
    if [[ "$line" =~ ^[A-Za-z] ]]; then
      if [[ "$in_services" == "yes" && "$line" != "services:" && "$inserted_marker" != "yes" ]]; then
        printf '%s\n' "$_WIZARD_MANAGED_SERVICES_MARKER" >> "$tmp_file"
        inserted_marker="yes"
      fi
      in_top_volumes="no"
      if [[ "$line" == "services:" ]]; then
        in_services="yes"
        current_service=""
      elif [[ "$line" =~ ^volumes:[[:space:]]*$ ]]; then
        in_top_volumes="yes"
        in_services="no"
        current_service=""
        continue  # skip volumes: header; regenerated at end of generate_docker_compose
      else
        in_services="no"
        current_service=""
      fi
      printf '%s\n' "$line" >> "$tmp_file"
      continue
    fi

    # Skip top-level volumes block content.
    if [[ "$in_top_volumes" == "yes" ]]; then
      continue
    fi

    # Track current service inside the services: block.
    if [[ "$in_services" == "yes" && "$line" =~ ^[[:space:]]{2}([A-Za-z0-9_-]+):[[:space:]]*$ ]]; then
      current_service="${BASH_REMATCH[1]}"
      current_root_service="$(_managed_service_root_name "$current_service")"
    fi

    # Skip managed services that are being removed or regenerated. Preserve
    # lightrag, user-added services, and unchanged managed service groups.
    if [[ "$in_services" == "yes" && -n "$current_service" ]] && \
      [[ "$current_service" != "lightrag" ]] && \
      [[ -n "$current_root_service" ]] && \
      _should_rewrite_wizard_managed_root_service "$current_root_service"; then
      continue
    fi

    printf '%s\n' "$line" >> "$tmp_file"
  done < "$compose_file"

  if [[ "$in_services" == "yes" && "$inserted_marker" != "yes" ]]; then
    printf '%s\n' "$_WIZARD_MANAGED_SERVICES_MARKER" >> "$tmp_file"
  fi

  mv "$tmp_file" "$compose_file"
}

_merge_managed_service_blocks() {
  local compose_file="$1"
  local service_blocks_file="$2"
  local tmp_file="${compose_file}.merge-services"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local inserted="no"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "$_WIZARD_MANAGED_SERVICES_MARKER" ]]; then
      if [[ -s "$service_blocks_file" ]]; then
        cat "$service_blocks_file" >> "$tmp_file"
        inserted="yes"
      fi
      continue
    fi

    printf '%s\n' "$line" >> "$tmp_file"
  done < "$compose_file"

  if [[ -s "$service_blocks_file" && "$inserted" != "yes" ]]; then
    cat "$service_blocks_file" >> "$tmp_file"
  fi

  mv "$tmp_file" "$compose_file"
}

_normalize_services_section_spacing() {
  local compose_file="$1"
  local tmp_file="${compose_file}.normalize-services"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local in_services="no"
  local pending_blank="no"
  local saw_service_content="no"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_services" == "yes" ]]; then
      if [[ "$line" == "$_WIZARD_MANAGED_SERVICES_MARKER" ]]; then
        continue
      fi

      if [[ -z "$line" ]]; then
        pending_blank="yes"
        continue
      fi

      if [[ ! "$line" =~ ^[[:space:]] ]]; then
        pending_blank="no"
        if [[ "$saw_service_content" == "yes" ]]; then
          printf '\n' >> "$tmp_file"
        fi
        printf '%s\n' "$line" >> "$tmp_file"
        in_services="no"
        continue
      fi

      if [[ "$pending_blank" == "yes" && "$saw_service_content" == "yes" ]]; then
        printf '\n' >> "$tmp_file"
      fi

      printf '%s\n' "$line" >> "$tmp_file"
      pending_blank="no"
      saw_service_content="yes"
      continue
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "services:" ]]; then
      in_services="yes"
      pending_blank="no"
      saw_service_content="no"
    fi
  done < "$compose_file"

  mv "$tmp_file" "$compose_file"
}

_strip_wrapping_quotes() {
  local value="$1"

  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value#\"}"
    value="${value%\"}"
  elif [[ "$value" == \'*\' ]]; then
    value="${value#\'}"
    value="${value%\'}"
  fi

  printf '%s' "$value"
}

read_service_environment_value() {
  local compose_file="$1"
  local service_name="$2"
  local wanted_key="$3"
  local line
  local entry_key=""
  local entry_value=""
  local service_header="  ${service_name}:"
  local in_service="no"
  local in_environment="no"

  if [[ ! -f "$compose_file" ]]; then
    return 1
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_service" == "yes" && "$in_environment" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{6}([A-Z0-9_]+):[[:space:]]*(.+)$ ]]; then
        entry_key="${BASH_REMATCH[1]}"
        if [[ "$entry_key" == "$wanted_key" ]]; then
          printf '%s' "$(_strip_wrapping_quotes "${BASH_REMATCH[2]}")"
          return 0
        fi
      elif [[ "$line" =~ ^[[:space:]]{6}-[[:space:]](.+)$ ]]; then
        entry_value="$(_strip_wrapping_quotes "${BASH_REMATCH[1]}")"
        entry_key="${entry_value%%=*}"
        if [[ "$entry_key" == "$wanted_key" && "$entry_value" == *=* ]]; then
          printf '%s' "${entry_value#*=}"
          return 0
        fi
      elif [[ ! "$line" =~ ^[[:space:]]{6} ]]; then
        in_environment="no"
      fi
    elif [[ "$in_service" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "$service_header" ]]; then
      in_service="no"
    fi

    if [[ "$line" == "$service_header" ]]; then
      in_service="yes"
      in_environment="no"
    elif [[ "$in_service" == "yes" && "$line" == "    environment:" ]]; then
      in_environment="yes"
    fi
  done < "$compose_file"

  return 1
}

_extract_named_volume_name() {
  local mount_spec="$1"
  local source=""

  mount_spec="$(_strip_wrapping_quotes "$mount_spec")"
  source="${mount_spec%%:*}"

  if [[ "$source" == "$mount_spec" || -z "$source" ]]; then
    return 1
  fi

  if [[ "$source" == .* || "$source" == /* || "$source" == "~"* ]]; then
    return 1
  fi

  if [[ "$source" == *"/"* || "$source" == *'$'* ]]; then
    return 1
  fi

  printf '%s' "$source"
}

_collect_referenced_named_volumes() {
  local compose_file="$1"
  local line
  local in_services="no"
  local current_service=""
  local in_volumes="no"
  local in_long_volume_entry="no"
  local long_volume_type=""
  local long_volume_source=""
  local volume_name=""
  local long_entry_key=""
  local long_entry_value=""
  local -A seen=()

  if [[ ! -f "$compose_file" ]]; then
    return 0
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "services:" ]]; then
      in_services="yes"
      current_service=""
      in_volumes="no"
      in_long_volume_entry="no"
      long_volume_type=""
      continue
    fi

    if [[ "$in_services" == "yes" && "$line" =~ ^[A-Za-z] && "$line" != "services:" ]]; then
      in_services="no"
      in_volumes="no"
      in_long_volume_entry="no"
      continue
    fi

    if [[ "$in_services" != "yes" ]]; then
      continue
    fi

    if [[ "$line" =~ ^[[:space:]]{2}([A-Za-z0-9_-]+):[[:space:]]*$ ]]; then
      current_service="${BASH_REMATCH[1]}"
      in_volumes="no"
      in_long_volume_entry="no"
      long_volume_type=""
      long_volume_source=""
      continue
    fi

    if [[ -z "$current_service" ]]; then
      continue
    fi

    if [[ "$line" == "    volumes:" ]]; then
      in_volumes="yes"
      in_long_volume_entry="no"
      long_volume_type=""
      long_volume_source=""
      continue
    fi

    if [[ "$in_volumes" != "yes" ]]; then
      continue
    fi

    if [[ "$line" =~ ^[[:space:]]{4}[^[:space:]-] || "$line" =~ ^[[:space:]]{2}[^[:space:]] ]]; then
      in_volumes="no"
      in_long_volume_entry="no"
      long_volume_type=""
      long_volume_source=""
      continue
    fi

    if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]](.+)$ ]]; then
      local volume_entry="${BASH_REMATCH[1]}"
      volume_entry="$(_strip_wrapping_quotes "$volume_entry")"
      in_long_volume_entry="no"
      long_volume_type=""
      long_volume_source=""

      if [[ "$volume_entry" =~ ^([A-Za-z_][A-Za-z0-9_-]*):[[:space:]]*(.*)$ ]]; then
        long_entry_key="${BASH_REMATCH[1]}"
        long_entry_value="$(_strip_wrapping_quotes "${BASH_REMATCH[2]}")"
        case "$long_entry_key" in
          type|source|target|read_only|bind|volume|tmpfs|consistency|nocopy|subpath)
            in_long_volume_entry="yes"
            case "$long_entry_key" in
              type)
                if [[ "$long_entry_value" == "volume" ]]; then
                  long_volume_type="volume"
                else
                  long_volume_type="other"
                fi
                ;;
              source)
                long_volume_source="$long_entry_value"
                ;;
            esac
            if [[ "$long_volume_type" == "volume" && -n "$long_volume_source" && -z "${seen[$long_volume_source]+set}" ]]; then
              seen["$long_volume_source"]=1
              printf '%s\n' "$long_volume_source"
            fi
            continue
            ;;
        esac
      fi

      volume_name="$(_extract_named_volume_name "$volume_entry")" || continue
      if [[ -z "${seen[$volume_name]+set}" ]]; then
        seen["$volume_name"]=1
        printf '%s\n' "$volume_name"
      fi
      continue
    fi

    if [[ "$in_long_volume_entry" == "yes" ]] && \
      [[ "$line" =~ ^[[:space:]]{8}([A-Za-z_][A-Za-z0-9_-]*):[[:space:]]*(.+)$ ]]; then
      long_entry_key="${BASH_REMATCH[1]}"
      long_entry_value="$(_strip_wrapping_quotes "${BASH_REMATCH[2]}")"
      case "$long_entry_key" in
        type)
          if [[ "$long_entry_value" == "volume" ]]; then
            long_volume_type="volume"
          else
            long_volume_type="other"
          fi
          ;;
        source)
          long_volume_source="$long_entry_value"
          ;;
      esac

      if [[ "$long_volume_type" == "volume" && -n "$long_volume_source" && -z "${seen[$long_volume_source]+set}" ]]; then
        seen["$long_volume_source"]=1
        printf '%s\n' "$long_volume_source"
      fi
    fi
  done < "$compose_file"
}

_trim_trailing_blank_lines_in_file() {
  local file="$1"
  local trim_file="${file}.trim-tail"
  _FILE_OPS_CLEANUP_TMP+=("$trim_file")

  awk '
    { lines[NR] = $0 }
    END {
      last = NR
      while (last > 0 && lines[last] == "") {
        last--
      }
      for (i = 1; i <= last; i++) {
        print lines[i]
      }
    }
  ' "$file" > "$trim_file"

  mv "$trim_file" "$file"
}

_append_referenced_volume_blocks() {
  local compose_file="$1"
  local -a referenced_volumes=()
  local volume_name
  local root_service

  while IFS= read -r volume_name; do
    if [[ -n "$volume_name" ]]; then
      referenced_volumes+=("$volume_name")
    fi
  done < <(_collect_referenced_named_volumes "$compose_file")

  if ((${#referenced_volumes[@]} == 0)); then
    return 0
  fi

  _trim_trailing_blank_lines_in_file "$compose_file"
  printf '\nvolumes:\n' >> "$compose_file"
  for volume_name in "${referenced_volumes[@]}"; do
    if _is_wizard_managed_volume_name "$volume_name"; then
      root_service="$(_managed_volume_root_name "$volume_name")"
      if [[ -n "${_FILE_OPS_VOLUME_BLOCKS[$volume_name]+set}" ]] && \
        _should_preserve_wizard_managed_root_service "$root_service"; then
        printf '%s' "${_FILE_OPS_VOLUME_BLOCKS[$volume_name]}" >> "$compose_file"
      else
        printf '  %s:\n' "$volume_name" >> "$compose_file"
      fi
    elif [[ -n "${_FILE_OPS_VOLUME_BLOCKS[$volume_name]+set}" ]]; then
      printf '%s' "${_FILE_OPS_VOLUME_BLOCKS[$volume_name]}" >> "$compose_file"
    else
      printf '  %s:\n' "$volume_name" >> "$compose_file"
    fi
  done
}

generate_docker_compose() {
  local output_file="${1:-${REPO_ROOT:-.}/docker-compose.yml}"
  local base_file="${REPO_ROOT:-.}/docker-compose.yml"
  local tmp_file="${output_file}.tmp"
  local service_blocks_file="${output_file}.services"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  _FILE_OPS_CLEANUP_TMP+=("$service_blocks_file")
  local template_file
  local lightrag_mounts=()
  local lightrag_env_entries=()
  local key
  local root_service

  # Prefer the existing generated compose as the starting point to preserve
  # any user customisations to the lightrag service.  Fall back to the base
  # docker-compose.yml when the output file doesn't exist yet.
  if [[ -f "$output_file" && "$output_file" != "$base_file" ]]; then
    _refresh_existing_managed_root_service_set_from_compose "$output_file"
    _collect_top_level_volume_blocks "$output_file"
    cp "$output_file" "$tmp_file"
    # Strip wizard-managed services and top-level volumes. User-managed
    # services are preserved, while volumes are rebuilt from final service
    # references after managed templates are appended.
    _strip_wizard_managed_services_and_top_level_volumes "$tmp_file"
  elif [[ -f "$base_file" ]]; then
    _refresh_existing_managed_root_service_set_from_compose "$base_file"
    _collect_top_level_volume_blocks "$base_file"
    cp "$base_file" "$tmp_file"
    _strip_wizard_managed_services_and_top_level_volumes "$tmp_file"
  else
    EXISTING_MANAGED_ROOT_SERVICE_SET=()
    _FILE_OPS_VOLUME_BLOCKS=()
    _FILE_OPS_VOLUME_ORDER=()
    printf 'services:\n' > "$tmp_file"
  fi

  prepare_lightrag_service_for_generated_compose "$tmp_file"
  normalize_lightrag_restart_policy "$tmp_file"
  # Remove stale wizard-managed keys from lightrag's environment so that
  # keys no longer in COMPOSE_ENV_OVERRIDES are not left behind.
  _strip_lightrag_wizard_environment_keys "$tmp_file"

  # Remove stale wizard-managed bind mounts from lightrag's volumes so that
  # mounts no longer needed (e.g. after SSL removal) are not left behind.
  _strip_lightrag_wizard_bind_mounts "$tmp_file"

  if [[ -n "${LIGHTRAG_COMPOSE_SERVER_PORT_MAPPING:-}" ]]; then
    _strip_lightrag_wizard_ports "$tmp_file"
    inject_lightrag_port_mapping "$tmp_file" "$LIGHTRAG_COMPOSE_SERVER_PORT_MAPPING"
  fi

  append_lightrag_ssl_mount lightrag_mounts "${COMPOSE_ENV_OVERRIDES[SSL_CERTFILE]:-}" || return 1
  append_lightrag_ssl_mount lightrag_mounts "${COMPOSE_ENV_OVERRIDES[SSL_KEYFILE]:-}" || return 1
  if ((${#lightrag_mounts[@]} > 0)); then
    inject_lightrag_bind_mounts "$tmp_file" "${lightrag_mounts[@]}"
  fi

  for key in "${!COMPOSE_ENV_OVERRIDES[@]}"; do
    lightrag_env_entries+=("${key}=${COMPOSE_ENV_OVERRIDES[$key]}")
  done
  if ((${#lightrag_env_entries[@]} > 0)); then
    inject_lightrag_environment_overrides "$tmp_file" "${lightrag_env_entries[@]}"
  fi

  repair_misplaced_lightrag_depends_on "$tmp_file"
  inject_lightrag_depends_on "$tmp_file" "${DOCKER_SERVICES[@]}"

  : > "$service_blocks_file"
  for service in "${DOCKER_SERVICES[@]}"; do
    root_service="$(_managed_service_root_name "$service")"
    if _should_preserve_wizard_managed_root_service "$root_service" && \
      _existing_managed_root_service_present "$root_service"; then
      continue
    fi

    template_file="$TEMPLATES_DIR/${service}.yml"
    if [[ "$service" == "milvus" ]]; then
      if [[ "${ENV_VALUES[MILVUS_DEVICE]:-cpu}" == "cuda" ]]; then
        if [[ -f "$TEMPLATES_DIR/${service}-gpu.yml" ]]; then
          template_file="$TEMPLATES_DIR/${service}-gpu.yml"
        fi
      fi
    fi
    if [[ "$service" == "qdrant" ]]; then
      if [[ "${ENV_VALUES[QDRANT_DEVICE]:-cpu}" == "cuda" ]]; then
        if [[ -f "$TEMPLATES_DIR/${service}-gpu.yml" ]]; then
          template_file="$TEMPLATES_DIR/${service}-gpu.yml"
        fi
      fi
    fi
    if [[ "$service" == "vllm-rerank" ]]; then
      if [[ "${ENV_VALUES[VLLM_RERANK_DEVICE]:-cpu}" == "cuda" ]]; then
        if [[ -f "$TEMPLATES_DIR/${service}-gpu.yml" ]]; then
          template_file="$TEMPLATES_DIR/${service}-gpu.yml"
        fi
      fi
    fi
    if [[ "$service" == "vllm-embed" ]]; then
      if [[ "${ENV_VALUES[VLLM_EMBED_DEVICE]:-cpu}" == "cuda" ]]; then
        if [[ -f "$TEMPLATES_DIR/${service}-gpu.yml" ]]; then
          template_file="$TEMPLATES_DIR/${service}-gpu.yml"
        fi
      fi
    fi
    if [[ ! -f "$template_file" ]]; then
      format_error "Missing docker template: $template_file" "Reinstall the setup scripts."
      return 1
    fi

    printf '\n' >> "$service_blocks_file"
    cat "$template_file" >> "$service_blocks_file"

    case "$service" in
      postgres)
        inject_service_environment_overrides "$service_blocks_file" "postgres" \
          "POSTGRES_USER=${ENV_VALUES[POSTGRES_USER]:-}" \
          "POSTGRES_PASSWORD=${ENV_VALUES[POSTGRES_PASSWORD]:-}" \
          "POSTGRES_DB=${ENV_VALUES[POSTGRES_DATABASE]:-}"
        ;;
      neo4j)
        inject_service_environment_overrides "$service_blocks_file" "neo4j" \
          "NEO4J_AUTH=${_COMPOSE_RAW_VALUE_PREFIX}\${NEO4J_USERNAME:?missing}/\${NEO4J_PASSWORD:?missing}" \
          "NEO4J_dbms_default__database=${ENV_VALUES[NEO4J_DATABASE]:-neo4j}"
        ;;
      mongodb)
        ;;
      redis)
        ;;
      milvus)
        ;;
      qdrant)
        ;;
      memgraph)
        ;;
      opensearch)
        ;;
      vllm-rerank)
        ;;
      vllm-embed)
        ;;
    esac
  done

  _merge_managed_service_blocks "$tmp_file" "$service_blocks_file"
  _normalize_services_section_spacing "$tmp_file"
  _append_referenced_volume_blocks "$tmp_file"

  mv "$tmp_file" "$output_file"
}

prepare_lightrag_service_for_generated_compose() {
  # Let the containerized app read the mounted .env itself. Keeping env_file
  # here would make Docker Compose re-parse the same secrets and expand '$'.
  local compose_file="$1"
  local tmp_file="${compose_file}.strip-env-file"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local in_lightrag="no"
  local in_env_file="no"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_env_file" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]] ]]; then
        continue
      fi
      in_env_file="no"
    fi

    if [[ "$in_lightrag" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  lightrag:" ]]; then
      in_lightrag="no"
    fi

    if [[ "$in_lightrag" == "yes" && "$line" == "    env_file:" ]]; then
      in_env_file="yes"
      continue
    fi

    if [[ "$in_lightrag" == "yes" && "$line" =~ ^[[:space:]]{4}container_name: ]]; then
      continue
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      in_env_file="no"
    fi
  done < "$compose_file"

  mv "$tmp_file" "$compose_file"
}

normalize_lightrag_restart_policy() {
  local compose_file="$1"
  local tmp_file="${compose_file}.normalize-lightrag-restart"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local in_lightrag="no"
  local in_deploy="no"
  local deploy_seen="no"
  local insert_blank_after_deploy="no"
  local skip_blank_after_removed_restart="no"
  local -a deploy_lines=()

  _trim_trailing_blank_lines() {
    local file="$1"
    local trim_file="${file}.trim"
    _FILE_OPS_CLEANUP_TMP+=("$trim_file")

    awk '
      { lines[NR] = $0 }
      END {
        last = NR
        while (last > 0 && lines[last] == "") {
          last--
        }
        for (i = 1; i <= last; i++) {
          print lines[i]
        }
      }
    ' "$file" > "$trim_file"

    mv "$trim_file" "$file"
  }

  _write_normalized_lightrag_deploy_block() {
    local deploy_line
    local skipping_restart_policy="no"

    printf '    deploy:\n' >> "$tmp_file"
    for deploy_line in "${deploy_lines[@]}"; do
      if [[ -z "$deploy_line" ]]; then
        continue
      fi

      if [[ "$skipping_restart_policy" == "yes" ]]; then
        if [[ "$deploy_line" =~ ^[[:space:]]{8} ]]; then
          continue
        fi
        skipping_restart_policy="no"
      fi

      if [[ "$deploy_line" == "      restart_policy:" ]]; then
        skipping_restart_policy="yes"
        continue
      fi

      printf '%s\n' "$deploy_line" >> "$tmp_file"
    done

    printf '      restart_policy:\n' >> "$tmp_file"
    printf '        condition: on-failure\n' >> "$tmp_file"
    printf '        max_attempts: 10\n' >> "$tmp_file"
  }

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_deploy" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{6} || -z "$line" ]]; then
        deploy_lines+=("$line")
        continue
      fi

      _trim_trailing_blank_lines "$tmp_file"
      _write_normalized_lightrag_deploy_block
      deploy_lines=()
      in_deploy="no"
      if [[ "$line" =~ ^[[:space:]]{2}[^[:space:]] || "$line" =~ ^[^[:space:]] ]]; then
        insert_blank_after_deploy="yes"
      fi
    fi

    if [[ "$in_lightrag" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  lightrag:" ]] || \
      [[ "$in_lightrag" == "yes" && "$line" =~ ^[^[:space:]] ]]; then
      if [[ "$deploy_seen" != "yes" ]]; then
        _trim_trailing_blank_lines "$tmp_file"
        _write_normalized_lightrag_deploy_block
        insert_blank_after_deploy="yes"
      fi
      in_lightrag="no"
      deploy_seen="no"
      skip_blank_after_removed_restart="no"
    fi

    if [[ "$in_lightrag" == "yes" && "$line" == "    deploy:" ]]; then
      in_deploy="yes"
      deploy_seen="yes"
      deploy_lines=()
      continue
    fi

    if [[ "$in_lightrag" == "yes" && "$line" =~ ^[[:space:]]{4}restart: ]]; then
      skip_blank_after_removed_restart="yes"
      continue
    fi

    if [[ "$skip_blank_after_removed_restart" == "yes" && "$in_lightrag" == "yes" ]]; then
      if [[ -z "$line" ]]; then
        continue
      fi
      skip_blank_after_removed_restart="no"
    fi

    if [[ "$insert_blank_after_deploy" == "yes" ]]; then
      printf '\n' >> "$tmp_file"
      insert_blank_after_deploy="no"
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      in_deploy="no"
      deploy_seen="no"
      insert_blank_after_deploy="no"
      skip_blank_after_removed_restart="no"
      deploy_lines=()
    fi
  done < "$compose_file"

  if [[ "$in_deploy" == "yes" ]]; then
    _trim_trailing_blank_lines "$tmp_file"
    _write_normalized_lightrag_deploy_block
    deploy_seen="yes"
  fi

  if [[ "$in_lightrag" == "yes" && "$deploy_seen" != "yes" ]]; then
    _trim_trailing_blank_lines "$tmp_file"
    _write_normalized_lightrag_deploy_block
  fi

  mv "$tmp_file" "$compose_file"
}

append_lightrag_ssl_mount() {
  local array_name="$1"
  local container_path="$2"
  local relative_host_path=""
  local mount_entry=""

  if [[ -z "$container_path" ]]; then
    return 0
  fi

  if [[ "$container_path" != /app/data/* ]]; then
    format_error "Unsupported SSL path: ${container_path}" "Use paths staged under /app/data."
    return 1
  fi

  relative_host_path="./data/${container_path#/app/data/}"
  mount_entry="${relative_host_path}:${container_path}:ro"
  local -n _arr_ref="$array_name"
  _arr_ref+=("$mount_entry")
}

format_yaml_value() {
  local value="$1"
  local escaped="${value//\\/\\\\}"

  escaped="${escaped//\"/\\\"}"
  escaped="${escaped//\$/\$\$}"
  printf '"%s"' "$escaped"
}

_COMPOSE_RAW_VALUE_PREFIX="__LIGHTRAG_RAW_COMPOSE__:"

format_compose_environment_value() {
  local value="$1"

  if [[ "$value" == "${_COMPOSE_RAW_VALUE_PREFIX}"* ]]; then
    printf '%s' "${value#${_COMPOSE_RAW_VALUE_PREFIX}}"
    return 0
  fi

  format_yaml_value "$value"
}

inject_service_environment_overrides() {
  local compose_file="$1"
  local service_name="$2"
  shift 2
  local entries=("$@")
  local tmp_file="${compose_file}.${service_name}.env"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line key value
  local in_service="no"
  local in_environment="no"
  local environment_style="mapping"
  local inserted="no"
  local service_header="  ${service_name}:"

  if ((${#entries[@]} == 0)); then
    return 0
  fi

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_service" == "yes" && "$in_environment" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]] ]]; then
        environment_style="list"
      elif [[ "$line" =~ ^[[:space:]]{6}[A-Z0-9_]+: ]]; then
        environment_style="mapping"
      fi

      if [[ "$line" =~ ^[[:space:]]{4}[^[:space:]] || "$line" =~ ^[[:space:]]{2}[^[:space:]] || "$line" =~ ^[^[:space:]] ]]; then
        if [[ "$inserted" == "no" ]]; then
          _write_service_environment_entries "$tmp_file" "$environment_style" "${entries[@]}"
          inserted="yes"
        fi
        in_environment="no"
      fi
    elif [[ "$in_service" == "yes" && \
            ( "$line" =~ ^[[:space:]]{2}[^[:space:]] || "$line" =~ ^[^[:space:]] ) && \
            "$line" != "$service_header" ]]; then
      if [[ "$inserted" == "no" ]]; then
        printf '    environment:\n' >> "$tmp_file"
        _write_service_environment_entries "$tmp_file" "mapping" "${entries[@]}"
        inserted="yes"
      fi
      in_service="no"
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "$service_header" ]]; then
      in_service="yes"
      in_environment="no"
      environment_style="mapping"
    elif [[ "$in_service" == "yes" && "$line" == "    environment:" ]]; then
      in_environment="yes"
      environment_style="mapping"
    fi
  done < "$compose_file"

  if [[ "$in_service" == "yes" && "$inserted" == "no" ]]; then
    if [[ "$in_environment" != "yes" ]]; then
      printf '    environment:\n' >> "$tmp_file"
    fi
    _write_service_environment_entries "$tmp_file" "$environment_style" "${entries[@]}"
  fi

  mv "$tmp_file" "$compose_file"
}

# Return success when a volume mount entry is a wizard-managed SSL cert/key
# bind mount (./data/certs/* -> /app/data/certs/*, optional :ro suffix).
_is_wizard_ssl_bind_mount() {
  local mount_spec="$1"
  local host_path=""
  local remainder=""
  local container_path=""
  local mode=""
  local host_suffix=""
  local container_suffix=""

  host_path="${mount_spec%%:*}"
  remainder="${mount_spec#*:}"
  if [[ "$remainder" == "$mount_spec" ]]; then
    return 1
  fi

  container_path="${remainder%%:*}"
  if [[ "$container_path" == "$remainder" ]]; then
    mode=""
  else
    mode="${remainder#${container_path}:}"
  fi

  if [[ "$host_path" != ./data/certs/* ]]; then
    return 1
  fi

  if [[ "$container_path" != /app/data/certs/* ]]; then
    return 1
  fi

  # Wizard-generated SSL mounts are read-only. Keep non-read-only mounts so
  # user-defined overrides under /app/data/certs are not stripped.
  if [[ -n "$mode" && "$mode" != "ro" ]]; then
    return 1
  fi

  host_suffix="${host_path#./data/certs/}"
  container_suffix="${container_path#/app/data/certs/}"
  [[ "$host_suffix" == "$container_suffix" ]]
}

# Remove wizard-managed SSL bind mounts from the lightrag service's volumes
# block, leaving persistent and user-added /app/data/* mounts intact.
_strip_lightrag_wizard_bind_mounts() {
  local compose_file="$1"
  local tmp_file="${compose_file}.strip-mounts"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local mount_spec
  local in_lightrag="no"
  local in_volumes="no"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_lightrag" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  lightrag:" ]]; then
        in_lightrag="no"
        in_volumes="no"
      elif [[ "$line" == "    volumes:" ]]; then
        in_volumes="yes"
      elif [[ "$in_volumes" == "yes" ]]; then
        if [[ "$line" =~ ^[[:space:]]{4}[^[:space:]] ]]; then
          in_volumes="no"
        elif [[ "$line" =~ ^[[:space:]]{6}-[[:space:]] ]]; then
          mount_spec="${line#      - }"
          mount_spec="${mount_spec%\"}"
          mount_spec="${mount_spec#\"}"
          mount_spec="${mount_spec%\'}"
          mount_spec="${mount_spec#\'}"
          if _is_wizard_ssl_bind_mount "$mount_spec"; then
            continue
          fi
        fi
      fi
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      in_volumes="no"
    fi
  done < "$compose_file"

  mv "$tmp_file" "$compose_file"
}

_is_wizard_lightrag_port_mapping() {
  local port_spec="$(_strip_wrapping_quotes "$1")"

  if [[ "$port_spec" == '${HOST:-0.0.0.0}:${PORT:-9621}:9621' || \
        "$port_spec" == '${PORT:-9621}:9621' ]]; then
    return 0
  fi

  case "$port_spec" in
    9621|9621/tcp|*:9621|*:9621/tcp)
      return 0
      ;;
  esac

  return 1
}

_strip_lightrag_wizard_ports() {
  local compose_file="$1"
  local tmp_file="${compose_file}.strip-lightrag-ports"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local port_spec=""
  local in_lightrag="no"
  local in_ports="no"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_lightrag" == "yes" && "$in_ports" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]](.+)$ ]]; then
        port_spec="${BASH_REMATCH[1]}"
        if _is_wizard_lightrag_port_mapping "$port_spec"; then
          continue
        fi
      elif [[ ! "$line" =~ ^[[:space:]]{6} ]]; then
        in_ports="no"
      fi
    elif [[ "$in_lightrag" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  lightrag:" ]]; then
      in_lightrag="no"
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      in_ports="no"
    elif [[ "$in_lightrag" == "yes" && "$line" == "    ports:" ]]; then
      in_ports="yes"
    fi
  done < "$compose_file"

  mv "$tmp_file" "$compose_file"
}

inject_lightrag_bind_mounts() {
  local compose_file="$1"
  shift
  local mounts=("$@")
  local tmp_file="${compose_file}.mounts"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local in_lightrag="no"
  local in_volumes="no"
  local inserted="no"

  if ((${#mounts[@]} == 0)); then
    return 0
  fi

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_lightrag" == "yes" && "$in_volumes" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{4}[^[:space:]-] || "$line" =~ ^[[:space:]]{2}[^[:space:]] || "$line" =~ ^(volumes|networks): ]]; then
        if [[ "$inserted" == "no" ]]; then
          for mount in "${mounts[@]}"; do
            printf '      - "%s"\n' "$mount" >> "$tmp_file"
          done
          inserted="yes"
        fi
        in_volumes="no"
      fi
    elif [[ "$in_lightrag" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  lightrag:" ]]; then
      if [[ "$inserted" == "no" ]]; then
        printf '    volumes:\n' >> "$tmp_file"
        for mount in "${mounts[@]}"; do
          printf '      - "%s"\n' "$mount" >> "$tmp_file"
        done
        inserted="yes"
      fi
      in_lightrag="no"
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      in_volumes="no"
    elif [[ "$in_lightrag" == "yes" && "$line" == "    volumes:" ]]; then
      in_volumes="yes"
    fi
  done < "$compose_file"

  if [[ "$in_lightrag" == "yes" && "$inserted" == "no" ]]; then
    if [[ "$in_volumes" != "yes" ]]; then
      printf '    volumes:\n' >> "$tmp_file"
    fi
    for mount in "${mounts[@]}"; do
      printf '      - "%s"\n' "$mount" >> "$tmp_file"
    done
  fi

  mv "$tmp_file" "$compose_file"
}

inject_lightrag_port_mapping() {
  local compose_file="$1"
  local port_mapping="$2"
  local tmp_file="${compose_file}.ports"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local in_lightrag="no"
  local in_ports="no"
  local inserted="no"

  if [[ -z "$port_mapping" ]]; then
    return 0
  fi

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_lightrag" == "yes" && "$in_ports" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{4}[^[:space:]-] || "$line" =~ ^[[:space:]]{2}[^[:space:]] || "$line" =~ ^(volumes|networks): ]]; then
        if [[ "$inserted" == "no" ]]; then
          printf '      - "%s"\n' "$port_mapping" >> "$tmp_file"
          inserted="yes"
        fi
        in_ports="no"
      fi
    elif [[ "$in_lightrag" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  lightrag:" ]]; then
      if [[ "$inserted" == "no" ]]; then
        printf '    ports:\n' >> "$tmp_file"
        printf '      - "%s"\n' "$port_mapping" >> "$tmp_file"
        inserted="yes"
      fi
      in_lightrag="no"
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      in_ports="no"
    elif [[ "$in_lightrag" == "yes" && "$line" == "    ports:" ]]; then
      in_ports="yes"
    fi
  done < "$compose_file"

  if [[ "$in_lightrag" == "yes" && "$inserted" == "no" ]]; then
    if [[ "$in_ports" != "yes" ]]; then
      printf '    ports:\n' >> "$tmp_file"
    fi
    printf '      - "%s"\n' "$port_mapping" >> "$tmp_file"
  fi

  mv "$tmp_file" "$compose_file"
}

repair_misplaced_lightrag_depends_on() {
  local compose_file="$1"
  local tmp_file="${compose_file}.repair-lightrag-depends-on"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local in_services="no"
  local in_lightrag="no"
  local lightrag_has_depends_on="no"
  local candidate_service=""
  local candidate_root_service=""
  local captured_block=""
  local candidate_header=""
  local inserted="no"
  local in_candidate_service="no"
  local skipping_candidate_depends_on="no"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "services:" ]]; then
      in_services="yes"
      continue
    fi

    if [[ "$in_services" == "yes" && "$line" =~ ^[^[:space:]] && "$line" != "services:" ]]; then
      break
    fi

    if [[ "$in_services" != "yes" ]]; then
      continue
    fi

    if [[ "$in_lightrag" == "yes" ]]; then
      if [[ "$line" == "    depends_on:" ]]; then
        lightrag_has_depends_on="yes"
        break
      fi

      if [[ "$line" =~ ^[[:space:]]{2}([A-Za-z0-9_-]+):[[:space:]]*$ ]] && \
        [[ "${BASH_REMATCH[1]}" != "lightrag" ]]; then
        candidate_service="${BASH_REMATCH[1]}"
        candidate_root_service="$(_managed_service_root_name "$candidate_service")"
        if [[ -z "$candidate_root_service" || "$candidate_root_service" == "milvus" ]]; then
          break
        fi
        in_lightrag="no"
      fi
      continue
    fi

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      continue
    fi

    if [[ -n "$candidate_service" ]]; then
      if [[ "$line" == "    depends_on:" ]]; then
        captured_block="    depends_on:"$'\n'
        continue
      fi

      if [[ -n "$captured_block" ]]; then
        if [[ "$line" =~ ^[[:space:]]{6} ]]; then
          captured_block+="${line}"$'\n'
          continue
        fi
        break
      fi

      if [[ "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  ${candidate_service}:" ]]; then
        break
      fi

      if [[ "$line" =~ ^[^[:space:]] ]]; then
        break
      fi
    fi
  done < "$compose_file"

  if [[ "$lightrag_has_depends_on" == "yes" || -z "$captured_block" || -z "$candidate_service" ]]; then
    return 0
  fi

  candidate_header="  ${candidate_service}:"
  : > "$tmp_file"
  in_lightrag="no"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$skipping_candidate_depends_on" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{6} ]]; then
        continue
      fi
      skipping_candidate_depends_on="no"
    fi

    if [[ "$in_lightrag" == "yes" && "$inserted" == "no" ]] && \
      [[ ( "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "  lightrag:" ) || "$line" =~ ^[^[:space:]] ]]; then
      printf '%s' "$captured_block" >> "$tmp_file"
      inserted="yes"
      in_lightrag="no"
    fi

    if [[ "$line" == "$candidate_header" ]]; then
      in_candidate_service="yes"
    elif [[ "$in_candidate_service" == "yes" ]] && \
      [[ ( "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "$candidate_header" ) || "$line" =~ ^[^[:space:]] ]]; then
      in_candidate_service="no"
    fi

    if [[ "$in_candidate_service" == "yes" && "$line" == "    depends_on:" ]]; then
      skipping_candidate_depends_on="yes"
      continue
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
    fi
  done < "$compose_file"

  if [[ "$in_lightrag" == "yes" && "$inserted" == "no" ]]; then
    printf '%s' "$captured_block" >> "$tmp_file"
  fi

  mv "$tmp_file" "$compose_file"
}

inject_lightrag_environment_overrides() {
  local compose_file="$1"
  shift
  inject_service_environment_overrides "$compose_file" "lightrag" "$@"
}

inject_lightrag_depends_on() {
  local compose_file="$1"
  shift
  local candidate_service
  local managed_services=()
  local tmp_file="${compose_file}.depends-on"
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local line
  local in_lightrag="no"
  local in_depends_on="no"
  local inserted="no"
  local insert_blank_after_depends_on="no"
  local current_dep_name=""
  local current_dep_block=""
  local dep_name=""
  local dep_tail=""
  local dep_service=""
  declare -A preserved_dep_blocks=()
  declare -A preserved_dep_seen=()
  local -a preserved_dep_order=()

  for candidate_service in "$@"; do
    if _is_wizard_managed_root_service_name "$candidate_service"; then
      managed_services+=("$candidate_service")
    fi
  done

  _record_preserved_depends_on_entry() {
    local service_name="$1"
    local block="$2"

    if [[ -z "$service_name" ]] || _is_wizard_managed_root_service_name "$service_name"; then
      return 0
    fi

    if [[ -n "${preserved_dep_seen[$service_name]+set}" ]]; then
      return 0
    fi

    preserved_dep_seen["$service_name"]=1
    preserved_dep_order+=("$service_name")
    preserved_dep_blocks["$service_name"]="$block"
  }

  _flush_current_depends_on_entry() {
    if [[ -z "$current_dep_name" ]]; then
      return 0
    fi

    _record_preserved_depends_on_entry "$current_dep_name" "$current_dep_block"
    current_dep_name=""
    current_dep_block=""
  }

  _trim_trailing_blank_lines() {
    local file="$1"
    local trim_file="${file}.trim"
    _FILE_OPS_CLEANUP_TMP+=("$trim_file")

    awk '
      { lines[NR] = $0 }
      END {
        last = NR
        while (last > 0 && lines[last] == "") {
          last--
        }
        for (i = 1; i <= last; i++) {
          print lines[i]
        }
      }
    ' "$file" > "$trim_file"

    mv "$trim_file" "$file"
  }

  _write_lightrag_depends_on_block() {
    local managed_service
    local preserved_service

    if ((${#preserved_dep_order[@]} == 0 && ${#managed_services[@]} == 0)); then
      inserted="yes"
      return 0
    fi

    printf '    depends_on:\n' >> "$tmp_file"

    for preserved_service in "${preserved_dep_order[@]}"; do
      printf '%s' "${preserved_dep_blocks[$preserved_service]}" >> "$tmp_file"
    done

    for managed_service in "${managed_services[@]}"; do
      printf '      %s:\n' "$managed_service" >> "$tmp_file"
      printf '        condition: service_healthy\n' >> "$tmp_file"
    done

    inserted="yes"
  }

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_depends_on" == "yes" ]]; then
      if [[ -n "$current_dep_name" && "$line" =~ ^[[:space:]]{8} ]]; then
        current_dep_block+="${line}"$'\n'
        continue
      fi

      if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]](.+)$ ]]; then
        _flush_current_depends_on_entry
        dep_service="$(_strip_wrapping_quotes "${BASH_REMATCH[1]}")"
        _record_preserved_depends_on_entry \
          "$dep_service" \
          "$(printf '      %s:\n        condition: service_started\n' "$dep_service")"
        continue
      fi

      if [[ "$line" =~ ^[[:space:]]{6}([A-Za-z0-9_.-]+):[[:space:]]*(.*)$ ]]; then
        _flush_current_depends_on_entry
        dep_name="${BASH_REMATCH[1]}"
        dep_tail="${BASH_REMATCH[2]}"
        current_dep_name="$(_strip_wrapping_quotes "$dep_name")"
        if [[ -n "$dep_tail" ]]; then
          current_dep_block="      ${current_dep_name}: ${dep_tail}"$'\n'
        else
          current_dep_block="      ${current_dep_name}:"$'\n'
        fi
        continue
      fi

      _flush_current_depends_on_entry
      if [[ "$inserted" == "no" ]]; then
        _trim_trailing_blank_lines "$tmp_file"
        _write_lightrag_depends_on_block
      fi
      in_depends_on="no"
    fi

    if [[ "$in_lightrag" == "yes" && "$line" == "    depends_on:" ]]; then
      in_depends_on="yes"
      continue
    fi

    if [[ "$in_lightrag" == "yes" && \
          ( "$line" =~ ^[[:space:]]{2}[^[:space:]] || "$line" =~ ^[^[:space:]] ) && \
          "$line" != "  lightrag:" ]]; then
      if [[ "$inserted" == "no" ]]; then
        _trim_trailing_blank_lines "$tmp_file"
        _write_lightrag_depends_on_block
        insert_blank_after_depends_on="yes"
      fi
      in_lightrag="no"
    fi

    if [[ "$insert_blank_after_depends_on" == "yes" ]]; then
      printf '\n' >> "$tmp_file"
      insert_blank_after_depends_on="no"
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      inserted="no"
      insert_blank_after_depends_on="no"
      in_depends_on="no"
      current_dep_name=""
      current_dep_block=""
      preserved_dep_blocks=()
      preserved_dep_seen=()
      preserved_dep_order=()
    fi
  done < "$compose_file"

  if [[ "$in_depends_on" == "yes" ]]; then
    _flush_current_depends_on_entry
    if [[ "$inserted" == "no" ]]; then
      _trim_trailing_blank_lines "$tmp_file"
      _write_lightrag_depends_on_block
    fi
  elif [[ "$in_lightrag" == "yes" && "$inserted" == "no" ]]; then
    _trim_trailing_blank_lines "$tmp_file"
    _write_lightrag_depends_on_block
  fi

  mv "$tmp_file" "$compose_file"
}

# Find the first generated compose file in priority order.
# Prints the path if found, empty string if not.
find_generated_compose_file() {
  local repo_root="${REPO_ROOT:-.}"
  local candidates=(
    "final:$repo_root/docker-compose.final.yml"
    "development:$repo_root/docker-compose.development.yml"
    "production:$repo_root/docker-compose.production.yml"
    "custom:$repo_root/docker-compose.custom.yml"
    "local:$repo_root/docker-compose.local.yml"
  )
  local candidate f

  for candidate in "${candidates[@]}"; do
    f="${candidate#*:}"
    if [[ -f "$f" ]]; then
      printf '%s' "$f"
      return 0
    fi
  done
  printf ''
}

# Detect service names in a compose file's services: block (excluding lightrag).
# Prints one service name per line.
detect_compose_services() {
  local compose_file="$1"
  local in_services="no"
  local line

  if [[ ! -f "$compose_file" ]]; then
    return 0
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "services:" ]]; then
      in_services="yes"
      continue
    fi
    if [[ "$in_services" == "yes" ]]; then
      if [[ "$line" =~ ^[^[:space:]] && "$line" != "services:" ]]; then
        in_services="no"
        continue
      fi
      if [[ "$line" =~ ^[[:space:]]{2}([A-Za-z0-9_-]+):[[:space:]]*$ ]]; then
        local svc_name="${BASH_REMATCH[1]}"
        if [[ "$svc_name" != "lightrag" ]]; then
          printf '%s\n' "$svc_name"
        fi
      fi
    fi
  done < "$compose_file"
}

detect_managed_root_services() {
  local compose_file="$1"
  local service_name
  local root_service
  declare -A seen_roots=()

  while IFS= read -r service_name; do
    root_service="$(_managed_service_root_name "$service_name")"
    if [[ -n "$root_service" && -z "${seen_roots[$root_service]+set}" ]]; then
      seen_roots["$root_service"]=1
      printf '%s\n' "$root_service"
    fi
  done < <(detect_compose_services "$compose_file")
}
