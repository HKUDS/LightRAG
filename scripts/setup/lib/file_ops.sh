# File operations for interactive setup.

# Registry of temp files created during this session; cleaned up on exit.
_FILE_OPS_CLEANUP_TMP=()
declare -A _FILE_OPS_VOLUME_BLOCKS=()
declare -a _FILE_OPS_VOLUME_ORDER=()
_file_ops_cleanup() {
  local f
  for f in "${_FILE_OPS_CLEANUP_TMP[@]:-}"; do
    rm -f "$f" 2>/dev/null || true
  done
}
trap '_file_ops_cleanup' EXIT INT TERM

# Keys whose values are always written with double quotes (e.g. may contain spaces).
_ALWAYS_QUOTED_KEYS="|WEBUI_TITLE|WEBUI_DESCRIPTION|"

format_env_value() {
  local value="$1"
  local key="${2:-}"
  local escaped

  if [[ -z "$value" ]]; then
    printf ''
    return
  fi

  if [[ -n "$key" && "$_ALWAYS_QUOTED_KEYS" == *"|${key}|"* ]] || \
     [[ "$value" =~ [[:space:]] || "$value" == *"\""* || "$value" == *"$"* || "$value" == *"#"* ]]; then
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
    if [[ "$line" =~ ^#[[:space:]]*([A-Z0-9_]+)=(.*)$ ]]; then
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
    if [[ "$line" =~ ^[A-Z0-9_]+= ]]; then
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
    elif [[ "$line" =~ ^#[[:space:]]*([A-Z0-9_]+)=(.*)$ ]]; then
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

  mv "$tmp_file" "$output_file"
}

# All environment keys the wizard may inject into the lightrag service via
# COMPOSE_ENV_OVERRIDES.  Used to remove stale entries before re-injection so
# keys no longer needed are not left behind in the compose file.
_WIZARD_COMPOSE_LIGHTRAG_KEYS=(
  "EMBEDDING_BINDING_HOST" "RERANK_BINDING_HOST" "LLM_BINDING_HOST"
  "REDIS_URI" "MONGO_URI" "NEO4J_URI" "MILVUS_URI" "QDRANT_URL" "MEMGRAPH_URI"
  "POSTGRES_HOST" "POSTGRES_PORT" "PORT" "HOST" "SSL_CERTFILE" "SSL_KEYFILE"
)

_is_wizard_managed_root_service_name() {
  local service_name="$1"

  case "$service_name" in
    postgres|neo4j|mongodb|redis|milvus|qdrant|memgraph|vllm-embed|vllm-rerank)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

_is_wizard_managed_service_name() {
  local service_name="$1"

  case "$service_name" in
    postgres|neo4j|mongodb|redis|milvus|milvus-etcd|milvus-minio|qdrant|memgraph|vllm-embed|vllm-rerank)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

_is_wizard_managed_volume_name() {
  local volume_name="$1"

  case "$volume_name" in
    postgres_data|neo4j_data|mongo_data|redis_data|milvus_data|milvus-etcd_data|milvus-minio_data|qdrant_data|memgraph_data|vllm_rerank_cache|vllm_embed_cache)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
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
      printf '      - %s\n' "$(format_yaml_value "${key}=${value}")" >> "$tmp_file"
    else
      printf '      %s: %s\n' "$key" "$(format_yaml_value "$value")" >> "$tmp_file"
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
  local line current_service=""
  local in_services="no"
  local in_top_volumes="no"

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Detect top-level (non-indented) keys.
    if [[ "$line" =~ ^[A-Za-z] ]]; then
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
    fi

    # Skip wizard-managed services; preserve lightrag and all user-added services.
    if [[ "$in_services" == "yes" && -n "$current_service" ]] && \
      [[ "$current_service" != "lightrag" ]] && \
      _is_wizard_managed_service_name "$current_service"; then
      continue
    fi

    printf '%s\n' "$line" >> "$tmp_file"
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
  local volume_name=""
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
      continue
    fi

    if [[ -z "$current_service" ]]; then
      continue
    fi

    if [[ "$line" == "    volumes:" ]]; then
      in_volumes="yes"
      in_long_volume_entry="no"
      long_volume_type=""
      continue
    fi

    if [[ "$in_volumes" != "yes" ]]; then
      continue
    fi

    if [[ "$line" =~ ^[[:space:]]{4}[^[:space:]-] || "$line" =~ ^[[:space:]]{2}[^[:space:]] ]]; then
      in_volumes="no"
      in_long_volume_entry="no"
      long_volume_type=""
      continue
    fi

    if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]](.+)$ ]]; then
      local volume_entry="${BASH_REMATCH[1]}"
      volume_entry="$(_strip_wrapping_quotes "$volume_entry")"
      in_long_volume_entry="no"
      long_volume_type=""

      if [[ "$volume_entry" == "type: volume" ]]; then
        in_long_volume_entry="yes"
        long_volume_type="volume"
        continue
      fi

      if [[ "$volume_entry" == "type: bind" || "$volume_entry" == "type: tmpfs" ]]; then
        in_long_volume_entry="yes"
        long_volume_type="other"
        continue
      fi

      volume_name="$(_extract_named_volume_name "$volume_entry")" || continue
      if [[ -z "${seen[$volume_name]+set}" ]]; then
        seen["$volume_name"]=1
        printf '%s\n' "$volume_name"
      fi
      continue
    fi

    if [[ "$in_long_volume_entry" == "yes" && "$long_volume_type" == "volume" ]] && \
      [[ "$line" =~ ^[[:space:]]{8}source:[[:space:]]*(.+)$ ]]; then
      volume_name="$(_strip_wrapping_quotes "${BASH_REMATCH[1]}")"
      if [[ -n "$volume_name" && -z "${seen[$volume_name]+set}" ]]; then
        seen["$volume_name"]=1
        printf '%s\n' "$volume_name"
      fi
    fi
  done < "$compose_file"
}

_append_referenced_volume_blocks() {
  local compose_file="$1"
  local -a referenced_volumes=()
  local volume_name

  while IFS= read -r volume_name; do
    if [[ -n "$volume_name" ]]; then
      referenced_volumes+=("$volume_name")
    fi
  done < <(_collect_referenced_named_volumes "$compose_file")

  if ((${#referenced_volumes[@]} == 0)); then
    return 0
  fi

  printf '\nvolumes:\n' >> "$compose_file"
  for volume_name in "${referenced_volumes[@]}"; do
    if _is_wizard_managed_volume_name "$volume_name"; then
      printf '  %s:\n' "$volume_name" >> "$compose_file"
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
  _FILE_OPS_CLEANUP_TMP+=("$tmp_file")
  local template_file
  local lightrag_mounts=()
  local lightrag_env_entries=()
  local key

  # Prefer the existing generated compose as the starting point to preserve
  # any user customisations to the lightrag service.  Fall back to the base
  # docker-compose.yml when the output file doesn't exist yet.
  if [[ -f "$output_file" && "$output_file" != "$base_file" ]]; then
    _collect_top_level_volume_blocks "$output_file"
    cp "$output_file" "$tmp_file"
    # Strip wizard-managed services and top-level volumes. User-managed
    # services are preserved, while volumes are rebuilt from final service
    # references after managed templates are appended.
    _strip_wizard_managed_services_and_top_level_volumes "$tmp_file"
  elif [[ -f "$base_file" ]]; then
    _collect_top_level_volume_blocks "$base_file"
    cp "$base_file" "$tmp_file"
    _strip_wizard_managed_services_and_top_level_volumes "$tmp_file"
  else
    _FILE_OPS_VOLUME_BLOCKS=()
    _FILE_OPS_VOLUME_ORDER=()
    printf 'services:\n' > "$tmp_file"
  fi

  prepare_lightrag_service_for_generated_compose "$tmp_file"
  # Remove stale wizard-managed keys from lightrag's environment so that
  # keys no longer in COMPOSE_ENV_OVERRIDES are not left behind.
  _strip_lightrag_wizard_environment_keys "$tmp_file"

  # Remove stale wizard-managed bind mounts from lightrag's volumes so that
  # mounts no longer needed (e.g. after SSL removal) are not left behind.
  _strip_lightrag_wizard_bind_mounts "$tmp_file"

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

  for service in "${DOCKER_SERVICES[@]}"; do
    template_file="$TEMPLATES_DIR/${service}.yml"
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

    printf '\n' >> "$tmp_file"
    cat "$template_file" >> "$tmp_file"

    case "$service" in
      postgres)
        inject_service_environment_overrides "$tmp_file" "postgres" \
          "POSTGRES_USER=${ENV_VALUES[POSTGRES_USER]:-}" \
          "POSTGRES_PASSWORD=${ENV_VALUES[POSTGRES_PASSWORD]:-}" \
          "POSTGRES_DB=${ENV_VALUES[POSTGRES_DATABASE]:-}"
        ;;
      neo4j)
        inject_service_environment_overrides "$tmp_file" "neo4j" \
          "NEO4J_AUTH=neo4j/${ENV_VALUES[NEO4J_PASSWORD]:-neo4j_password}" \
          "NEO4J_dbms_default__database=${ENV_VALUES[NEO4J_DATABASE]:-neo4j}"
        ;;
      mongodb)
        ;;
      redis)
        ;;
      milvus)
        inject_service_environment_overrides "$tmp_file" "milvus" \
          "MINIO_ACCESS_KEY_ID=${ENV_VALUES[MINIO_ACCESS_KEY_ID]:-minioadmin}" \
          "MINIO_SECRET_ACCESS_KEY=${ENV_VALUES[MINIO_SECRET_ACCESS_KEY]:-minioadmin}"
        inject_service_environment_overrides "$tmp_file" "milvus-minio" \
          "MINIO_ROOT_USER=${ENV_VALUES[MINIO_ACCESS_KEY_ID]:-minioadmin}" \
          "MINIO_ROOT_PASSWORD=${ENV_VALUES[MINIO_SECRET_ACCESS_KEY]:-minioadmin}"
        ;;
      qdrant)
        ;;
      memgraph)
        ;;
      vllm-rerank)
        ;;
      vllm-embed)
        ;;
    esac
  done

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

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "  lightrag:" ]]; then
      in_lightrag="yes"
      in_env_file="no"
    fi
  done < "$compose_file"

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

      if [[ "$line" =~ ^[[:space:]]{4}[^[:space:]] || "$line" =~ ^[[:space:]]{2}[^[:space:]] || "$line" =~ ^(volumes|networks): ]]; then
        if [[ "$inserted" == "no" ]]; then
          _write_service_environment_entries "$tmp_file" "$environment_style" "${entries[@]}"
          inserted="yes"
        fi
        in_environment="no"
      fi
    elif [[ "$in_service" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "$service_header" ]]; then
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

inject_lightrag_environment_overrides() {
  local compose_file="$1"
  shift
  inject_service_environment_overrides "$compose_file" "lightrag" "$@"
}

# Find the first generated compose file in priority order.
# Prints the path if found, empty string if not.
find_generated_compose_file() {
  local repo_root="${REPO_ROOT:-.}"
  local candidates=(
    "$repo_root/docker-compose.final.yml"
    "$repo_root/docker-compose.development.yml"
    "$repo_root/docker-compose.production.yml"
    "$repo_root/docker-compose.custom.yml"
    "$repo_root/docker-compose.local.yml"
  )
  local f
  for f in "${candidates[@]}"; do
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

  while IFS= read -r service_name; do
    if _is_wizard_managed_root_service_name "$service_name"; then
      printf '%s\n' "$service_name"
    fi
  done < <(detect_compose_services "$compose_file")
}
