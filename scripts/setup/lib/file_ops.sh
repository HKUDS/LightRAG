# File operations for interactive setup.

format_env_value() {
  local value="$1"
  local escaped

  if [[ -z "$value" ]]; then
    printf ''
    return
  fi

  if [[ "$value" =~ [[:space:]] || "$value" == *"\""* || "$value" == *"$"* ]]; then
    # Double-quoted .env values only need escaping for backslash and double quote.
    # Do not escape '$': python-dotenv preserves plain '$' literally, while '\$'
    # changes the loaded value.
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
    cp "$env_file" "$backup_file"
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
  local line key value
  local -A written_keys=()

  if [[ ! -f "$template_file" ]]; then
    echo "env.example not found at $template_file" >&2
    return 1
  fi

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[A-Z0-9_]+= ]]; then
      key="${line%%=*}"
      if [[ -z "${written_keys[$key]+set}" ]]; then
        if [[ -n "${ENV_VALUES[$key]+set}" ]]; then
          value="${ENV_VALUES[$key]}"
          printf '%s=%s\n' "$key" "$(format_env_value "$value")" >> "$tmp_file"
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
    elif [[ "$line" =~ ^#[[:space:]]*([A-Z0-9_]+)= ]]; then
      key="${BASH_REMATCH[1]}"
      if [[ -z "${written_keys[$key]+set}" && -n "${ENV_VALUES[$key]+set}" ]]; then
        value="${ENV_VALUES[$key]}"
        printf '%s=%s\n' "$key" "$(format_env_value "$value")" >> "$tmp_file"
        written_keys["$key"]=1
      else
        printf '%s\n' "$line" >> "$tmp_file"
      fi
    else
      printf '%s\n' "$line" >> "$tmp_file"
    fi
  done < "$template_file"

  mv "$tmp_file" "$output_file"
}

generate_docker_compose() {
  local output_file="${1:-${REPO_ROOT:-.}/docker-compose.yml}"
  local base_file="${REPO_ROOT:-.}/docker-compose.yml"
  local tmp_file="${output_file}.tmp"
  local template_file
  local volume_names=()
  local lightrag_mounts=()
  local lightrag_env_entries=()
  local key

  if [[ -f "$base_file" ]]; then
    cp "$base_file" "$tmp_file"
  else
    printf 'services:\n' > "$tmp_file"
  fi

  prepare_lightrag_service_for_generated_compose "$tmp_file"

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
        volume_names+=("postgres_data")
        ;;
      neo4j)
        inject_service_environment_overrides "$tmp_file" "neo4j" \
          "NEO4J_AUTH=neo4j/${ENV_VALUES[NEO4J_PASSWORD]:-neo4j_password}" \
          "NEO4J_dbms_default__database=${ENV_VALUES[NEO4J_DATABASE]:-neo4j}"
        volume_names+=("neo4j_data")
        ;;
      mongodb)
        volume_names+=("mongo_data")
        ;;
      redis)
        volume_names+=("redis_data")
        ;;
      milvus)
        inject_service_environment_overrides "$tmp_file" "milvus" \
          "MINIO_ACCESS_KEY_ID=${ENV_VALUES[MINIO_ACCESS_KEY_ID]:-minioadmin}" \
          "MINIO_SECRET_ACCESS_KEY=${ENV_VALUES[MINIO_SECRET_ACCESS_KEY]:-minioadmin}"
        inject_service_environment_overrides "$tmp_file" "minio" \
          "MINIO_ROOT_USER=${ENV_VALUES[MINIO_ACCESS_KEY_ID]:-minioadmin}" \
          "MINIO_ROOT_PASSWORD=${ENV_VALUES[MINIO_SECRET_ACCESS_KEY]:-minioadmin}"
        volume_names+=("milvus_data" "etcd_data" "minio_data")
        ;;
      qdrant)
        volume_names+=("qdrant_data")
        ;;
      memgraph)
        volume_names+=("memgraph_data")
        ;;
      vllm-rerank)
        volume_names+=("vllm_rerank_cache")
        ;;
    esac
  done

  if ((${#volume_names[@]} > 0)); then
    printf '\nvolumes:\n' >> "$tmp_file"
    for volume in "${volume_names[@]}"; do
      printf '  %s:\n' "$volume" >> "$tmp_file"
    done
  fi

  mv "$tmp_file" "$output_file"
}

prepare_lightrag_service_for_generated_compose() {
  # Let the containerized app read the mounted .env itself. Keeping env_file
  # here would make Docker Compose re-parse the same secrets and expand '$'.
  local compose_file="$1"
  local tmp_file="${compose_file}.strip-env-file"
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
  eval "$array_name+=(\"\$mount_entry\")"
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
  local line key value
  local in_service="no"
  local in_environment="no"
  local inserted="no"
  local service_header="  ${service_name}:"

  if ((${#entries[@]} == 0)); then
    return 0
  fi

  : > "$tmp_file"

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$in_service" == "yes" && "$in_environment" == "yes" ]]; then
      if [[ "$line" =~ ^[[:space:]]{4}[^[:space:]] || "$line" =~ ^[[:space:]]{2}[^[:space:]] || "$line" =~ ^(volumes|networks): ]]; then
        if [[ "$inserted" == "no" ]]; then
          for entry in "${entries[@]}"; do
            key="${entry%%=*}"
            value="${entry#*=}"
            printf '      %s: %s\n' "$key" "$(format_yaml_value "$value")" >> "$tmp_file"
          done
          inserted="yes"
        fi
        in_environment="no"
      fi
    elif [[ "$in_service" == "yes" && "$line" =~ ^[[:space:]]{2}[^[:space:]] && "$line" != "$service_header" ]]; then
      if [[ "$inserted" == "no" ]]; then
        printf '    environment:\n' >> "$tmp_file"
        for entry in "${entries[@]}"; do
          key="${entry%%=*}"
          value="${entry#*=}"
          printf '      %s: %s\n' "$key" "$(format_yaml_value "$value")" >> "$tmp_file"
        done
        inserted="yes"
      fi
      in_service="no"
    fi

    printf '%s\n' "$line" >> "$tmp_file"

    if [[ "$line" == "$service_header" ]]; then
      in_service="yes"
      in_environment="no"
    elif [[ "$in_service" == "yes" && "$line" == "    environment:" ]]; then
      in_environment="yes"
    fi
  done < "$compose_file"

  if [[ "$in_service" == "yes" && "$inserted" == "no" ]]; then
    if [[ "$in_environment" != "yes" ]]; then
      printf '    environment:\n' >> "$tmp_file"
    fi
    for entry in "${entries[@]}"; do
      key="${entry%%=*}"
      value="${entry#*=}"
      printf '      %s: %s\n' "$key" "$(format_yaml_value "$value")" >> "$tmp_file"
    done
  fi

  mv "$tmp_file" "$compose_file"
}

inject_lightrag_bind_mounts() {
  local compose_file="$1"
  shift
  local mounts=("$@")
  local tmp_file="${compose_file}.mounts"
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
