# File operations for interactive setup.

format_env_value() {
  local value="$1"
  local escaped

  if [[ -z "$value" ]]; then
    printf ''
    return
  fi

  if [[ "$value" =~ [[:space:]] || "$value" == *"\""* || "$value" == *"$"* ]]; then
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
        printf '%s\n' "$line" >> "$tmp_file"
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

  if [[ -f "$base_file" ]]; then
    cp "$base_file" "$tmp_file"
  else
    printf 'services:\n' > "$tmp_file"
  fi

  if [[ -n "${ENV_VALUES[SSL_CERTFILE]:-}" ]]; then
    lightrag_mounts+=("${ENV_VALUES[SSL_CERTFILE]}:${ENV_VALUES[SSL_CERTFILE]}:ro")
  fi
  if [[ -n "${ENV_VALUES[SSL_KEYFILE]:-}" ]]; then
    lightrag_mounts+=("${ENV_VALUES[SSL_KEYFILE]}:${ENV_VALUES[SSL_KEYFILE]}:ro")
  fi
  if ((${#lightrag_mounts[@]} > 0)); then
    inject_lightrag_bind_mounts "$tmp_file" "${lightrag_mounts[@]}"
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
        volume_names+=("postgres_data")
        ;;
      neo4j)
        volume_names+=("neo4j_data")
        ;;
      mongodb)
        volume_names+=("mongo_data")
        ;;
      redis)
        volume_names+=("redis_data")
        ;;
      milvus)
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
