# Preset loading helpers.
# shellcheck disable=SC2034

apply_preset() {
  local entry key value

  for entry in "$@"; do
    key="${entry%%=*}"
    value="${entry#*=}"
    if [[ -z "${ENV_VALUES[$key]+set}" || -z "${ENV_VALUES[$key]}" ]]; then
      ENV_VALUES["$key"]="$value"
    fi
  done
}

apply_preset_overwrite() {
  local entry key value

  for entry in "$@"; do
    key="${entry%%=*}"
    value="${entry#*=}"
    ENV_VALUES["$key"]="$value"
  done
}

load_preset() {
  local preset_name="$1"

  case "$preset_name" in
    development)
      apply_preset "${PRESET_DEVELOPMENT[@]}"
      ;;
    production)
      apply_preset "${PRESET_PRODUCTION[@]}"
      ;;
    local)
      apply_preset "${PRESET_LOCAL[@]}"
      ;;
    *)
      echo "Unknown preset: $preset_name" >&2
      return 1
      ;;
  esac
}

load_storage_preset_overwrite() {
  local preset_name="$1"

  case "$preset_name" in
    development)
      apply_preset_overwrite "${PRESET_DEVELOPMENT[@]:0:4}"
      ;;
    production)
      apply_preset_overwrite "${PRESET_PRODUCTION[@]:0:4}"
      ;;
    local)
      apply_preset_overwrite "${PRESET_LOCAL[@]:0:4}"
      ;;
    *)
      echo "Unknown preset: $preset_name" >&2
      return 1
      ;;
  esac
}
