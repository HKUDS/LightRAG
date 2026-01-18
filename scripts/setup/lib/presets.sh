# Preset loading helpers.
# shellcheck disable=SC2034

apply_preset() {
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
