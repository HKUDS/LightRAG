# Preset helpers for setup-managed overrides.

apply_preset_overwrite() {
  local entry key value

  for entry in "$@"; do
    key="${entry%%=*}"
    value="${entry#*=}"
    ENV_VALUES["$key"]="$value"
  done
}
