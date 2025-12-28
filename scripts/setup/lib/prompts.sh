# Prompt helpers for interactive setup.

mask_sensitive_input() {
  local prompt="$1"
  local value

  read -r -s -p "$prompt" value
  echo >&2
  printf '%s' "$value"
}

prompt_with_default() {
  local prompt="$1"
  local default="$2"
  local value

  if [[ -n "$default" ]]; then
    read -r -p "$prompt [$default]: " value
  else
    read -r -p "$prompt: " value
  fi

  if [[ -z "$value" ]]; then
    value="$default"
  fi

  printf '%s' "$value"
}

confirm() {
  local prompt="$1"
  local response

  read -r -p "$prompt [y/N]: " response
  case "${response,,}" in
    y|yes)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

prompt_until_valid() {
  local prompt="$1"
  local default="$2"
  local validator="$3"
  shift 3
  local value

  while true; do
    value="$(prompt_with_default "$prompt" "$default")"
    if "$validator" "$value" "$@"; then
      printf '%s' "$value"
      return 0
    fi
    echo "Invalid value. Please try again."
  done
}

prompt_secret_until_valid() {
  local prompt="$1"
  local validator="$2"
  shift 2
  local value

  while true; do
    value="$(mask_sensitive_input "$prompt")"
    if "$validator" "$value" "$@"; then
      printf '%s' "$value"
      return 0
    fi
    echo "Invalid value. Please try again."
  done
}
