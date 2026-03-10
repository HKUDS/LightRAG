# Prompt helpers for interactive setup.

CLEAR_INPUT_SENTINEL="__LIGHTRAG_CLEAR__"

_truncate_for_display() {
  local value="$1"
  local max=50
  if [[ ${#value} -gt $max ]]; then
    printf '%s' "${value:0:$max}..."
  else
    printf '%s' "$value"
  fi
}

mask_sensitive_input() {
  local prompt="$1"
  local value

  read -r -p "$prompt" value
  echo >&2
  printf '%s' "$value"
}

prompt_secret_with_default() {
  local prompt="$1"
  local default="${2:-}"
  local value

  if [[ -n "$default" ]]; then
    local display_default
    display_default="$(_truncate_for_display "$default")"
    read -r -p "$prompt [${display_default}]: " value
  else
    read -r -p "$prompt" value
  fi
  echo >&2

  if [[ -z "$value" ]]; then
    value="$default"
  fi

  printf '%s' "$value"
}

prompt_clearable_with_default() {
  local prompt="$1"
  local default="${2:-}"
  local value
  local prompt_text="$prompt"

  if [[ -n "$default" ]]; then
    prompt_text="$prompt (Enter to keep, type 'clear' to remove)"
  else
    prompt_text="$prompt (type 'clear' to remove)"
  fi

  value="$(prompt_with_default "$prompt_text" "$default")"
  if [[ "${value,,}" == "clear" ]]; then
    printf '%s' "$CLEAR_INPUT_SENTINEL"
    return 0
  fi

  printf '%s' "$value"
}

prompt_clearable_secret_with_default() {
  local prompt="$1"
  local default="${2:-}"
  local value
  local prompt_text="$prompt"

  if [[ -n "$default" ]]; then
    prompt_text="$prompt (Enter to keep, type 'clear' to remove)"
  else
    prompt_text="$prompt (type 'clear' to remove)"
  fi

  value="$(prompt_secret_with_default "$prompt_text" "$default")"
  if [[ "${value,,}" == "clear" ]]; then
    printf '%s' "$CLEAR_INPUT_SENTINEL"
    return 0
  fi

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

confirm_default_no() {
  local prompt="$1"
  local response
  while true; do
    read -r -n 1 -p "$prompt [y/N]: " response
    echo
    case "$response" in
      y|Y) return 0 ;;
      n|N|"") return 1 ;;
    esac
  done
}

confirm_default_yes() {
  local prompt="$1"
  local response
  while true; do
    read -r -n 1 -p "$prompt [Y/n]: " response
    echo
    case "$response" in
      y|Y|"") return 0 ;;
      n|N) return 1 ;;
    esac
  done
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

prompt_secret_until_valid_with_default() {
  local prompt="$1"
  local default="$2"
  local validator="$3"
  shift 3
  local value

  while true; do
    value="$(prompt_secret_with_default "$prompt" "$default")"
    if "$validator" "$value" "$@"; then
      printf '%s' "$value"
      return 0
    fi
    echo "Invalid value. Please try again."
  done
}

prompt_required_secret() {
  local prompt="$1"
  local value

  while true; do
    value="$(mask_sensitive_input "$prompt")"
    if [[ -n "$value" ]]; then
      printf '%s' "$value"
      return 0
    fi
    echo "Value cannot be empty. Please try again."
  done
}
