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

style_prompt_text() {
  local prompt="$1"

  if [[ -n "${COLOR_YELLOW:-}" && "$prompt" == *Docker* ]]; then
    prompt="${prompt//Docker/${COLOR_YELLOW}Docker${COLOR_RESET}}"
  fi

  printf '%s' "$prompt"
}

confirm_default_no() {
  local prompt="$1"
  local response
  local styled_prompt

  styled_prompt="$(style_prompt_text "$prompt")"
  while true; do
    read -r -n 1 -p "$styled_prompt [y/N]: " response
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
  local styled_prompt

  styled_prompt="$(style_prompt_text "$prompt")"
  while true; do
    read -r -n 1 -p "$styled_prompt [Y/n]: " response
    echo
    case "$response" in
      y|Y|"") return 0 ;;
      n|N) return 1 ;;
    esac
  done
}

confirm_required_yes_no() {
  local prompt="$1"
  local response
  local styled_prompt

  styled_prompt="$(style_prompt_text "$prompt")"

  while true; do
    printf '%b' "$styled_prompt [yes/no]: " >&2
    read -r response
    case "${response,,}" in
      yes) return 0 ;;
      no) return 1 ;;
      *)
        echo "Please type 'yes' or 'no'." >&2
        ;;
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

prompt_choice() {
  local prompt="$1"
  local default="$2"
  shift 2
  local options=("$@")
  local choice
  local index=1
  local default_index=""
  local count="${#options[@]}"

  for option in "${options[@]}"; do
    if [[ "$option" == "$default" ]]; then
      default_index="$index"
    fi
    index=$((index + 1))
  done

  while true; do
    printf '%s\n' "${COLOR_BLUE}${prompt}${COLOR_RESET} options:" >&2
    index=1
    for option in "${options[@]}"; do
      if [[ "$index" == "$default_index" ]]; then
        printf '  %s) %s%s%s\n' \
          "${COLOR_GREEN}${index}${COLOR_RESET}" \
          "${COLOR_YELLOW}" \
          "$option" \
          "${COLOR_RESET}" >&2
      else
        printf '  %s) %s\n' "${COLOR_GREEN}${index}${COLOR_RESET}" "$option" >&2
      fi
      index=$((index + 1))
    done
    if [[ -n "$default_index" ]]; then
      printf 'Enter number (default: %s): ' "$default_index" >&2
    else
      printf 'Enter number: ' >&2
    fi

    if ((count <= 9)); then
      read -r -n 1 choice
      printf '\n' >&2
    else
      read -r choice
    fi

    if [[ -z "$choice" ]]; then
      if [[ -n "$default_index" ]]; then
        printf '%s' "${options[default_index-1]}"
        return 0
      fi
    elif [[ "$choice" =~ ^[0-9]+$ ]] && ((choice >= 1 && choice <= count)); then
      printf '%s' "${options[choice-1]}"
      return 0
    fi

    printf '%s\n' "${COLOR_YELLOW}Invalid selection.${COLOR_RESET} Please enter a number between 1 and ${count}." >&2
  done
}
