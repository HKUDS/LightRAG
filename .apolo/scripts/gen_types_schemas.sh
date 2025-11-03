#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PACKAGE_DIR=".apolo/src/apolo_apps_lightrag"

INPUT_SCHEMA="${APP_PACKAGE_DIR}/schemas/LightRAGAppInputs.json"
OUTPUT_SCHEMA="${APP_PACKAGE_DIR}/schemas/LightRAGAppOutputs.json"

if command -v poetry >/dev/null 2>&1; then
    APP_TYPES_CMD=(poetry run app-types)
elif [[ -x "${REPO_ROOT}/.venv/bin/app-types" ]]; then
    APP_TYPES_CMD=("${REPO_ROOT}/.venv/bin/app-types")
elif command -v app-types >/dev/null 2>&1; then
    APP_TYPES_CMD=(app-types)
else
    echo "app-types CLI not found. Install dependencies via 'poetry install --with dev'." >&2
    exit 1
fi

(
    cd "${REPO_ROOT}"
    "${APP_TYPES_CMD[@]}" dump-types-schema "${APP_PACKAGE_DIR}" LightRAGAppInputs "${INPUT_SCHEMA}"
    "${APP_TYPES_CMD[@]}" dump-types-schema "${APP_PACKAGE_DIR}" LightRAGAppOutputs "${OUTPUT_SCHEMA}"
)
