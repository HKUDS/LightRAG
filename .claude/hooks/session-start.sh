#!/bin/bash
# SessionStart hook: install backend + WebUI dependencies so that
# tests and linters work out of the box in Claude Code on the web.
set -euo pipefail

# Only run in the remote (Claude Code on the web) environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# --- Python backend (uv) ---
# `uv sync` is idempotent and reuses the cached .venv across sessions.
if command -v uv >/dev/null 2>&1; then
  echo "[session-start] Syncing Python dependencies (api, test extras)..."
  uv sync --extra api --extra test
fi

# --- WebUI frontend (bun) ---
if command -v bun >/dev/null 2>&1 && [ -f lightrag_webui/package.json ]; then
  echo "[session-start] Installing WebUI dependencies..."
  (cd lightrag_webui && bun install --frozen-lockfile)
fi

echo "[session-start] Dependencies ready."
