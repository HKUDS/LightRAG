#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load test environment (LLM, embedding configs)
if [ ! -f "$SCRIPT_DIR/.env-test" ]; then
    echo "Error: $SCRIPT_DIR/.env-test not found. Create it with LLM/embedding config."
    exit 1
fi
set -a
source "$SCRIPT_DIR/.env-test"
set +a

# Working directory
export WORKING_DIR="/tmp/lightrag_ws_test"
rm -rf "$WORKING_DIR"
mkdir -p "$WORKING_DIR"

echo "Starting LightRAG server on port 9621..."
echo "Working dir: $WORKING_DIR"
python -m lightrag.api.lightrag_server --port 9621 --working-dir "$WORKING_DIR"
