#!/bin/bash

# LightRAG server startup script for local deployment
# Usage: ./run.sh [IP] [PORT]
#   IP: optional IP address to bind server to (default: 0.0.0.0)
#   PORT: optional port number (default: 9621 for API, 5173 for Web UI)

set -e

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading configuration from .env..."
    source .env
fi

# Default values
HOST="${1:-0.0.0.0}"
API_PORT="${2:-9621}"
WEBUI_PORT=$((API_PORT + 1))

# Use nvm to load the correct Node.js version
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh"
    nvm use 20
fi

# Create necessary directories
mkdir -p data/rag_storage data/inputs logs

# Stop any existing servers
echo "Stopping any existing LightRAG server..."
pkill -f "lightrag-server" || true
pkill -f "node.*vite" || true

# Clean up old files
rm -f dev.log

# Start LightRAG server in background
echo "Starting LightRAG server on ${HOST}:${API_PORT}..."
lightrag-server --host "${HOST}" --port "${API_PORT}" > logs/server.log 2>&1 &
SERVER_PID=$!

# Wait for server to initialize
echo "Waiting for LightRAG server to start..."
sleep 5

# Start Web UI in background
echo "Starting Web UI on ${HOST}:${WEBUI_PORT}..."
cd lightrag_webui
nohup npm run dev > ../dev.log 2>&1 &
WEBUI_PID=$!

# Wait for web UI to start
sleep 5

# Check if servers are running
sleep 3

echo ""
echo "========================================"
echo "LightRAG Server Started!"
echo "========================================"
echo "API Server:   http://${HOST}:${API_PORT}"
echo "API Docs:     http://${HOST}:${API_PORT}/docs"
echo "Web UI:       http://${HOST}:${WEBUI_PORT}/webui/"
echo "========================================"
echo ""
echo "Process IDs:"
echo "  LightRAG API:  $SERVER_PID"
echo "  Web UI:        $WEBUI_PID"
echo ""
echo "Logs:"
echo "  Server Log:    ./logs/server.log"
echo "  Web UI Log:    ./dev.log"
echo ""
echo "To stop servers: ./stop.sh"
echo "========================================"

# Save PIDs to file for stop.sh
echo "${SERVER_PID}" > .pids
echo "${WEBUI_PID}" >> .pids

exit 0