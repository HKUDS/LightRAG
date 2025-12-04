#!/bin/bash

# Start LightRAG Development Stack
# This script starts all services: PostgreSQL, Redis, API Server, and WebUI

set -e

# Configuration
LLM_MODEL="mistral-nemo:latest"
EMBEDDING_MODEL="bge-m3:latest"
EMBEDDING_DIM=1024
OLLAMA_HOST="http://localhost:11434"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Cleanup function for error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo -e "\n${RED}Script failed with exit code $exit_code${NC}"
        echo -e "${YELLOW}Cleaning up started background processes...${NC}"
        if [ -n "$API_PID" ]; then kill $API_PID 2>/dev/null || true; fi
        if [ -n "$WEBUI_PID" ]; then kill $WEBUI_PID 2>/dev/null || true; fi
    fi
    exit $exit_code
}

# Trap errors
trap cleanup ERR SIGINT SIGTERM

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🚀 Starting LightRAG Development Stack                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"

# -----------------------------------------------------------------------------
# 1. Prerequisites Checks
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/6]${NC} Checking prerequisites..."

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: '$1' is not installed or not in PATH.${NC}"
        exit 1
    fi
}

check_command docker
check_command python
check_command npm
check_command curl

# Check if Ollama is running
if ! curl -s "$OLLAMA_HOST/api/tags" > /dev/null; then
    echo -e "${RED}Error: Ollama is not running at $OLLAMA_HOST${NC}"
    echo -e "Please start Ollama and try again."
    exit 1
fi
echo -e "${GREEN}✓${NC} Ollama is running"

# -----------------------------------------------------------------------------
# 2. Model Validation
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6]${NC} Checking AI Models..."

ensure_model() {
    local model=$1
    echo -n "   Checking model '$model'... "
    if curl -s "$OLLAMA_HOST/api/tags" | grep -q "\"$model\""; then
        echo -e "${GREEN}Found${NC}"
    else
        echo -e "${YELLOW}Missing${NC}"
        echo -e "   ${BLUE}Attempting to pull '$model'...${NC}"
        if command -v ollama &> /dev/null; then
            ollama pull "$model"
        else
            echo -e "${RED}Error: 'ollama' CLI not found. Cannot pull model automatically.${NC}"
            echo -e "Please run: ${YELLOW}ollama pull $model${NC}"
            exit 1
        fi
    fi
}

ensure_model "$LLM_MODEL"
ensure_model "$EMBEDDING_MODEL"

# -----------------------------------------------------------------------------
# 3. Start Infrastructure (Docker)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6]${NC} Starting Docker containers (PostgreSQL + Redis)..."

# Change to project root
cd "$PROJECT_ROOT"

# Check ports
check_port() {
    local port=$1
    local name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}Warning: Port $port ($name) is already in use.${NC}"
    fi
}

check_port 5433 "PostgreSQL"
check_port 6380 "Redis"

docker-compose -f docker-compose.test-db.yml up -d --build
echo -e "${GREEN}✓${NC} Docker containers started"

# -----------------------------------------------------------------------------
# 4. Wait for Database
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6]${NC} Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
  if docker exec lightrag-audit-postgres pg_isready -U lightrag -d lightrag_audit > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} PostgreSQL is ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo -e "${RED}✗${NC} PostgreSQL failed to start after 30 seconds"
    exit 1
  fi
  sleep 1
done

# -----------------------------------------------------------------------------
# 5. Start API Server
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6]${NC} Starting LightRAG API Server on port 9621..."
check_port 9621 "API Server"

# Export environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_USER=lightrag
export POSTGRES_PASSWORD=lightrag123
export POSTGRES_DATABASE=lightrag_audit
export REDIS_HOST=localhost
export REDIS_PORT=6380
export LIGHTRAG_MULTI_TENANT_STRICT=true
export LIGHTRAG_REQUIRE_USER_AUTH=true
export AUTH_USER=admin
export AUTH_PASS=admin123
export LLM_BINDING=ollama
export LLM_BINDING_HOST=$OLLAMA_HOST
export LLM_MODEL=$LLM_MODEL
export EMBEDDING_BINDING=ollama
export EMBEDDING_BINDING_HOST=$OLLAMA_HOST
export EMBEDDING_MODEL=$EMBEDDING_MODEL
export EMBEDDING_DIM=$EMBEDDING_DIM

# Start API server in background
python -m lightrag.api.lightrag_server --host 0.0.0.0 --port 9621 > /tmp/lightrag-api.log 2>&1 &
API_PID=$!
echo $API_PID > /tmp/lightrag-api.pid

echo -e "${GREEN}✓${NC} API Server started (PID: $API_PID)"

# Wait for API
echo -e "\n   Waiting for API to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:9621/health > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓${NC} API is ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo -e "   ${RED}✗${NC} API failed to start after 30 seconds"
    echo -e "${YELLOW}Last 20 lines of API log:${NC}"
    tail -n 20 /tmp/lightrag-api.log
    exit 1
  fi
  sleep 1
done

# -----------------------------------------------------------------------------
# 6. Start WebUI
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6]${NC} Starting WebUI dev server on port 5173..."
cd "$PROJECT_ROOT/lightrag_webui"
check_port 5173 "WebUI"

export VITE_API_BASE_URL=http://localhost:9621
npm run dev > /tmp/lightrag-webui.log 2>&1 &
WEBUI_PID=$!
echo $WEBUI_PID > /tmp/lightrag-webui.pid

echo -e "${GREEN}✓${NC} WebUI Server started (PID: $WEBUI_PID)"

# Wait for WebUI
echo -e "\n   Waiting for WebUI to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:5173/ > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓${NC} WebUI is ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo -e "   ${YELLOW}⚠${NC}  WebUI may take longer to start, continuing anyway..."
    break
  fi
  sleep 1
done

# -----------------------------------------------------------------------------
# Final Status
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   ✅ Stack Started Successfully!                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${GREEN}Services Running:${NC}"
echo -e "  • PostgreSQL:  localhost:5433 (lightrag_audit)"
echo -e "  • Redis:       localhost:6380"
echo -e "  • API:         ${BLUE}http://localhost:9621${NC}"
echo -e "  • WebUI:       ${BLUE}http://localhost:5173${NC}"

echo -e "\n${GREEN}Configuration:${NC}"
echo -e "  • LLM Model:       $LLM_MODEL"
echo -e "  • Embedding Model: $EMBEDDING_MODEL ($EMBEDDING_DIM dim)"

echo -e "\n${GREEN}Process IDs:${NC}"
echo -e "  • API Server:  $API_PID"
echo -e "  • WebUI:       $WEBUI_PID"

echo -e "\n${YELLOW}To view logs:${NC}"
echo -e "  • API:   tail -f /tmp/lightrag-api.log"
echo -e "  • WebUI: tail -f /tmp/lightrag-webui.log"

echo -e "\n${YELLOW}To stop the stack:${NC}"
echo -e "  • Run: bash scripts/stop-dev-stack.sh"

echo -e "\n${RED}NOTE:${NC} If you see 'dimension mismatch' errors, run: ${YELLOW}bash scripts/clean-dev-stack.sh${NC}"

echo ""
