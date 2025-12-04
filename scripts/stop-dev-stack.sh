#!/bin/bash

# Stop LightRAG Development Stack
# This script gracefully stops all services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🛑 Stopping LightRAG Development Stack                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Stop API Server
echo -e "\n${YELLOW}[1/3]${NC} Stopping API Server..."
if [ -f /tmp/lightrag-api.pid ]; then
  API_PID=$(cat /tmp/lightrag-api.pid)
  if ps -p $API_PID > /dev/null 2>&1; then
    kill $API_PID 2>/dev/null || true
    sleep 1
    if ps -p $API_PID > /dev/null 2>&1; then
      kill -9 $API_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✓${NC} API Server stopped (PID: $API_PID)"
  else
    echo -e "${YELLOW}⚠${NC}  API Server not running"
  fi
  rm -f /tmp/lightrag-api.pid
else
  # Fallback: Kill by process name/pattern
  pkill -f "lightrag.api.lightrag_server" || true
  echo -e "${GREEN}✓${NC} API Server stopped (by pattern)"
fi

# Stop WebUI
echo -e "\n${YELLOW}[2/3]${NC} Stopping WebUI Server..."
if [ -f /tmp/lightrag-webui.pid ]; then
  WEBUI_PID=$(cat /tmp/lightrag-webui.pid)
  if ps -p $WEBUI_PID > /dev/null 2>&1; then
    kill $WEBUI_PID 2>/dev/null || true
    sleep 1
    if ps -p $WEBUI_PID > /dev/null 2>&1; then
      kill -9 $WEBUI_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✓${NC} WebUI Server stopped (PID: $WEBUI_PID)"
  else
    echo -e "${YELLOW}⚠${NC}  WebUI Server not running"
  fi
  rm -f /tmp/lightrag-webui.pid
else
  # Fallback: Kill by process name/pattern
  pkill -f "npm run dev" || true
  echo -e "${GREEN}✓${NC} WebUI Server stopped (by pattern)"
fi

# Stop Docker containers
echo -e "\n${YELLOW}[3/3]${NC} Stopping Docker containers..."
cd "$PROJECT_ROOT"
docker-compose -f docker-compose.test-db.yml down

echo -e "${GREEN}✓${NC} Docker containers stopped"

# Final status
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   ✅ Stack Stopped Successfully!                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${YELLOW}Note:${NC} Data is persisted in Docker volumes"
echo -e "        Use 'clean-dev-stack.sh' to remove all data and containers"

echo ""
