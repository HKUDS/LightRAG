#!/bin/bash
# ============================================================================
# LightRAG Hybrid Development Stack - Stop Script
# ============================================================================
# Stops all development services started by dev-start.sh
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.dev-db.yml"

# PID files
API_PID_FILE="/tmp/lightrag-dev-api.pid"
WEBUI_PID_FILE="/tmp/lightrag-dev-webui.pid"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${CYAN}${BOLD}🛑 Stopping LightRAG Development Stack${NC}                                  ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop WebUI
echo -e "  ${CYAN}▶${NC} Stopping WebUI..."
if [ -f "$WEBUI_PID_FILE" ]; then
    WEBUI_PID=$(cat "$WEBUI_PID_FILE")
    if kill -0 "$WEBUI_PID" 2>/dev/null; then
        kill "$WEBUI_PID" 2>/dev/null || true
        # Also kill any child processes (vite spawns children)
        pkill -P "$WEBUI_PID" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} WebUI stopped (PID: $WEBUI_PID)"
    else
        echo -e "  ${YELLOW}⚠${NC} WebUI process not running"
    fi
    rm -f "$WEBUI_PID_FILE"
else
    echo -e "  ${YELLOW}⚠${NC} No WebUI PID file found"
fi

# Also kill any remaining vite/bun processes on port 5173
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

# Stop API Server
echo -e "  ${CYAN}▶${NC} Stopping API Server..."
if [ -f "$API_PID_FILE" ]; then
    API_PID=$(cat "$API_PID_FILE")
    if kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} API Server stopped (PID: $API_PID)"
    else
        echo -e "  ${YELLOW}⚠${NC} API Server process not running"
    fi
    rm -f "$API_PID_FILE"
else
    echo -e "  ${YELLOW}⚠${NC} No API Server PID file found"
fi

# Also kill any remaining python processes on port 9621
lsof -ti:9621 | xargs kill -9 2>/dev/null || true

# Stop Docker containers
echo -e "  ${CYAN}▶${NC} Stopping Docker containers (PostgreSQL + Redis)..."
if [ -f "$COMPOSE_FILE" ]; then
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Docker containers stopped"
else
    echo -e "  ${YELLOW}⚠${NC} Docker compose file not found, attempting manual cleanup..."
    docker stop lightrag-dev-postgres lightrag-dev-redis 2>/dev/null || true
    docker rm lightrag-dev-postgres lightrag-dev-redis 2>/dev/null || true
fi

# Clean up log files (optional)
echo -e "  ${CYAN}▶${NC} Cleaning up temporary files..."
rm -f /tmp/lightrag-dev-api.log /tmp/lightrag-dev-webui.log 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Temporary files cleaned"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${BOLD}✅ All services stopped successfully${NC}                                     ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}To start again:${NC} ${YELLOW}./dev-start.sh${NC}  or  ${YELLOW}make dev${NC}"
echo ""
