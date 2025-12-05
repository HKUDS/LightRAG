#!/bin/bash
# ============================================================================
# LightRAG Development Stack - Status Check
# ============================================================================
# Shows status of all development services
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Load environment
if [ -f "$ENV_FILE" ]; then
    set -a
    source <(grep -v '^\s*#' "$ENV_FILE" | grep -v '^\s*$' | sed 's/\r$//')
    set +a
fi

# Extract Redis port from REDIS_URI
if [ -n "$REDIS_URI" ]; then
    REDIS_PORT=$(echo "$REDIS_URI" | sed -n 's/.*:\([0-9]*\)$/\1/p')
fi
REDIS_PORT=${REDIS_PORT:-16379}
POSTGRES_PORT=${POSTGRES_PORT:-15432}
API_PORT=${PORT:-9621}
WEBUI_PORT=5173

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${CYAN}${BOLD}📊 LightRAG Development Stack Status${NC}                                    ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

check_service() {
    local name=$1
    local port=$2
    local type=$3

    if [ "$type" = "docker" ]; then
        local container=$4
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            echo -e "  ${GREEN}●${NC} ${BOLD}$name${NC} ${DIM}(Docker)${NC}"
            echo -e "    └─ Running on port $port"
            return 0
        else
            echo -e "  ${RED}○${NC} ${BOLD}$name${NC} ${DIM}(Docker)${NC}"
            echo -e "    └─ Not running"
            return 1
        fi
    else
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo -e "  ${GREEN}●${NC} ${BOLD}$name${NC} ${DIM}(Native)${NC}"
            echo -e "    └─ Running on port $port"
            return 0
        else
            echo -e "  ${RED}○${NC} ${BOLD}$name${NC} ${DIM}(Native)${NC}"
            echo -e "    └─ Not running"
            return 1
        fi
    fi
}

echo -e "${CYAN}${BOLD}🗄️  Database Services:${NC}"
check_service "PostgreSQL" "$POSTGRES_PORT" "docker" "lightrag-dev-postgres"
check_service "Redis" "$REDIS_PORT" "docker" "lightrag-dev-redis"

echo ""
echo -e "${CYAN}${BOLD}🖥️  Application Services:${NC}"
check_service "API Server" "$API_PORT" "native"
check_service "WebUI" "$WEBUI_PORT" "native"

echo ""
echo -e "${CYAN}${BOLD}🔗 Quick Links:${NC}"
if lsof -Pi :$WEBUI_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "  • WebUI:      ${BLUE}http://localhost:$WEBUI_PORT${NC}"
fi
if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "  • API:        ${BLUE}http://localhost:$API_PORT${NC}"
    echo -e "  • API Docs:   ${BLUE}http://localhost:$API_PORT/docs${NC}"
fi

echo ""
