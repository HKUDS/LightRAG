#!/bin/bash

# Clean LightRAG Development Stack
# This script completely removes all containers, volumes, and processes
# WARNING: This will delete all data in the development environment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${RED}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║     ⚠️  Cleaning LightRAG Development Stack (ALL DATA WILL BE LOST)  ║${NC}"
echo -e "${RED}╚══════════════════════════════════════════════════════════════════════╝${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Confirm action
echo -e "\n${YELLOW}This will:${NC}"
echo -e "  • Stop all running services"
echo -e "  • Remove all Docker containers"
echo -e "  • Remove all Docker volumes (DATABASE DATA WILL BE DELETED)"
echo -e "  • Kill all background processes"
echo -e "  • Clear log files"

read -p "$(echo -e ${YELLOW}Are you sure? Type 'yes' to confirm: ${NC})" -n 3 -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
  echo -e "${YELLOW}Cancelled.${NC}"
  exit 1
fi

echo -e "\n${RED}Starting cleanup...${NC}\n"

# Step 1: Kill processes
echo -e "${YELLOW}[1/5]${NC} Killing background processes..."

if [ -f /tmp/lightrag-api.pid ]; then
  API_PID=$(cat /tmp/lightrag-api.pid)
  if ps -p $API_PID > /dev/null 2>&1; then
    kill -9 $API_PID 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} API Server killed"
  fi
  rm -f /tmp/lightrag-api.pid
fi

if [ -f /tmp/lightrag-webui.pid ]; then
  WEBUI_PID=$(cat /tmp/lightrag-webui.pid)
  if ps -p $WEBUI_PID > /dev/null 2>&1; then
    kill -9 $WEBUI_PID 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} WebUI Server killed"
  fi
  rm -f /tmp/lightrag-webui.pid
fi

# Force kill by pattern as fallback
pkill -9 -f "lightrag.api.lightrag_server" 2>/dev/null || true
pkill -9 -f "npm run dev" 2>/dev/null || true

echo -e "${GREEN}✓${NC} All background processes terminated"

# Step 2: Stop Docker containers
echo -e "\n${YELLOW}[2/5]${NC} Stopping Docker containers..."
cd "$PROJECT_ROOT"
docker-compose -f docker-compose.test-db.yml down > /dev/null 2>&1 || true
echo -e "${GREEN}✓${NC} Docker containers stopped"

# Step 3: Remove Docker volumes
echo -e "\n${YELLOW}[3/5]${NC} Removing Docker volumes..."
docker volume rm lightrag_postgres_audit_data 2>/dev/null || true
echo -e "${GREEN}✓${NC} Docker volumes removed"

# Step 4: Clean log files
echo -e "\n${YELLOW}[4/5]${NC} Cleaning log files..."
rm -f /tmp/lightrag-api.log
rm -f /tmp/lightrag-webui.log
rm -f /tmp/lightrag-api.pid
rm -f /tmp/lightrag-webui.pid
echo -e "${GREEN}✓${NC} Log files cleaned"

# Step 5: Clean local storage
echo -e "\n${YELLOW}[5/5]${NC} Cleaning local storage..."
rm -rf "$PROJECT_ROOT/rag_storage/lightrag_multitenant" 2>/dev/null || true
rm -rf "$PROJECT_ROOT/rag_storage/default" 2>/dev/null || true
mkdir -p "$PROJECT_ROOT/rag_storage"
echo -e "${GREEN}✓${NC} Local storage cleaned"

# Final status
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   ✅ Cleanup Completed Successfully!                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  • Run 'bash scripts/start-dev-stack.sh' to restart with fresh data"

echo ""
