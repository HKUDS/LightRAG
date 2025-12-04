# ============================================================================
# LightRAG Development Makefile
# ============================================================================
# Quick commands for local development
# Uses .env file at project root for configuration
# ============================================================================

.PHONY: help dev dev-start dev-stop dev-status dev-logs dev-logs-api dev-logs-webui \
        db-only db-stop db-shell db-logs clean-db install test lint

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
CYAN := \033[0;36m
RED := \033[0;31m
BOLD := \033[1m
DIM := \033[2m
NC := \033[0m

# ============================================================================
# HELP - Default target
# ============================================================================
help:
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║$(NC)  $(CYAN)$(BOLD)⚡ LightRAG Development Commands$(NC)                                        $(BLUE)║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🚀 Quick Start:$(NC)"
	@echo "  $(YELLOW)make dev$(NC)              Start full dev stack (Docker DBs + local API/WebUI)"
	@echo "    $(YELLOW)make dev CONFIRM_KILL=yes$(NC)  Auto-kill processes/containers occupying dev ports (use carefully)"
	@echo ""
	@echo "$(GREEN)$(BOLD)📋 Development Commands:$(NC)"
	@echo "  $(YELLOW)make dev$(NC)              Start everything (PostgreSQL, Redis, API, WebUI)"
	@echo "  $(YELLOW)make dev-stop$(NC)         Stop all development services"
	@echo "  $(YELLOW)make dev-status$(NC)       Check status of all services"
	@echo "  $(YELLOW)make dev-logs$(NC)         View all logs (API + WebUI)"
	@echo "  $(YELLOW)make dev-logs-api$(NC)     View API server logs only"
	@echo "  $(YELLOW)make dev-logs-webui$(NC)   View WebUI logs only"
	@echo ""
	@echo "$(GREEN)$(BOLD)🗄️  Database Only (Docker):$(NC)"
	@echo "  $(YELLOW)make db-only$(NC)          Start only PostgreSQL + Redis in Docker"
	@echo "  $(YELLOW)make db-stop$(NC)          Stop PostgreSQL + Redis containers"
	@echo "  $(YELLOW)make db-shell$(NC)         Connect to PostgreSQL shell"
	@echo "  $(YELLOW)make db-logs$(NC)          View database container logs"
	@echo "  $(YELLOW)make clean-db$(NC)         Remove database volumes (⚠️ deletes data!)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🔧 Setup & Utilities:$(NC)"
	@echo "  $(YELLOW)make install$(NC)          Install Python + WebUI dependencies"
	@echo "  $(YELLOW)make test$(NC)             Run tests"
	@echo "  $(YELLOW)make lint$(NC)             Run linters"
	@echo ""
	@echo "$(GREEN)$(BOLD)📡 Service URLs (when running):$(NC)"
	@echo "  • WebUI:        $(BLUE)http://localhost:5173$(NC)"
	@echo "  • API Server:   $(BLUE)http://localhost:9621$(NC)"
	@echo "  • API Docs:     $(BLUE)http://localhost:9621/docs$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🔐 Default Credentials:$(NC)"
	@echo "  • Username:     $(YELLOW)admin$(NC)"
	@echo "  • Password:     $(YELLOW)admin123$(NC)"
	@echo ""
	@echo "$(DIM)Configuration: .env at project root$(NC)"
	@echo ""

# ============================================================================
# DEVELOPMENT - Full Stack
# ============================================================================

## Start full development stack
dev: dev-start

## Start development stack (alias)
dev-start:
	@chmod +x ./dev-start.sh
	@if [ "$(CONFIRM_KILL)" = "yes" ] || [ "$(CONFIRM_KILL)" = "true" ]; then \
		./dev-start.sh --yes ; \
	else \
		./dev-start.sh ; \
	fi

## Stop development stack
dev-stop:
	@chmod +x ./dev-stop.sh
	@./dev-stop.sh

## Check status of services
dev-status:
	@chmod +x ./dev-status.sh
	@./dev-status.sh

## View all logs
dev-logs:
	@echo "$(CYAN)$(BOLD)📄 API Server Logs:$(NC)"
	@tail -n 50 /tmp/lightrag-dev-api.log 2>/dev/null || echo "  No API logs found"
	@echo ""
	@echo "$(CYAN)$(BOLD)📄 WebUI Logs:$(NC)"
	@tail -n 50 /tmp/lightrag-dev-webui.log 2>/dev/null || echo "  No WebUI logs found"

## View API logs
dev-logs-api:
	@tail -f /tmp/lightrag-dev-api.log

## View WebUI logs
dev-logs-webui:
	@tail -f /tmp/lightrag-dev-webui.log

# ============================================================================
# DATABASE ONLY - Docker
# ============================================================================

## Start only database services (PostgreSQL + Redis)
db-only:
	@echo "$(CYAN)$(BOLD)🗄️  Starting PostgreSQL + Redis...$(NC)"
	@docker compose -f docker-compose.dev-db.yml --env-file .env up -d
	@echo ""
	@echo "$(GREEN)✓ Database services started$(NC)"
	@echo ""
	@echo "  PostgreSQL: localhost:$${POSTGRES_PORT:-15432}"
	@echo "  Redis:      localhost:$${REDIS_PORT:-16379}"
	@echo ""
	@echo "$(DIM)To start the full stack, run: make dev$(NC)"

## Stop database services
db-stop:
	@echo "$(CYAN)$(BOLD)🛑 Stopping PostgreSQL + Redis...$(NC)"
	@docker compose -f docker-compose.dev-db.yml down
	@echo "$(GREEN)✓ Database services stopped$(NC)"

## Connect to PostgreSQL shell
db-shell:
	@docker exec -it lightrag-dev-postgres psql -U $${POSTGRES_USER:-lightrag} -d $${POSTGRES_DATABASE:-lightrag_multitenant}

## View database logs
db-logs:
	@docker compose -f docker-compose.dev-db.yml logs -f

## Remove database volumes (WARNING: deletes all data!)
clean-db:
	@echo "$(RED)$(BOLD)⚠️  WARNING: This will delete all database data!$(NC)"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	@docker compose -f docker-compose.dev-db.yml down -v
	@echo "$(GREEN)✓ Database volumes removed$(NC)"

# ============================================================================
# SETUP & UTILITIES
# ============================================================================

## Install all dependencies
install:
	@echo "$(CYAN)$(BOLD)📦 Installing Python dependencies...$(NC)"
	@pip install -e ".[dev]" || pip install -e .
	@echo ""
	@echo "$(CYAN)$(BOLD)📦 Installing WebUI dependencies...$(NC)"
	@cd lightrag_webui && (bun install 2>/dev/null || npm install)
	@echo ""
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

## Run tests
test:
	@echo "$(CYAN)$(BOLD)🧪 Running tests...$(NC)"
	@python -m pytest tests/ -v

## Run linters
lint:
	@echo "$(CYAN)$(BOLD)🔍 Running linters...$(NC)"
	@python -m ruff check lightrag/ || true
	@cd lightrag_webui && (bun run lint 2>/dev/null || npm run lint) || true
