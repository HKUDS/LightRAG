SHELL := /bin/bash
SETUP_SCRIPT := scripts/setup/setup.sh
SETUP_BASH ?= $(or $(firstword $(wildcard /opt/homebrew/bin/bash /usr/local/bin/bash /opt/local/bin/bash)),$(shell command -v bash 2>/dev/null),bash)
SETUP_OPTS ?=
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_BLUE := \033[34m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m

ifeq ($(NO_COLOR),1)
COLOR_RESET :=
COLOR_BOLD :=
COLOR_BLUE :=
COLOR_GREEN :=
COLOR_YELLOW :=
endif

.PHONY: help dev configure env-base env-storage env-server env-validate env-backup env-security-check env-base-rewrite env-storage-rewrite env base storage server validate backup security security-check base-rewrite storage-rewrite

help:
	@printf "$(COLOR_BOLD)Interactive setup targets$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)make dev$(COLOR_RESET)                    Bootstrap local dev+test+offline env with uv + bun\n"
	@printf "  $(COLOR_GREEN)make env-base$(COLOR_RESET)               Configure LLM, embedding, and reranker (run first)\n"
	@printf "  $(COLOR_GREEN)make env-storage$(COLOR_RESET)            Configure storage backends and databases\n"
	@printf "  $(COLOR_GREEN)make env-server$(COLOR_RESET)             Configure server, security, and SSL\n"
	@printf "  $(COLOR_GREEN)make env-validate$(COLOR_RESET)           Validate existing .env\n"
	@printf "  $(COLOR_GREEN)make env-security-check$(COLOR_RESET)     Audit existing .env for security risks\n"
	@printf "  $(COLOR_GREEN)make env-backup$(COLOR_RESET)             Backup current .env\n"
	@printf "  $(COLOR_GREEN)make env-base-rewrite$(COLOR_RESET)       Force-regenerate wizard-managed compose services during base setup\n"
	@printf "  $(COLOR_GREEN)make env-storage-rewrite$(COLOR_RESET)    Force-regenerate wizard-managed compose services during storage setup\n"
	@printf "  $(COLOR_GREEN)make base$(COLOR_RESET)                   Short form of make env-base (all env prefix can be stripped)\n"
	@printf "\n"
	@printf "$(COLOR_BOLD)Typical workflow$(COLOR_RESET)\n"
	@printf "  1. make dev            # install backend/test deps and build frontend\n"
	@printf "  2. make env-base       # set LLM/embedding/reranker\n"
	@printf "  3. make env-storage    # set storage backends (optional)\n"
	@printf "  4. make env-server     # set port/security/SSL (optional)\n\n"
	@printf "$(COLOR_BOLD)Examples$(COLOR_RESET)\n"
	@printf "  make dev\n"
	@printf "  make env-base\n"
	@printf "  make env-storage SETUP_OPTS=--debug\n"
	@printf "  make env-server\n\n"
	@printf "  make env-storage-rewrite\n\n"
	@printf "  make env-security-check\n\n"
	@printf "$(COLOR_BOLD)Compose Output$(COLOR_RESET)\n"
	@printf "  Bundled service images are defined in scripts/setup/templates/*.yml.\n"
	@printf "  Compose file output: docker-compose.final.yml\n"

dev:
	@if ! command -v uv >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)uv is required for make dev.$(COLOR_RESET)\n"; \
		printf "Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n"; \
		printf "Unix/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh\n"; \
		printf "Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n"; \
		exit 1; \
	fi
	@if ! command -v bun >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)bun is required for make dev.$(COLOR_RESET)\n"; \
		printf "Install Bun first: https://bun.sh/docs/installation\n"; \
		printf "macOS/Linux: curl -fsSL https://bun.sh/install | bash\n"; \
		printf "Windows: powershell -c \"irm bun.sh/install.ps1 | iex\"\n"; \
		exit 1; \
	fi
	@printf "$(COLOR_BLUE)Syncing backend and test dependencies with uv...$(COLOR_RESET)\n"
	@uv sync --extra test --extra offline
	@printf "$(COLOR_BLUE)Installing frontend dependencies with Bun...$(COLOR_RESET)\n"
	@cd lightrag_webui && bun install --frozen-lockfile
	@printf "$(COLOR_BLUE)Building frontend assets...$(COLOR_RESET)\n"
	@cd lightrag_webui && bun run build
	@printf "$(COLOR_GREEN)Development environment is ready.$(COLOR_RESET)\n"
	@printf "Next steps:\n"
	@printf "  source .venv/bin/activate\n"
	@printf "  make env-base\n"
	@printf "  lightrag-server\n"

env-base env base configure:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --base $(SETUP_OPTS)

env-storage storage:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --storage $(SETUP_OPTS)

env-base-rewrite base-rewrite:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --base --rewrite-compose $(SETUP_OPTS)

env-storage-rewrite storage-rewrite:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --storage --rewrite-compose $(SETUP_OPTS)

env-server server:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --server $(SETUP_OPTS)

env-validate validate:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --validate $(SETUP_OPTS)

env-security-check security security-check:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --security-check $(SETUP_OPTS)

env-backup backup:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --backup $(SETUP_OPTS)
