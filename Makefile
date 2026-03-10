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

.PHONY: help env-base env-storage env-server env-validate env-backup

help:
	@printf "$(COLOR_BOLD)Interactive setup targets$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)make env-base$(COLOR_RESET)         Configure LLM, embedding, and reranker (run first)\n"
	@printf "  $(COLOR_GREEN)make env-storage$(COLOR_RESET)      Configure storage backends and databases\n"
	@printf "  $(COLOR_GREEN)make env-server$(COLOR_RESET)       Configure server, security, and SSL\n"
	@printf "  $(COLOR_GREEN)make env-validate$(COLOR_RESET)     Validate existing .env\n"
	@printf "  $(COLOR_GREEN)make env-backup$(COLOR_RESET)       Backup current .env\n\n"
	@printf "$(COLOR_BOLD)Typical workflow$(COLOR_RESET)\n"
	@printf "  1. make env-base       # set LLM/embedding/reranker\n"
	@printf "  2. make env-storage    # set storage backends (optional)\n"
	@printf "  3. make env-server     # set port/security/SSL (optional)\n\n"
	@printf "$(COLOR_BOLD)Examples$(COLOR_RESET)\n"
	@printf "  make env-base\n"
	@printf "  make env-storage SETUP_OPTS=--debug\n"
	@printf "  make env-server\n\n"
	@printf "$(COLOR_BOLD)Compose Output$(COLOR_RESET)\n"
	@printf "  Bundled service images are defined in scripts/setup/templates/*.yml.\n"
	@printf "  Compose file output: docker-compose.final.yml\n"

configure:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --base $(SETUP_OPTS)

env-base:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --base $(SETUP_OPTS)

env-storage:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --storage $(SETUP_OPTS)

env-server:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --server $(SETUP_OPTS)

env-validate:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --validate $(SETUP_OPTS)

env-backup:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --backup $(SETUP_OPTS)
