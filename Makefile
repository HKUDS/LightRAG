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

.PHONY: help setup configure setup-quick setup-quick-vllm setup-production setup-validate setup-backup setup-help

help:
	@printf "$(COLOR_BOLD)Interactive setup targets$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)make setup$(COLOR_RESET)            Full wizard (development/production/custom)\n"
	@printf "  $(COLOR_GREEN)make setup-quick$(COLOR_RESET)      Development preset, minimal prompts\n"
	@printf "  $(COLOR_GREEN)make setup-quick-vllm$(COLOR_RESET) Development preset + local vLLM embedding + optional reranker\n"
	@printf "  $(COLOR_GREEN)make setup-production$(COLOR_RESET) Production preset + SSL/security prompts\n"
	@printf "  $(COLOR_GREEN)make setup-validate$(COLOR_RESET)   Validate existing .env\n"
	@printf "  $(COLOR_GREEN)make setup-backup$(COLOR_RESET)     Backup current .env\n"
	@printf "  $(COLOR_GREEN)make setup-help$(COLOR_RESET)       Script help and options\n\n"
	@printf "$(COLOR_BOLD)Install types$(COLOR_RESET)\n"
	@printf "  $(COLOR_BLUE)development$(COLOR_RESET): local JSON/NetworkX defaults, fastest to start\n"
	@printf "  $(COLOR_BLUE)production$(COLOR_RESET): database-backed defaults, security prompts, docker services optional\n"
	@printf "  $(COLOR_BLUE)custom$(COLOR_RESET): pick each storage backend manually\n\n"
	@printf "$(COLOR_BOLD)Examples$(COLOR_RESET)\n"
	@printf "  make setup\n"
	@printf "  make setup-quick\n"
	@printf "  make setup-production SETUP_OPTS=--debug\n\n"
	@printf "$(COLOR_BOLD)Image Settings$(COLOR_RESET)\n"
	@printf "  Wizard will show image settings for selected services and let you override them.\n"
	@printf "  You can also edit POSTGRES_IMAGE, NEO4J_IMAGE_TAG, etc. in .env.\n"
	@printf "  Compose file output: docker-compose.<development|production|custom>.yml\n"

setup:
	@$(SETUP_BASH) $(SETUP_SCRIPT) $(SETUP_OPTS)

configure:
	@$(SETUP_BASH) $(SETUP_SCRIPT) $(SETUP_OPTS)

setup-quick:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --quick $(SETUP_OPTS)

setup-quick-vllm:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --quick-vllm $(SETUP_OPTS)

setup-production:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --production $(SETUP_OPTS)

setup-validate:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --validate $(SETUP_OPTS)

setup-backup:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --backup $(SETUP_OPTS)

setup-help:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --help $(SETUP_OPTS)
