# Makefile for LightRAG Helm packaging

# ------------------------------------------------------------------------------
# Core chart configuration (mirrors upstream LightRAG project)
# ------------------------------------------------------------------------------
CHART_NAME := lightrag-minimal
CHART_DIR := k8s-deploy/$(CHART_NAME)
CHART_PACKAGE_DIR := dist/charts
HELM_REGISTRY := ghcr.io/neuro-inc/helm-charts

RAW_VERSION := $(if $(VERSION),$(VERSION),$(shell git describe --tags --always --dirty 2>/dev/null))
SANITIZED_VERSION := $(shell python -c 'import re; raw = "$(RAW_VERSION)".strip(); raw = raw[1:] if raw.startswith("v") else raw; raw = raw or "0.0.0"; print(re.sub(r"[^0-9A-Za-z.\-]", "-", raw) or "0.0.0")')
CHART_VERSION := $(SANITIZED_VERSION)
CHART_PACKAGE := $(CHART_PACKAGE_DIR)/$(CHART_NAME)-$(CHART_VERSION).tgz

GITHUB_USERNAME := $(shell echo "$$APOLO_GITHUB_TOKEN" | base64 -d 2>/dev/null | cut -d: -f1 2>/dev/null || echo "oauth2")

# ------------------------------------------------------------------------------
# Apolo tooling configuration
# ------------------------------------------------------------------------------
POETRY ?= poetry
IMAGE_NAME ?= app-lightrag
IMAGE_TAG ?= latest
HOOKS_IMAGE_TARGET ?= ghcr.io/neuro-inc/lightrag

APP_CHART_NAME := lightrag
APP_CHART_DIR := k8s-deploy/$(APP_CHART_NAME)
APP_CHART_PACKAGE := $(CHART_PACKAGE_DIR)/$(APP_CHART_NAME)-$(CHART_VERSION).tgz

define HELP_TEXT
Available targets:
  install              - Install Poetry environment and pre-commit hooks
  lint                 - Run pre-commit across the repository
  test-unit            - Execute unit tests for Apolo integrations
  gen-types-schemas    - Regenerate LightRAG JSON schemas
  helm-package         - Package the LightRAG Helm chart (version: $(CHART_VERSION))
  helm-push            - Package and push the minimal chart to $(HELM_REGISTRY)
  helm-package-app     - Package the full LightRAG app chart
  helm-push-app        - Package and push the full app chart
  build-hook-image     - Build the hooks helper image
  push-hook-image      - Push the hooks helper image
  clean                - Remove packaged charts from $(CHART_PACKAGE_DIR)

Set VERSION=1.2.3 to override the git-derived chart version.
endef
export HELP_TEXT

# ------------------------------------------------------------------------------
# Phony targets
# ------------------------------------------------------------------------------
.PHONY: all help test clean
.PHONY: install setup lint format test-unit gen-types-schemas
.PHONY: build-hook-image push-hook-image helm-package helm-push helm-package-app helm-push-app

# ------------------------------------------------------------------------------
# User-facing helpers
# ------------------------------------------------------------------------------
all: help

help:
	@printf '%s\n' "$$HELP_TEXT"

install setup:
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) install --with dev
	$(POETRY) run pre-commit install

lint format:
ifdef CI
	$(POETRY) run pre-commit run --all-files --show-diff-on-failure
else
	$(POETRY) run pre-commit run --all-files || $(POETRY) run pre-commit run --all-files
endif

test-unit:
	$(POETRY) run pytest -vvs --cov=.apolo --cov-report xml:.coverage.unit.xml .apolo/tests/unit

test: test-unit

gen-types-schemas:
	@.apolo/scripts/gen_types_schemas.sh

build-hook-image:
	docker build \
		-t $(IMAGE_NAME):latest \
		-f hooks.Dockerfile \
		.

push-hook-image: build-hook-image
	docker tag $(IMAGE_NAME):latest $(HOOKS_IMAGE_TARGET):$(IMAGE_TAG)
	docker push $(HOOKS_IMAGE_TARGET):$(IMAGE_TAG)

# ------------------------------------------------------------------------------
# Upstream Helm packaging targets (kept minimal to ease syncing with source)
# ------------------------------------------------------------------------------
helm-package:
	@if [ -z "$(CHART_VERSION)" ]; then \
		echo "Error: unable to determine chart version."; \
		exit 1; \
	fi
	@echo "Packaging $(CHART_NAME) chart version $(CHART_VERSION)..."
	@mkdir -p $(CHART_PACKAGE_DIR)
	helm dependency update $(CHART_DIR) >/dev/null
	helm package $(CHART_DIR) \
		--version $(CHART_VERSION) \
		--app-version $(CHART_VERSION) \
		-d $(CHART_PACKAGE_DIR)
	@echo "✅ Chart packaged at $(CHART_PACKAGE)"

helm-push: helm-package
	@if [ -z "$(APOLO_GITHUB_TOKEN)" ]; then \
		echo "Error: APOLO_GITHUB_TOKEN not set. Please export a token with write:packages."; \
		exit 1; \
	fi
	@echo "Logging into Helm registry ghcr.io as $(GITHUB_USERNAME)..."
	echo "$(APOLO_GITHUB_TOKEN)" | helm registry login ghcr.io -u $(GITHUB_USERNAME) --password-stdin >/dev/null
	@echo "Pushing chart $(CHART_NAME):$(CHART_VERSION) to $(HELM_REGISTRY)..."
	helm push $(CHART_PACKAGE) oci://$(HELM_REGISTRY)
	@echo "✅ Chart pushed to $(HELM_REGISTRY)"

clean:
	@echo "Removing packaged charts..."
	rm -rf $(CHART_PACKAGE_DIR)
	@echo "✅ Cleaned"

helm-package-app:
	@if [ -z "$(CHART_VERSION)" ]; then \
		echo "Error: unable to determine chart version."; \
		exit 1; \
	fi
	@echo "Packaging $(APP_CHART_NAME) chart version $(CHART_VERSION)..."
	@mkdir -p $(CHART_PACKAGE_DIR)
	helm dependency update $(APP_CHART_DIR) >/dev/null
	helm package $(APP_CHART_DIR) \
		--version $(CHART_VERSION) \
		--app-version $(CHART_VERSION) \
		-d $(CHART_PACKAGE_DIR)
	@echo "✅ Chart packaged at $(APP_CHART_PACKAGE)"

helm-push-app: helm-package-app
	@if [ -z "$(APOLO_GITHUB_TOKEN)" ]; then \
		echo "Error: APOLO_GITHUB_TOKEN not set. Please export a token with write:packages."; \
		exit 1; \
	fi
	@echo "Logging into Helm registry ghcr.io as $(GITHUB_USERNAME)..."
	echo "$(APOLO_GITHUB_TOKEN)" | helm registry login ghcr.io -u $(GITHUB_USERNAME) --password-stdin >/dev/null
	@echo "Pushing chart $(APP_CHART_NAME):$(CHART_VERSION) to $(HELM_REGISTRY)..."
	helm push $(APP_CHART_PACKAGE) oci://$(HELM_REGISTRY)
	@echo "✅ Chart pushed to $(HELM_REGISTRY)"
