# Makefile for LightRAG Helm packaging

# Configuration
CHART_NAME := lightrag-minimal
CHART_DIR := k8s-deploy/$(CHART_NAME)
CHART_PACKAGE_DIR := dist/charts
HELM_REGISTRY := ghcr.io/neuro-inc/helm-charts

RAW_VERSION := $(if $(VERSION),$(VERSION),$(shell git describe --tags --always --dirty 2>/dev/null))
SANITIZED_VERSION := $(shell python -c 'import re; raw = "$(RAW_VERSION)".strip(); raw = raw[1:] if raw.startswith("v") else raw; raw = raw or "0.0.0"; sanitized = re.sub(r"[^0-9A-Za-z.\-]", "-", raw); print(sanitized or "0.0.0")')
CHART_VERSION := $(SANITIZED_VERSION)
CHART_PACKAGE := $(CHART_PACKAGE_DIR)/$(CHART_NAME)-$(CHART_VERSION).tgz

GITHUB_USERNAME := $(shell echo "$$APOLO_GITHUB_TOKEN" | base64 -d 2>/dev/null | cut -d: -f1 2>/dev/null || echo "oauth2")

HOOKS_IMAGE_REPO ?= ghcr.io/neuro-inc/app-lightrag
BUILD_IMAGE_TAG ?= $(CHART_VERSION)
IMAGE_TAG ?= $(BUILD_IMAGE_TAG)
HOOKS_BUILD_IMAGE := $(HOOKS_IMAGE_REPO):$(BUILD_IMAGE_TAG)
HOOKS_PUBLISH_IMAGE := $(HOOKS_IMAGE_REPO):$(IMAGE_TAG)

define HELP_MESSAGE
Available targets:
  helm-package         - Package the LightRAG Helm chart (version: $(CHART_VERSION))
  helm-push            - Package and push the chart to $(HELM_REGISTRY)
  clean                - Remove packaged charts from $(CHART_PACKAGE_DIR)
  hooks-build          - Build the pre-commit hooks image $(HOOKS_BUILD_IMAGE)
  hooks-publish        - Build and push the hooks image to its registry

Set VERSION=1.2.3 to override the git-derived chart version.
endef
export HELP_MESSAGE

.PHONY: all help helm-package helm-push clean test hooks-build hooks-publish build-hook-image push-hook-image

all: help

help:
	@printf "%s\n" "$$HELP_MESSAGE"

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

test:
	@echo "No automated tests for Helm packaging. Use 'helm test' as needed."

hooks-build:
	@echo "Building hooks image $(HOOKS_BUILD_IMAGE)..."
	docker build \
		--file hooks.Dockerfile \
		--tag $(HOOKS_BUILD_IMAGE) \
		.
	@echo "✅ Hooks image built: $(HOOKS_BUILD_IMAGE)"

hooks-publish: hooks-build
	@echo "Tagging hooks image as $(HOOKS_PUBLISH_IMAGE)..."
	@if [ "$(HOOKS_PUBLISH_IMAGE)" != "$(HOOKS_BUILD_IMAGE)" ]; then \
		docker tag $(HOOKS_BUILD_IMAGE) $(HOOKS_PUBLISH_IMAGE); \
	fi
	@echo "Pushing hooks image $(HOOKS_PUBLISH_IMAGE)..."
	docker push $(HOOKS_PUBLISH_IMAGE)
	@echo "✅ Hooks image pushed to $(HOOKS_PUBLISH_IMAGE)"

build-hook-image: hooks-build

push-hook-image: hooks-publish
