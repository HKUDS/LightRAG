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

.PHONY: help helm-package helm-push clean

help:
	@echo "Available targets:"
	@echo "  helm-package         - Package the LightRAG Helm chart (version: $(CHART_VERSION))"
	@echo "  helm-push            - Package and push the chart to $(HELM_REGISTRY)"
	@echo "  clean                - Remove packaged charts from $(CHART_PACKAGE_DIR)"
	@echo "\nSet VERSION=1.2.3 to override the git-derived chart version."

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
