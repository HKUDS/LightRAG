# Memory Connector Makefile
# As per user's answer: Go 1.21, GitHub Releases + Docker + install script

# Variables
BINARY_NAME=memory-connector
BUILD_DIR=bin
CMD_DIR=cmd/memory-connector
GO=go
GOFLAGS=-v
VERSION?=0.1.0
GIT_COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE=$(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# Build flags
LDFLAGS=-ldflags "-X main.Version=$(VERSION) -X main.GitCommit=$(GIT_COMMIT) -X main.BuildDate=$(BUILD_DATE)"

# Default target
.PHONY: all
all: clean deps build

# Install dependencies
.PHONY: deps
deps:
	@echo "Installing dependencies..."
	$(GO) mod download
	$(GO) mod tidy

# Build the binary
.PHONY: build
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) $(CMD_DIR)/main.go

# Build for all platforms (as per user's answer: multi-platform distribution)
.PHONY: build-all
build-all:
	@echo "Building for all platforms..."
	@mkdir -p $(BUILD_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 $(CMD_DIR)/main.go
	GOOS=linux GOARCH=arm64 $(GO) build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-arm64 $(CMD_DIR)/main.go
	GOOS=darwin GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64 $(CMD_DIR)/main.go
	GOOS=darwin GOARCH=arm64 $(GO) build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 $(CMD_DIR)/main.go
	GOOS=windows GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-windows-amd64.exe $(CMD_DIR)/main.go

# Run tests (as per user's answer: 90%+ coverage target)
.PHONY: test
test:
	@echo "Running tests..."
	$(GO) test -v -race -cover ./...

# Run tests with coverage
.PHONY: test-coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(GO) test -v -race -coverprofile=coverage.out -covermode=atomic ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Run linter
.PHONY: lint
lint:
	@echo "Running linter..."
	@which golangci-lint > /dev/null || (echo "Installing golangci-lint..." && go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest)
	golangci-lint run ./...

# Format code
.PHONY: fmt
fmt:
	@echo "Formatting code..."
	$(GO) fmt ./...
	@which goimports > /dev/null || (echo "Installing goimports..." && go install golang.org/x/tools/cmd/goimports@latest)
	goimports -w .

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning..."
	rm -rf $(BUILD_DIR)
	rm -f coverage.out coverage.html

# Install the binary
.PHONY: install
install: build
	@echo "Installing $(BINARY_NAME)..."
	cp $(BUILD_DIR)/$(BINARY_NAME) /usr/local/bin/$(BINARY_NAME)

# Uninstall the binary
.PHONY: uninstall
uninstall:
	@echo "Uninstalling $(BINARY_NAME)..."
	rm -f /usr/local/bin/$(BINARY_NAME)

# Run the binary
.PHONY: run
run: build
	@echo "Running $(BINARY_NAME)..."
	./$(BUILD_DIR)/$(BINARY_NAME)

# Docker build (as per user's answer: Docker distribution)
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t memory-connector:$(VERSION) -t memory-connector:latest .

# Docker run
.PHONY: docker-run
docker-run:
	docker run --rm -v $(PWD)/configs:/app/configs -v $(PWD)/data:/app/data memory-connector:latest

# Generate install script (as per user's answer: install script distribution)
.PHONY: generate-install-script
generate-install-script:
	@echo "Generating install script..."
	@mkdir -p scripts
	@cat > scripts/install.sh <<'EOF'
	#!/bin/bash
	# Memory Connector Installation Script
	set -e
	VERSION="$(VERSION)"
	BINARY_NAME="memory-connector"
	INSTALL_DIR="/usr/local/bin"
	OS=$$(uname -s | tr '[:upper:]' '[:lower:]')
	ARCH=$$(uname -m)
	if [ "$$ARCH" = "x86_64" ]; then ARCH="amd64"; fi
	if [ "$$ARCH" = "aarch64" ]; then ARCH="arm64"; fi
	DOWNLOAD_URL="https://github.com/kamir/memory-connector/releases/download/v$$VERSION/$$BINARY_NAME-$$OS-$$ARCH"
	if [ "$$OS" = "windows" ]; then DOWNLOAD_URL="$$DOWNLOAD_URL.exe"; fi
	echo "Installing $$BINARY_NAME v$$VERSION for $$OS-$$ARCH..."
	curl -fsSL $$DOWNLOAD_URL -o /tmp/$$BINARY_NAME
	chmod +x /tmp/$$BINARY_NAME
	sudo mv /tmp/$$BINARY_NAME $$INSTALL_DIR/$$BINARY_NAME
	echo "$$BINARY_NAME installed successfully to $$INSTALL_DIR/$$BINARY_NAME"
	$$BINARY_NAME version
	EOF
	@chmod +x scripts/install.sh
	@echo "Install script generated: scripts/install.sh"

# Development mode - watch for changes and rebuild
.PHONY: dev
dev:
	@which air > /dev/null || (echo "Installing air..." && go install github.com/cosmtrek/air@latest)
	air

# Help
.PHONY: help
help:
	@echo "Memory Connector Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all                    - Clean, install deps, and build (default)"
	@echo "  deps                   - Install Go dependencies"
	@echo "  build                  - Build binary for current platform"
	@echo "  build-all              - Build binaries for all platforms"
	@echo "  test                   - Run tests"
	@echo "  test-coverage          - Run tests with coverage report"
	@echo "  lint                   - Run linter"
	@echo "  fmt                    - Format code"
	@echo "  clean                  - Remove build artifacts"
	@echo "  install                - Install binary to /usr/local/bin"
	@echo "  uninstall              - Remove binary from /usr/local/bin"
	@echo "  run                    - Build and run the binary"
	@echo "  docker-build           - Build Docker image"
	@echo "  docker-run             - Run Docker container"
	@echo "  generate-install-script - Generate installation script"
	@echo "  dev                    - Run in development mode with auto-reload"
	@echo "  help                   - Show this help message"
