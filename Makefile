# Development commands (npm-style)
# Usage: make <command>

.PHONY: lint format typecheck check fix test clean dev

# Lint code with ruff
lint:
	uv run ruff check .

# Format code with ruff
format:
	uv run ruff format .

# Type-check with ty
typecheck:
	uv run ty check .

# Run all checks (lint + format check + typecheck)
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check .

# Fix linting issues and format code
fix:
	uv run ruff check . --fix
	uv run ruff format .

# Run tests
test:
	uv run pytest

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Start development server
dev:
	uv run lightrag-server

# Install dependencies
install:
	uv sync --extra api --extra test --extra lint

# Update lockfile
lock:
	uv lock --upgrade
