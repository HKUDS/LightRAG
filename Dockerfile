# syntax=docker/dockerfile:1

# Frontend build stage
FROM oven/bun:1 AS frontend-builder

WORKDIR /app

# Copy frontend source code
COPY lightrag_webui/ ./lightrag_webui/

# Build frontend assets for inclusion in the API package
RUN --mount=type=cache,target=/root/.bun/install/cache \
    cd lightrag_webui \
    && bun install --frozen-lockfile \
    && bun run build

# Python build stage - using uv for faster package installation
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install system deps (Rust is required by some wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Ensure shared data directory exists for uv caches
RUN mkdir -p /root/.local/share/uv

# Copy project metadata and sources
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Install base, API, and offline extras without the project to improve caching
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --no-install-project --no-editable

# Copy project sources after dependency layer
COPY lightrag/ ./lightrag/

# Include pre-built frontend assets from the previous stage
COPY --from=frontend-builder /app/lightrag/api/webui ./lightrag/api/webui

# Sync project in non-editable mode and ensure pip is available for runtime installs
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --no-editable \
    && /app/.venv/bin/python -m ensurepip --upgrade

# Prepare offline cache directory and pre-populate tiktoken data
# Use uv run to execute commands from the virtual environment
RUN mkdir -p /app/data/tiktoken \
    && uv run lightrag-download-cache --cache-dir /app/data/tiktoken || status=$?; \
    if [ -n "${status:-}" ] && [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Install uv for package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

# Copy installed packages and application code
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/lightrag ./lightrag
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Ensure the installed scripts are on PATH
ENV PATH=/app/.venv/bin:/root/.local/bin:$PATH

# Install dependencies with uv sync (uses locked versions from uv.lock)
# And ensure pip is available for runtime installs
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --no-editable \
    && /app/.venv/bin/python -m ensurepip --upgrade

# Create persistent data directories AFTER package installation
# Note: /app/lightrag/prompts can be overridden via volume mount for easy prompt customization
RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/data/tiktoken /app/lightrag/prompts

# Copy offline cache into the newly created directory
COPY --from=builder /app/data/tiktoken /app/data/tiktoken

# Point to the prepared cache
ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

# Expose API port
EXPOSE 9621

ENTRYPOINT ["python", "-m", "lightrag.api.lightrag_server"]
