# syntax=docker/dockerfile:1

# ==========================================
# 1. Frontend Build Stage
# ==========================================
FROM oven/bun:1 AS frontend-builder
WORKDIR /app
COPY lightrag_webui/ ./lightrag_webui/
RUN --mount=type=cache,target=/root/.bun/install/cache \
    cd lightrag_webui \
    && bun install --frozen-lockfile \
    && bun run build

# ==========================================
# 2. Python Builder Stage
# ==========================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# 安裝必要工具，加入 dos2unix 用於處理 Windows 換行符號問題
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl build-essential pkg-config git dos2unix \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"
RUN mkdir -p /root/.local/share/uv

# 複製依賴定義
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# 安裝依賴
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --no-install-project --no-editable

# 複製源代碼
COPY lightrag/ ./lightrag/
COPY --from=frontend-builder /app/lightrag/api/webui ./lightrag/api/webui

# 再次 Sync 確保環境完整
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --no-editable \
    && /app/.venv/bin/python -m ensurepip --upgrade

# 下載 Tiktoken Cache
RUN mkdir -p /app/data/tiktoken \
    && uv run lightrag-download-cache --cache-dir /app/data/tiktoken || status=$?; \
    if [ -n "${status:-}" ] && [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi

# ==========================================
# 3. Final Runtime Stage
# ==========================================
FROM python:3.12-slim

WORKDIR /app

# Runtime 安裝 dos2unix 確保萬無一失
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr \
    git git-lfs dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1

# 複製構建好的環境和代碼
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/lightrag ./lightrag
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

ENV PATH=/app/.venv/bin:/root/.local/bin:$PATH

# 建立數據目錄
RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/data/tiktoken \
    && chmod -R 777 /app/data

COPY --from=builder /app/data/tiktoken /app/data/tiktoken

ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

EXPOSE 9621

# --- 修復重點區 ---
COPY entrypoint.sh /app/entrypoint.sh
# 1. 使用 dos2unix 強制將 CRLF 轉為 LF
# 2. 賦予執行權限
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]