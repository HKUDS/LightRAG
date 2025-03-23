# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app

# 添加镜像源并安装依赖
RUN mkdir -p /etc/apt && \
    echo "deb https://mirrors.ustc.edu.cn/debian bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/debian bullseye-updates main contrib non-free" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env

# Copy only requirements files first to leverage Docker cache
COPY requirements.txt .
COPY lightrag/api/requirements.txt ./lightrag/api/

# Install dependencies
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir -r lightrag/api/requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY ./lightrag ./lightrag
COPY setup.py .

RUN pip install .
# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Docker data directories
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

# Expose the default port
EXPOSE 9621

# Set entrypoint
ENTRYPOINT ["python", "-m", "lightrag.api.lightrag_server"]
