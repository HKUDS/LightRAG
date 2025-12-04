# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Upgrade pip、setuptools and wheel to the latest version
RUN pip install --upgrade pip setuptools wheel

# Install Rust and required build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env

# Copy pyproject.toml and source code for dependency installation
COPY pyproject.toml .
COPY setup.py .
COPY lightrag/ ./lightrag/

# Install dependencies
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --user --no-cache-dir --use-pep517 .
RUN pip install --user --no-cache-dir --use-pep517 .[api]

# Install depndencies for default storage
RUN pip install --user --no-cache-dir nano-vectordb networkx
# Install depndencies for default LLM
RUN pip install --user --no-cache-dir openai ollama tiktoken
# Install depndencies for default document loader
RUN pip install --user --no-cache-dir pypdf2 python-docx python-pptx openpyxl

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY ./lightrag ./lightrag
COPY setup.py .
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Copy scripts directory for demo tenant initialization
COPY scripts/ /app/scripts/

RUN pip install --use-pep517 ".[api]"
# Make scripts executable
RUN chmod +x /app/docker-entrypoint.sh && \
    find /app/scripts -name "*.py" -type f -exec chmod +x {} \;

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Docker data directories
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

# Expose the default port
EXPOSE 9621

# Set entrypoint to our script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
