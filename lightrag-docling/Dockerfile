# LightRAG Docling Service Dockerfile
# Multi-stage build for optimized container size

# ============================================================================
# Build Stage - Install dependencies and build artifacts
# ============================================================================
FROM python:3.11-slim-bullseye as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Production Stage - Minimal runtime image
# ============================================================================
FROM python:3.11-slim-bullseye

# Set build labels
LABEL maintainer="LightRAG Team"
LABEL version="${VERSION}"
LABEL build-date="${BUILD_DATE}"
LABEL vcs-ref="${VCS_REF}"
LABEL description="LightRAG Docling Document Processing Service"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Required for docling ML models
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Required for PDF processing
    poppler-utils \
    # Required for health checks
    curl \
    # Required for process monitoring
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r docling && useradd -r -g docling -u 10001 docling

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/docling/.local

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY __init__.py .

# Create necessary directories and set permissions
RUN mkdir -p /app/cache /app/logs /app/tmp \
    && chown -R docling:docling /app \
    && chmod -R 755 /app

# Set environment variables
ENV PYTHONPATH="/app:/home/docling/.local/lib/python3.11/site-packages"
ENV PATH="/home/docling/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Docling service specific environment variables
ENV DOCLING_HOST=0.0.0.0
ENV DOCLING_PORT=8080
ENV DOCLING_CACHE_DIR=/app/cache
ENV DOCLING_LOG_LEVEL=INFO
ENV DOCLING_WORKERS=1

# Switch to non-root user
USER docling:docling

# Expose service port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["python", "-m", "uvicorn", "src.docling_service:app"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--workers", "1"]