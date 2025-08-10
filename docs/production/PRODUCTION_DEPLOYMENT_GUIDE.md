# 🚀 LightRAG Production Deployment Guide

> **📘 This document has been consolidated into a comprehensive production guide.**
> **Please use: [Complete Production Deployment Guide](PRODUCTION_DEPLOYMENT_COMPLETE.md)**

**Status**: 🚨 **DEPRECATED** - Redirects to consolidated guide
**Replacement**: [PRODUCTION_DEPLOYMENT_COMPLETE.md](PRODUCTION_DEPLOYMENT_COMPLETE.md)
**Action Required**: Update bookmarks and references to use the new consolidated guide.

This document previously provided comprehensive production deployment instructions but has been merged with other production guides to create a single authoritative source. The new consolidated guide includes all the information from this document plus additional deployment scenarios and enhanced troubleshooting.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Configuration](#detailed-configuration)
- [Enhanced Document Processing](#enhanced-document-processing)
- [MCP Server Setup](#mcp-server-setup)
- [Security Setup](#security-setup)
- [Monitoring & Observability](#monitoring--observability)
- [Backup & Disaster Recovery](#backup--disaster-recovery)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## 🎯 Overview

This production deployment includes:

- **🔐 Enterprise Authentication**: Phase 1 security features with bcrypt, rate limiting, audit logging
- **🐳 Container Orchestration**: Docker Compose with multi-service architecture
- **🤖 MCP Integration**: Model Context Protocol server for Claude CLI integration
- **📄 Enhanced Document Processing**: Docling service with OCR, table recognition, and figure extraction
- **📊 Monitoring Stack**: Prometheus, Grafana, Jaeger tracing
- **🔄 Load Balancing**: Nginx reverse proxy with SSL termination
- **💾 Data Persistence**: PostgreSQL with pgvector, Redis caching
- **📝 Comprehensive Logging**: Structured logging with log aggregation
- **🔄 Automated Backups**: Database and data backups with cloud storage
- **🚨 Health Monitoring**: Multi-tier health checks and alerting

## ✅ Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 100GB+ SSD storage
- **Network**: Static IP with domain name

### Software Requirements

- Docker 24.0+
- Docker Compose 2.20+
- Git 2.30+
- SSL certificates (Let's Encrypt or commercial)

### External Services

- **LLM Provider**: OpenAI, Anthropic, or compatible API
- **Email Service**: SMTP for notifications (optional)
- **Cloud Storage**: AWS S3, Google Cloud, or compatible (optional)

## ⚡ Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/Ajith-82/LightRAG.git
cd LightRAG

# Copy production environment template
cp production.env .env

# Create necessary directories
mkdir -p data/{rag_storage,inputs,backups} logs certs
```

### 2. Configure Environment

Edit `.env` file with your settings:

```bash
# Core Settings
NODE_ENV=production
DEBUG=false

# Database (Required)
POSTGRES_HOST=postgres
POSTGRES_USER=lightrag_prod
POSTGRES_PASSWORD=your-secure-password-here
POSTGRES_DATABASE=lightrag_production

# LLM Configuration (Required)
LLM_BINDING=openai
LLM_API_KEY=your-openai-api-key
EMBEDDING_API_KEY=your-openai-api-key

# Security (Recommended)
JWT_SECRET_KEY=your-super-secure-jwt-secret
BCRYPT_ROUNDS=12
RATE_LIMIT_ENABLED=true

# Monitoring (Optional)
GRAFANA_ADMIN_PASSWORD=your-grafana-password
```

### 3. Deploy

```bash
# Start production environment
docker compose -f docker-compose.production.yml up -d

# Optional: Start with enhanced document processing (includes Docling service)
docker compose -f docker-compose.production.yml -f docker-compose.enhanced.yml --profile enhanced-processing up -d

# Check status
docker compose -f docker-compose.production.yml ps

# View logs
docker compose -f docker-compose.production.yml logs -f lightrag
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost/health

# API test
curl -X POST http://localhost/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, world!", "mode": "naive"}'
```

## ⚙️ Detailed Configuration

### Environment Configuration

The production environment supports 200+ configuration options across:

#### Core Application
```bash
# Server Configuration
HOST=0.0.0.0
PORT=9621
WORKERS=4
WORKER_TIMEOUT=300

# Performance
MAX_PARALLEL_INSERT=4
CHUNK_TOKEN_SIZE=1200
LLM_MAX_ASYNC=4
```

#### Security (Phase 1 Authentication)
```bash
# Authentication
AUTH_ENABLED=true
JWT_SECRET_KEY=your-jwt-secret
JWT_EXPIRATION_HOURS=24

# Password Security
BCRYPT_ROUNDS=12
PASSWORD_MIN_LENGTH=12
PASSWORD_LOCKOUT_ATTEMPTS=5

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BLOCK_DURATION=3600

# Security Headers
SECURITY_HEADERS_ENABLED=true
CSP_DEFAULT_SRC='self'
HSTS_MAX_AGE=31536000

# Audit Logging
AUDIT_LOGGING_ENABLED=true
AUDIT_LOG_LEVEL=INFO
AUDIT_ENABLE_ANALYTICS=true
```

#### Database & Storage
```bash
# PostgreSQL
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=30
POSTGRES_SSLMODE=require

# Redis
REDIS_POOL_SIZE=20
REDIS_SOCKET_KEEPALIVE=true

# Storage Backends
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
```

## 📄 Enhanced Document Processing

### Overview

LightRAG includes an optional **Enhanced Document Processing** service powered by Docling, providing advanced OCR, table recognition, and figure extraction capabilities. This significantly improves document processing quality compared to basic text extraction.

### Features

#### Basic Document Processing (Default)
- Simple text extraction using PyPDF2, python-docx
- Fast processing (< 1 second)
- Limited accuracy for complex documents
- No OCR capability

#### Enhanced Document Processing (Optional)
- **🔍 OCR (Optical Character Recognition)**: Extract text from images and scanned documents
- **📊 Table Recognition**: Preserve table structure and relationships
- **🖼️ Figure Extraction**: Identify and describe images, charts, diagrams
- **📐 Layout Understanding**: Recognize headers, paragraphs, multi-column layouts
- **📋 Structured Metadata**: Enhanced document structure understanding
- **💾 Intelligent Caching**: ML model caching for performance optimization

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LightRAG      │───▶│  Docling Service │───▶│  ML Models      │
│   Main App      │    │  (Port 8080)     │    │  (EasyOCR, etc) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Fallback       │    │  Cache & Logs    │
│  Processors     │    │                  │
└─────────────────┘    └──────────────────┘
```

### Configuration

#### Environment Variables

Add these to your `.env` file to enable enhanced processing:

```bash
# Enhanced Document Processing
LIGHTRAG_ENHANCED_PROCESSING=true
DOCLING_SERVICE_MODE=auto

# Docling Service Configuration
DOCLING_SERVICE_URL=http://docling-service:8080
DOCLING_SERVICE_TIMEOUT=300
DOCLING_SERVICE_RETRIES=3
DOCLING_FALLBACK_ENABLED=true

# Docling Processing Settings
DOCLING_DEFAULT_EXPORT_FORMAT=markdown
DOCLING_DEFAULT_MAX_WORKERS=3
DOCLING_DEFAULT_ENABLE_OCR=true
DOCLING_DEFAULT_ENABLE_TABLE_STRUCTURE=true
DOCLING_DEFAULT_ENABLE_FIGURES=true

# Docling Cache Settings
DOCLING_CACHE_ENABLED=true
DOCLING_CACHE_MAX_SIZE_GB=5
DOCLING_CACHE_TTL_HOURS=168

# Docling Resource Limits
DOCLING_MAX_FILE_SIZE_MB=100
DOCLING_MAX_BATCH_SIZE=10
DOCLING_REQUEST_TIMEOUT_SECONDS=300

# Security (Optional)
DOCLING_API_KEY=your-docling-api-key
DOCLING_CORS_ORIGINS="*"

# IMPORTANT: Do NOT set DOCLING_DEBUG (causes Pydantic validation errors)
```

#### Docker Compose Profiles

Enhanced processing uses Docker Compose profiles for optional deployment:

```bash
# Basic deployment (without Docling)
docker compose -f docker-compose.production.yml up -d

# Enhanced deployment (with Docling service)
docker compose -f docker-compose.production.yml -f docker-compose.enhanced.yml --profile enhanced-processing up -d
```

### Deployment

#### Method 1: Full Production Stack with Enhanced Processing

```bash
# Clone repository
git clone https://github.com/Ajith-82/LightRAG.git
cd LightRAG

# Configure environment with enhanced processing
cp production.env .env
# Edit .env to add Docling settings (see Configuration section above)

# Deploy with enhanced processing
docker compose -f docker-compose.production.yml -f docker-compose.enhanced.yml --profile enhanced-processing up -d
```

#### Method 2: Add Enhanced Processing to Existing Deployment

```bash
# Stop existing deployment
docker compose -f docker-compose.production.yml down

# Update environment configuration
# Add Docling settings to .env file

# Restart with enhanced processing
docker compose -f docker-compose.production.yml -f docker-compose.enhanced.yml --profile enhanced-processing up -d
```

### Service Verification

#### Health Checks

```bash
# Check Docling service health
curl http://localhost:8080/health

# Expected response for working service:
{
  "status": "healthy",
  "docling_available": true,
  "cache_available": true,
  "memory_usage_mb": 2500,
  "supported_formats": [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html"]
}

# If docling_available is false, check troubleshooting section
```

#### Processing Test

```bash
# Test enhanced processing via API
curl -X POST http://localhost/api/documents/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-document.pdf" \
  -F "enable_enhanced_processing=true"

# Check processing result
curl http://localhost/api/documents/list?enhanced_only=true
```

### Performance Characteristics

#### Processing Comparison

| Feature | Basic Processing | Enhanced Processing |
|---------|------------------|---------------------|
| **Speed** | ~0.2s per document | ~10-30s per document |
| **Accuracy** | Text extraction only | Full OCR + ML analysis |
| **OCR Support** | ❌ None | ✅ Full OCR |
| **Tables** | ❌ Lost structure | ✅ Structure preserved |
| **Images** | ❌ Ignored | ✅ Analyzed and described |
| **Memory Usage** | ~100MB | ~2.5GB (with models) |
| **Cache Benefits** | Minimal | Significant (80% faster on cache hit) |

#### Resource Requirements

- **CPU**: 2-4 cores (ML processing intensive)
- **Memory**: 4-6GB RAM (ML models + processing)
- **Storage**: 10-20GB for model cache
- **Network**: Internet access for initial model download

### Troubleshooting

#### Known Issue: Service Shows "degraded" Status

**Root Cause**: This issue was caused by the `DOCLING_DEBUG=false` environment variable causing a Pydantic validation error in Docling's internal configuration system.

**Solution**:
```bash
# DO NOT set DOCLING_DEBUG environment variable
# Remove any existing DOCLING_DEBUG settings

# Check service status
curl http://localhost:8080/health

# If docling_available is still false, check logs
docker logs lightrag_docling_prod
```

#### Other Common Issues

**1. Models Not Downloading**

```bash
# Check internet connectivity from container
docker exec lightrag_docling_prod curl -I https://github.com

# Check available disk space
docker exec lightrag_docling_prod df -h

# Manual model download test
docker exec lightrag_docling_prod python -c "
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
print('Models loaded successfully')
"
```

**2. High Memory Usage**

```bash
# Monitor memory usage
docker stats lightrag_docling_prod

# Reduce worker count if needed
DOCLING_DEFAULT_MAX_WORKERS=1

# Restart service
docker compose restart docling-service
```

**3. Processing Timeout**

```bash
# Increase timeout settings
DOCLING_SERVICE_TIMEOUT=600
DOCLING_REQUEST_TIMEOUT_SECONDS=600

# For large files, consider preprocessing
DOCLING_MAX_FILE_SIZE_MB=200
```

#### Performance Optimization

```bash
# Optimize for production workloads
DOCLING_DEFAULT_MAX_WORKERS=2        # Reduce for stability
DOCLING_CACHE_ENABLED=true           # Enable caching
DOCLING_CACHE_MAX_SIZE_GB=10         # Increase cache size

# Resource limits in docker-compose
deploy:
  resources:
    limits:
      memory: 6G
      cpus: '3.0'
    reservations:
      memory: 3G
      cpus: '1.5'
```

### Monitoring Enhanced Processing

#### Metrics

Enhanced processing provides additional metrics:

```bash
# Docling service metrics
curl http://localhost:8080/metrics

# Processing performance
curl http://localhost:8080/cache/stats

# Model status
curl http://localhost:8080/config
```

#### Log Analysis

```bash
# Docling service logs
docker logs lightrag_docling_prod -f

# Processing performance logs
docker logs lightrag_docling_prod | grep "processing_time"

# Cache hit rate
docker logs lightrag_docling_prod | grep "cache_hit"
```

### Quality Comparison

The enhanced processing provides significant quality improvements:

#### Before (Basic Processing)
```
Processing Time: 0.22 seconds
Word Count: 8,531
Features: Basic text extraction only
OCR: Not available
Table Recognition: Not available
Figure Extraction: Not available
```

#### After (Enhanced Processing)
```
Processing Time: ~12 seconds
Word Count: More accurate count with OCR
Features: Full OCR + ML processing
OCR: ✅ Available and active
Table Recognition: ✅ Structure preserved
Figure Extraction: ✅ Images analyzed
Metadata: Enhanced structure understanding
Cache Hit: 80% faster on repeated processing
```

### Security Considerations

#### Network Security
- Docling service runs in isolated Docker network
- Only LightRAG main service can access Docling service
- Optional API key authentication available

#### Data Privacy
- Documents processed in memory only
- No data sent to external services
- ML models run locally in container
- Optional caching can be disabled for sensitive documents

#### Resource Security
- Non-root container execution
- Read-only filesystem where possible
- Resource limits prevent resource exhaustion
- Health checks ensure service availability

### Docker Services Configuration

#### Application Service
- **Image**: Custom production-optimized image
- **Resources**: 8GB RAM, 4 CPU cores
- **Health Checks**: Multi-tier health monitoring
- **Security**: Non-root user, readonly filesystem options

#### MCP Server Service
- **Image**: Custom MCP server image
- **Resources**: 2GB RAM, 2 CPU cores
- **Features**: 11 tools, 3 resources for Claude CLI integration
- **Security**: Isolated network, API authentication

#### Database Service
- **Image**: shangor/postgres-for-rag:v1.0 (PostgreSQL + pgvector + AGE)
- **Performance**: Tuned for RAG workloads
- **Persistence**: Named volume with backup integration
- **Security**: Network isolation, credential management

#### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Dashboards and visualization
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation (optional)

## 🤖 MCP Server Setup

### Overview

The Model Context Protocol (MCP) server provides Claude CLI integration with 11 specialized tools and 3 resources for seamless LightRAG interaction. This enables natural language interaction with your RAG system through Claude.

### MCP Server Features

#### Available Tools (11)
1. **lightrag_query** - Execute RAG queries with different modes
2. **lightrag_insert_text** - Insert text documents directly
3. **lightrag_insert_file** - Upload and process files
4. **lightrag_batch_insert** - Batch document processing
5. **lightrag_list_documents** - List processed documents
6. **lightrag_get_document** - Retrieve document details
7. **lightrag_delete_document** - Remove documents
8. **lightrag_get_graph** - Export knowledge graph
9. **lightrag_search_entities** - Search graph entities
10. **lightrag_health_check** - System health monitoring
11. **lightrag_clear_cache** - Cache management

#### Available Resources (3)
1. **lightrag://system/config** - System configuration
2. **lightrag://system/stats** - Performance statistics
3. **lightrag://system/health** - Health status

### Production Configuration

#### MCP Environment Variables

Add these to your `.env` file:

```bash
# MCP Server Configuration
MCP_SERVER_ENABLED=true
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8080
MCP_SERVER_WORKERS=2

# LightRAG API Integration
LIGHTRAG_API_URL=http://lightrag:9621
LIGHTRAG_API_TIMEOUT=300
LIGHTRAG_API_RETRIES=3

# MCP Features
MCP_ENABLE_STREAMING=true
MCP_ENABLE_DOCUMENT_UPLOAD=true
MCP_ENABLE_BATCH_PROCESSING=true
MCP_ENABLE_GRAPH_OPERATIONS=true

# MCP Security
MCP_AUTH_ENABLED=true
MCP_API_KEY=your-mcp-api-key
MCP_CORS_ORIGINS=["https://claude.ai"]

# MCP Performance
MCP_MAX_CONCURRENT_REQUESTS=10
MCP_REQUEST_TIMEOUT=300
MCP_CACHE_ENABLED=true
MCP_CACHE_TTL=3600

# MCP Logging
MCP_LOG_LEVEL=INFO
MCP_LOG_FORMAT=json
MCP_ENABLE_ACCESS_LOG=true
```

### Docker Compose Integration

Add MCP server to your `docker-compose.production.yml`:

```yaml
services:
  lightrag-mcp:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    container_name: lightrag_mcp
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      # Load from .env file
      - MCP_SERVER_HOST=0.0.0.0
      - MCP_SERVER_PORT=8080
      - LIGHTRAG_API_URL=http://lightrag:9621
      - MCP_ENABLE_STREAMING=${MCP_ENABLE_STREAMING:-true}
      - MCP_AUTH_ENABLED=${MCP_AUTH_ENABLED:-true}
      - MCP_API_KEY=${MCP_API_KEY}
    networks:
      - lightrag-network
    depends_on:
      lightrag:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
    volumes:
      - ./logs:/app/logs:rw
      - /tmp:/tmp:rw
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.mcp.rule=Host(`mcp.yourdomain.com`)"
      - "traefik.http.services.mcp.loadbalancer.server.port=8080"
```

### Claude CLI Integration

#### 1. Install Claude CLI

```bash
# Install Claude CLI
npm install -g @anthropic-ai/claude-cli

# Or using pip
pip install claude-cli
```

#### 2. Configure MCP Server

```bash
# Add MCP server to Claude CLI configuration
claude config mcp add lightrag-mcp "http://localhost:8080" --api-key your-mcp-api-key

# Or using local Python module
claude config mcp add lightrag-mcp python -m lightrag_mcp
```

#### 3. Verify Integration

```bash
# Test MCP server connectivity
claude mcp lightrag_health_check

# List available tools
claude mcp --list-tools

# Test basic query
claude mcp lightrag_query "What are the main themes in my documents?" --mode hybrid
```

### MCP Server Usage Examples

#### Document Management

```bash
# Upload a document
claude mcp lightrag_insert_file "/path/to/document.pdf" --description "Product manual"

# Insert text directly
claude mcp lightrag_insert_text "Important company policy update..." --title "Policy Update"

# Batch processing
claude mcp lightrag_batch_insert --directory "/path/to/documents" --file-types "pdf,docx,txt"

# List all documents
claude mcp lightrag_list_documents --limit 20 --offset 0
```

#### Knowledge Graph Operations

```bash
# Export knowledge graph
claude mcp lightrag_get_graph --format json --max-nodes 100

# Search entities
claude mcp lightrag_search_entities "artificial intelligence" --limit 10

# Get specific entity details
claude mcp lightrag_search_entities "machine learning" --include-relationships true
```

#### Query Operations

```bash
# Local context queries
claude mcp lightrag_query "How does authentication work?" --mode local

# Global knowledge queries
claude mcp lightrag_query "What are the system requirements?" --mode global

# Hybrid queries (recommended)
claude mcp lightrag_query "Explain the deployment process" --mode hybrid

# Streaming queries
claude mcp lightrag_query "Summarize all security features" --mode hybrid --stream true
```

### MCP Monitoring

#### Health Checks

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed system status
claude mcp lightrag_health_check --detailed

# Resource monitoring
claude mcp resource "lightrag://system/stats"
```

#### Logs and Debugging

```bash
# View MCP server logs
docker logs lightrag_mcp -f

# Enable debug mode
MCP_LOG_LEVEL=DEBUG docker-compose restart lightrag-mcp

# Check resource usage
claude mcp resource "lightrag://system/config"
```

### Performance Optimization

#### MCP Server Tuning

```bash
# Adjust worker processes
MCP_SERVER_WORKERS=4

# Optimize timeouts
MCP_REQUEST_TIMEOUT=600
LIGHTRAG_API_TIMEOUT=600

# Enable caching
MCP_CACHE_ENABLED=true
MCP_CACHE_TTL=7200

# Concurrent request limits
MCP_MAX_CONCURRENT_REQUESTS=20
```

#### Connection Pooling

```bash
# HTTP connection optimization
MCP_HTTP_POOL_SIZE=20
MCP_HTTP_POOL_MAXSIZE=50
MCP_HTTP_KEEP_ALIVE=true

# Retry configuration
LIGHTRAG_API_RETRIES=5
MCP_RETRY_BACKOFF=exponential
```

### Security Configuration

#### Authentication

```bash
# Enable MCP authentication
MCP_AUTH_ENABLED=true
MCP_API_KEY=your-secure-mcp-api-key-here

# Configure CORS for Claude.ai
MCP_CORS_ORIGINS=["https://claude.ai", "https://your-domain.com"]

# Rate limiting
MCP_RATE_LIMIT_ENABLED=true
MCP_RATE_LIMIT_REQUESTS=100
MCP_RATE_LIMIT_WINDOW=60
```

#### Network Security

```bash
# Bind to specific interface
MCP_SERVER_HOST=127.0.0.1  # Local only
# or
MCP_SERVER_HOST=0.0.0.0    # All interfaces

# Use custom port
MCP_SERVER_PORT=8080

# Enable TLS (recommended for production)
MCP_TLS_ENABLED=true
MCP_TLS_CERT_PATH=/certs/mcp-cert.pem
MCP_TLS_KEY_PATH=/certs/mcp-key.pem
```

### Troubleshooting MCP Issues

#### Common Problems

1. **Connection Refused**:
```bash
# Check if MCP server is running
docker ps | grep lightrag_mcp

# Verify port binding
netstat -tlnp | grep 8080

# Check firewall
sudo ufw status | grep 8080
```

2. **Authentication Errors**:
```bash
# Verify API key
echo $MCP_API_KEY

# Test authentication
curl -H "Authorization: Bearer $MCP_API_KEY" http://localhost:8080/health
```

3. **Performance Issues**:
```bash
# Monitor resource usage
docker stats lightrag_mcp

# Check API response times
claude mcp lightrag_health_check --benchmark
```

#### Debug Configuration

```bash
# Enable verbose logging
MCP_LOG_LEVEL=DEBUG
MCP_ENABLE_ACCESS_LOG=true
MCP_LOG_FORMAT=detailed

# Restart with debug mode
docker-compose restart lightrag-mcp
```

### MCP Server Scaling

#### Horizontal Scaling

```bash
# Scale MCP server instances
docker-compose up -d --scale lightrag-mcp=3

# Load balancer configuration (Nginx)
upstream mcp_backend {
    server lightrag-mcp-1:8080;
    server lightrag-mcp-2:8080;
    server lightrag-mcp-3:8080;
}
```

#### High Availability

```yaml
# docker-compose.production.yml
services:
  lightrag-mcp:
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 10s
```

## 🔐 Security Setup

### SSL/TLS Configuration

#### Production SSL Certificates

1. **Generate SSL Certificates**:

```bash
# Using Let's Encrypt (recommended)
certbot certonly --standalone -d your-domain.com

# Copy certificates
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem certs/cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem certs/key.pem
```

#### Self-Signed SSL Certificates (Local Testing)

For local development and testing environments, you can generate self-signed certificates:

1. **Generate Self-Signed Certificate**:

```bash
# Create certificate directory
mkdir -p certs

# Generate private key
openssl genrsa -out certs/key.pem 2048

# Generate certificate signing request
openssl req -new -key certs/key.pem -out certs/cert.csr -subj "/C=US/ST=State/L=City/O=Organization/OU=IT/CN=localhost"

# Generate self-signed certificate (valid for 365 days)
openssl x509 -req -in certs/cert.csr -signkey certs/key.pem -out certs/cert.pem -days 365

# Clean up CSR file
rm certs/cert.csr

# Set proper permissions
chmod 600 certs/key.pem
chmod 644 certs/cert.pem
```

2. **Generate with Subject Alternative Names** (for multiple domains/IPs):

```bash
# Create config file for SAN certificate
cat > certs/san.conf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Organization
OU = IT Department
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
DNS.3 = lightrag.local
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate certificate with SAN
openssl req -new -x509 -key certs/key.pem -out certs/cert.pem -days 365 -config certs/san.conf -extensions v3_req

# Clean up config file
rm certs/san.conf
```

3. **Quick One-Command Generation**:

```bash
# Generate self-signed certificate in one command
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=LightRAG/CN=localhost" -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1"
```

4. **Trust Self-Signed Certificate** (Optional):

```bash
# Ubuntu/Debian
sudo cp certs/cert.pem /usr/local/share/ca-certificates/lightrag-local.crt
sudo update-ca-certificates

# CentOS/RHEL
sudo cp certs/cert.pem /etc/pki/ca-trust/source/anchors/lightrag-local.crt
sudo update-ca-trust

# macOS (if accessing from Mac)
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain certs/cert.pem
```

5. **Verify Certificate**:

```bash
# Check certificate details
openssl x509 -in certs/cert.pem -text -noout

# Test SSL connection
openssl s_client -connect localhost:443 -servername localhost < /dev/null

# Verify certificate chain
openssl verify certs/cert.pem
```

#### SSL Configuration

2. **Update Nginx Configuration**:

Edit `nginx/conf.d/lightrag.conf` to enable HTTPS section.

### Firewall Configuration

```bash
# Ubuntu/Debian
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 5432/tcp  # Block direct database access
ufw deny 6379/tcp  # Block direct Redis access
ufw enable

# CentOS/RHEL
firewall-cmd --permanent --add-service=http
firewall-cmd --permanent --add-service=https
firewall-cmd --reload
```

### Authentication Setup

The production deployment includes Phase 1 authentication features:

1. **User Registration**:
```bash
curl -X POST http://localhost/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "SecurePass123!",
    "email": "admin@example.com"
  }'
```

2. **Login**:
```bash
curl -X POST http://localhost/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "SecurePass123!"}'
```

### Security Monitoring

- **Audit Logs**: `/app/logs/audit.log`
- **Security Events**: Authentication, authorization, rate limiting
- **Anomaly Detection**: Automated analysis of security patterns
- **Failed Login Tracking**: Account lockout protection

## 📊 Monitoring & Observability

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/your-password):

1. **LightRAG Overview**: Application metrics, request rates, error rates
2. **System Resources**: CPU, memory, disk usage
3. **Database Performance**: PostgreSQL metrics, query performance
4. **Security Dashboard**: Authentication events, rate limiting, audit logs

### Prometheus Metrics

Available metrics at `http://localhost:9090`:

- **Application Metrics**: Request duration, error rates, active sessions
- **Database Metrics**: Connection pool, query performance, storage usage
- **System Metrics**: CPU, memory, disk, network usage
- **Security Metrics**: Authentication events, rate limiting hits

### Log Aggregation

Structured JSON logging with correlation IDs:

```json
{
  "timestamp": "2025-01-30T12:00:00Z",
  "level": "INFO",
  "service": "lightrag",
  "request_id": "req-123",
  "user_id": "user-456",
  "event": "query_processed",
  "duration_ms": 1234,
  "mode": "hybrid"
}
```

### Health Checks

Multi-tier health monitoring:

- **Liveness**: `/health/live` - Basic application status
- **Readiness**: `/health/ready` - Ready to serve requests
- **Deep Health**: `/health` - Full system health check

Health check includes:
- Database connectivity
- Redis connectivity
- System resources (CPU, memory, disk)
- LLM API connectivity (optional)

## 💾 Backup & Disaster Recovery

### Automated Backups

Backups run automatically via cron:

- **Database Backups**: Daily at 1:00 AM
- **Data Backups**: Daily at 2:00 AM
- **Retention**: 7 days (database), 30 days (data)

### Manual Backup

```bash
# Database backup
docker exec lightrag_backup /app/scripts/backup-database.sh

# Data backup
docker exec lightrag_backup /app/scripts/backup-data.sh
```

### Cloud Storage Integration

Configure cloud storage for off-site backups:

```bash
# AWS S3
AWS_S3_BUCKET=your-backup-bucket
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Generic S3-compatible (MinIO, etc.)
RCLONE_CONFIG=your-rclone-config
RCLONE_REMOTE=your-remote-name
```

### Disaster Recovery

1. **Database Recovery**:
```bash
# Restore from backup
gunzip -c backup.sql.gz | psql -h postgres -U lightrag_prod lightrag_production
```

2. **Data Recovery**:
```bash
# Extract data backup
tar -xzf lightrag_data_backup.tar.gz -C /app/data/
```

3. **Point-in-Time Recovery**: Use PostgreSQL WAL archiving for precise recovery.

## ⚡ Performance Tuning

### Application Tuning

```bash
# Worker Configuration
WORKERS=4                    # Number of Gunicorn workers
MAX_WORKERS=8               # Maximum workers
WORKER_TIMEOUT=300          # Request timeout

# LLM Performance
LLM_MAX_ASYNC=4             # Concurrent LLM requests
LLM_TIMEOUT=300             # LLM request timeout
EMBEDDING_MAX_ASYNC=8       # Concurrent embedding requests

# Processing Performance
MAX_PARALLEL_INSERT=4       # Concurrent document processing
CHUNK_TOKEN_SIZE=1200       # Optimal chunk size
ENTITY_EXTRACT_MAX_GLEANING=2  # Entity extraction passes
```

### Database Tuning

PostgreSQL configuration in `postgres/config/postgresql.conf`:

```ini
# Memory Settings
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 256MB
maintenance_work_mem = 1GB

# Performance Settings
max_connections = 100
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Logging
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
```

### Redis Tuning

```bash
# Memory Management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Application Won't Start

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs lightrag

# Common causes:
# - Database connection issues
# - Missing environment variables
# - Port conflicts
# - SSL certificate issues
```

#### 2. Enhanced Document Processing Issues

```bash
# Check Docling service status
curl http://localhost:8080/health

# If docling_available is false:
# - Ensure DOCLING_DEBUG is NOT set in environment variables
# - Check for Pydantic validation errors in logs
# - Verify models downloaded successfully

# Check Docling service logs
docker logs lightrag_docling_prod

# Common solutions:
# - Remove DOCLING_DEBUG environment variable
# - Increase memory allocation for ML models
# - Verify internet connectivity for model downloads
```

#### 3. Database Connection Issues

```bash
# Test database connectivity
docker exec -it lightrag_postgres psql -U lightrag_prod -d lightrag_production

# Check database logs
docker-compose -f docker-compose.production.yml logs postgres
```

#### 4. High Memory Usage

```bash
# Monitor resource usage
docker stats

# Tune worker settings
WORKERS=2                   # Reduce workers
LLM_MAX_ASYNC=2            # Reduce concurrent requests
MAX_PARALLEL_INSERT=2      # Reduce parallel processing
```

#### 5. Performance Issues

```bash
# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=lightrag_request_duration_seconds

# Database performance
SELECT * FROM pg_stat_activity;
SELECT * FROM pg_stat_user_tables;
```

### Debug Mode

For troubleshooting, enable debug mode (not for production):

```bash
DEBUG=true
LOG_LEVEL=DEBUG
VERBOSE_LOGGING=true
```

### Log Analysis

```bash
# Application logs
docker-compose -f docker-compose.production.yml logs -f lightrag

# Audit logs
docker exec lightrag_app tail -f /app/logs/audit.log

# Database logs
docker-compose -f docker-compose.production.yml logs postgres
```

## 🔄 Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor health checks and alerts
- Review error logs and security events
- Verify backup completion

#### Weekly
- Update SSL certificates if needed
- Review performance metrics
- Clean up old logs and temporary files

#### Monthly
- Update Docker images
- Review and update security configurations
- Performance optimization review
- Disaster recovery testing

### Updates and Upgrades

1. **Application Updates**:
```bash
# Pull latest image
docker-compose -f docker-compose.production.yml pull lightrag

# Restart services
docker-compose -f docker-compose.production.yml up -d
```

2. **Database Migrations**:
```bash
# Run migrations
docker exec lightrag_app python -c "
import asyncio
from lightrag.api.migrations.auth_phase1_migration import AuthPhase1Migration

async def run():
    migration = AuthPhase1Migration()
    await migration.run()

asyncio.run(run())
"
```

3. **Security Updates**:
```bash
# Update system packages
apt update && apt upgrade

# Update Docker images
docker-compose -f docker-compose.production.yml pull
```

### Scaling

#### Horizontal Scaling

```bash
# Scale application instances
docker-compose -f docker-compose.production.yml up -d --scale lightrag=3

# Load balancer configuration required
```

#### Vertical Scaling

```yaml
# Update docker-compose.production.yml
services:
  lightrag:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8.0'
```

### Monitoring and Alerting

Set up alerts for:

- **Application Health**: Service down, high error rates
- **Resource Usage**: High CPU, memory, disk usage
- **Security Events**: Failed logins, rate limiting hits
- **Database Issues**: Connection failures, slow queries
- **Backup Failures**: Failed backups, storage issues

## 📞 Support

For production support:

1. **Documentation**: Check this guide and project README
2. **Logs**: Collect relevant logs before reporting issues
3. **Health Status**: Include health check results
4. **Configuration**: Review environment configuration
5. **GitHub Issues**: Report bugs and feature requests

---

**🎯 This production deployment provides enterprise-grade reliability, security, and performance for LightRAG.**

For additional configuration options and advanced topics, see:
- [Authentication Documentation](docs/security/AUTHENTICATION_IMPROVEMENT_PLAN.md)
- [MCP Integration Guide](docs/integration_guides/MCP_IMPLEMENTATION_SUMMARY.md)
- [Performance Optimization Guide](docs/performance-optimization.md)
