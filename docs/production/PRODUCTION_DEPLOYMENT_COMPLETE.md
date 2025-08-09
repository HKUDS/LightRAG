# 🚀 Complete LightRAG Production Deployment Guide

**Last Updated**: 2025-08-09
**Status**: Authoritative Production Guide
**Audience**: DevOps Engineers, System Administrators, Production Teams

> **📘 This is the single authoritative guide for production LightRAG deployment.** All other production guides in this directory are deprecated and redirect to this document.

## 📋 Table of Contents

- [Overview](#overview)
- [Deployment Options](#deployment-options)
- [Prerequisites](#prerequisites)
- [Quick Start Deployments](#quick-start-deployments)
- [Architecture Options](#architecture-options)
- [Configuration Reference](#configuration-reference)
- [Security Setup](#security-setup)
- [Monitoring & Observability](#monitoring--observability)
- [Backup & Disaster Recovery](#backup--disaster-recovery)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Maintenance & Scaling](#maintenance--scaling)

## 🎯 Overview

This guide covers **all production deployment scenarios** for LightRAG with enterprise-grade security, monitoring, and reliability features.

### Production Features Included

- **🔐 Enterprise Authentication**: Phase 1 security with bcrypt, JWT, rate limiting, audit logging
- **🐳 Container Orchestration**: Docker Compose with multi-service architecture
- **🤖 MCP Integration**: Model Context Protocol server for Claude CLI integration
- **📄 Enhanced Document Processing**: Docling service with OCR, table recognition, figure extraction
- **📊 Monitoring Stack**: Prometheus, Grafana, Jaeger tracing, log aggregation
- **🔄 Load Balancing**: Nginx reverse proxy with SSL termination
- **💾 Data Persistence**: Multiple database backends (PostgreSQL, Redis, MongoDB, Neo4j)
- **🔄 Automated Backups**: Database and data backups with cloud storage support
- **🚨 Health Monitoring**: Multi-tier health checks and alerting

## 🏗️ Deployment Options

Choose the deployment that matches your requirements:

### Option 1: Full Enterprise Stack (Recommended)
**Best for**: Production environments requiring all features
- LightRAG + Web UI + API + MCP Server
- PostgreSQL + Redis + monitoring stack
- Enhanced document processing (Docling)
- Full security + SSL + authentication

### Option 2: Simple Production Stack
**Best for**: Basic production deployments
- LightRAG + database + basic monitoring
- No enhanced document processing
- Simple authentication

### Option 3: xAI + Ollama Stack
**Best for**: Organizations using xAI Grok models
- LightRAG configured for xAI Grok models
- Local Ollama for embeddings
- Optimized for xAI timeout handling

### Option 4: High-Performance Stack
**Best for**: Large-scale deployments
- Horizontal scaling ready
- Advanced database configurations
- Performance optimization

## ✅ Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Ubuntu 20.04+, CentOS 8+ | Ubuntu 22.04 LTS |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8GB | 16-32GB |
| **Storage** | 100GB SSD | 500GB+ SSD |
| **Network** | Static IP | Domain + SSL |

### Software Requirements

- **Docker**: 24.0+ with Docker Compose V2
- **Git**: 2.30+
- **SSL Certificates**: Let's Encrypt or commercial (production)
- **Domain Name**: For SSL and external access (production)

### External Service Requirements

- **LLM Provider**: OpenAI, Anthropic, xAI, or compatible API
- **Email Service**: SMTP for notifications (optional)
- **Cloud Storage**: AWS S3, Google Cloud, or compatible for backups (optional)

## ⚡ Quick Start Deployments

### Full Enterprise Stack (5-minute setup)

```bash
# 1. Clone repository
git clone https://github.com/Ajith-82/LightRAG.git
cd LightRAG

# 2. Copy production environment template
cp production.env .env

# 3. Configure environment (edit .env file)
# Set: POSTGRES_PASSWORD, LLM_API_KEY, JWT_SECRET_KEY, GRAFANA_ADMIN_PASSWORD

# 4. Create directories
mkdir -p data/{rag_storage,inputs,backups} logs certs

# 5. Deploy enterprise stack
docker compose -f docker-compose.production.yml -f docker-compose.enhanced.yml --profile enhanced-processing up -d

# 6. Verify deployment
curl http://localhost/health
docker compose -f docker-compose.production.yml ps
```

### Simple Production Stack

```bash
# 1-3. Same as above

# 4. Deploy basic stack
docker compose -f docker-compose.production.yml up -d

# 5. Verify
curl http://localhost:9621/health
```

### xAI + Ollama Stack

```bash
# 1. Clone and setup
git clone https://github.com/Ajith-82/LightRAG.git
cd LightRAG

# 2. Configure for xAI
cp production.env .env
# Edit .env file with these key settings:
cat >> .env << EOF
# xAI Configuration
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
LLM_BINDING_API_KEY=your_xai_api_key_here
MAX_ASYNC=2
TIMEOUT=240

# Ollama Embeddings
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_BINDING_HOST=http://host.docker.internal:11434

# Storage
LIGHTRAG_KV_STORAGE=RedisKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
EOF

# 3. Ensure Ollama is running locally
ollama pull bge-m3:latest

# 4. Deploy
docker compose -f docker-compose.production.yml up -d

# 5. Test deployment
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, world!", "mode": "naive"}'
```

## 🏛️ Architecture Options

### Full Enterprise Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   LightRAG       │────│   PostgreSQL    │
│   (Nginx + SSL) │    │   + Web UI       │    │   + pgvector    │
└─────────────────┘    │   + MCP Server   │    └─────────────────┘
                       └──────────────────┘             │
                                │                       │
            ┌───────────────────┼───────────┐          │
            │                   │           │          │
    ┌───────▼────────┐ ┌────────▼────────┐  │  ┌─────▼─────┐
    │  Docling       │ │  External LLM   │  │  │   Redis   │
    │  Service       │ │  (xAI/OpenAI)   │  │  │   Cache   │
    │  (Enhanced)    │ │                 │  │  │           │
    └────────────────┘ └─────────────────┘  │  └───────────┘
                                            │
    ┌─────────────────────────────────────┐│
    │         Monitoring Stack            ││
    │  Prometheus + Grafana + Jaeger      ││
    │  + Loki + Alertmanager             ││
    └─────────────────────────────────────┘│
                                          │
    ┌─────────────────────────────────────▼┐
    │            Backup Service            │
    │     Database + Data + Cloud          │
    └──────────────────────────────────────┘
```

### Component Specifications

| Component | Purpose | Resources | High Availability |
|-----------|---------|-----------|------------------|
| **LightRAG Core** | Main application | 4-8 CPU, 8-16GB RAM | Horizontal scaling ready |
| **PostgreSQL** | Primary database | 4-8 CPU, 16-32GB RAM | Master-replica supported |
| **Redis** | Caching layer | 2-4 CPU, 4-8GB RAM | Cluster mode available |
| **Nginx** | Load balancer/proxy | 2 CPU, 2-4GB RAM | Multiple instances |
| **MCP Server** | Claude CLI integration | 2 CPU, 2GB RAM | Stateless scaling |
| **Docling** | Enhanced processing | 4-6 CPU, 4-8GB RAM | Processing queue |
| **Monitoring** | Observability stack | 4 CPU, 8GB RAM | Distributed setup |

## ⚙️ Configuration Reference

### Core Environment Variables

```bash
# =============================================================================
# CORE APPLICATION SETTINGS
# =============================================================================

# Server Configuration
HOST=0.0.0.0
PORT=9621
WORKERS=4                        # Gunicorn worker processes
WORKER_TIMEOUT=300              # Request timeout (seconds)
DEBUG=false                     # Never true in production
NODE_ENV=production

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# OpenAI Configuration
LLM_BINDING=openai
LLM_API_KEY=your-openai-api-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_API_KEY=your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-large

# xAI Configuration (Alternative)
LLM_BINDING=xai
LLM_BINDING_API_KEY=your-xai-api-key
LLM_MODEL=grok-3-mini
MAX_ASYNC=2                     # Critical for xAI stability
TIMEOUT=240                     # 4 minutes for complex operations

# Ollama Configuration (For Embeddings)
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
EMBEDDING_BINDING_HOST=http://host.docker.internal:11434

# =============================================================================
# DATABASE & STORAGE CONFIGURATION
# =============================================================================

# PostgreSQL (Primary Database)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=lightrag_prod
POSTGRES_PASSWORD=your-secure-postgres-password
POSTGRES_DATABASE=lightrag_production
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=30
POSTGRES_SSLMODE=require        # Enforce SSL

# Redis (Caching)
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_POOL_SIZE=20
REDIS_SOCKET_KEEPALIVE=true

# Storage Backend Selection
LIGHTRAG_KV_STORAGE=PGKVStorage            # or RedisKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage    # or MilvusVectorDBStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage      # or Neo4JStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# Authentication (Phase 1)
AUTH_ENABLED=true
JWT_SECRET_KEY=your-super-secure-jwt-secret-key
JWT_EXPIRATION_HOURS=24
BCRYPT_ROUNDS=12

# Password Security
PASSWORD_MIN_LENGTH=12
PASSWORD_LOCKOUT_ATTEMPTS=5
PASSWORD_LOCKOUT_DURATION=3600

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

# =============================================================================
# ENHANCED DOCUMENT PROCESSING
# =============================================================================

# Docling Service Configuration
LIGHTRAG_ENHANCED_PROCESSING=true
DOCLING_SERVICE_URL=http://docling-service:8080
DOCLING_SERVICE_TIMEOUT=300
DOCLING_FALLBACK_ENABLED=true

# Docling Processing Features
DOCLING_DEFAULT_ENABLE_OCR=true
DOCLING_DEFAULT_ENABLE_TABLE_STRUCTURE=true
DOCLING_DEFAULT_ENABLE_FIGURES=true
DOCLING_DEFAULT_MAX_WORKERS=3

# Docling Resource Limits
DOCLING_MAX_FILE_SIZE_MB=100
DOCLING_MAX_BATCH_SIZE=10
DOCLING_CACHE_ENABLED=true
DOCLING_CACHE_MAX_SIZE_GB=5
DOCLING_CACHE_TTL_HOURS=168     # 7 days

# =============================================================================
# MCP SERVER CONFIGURATION
# =============================================================================

# MCP Server Settings
MCP_SERVER_ENABLED=true
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8080
MCP_SERVER_WORKERS=2

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

# =============================================================================
# MONITORING & OBSERVABILITY
# =============================================================================

# Monitoring Settings
GRAFANA_ADMIN_PASSWORD=your-grafana-password
PROMETHEUS_RETENTION_TIME=15d
JAEGER_ENABLED=true
LOKI_ENABLED=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_ACCESS_LOG=true
LOG_CORRELATION_ID=true

# Metrics Collection
METRICS_ENABLED=true
METRICS_EXPORT_INTERVAL=60
CUSTOM_METRICS_ENABLED=true

# =============================================================================
# BACKUP & RECOVERY
# =============================================================================

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 1 * * *"     # Daily at 1 AM
BACKUP_RETENTION_DAYS=7

# Cloud Storage (Optional)
BACKUP_CLOUD_ENABLED=false
AWS_S3_BUCKET=your-backup-bucket
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Processing Performance
MAX_PARALLEL_INSERT=4           # Concurrent document processing
CHUNK_TOKEN_SIZE=1200          # Optimal chunk size
LLM_MAX_ASYNC=4                # Concurrent LLM requests
EMBEDDING_MAX_ASYNC=8          # Concurrent embedding requests
ENTITY_EXTRACT_MAX_GLEANING=2  # Entity extraction passes

# Memory Management
PYTHON_MEMORY_LIMIT=8G
WORKER_MEMORY_LIMIT=2G
CLEANUP_INTERVAL=3600          # Memory cleanup interval (seconds)

# =============================================================================
# SSL/TLS CONFIGURATION
# =============================================================================

# SSL Settings
SSL_ENABLED=true
SSL_CERT_PATH=/certs/cert.pem
SSL_KEY_PATH=/certs/key.pem
SSL_REDIRECT_HTTP=true

# Let's Encrypt Integration
LETSENCRYPT_ENABLED=false
LETSENCRYPT_EMAIL=admin@yourdomain.com
LETSENCRYPT_DOMAINS=yourdomain.com,api.yourdomain.com
```

### Docker Compose Files

The production deployment uses layered Docker Compose files:

- **`docker-compose.production.yml`**: Base production services
- **`docker-compose.enhanced.yml`**: Enhanced document processing
- **`docker-compose.monitoring.yml`**: Full monitoring stack
- **`docker-compose.backup.yml`**: Backup and recovery services

## 🔐 Security Setup

### SSL/TLS Configuration

#### Production SSL with Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot

# Generate certificates
certbot certonly --standalone -d yourdomain.com

# Copy to project
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem certs/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem certs/key.pem

# Set permissions
chmod 600 certs/key.pem
chmod 644 certs/cert.pem

# Auto-renewal
echo "0 2 * * * certbot renew --quiet && docker compose -f docker-compose.production.yml restart nginx" | crontab -
```

#### Self-Signed SSL (Development/Testing)

```bash
# Quick self-signed certificate with SAN support
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem \
  -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=LightRAG/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1"

# Set permissions
chmod 600 certs/key.pem
chmod 644 certs/cert.pem
```

### Firewall Configuration

```bash
# Ubuntu/Debian
sudo ufw allow 80/tcp          # HTTP
sudo ufw allow 443/tcp         # HTTPS
sudo ufw deny 5432/tcp         # Block direct database access
sudo ufw deny 6379/tcp         # Block direct Redis access
sudo ufw deny 9621/tcp         # Block direct app access (use nginx)
sudo ufw enable

# CentOS/RHEL/Fedora
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### Authentication Configuration

```bash
# Create first admin user
curl -X POST http://localhost/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "SecurePass123!",
    "email": "admin@yourdomain.com",
    "role": "admin"
  }'

# Login and get JWT token
curl -X POST http://localhost/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "SecurePass123!"}' \
  | jq -r '.access_token'
```

## 📊 Monitoring & Observability

### Grafana Dashboards

Access Grafana at `https://yourdomain.com/grafana` (admin/your-password):

1. **LightRAG Application Dashboard**
   - Request rates, response times, error rates
   - Active sessions, query performance
   - LLM API usage and costs

2. **System Resources Dashboard**
   - CPU, memory, disk, network usage
   - Container resource utilization
   - Storage backend performance

3. **Security Dashboard**
   - Authentication events, failed logins
   - Rate limiting incidents
   - Audit log analysis

4. **Business Metrics Dashboard**
   - Document processing volume
   - Query types and success rates
   - User activity patterns

### Prometheus Metrics

Key metrics available at `https://yourdomain.com/metrics`:

```promql
# Application Performance
lightrag_request_duration_seconds
lightrag_request_total
lightrag_error_rate
lightrag_active_sessions

# Document Processing
lightrag_documents_processed_total
lightrag_processing_duration_seconds
lightrag_processing_errors_total

# LLM Usage
lightrag_llm_requests_total
lightrag_llm_tokens_used
lightrag_llm_response_time

# Storage Performance
lightrag_database_connections
lightrag_database_query_duration
lightrag_cache_hit_rate
```

### Alerting Rules

Critical alerts configured in Prometheus:

```yaml
# Application Health
- alert: LightRAGDown
  expr: up{job="lightrag"} == 0
  for: 1m
  labels:
    severity: critical

# High Error Rate
- alert: HighErrorRate
  expr: rate(lightrag_request_total{status=~"5.."}[5m]) > 0.1
  for: 2m
  labels:
    severity: warning

# Database Issues
- alert: DatabaseConnectionHigh
  expr: lightrag_database_connections > 80
  for: 5m
  labels:
    severity: warning

# Resource Usage
- alert: HighMemoryUsage
  expr: container_memory_usage_bytes{name="lightrag"} / container_spec_memory_limit_bytes > 0.9
  for: 5m
  labels:
    severity: critical
```

### Log Aggregation

Structured JSON logging with ELK Stack integration:

```json
{
  "timestamp": "2025-08-09T14:30:00.123Z",
  "level": "INFO",
  "service": "lightrag-api",
  "version": "2.1.0",
  "request_id": "req-abc123",
  "user_id": "user-456",
  "session_id": "sess-789",
  "event": "document_processed",
  "document_id": "doc-321",
  "processing_time_ms": 2340,
  "mode": "enhanced",
  "metadata": {
    "file_type": "pdf",
    "pages": 15,
    "ocr_enabled": true
  }
}
```

## 💾 Backup & Disaster Recovery

### Automated Backup Strategy

```bash
# Backup Types and Schedule
Database Backups:    Daily 01:00 AM (7-day retention)
Data Backups:        Daily 02:00 AM (30-day retention)
Config Backups:      Weekly Sundays (90-day retention)
Full System:         Monthly 1st (12-month retention)
```

### Manual Backup Commands

```bash
# Database backup
docker exec lightrag_postgres pg_dump -U lightrag_prod lightrag_production | gzip > backup_$(date +%Y%m%d).sql.gz

# Data directory backup
tar -czf rag_storage_backup_$(date +%Y%m%d).tar.gz data/rag_storage/

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env certs/ nginx/ postgres/

# Complete system backup
docker run --rm -v /opt/lightrag:/source -v /backups:/backup alpine tar -czf /backup/full_backup_$(date +%Y%m%d).tar.gz -C /source .
```

### Cloud Storage Integration

```bash
# AWS S3 Configuration
AWS_S3_BUCKET=lightrag-backups-prod
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# Automated cloud upload
docker exec lightrag_backup aws s3 sync /backups s3://lightrag-backups-prod/$(hostname)/
```

### Disaster Recovery Procedures

#### Recovery Time Objectives (RTO)
- **Database Recovery**: < 15 minutes
- **Application Recovery**: < 5 minutes
- **Full System Recovery**: < 30 minutes

#### Recovery Point Objectives (RPO)
- **Database**: < 1 hour (WAL archiving)
- **Documents**: < 24 hours (daily backups)
- **Configuration**: < 7 days (weekly backups)

#### Recovery Steps

```bash
# 1. Database Recovery
gunzip -c backup_20250809.sql.gz | docker exec -i lightrag_postgres psql -U lightrag_prod lightrag_production

# 2. Data Recovery
tar -xzf rag_storage_backup_20250809.tar.gz -C data/

# 3. Application Restart
docker compose -f docker-compose.production.yml restart

# 4. Verification
curl http://localhost/health
docker compose -f docker-compose.production.yml ps
```

## ⚡ Performance Tuning

### Application-Level Optimization

```bash
# Worker Configuration
WORKERS=8                        # Scale with CPU cores
MAX_WORKERS=16                   # 2x workers for burst capacity
WORKER_TIMEOUT=600              # Increase for large documents

# Concurrent Processing
LLM_MAX_ASYNC=8                 # Scale with LLM provider limits
EMBEDDING_MAX_ASYNC=16          # Usually higher than LLM
MAX_PARALLEL_INSERT=8           # Balance with system resources

# Memory Management
CHUNK_TOKEN_SIZE=1200           # Optimal for most LLMs
ENTITY_EXTRACT_MAX_GLEANING=1   # Reduce for speed, increase for quality
CLEANUP_INTERVAL=1800           # Clean up every 30 minutes
```

### Database Performance

#### PostgreSQL Optimization

```ini
# /postgres/config/postgresql.conf

# Memory Settings (for 32GB system)
shared_buffers = 8GB                    # 25% of RAM
effective_cache_size = 24GB             # 75% of RAM
work_mem = 256MB                        # Per query operation
maintenance_work_mem = 2GB              # For VACUUM, CREATE INDEX

# Connection Settings
max_connections = 200                   # Balance with application workers
max_prepared_transactions = 200         # For 2PC transactions

# Performance Settings
checkpoint_completion_target = 0.9      # Spread checkpoint I/O
wal_buffers = 64MB                      # WAL write buffer
default_statistics_target = 500         # Query planner statistics
random_page_cost = 1.1                  # For SSD storage
effective_io_concurrency = 200          # For SSD storage

# Vacuum Settings
autovacuum = on
autovacuum_max_workers = 6
autovacuum_naptime = 15s

# Logging (for monitoring)
log_min_duration_statement = 1000       # Log slow queries
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
```

#### Database Indexing Strategy

```sql
-- Critical indexes for LightRAG performance
CREATE INDEX CONCURRENTLY idx_documents_status ON documents(status);
CREATE INDEX CONCURRENTLY idx_documents_created ON documents(created_at);
CREATE INDEX CONCURRENTLY idx_chunks_document_id ON chunks(document_id);
CREATE INDEX CONCURRENTLY idx_entities_type ON entities(entity_type);
CREATE INDEX CONCURRENTLY idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX CONCURRENTLY idx_relationships_target ON relationships(target_entity_id);

-- Vector similarity indexes (if using pgvector)
CREATE INDEX CONCURRENTLY idx_embeddings_vector ON embeddings USING ivfflat (embedding) WITH (lists = 1000);

-- Full-text search indexes
CREATE INDEX CONCURRENTLY idx_documents_content_fts ON documents USING gin(to_tsvector('english', content));
```

### Redis Performance

```bash
# /redis/redis.conf

# Memory Management
maxmemory 8gb                           # Set appropriate limit
maxmemory-policy allkeys-lru            # Eviction policy

# Persistence (balance durability vs performance)
save 900 1                              # Save if 1 key changes in 900s
save 300 10                             # Save if 10 keys change in 300s
save 60 10000                           # Save if 10k keys change in 60s
appendonly yes                          # Enable AOF for durability
appendfsync everysec                    # Sync every second

# Network
tcp-keepalive 300                       # Keep connections alive
timeout 0                               # No client timeout

# Performance
lazyfree-lazy-eviction yes              # Non-blocking eviction
lazyfree-lazy-expire yes                # Non-blocking expiration
```

### Container Resource Optimization

```yaml
# docker-compose.production.yml resource limits
services:
  lightrag:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8.0'
        reservations:
          memory: 8G
          cpus: '4.0'

  postgres:
    deploy:
      resources:
        limits:
          memory: 32G
          cpus: '8.0'
        reservations:
          memory: 16G
          cpus: '4.0'

  redis:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

### Load Testing & Benchmarks

```bash
# Install load testing tools
pip install locust httpx

# Create load test script
cat > load_test.py << 'EOF'
from locust import HttpUser, task, between
import json

class LightRAGUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Login and get token
        response = self.client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})

    @task(3)
    def query_hybrid(self):
        self.client.post("/query", json={
            "query": "What is the main topic?",
            "mode": "hybrid"
        })

    @task(1)
    def query_local(self):
        self.client.post("/query", json={
            "query": "Explain the process",
            "mode": "local"
        })

    @task(1)
    def health_check(self):
        self.client.get("/health")
EOF

# Run load test
locust -f load_test.py --host=https://yourdomain.com --users 50 --spawn-rate 5 --run-time 5m
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Application Won't Start

**Symptoms**: Container exits immediately, health checks fail

```bash
# Diagnostic steps
docker compose -f docker-compose.production.yml logs lightrag

# Common causes and solutions:
# Database connection failed
export POSTGRES_PASSWORD="correct-password"

# Missing environment variables
grep -E "^[A-Z]" .env | wc -l     # Should be 50+ variables

# Port conflicts
netstat -tlnp | grep :9621        # Check if port is in use
sudo lsof -i :9621               # Find process using port

# SSL certificate issues
openssl x509 -in certs/cert.pem -text -noout
```

#### 2. Enhanced Document Processing Issues

**Symptoms**: Docling service shows "degraded" status, OCR not working

```bash
# Check Docling service status
curl http://localhost:8080/health

# Common issues:
# DOCLING_DEBUG environment variable causes Pydantic errors
unset DOCLING_DEBUG
grep -v DOCLING_DEBUG .env > .env.tmp && mv .env.tmp .env

# Insufficient memory for ML models
docker stats lightrag_docling_prod    # Should have 4GB+ available

# Model download failures
docker exec lightrag_docling_prod python -c "
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
print('Models loaded successfully')
"

# Restart service
docker compose restart docling-service
```

#### 3. xAI API Timeout Issues

**Symptoms**: "504 Gateway Timeout", slow responses from xAI

```bash
# Verify xAI configuration
echo "MAX_ASYNC: $MAX_ASYNC (should be 2)"
echo "TIMEOUT: $TIMEOUT (should be 240+)"
echo "LLM_MODEL: $LLM_MODEL (use grok-3-mini for best performance)"

# Test xAI API directly
curl -X POST https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grok-3-mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'

# Monitor xAI request patterns
docker logs lightrag | grep -i "xai\|timeout\|retry"
```

#### 4. High Memory Usage

**Symptoms**: System becomes unresponsive, OOM kills

```bash
# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Identify memory-hungry containers
docker stats --no-stream | sort -k4 -h

# Reduce resource usage
export WORKERS=2                    # Reduce Gunicorn workers
export LLM_MAX_ASYNC=2             # Reduce concurrent LLM calls
export MAX_PARALLEL_INSERT=2       # Reduce document processing
export DOCLING_DEFAULT_MAX_WORKERS=1  # Reduce Docling workers

# Restart with lower limits
docker compose -f docker-compose.production.yml restart
```

#### 5. Database Performance Issues

**Symptoms**: Slow queries, high database CPU usage

```bash
# Check database performance
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;"

# Check active connections
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "
SELECT COUNT(*), state FROM pg_stat_activity GROUP BY state;"

# Analyze slow queries
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "
SELECT query, mean_time, calls
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY mean_time DESC;"

# Check table sizes
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "
SELECT schemaname,tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;"

# Run maintenance
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "
VACUUM ANALYZE; REINDEX DATABASE lightrag_production;"
```

#### 6. Authentication Issues

**Symptoms**: 401 Unauthorized, JWT token errors

```bash
# Verify JWT configuration
echo "JWT_SECRET_KEY length: ${#JWT_SECRET_KEY} (should be 32+ chars)"
echo "JWT_EXPIRATION_HOURS: $JWT_EXPIRATION_HOURS"

# Test authentication flow
# 1. Register user
curl -X POST http://localhost/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "TestPass123!", "email": "test@example.com"}'

# 2. Login
TOKEN=$(curl -X POST http://localhost/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "TestPass123!"}' \
  | jq -r '.access_token')

# 3. Test authenticated request
curl -H "Authorization: Bearer $TOKEN" http://localhost/health

# Check audit logs
tail -f logs/audit.log | grep -E "login|auth|401|403"
```

#### 7. SSL/TLS Issues

**Symptoms**: Certificate errors, HTTPS not working

```bash
# Verify certificate validity
openssl x509 -in certs/cert.pem -text -noout | grep -E "Not Before|Not After"

# Test SSL connection
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com < /dev/null

# Check certificate chain
curl -vI https://yourdomain.com

# Renew Let's Encrypt certificates
certbot renew --dry-run
certbot renew
docker compose -f docker-compose.production.yml restart nginx
```

### Debug Mode Configuration

For troubleshooting, temporarily enable debug mode (never in production):

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG
export VERBOSE_LOGGING=true

# Enable SQL query logging
export POSTGRES_LOG_STATEMENT=all

# Enable detailed MCP logging
export MCP_LOG_LEVEL=DEBUG

# Restart services
docker compose -f docker-compose.production.yml restart

# Disable debug mode after troubleshooting
export DEBUG=false
export LOG_LEVEL=INFO
export VERBOSE_LOGGING=false
```

### Performance Monitoring for Troubleshooting

```bash
# Real-time performance monitoring
watch -n 1 'docker stats --no-stream'

# Database connections
watch -n 5 'docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "SELECT COUNT(*) FROM pg_stat_activity;"'

# API response times
while true; do
  curl -w "%{time_total}s\n" -o /dev/null -s http://localhost/health
  sleep 5
done

# Memory usage trends
docker stats --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}" | tee -a memory_usage.log
```

## 🔄 Maintenance & Scaling

### Regular Maintenance Schedule

#### Daily Tasks (Automated)
- Health check monitoring
- Log rotation and cleanup
- Database and data backups
- Security event analysis
- Performance metrics review

#### Weekly Tasks
```bash
# Update SSL certificates if needed
certbot renew --quiet

# Database maintenance
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "VACUUM ANALYZE;"

# Clean old logs
find logs/ -name "*.log" -type f -mtime +7 -delete

# Update system packages
apt update && apt list --upgradable
```

#### Monthly Tasks
```bash
# Update Docker images
docker compose -f docker-compose.production.yml pull

# Security updates
apt upgrade -y

# Performance optimization review
docker stats --no-stream
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "
SELECT * FROM pg_stat_user_tables ORDER BY seq_scan DESC LIMIT 10;"

# Backup verification
aws s3 ls s3://lightrag-backups-prod/$(hostname)/ --recursive | tail -10

# Disaster recovery testing (staging environment)
# Test restore procedures on staging
```

### Scaling Strategies

#### Horizontal Scaling (Multiple Instances)

```bash
# Scale application instances
docker compose -f docker-compose.production.yml up -d --scale lightrag=3

# Update nginx load balancer configuration
cat > nginx/conf.d/upstream.conf << 'EOF'
upstream lightrag_backend {
    least_conn;
    server lightrag-1:9621 max_fails=3 fail_timeout=30s;
    server lightrag-2:9621 max_fails=3 fail_timeout=30s;
    server lightrag-3:9621 max_fails=3 fail_timeout=30s;
}
EOF

# Restart nginx
docker compose -f docker-compose.production.yml restart nginx
```

#### Vertical Scaling (More Resources)

```yaml
# Update docker-compose.production.yml
services:
  lightrag:
    deploy:
      resources:
        limits:
          memory: 32G        # Increase from 16G
          cpus: '16.0'       # Increase from 8.0
        reservations:
          memory: 16G        # Increase from 8G
          cpus: '8.0'        # Increase from 4.0

  postgres:
    deploy:
      resources:
        limits:
          memory: 64G        # Increase from 32G
          cpus: '16.0'       # Increase from 8.0
```

#### Database Scaling

```bash
# PostgreSQL Read Replicas
cat > docker-compose.replica.yml << 'EOF'
version: '3.8'
services:
  postgres-replica:
    image: shangor/postgres-for-rag:v1.0
    environment:
      PGUSER: lightrag_prod
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_MASTER_SERVICE: postgres
      POSTGRES_REPLICA_USER: replica_user
      POSTGRES_REPLICA_PASSWORD: replica_password
    command: |
      bash -c "
        pg_basebackup -h postgres -D /var/lib/postgresql/data -U replica_user -v -P -W
        echo 'standby_mode = on' >> /var/lib/postgresql/data/postgresql.conf
        echo 'primary_conninfo = ''host=postgres port=5432 user=replica_user''' >> /var/lib/postgresql/data/postgresql.conf
        postgres
      "
    depends_on:
      - postgres
EOF

# Redis Cluster (for larger deployments)
cat > docker-compose.redis-cluster.yml << 'EOF'
version: '3.8'
services:
  redis-1:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
  redis-2:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
  redis-3:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
EOF
```

#### Auto-Scaling with Docker Swarm

```bash
# Initialize Docker Swarm
docker swarm init

# Create auto-scaling service
docker service create \
  --name lightrag \
  --replicas 2 \
  --limit-memory 16g \
  --limit-cpu 8 \
  --env-file .env \
  --network lightrag-network \
  --publish 9621:9621 \
  lightrag:production

# Auto-scale based on CPU
docker service update \
  --replicas-max-per-node 2 \
  --constraint-add node.labels.environment==production \
  lightrag

# Monitor scaling
watch docker service ps lightrag
```

### Capacity Planning

#### Resource Monitoring and Forecasting

```bash
# Collect baseline metrics
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" --no-stream > baseline_metrics.txt

# Monitor growth trends
cat > collect_metrics.sh << 'EOF'
#!/bin/bash
DATE=$(date '+%Y-%m-%d %H:%M:%S')
CPU=$(docker stats --format "{{.CPUPerc}}" --no-stream lightrag | sed 's/%//')
MEM=$(docker stats --format "{{.MemUsage}}" --no-stream lightrag | cut -d'/' -f1)
DOCS=$(curl -s http://localhost/api/documents/count | jq '.count')
QUERIES=$(curl -s http://localhost/metrics | grep lightrag_request_total | tail -1 | awk '{print $2}')

echo "$DATE,$CPU,$MEM,$DOCS,$QUERIES" >> capacity_metrics.csv
EOF

# Run every 15 minutes
crontab -e
# Add: */15 * * * * /opt/lightrag/collect_metrics.sh
```

#### Capacity Thresholds and Alerts

```yaml
# prometheus/alert_rules.yml
groups:
- name: capacity_planning
  rules:
  - alert: HighCPUUtilization
    expr: container_cpu_usage_seconds_total > 0.8
    for: 10m
    labels:
      severity: warning
      action: scale_up

  - alert: HighMemoryUtilization
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.85
    for: 5m
    labels:
      severity: warning
      action: scale_up

  - alert: DatabaseConnectionsHigh
    expr: pg_stat_database_numbackends > 150
    for: 5m
    labels:
      severity: warning
      action: optimize_or_scale
```

### Upgrade Procedures

#### Application Updates

```bash
# 1. Backup before upgrade
docker exec lightrag_backup /app/scripts/backup-database.sh
docker exec lightrag_backup /app/scripts/backup-data.sh

# 2. Pull latest images
docker compose -f docker-compose.production.yml pull

# 3. Rolling update (zero downtime)
docker compose -f docker-compose.production.yml up -d --scale lightrag=2
sleep 60  # Wait for new instance to be healthy
docker compose -f docker-compose.production.yml up -d --scale lightrag=1 --no-recreate

# 4. Verify update
curl http://localhost/health
docker compose -f docker-compose.production.yml logs lightrag | tail -50

# 5. Run database migrations if needed
docker exec lightrag_app python -m lightrag.api.migrations.run_all
```

#### Database Schema Updates

```bash
# 1. Create migration script
cat > migration_v2.1.0.sql << 'EOF'
BEGIN;

-- Add new columns
ALTER TABLE documents ADD COLUMN IF NOT EXISTS enhanced_metadata JSONB;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_enhanced_metadata
  ON documents USING gin(enhanced_metadata);

-- Update existing data
UPDATE documents SET enhanced_metadata = '{}' WHERE enhanced_metadata IS NULL;

COMMIT;
EOF

# 2. Test migration on staging
docker exec lightrag_postgres_staging psql -U lightrag_prod -d lightrag_production -f migration_v2.1.0.sql

# 3. Apply to production (during maintenance window)
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -f migration_v2.1.0.sql

# 4. Verify migration
docker exec lightrag_postgres psql -U lightrag_prod -d lightrag_production -c "
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'documents' AND column_name = 'enhanced_metadata';"
```

### Monitoring and Alerting Setup

#### Comprehensive Alert Configuration

```yaml
# alertmanager/config.yml
global:
  smtp_smarthost: 'smtp.yourdomain.com:587'
  smtp_from: 'alerts@yourdomain.com'
  smtp_auth_username: 'alerts@yourdomain.com'
  smtp_auth_password: 'your-email-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://alertmanager-webhook:5000/alerts'

- name: 'critical-alerts'
  email_configs:
  - to: 'devops@yourdomain.com'
    subject: 'CRITICAL: LightRAG Alert'
    body: |
      Alert: {{ .GroupLabels.alertname }}
      Summary: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
      Details: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts-critical'

- name: 'warning-alerts'
  email_configs:
  - to: 'team@yourdomain.com'
    subject: 'WARNING: LightRAG Alert'
    body: |
      Alert: {{ .GroupLabels.alertname }}
      Summary: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
```

---

## 📞 Production Support

### Support Escalation Matrix

| Issue Severity | Response Time | Escalation Path |
|---------------|---------------|-----------------|
| **Critical** (System Down) | 15 minutes | DevOps → Engineering Manager → CTO |
| **High** (Performance Impact) | 1 hour | DevOps → Senior Engineer → Engineering Manager |
| **Medium** (Feature Issue) | 4 hours | Support → Engineer → Team Lead |
| **Low** (General Questions) | 24 hours | Support → Documentation → Engineer |

### Health Check Endpoints

```bash
# Basic health
curl https://yourdomain.com/health
# Response: {"status": "healthy", "timestamp": "2025-08-09T14:30:00Z"}

# Detailed health
curl https://yourdomain.com/health?detailed=true
# Response: Full system status including dependencies

# Component health
curl https://yourdomain.com/health/database
curl https://yourdomain.com/health/llm
curl https://yourdomain.com/health/storage
```

### Log Collection for Support

```bash
# Collect comprehensive logs for support
cat > collect_support_logs.sh << 'EOF'
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="support_logs_${TIMESTAMP}"

mkdir -p $LOGDIR

# Application logs
docker compose -f docker-compose.production.yml logs lightrag > $LOGDIR/lightrag.log
docker compose -f docker-compose.production.yml logs postgres > $LOGDIR/postgres.log
docker compose -f docker-compose.production.yml logs redis > $LOGDIR/redis.log

# System information
docker stats --no-stream > $LOGDIR/docker_stats.txt
docker compose -f docker-compose.production.yml ps > $LOGDIR/services_status.txt
df -h > $LOGDIR/disk_usage.txt
free -h > $LOGDIR/memory_usage.txt

# Configuration (sanitized)
cp .env $LOGDIR/config.env
sed -i 's/=.*/=***REDACTED***/g' $LOGDIR/config.env

# Health checks
curl -s http://localhost/health > $LOGDIR/health_check.json
curl -s http://localhost/metrics > $LOGDIR/metrics.txt

# Create archive
tar -czf $LOGDIR.tar.gz $LOGDIR
rm -rf $LOGDIR

echo "Support logs collected in: $LOGDIR.tar.gz"
EOF

chmod +x collect_support_logs.sh
./collect_support_logs.sh
```

### Emergency Procedures

#### Service Recovery

```bash
# Emergency restart procedure
docker compose -f docker-compose.production.yml down
docker system prune -f
docker compose -f docker-compose.production.yml up -d

# Database recovery
docker exec lightrag_postgres pg_isready -U lightrag_prod
# If failed, restore from backup
gunzip -c backup_latest.sql.gz | docker exec -i lightrag_postgres psql -U lightrag_prod lightrag_production

# Data recovery
tar -xzf rag_storage_backup_latest.tar.gz -C data/
```

#### Rollback Procedure

```bash
# Rollback to previous version
docker tag lightrag:production lightrag:broken-$(date +%Y%m%d)
docker tag lightrag:production-prev lightrag:production
docker compose -f docker-compose.production.yml up -d

# Database rollback (if schema changes)
# Restore from pre-upgrade backup
docker exec -i lightrag_postgres psql -U lightrag_prod lightrag_production < backup_pre_upgrade.sql
```

---

## 🏆 Production Readiness Checklist

### Security ✅
- [ ] SSL/TLS certificates configured and valid
- [ ] Authentication enabled with strong passwords
- [ ] Rate limiting configured
- [ ] Firewall rules configured
- [ ] Audit logging enabled
- [ ] Security headers configured
- [ ] Database connections encrypted
- [ ] API keys properly secured

### Monitoring ✅
- [ ] Health checks responding correctly
- [ ] Prometheus metrics collection enabled
- [ ] Grafana dashboards configured
- [ ] Critical alerts configured
- [ ] Log aggregation working
- [ ] Performance monitoring enabled
- [ ] Capacity monitoring enabled

### Backup & Recovery ✅
- [ ] Database backups automated and tested
- [ ] Data backups automated and tested
- [ ] Backup retention policies configured
- [ ] Cloud storage integration tested
- [ ] Disaster recovery procedures documented
- [ ] Recovery testing scheduled

### Performance ✅
- [ ] Resource limits configured appropriately
- [ ] Database indexes optimized
- [ ] Connection pooling configured
- [ ] Caching strategy implemented
- [ ] Load testing completed
- [ ] Performance baselines established

### Documentation ✅
- [ ] Deployment procedures documented
- [ ] Configuration guide complete
- [ ] Troubleshooting guide available
- [ ] Monitoring runbooks created
- [ ] Emergency procedures documented
- [ ] Team training completed

---

## 📚 Additional Resources

### Related Documentation
- [Authentication Setup Guide](../security/AUTHENTICATION_IMPROVEMENT_PLAN.md)
- [MCP Integration Guide](../integration_guides/MCP_IMPLEMENTATION_SUMMARY.md)
- [Security Hardening Guide](../security/SECURITY_HARDENING.md)
- [xAI Integration Troubleshooting](../integration_guides/TROUBLESHOOTING_XAI.md)
- [System Architecture Overview](../architecture/SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md)

### Configuration Templates
- [`production.env`](../../production.env) - Production environment template
- [`docker-compose.production.yml`](../../docker-compose.production.yml) - Main production stack
- [`docker-compose.enhanced.yml`](../../docker-compose.enhanced.yml) - Enhanced processing
- [`nginx/conf.d/lightrag.conf`](../../nginx/conf.d/lightrag.conf) - Nginx configuration

### Support Resources
- [GitHub Issues](https://github.com/HKUDS/LightRAG/issues) - Bug reports and feature requests
- [Discussions](https://github.com/HKUDS/LightRAG/discussions) - Community support
- [Documentation](../../docs/README.md) - Complete documentation index

---

**🎯 This is the complete production deployment guide for LightRAG.**
**All deployment scenarios, configurations, and operational procedures are covered in this single authoritative document.**

For questions or issues not covered in this guide, please check the troubleshooting section or create a GitHub issue with your specific environment details and error messages.
