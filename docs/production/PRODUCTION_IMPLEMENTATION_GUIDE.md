# LightRAG Production Implementation Guide

## Overview

This comprehensive guide provides step-by-step instructions for deploying LightRAG in production with the following components:

- **LightRAG Core + Web UI**: Complete RAG system with React frontend
- **xAI Grok**: Large Language Model for text generation
- **Local Ollama**: Embedding model service
- **PostgreSQL**: Production database with all storage backends
- **Authentication**: Enterprise-grade security
- **Monitoring**: Comprehensive observability and alerting

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Database Configuration](#database-configuration)
5. [Ollama Embedding Setup](#ollama-embedding-setup)
6. [xAI Integration](#xai-integration)
7. [LightRAG Application Deployment](#lightrag-application-deployment)
8. [Authentication & Security](#authentication--security)
9. [Monitoring & Observability](#monitoring--observability)
10. [SSL/TLS Configuration](#ssltls-configuration)
11. [Backup & Recovery](#backup--recovery)
12. [Performance Tuning](#performance-tuning)
13. [Troubleshooting](#troubleshooting)
14. [Maintenance Procedures](#maintenance-procedures)

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│                 │    │                  │    │                 │
│   Load Balancer │────│   LightRAG       │────│   PostgreSQL    │
│   (Nginx/HAP)   │    │   + Web UI       │    │   Database      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                │
                    ┌───────────┼───────────┐
                    │                       │
            ┌───────▼────────┐    ┌────────▼────────┐
            │                │    │                 │
            │  xAI Grok API  │    │ Local Ollama    │
            │  (LLM Service) │    │ (Embeddings)    │
            │                │    │                 │
            └────────────────┘    └─────────────────┘

            ┌─────────────────────────────────────┐
            │                                     │
            │         Monitoring Stack            │
            │  Prometheus + Grafana + Alerting    │
            │                                     │
            └─────────────────────────────────────┘
```

### Component Specifications

| Component | Purpose | Resource Requirements |
|-----------|---------|----------------------|
| LightRAG + Web UI | Main application server | 4 CPU, 8GB RAM, 50GB SSD |
| PostgreSQL | All storage backends | 4 CPU, 16GB RAM, 500GB SSD |
| Local Ollama | Embedding generation | 8 CPU, 32GB RAM, 100GB SSD |
| xAI Grok | LLM text generation | API service (external) |
| Nginx | Load balancer/proxy | 2 CPU, 4GB RAM, 20GB SSD |
| Monitoring | Observability stack | 2 CPU, 8GB RAM, 100GB SSD |

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 22.04 LTS or CentOS 8+ (recommended)
- **Docker**: Version 24.0+ with Docker Compose V2
- **Hardware**: Minimum 16 CPU cores, 64GB RAM, 1TB SSD
- **Network**: High-bandwidth internet connection for xAI API access
- **Domain**: Valid domain name with SSL certificates

### Required Accounts & API Keys

1. **xAI Account**: Get API key from [x.ai](https://x.ai)
2. **Domain & SSL**: SSL certificates for HTTPS
3. **Monitoring**: Optional external monitoring services

## Infrastructure Setup

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose V2
sudo apt install docker-compose-plugin

# Create application directory
sudo mkdir -p /opt/lightrag
sudo chown $USER:$USER /opt/lightrag
cd /opt/lightrag

# Clone LightRAG repository
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
```

### 2. Directory Structure Setup

```bash
# Create production directories
mkdir -p {data/{rag_storage,inputs,postgres},logs,certs,backups,monitoring}

# Set proper permissions
sudo chown -R $USER:$USER /opt/lightrag
chmod -R 755 /opt/lightrag
```

## Database Configuration

### 1. PostgreSQL Production Setup

Create `production-postgres.env`:

```bash
# PostgreSQL Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=lightrag_prod
POSTGRES_PASSWORD=your_secure_postgres_password_here
POSTGRES_DATABASE=lightrag_production
POSTGRES_MAX_CONNECTIONS=25

# PostgreSQL SSL Configuration
POSTGRES_SSL_MODE=require
```

### 2. Database Initialization

Create `docker-compose.postgres.yml`:

```yaml
version: '3.8'

services:
  postgres:
    container_name: lightrag_postgres_prod
    image: shangor/postgres-for-rag:v1.0
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DATABASE}
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --lc-collate=C --lc-ctype=C"
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
      - ./postgres/init:/docker-entrypoint-initdb.d:ro
      - ./postgres/config/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    ports:
      - "5432:5432"
    networks:
      - lightrag-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DATABASE}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '4'
        reservations:
          memory: 8G
          cpus: '2'
    command: >
      postgres -c config_file=/etc/postgresql/postgresql.conf
               -c max_connections=200
               -c shared_buffers=4GB
               -c effective_cache_size=12GB
               -c maintenance_work_mem=1GB
               -c checkpoint_completion_target=0.9
               -c wal_buffers=32MB
               -c default_statistics_target=100
               -c random_page_cost=1.1
               -c effective_io_concurrency=200

networks:
  lightrag-network:
    driver: bridge

volumes:
  postgres_prod_data:
    driver: local
```

Start PostgreSQL:

```bash
# Load environment variables
export $(cat production-postgres.env | xargs)

# Start PostgreSQL
docker-compose -f docker-compose.postgres.yml up -d

# Verify database is running
docker-compose -f docker-compose.postgres.yml logs postgres
docker-compose -f docker-compose.postgres.yml exec postgres pg_isready
```

## Ollama Embedding Setup

### 1. Ollama Production Configuration

Create `docker-compose.ollama.yml`:

```yaml
version: '3.8'

services:
  ollama:
    container_name: lightrag_ollama_prod
    image: ollama/ollama:latest
    restart: always
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
    networks:
      - lightrag-network
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_ORIGINS=*
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 32G
          cpus: '8'
        reservations:
          memory: 16G
          cpus: '4'

networks:
  lightrag-network:
    external: true

volumes:
  ollama_models:
    driver: local
```

### 2. Download and Configure Embedding Model

```bash
# Start Ollama service
docker-compose -f docker-compose.ollama.yml up -d

# Download embedding model (choose one)
docker exec lightrag_ollama_prod ollama pull bge-m3:latest
# OR for better performance
docker exec lightrag_ollama_prod ollama pull nomic-embed-text:latest

# Verify model is available
docker exec lightrag_ollama_prod ollama list

# Test embedding generation
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3:latest", "prompt": "Test embedding generation"}'
```

### 3. Ollama Performance Optimization

Create `ollama-config.json`:

```json
{
  "num_ctx": 4096,
  "num_batch": 512,
  "num_gqa": 8,
  "num_gpu": 1,
  "num_thread": 8,
  "num_predict": -1,
  "temperature": 0.0,
  "top_p": 0.9,
  "repeat_last_n": 64,
  "repeat_penalty": 1.1,
  "seed": -1,
  "tfs_z": 1.0,
  "typical_p": 1.0,
  "mirostat": 0,
  "mirostat_eta": 0.1,
  "mirostat_tau": 5.0
}
```

## xAI Integration

### 1. xAI API Configuration

Create `xai-config.env`:

```bash
# xAI Configuration (Tested and Optimized)
XAI_API_KEY=your_xai_api_key_here
XAI_API_BASE=https://api.x.ai/v1
XAI_MODEL=grok-3-mini
XAI_TIMEOUT=240
XAI_MAX_RETRIES=3

# Performance Settings for xAI
XAI_MAX_ASYNC=2  # Reduced concurrency to prevent timeouts
XAI_TEMPERATURE=0.0
XAI_MAX_TOKENS=4000
```

### 2. xAI Model Options

| Model | Use Case | Performance | Cost |
|-------|----------|-------------|------|
| `grok-3-mini` | **Recommended** - Fast, cost-effective | High speed | Low |
| `grok-2-1212` | High-quality responses | Medium speed | Medium |
| `grok-2-vision-1212` | Vision + text processing | Medium speed | High |

### 3. xAI Rate Limiting & Error Handling

```bash
# xAI-specific optimizations
XAI_RATE_LIMIT_REQUESTS_PER_MINUTE=50
XAI_RATE_LIMIT_TOKENS_PER_MINUTE=40000
XAI_BACKOFF_FACTOR=2
XAI_CIRCUIT_BREAKER_THRESHOLD=5
```

## LightRAG Application Deployment

### 1. Production Environment Configuration

Create `production.env`:

```bash
#####################################
### LightRAG Production Configuration
#####################################

### Server Configuration
HOST=0.0.0.0
PORT=9621
WORKERS=4
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO

### Web UI Configuration
WEBUI_TITLE='Production RAG System'
WEBUI_DESCRIPTION="Enterprise Knowledge Graph RAG System"

### Directory Configuration
WORKING_DIR=/app/data/rag_storage
INPUT_DIR=/app/data/inputs
LOG_DIR=/app/logs

### Authentication Configuration
AUTH_ACCOUNTS='admin:SecureAdminPassword123!,user:UserPassword456!'
TOKEN_SECRET=your_jwt_secret_key_minimum_32_characters
TOKEN_EXPIRE_HOURS=24
GUEST_TOKEN_EXPIRE_HOURS=2
JWT_ALGORITHM=HS256

### API Security
LIGHTRAG_API_KEY=your_secure_api_key_here
WHITELIST_PATHS=/health,/api/health

### Enhanced Security
PASSWORD_MIN_LENGTH=12
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_NUMBERS=true
PASSWORD_REQUIRE_SPECIAL_CHARS=true
PASSWORD_LOCKOUT_ATTEMPTS=5
PASSWORD_LOCKOUT_DURATION_MINUTES=30

### Rate Limiting
RATE_LIMITING_ENABLED=true
RATE_LIMIT_AUTH=10/minute
RATE_LIMIT_GENERAL=200/minute
RATE_LIMIT_UPLOAD=20/minute
RATE_LIMIT_QUERY=100/minute
RATE_LIMIT_GRAPH=60/minute
RATE_LIMIT_ADMIN=500/minute

### Redis for Rate Limiting
REDIS_URL=redis://redis:6379

### Security Headers
SECURITY_ENABLE_CSP=true
SECURITY_ENABLE_HSTS=true
SECURITY_ENABLE_X_HEADERS=true
SECURITY_HIDE_SERVER_HEADER=true

### SSL Configuration
SSL=true
SSL_CERTFILE=/app/certs/fullchain.pem
SSL_KEYFILE=/app/certs/privkey.pem

### Audit Logging
AUDIT_LOG_FILE=/app/logs/audit.log
AUDIT_LOG_LEVEL=INFO
AUDIT_MAX_FILE_SIZE=104857600
AUDIT_BACKUP_COUNT=10
AUDIT_STRUCTURED_LOGGING=true
AUDIT_ASYNC_LOGGING=true

### Query Configuration
ENABLE_LLM_CACHE=true
COSINE_THRESHOLD=0.2
TOP_K=40
CHUNK_TOP_K=10
MAX_ENTITY_TOKENS=10000
MAX_RELATION_TOKENS=10000
MAX_TOTAL_TOKENS=30000
RELATED_CHUNK_NUMBER=5

### Reranking (Optional)
ENABLE_RERANK=false
MIN_RERANK_SCORE=0.0

### Document Processing
SUMMARY_LANGUAGE=English
ENABLE_LLM_CACHE_FOR_EXTRACT=true
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
FORCE_LLM_SUMMARY_ON_MERGE=4
MAX_TOKENS=10000
MAX_GLEANING=1

### Concurrency Configuration (xAI Optimized)
MAX_ASYNC=2  # Reduced for xAI stability
MAX_PARALLEL_INSERT=2
EMBEDDING_FUNC_MAX_ASYNC=4
EMBEDDING_BATCH_NUM=10

### xAI LLM Configuration
TIMEOUT=240
TEMPERATURE=0
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
LLM_BINDING_HOST=https://api.x.ai/v1
LLM_BINDING_API_KEY=${XAI_API_KEY}

### Ollama Embedding Configuration
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
EMBEDDING_BINDING_HOST=http://ollama:11434

### PostgreSQL Storage Configuration
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage

### PostgreSQL Connection
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=lightrag_prod
POSTGRES_PASSWORD=your_secure_postgres_password_here
POSTGRES_DATABASE=lightrag_production
POSTGRES_MAX_CONNECTIONS=25
POSTGRES_SSL_MODE=require
```

### 2. Production Docker Compose

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  # ===================================================================
  # LightRAG Main Application
  # ===================================================================
  lightrag:
    container_name: lightrag_prod
    build:
      context: .
      dockerfile: Dockerfile.production
    restart: always
    ports:
      - "9621:9621"
    volumes:
      - ./data/rag_storage:/app/data/rag_storage
      - ./data/inputs:/app/data/inputs
      - ./logs:/app/logs
      - ./certs:/app/certs:ro
      - ./production.env:/app/.env:ro
    env_file:
      - production.env
    networks:
      - lightrag-network
    depends_on:
      postgres:
        condition: service_healthy
      ollama:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "https://localhost:9621/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=1G
    user: "1000:1000"

  # ===================================================================
  # PostgreSQL Database
  # ===================================================================
  postgres:
    container_name: lightrag_postgres_prod
    image: shangor/postgres-for-rag:v1.0
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DATABASE}
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
      - ./postgres/init:/docker-entrypoint-initdb.d:ro
      - ./backups:/backups
    networks:
      - lightrag-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DATABASE}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '4'
        reservations:
          memory: 8G
          cpus: '2'

  # ===================================================================
  # Ollama Embedding Service
  # ===================================================================
  ollama:
    container_name: lightrag_ollama_prod
    image: ollama/ollama:latest
    restart: always
    volumes:
      - ollama_models:/root/.ollama
    networks:
      - lightrag-network
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_ORIGINS=*
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 32G
          cpus: '8'
        reservations:
          memory: 16G
          cpus: '4'

  # ===================================================================
  # Redis for Rate Limiting
  # ===================================================================
  redis:
    container_name: lightrag_redis_prod
    image: redis:7-alpine
    restart: always
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - lightrag-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
        reservations:
          memory: 1G
          cpus: '0.5'

  # ===================================================================
  # Nginx Reverse Proxy
  # ===================================================================
  nginx:
    container_name: lightrag_nginx_prod
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./certs:/etc/nginx/certs:ro
      - ./logs/nginx:/var/log/nginx
    networks:
      - lightrag-network
    depends_on:
      - lightrag
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  lightrag-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres_prod_data:
    driver: local
  ollama_models:
    driver: local
  redis_data:
    driver: local
```

### 3. Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Security
    server_tokens off;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied expired no-cache no-store private must-revalidate max-age=0;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/json
        application/xml+rss;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=1r/s;

    include /etc/nginx/conf.d/*.conf;
}
```

Create `nginx/conf.d/lightrag.conf`:

```nginx
upstream lightrag_backend {
    server lightrag:9621;
    keepalive 32;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";

    # File upload size
    client_max_body_size 100M;

    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://lightrag_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Authentication endpoints (strict rate limiting)
    location ~ ^/(login|auth-status) {
        limit_req zone=auth burst=5 nodelay;
        proxy_pass http://lightrag_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # API endpoints (moderate rate limiting)
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://lightrag_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300;
        proxy_connect_timeout 10;
        proxy_send_timeout 300;
    }

    # Web UI and other endpoints
    location / {
        limit_req zone=api burst=10 nodelay;
        proxy_pass http://lightrag_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Static assets caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        proxy_pass http://lightrag_backend;
    }
}
```

## Authentication & Security

### 1. User Management

```bash
# Create secure passwords
python3 -c "
import secrets
import string
alphabet = string.ascii_letters + string.digits + '!@#$%^&*'
admin_password = ''.join(secrets.choice(alphabet) for i in range(16))
user_password = ''.join(secrets.choice(alphabet) for i in range(16))
print(f'Admin password: {admin_password}')
print(f'User password: {user_password}')
"

# Generate JWT secret
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. SSL/TLS Certificate Setup

```bash
# Using Let's Encrypt (recommended)
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./certs/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./certs/
sudo chown $USER:$USER ./certs/*.pem
sudo chmod 600 ./certs/*.pem
```

### 3. Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Verify firewall status
sudo ufw status verbose
```

### 4. Security Hardening Script

Create `scripts/security-hardening.sh`:

```bash
#!/bin/bash
set -e

echo "🔒 Applying security hardening..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install security tools
sudo apt install -y fail2ban ufw logwatch rkhunter

# Configure fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo sed -i 's/bantime = 10m/bantime = 1h/' /etc/fail2ban/jail.local
sudo sed -i 's/findtime = 10m/findtime = 20m/' /etc/fail2ban/jail.local
sudo sed -i 's/maxretry = 5/maxretry = 3/' /etc/fail2ban/jail.local

# Start security services
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Configure automatic security updates
echo 'Unattended-Upgrade::Automatic-Reboot "false";' | sudo tee -a /etc/apt/apt.conf.d/50unattended-upgrades
sudo systemctl enable unattended-upgrades

# Secure shared memory
echo 'tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0' | sudo tee -a /etc/fstab

# Disable unused services
sudo systemctl disable avahi-daemon 2>/dev/null || true
sudo systemctl disable cups 2>/dev/null || true
sudo systemctl disable bluetooth 2>/dev/null || true

echo "✅ Security hardening completed"
```

## Monitoring & Observability

### 1. Monitoring Stack Setup

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  # ===================================================================
  # Prometheus
  # ===================================================================
  prometheus:
    container_name: lightrag_prometheus
    image: prom/prometheus:latest
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - lightrag-network

  # ===================================================================
  # Grafana
  # ===================================================================
  grafana:
    container_name: lightrag_grafana
    image: grafana/grafana:latest
    restart: always
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure_grafana_password
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SMTP_ENABLED=true
      - GF_SMTP_HOST=smtp.gmail.com:587
      - GF_SMTP_USER=your-email@gmail.com
      - GF_SMTP_PASSWORD=your-app-password
      - GF_SMTP_FROM_ADDRESS=your-email@gmail.com
    networks:
      - lightrag-network

  # ===================================================================
  # Node Exporter
  # ===================================================================
  node_exporter:
    container_name: lightrag_node_exporter
    image: prom/node-exporter:latest
    restart: always
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - lightrag-network

  # ===================================================================
  # cAdvisor
  # ===================================================================
  cadvisor:
    container_name: lightrag_cadvisor
    image: gcr.io/cadvisor/cadvisor:latest
    restart: always
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker:/var/lib/docker:ro
      - /dev/disk:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - lightrag-network

  # ===================================================================
  # Alertmanager
  # ===================================================================
  alertmanager:
    container_name: lightrag_alertmanager
    image: prom/alertmanager:latest
    restart: always
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    networks:
      - lightrag-network

networks:
  lightrag-network:
    external: true

volumes:
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  alertmanager_data:
    driver: local
```

### 2. Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # LightRAG Application
  - job_name: 'lightrag'
    static_configs:
      - targets: ['lightrag:9621']
    metrics_path: '/metrics'
    scrape_interval: 30s

  # Node Exporter
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node_exporter:9100']

  # cAdvisor
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  # Ollama
  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    metrics_path: '/metrics'

  # Nginx
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
```

### 3. Alert Rules

Create `monitoring/alert_rules.yml`:

```yaml
groups:
  - name: lightrag_alerts
    rules:
      # Application Health
      - alert: LightRAGDown
        expr: up{job="lightrag"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LightRAG application is down"
          description: "LightRAG has been down for more than 1 minute"

      # Database Health
      - alert: PostgreSQLDown
        expr: up{job="postgres"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "PostgreSQL database is not responding"

      # System Resources
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 90% for more than 5 minutes"

      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 85% for more than 10 minutes"

      # Disk Space
      - alert: DiskSpaceLow
        expr: node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"} < 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Disk space running low"
          description: "Less than 10% disk space remaining on {{ $labels.device }}"

      # API Response Time
      - alert: HighAPIResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="lightrag"}[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API response time"
          description: "95th percentile response time is above 5 seconds"

      # xAI API Errors
      - alert: XAIAPIErrors
        expr: rate(lightrag_xai_api_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High xAI API error rate"
          description: "xAI API error rate is above 10% for the last 2 minutes"
```

### 4. Alertmanager Configuration

Create `monitoring/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'your-email@gmail.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email-notifications'

receivers:
  - name: 'email-notifications'
    email_configs:
      - to: 'admin@your-domain.com'
        subject: '🚨 LightRAG Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          Time: {{ .StartsAt }}
          {{ end }}

  - name: 'slack-notifications'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
        title: 'LightRAG Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

### 5. Grafana Dashboard

Create `monitoring/grafana/dashboards/lightrag-dashboard.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "LightRAG Production Dashboard",
    "tags": ["lightrag", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"lightrag\"}",
            "legendFormat": "LightRAG Status"
          }
        ]
      },
      {
        "id": 2,
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"lightrag\"}[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "id": 3,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"lightrag\"}[5m]))",
            "legendFormat": "95th Percentile"
          }
        ]
      },
      {
        "id": 4,
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100",
            "legendFormat": "Memory Available %"
          },
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Backup & Recovery

### 1. Automated Backup Script

Create `scripts/backup-production.sh`:

```bash
#!/bin/bash
set -e

# Configuration
BACKUP_DIR="/opt/lightrag/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "🔄 Starting production backup: $DATE"

# 1. PostgreSQL Database Backup
echo "📊 Backing up PostgreSQL database..."
docker-compose -f docker-compose.production.yml exec -T postgres pg_dump \
    -U lightrag_prod -d lightrag_production --verbose \
    > "$BACKUP_DIR/postgres_backup_$DATE.sql"

# Compress database backup
gzip "$BACKUP_DIR/postgres_backup_$DATE.sql"

# 2. LightRAG Data Backup
echo "📁 Backing up LightRAG data..."
tar -czf "$BACKUP_DIR/rag_storage_backup_$DATE.tar.gz" -C ./data rag_storage

# 3. Configuration Backup
echo "⚙️ Backing up configuration..."
tar -czf "$BACKUP_DIR/config_backup_$DATE.tar.gz" \
    production.env \
    docker-compose.production.yml \
    nginx/ \
    monitoring/ \
    certs/

# 4. Logs Backup
echo "📋 Backing up logs..."
tar -czf "$BACKUP_DIR/logs_backup_$DATE.tar.gz" logs/

# 5. Create backup manifest
echo "📝 Creating backup manifest..."
cat > "$BACKUP_DIR/backup_manifest_$DATE.txt" << EOF
LightRAG Production Backup
Date: $DATE
Files:
- postgres_backup_$DATE.sql.gz
- rag_storage_backup_$DATE.tar.gz
- config_backup_$DATE.tar.gz
- logs_backup_$DATE.tar.gz

System Info:
$(docker-compose -f docker-compose.production.yml ps)

Database Size:
$(docker-compose -f docker-compose.production.yml exec -T postgres psql -U lightrag_prod -d lightrag_production -c "SELECT pg_size_pretty(pg_database_size('lightrag_production'));" -t)

Storage Usage:
$(df -h ./data/rag_storage)
EOF

# 6. Clean old backups
echo "🧹 Cleaning old backups..."
find "$BACKUP_DIR" -name "*backup_*" -type f -mtime +$RETENTION_DAYS -delete

# 7. Upload to cloud storage (optional)
if [ "$CLOUD_BACKUP_ENABLED" = "true" ]; then
    echo "☁️ Uploading to cloud storage..."
    # AWS S3 example
    # aws s3 cp "$BACKUP_DIR/" s3://your-backup-bucket/lightrag/ --recursive --exclude "*" --include "*$DATE*"

    # Google Cloud example
    # gsutil -m cp "$BACKUP_DIR/*$DATE*" gs://your-backup-bucket/lightrag/
fi

# Calculate backup sizes
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "✅ Backup completed successfully!"
echo "📊 Total backup size: $TOTAL_SIZE"
echo "📁 Backup location: $BACKUP_DIR"

# Send notification
if [ "$NOTIFICATION_ENABLED" = "true" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"✅ LightRAG backup completed successfully!\nDate: $DATE\nSize: $TOTAL_SIZE\"}" \
        "$SLACK_WEBHOOK_URL" || true
fi
```

### 2. Recovery Procedures

Create `scripts/restore-production.sh`:

```bash
#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_date>"
    echo "Example: $0 20241201_143000"
    echo ""
    echo "Available backups:"
    ls -la /opt/lightrag/backups/*backup_*.gz | head -10
    exit 1
fi

BACKUP_DATE=$1
BACKUP_DIR="/opt/lightrag/backups"

echo "🔄 Starting production restoration: $BACKUP_DATE"

# Verify backup files exist
for file in "postgres_backup_${BACKUP_DATE}.sql.gz" "rag_storage_backup_${BACKUP_DATE}.tar.gz" "config_backup_${BACKUP_DATE}.tar.gz"; do
    if [ ! -f "$BACKUP_DIR/$file" ]; then
        echo "❌ Backup file not found: $file"
        exit 1
    fi
done

# Stop services
echo "🛑 Stopping services..."
docker-compose -f docker-compose.production.yml down

# 1. Restore PostgreSQL Database
echo "📊 Restoring PostgreSQL database..."
docker-compose -f docker-compose.production.yml up -d postgres
sleep 30

# Drop and recreate database
docker-compose -f docker-compose.production.yml exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS lightrag_production;"
docker-compose -f docker-compose.production.yml exec postgres psql -U postgres -c "CREATE DATABASE lightrag_production OWNER lightrag_prod;"

# Restore database
gunzip -c "$BACKUP_DIR/postgres_backup_${BACKUP_DATE}.sql.gz" | \
    docker-compose -f docker-compose.production.yml exec -T postgres psql -U lightrag_prod -d lightrag_production

# 2. Restore LightRAG Data
echo "📁 Restoring LightRAG data..."
rm -rf ./data/rag_storage
tar -xzf "$BACKUP_DIR/rag_storage_backup_${BACKUP_DATE}.tar.gz" -C ./data

# 3. Restore Configuration (if needed)
if [ "$RESTORE_CONFIG" = "true" ]; then
    echo "⚙️ Restoring configuration..."
    tar -xzf "$BACKUP_DIR/config_backup_${BACKUP_DATE}.tar.gz"
fi

# Start all services
echo "🚀 Starting all services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 60

# Verify restoration
echo "✅ Verifying restoration..."
curl -f http://localhost:9621/health || echo "⚠️ Health check failed"

echo "✅ Restoration completed successfully!"
echo "📊 Restored from backup: $BACKUP_DATE"
```

### 3. Automated Backup Scheduling

Add to crontab:

```bash
# Add to crontab (crontab -e)
# Daily backup at 2 AM
0 2 * * * /opt/lightrag/LightRAG/scripts/backup-production.sh >> /var/log/lightrag-backup.log 2>&1

# Weekly database optimization at 1 AM on Sundays
0 1 * * 0 /opt/lightrag/LightRAG/scripts/maintenance.sh >> /var/log/lightrag-maintenance.log 2>&1
```

## Performance Tuning

### 1. System-Level Optimizations

Create `scripts/performance-tuning.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Applying performance optimizations..."

# Kernel parameters for high-performance applications
cat >> /etc/sysctl.conf << EOF
# Network performance
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 16384 16777216
net.ipv4.tcp_wmem = 4096 16384 16777216
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr

# File system performance
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.swappiness = 10

# PostgreSQL optimizations
kernel.shmmax = 17179869184
kernel.shmall = 4194304
kernel.sem = 250 32000 100 128

# File descriptor limits
fs.file-max = 65536
EOF

# Apply kernel parameters
sysctl -p

# Docker daemon optimization
cat > /etc/docker/daemon.json << EOF
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "Hard": 64000,
      "Name": "nofile",
      "Soft": 64000
    }
  },
  "live-restore": true
}
EOF

# Restart Docker
systemctl restart docker

echo "✅ Performance optimizations applied"
```

### 2. Application-Level Optimizations

Update `production.env` with performance settings:

```bash
# High-performance settings
MAX_ASYNC=2  # Optimized for xAI
MAX_PARALLEL_INSERT=4
EMBEDDING_FUNC_MAX_ASYNC=8
EMBEDDING_BATCH_NUM=20

# Database connection pooling
POSTGRES_MAX_CONNECTIONS=50

# Caching optimizations
ENABLE_LLM_CACHE=true
ENABLE_LLM_CACHE_FOR_EXTRACT=true

# Query optimizations
TOP_K=60
CHUNK_TOP_K=15
MAX_TOTAL_TOKENS=40000
```

### 3. Database Performance Tuning

Update PostgreSQL configuration in `postgres/config/postgresql.conf`:

```conf
# Memory settings (for 16GB RAM server)
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
work_mem = 64MB

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 32MB
max_wal_size = 4GB
min_wal_size = 1GB

# Query planner
random_page_cost = 1.1
effective_io_concurrency = 200
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8

# Connection settings
max_connections = 200
```

## Troubleshooting

### 1. Common Issues and Solutions

#### xAI API Issues

```bash
# Check xAI API connectivity
curl -H "Authorization: Bearer $XAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model": "grok-3-mini", "messages": [{"role": "user", "content": "test"}]}' \
     https://api.x.ai/v1/chat/completions

# Monitor xAI rate limits
docker-compose -f docker-compose.production.yml logs lightrag | grep -i "rate\|limit"

# Solution: Reduce MAX_ASYNC if getting timeouts
echo "MAX_ASYNC=1" >> production.env
```

#### Ollama Embedding Issues

```bash
# Check Ollama service
curl http://localhost:11434/api/tags

# Check model availability
docker exec lightrag_ollama_prod ollama list

# Pull model if missing
docker exec lightrag_ollama_prod ollama pull bge-m3:latest

# Check embedding generation
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3:latest", "prompt": "test"}'
```

#### PostgreSQL Connection Issues

```bash
# Check PostgreSQL status
docker-compose -f docker-compose.production.yml exec postgres pg_isready

# Check connections
docker-compose -f docker-compose.production.yml exec postgres psql \
    -U lightrag_prod -d lightrag_production \
    -c "SELECT count(*) FROM pg_stat_activity;"

# Check locks
docker-compose -f docker-compose.production.yml exec postgres psql \
    -U lightrag_prod -d lightrag_production \
    -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

#### Memory Issues

```bash
# Check memory usage
docker stats

# Check system memory
free -h

# Check swap usage
swapon -s

# Clean up Docker resources
docker system prune -f
docker volume prune -f
```

### 2. Diagnostic Scripts

Create `scripts/diagnose.sh`:

```bash
#!/bin/bash

echo "🔍 LightRAG Production Diagnostics"
echo "=================================="

# System information
echo "📊 System Information:"
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | awk 'NR==2{printf "%.2f GB", $2/1024/1024}')"
echo "Disk Space: $(df -h / | awk 'NR==2{print $4 " available"}')"
echo "Uptime: $(uptime -p)"
echo ""

# Docker status
echo "🐳 Docker Services:"
docker-compose -f docker-compose.production.yml ps
echo ""

# Service health checks
echo "🏥 Health Checks:"
echo -n "LightRAG: "
curl -sf http://localhost:9621/health && echo "✅ OK" || echo "❌ FAILED"
echo -n "PostgreSQL: "
docker-compose -f docker-compose.production.yml exec postgres pg_isready && echo "✅ OK" || echo "❌ FAILED"
echo -n "Ollama: "
curl -sf http://localhost:11434/api/tags && echo "✅ OK" || echo "❌ FAILED"
echo -n "Redis: "
docker-compose -f docker-compose.production.yml exec redis redis-cli ping && echo "✅ OK" || echo "❌ FAILED"
echo ""

# Resource usage
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""

# Recent logs
echo "📋 Recent Errors:"
docker-compose -f docker-compose.production.yml logs --tail=10 lightrag | grep -i error || echo "No recent errors"
echo ""

# Database status
echo "🗄️ Database Status:"
docker-compose -f docker-compose.production.yml exec postgres psql \
    -U lightrag_prod -d lightrag_production \
    -c "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"

# Check certificates
echo "🔒 SSL Certificates:"
if [ -f "./certs/fullchain.pem" ]; then
    openssl x509 -in ./certs/fullchain.pem -noout -dates
else
    echo "❌ SSL certificate not found"
fi
```

### 3. Performance Monitoring

Create `scripts/performance-check.sh`:

```bash
#!/bin/bash

echo "📈 Performance Check"
echo "==================="

# API response time test
echo "🌐 API Response Time:"
time curl -sf http://localhost:9621/health > /dev/null

# Database query performance
echo "🗄️ Database Performance:"
docker-compose -f docker-compose.production.yml exec postgres psql \
    -U lightrag_prod -d lightrag_production \
    -c "SELECT query, total_exec_time, calls, mean_exec_time FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 5;"

# Memory usage by service
echo "💾 Memory Usage by Service:"
docker stats --no-stream --format "{{.Name}}: {{.MemUsage}}"

# Disk I/O
echo "💿 Disk I/O:"
iostat -x 1 1 | grep -E "(Device|sda|nvme)"

# Network connections
echo "🌐 Network Connections:"
netstat -tuln | grep -E ":(80|443|9621|5432|11434|6379)"
```

## Maintenance Procedures

### 1. Regular Maintenance Script

Create `scripts/maintenance.sh`:

```bash
#!/bin/bash
set -e

echo "🔧 Running maintenance procedures..."

# 1. Update system packages (if enabled)
if [ "$AUTO_UPDATE" = "true" ]; then
    echo "📦 Updating system packages..."
    sudo apt update && sudo apt upgrade -y
fi

# 2. Docker cleanup
echo "🧹 Cleaning Docker resources..."
docker system prune -f
docker image prune -f
docker volume prune -f

# 3. Database maintenance
echo "🗄️ Database maintenance..."
docker-compose -f docker-compose.production.yml exec postgres psql \
    -U lightrag_prod -d lightrag_production \
    -c "VACUUM ANALYZE;" > /dev/null

# 4. Log rotation
echo "📋 Rotating logs..."
find ./logs -name "*.log" -size +100M -exec gzip {} \;
find ./logs -name "*.gz" -mtime +30 -delete

# 5. Certificate renewal check
echo "🔒 Checking SSL certificates..."
if openssl x509 -checkend 604800 -noout -in ./certs/fullchain.pem; then
    echo "✅ Certificate is valid for at least 7 days"
else
    echo "⚠️ Certificate expires within 7 days - renewal needed"
    # Auto-renewal with certbot
    sudo certbot renew --quiet
fi

# 6. Health check all services
echo "🏥 Health check..."
./scripts/diagnose.sh > /dev/null

# 7. Update monitoring dashboards
echo "📊 Updating monitoring..."
curl -sf http://localhost:3000/api/health > /dev/null || echo "⚠️ Grafana not responding"

# 8. Backup verification
echo "💾 Backup verification..."
LATEST_BACKUP=$(ls -t /opt/lightrag/backups/*postgres_backup_*.sql.gz | head -1)
if [ -n "$LATEST_BACKUP" ]; then
    echo "✅ Latest backup: $(basename $LATEST_BACKUP)"
else
    echo "⚠️ No recent backups found"
fi

echo "✅ Maintenance completed successfully"
```

### 2. Update Procedures

Create `scripts/update-production.sh`:

```bash
#!/bin/bash
set -e

echo "🔄 Updating LightRAG production deployment..."

# Backup before update
echo "💾 Creating pre-update backup..."
./scripts/backup-production.sh

# Pull latest code
echo "📥 Pulling latest code..."
git fetch origin
git pull origin main

# Update Docker images
echo "🐳 Updating Docker images..."
docker-compose -f docker-compose.production.yml pull

# Restart services with zero downtime
echo "🔄 Restarting services..."
docker-compose -f docker-compose.production.yml up -d --no-deps lightrag

# Wait for health check
echo "⏳ Waiting for health check..."
sleep 30

# Verify deployment
if curl -sf http://localhost:9621/health; then
    echo "✅ Update completed successfully"
else
    echo "❌ Update failed - rolling back..."
    ./scripts/restore-production.sh $(date +%Y%m%d)
    exit 1
fi
```

## Deployment Checklist

### Pre-Deployment Checklist

- [ ] Server meets minimum requirements
- [ ] Domain name configured with DNS
- [ ] SSL certificates obtained and installed
- [ ] xAI API key obtained and tested
- [ ] Backup storage configured
- [ ] Monitoring alerts configured
- [ ] Firewall rules applied
- [ ] Security hardening completed

### Deployment Steps

1. **Infrastructure Setup**
   ```bash
   # 1. Clone repository
   git clone https://github.com/HKUDS/LightRAG.git
   cd LightRAG

   # 2. Apply security hardening
   ./scripts/security-hardening.sh

   # 3. Configure environment
   cp production.env.example production.env
   # Edit production.env with your settings
   ```

2. **Service Deployment**
   ```bash
   # 1. Start PostgreSQL
   docker-compose -f docker-compose.postgres.yml up -d

   # 2. Start Ollama and download models
   docker-compose -f docker-compose.ollama.yml up -d
   docker exec lightrag_ollama_prod ollama pull bge-m3:latest

   # 3. Start main application
   docker-compose -f docker-compose.production.yml up -d

   # 4. Start monitoring stack
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

3. **Verification**
   ```bash
   # Run diagnostics
   ./scripts/diagnose.sh

   # Test all endpoints
   curl https://your-domain.com/health
   curl https://your-domain.com/webui
   curl https://your-domain.com/docs
   ```

### Post-Deployment Checklist

- [ ] All services running and healthy
- [ ] Web UI accessible
- [ ] API endpoints responding
- [ ] Authentication working
- [ ] Monitoring dashboards configured
- [ ] Alerts configured and tested
- [ ] Backup system verified
- [ ] SSL certificates valid
- [ ] Performance baseline established
- [ ] Documentation updated

## Conclusion

This comprehensive guide provides a complete production deployment of LightRAG with enterprise-grade features:

- **High Availability**: Load balancer, health checks, automatic restarts
- **Security**: Authentication, SSL/TLS, rate limiting, security headers
- **Monitoring**: Prometheus, Grafana, alerting, performance metrics
- **Backup & Recovery**: Automated backups, point-in-time recovery
- **Performance**: Optimized configurations, resource management
- **Maintenance**: Automated maintenance, update procedures

The deployment is designed for enterprise use with proper security, monitoring, and maintenance procedures. Follow the checklist and procedures for a successful production deployment.

For support and troubleshooting, refer to the diagnostic scripts and monitoring dashboards provided in this guide.
