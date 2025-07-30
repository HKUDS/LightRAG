# 🚀 LightRAG Production Deployment Guide

This comprehensive guide walks you through deploying LightRAG in a production environment with enterprise-grade security, monitoring, and reliability features.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Configuration](#detailed-configuration)
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
git clone https://github.com/HKUDS/LightRAG.git
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
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f lightrag
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

### Docker Services Configuration

#### Application Service
- **Image**: Custom production-optimized image
- **Resources**: 8GB RAM, 4 CPU cores
- **Health Checks**: Multi-tier health monitoring
- **Security**: Non-root user, readonly filesystem options

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

## 🔐 Security Setup

### SSL/TLS Configuration

1. **Generate SSL Certificates**:

```bash
# Using Let's Encrypt (recommended)
certbot certonly --standalone -d your-domain.com

# Copy certificates
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem certs/cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem certs/key.pem
```

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

#### 2. Database Connection Issues

```bash
# Test database connectivity
docker exec -it lightrag_postgres psql -U lightrag_prod -d lightrag_production

# Check database logs
docker-compose -f docker-compose.production.yml logs postgres
```

#### 3. High Memory Usage

```bash
# Monitor resource usage
docker stats

# Tune worker settings
WORKERS=2                   # Reduce workers
LLM_MAX_ASYNC=2            # Reduce concurrent requests
MAX_PARALLEL_INSERT=2      # Reduce parallel processing
```

#### 4. Performance Issues

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