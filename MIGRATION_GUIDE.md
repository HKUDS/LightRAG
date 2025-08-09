# LightRAG Migration Guide

**Version**: Pre-Release to v1.5.0
**Last Updated**: 2025-01-29
**Target Audience**: Developers, DevOps Engineers, System Administrators

---

## 📋 Overview

This guide helps you migrate from earlier LightRAG versions to the upcoming v1.5.0 stable release. It covers breaking changes, configuration updates, and new features that may affect your deployment.

## 🚨 Critical Breaking Changes

### 1. Deprecated Function Removal (v0.2.0)
**Impact**: HIGH - Functions will be completely removed

**Affected Functions**:
```python
# ❌ DEPRECATED - Will be removed in v0.2.0
lightrag.search_old()          # lightrag/lightrag.py:727
lightrag.query_deprecated()    # lightrag/lightrag.py:739
lightrag.insert_legacy()       # lightrag/lightrag.py:1694
lightrag.batch_process_old()   # lightrag/lightrag.py:1716
operate.handle_deprecated()    # lightrag/operate.py:3006
```

**Migration Actions**:
```python
# ✅ NEW - Use these replacements
lightrag.search()              # New search method
lightrag.query()               # Standard query function
lightrag.insert()              # Current insert method
lightrag.batch_process()       # New batch processing
operate.handle()               # Current operation handler
```

### 2. Storage Backend Lock-in
**Impact**: HIGH - Cannot change after document addition

**Current Limitation**:
- Once documents are added, storage backends cannot be changed
- Requires complete data migration to switch backends

**Migration Strategy**:
```bash
# 1. Export existing data (if possible)
python export_data.py --output backup_data.json

# 2. Initialize new storage backend
# Update .env with new storage configuration

# 3. Re-process all documents with new backend
python reprocess_documents.py --input backup_data.json
```

### 3. Configuration File Format Changes
**Impact**: MEDIUM - Environment file updates required

**Required Changes**:
```bash
# ❌ OLD FORMAT (.env)
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=ollama

# ✅ NEW FORMAT (.env)
LLM_BINDING=openai
EMBEDDING_BINDING=ollama
```

## 🔄 Configuration Migration

### Environment File Updates

**1. Copy New Template**:
```bash
# Backup existing configuration
cp .env .env.backup

# Copy new template
cp env.example .env

# Merge your settings from backup
```

**2. Key Configuration Changes**:

| Old Variable | New Variable | Notes |
|-------------|-------------|--------|
| `LLM_PROVIDER` | `LLM_BINDING` | Renamed for consistency |
| `EMBEDDING_PROVIDER` | `EMBEDDING_BINDING` | Renamed for consistency |
| `MAX_CONCURRENT` | `MAX_ASYNC` | Clearer naming |
| `JWT_SECRET` | `JWT_SECRET_KEY` | More explicit |

**3. New Required Variables**:
```bash
# Add these new variables
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO
RATE_LIMIT_ENABLED=true
AUTH_ENABLED=true
JWT_EXPIRE_HOURS=24
```

### xAI Integration Updates

**Critical Configuration**:
```bash
# ✅ REQUIRED for xAI stability
MAX_ASYNC=2  # Prevents timeout issues
XAI_API_KEY=your_key_here
XAI_API_BASE=https://api.x.ai/v1
```

**Available Models**:
```bash
# Supported xAI models
LLM_MODEL=grok-3-mini          # Recommended
LLM_MODEL=grok-2-1212          # Alternative
LLM_MODEL=grok-2-vision-1212   # Vision support
```

## 🏗️ Infrastructure Migration

### Docker Deployment Updates

**1. New Production Docker Compose**:
```bash
# ❌ OLD - Development only
docker-compose up

# ✅ NEW - Production ready with security
docker-compose -f docker-compose.production.yml up -d
```

**2. Security-Hardened Containers**:
```dockerfile
# New security features in Dockerfile.production
USER 1000:1000                    # Non-root user
WORKDIR /app --read-only          # Read-only filesystem
CAPABILITIES --drop=ALL           # Minimal capabilities
```

**3. Production Environment Setup**:
```bash
# 1. Copy production template
cp production.env .env

# 2. Configure database credentials
vim .env  # Edit database settings

# 3. Start production stack
docker-compose -f docker-compose.production.yml up -d
```

### Kubernetes Migration

**New K8s Support**:
```bash
cd k8s-deploy/

# 1. Install dependencies (databases)
./databases/01-prepare.sh
./databases/02-install-database.sh

# 2. Deploy LightRAG
./install_lightrag.sh

# 3. Verify deployment
kubectl get pods -n lightrag
```

## 🔐 Security Migration

### Authentication System Updates

**1. JWT Configuration**:
```bash
# Generate secure JWT secret (>32 characters)
openssl rand -hex 32

# Add to .env
JWT_SECRET_KEY=your_generated_secret_here
JWT_EXPIRE_HOURS=24
```

**2. Rate Limiting Setup**:
```bash
# Enable rate limiting
RATE_LIMIT_ENABLED=true

# Configure limits (optional)
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

**3. Audit Logging**:
```bash
# Audit logs location
logs/audit.log

# Monitor with
tail -f logs/audit.log
```

## 📊 Database Migration

### PostgreSQL Production Setup

**1. Database Initialization**:
```sql
-- Create production database
CREATE DATABASE lightrag;
CREATE USER lightrag WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE lightrag TO lightrag;
```

**2. Connection Configuration**:
```bash
# Update .env with PostgreSQL settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lightrag
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=secure_password
```

### Redis Caching Setup

**1. Redis Configuration**:
```bash
# Add Redis for improved performance
REDIS_URL=redis://localhost:6379/0
```

**2. Cache Verification**:
```bash
# Test Redis connection
docker-compose exec redis redis-cli ping
# Should return: PONG
```

## 🧪 Testing Migration

### New Test Framework

**1. Install Test Dependencies**:
```bash
# Install testing requirements
pip install -e ".[test]"
```

**2. Run Test Suite**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lightrag tests/

# Run specific test categories
pytest tests/test_health.py -v
```

**3. Health Check Validation**:
```bash
# Test health endpoints
curl http://localhost:9621/health
curl http://localhost:9621/api/health
```

## 📱 Web UI Migration

### React Frontend Setup

**1. Install Dependencies**:
```bash
cd lightrag_webui/

# Using Bun (recommended)
bun install
bun run dev

# Using Node.js
npm install
npm run dev-no-bun
```

**2. Production Build**:
```bash
# Build for production
bun run build

# Serve built assets
bun run preview
```

## 🔧 API Migration

### New API Endpoints

**1. Updated Health Checks**:
```bash
# Basic health
GET /health

# Detailed health with dependencies
GET /api/health
```

**2. Authentication Endpoints**:
```bash
# Login (if auth enabled)
POST /api/auth/login

# Token verification
GET /api/auth/verify
```

**3. Ollama Compatibility**:
```bash
# Chat interface (Ollama-compatible)
POST /api/chat
```

## 🔄 MCP Integration

### Model Context Protocol Setup

**1. Install MCP Dependencies**:
```bash
pip install mcp httpx pydantic aiofiles typing-extensions
```

**2. Start MCP Server**:
```bash
# Start MCP server
python -m lightrag_mcp

# Configure environment
LIGHTRAG_API_URL=http://localhost:9621
MCP_ENABLE_STREAMING=true
```

**3. Claude CLI Integration**:
```bash
# Setup Claude CLI with MCP
claude config mcp add lightrag-mcp python -m lightrag_mcp

# Test functionality
claude mcp lightrag_health_check
```

## 📋 Post-Migration Checklist

### ✅ Verification Steps

**1. Core Functionality**:
- [ ] Documents upload successfully
- [ ] Queries return expected results
- [ ] Storage backends are working
- [ ] LLM integrations are functional

**2. Production Features**:
- [ ] Authentication is working (if enabled)
- [ ] Rate limiting is active
- [ ] Audit logs are being written
- [ ] Health checks pass

**3. Performance**:
- [ ] Response times are acceptable
- [ ] Memory usage is reasonable
- [ ] Concurrent operations work
- [ ] Error handling is proper

**4. Security**:
- [ ] JWT tokens are secure
- [ ] API keys are protected
- [ ] Container security is active
- [ ] Network access is controlled

## 🚨 Rollback Plan

### If Migration Fails

**1. Restore Backup**:
```bash
# Restore environment file
cp .env.backup .env

# Restore data (if applicable)
cp -r rag_storage.backup/ rag_storage/
```

**2. Restart Services**:
```bash
# Stop new version
docker-compose -f docker-compose.production.yml down

# Start old version
docker-compose up -d
```

**3. Verify Rollback**:
```bash
# Test basic functionality
curl http://localhost:9621/health
```

## 📞 Support & Troubleshooting

### Common Migration Issues

**1. xAI Timeout Errors**:
```bash
# Solution: Set MAX_ASYNC=2
echo "MAX_ASYNC=2" >> .env
```

**2. Authentication Failures**:
```bash
# Check JWT secret length (must be >32 chars)
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> .env
```

**3. Database Connection Issues**:
```bash
# Verify database is running
docker-compose exec postgres psql -U lightrag -d lightrag -c "SELECT version();"
```

### Getting Help

- **Documentation**: `docs/DOCUMENTATION_INDEX.md`
- **Known Issues**: `KNOWN_ISSUES.md`
- **Security Guide**: `docs/security/SECURITY_HARDENING.md`
- **GitHub Issues**: Report with detailed error logs

---

## 📅 Migration Timeline

### Immediate Actions (Pre v1.5.0)
- [ ] Update deprecated function calls
- [ ] Migrate configuration files
- [ ] Test new authentication system
- [ ] Verify xAI integration settings

### Next Release Preparation (v2.0.0)
- [ ] Plan storage backend migration strategy
- [ ] Evaluate new security features
- [ ] Review performance optimizations
- [ ] Update integration patterns

---

**⚠️ Important**: Always test migrations in a development environment before applying to production systems. Keep backups of all configuration and data files.
