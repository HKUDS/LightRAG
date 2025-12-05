# Quick Reference Guide - LightRAG Multi-Tenant Stack

## 🚀 First Time Setup (5 minutes)

```bash
# 1. Enter the starter directory
cd starter

# 2. Initialize environment
make setup

# 3. Edit .env with your API keys (if needed)
# nano .env

# 4. Start all services
make up

# 5. Initialize database (wait 30 seconds after 'make up')
make init-db

# 6. Verify everything is working
make api-health

# 7. Open in browser
# http://localhost:3000
```

---

## 📊 Monitoring & Debugging

```bash
# View all service logs in real-time
make logs

# View only API logs
make logs-api

# Check service status
make status

# Check API health
make api-health

# Connect to database
make db-shell
```

---

## 🗄️ Database Operations

```bash
# Create a backup
make db-backup
# Creates: ./backups/lightrag_backup_YYYYMMDD_HHMMSS.sql

# Restore from latest backup
make db-restore

# Connect to database shell
make db-shell
# Common commands:
# \dt                    - list all tables
# \di                    - list all indexes
# \d documents           - show documents table structure
# SELECT count(*) FROM documents;
# SELECT distinct tenant_id FROM documents;
# \q                     - quit

# WARNING: Delete and reinitialize database
make db-reset
```

---

## 🔄 Service Management

```bash
# Start services
make up

# Stop services
make down

# Restart services
make restart

# See what's running
make ps

# Stop and remove everything (keep data)
make down

# Complete reset (WARNING: deletes all data!)
make reset
```

---

## 🧪 Testing Multi-Tenant Features

```bash
# Test with curl - Insert document for tenant "acme-corp"
curl -X POST http://localhost:9621/api/v1/insert \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Id: acme-corp" \
  -H "X-KB-Id: kb-prod" \
  -d '{"document": "Sample document content"}'

# Test isolation - query as different tenant
curl "http://localhost:9621/api/v1/query" \
  -H "X-Tenant-Id: acme-corp" \
  -H "X-KB-Id: kb-prod" \
  -G --data-urlencode "param=test"

# Check isolation
curl "http://localhost:9621/api/v1/query" \
  -H "X-Tenant-Id: techstart" \
  -H "X-KB-Id: kb-main" \
  -G --data-urlencode "param=test"

# Run test suite
make test

# Run isolation tests
make test-isolation
```

---

## 🔧 Configuration Cheat Sheet

### Change API Port
```bash
# In .env
API_PORT=9622

# Restart
make restart
```

### Change WebUI Port
```bash
# In .env
WEBUI_PORT=3001

# Restart
make restart
```

### Change Database Password
```bash
# In .env
POSTGRES_PASSWORD=your_new_password

# WARNING: Database must be reset if already initialized
make db-reset
make init-db
```

### Use Different LLM (OpenAI)
```bash
# In .env
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_BINDING_API_KEY=sk-...

# Restart API
docker compose -p lightrag-multitenant restart lightrag-api
```

### Use Different Embedding Model
```bash
# In .env
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY=sk-...

# Restart API
docker compose -p lightrag-multitenant restart lightrag-api
```

---

## 📱 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| WebUI | `http://localhost:3000` | Upload documents, visualize graph, query |
| API | `http://localhost:9621` | RESTful API access |
| API Health | `http://localhost:9621/health` | Check API status |
| PostgreSQL | `localhost:5432` | Database (use `make db-shell`) |
| Redis | `localhost:6379` | Cache (internal) |

---

## 🐛 Troubleshooting Quick Fixes

### "Port already in use"
```bash
# Find process using port 9621
lsof -i :9621

# Kill it
kill -9 <PID>

# Or change port in .env and restart
```

### "Database is not ready"
```bash
# Wait longer and try again
sleep 30
make init-db

# Or check logs
make logs-db
```

### "API not responding"
```bash
# Restart API
docker compose -p lightrag-multitenant restart lightrag-api

# Check logs
make logs-api

# Check health
make api-health
```

### "Can't connect to database"
```bash
# Check database is running
make status

# Try connecting directly
make db-shell

# Reinitialize if needed
make db-reset
make init-db
```

### "WebUI won't load"
```bash
# Check if it's running
make status

# View logs
make logs-webui

# Rebuild if needed
docker compose -p lightrag-multitenant down
docker compose -p lightrag-multitenant build --no-cache lightrag-webui
make up
```

---

## 📋 Sample Tenants & KBs

Default data created by `make init-db`:

```
Tenant: acme-corp
├── kb-prod     (Production knowledge base)
└── kb-dev      (Development knowledge base)

Tenant: techstart
├── kb-main     (Main knowledge base)
└── kb-backup   (Backup knowledge base)
```

---

## 🔐 Multi-Tenant Isolation

### How to Create Document for Specific Tenant

```python
import requests

headers = {
    "X-Tenant-Id": "acme-corp",
    "X-KB-Id": "kb-prod",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:9621/api/v1/insert",
    headers=headers,
    json={"document": "Your document content"}
)
```

### How to Verify Isolation

```bash
# Insert for tenant "acme-corp"
curl -X POST http://localhost:9621/api/v1/insert \
  -H "X-Tenant-Id: acme-corp" \
  -H "X-KB-Id: kb-prod" \
  -d '{"document": "acme-corp only"}'

# Query as "acme-corp" - should return data
curl "http://localhost:9621/api/v1/query" \
  -H "X-Tenant-Id: acme-corp" \
  -H "X-KB-Id: kb-prod" \
  -G --data-urlencode "param=acme"

# Query as "techstart" - should NOT return data
curl "http://localhost:9621/api/v1/query" \
  -H "X-Tenant-Id: techstart" \
  -H "X-KB-Id: kb-main" \
  -G --data-urlencode "param=acme"
  # Result: Empty or different data
```

---

## 💾 Backup & Restore Workflow

```bash
# Before making major changes
make db-backup
# Creates: ./backups/lightrag_backup_20251120_143022.sql

# Do your work...

# If something goes wrong
make db-restore

# Or manually restore specific backup
psql -U lightrag -d lightrag_multitenant < ./backups/lightrag_backup_20251120_143022.sql
```

---

## 🚨 Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Connection refused" | Services not running | `make up` |
| "Database does not exist" | Database not initialized | `make init-db` |
| "Port already in use" | Port conflict | Change port in `.env` |
| "Health check failed" | Service taking too long to start | Wait 30-60 seconds |
| "Permission denied" | Database permissions | `make db-reset && make init-db` |
| "Tenant not found" | Wrong tenant name | Check with `make db-shell` |

---

## 📚 Full Help

```bash
make help
```

Shows all available commands with descriptions.

---

## 🔗 Useful Links

- **LightRAG Docs**: https://github.com/HKUDS/LightRAG
- **Docker Docs**: https://docs.docker.com/
- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **pgvector Docs**: https://github.com/pgvector/pgvector

---

## ✅ Health Check Verification

```bash
# API Health
curl http://localhost:9621/health

# Database Connection
make db-shell
# SELECT count(*) FROM tenants;
# \q

# Redis Connection
docker compose -p lightrag-multitenant exec redis redis-cli -a redis_secure_password ping
```

---

## 🎯 Next Steps

1. ✅ **Setup Complete** - Services are running
2. 📤 **Upload Documents** - Use WebUI at http://localhost:3000
3. 🔍 **Create Queries** - Ask questions about your documents
4. 📊 **Visualize KB** - View knowledge graph in WebUI
5. 🚀 **Go to Production** - Configure security, backups, monitoring

---

**Version**: 1.0
**Last Updated**: November 20, 2025
**Status**: Production Ready
