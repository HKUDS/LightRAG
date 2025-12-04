# LightRAG Multi-Tenant Docker - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup Environment
```bash
cd starter
make setup           # Initialize .env from template
```

### Step 2: Start Services with Multi-Tenant Demo Mode
```bash
make up             # Starts all services (PostgreSQL, Redis, API, WebUI)
make init-db        # Initialize database with 2 demo tenants
```

### Step 3: Access the Services
```
Web UI:     http://localhost:3001
API Docs:   http://localhost:8000/docs
API Base:   http://localhost:8000
```

## 📊 Default Multi-Tenant Configuration

By default, the system starts with **MULTITENANT_MODE=demo** which provides:

### Pre-configured Demo Tenants

| Tenant | Knowledge Bases | Purpose |
|--------|-----------------|---------|
| **acme-corp** | kb-prod, kb-dev | Enterprise tenant with production and dev KBs |
| **techstart** | kb-main, kb-backup | Startup tenant with main and backup KBs |

### Using the Multi-Tenant API

All API requests should include tenant headers:

```bash
# Example: Query as acme-corp
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-Tenant-ID: acme-corp" \
  -H "X-KB-ID: kb-prod" \
  -H "Content-Type: application/json" \
  -d '{"query": "your query here"}'

# Example: Query as techstart
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-Tenant-ID: techstart" \
  -H "X-KB-ID: kb-main" \
  -H "Content-Type: application/json" \
  -d '{"query": "your query here"}'
```

## 📋 Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| Web UI | http://localhost:3001 | User interface |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| API Redoc | http://localhost:8000/redoc | Alternative API docs |
| API Health | http://localhost:8000/health | API health check |
| PostgreSQL | internal-only (container network) | Database (not exposed to host by default) |
| Redis | localhost:6379 | Cache (internal use) |

## 🎯 Useful Make Commands

```bash
# Service Management
make up              # Start all services
make down            # Stop all services
make restart         # Restart all services
make status          # Show service status
make logs            # Stream logs from all services
make logs-api        # Stream logs from API only

# Database Operations
make init-db         # Initialize database with multi-tenant schema
make db-shell        # Connect to PostgreSQL shell
make db-backup       # Create database backup
make db-restore      # Restore from backup
make db-reset        # Reset database (WARNING: deletes all data)

# Testing & Health
make api-health      # Check API health
make test            # Run tests for current mode
make test-all-scenarios  # Run all 3 testing scenarios

# Cleanup
make clean           # Remove stopped containers
make reset           # Full system reset (WARNING: deletes volumes)
```

## 🔐 Demo Database Credentials (local/dev only)

```text
PostgreSQL User:     lightrag
PostgreSQL Password: lightrag_secure_password
PostgreSQL Database: lightrag_multitenant
PostgreSQL Host:     postgres (inside Docker container)
PostgreSQL Port:     5432 (internal-only; not forwarded to host)

Demo Login:
Username: admin
Password: password
```

## 🛠️ Testing Different Modes

### Multi-Tenant Demo Mode (DEFAULT)
```bash
# Already set in .env as MULTITENANT_MODE=demo
make up
make init-db

# Test with both tenants
curl -H "X-Tenant-ID: acme-corp" -H "X-KB-ID: kb-prod" \
  http://localhost:8000/health
```

### Single-Tenant Isolation Mode
```bash
# Switch to single tenant with multiple KBs
echo "MULTITENANT_MODE=on" >> .env
make restart
make init-db
```

### Backward Compatibility Mode
```bash
# Switch to single-tenant compatibility (like main branch)
echo "MULTITENANT_MODE=off" >> .env
make restart
make init-db

# No tenant headers required
curl http://localhost:8000/health
```

## 📊 Database Tables Created

The initialization script (`init-postgres.sql`) creates:

**Multi-Tenant Metadata**:
- `tenants` - Tenant information
- `knowledge_bases` - KB metadata per tenant

**Document Storage** (tenant-isolated):
- `documents` - Document metadata
- `document_chunks` - Document segments
- `document_status` - Processing status
- `embeddings_document` - Document embeddings

**Knowledge Graph** (tenant-isolated):
- `entities` - Entity records
- `relations` - Relationship records
- `embeddings_entity` - Entity embeddings
- `embeddings_relation` - Relation embeddings

**Caching & Storage** (tenant-isolated):
- `kv_storage` - Key-value cache
- `llm_cache` - LLM response cache

All tables use composite keys: `(tenant_id, kb_id, id)`

## 🔍 Verify Multi-Tenant Isolation

### From the Database
```bash
# Connect to database
make db-shell

# Check tenants
SELECT tenant_id, name FROM tenants;

# Check KBs
SELECT tenant_id, kb_id, name FROM knowledge_bases;

# Verify data isolation
SELECT COUNT(*) FROM documents WHERE tenant_id='acme-corp';
SELECT COUNT(*) FROM documents WHERE tenant_id='techstart';

# Exit
\q
```

### From the API
```bash
# Health check
curl http://localhost:8000/health

# API docs (shows available endpoints)
curl http://localhost:8000/docs
```

## 🆘 Troubleshooting

### Services Won't Start
```bash
# Check if ports are in use
lsof -i :3001  # WebUI
lsof -i :8000  # API
lsof -i :5432  # PostgreSQL (if published)

# View logs for errors
make logs

# Kill conflicting process and restart
kill -9 <PID>
make restart
```

### Database Connection Issues
```bash
# Check PostgreSQL is healthy
make status

# View database logs
make logs-db

# Verify database exists
make db-shell
\l  # List databases
\q  # Exit
```

### API Not Responding
```bash
# Check API health
make api-health

# Check API logs
make logs-api

# If needed, restart API
docker compose -p lightrag-multitenant restart lightrag-api
```

### Reset Everything
```bash
# WARNING: This deletes all data!
make reset           # Full system reset (stops, removes containers and volumes)
make setup           # Re-initialize
make up              # Start fresh
make init-db         # Initialize with clean database
```

## 📚 Configuration Options

Edit `.env` to customize:

```env
# Testing Mode (MUST be demo, on, or off)
MULTITENANT_MODE=demo    # demo ← DEFAULT for testing with 2 tenants

# Default Tenant/KB (when not provided in headers)
DEFAULT_TENANT=default
DEFAULT_KB=default

# LLM Configuration
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_API_KEY=your_key_here

# Embedding Model
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_BINDING_HOST=http://host.docker.internal:11434

# Database
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=lightrag_secure_password
POSTGRES_DATABASE=lightrag_multitenant

# Server Ports
API_PORT=8000
WEBUI_PORT=3001
```

## 🎯 Key Features

✅ **Multi-Tenant Isolation**: Complete data separation between tenants  
✅ **Multiple KBs per Tenant**: Each tenant can have multiple knowledge bases  
✅ **Composite Keys**: Database-level isolation with (tenant_id, kb_id, id)  
✅ **Vector Search**: pgvector with HNSW indexing  
✅ **Graph Database**: Apache AGE for entity relationships  
✅ **Caching**: Redis for LLM response caching  
✅ **Health Checks**: Automatic service health monitoring  

## 📖 More Information

For detailed testing strategies and advanced configuration, see:
- `docs/adr/008-multi-tenant-testing-strategy.md` - Complete testing guide
- `QUICK_REFERENCE.md` - Command reference guide
- `README.md` - Project overview

## ⚡ Performance Tips

### For Better Search
- Use appropriate tenant and KB headers
- Database indexes are automatically created
- Vector search uses HNSW for fast similarity search

### For Better Caching
- LLM responses are cached in Redis by default
- Set `ENABLE_LLM_CACHE=true` (default)

### For Multi-Tenant Workloads
- Data is automatically isolated by tenant_id and kb_id
- No risk of cross-tenant data leakage

---

**Last Updated**: November 22, 2025  
**Status**: ✅ Ready to Use - Multi-Tenant Demo Mode Enabled by Default  
**Platform**: macOS/Linux/Windows (with Docker)
