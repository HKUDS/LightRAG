# LightRAG Multi-Tenant Stack with PostgreSQL

A complete, production-ready multi-tenant RAG (Retrieval-Augmented Generation) system using LightRAG with PostgreSQL as the backend.

## 🚀 Quick Start

```bash
# 1. Initialize environment (first time only)
make setup

# 2. Start all services
make up

# 3. Initialize database schema
make init-db

# 4. View service status
make status

# 5. Access the application
# WebUI:       http://localhost:3001
# API Server:  http://localhost:8000
# API Docs:    http://localhost:8000/docs

## 🔐 Demo credentials (local/dev only)

Use the following defaults when running the stack locally for demonstrations or testing. These come from `starter/env.example` — change them in `.env` for any shared or production deployments.

```
User:     lightrag
Password: lightrag_secure_password
Database: lightrag_multitenant
Host:     postgres (inside Docker)
Port:     5432 (internal-only; not forwarded to localhost by default)
```

⚠️ Note: These credentials are for development/demos only. Always pick strong, unique passwords for production and avoid committing secrets to source control.
```

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  LightRAG Multi-Tenant Stack                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Web UI (React)                        │  │
│  │              http://localhost:3001                       │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                        │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │         LightRAG API Server (FastAPI)                   │  │
│  │             http://localhost:8000                       │  │
│  │                                                          │  │
│  │  Multi-Tenant Context:  (tenant_id, kb_id)             │  │
│  │  - Enforces data isolation at API level                │  │
│  │  - Routes queries to appropriate backends              │  │
│  │  - Manages document processing                         │  │
│  └────────────────────┬────────────────────────────────────┘  │
│                       │                                        │
│       ┌───────────────┼───────────────┐                       │
│       │               │               │                       │
│  ┌────▼──────┐  ┌─────▼──────┐  ┌────▼──────┐               │
│  │ PostgreSQL│  │   Redis    │  │  Embedding│               │
│  │ Storage   │  │   Cache    │  │  Service  │               │
│  │           │  │            │  │  (Ollama) │               │
│  │ - KV      │  │ LLM cache  │  │           │               │
│  │ - Documents│ │ Session    │  │ bge-m3    │               │
│  │ - Entities│  │ Temporary  │  │           │               │
│  │ - Relations│ │ Data       │  │           │               │
│  │ - Vectors │  │            │  │           │               │
│  └───────────┘  └────────────┘  └───────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

### Multi-Tenant Data Isolation
- **Composite Key Pattern**: Each resource identified by `(tenant_id, kb_id, resource_id)`
- **Database-Level Enforcement**: Queries automatically scoped to tenant/KB
- **Cross-Tenant Access Prevention**: Impossible to retrieve data from other tenants
- **Complete Isolation**: Works across all 10 storage backends

### Storage Architecture
- **PostgreSQL**: Primary storage with pgvector extension
  - Key-Value storage (PGKVStorage)
  - Document metadata (PGDocStatusStorage)
  - Knowledge graph (PGGraphStorage)
  - Vector embeddings (PGVectorStorage)
- **Redis**: Caching and session management
- **Embedding Service**: Ollama (configurable to OpenAI, Jina, etc.)

### Supported Tenants & Knowledge Bases
Default sample data (automatically created):
```
Tenant: 595ea68b-0f3a-4dbe-8a86-9276a1bbd10c
  └─ kb-prod     (Production KB)
  └─ kb-dev      (Development KB)

Tenant: 44bf3e0d-d633-4dea-9b74-3e24140cd7e3
  └─ kb-main     (Main KB)
  └─ kb-backup   (Backup KB)
```

## 📖 Makefile Commands

### Setup & Configuration
```bash
make help              # Show all available commands
make setup             # Initialize .env file (first time only)
make init-db           # Initialize PostgreSQL database schema
```

### Service Control
```bash
make up                # Start all services
make down              # Stop all services
make restart           # Restart all services
make logs              # Stream logs from all services
make logs-api          # Stream logs from API only
make logs-db           # Stream logs from PostgreSQL only
make logs-webui        # Stream logs from WebUI only
make status            # Show status of all running services
```

### Database Management
```bash
make db-shell          # Connect to PostgreSQL interactive shell
make db-backup         # Create database backup
make db-restore        # Restore from latest backup
make db-reset          # Delete and reinitialize database (⚠️ WARNING)
```

### Health & Testing
```bash
make api-health        # Check API health status
make test              # Run multi-tenant tests
make test-isolation    # Run tenant isolation tests
```

### Cleanup & Maintenance
```bash
make clean             # Remove stopped containers and dangling images
make reset             # Full system reset (⚠️ WARNING: deletes all data)
make prune             # Prune unused Docker resources
```

## 🔧 Configuration

### Environment Variables

Edit `.env` file to configure:

```bash
# LLM Provider (OpenAI, Ollama, Azure, etc.)
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_BINDING_API_KEY=your_api_key_here

# Embedding Service
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_BINDING_HOST=http://localhost:11434

# Database Credentials
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=lightrag_secure_password

# Multi-Tenant Settings
DEFAULT_TENANT=default
DEFAULT_KB=default

# See env.template.example for all available options
```

## 🔐 Security & Multi-Tenant Isolation

### Isolation Guarantees

1. **Database-Level Filtering**: Every query includes `tenant_id` and `kb_id` constraints
2. **Composite Key Constraints**: Prevents accidental ID collisions between tenants
3. **No Application-Level Trust**: Storage layer enforces isolation even if app code has bugs
4. **Audit Trail**: All operations include tenant context for traceability

### Best Practices

✅ **DO:**
- Always pass tenant context to every operation
- Use support module helpers for queries
- Create composite indexes on (tenant_id, kb_id, ...)
- Validate tenant context early in request pipeline
- Log all tenant-related operations

❌ **DON'T:**
- Query without tenant filtering
- Hardcode tenant IDs in application code
- Assume application code enforces isolation
- Skip index creation after migration
- Mix tenants in a single transaction

## 📋 Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| **WebUI** | `http://localhost:3001` | Interactive frontend for document upload, KB visualization, queries |
| **API Server** | `http://localhost:8000` | RESTful API for programmatic access |
| **PostgreSQL** | `internal-only (container network)` | Database backend (not exposed to host by default) |
| **Redis** | `localhost:6379` | Cache backend (internal only) |
| **Health Check** | `http://localhost:8000/health` | API health status |

## 🧪 Testing Multi-Tenant Features

### Run All Multi-Tenant Tests
```bash
make test
```

### Run Specific Test Suites
```bash
# Test tenant isolation
make test-isolation

# Test PostgreSQL backend
pytest tests/test_multi_tenant_backends.py::TestPostgreSQLTenantSupport -v

# Test data integrity
pytest tests/test_multi_tenant_backends.py::TestDataIntegrity -v
```

### Manual Testing

1. **Create document for tenant "595ea68b-0f3a-4dbe-8a86-9276a1bbd10c"**:
```bash
curl -X POST http://localhost:8000/api/v1/insert \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Id: 595ea68b-0f3a-4dbe-8a86-9276a1bbd10c" \
  -H "X-KB-Id: kb-prod" \
  -d '{"document": "Sample document"}'
```

2. **Query as "595ea68b-0f3a-4dbe-8a86-9276a1bbd10c"**:
```bash
curl "http://localhost:8000/api/v1/query" \
  -H "X-Tenant-Id: 595ea68b-0f3a-4dbe-8a86-9276a1bbd10c" \
  -H "X-KB-Id: kb-prod" \
  -G --data-urlencode "param=test"
```

3. **Verify isolation** - query with different tenant:
```bash
curl "http://localhost:8000/api/v1/query" \
  -H "X-Tenant-Id: 44bf3e0d-d633-4dea-9b74-3e24140cd7e3" \
  -H "X-KB-Id: kb-main" \
  -G --data-urlencode "param=test"
# Should return different or empty results
```

## 📦 Docker Services

### PostgreSQL (pgvector/pgvector:pg15-latest)
- **Purpose**: Primary data storage with vector support
- **Volume**: `postgres_data` (persists database files)
- **Port**: 5432 (internal), configurable via `POSTGRES_PORT`
- **Health Check**: Every 10 seconds

### Redis (redis:7-alpine)
- **Purpose**: Caching, sessions, temporary data
- **Volume**: `redis_data` (persists snapshot)
- **Port**: 6379 (internal), configurable via `REDIS_PORT`
- **Health Check**: Every 10 seconds

### LightRAG API
- **Port**: 8621 (internal), 8000 (external/host)
- **Volume**: `./data/*` (documents, storage, tiktoken cache)
- **Dependencies**: PostgreSQL, Redis
- **Health Check**: Every 30 seconds
- **Resources**: Limited to 2 CPUs / 4GB RAM

### Web UI
- **Port**: 3000 (internal), 3001 (external/host)
- **Framework**: React + Vite
- **Dependencies**: LightRAG API
- **Health Check**: Every 30 seconds

## 🐛 Troubleshooting

### Services not starting?
```bash
# Check service status
make status

# View detailed logs
make logs

# Check specific service
make logs-api
```

### Database connection error?
```bash
# Verify database is ready
make api-health

# Check PostgreSQL directly
make db-shell

# Reinitialize database
make db-reset
```

### API responding slowly?
```bash
# Check resource usage
docker stats lightrag-api

# View API logs for errors
make logs-api

# Restart API service
docker compose -p lightrag-multitenant restart lightrag-api
```

### Data isolation issues?
```bash
# Check tenant context in logs
make logs | grep -i tenant

# Verify database schema
make db-shell
# \dt  (list tables)
# \di  (list indexes)
```

## 📂 Directory Structure

```
starter/
├── Makefile                    # Main command interface
├── docker-compose.yml          # Docker services definition
├── env.template.example        # Environment variables template
├── init-postgres.sql          # PostgreSQL initialization (optional)
├── README.md                  # This file
├── data/
│   ├── inputs/               # Document input directory
│   ├── rag_storage/          # LightRAG storage
│   └── tiktoken/             # Tiktoken cache
└── backups/                   # Database backups (created by make db-backup)
```

## 🔄 Data Migration

### Backup Database
```bash
make db-backup
# Backs up to: ./backups/lightrag_backup_YYYYMMDD_HHMMSS.sql
```

### Restore Database
```bash
make db-restore
# Restores from latest backup in ./backups/
```

### Export Data for Another Tenant
```bash
# Export
make db-shell
\COPY (SELECT * FROM documents WHERE tenant_id='acme-corp') TO 'acme-corp-export.csv' CSV HEADER;
\q

# Import
make db-shell
\COPY documents FROM 'acme-corp-export.csv' CSV HEADER;
\q
```

## 🚀 Production Deployment

For production deployments:

1. **Use strong passwords**: Update `POSTGRES_PASSWORD` and `REDIS_PASSWORD`
2. **Enable SSL**: Uncomment SSL configuration in `.env`
3. **Use external LLM provider**: Configure production API keys
4. **Set up monitoring**: Monitor logs and health endpoints
5. **Regular backups**: Schedule `make db-backup` via cron
6. **Resource limits**: Adjust resource limits in docker-compose.yml
7. **Network isolation**: Use only internal networks, expose via proxy

## 📝 API Usage Examples

### Using Multi-Tenant Context

```python
import requests

BASE_URL = "http://localhost:8000"

# Headers with tenant context
headers = {
    "X-Tenant-Id": "acme-corp",
    "X-KB-Id": "kb-prod",
    "Content-Type": "application/json"
}

# Insert document
response = requests.post(
    f"{BASE_URL}/api/v1/insert",
    headers=headers,
    json={"document": "Company policy document"}
)

# Query with tenant isolation
response = requests.get(
    f"{BASE_URL}/api/v1/query",
    headers=headers,
    params={"param": "policy"},
    params={"top_k": 5}
)

# Results are automatically isolated to acme-corp/kb-prod
print(response.json())
```

### Python SDK Example

```python
from lightrag import LightRAG

# Initialize with tenant context
rag = LightRAG(
    tenant_id="acme-corp",
    kb_id="kb-prod",
    storage_type="PostgreSQL",
    llm_model_name="gpt-4o",
    embedding_model_name="bge-m3:latest"
)

# Insert document (automatically scoped to tenant/kb)
rag.insert("Company documentation", source="internal")

# Query (automatically scoped to tenant/kb)
results = rag.query("What is the company policy?")
print(results)
```

## 📚 Documentation References

- **Multi-Tenant Architecture**: See `docs/0001-multi-tenant-architecture.md`
- **LightRAG Documentation**: https://github.com/HKUDS/LightRAG
- **PostgreSQL Vector Extension**: https://github.com/pgvector/pgvector
- **Docker Compose Documentation**: https://docs.docker.com/compose/

## 🆘 Support & Issues

### Common Issues

**Q: Port already in use?**
```bash
# Change port in .env
WEBUI_PORT=3001
API_PORT=9622
POSTGRES_PORT=5433
```

**Q: Out of memory?**
```bash
# Reduce resource limits in docker-compose.yml or adjust system resources
```

**Q: API not responding?**
```bash
# Check if services are running
make ps

# View logs
make logs

# Restart services
make down && make up
```

**Q: Database errors?**
```bash
# Connect to database shell
make db-shell

# Check table structure
\d documents

# Check indexes
\di
```

## 📄 License

LightRAG is licensed under MIT License. See LICENSE file for details.

## 🙋 Contributing

Contributions are welcome! Please refer to the main LightRAG repository for contribution guidelines.

---

**Last Updated**: November 20, 2025
**Status**: Production Ready
**Version**: 1.0

For more information about multi-tenant features, see the architecture documentation in `docs/0001-multi-tenant-architecture.md`
