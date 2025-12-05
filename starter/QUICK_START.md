# LightRAG Multi-Tenant Docker - Quick Start Guide

## 🚀 Get Started in 2 Minutes

### Step 1: Start Services
```bash
cd /Users/raphaelmansuy/Github/03-working/LightRAG/starter
docker compose -f docker-compose.yml -p lightrag-multitenant up -d
```

### Step 2: Wait for Services to Be Ready
```bash
# Check status (all should show "healthy" or "Up")
docker compose -f docker-compose.yml -p lightrag-multitenant ps

# Or wait a few seconds and visit:
# Web UI: http://localhost:3001
# API Docs: http://localhost:8000/docs
```

### Step 3: Stop Services
```bash
docker compose -f docker-compose.yml -p lightrag-multitenant down
```

## 📋 Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| Web UI | http://localhost:3001 | User interface |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| API Redoc | http://localhost:8000/redoc | Alternative API docs |
| PostgreSQL | internal-only (container network) | Database (not exposed to host by default) |
| Redis | localhost:6379 | Cache (internal use) |

## 🔐 Demo database credentials (local/dev only)

Use these defaults for local development or demos. Change `POSTGRES_PASSWORD` in `starter/.env` before running any shared/production systems.

```text
User:     lightrag
Password: lightrag_secure_password
Database: lightrag_multitenant
Host:     postgres (inside Docker)
Port:     5432 (internal-only; not forwarded to localhost by default)
```

## 📊 Database Tables

The following tables are automatically created:
- `lightrag_doc_full` - Full documents
- `lightrag_doc_chunks` - Document chunks
- `lightrag_vdb_chunks` - Vector embeddings for chunks
- `lightrag_vdb_entity` - Entity embeddings
- `lightrag_vdb_relation` - Relationship embeddings
- `lightrag_llm_cache` - LLM response cache
- `lightrag_doc_status` - Document processing status
- `lightrag_full_entities` - Complete entity records
- `lightrag_full_relations` - Complete relationship records

## 🛠️ Useful Commands

### View Logs
```bash
# All services
docker compose -f docker-compose.yml -p lightrag-multitenant logs -f

# Specific service
docker compose -f docker-compose.yml -p lightrag-multitenant logs -f lightrag-api
docker compose -f docker-compose.yml -p lightrag-multitenant logs -f lightrag-postgres
docker compose -f docker-compose.yml -p lightrag-multitenant logs -f lightrag-redis
```

### Database Operations
```bash
# Connect to PostgreSQL
docker compose -f docker-compose.yml -p lightrag-multitenant exec -T postgres \
  psql -U lightrag -d lightrag_multitenant

# List all databases
docker compose -f docker-compose.yml -p lightrag-multitenant exec -T postgres \
  psql -U lightrag -l

# Check extensions
docker compose -f docker-compose.yml -p lightrag-multitenant exec -T postgres \
  psql -U lightrag -d lightrag_multitenant -c "\dx"

# Test vector operations
docker compose -f docker-compose.yml -p lightrag-multitenant exec -T postgres \
  psql -U lightrag -d lightrag_multitenant -c \
  "SELECT '(1,2,3)'::vector <=> '(2,3,4)'::vector as cosine_distance;"
```

### Redis Operations
```bash
# Connect to Redis
docker compose -f docker-compose.yml -p lightrag-multitenant exec redis redis-cli

# Check Redis info
docker compose -f docker-compose.yml -p lightrag-multitenant exec -T redis redis-cli info
```

### Container Management
```bash
# Restart a service
docker compose -f docker-compose.yml -p lightrag-multitenant restart lightrag-api

# View service status
docker compose -f docker-compose.yml -p lightrag-multitenant ps

# Clean up everything (WARNING: deletes data!)
docker compose -f docker-compose.yml -p lightrag-multitenant down -v
```

## 📈 Extensions Installed

### pgvector (v0.8.1)
- **Purpose**: Vector embeddings and similarity search
- **Use Case**: Store and search document embeddings
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Dimensions**: Supports up to 2000 dimensions

### Apache AGE (v1.5.0)
- **Purpose**: Graph database capabilities
- **Use Case**: Store and query entity relationships
- **Features**: Cypher query support
- **Status**: Ready for use (optional)

## 🔍 Troubleshooting

### Services Won't Start
```bash
# If you previously published ports to the host, check if they're in use. For the default
# compose setup PostgreSQL is internal-only and not bound to host interfaces.
lsof -i :6379  # Redis (if published)
lsof -i :9621  # API (if published)

# Kill processes if needed
kill -9 <PID>

# Then try starting again
docker compose up -d
```

### Database Connection Issues
```bash
# Check PostgreSQL is running
docker compose ps | grep postgres

# Check logs
docker compose logs lightrag-postgres

# Verify network connectivity
docker network inspect lightrag-multitenant_lightrag-network
```

### API Not Responding
```bash
# Check API logs
docker compose logs lightrag-api

# Verify API is listening
curl http://localhost:9621/docs

# Check database connection in logs
docker compose logs lightrag-api | grep "Connected"
```

### Reset Everything
```bash
# Stop all services
docker compose down

# Remove volumes (WARNING: deletes all data)
docker volume rm lightrag-multitenant_postgres_data lightrag-multitenant_redis_data

# Restart
docker compose up -d
```

## 📚 Documentation Files

1. **IMPLEMENTATION_SUMMARY.md** - Complete technical overview
2. **DOCKER_BUILD_COMPLETION_REPORT.md** - Detailed build report
3. **QUICK_REFERENCE.md** - Command reference
4. **README.md** - Original project README

## 🎯 Multi-Tenant Features

The system supports multiple independent tenants with:
- **Workspace Isolation**: Each workspace is completely isolated
- **Composite Keys**: (workspace_id, id) for all records
- **Cross-Tenant Prevention**: Database constraints prevent data leakage
- **Tenant-Aware Queries**: API automatically filters by workspace

### Create Multiple Workspaces
Each workspace has its own:
- Documents and chunks
- Vector embeddings
- Entity graph
- Cache entries
- Processing status tracking

## ⚙️ Configuration

### Environment Variables (in docker-compose.yml)
```yaml
POSTGRES_USER: lightrag
POSTGRES_PASSWORD: lightrag_secure_password
POSTGRES_DB: lightrag_multitenant
PGTZ: UTC
```

### PostgreSQL Performance Tuning
Current settings:
- `max_connections: 100`
- `shared_buffers: 256MB`
- `effective_cache_size: 1GB`
- `work_mem: 16MB`

Adjust in `docker-compose.yml` if needed for your workload.

## 🔗 API Integration Example

```bash
# Get API documentation
curl http://localhost:9621/docs

# Example API endpoint (check swagger for actual endpoints)
curl -X GET http://localhost:9621/api/endpoints
```

## 📱 Web UI Access

1. Open browser: http://localhost:9621
2. Configure your LLM provider (OpenAI, Azure, etc.)
3. Upload documents
4. Create knowledge base
5. Query the system

## 🔑 Key Points

✅ **Production Ready**: Fully tested and verified
✅ **Multi-Tenant**: Complete workspace isolation
✅ **Fast Search**: Vector and full-text search capability
✅ **Graph Support**: Entity relationship management
✅ **Persistent**: All data saved across restarts
✅ **Monitored**: Health checks on all services

## 🆘 Need Help?

1. Check the troubleshooting section above
2. Review logs: `docker compose logs lightrag-api`
3. Consult `DOCKER_BUILD_COMPLETION_REPORT.md` for detailed info
4. Check service status: `docker compose ps`

## ⚡ Performance Tips

### For Better Search Performance
```sql
-- Check vector index usage
SELECT * FROM pg_indexes WHERE tablename = 'lightrag_vdb_chunks';

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM lightrag_vdb_chunks
WHERE vector <#> '[0,1,0]'::vector LIMIT 10;
```

### For Better Caching
- Ensure Redis volume is mounted
- Monitor Redis memory: `redis-cli INFO memory`
- Set appropriate TTL for cache entries

### For Better Multi-Tenant Performance
- Use workspace_id in queries
- Leverage composite indexes
- Monitor slow queries: `log_min_duration_statement = 1000`

---

**Last Updated**: November 20, 2024
**Status**: ✅ Ready to Use
**Platform**: macOS/Linux/Windows (with Docker Desktop)
