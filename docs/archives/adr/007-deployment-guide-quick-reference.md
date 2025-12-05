# ADR 007: Deployment Guide and Quick Reference

## Status: Proposed

## Summary of Multi-Tenant Architecture

### Core Components

| Component | Purpose | Responsibility |
|-----------|---------|-----------------|
| **Tenant** | Top-level isolation boundary | Grouping of knowledge bases |
| **Knowledge Base** | Domain-specific RAG system | Contains documents, entities, relationships |
| **TenantContext** | Request-scoped isolation | Passed through entire call stack |
| **RAGManager** | Instance caching | Creates/caches LightRAG per tenant/KB |
| **Storage Layer Filters** | Defense in depth | All queries scoped to tenant/KB |

### Key Design Decisions

```
┌──────────────────────────────────────┐
│   Composite Isolation Strategy       │
├──────────────────────────────────────┤
│ Tenant ID (UUID)                     │
│ └─ Knowledge Base ID (UUID)          │
│    └─ Composite Key: t:k:entity_id   │
│       └─ Storage filters all queries  │
└──────────────────────────────────────┘
```

### Files Modified/Created

**New Files (11 total)**:
1. `lightrag/models/tenant.py` - Tenant/KB models
2. `lightrag/services/tenant_service.py` - Tenant management
3. `lightrag/tenant_rag_manager.py` - Instance caching
4. `lightrag/api/dependencies.py` - DI for tenant context
5. `lightrag/api/models/requests.py` - API request models
6. `lightrag/api/routers/tenant_routes.py` - Tenant endpoints
7. `tests/test_tenant_isolation.py` - Unit tests
8. `tests/test_api_tenant_routes.py` - Integration tests
9. `scripts/migrate_workspace_to_tenant.py` - Migration script
10. `lightrag/kg/migrations/001_add_tenant_schema.sql` - DB schema
11. `lightrag/kg/migrations/mongo_001_add_tenant_collections.py` - MongoDB schema

**Modified Files (7 total)**:
1. `lightrag/base.py` - Add tenant/kb to StorageNameSpace
2. `lightrag/lightrag.py` - Add tenant context to query/insert
3. `lightrag/kg/postgres_impl.py` - Add tenant filtering to all queries
4. `lightrag/kg/json_kv_impl.py` - Add tenant/kb directories
5. `lightrag/api/lightrag_server.py` - Register new routes
6. `lightrag/api/auth.py` - Tenant-aware JWT validation
7. `lightrag/api/config.py` - Add tenant configuration

## Quick Start for Developers

### 1. Setting Up Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL for tenant metadata
docker run -d --name lightrag-postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15

# Run migrations
psql postgresql://postgres:password@localhost:5432/postgres < \
  lightrag/kg/migrations/001_add_tenant_schema.sql

# Set environment variables
export LIGHTRAG_KV_STORAGE=PGKVStorage
export TENANT_DB_HOST=localhost
export TENANT_DB_USER=postgres
export TENANT_DB_PASSWORD=password
```

### 2. Testing Locally

```bash
# Run unit tests
pytest tests/test_tenant_isolation.py -v

# Run integration tests
pytest tests/test_api_tenant_routes.py -v

# Run with coverage
pytest --cov=lightrag tests/ --cov-report=html

# Test tenant isolation (should fail if not working)
pytest tests/test_tenant_isolation.py::TestTenantIsolation::test_cross_tenant_data_isolation -v
```

### 3. Manual Testing via cURL

```bash
# 1. Create tenant (admin)
ADMIN_TOKEN="eyJhbGc..."  # From auth system
curl -X POST http://localhost:9621/api/v1/tenants \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_name": "Test Tenant"}'

# Response:
# {
#   "status": "success",
#   "data": {
#     "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
#     "tenant_name": "Test Tenant",
#     "is_active": true,
#     "created_at": "2025-11-20T10:00:00Z"
#   }
# }

TENANT_ID="550e8400-e29b-41d4-a716-446655440000"

# 2. Create knowledge base
curl -X POST http://localhost:9621/api/v1/tenants/$TENANT_ID/knowledge-bases \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"kb_name": "Test KB"}'

KB_ID="660e8400-e29b-41d4-a716-446655440000"

# 3. Create API key for tenant
curl -X POST http://localhost:9621/api/v1/tenants/$TENANT_ID/api-keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "key_name": "test-key",
    "knowledge_base_ids": ["'$KB_ID'"],
    "permissions": ["query:run", "document:read"]
  }'

# Response includes: {"key": "sk-..."}
API_KEY="sk-..."

# 4. Add document with API key
curl -X POST http://localhost:9621/api/v1/tenants/$TENANT_ID/knowledge-bases/$KB_ID/documents/add \
  -H "X-API-Key: $API_KEY" \
  -F "file=@test_document.pdf"

# 5. Query knowledge base
curl -X POST http://localhost:9621/api/v1/tenants/$TENANT_ID/knowledge-bases/$KB_ID/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "mode": "mix",
    "top_k": 10
  }'

# 6. Verify cross-tenant isolation (should fail)
TENANT_B_ID="770e8400-e29b-41d4-a716-446655440001"
curl -X GET http://localhost:9621/api/v1/tenants/$TENANT_B_ID \
  -H "X-API-Key: $API_KEY"

# Response: 403 Forbidden (API key only for Tenant A)
```

## Backward Compatibility

### Migrating from Workspace to Tenant

```bash
# 1. Backup existing data
cp -r ./rag_storage ./rag_storage.backup

# 2. Run migration script
python scripts/migrate_workspace_to_tenant.py \
  --working-dir ./rag_storage

# 3. Verify migration
python -c "
from lightrag.services.tenant_service import TenantService
import asyncio

async def verify():
    service = TenantService(...)
    tenants = await service.list_all_tenants()
    for t in tenants:
        print(f'Tenant: {t.tenant_id} ({t.tenant_name})')
        kbs = await service.list_knowledge_bases(t.tenant_id)
        for kb in kbs:
            print(f'  KB: {kb.kb_id} ({kb.kb_name})')

asyncio.run(verify())
"

# 4. Test that old workspace still accessible via tenant
# Legacy workspace 'myworkspace' becomes tenant 'myworkspace'
```

## Configuration Examples

### Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: lightrag
      POSTGRES_PASSWORD: secret
    ports:
      - "5432:5432"
    volumes:
      - ./lightrag/kg/migrations/001_add_tenant_schema.sql:/docker-entrypoint-initdb.d/01_schema.sql

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  lightrag:
    build: .
    environment:
      # Tenant Configuration
      TENANT_ENABLED: "true"
      MAX_CACHED_INSTANCES: "100"

      # Storage Configuration
      LIGHTRAG_KV_STORAGE: "PGKVStorage"
      LIGHTRAG_VECTOR_STORAGE: "PGVectorStorage"
      LIGHTRAG_GRAPH_STORAGE: "PGGraphStorage"

      # Database
      PG_HOST: "postgres"
      PG_DATABASE: "lightrag"
      PG_USER: "postgres"
      PG_PASSWORD: "secret"

      # LLM Configuration
      LLM_BINDING: "openai"
      LLM_MODEL: "gpt-4o-mini"
      LLM_BINDING_API_KEY: "${OPENAI_API_KEY}"

      # Embedding Configuration
      EMBEDDING_BINDING: "openai"
      EMBEDDING_MODEL: "text-embedding-3-small"
      EMBEDDING_DIM: "1536"

      # Authentication
      JWT_ALGORITHM: "HS256"
      TOKEN_SECRET: "your-secret-key-change-in-production"
      TOKEN_EXPIRE_HOURS: "24"

      # API
      CORS_ORIGINS: "*"
      LOG_LEVEL: "INFO"

    ports:
      - "9621:9621"

    depends_on:
      - postgres
      - redis

    volumes:
      - ./rag_storage:/app/rag_storage
```

### Environment Variables

```bash
# Tenant Manager
TENANT_ENABLED=true
MAX_CACHED_INSTANCES=100
TENANT_CONFIG_SYNC_INTERVAL=300

# Database
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage

# PostgreSQL Connection
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=lightrag
PG_USER=postgres
PG_PASSWORD=secret

# Authentication
JWT_ALGORITHM=HS256
TOKEN_SECRET=your-secret-key
TOKEN_EXPIRE_HOURS=24
GUEST_TOKEN_EXPIRE_HOURS=1

# LLM Configuration
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_API_KEY=${OPENAI_API_KEY}
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small

# Quotas
MAX_DOCUMENTS=10000
MAX_STORAGE_GB=100
MAX_KB_PER_TENANT=50

# Rate Limiting
RATE_LIMIT_QUERIES_PER_MINUTE=100
RATE_LIMIT_DOCUMENTS_PER_HOUR=50
RATE_LIMIT_API_CALLS_PER_MONTH=100000

# Monitoring
LOG_LEVEL=INFO
ENABLE_AUDIT_LOGGING=true
AUDIT_LOG_RETENTION_DAYS=90
```

## Monitoring and Observability

### Metrics to Track

```python
# Key metrics for multi-tenant system

METRICS = {
    "tenant_management": {
        "active_tenants": "Gauge",
        "total_kbs": "Gauge",
        "tenant_creation_time": "Histogram",
    },
    "isolation": {
        "cross_tenant_access_attempts": "Counter",  # Should be 0
        "cross_kb_access_attempts": "Counter",      # Should be 0
        "isolation_violations": "Counter",           # Should be 0
    },
    "performance": {
        "query_latency_per_tenant": "Histogram",
        "document_processing_time": "Histogram",
        "rag_instance_cache_hits": "Counter",
        "rag_instance_cache_misses": "Counter",
    },
    "security": {
        "failed_auth_attempts": "Counter",
        "permission_denials": "Counter",
        "api_key_usage": "Counter (per key)",
    },
    "quotas": {
        "storage_used_per_tenant": "Gauge",
        "documents_per_tenant": "Gauge",
        "api_calls_per_tenant": "Counter",
    }
}
```

### Example Prometheus Queries

```promql
# Average query latency per tenant
histogram_quantile(0.95, query_latency_per_tenant) by (tenant_id)

# Cache hit rate
rag_instance_cache_hits / (rag_instance_cache_hits + rag_instance_cache_misses)

# Failed auth attempts
rate(failed_auth_attempts[5m])

# Cross-tenant access attempts (should be 0)
cross_tenant_access_attempts
```

### Logging

```python
# Structured logging for debugging

import structlog

logger = structlog.get_logger()

# Example log entry
logger.info(
    "query_executed",
    user_id="user-123",
    tenant_id="acme",
    kb_id="docs",
    query="What is...",
    mode="mix",
    latency_ms=145,
    result_count=5,
    request_id="req-abc-123"
)
```

## Rollout Strategy

### Phase 1: Soft Launch (Week 1)
```
- Deploy with TENANT_ENABLED=false (features off)
- Run in parallel with existing system
- Test against staging data
- Monitor for issues: 0 expected
```

### Phase 2: Closed Beta (Week 2)
```
- TENANT_ENABLED=true for 10% of traffic
- Small set of trusted customers
- Monitor metrics closely
- Rollback plan ready
```

### Phase 3: Gradual Rollout (Week 3)
```
- 25% → 50% → 100%
- Staggered by time of day
- Monitor isolation violations (should be 0)
- Customer education happening
```

### Phase 4: Full Production (Week 4)
```
- 100% of traffic on multi-tenant system
- Legacy workspace mode deprecated (6-month timeline)
- Full monitoring and alerting active
- Support team trained
```

## Troubleshooting Guide

### Issue: Cross-Tenant Data Visible

```
Symptom: User can see Tenant B data while using Tenant A credentials
Solution:
1. Check TokenPayload.tenant_id == request.path.tenant_id
2. Check storage filters include WHERE tenant_id = ? AND kb_id = ?
3. Review TenantContext creation in get_tenant_context()
4. Check RAGManager.get_rag_instance() is called with correct IDs
```

### Issue: Slow Queries

```
Symptom: Queries taking >1 second
Solution:
1. Check indexes on (tenant_id, kb_id) columns
2. Verify RAG instance cache is working (check metrics)
3. Check if instance is being recompiled every request
4. Profile with: SELECT * FROM documents WHERE tenant_id=? AND kb_id=?
```

### Issue: High Memory Usage

```
Symptom: Memory growing over time
Solution:
1. Check MAX_CACHED_INSTANCES setting (default 100)
2. Monitor rag_instance_cache_size metric
3. Verify finalize_storages() called on eviction
4. Check for memory leaks in embedding cache
```

## Support and Resources

### Documentation
- Architecture Overview: `adr/001-multi-tenant-architecture-overview.md`
- Implementation Guide: `adr/002-implementation-strategy.md`
- Data Models: `adr/003-data-models-and-storage.md`
- API Design: `adr/004-api-design.md`
- Security: `adr/005-security-analysis.md`
- Diagrams & Alternatives: `adr/006-architecture-diagrams-alternatives.md`

### Code Examples
- See `examples/multi_tenant_demo.py` for complete usage example
- See `tests/test_api_tenant_routes.py` for API testing examples
- See `scripts/migrate_workspace_to_tenant.py` for migration examples

### Getting Help
- GitHub Issues: [LightRAG/issues](https://github.com/HKUDS/LightRAG/issues)
- Discussions: [LightRAG/discussions](https://github.com/HKUDS/LightRAG/discussions)
- Discord: [LightRAG Community](https://discord.gg/yF2MmDJyGJ)

## Success Criteria

Multi-tenant implementation is successful when:

✓ **Functional Requirements Met**
- [ ] All API endpoints working with tenant/KB routing
- [ ] Data isolation verified (cross-tenant access prevents)
- [ ] RBAC enforcement working correctly
- [ ] Audit logging capturing all operations
- [ ] Migration from workspace to tenant successful

✓ **Performance Targets Met**
- [ ] Query latency < 200ms p99 (including tenant filtering)
- [ ] Storage overhead < 3%
- [ ] Instance cache hit rate > 90%
- [ ] API response time < 150ms average

✓ **Security Requirements Met**
- [ ] Zero cross-tenant data access
- [ ] JWT token validation in all requests
- [ ] Permission checking on every operation
- [ ] Rate limiting preventing abuse
- [ ] Audit logs tamper-proof and retained

✓ **Operational Readiness**
- [ ] Monitoring/alerting configured
- [ ] Runbooks for common issues
- [ ] Disaster recovery plan tested
- [ ] Support team trained
- [ ] Documentation complete

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Deployment Timeline**: 4 weeks
**Success Criteria**: All items checked off
**Status**: Ready for Implementation
