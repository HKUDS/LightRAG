# ADR 001: Multi-Tenant, Multi-Knowledge-Base Architecture for LightRAG

## Status: Proposed

## Context

### Current State
LightRAG is a retrieval-augmented generation system that currently operates as a single-instance system with basic workspace-level data isolation. The existing architecture uses:

- **Workspace concept**: Directory-based or database-field-based isolation for file/database storage
- **Single LightRAG instance**: One RAG system per server process, configured at startup
- **Basic authentication**: JWT tokens and API key support without tenant/knowledge-base awareness
- **Shared configuration**: All data uses the same LLM, embedding, and storage configurations

### Limitations of Current Architecture
1. **No true multi-tenancy**: Cannot serve multiple independent tenants securely
2. **No knowledge base isolation**: All data belongs to a single knowledge base
3. **Shared compute resources**: LLM and embedding calls are shared across all workspaces
4. **Static configuration**: All tenants must use the same models and settings
5. **Cross-tenant data leak risk**: Workspace isolation is not cryptographically enforced
6. **No resource quotas**: No limits on storage, compute, or API usage per tenant
7. **Authentication limitations**: JWT tokens don't support fine-grained access control

### Existing Code Evidence
- **Workspace in base.py**: `StorageNameSpace` class (line 176) includes `workspace` field for basic isolation
- **Namespace concept**: `NameSpace` class in `namespace.py` defines storage categories but no tenant/KB concept
- **Storage implementations**: Each storage type (PostgreSQL, JSON, Neo4j) implements workspace filtering:
  - `PostgreSQLDB` constructor accepts workspace parameter (line 56 in postgres_impl.py)
  - `JsonKVStorage` creates workspace directories (line 30-39 in json_kv_impl.py)
- **API configuration**: `lightrag_server.py` accepts `--workspace` flag but no tenant/KB parameters
- **Authentication**: `auth.py` provides JWT tokens with roles but no tenant/KB scoping

### Business Requirements
Organizations deploying LightRAG need to:
1. Serve multiple independent customers (tenants) from a single instance
2. Support multiple knowledge bases per tenant for different use cases
3. Enforce complete data isolation between tenants
4. Manage per-tenant resource quotas and billing
5. Support per-tenant configuration (models, parameters, API keys)
6. Provide audit trails and access logs per tenant

## Decision

### High-Level Architecture
Implement a **multi-tenant, multi-knowledge-base (MT-MKB)** architecture that:

1. **Adds tenant abstraction layer** above the current workspace concept
2. **Introduces knowledge base concept** as a first-class entity
3. **Implements tenant-aware routing** at the API level
4. **Enforces data isolation** through composite keys and access control
5. **Supports per-tenant/KB configuration** for models and parameters
6. **Adds role-based access control (RBAC)** for fine-grained permissions

### Core Design Principles
1. **Backward Compatibility**: Existing single-workspace setups continue to work
2. **Layered Isolation**: Tenant > Knowledge Base > Document > Chunk/Entity
3. **Zero Trust**: All data access requires explicit tenant/KB context
4. **Default Deny**: Cross-tenant access is explicitly blocked unless authorized
5. **Audit Trail**: All operations logged with tenant/KB context
6. **Resource Aware**: Quotas and limits per tenant/KB

### Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (Single Instance)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  API Router      │  │ Auth/Middleware  │  │  Request Handler │
│  │  Layer           │  │ (Tenant Extract) │  │  Layer           │
│  └──────┬───────────┘  └──────┬───────────┘  └──────┬───────────┘
│         │                      │                      │
│  ┌──────▼──────────────────────▼──────────────────────▼──────┐
│  │        Tenant Context (TenantID + KnowledgeBaseID)       │
│  │        Injected via Dependency Injection / Middleware    │
│  └──────┬─────────────────────────────────────────────────────┘
│         │
│  ┌──────▼──────────────────────────────────────────────────────┐
│  │         Tenant-Aware LightRAG Instance Manager             │
│  │         (Caches instances per tenant)                      │
│  └──────┬─────────────────────────────────────────────────────┘
│         │
│  ┌──────▼──────────────────────────────────────────────────────┐
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐        │
│  │  │  Tenant 1   │  │  Tenant 2   │  │  Tenant N    │        │
│  │  │  KB1, KB2   │  │  KB1, KB3   │  │  KB1, ...    │        │
│  │  └─────────────┘  └─────────────┘  └──────────────┘        │
│  │                                                             │
│  │  Multiple LightRAG Instances (per tenant or cached)        │
│  └──────┬──────────────────────────────────────────────────────┘
│         │
│  ┌──────▼──────────────────────────────────────────────────────┐
│  │         Storage Access Layer with Tenant Filtering         │
│  │         (Adds tenant/KB filters to all queries)            │
│  └──────┬─────────────────────────────────────────────────────┘
│         │
│  ┌──────▼──────────────────────────────────────────────────────┐
│  │                                                              │
│  │  ┌────────────────┐  ┌────────────┐  ┌────────────────┐   │
│  │  │  PostgreSQL    │  │  Neo4j     │  │  Redis/Milvus │   │
│  │  │  (Shared DB)   │  │  (Shared)  │  │  (Shared)      │   │
│  │  └────────────────┘  └────────────┘  └────────────────┘   │
│  │                                                              │
│  │  All queries filtered by tenant/KB at storage layer        │
│  └────────────────────────────────────────────────────────────┘
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Tenant Model
- **TenantID**: Unique identifier (UUID or slug)
- **TenantName**: Human-readable name
- **Configuration**: Per-tenant LLM, embedding, and rerank model configs
- **ResourceQuotas**: Storage, API calls, concurrent requests limits
- **CreatedAt/UpdatedAt**: Audit timestamps

#### 2. Knowledge Base Model
- **KnowledgeBaseID**: Unique within tenant
- **TenantID**: Parent tenant reference
- **KBName**: Display name
- **Description**: Purpose and content overview
- **Configuration**: Per-KB indexing and query parameters
- **Status**: Active/Archived
- **Metadata**: Custom fields for tenant-specific data

#### 3. Storage Isolation Strategy
All storage operations will include tenant/KB filters:
- **Document storage**: `workspace = f"{tenant_id}_{kb_id}"`
- **Vector storage**: Add `tenant_id` and `kb_id` metadata fields
- **Graph storage**: Store tenant/KB info as node/edge attributes
- **KV storage**: Prefix keys with `tenant_id:kb_id:entity_id`

#### 4. API Routing
```
POST   /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/add
GET    /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/documents/{doc_id}
POST   /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/query
GET    /api/v1/tenants/{tenant_id}/knowledge-bases/{kb_id}/graph
```

#### 5. Authentication & Authorization
```python
# JWT Token Payload
{
    "sub": "user_id",                    # User identifier
    "tenant_id": "tenant_uuid",          # Assigned tenant
    "knowledge_base_ids": ["kb1", "kb2"], # Accessible KBs
    "role": "admin|editor|viewer",       # Role within tenant
    "exp": 1234567890,                   # Expiration
    "permissions": {
        "create_kb": true,
        "delete_documents": true,
        "run_queries": true
    }
}
```

#### 6. Dependency Injection for Tenant Context
```python
# FastAPI dependency to extract and validate tenant context
async def get_tenant_context(
    tenant_id: str,
    kb_id: str,
    token: str = Depends(get_auth_token)
) -> TenantContext:
    # Verify user can access this tenant/KB
    # Return validated context object
    pass
```

## Consequences

### Positive
1. **True Multi-Tenancy**: Complete data isolation between tenants
2. **Scalability**: Support hundreds of tenants in single instance
3. **Cost Efficiency**: Shared infrastructure reduces per-tenant costs
4. **Flexibility**: Per-tenant model and parameter configuration
5. **Security**: Fine-grained access control and audit trails
6. **Resource Management**: Per-tenant quotas prevent resource abuse
7. **Operational Simplicity**: Single instance to manage

### Negative/Tradeoffs
1. **Increased Complexity**: More code, more testing required (~2-3x development effort)
2. **Performance Overhead**: Tenant/KB filtering on every query (~5-10% latency impact)
3. **Storage Overhead**: Tenant/KB metadata increases storage footprint (~2-3%)
4. **Operational Complexity**: More configuration options, training needed
5. **Breaking Changes**: API endpoints change, requires migration scripts
6. **Backward Compatibility**: Existing workspaces need migration strategy

### Security Considerations
1. **Data Isolation**: Tenant-aware queries prevent cross-tenant leaks
2. **Authentication**: JWT tokens must include tenant scope
3. **Authorization**: RBAC prevents unauthorized access to KBs
4. **Audit Trail**: All operations logged for compliance
5. **Key Management**: Per-tenant API keys need separate management
6. **Potential Vulnerabilities**:
   - Parameter injection in tenant/KB IDs (mitigate: strict validation)
   - JWT token hijacking (mitigate: short expiry, rate limiting)
   - Side-channel attacks via timing (mitigate: constant-time comparisons)
   - Resource exhaustion (mitigate: quotas and rate limiting)

### Performance Impact
- **Query Latency**: +5-10% from additional filtering
- **Storage Size**: +2-3% for tenant/KB metadata
- **Memory Usage**: +20-30% from maintaining multiple LightRAG instances
- **CPU Usage**: +10-15% from authentication/authorization checks

### Migration Path for Existing Deployments
1. **Phase 1**: Deploy with backward compatibility (single tenant = existing workspace)
2. **Phase 2**: Provide migration script to convert workspaces to tenants
3. **Phase 3**: Support hybrid mode (legacy workspaces + new tenants)
4. **Phase 4**: Deprecate workspace mode in favor of tenant mode

## Implementation Plan (Summary)

See `002-implementation-strategy.md` for detailed step-by-step implementation guide.

### High-Level Phases
1. **Phase 1 (2-3 weeks)**: Core infrastructure
   - Database schema changes
   - Tenant/KB models
   - Storage access layer updates

2. **Phase 2 (2-3 weeks)**: API layer
   - Tenant-aware routing
   - Request/response models
   - Authentication/authorization

3. **Phase 3 (1-2 weeks)**: LightRAG integration
   - Instance manager
   - Per-tenant configurations
   - Query execution

4. **Phase 4 (1 week)**: Testing & deployment
   - Unit/integration tests
   - Migration scripts
   - Documentation

## Alternatives Considered

### 1. Separate Database Per Tenant
- **Approach**: Each tenant gets its own database/storage instance
- **Rejected because**:
  - Massive operational overhead (n×database connections, backups, upgrades)
  - Expensive (n×database licensing)
  - Complex to manage tenants across instances
  - Makes sharing resources impossible

### 2. Dedicated Server Instance Per Tenant
- **Approach**: Each tenant runs their own LightRAG instance
- **Rejected because**:
  - Massive resource waste (minimum resources per instance)
  - Very expensive at scale (n×server costs)
  - Difficult to manage and monitor
  - Cannot share LLM/embedding infrastructure

### 3. Simple Workspace Extension
- **Approach**: Just rename "workspace" to "tenant"
- **Rejected because**:
  - No knowledge base concept (multiple KB per tenant fails)
  - Cannot enforce cross-tenant access prevention
  - No RBAC or fine-grained permissions
  - Cannot manage per-tenant configuration
  - No resource quotas

### 4. Sharding by Tenant Hash
- **Approach**: Hash tenant ID to determine shard, send queries to correct shard
- **Rejected because**:
  - Breaks operational simplicity (multiple instances to manage)
  - Rebalancing is complex when adding/removing tenants
  - Doesn't reduce resource overhead

## Evidence/References

### Code References
- **Storage base class**: `lightrag/base.py:176-185` (StorageNameSpace)
- **Namespace constants**: `lightrag/namespace.py` (NameSpace class)
- **Workspace implementation**: `lightrag/kg/json_kv_impl.py:28-39` (JsonKVStorage)
- **PostgreSQL workspace support**: `lightrag/kg/postgres_impl.py:44-59`
- **API server architecture**: `lightrag/api/lightrag_server.py:1-300`
- **Authentication**: `lightrag/api/auth.py` (JWT token management)
- **Config**: `lightrag/api/config.py:200-220` (workspace argument)

### Related Documentation
- Current workspace isolation documented in `lightrag/api/README-zh.md:165-173`
- Storage implementations in `lightrag/kg/` directory

## Next Steps
1. Review and approve this ADR
2. Create detailed design documents for each component (see ADR 002-007)
3. Conduct security review of proposed architecture
4. Estimate development effort and allocate resources
5. Create implementation tickets and sprint planning

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Author**: Architecture Design Process
**Status**: Proposed - Awaiting Review and Approval
