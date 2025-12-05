# ADR 006: Architecture Diagrams and Alternatives Analysis

## Status: Proposed

## Proposed Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LightRAG Multi-Tenant System                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                      FastAPI Application                         │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │                                                                   │      │
│  │  ┌─────────────────────────────────────────────────────────┐    │      │
│  │  │         Request Middleware Layer                        │    │      │
│  │  ├─────────────────────────────────────────────────────────┤    │      │
│  │  │ • CORS Middleware                                      │    │      │
│  │  │ • HTTPS Redirect                                       │    │      │
│  │  │ • Rate Limiting (per tenant)                           │    │      │
│  │  │ • Request Logging & Audit                              │    │      │
│  │  │ • Idempotency Key Handling                             │    │      │
│  │  └─────────────────────────────────────────────────────────┘    │      │
│  │                          ↓                                        │      │
│  │  ┌─────────────────────────────────────────────────────────┐    │      │
│  │  │      Authentication & Tenant Context Extraction        │    │      │
│  │  ├─────────────────────────────────────────────────────────┤    │      │
│  │  │ 1. Parse JWT token or API key from headers             │    │      │
│  │  │ 2. Validate signature and expiration                   │    │      │
│  │  │ 3. Extract tenant_id, kb_id, user_id, permissions      │    │      │
│  │  │ 4. Verify token.tenant_id == path.tenant_id            │    │      │
│  │  │ 5. Verify user can access kb_id                        │    │      │
│  │  │ → Returns TenantContext object                          │    │      │
│  │  └─────────────────────────────────────────────────────────┘    │      │
│  │                          ↓                                        │      │
│  │  ┌─────────────────────────────────────────────────────────┐    │      │
│  │  │         API Routing Layer                               │    │      │
│  │  ├─────────────────────────────────────────────────────────┤    │      │
│  │  │ /api/v1/tenants/{tenant_id}/                           │    │      │
│  │  │ ├─ knowledge-bases/{kb_id}/documents/*                │    │      │
│  │  │ ├─ knowledge-bases/{kb_id}/query*                     │    │      │
│  │  │ ├─ knowledge-bases/{kb_id}/graph/*                    │    │      │
│  │  │ ├─ knowledge-bases/*                                  │    │      │
│  │  │ └─ api-keys/*                                         │    │      │
│  │  └─────────────────────────────────────────────────────────┘    │      │
│  │                          ↓                                        │      │
│  │  ┌─────────────────────────────────────────────────────────┐    │      │
│  │  │    Request Handlers (with TenantContext injected)       │    │      │
│  │  ├─────────────────────────────────────────────────────────┤    │      │
│  │  │ • Validate permissions on TenantContext                │    │      │
│  │  │ • Get tenant-specific RAG instance                     │    │      │
│  │  │ • Pass context to business logic                       │    │      │
│  │  │ • Return response with audit trail                     │    │      │
│  │  └─────────────────────────────────────────────────────────┘    │      │
│  │                                                                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │              Tenant-Aware LightRAG Instance Manager              │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │                                                                   │      │
│  │  Instance Cache:                                                 │      │
│  │  ┌─────────────────────────────────────────────────────────┐    │      │
│  │  │ (tenant_1, kb_1) → LightRAG@memory                     │    │      │
│  │  │ (tenant_1, kb_2) → LightRAG@memory                     │    │      │
│  │  │ (tenant_2, kb_1) → LightRAG@memory                     │    │      │
│  │  │ (tenant_3, kb_1) → LightRAG@memory                     │    │      │
│  │  │ ...                                                     │    │      │
│  │  │ Max: 100 instances (configurable)                      │    │      │
│  │  └─────────────────────────────────────────────────────────┘    │      │
│  │                                                                   │      │
│  │  Each LightRAG instance:                                         │      │
│  │  • Uses tenant-specific configuration (LLM, embedding models)   │      │
│  │  • Works with dedicated namespace: {tenant_id}_{kb_id}          │      │
│  │  • Isolated storage connections                                 │      │
│  │  └─────────────────────────────────────────────────────────────┘    │      │
│  │                                                                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │              Storage Access Layer (with Tenant Filtering)        │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │                                                                   │      │
│  │  Query Modification:                                             │      │
│  │  Before:  SELECT * FROM documents WHERE doc_id = 'abc'          │      │
│  │  After:   SELECT * FROM documents                               │      │
│  │           WHERE tenant_id = 'acme' AND kb_id = 'docs'           │      │
│  │           AND doc_id = 'abc'                                    │      │
│  │                                                                   │      │
│  │  • All queries automatically scoped to current tenant/KB         │      │
│  │  • Prevents accidental cross-tenant data access                 │      │
│  │  • Storage layer enforces isolation (defense in depth)          │      │
│  │                                                                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    Storage Backends (Shared)                     │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │                                                                   │      │
│  │  ┌─────────────────┐  ┌─────────────┐  ┌────────────────────┐  │      │
│  │  │   PostgreSQL    │  │   Neo4j     │  │  Milvus/Qdrant    │  │      │
│  │  │  (Shared DB)    │  │  (Shared)   │  │   (Vector Store)   │  │      │
│  │  ├─────────────────┤  ├─────────────┤  ├────────────────────┤  │      │
│  │  │ • Documents     │  │ • Entities  │  │ • Embeddings       │  │      │
│  │  │ • Chunks        │  │ • Relations │  │ • Entity vectors   │  │      │
│  │  │ • Entities      │  │             │  │                    │  │      │
│  │  │ • API Keys      │  │ Each node   │  │ Each vector        │  │      │
│  │  │ • Tenants       │  │ tagged with │  │ tagged with        │  │      │
│  │  │ • KBs           │  │ tenant_id + │  │ tenant_id + kb_id  │  │      │
│  │  │                 │  │ kb_id       │  │                    │  │      │
│  │  │ Filtered by:    │  │             │  │ Filtered by:       │  │      │
│  │  │ tenant_id,      │  │ Filtered by:│  │ tenant_id,         │  │      │
│  │  │ kb_id in WHERE  │  │ tenant_id + │  │ kb_id in query     │  │      │
│  │  │                 │  │ kb_id       │  │                    │  │      │
│  │  └─────────────────┘  └─────────────┘  └────────────────────┘  │      │
│  │                                                                   │      │
│  │  All with tenant/KB isolation at schema/data level              │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagrams

### Query Execution Flow

```
1. Client Request
   ├─ POST /api/v1/tenants/acme/knowledge-bases/docs/query
   ├─ Body: {"query": "What is..."}
   └─ Header: Authorization: Bearer <token>
          │
          ▼
2. Middleware Validation
   ├─ Extract tenant_id, kb_id from URL path
   ├─ Extract token from Authorization header
   ├─ Validate token signature and expiration
   ├─ Extract user_id, tenant_id_in_token, permissions
   └─ VERIFY: tenant_id (path) == tenant_id_in_token
          │
          ▼
3. Dependency Injection
   ├─ Create TenantContext(
   │   tenant_id="acme",
   │   kb_id="docs",
   │   user_id="john",
   │   role="editor",
   │   permissions={"query:run": true}
   └─ )
          │
          ▼
4. Handler Authorization
   ├─ Check TenantContext.permissions["query:run"] == true
   └─ If false → 403 Forbidden
          │
          ▼
5. Get RAG Instance
   ├─ RAGManager.get_instance(tenant_id="acme", kb_id="docs")
   ├─ Check cache → Found → Use cached instance
   └─ (If not cached: create new with tenant config)
          │
          ▼
6. Execute Query
   ├─ RAG.aquery(query="What is...", tenant_context=context)
   │  └─ All internal queries will include tenant/kb filters:
   │     └─ Storage layer automatically adds:
   │        WHERE tenant_id='acme' AND kb_id='docs'
          │
          ▼
7. Storage Layer Filtering
   ├─ Vector search: Find embeddings WHERE tenant_id='acme' AND kb_id='docs'
   ├─ Graph query: Match entities {tenant_id:'acme', kb_id:'docs'}
   ├─ KV lookup: Get items with key prefix 'acme:docs:'
   └─ Returns only acme/docs data (NO cross-tenant leakage possible)
          │
          ▼
8. Response Generation
   ├─ RAG generates response from filtered data
   ├─ Response object created
   └─ Handler receives response with TenantContext
          │
          ▼
9. Audit Logging
   ├─ Log: {
   │   user_id: "john",
   │   tenant_id: "acme",
   │   kb_id: "docs",
   │   action: "query_executed",
   │   status: "success",
   │   timestamp: <now>
   └─ }
          │
          ▼
10. Response Returned to Client
    └─ HTTP 200 with query result
```

### Document Upload Flow

```
1. Client uploads document
   ├─ POST /api/v1/tenants/acme/knowledge-bases/docs/documents/add
   ├─ File: document.pdf
   └─ Header: Authorization: Bearer <token>
          │
          ▼
2. Authentication & Authorization
   ├─ Validate token, extract TenantContext
   ├─ Check permission: document:create
   └─ Verify tenant_id matches path and token
          │
          ▼
3. File Validation
   ├─ Check file type (PDF, DOCX, etc.)
   ├─ Check file size < quota
   ├─ Sanitize filename
   └─ Generate unique doc_id
          │
          ▼
4. Queue Document Processing
   ├─ Store temp file: /{working_dir}/{tenant_id}/{kb_id}/__tmp__/{doc_id}
   ├─ Create DocStatus record with status="processing"
   ├─ Return to client: {status: "processing", track_id: "..."}
   └─ Start async processing task
          │
          ▼
5. Async Document Processing (background task)
   ├─ Get RAG instance for (acme, docs)
   ├─ Insert document:
   │  └─ RAG.ainsert(file_path, tenant_id="acme", kb_id="docs")
   │     └─ Internal processing automatically tags data with:
   │        └─ tenant_id="acme", kb_id="docs"
   │
   ├─ Update DocStatus:
   │  ├─ status → "success"
   │  ├─ chunks_processed → 42
   │  └─ entities_extracted → 15
   │
   └─ Move file: __tmp__ → {kb_id}/documents/
          │
          ▼
6. Storage Writes (tenant-scoped)
   ├─ PostgreSQL:
   │  └─ INSERT INTO chunks (tenant_id, kb_id, doc_id, content)
   │     VALUES ('acme', 'docs', 'doc-123', '...')
   │
   ├─ Neo4j:
   │  └─ CREATE (e:Entity {tenant_id:'acme', kb_id:'docs', name:'...'})-[:IN_KB]->(kb)
   │
   └─ Milvus:
      └─ Insert vector with metadata: {tenant_id:'acme', kb_id:'docs'}
          │
          ▼
7. Client Polls for Status
   ├─ GET /api/v1/tenants/acme/knowledge-bases/docs/documents/{doc_id}/status
   ├─ Returns: {status: "success", chunks: 42, entities: 15}
   └─ Client confirms upload complete
```

## Alternatives Considered

### Alternative 1: Separate Database Per Tenant

**Architecture:**
- Each tenant gets dedicated PostgreSQL database
- Separate Neo4j instances per tenant
- Separate Milvus collections per tenant

```
Tenant A Server → PostgreSQL A
                → Neo4j A
                → Milvus A

Tenant B Server → PostgreSQL B
                → Neo4j B
                → Milvus B
```

**Pros:**
- Maximum isolation (physical separation)
- Easier compliance (HIPAA, GDPR)
- Better disaster recovery per tenant
- Easier scaling (scale out per tenant)

**Cons:**
- ❌ Massive operational overhead
  - Each database needs separate backup, upgrade, monitoring
  - 100 tenants = 100 databases to manage
  - Database licensing costs multiply (100x more expensive)
- ❌ Complex deployment & maintenance
  - Infrastructure-as-Code becomes complex
  - Database credentials management nightmare
  - Harder debugging with distributed databases
- ❌ Impossible resource sharing
  - Cannot leverage shared compute resources
  - Cannot optimize resource usage globally
  - Waste of resources (each DB has minimum overhead)
- ❌ Cross-tenant features impossible
  - Data sharing between tenants difficult
  - Consolidated reporting/analytics hard to implement

**Decision: REJECTED**
Too expensive and operationally complex for moderate scale.

---

### Alternative 2: Dedicated Server Per Tenant

**Architecture:**
- Each tenant runs own LightRAG instance
- Own Python process, own configurations
- Own memory/CPU allocation

```
Tenant A    → LightRAG Process A (port 9621)
Tenant B    → LightRAG Process B (port 9622)
Tenant C    → LightRAG Process C (port 9623)
```

**Pros:**
- Complete isolation (separate processes)
- Easy to manage per-tenant configs
- Can use different models per tenant

**Cons:**
- ❌ Massive resource waste
  - Minimum ~500MB RAM per instance × 100 tenants = 50GB+ RAM
  - Minimum CPU overhead per process
- ❌ Extremely expensive at scale
  - 100 tenants × 4GB allocated = 400GB RAM needed
  - Infrastructure costs prohibitive
- ❌ Operational nightmare
  - 100 processes to monitor
  - 100 upgrades/patches to manage
  - Complex deployment orchestration
- ❌ Poor utilization
  - Most tenants underutilize their resources
  - Cannot rebalance resources dynamically
  - Peak loads unpredictable per tenant

**Decision: REJECTED**
Not economically viable for enterprise deployments.

---

### Alternative 3: Simple Workspace Rename (No Knowledge Base)

**Architecture:**
- Rename "workspace" to "tenant"
- No KB concept
- Assume 1 KB per tenant

```
POST /api/v1/workspaces/{workspace_id}/query
→ becomes
POST /api/v1/tenants/{tenant_id}/query
```

**Pros:**
- Minimal code changes
- Backward compatible
- Quick implementation (1 week)

**Cons:**
- ❌ No knowledge base isolation
  - Tenant with multiple unrelated KBs must share config
  - Cannot have tenant-specific KB settings
  - All data mixed together
- ❌ Cannot enforce cross-tenant access prevention
  - Workspace is just a directory/field
  - No API-level enforcement
  - Easy to make mistakes
- ❌ No RBAC
  - Cannot grant access to specific KBs
  - All-or-nothing tenant access
  - No fine-grained permissions
- ❌ No tenant-specific configuration
  - All tenants must use same LLM/embedding models
  - Cannot customize per tenant needs
- ❌ Limited compliance features
  - No audit trails of who accessed what
  - Difficult to enforce data residency
  - No resource quotas

**Decision: REJECTED**
Doesn't meet business requirements for true multi-tenancy.

---

### Alternative 4: Shared Single LightRAG for All Tenants

**Architecture:**
- One LightRAG instance for all tenants
- Single namespace, single graph
- Tenant filtering only at API layer

```
API Layer → Filters query by tenant → Single LightRAG Instance
```

**Pros:**
- Minimal resource usage
- Single deployment
- Simple to maintain

**Cons:**
- ❌ Data isolation risk is CRITICAL
  - Single point of failure for all tenants
  - One query mistake → cross-tenant data leak
  - Cannot be patched without affecting all
- ❌ Performance bottleneck
  - Single instance cannot scale with tenants
  - All LLM calls compete for resources
  - All embedding calls serialized
- ❌ Tenant-specific configuration impossible
  - All tenants share same models
  - Cannot customize chunk size, top_k, etc per tenant
- ❌ No blast radius isolation
  - One tenant's bad data can corrupt all
  - One tenant's quota exhaustion affects all
- ❌ Compliance impossible
  - Data residency requirements: cannot guarantee where data is
  - GDPR right to deletion: must delete entire system
  - Audit requirements: cannot track per-tenant operations

**Decision: REJECTED**
Unacceptable security and operational risks.

---

### Alternative 5: Sharding by Tenant Hash

**Architecture:**
- Hash tenant ID
- Route to specific shard server
- Multiple instances with different tenant ranges

```
Tenant Hash % 3
├─ Shard 0: LightRAG A (tenants 0, 3, 6, 9...)
├─ Shard 1: LightRAG B (tenants 1, 4, 7, 10...)
└─ Shard 2: LightRAG C (tenants 2, 5, 8, 11...)
```

**Pros:**
- Distributes load across instances
- Better than single instance
- Can grow to 3+ instances

**Cons:**
- ❌ Breaks operational simplicity
  - Need load balancer + routing logic
  - Shards must be preconfigured
  - Adding tenant requires determining shard
- ❌ Rebalancing is complex
  - Adding new shard requires data migration
  - Tenant addition might change shard assignment
  - Hotspots impossible to fix dynamically
- ❌ Doesn't reduce fundamental costs
  - Still need multiple instances
  - Each instance has full overhead
  - Only slightly better than per-tenant instances
- ❌ More complex than multi-tenant single instance
  - Routing logic adds latency
  - Debugging harder (data could be on any shard)
  - Cross-shard features harder to implement

**Decision: REJECTED**
Introduces complexity without enough benefit over single instance per tenant approach.

---

### Comparison Table

| Approach | Isolation | Cost | Complexity | Scalability | Selected |
|----------|-----------|------|-----------|-------------|----------|
| **Proposed: Single Instance Multi-Tenant** | ✓ Good | ✓ Low | ✓ Medium | ✓ Excellent | **✓ YES** |
| Alt 1: DB Per Tenant | ✓✓ Perfect | ✗✗ 100x | ✗✗ Very High | ✗ Limited | ✗ |
| Alt 2: Server Per Tenant | ✓ Good | ✗✗ 50x | ✗ High | ✗ Limited | ✗ |
| Alt 3: Workspace Rename | ~ Weak | ✓ Very Low | ✓ Very Low | ✓ Good | ✗ |
| Alt 4: Single Instance | ✗ Poor | ✓ Very Low | ✓ Very Low | ✗ Poor | ✗ |
| Alt 5: Sharding | ✓ Good | ✗ 10-20x | ✗✗ High | ✓ Good | ✗ |

## Why This Approach Wins

The proposed **single instance, multi-tenant, multi-KB** architecture offers the optimal balance:

1. **Security**: Complete tenant isolation through multiple layers
2. **Cost**: Efficient resource sharing (100 tenants ≈ 1.1x cost of single tenant)
3. **Complexity**: Manageable (dependency injection handles most complexity)
4. **Scalability**: Single instance can serve 100s of tenants, scales vertically well
5. **Compliance**: Audit trails and data isolation support compliance needs
6. **Features**: Supports RBAC, per-tenant config, resource quotas

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Related Files**: 001-multi-tenant-architecture-overview.md
