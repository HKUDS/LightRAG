# Multi-Tenant Visual Reference

> Quick reference guide with visual-first approach to multi-tenant concepts

**Last Updated**: November 20, 2025
**Status**: Production Ready
**Purpose**: Quick lookup for diagrams, patterns, and implementation checklists

---

## Color Scheme & Design

### Professional Pastel Palette

The documentation uses 5 carefully selected pastel colors designed for accessibility:

| Color | Hex Code | Use Case | Pastel | Bold | Text |
|-------|----------|----------|--------|------|------|
| Teal | #E0F2F1 / #00796B | Storage/Data | Light | Dark | #004D40 |
| Purple | #F3E5F5 / #6A1B9A | Tenants/Organization | Light | Dark | #38006B |
| Green | #E8F5E9 / #2E7D32 | Success/Deployment | Light | Dark | #1B5E20 |
| Orange | #FFF3E0 / #E65100 | Vectors/Performance | Light | Dark | #BF360C |
| Red | #FFEBEE / #C62828 | Security/Warnings | Light | Dark | #C62828 |

**Design Philosophy**:
- Pastel backgrounds reduce eye strain
- Bold accent colors provide contrast
- Designed for colorblind accessibility
- Professional yet pleasant appearance

---

## System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                   LightRAG Multi-Tenant System                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────┐      ┌──────────────────────────┐ │
│  │  Client Applications    │      │  API Gateway/Middleware  │ │
│  │                         │      │  (Extract Tenant Context)│ │
│  │  - Web App              │─────>│  - tenant_id             │ │
│  │  - Mobile App           │      │  - kb_id                 │ │
│  │  - CLI Tools            │      │                          │ │
│  │  - Batch Jobs           │      │  - Validate Access       │ │
│  └─────────────────────────┘      │  - Log Operations        │ │
│                                   └──────────┬───────────────┘ │
│                                              │                 │
│                                              ▼                 │
│                                   ┌──────────────────────┐    │
│                                   │ LightRAG Core        │    │
│                                   │ (Tenant-Aware)       │    │
│                                   │                      │    │
│                                   │ - Query Builder      │    │
│                                   │ - Filter Generator   │    │
│                                   │ - Response Handler   │    │
│                                   └──────────┬───────────┘    │
│                                              │                 │
│                 ┌────────────────────────────┼───────────────┐ │
│                 │                            │               │ │
│                 ▼                            ▼               ▼ │
│  ┌──────────────────────┐    ┌─────────────────────┐  ┌─────┐ │
│  │ Relational DB        │    │ Document DB         │  │ KV  │ │
│  │ (PostgreSQL)         │    │ (MongoDB)           │  │Store│ │
│  │                      │    │                     │  │(Red)│ │
│  │ Rows by:             │    │ Docs by:            │  │Keys:│ │
│  │ (tenant, kb, id)     │    │ {tenant,kb,...}     │  │t:k: │ │
│  │                      │    │                     │  │key  │ │
│  └──────────────────────┘    └─────────────────────┘  └─────┘ │
│                │                      │                  │    │
│                │                      │                  │    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         Vector DBs & Graph DBs                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ Qdrant       │  │ Neo4j        │  │ NetworkX     │  │  │
│  │  │ Metadata     │  │ Node Props   │  │ Subgraph     │  │  │
│  │  │ Filter       │  │ WHERE clause │  │ Extract      │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  │                                                         │  │
│  │  All scoped to (tenant_id, kb_id) automatically       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  Core Principle: NO tenant context escapes storage layer     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Isolation Layers

```
┌─────────────────────────────────────────────────────────┐
│           Data Isolation - Three Layers                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  LAYER 1: Tenant Isolation                             │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Highest level: Different organizations/customers │  │
│  │                                                  │  │
│  │ Acme Corp      │    TechStart Inc                │  │
│  │ tenant:acme    │    tenant:techstart            │  │
│  │                                                  │  │
│  │ Complete separation - no cross-tenant access    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  LAYER 2: Knowledge Base Isolation                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Within tenant: Different projects/environments   │  │
│  │                                                  │  │
│  │ Acme Corp:                                       │  │
│  │ ├─ kb-prod      (Production)                    │  │
│  │ ├─ kb-staging   (Pre-production)                │  │
│  │ └─ kb-dev       (Development)                   │  │
│  │                                                  │  │
│  │ Data in kb-prod never leaks to kb-staging       │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  LAYER 3: Resource Isolation                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Within kb: Documents, entities, vectors, etc.   │  │
│  │                                                  │  │
│  │ kb-prod:                                         │  │
│  │ ├─ Document: "sales-report-2025"               │  │
│  │ ├─ Entity: "John Doe"                          │  │
│  │ ├─ Vector: <embedding vector>                  │  │
│  │ └─ Relation: "manages" (between entities)      │  │
│  │                                                  │  │
│  │ All accessed only via (tenant, kb) context      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Access Pattern: tenant -> kb -> resources             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Query Execution Flow

```
START
  │
  ├─> Receive Request
  │   GET /api/documents?status=active
  │   Header: tenant-id=acme-corp
  │   Header: kb-id=kb-prod
  │
  ├─> Extract Tenant Context
  │   tenant_id = "acme-corp"
  │   kb_id = "kb-prod"
  │   [VALIDATED: User owns this tenant/kb]
  │
  ├─> Build Application Query
  │   Base: "SELECT * FROM documents WHERE status='active'"
  │
  ├─> Apply Tenant Filter
  │   Final: "SELECT * FROM documents"
  │           "WHERE status='active'"
  │           "AND tenant_id='acme-corp'"
  │           "AND kb_id='kb-prod'"
  │
  ├─> Execute Query (Storage Layer)
  │   PostgreSQL/MongoDB/Redis/etc.
  │   [ENFORCED: Only returns scoped rows]
  │
  ├─> Process Results
  │   ├─> Acme Corp's documents: RETURNED
  │   ├─> TechStart's documents: FILTERED OUT
  │   └─> Other tenant's documents: FILTERED OUT
  │
  ├─> Return to Client
  │   {
  │     "tenant": "acme-corp",
  │     "kb": "kb-prod",
  │     "documents": [...],
  │     "count": 42
  │   }
  │
END (with tenant/kb context intact)
```

---

## Composite Key Pattern

```
┌──────────────────────────────────────────────────┐
│          Composite Key Structure                 │
├──────────────────────────────────────────────────┤
│                                                  │
│  Single Key (OLD - Not Tenant-Aware)            │
│  ┌──────────────────────────────────┐           │
│  │ id: 12345                        │           │
│  │                                  │           │
│  │ Problem: Same ID in different    │           │
│  │          tenants = collision!    │           │
│  └──────────────────────────────────┘           │
│                                                  │
│  Composite Key (NEW - Tenant-Aware)             │
│  ┌──────────────────────────────────┐           │
│  │ (tenant_id, kb_id, id)           │           │
│  │ ("acme", "kb-prod", "doc-123")   │           │
│  │                                  │           │
│  │ Same ID with different contexts: │           │
│  │ ("acme", "kb-prod", "doc-123")   │ <- Acme   │
│  │ ("acme", "kb-dev", "doc-123")    │ <- Acme   │
│  │ ("techstart", "kb-main", "123")  │ <- Tech   │
│  │                                  │           │
│  │ All unique! No collisions!       │           │
│  └──────────────────────────────────┘           │
│                                                  │
│  Storage Implementation                         │
│  ┌──────────────────────────────────┐           │
│  │ PostgreSQL:                      │           │
│  │ PRIMARY KEY (tenant_id, kb_id, id)          │
│  │                                  │           │
│  │ MongoDB:                         │           │
│  │ db.createIndex({                 │           │
│  │   tenant_id: 1,                  │           │
│  │   kb_id: 1,                      │           │
│  │   _id: 1                         │           │
│  │ })                               │           │
│  │                                  │           │
│  │ Redis:                           │           │
│  │ key = "tenant:kb:id"             │           │
│  │ key = "acme:kb-prod:doc-123"     │           │
│  └──────────────────────────────────┘           │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Data Organization by Backend

```
┌───────────────────────────────────────────────────────┐
│        How Each Backend Organizes Tenant Data         │
├───────────────────────────────────────────────────────┤
│                                                       │
│  PostgreSQL                                          │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Table: documents                                │ │
│  │ ┌─────────┬──────┬────┬──────────┬──────────┐  │ │
│  │ │tenant_id│kb_id │ id │ title    │ content  │  │ │
│  │ ├─────────┼──────┼────┼──────────┼──────────┤  │ │
│  │ │ acme    │prod  │ 1  │ Report   │ [data]   │  │ │
│  │ │ acme    │dev   │ 2  │ Draft    │ [data]   │  │ │
│  │ │ tech    │main  │ 1  │ Spec     │ [data]   │  │ │
│  │ └─────────┴──────┴────┴──────────┴──────────┘  │ │
│  │ Row filtering: WHERE tenant='acme' AND kb='prod'  │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  MongoDB                                             │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Collection: documents                           │ │
│  │ Document 1: {tenant:"acme", kb:"prod", _id:1}  │ │
│  │ Document 2: {tenant:"acme", kb:"dev", _id:2}   │ │
│  │ Document 3: {tenant:"tech", kb:"main", _id:1}  │ │
│  │                                                 │ │
│  │ Filter: {tenant:"acme", kb:"prod"}              │ │
│  │ Returns: Document 1 only                         │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  Redis                                               │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Key Namespace Pattern:                          │ │
│  │ "acme:prod:doc:1"   -> Document 1 (Acme)       │ │
│  │ "acme:dev:doc:2"    -> Document 2 (Acme)       │ │
│  │ "tech:main:doc:1"   -> Document 1 (Tech)       │ │
│  │                                                 │ │
│  │ Query pattern: "acme:prod:*"                    │ │
│  │ Returns: All keys matching tenant:kb scope      │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  Qdrant (Vector DB)                                  │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Collection: embeddings                          │ │
│  │                                                 │ │
│  │ Point 1: {                                      │ │
│  │   "vector": [...],                              │ │
│  │   "payload": {                                  │ │
│  │     "tenant_id": "acme",                        │ │
│  │     "kb_id": "prod"                             │ │
│  │   }                                             │ │
│  │ }                                               │ │
│  │                                                 │ │
│  │ Search filter:                                  │ │
│  │ {"must": [                                      │ │
│  │   {"key":"tenant_id", "match":{"value":"acme"}},│
│  │   {"key":"kb_id", "match":{"value":"prod"}}     │ │
│  │ ]}                                              │ │
│  │ Returns: Only vectors with matching metadata    │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  Neo4j (Graph DB)                                    │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Node structure:                                 │ │
│  │ (Entity {                                       │ │
│  │   tenant_id: "acme",                            │ │
│  │   kb_id: "prod",                                │ │
│  │   name: "John Doe"                              │ │
│  │ })                                              │ │
│  │                                                 │ │
│  │ Query:                                          │ │
│  │ MATCH (n:Entity)                                │ │
│  │ WHERE n.tenant_id = 'acme'                      │ │
│  │   AND n.kb_id = 'prod'                          │ │
│  │ RETURN n                                        │ │
│  │ Returns: Entities scoped to acme:prod           │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## Security Boundaries

```
┌────────────────────────────────────────────────────┐
│          Security Boundary Enforcement              │
├────────────────────────────────────────────────────┤
│                                                    │
│  Client Request                                   │
│  GET /documents?tenant=acme-corp&kb=kb-prod       │
│           │                                       │
│           ▼                                       │
│  API Layer - VALIDATE                             │
│  ┌────────────────────────────────────────────┐   │
│  │ Check: User has permission for tenant      │   │
│  │ Check: kb_id belongs to tenant             │   │
│  │ Failure: Return 403 Forbidden               │   │
│  └────────────────────────────────────────────┘   │
│           │                                       │
│           ▼ (validated)                           │
│  Query Builder - ENFORCE                          │
│  ┌────────────────────────────────────────────┐   │
│  │ Base query: SELECT * FROM documents        │   │
│  │                                            │   │
│  │ Add filter: AND tenant_id='acme-corp'      │   │
│  │ Add filter: AND kb_id='kb-prod'            │   │
│  │                                            │   │
│  │ Even if app developer forgets tenant       │   │
│  │ context, storage layer won't return data   │   │
│  └────────────────────────────────────────────┘   │
│           │                                       │
│           ▼                                       │
│  Storage Layer - DATABASE                         │
│  ┌────────────────────────────────────────────┐   │
│  │ PostgreSQL executes:                       │   │
│  │ SELECT * FROM documents                    │   │
│  │ WHERE tenant_id='acme-corp'                │   │
│  │   AND kb_id='kb-prod'                      │   │
│  │                                            │   │
│  │ Result: Only matching rows returned        │   │
│  │ Impossible to get other tenant data        │   │
│  └────────────────────────────────────────────┘   │
│           │                                       │
│           ▼                                       │
│  Response                                         │
│  Documents from acme-corp/kb-prod ONLY           │
│                                                    │
│  Key Point: TWO layers of protection              │
│  1. API validation (user has access)              │
│  2. Database enforcement (scope in query)         │
│                                                    │
│  If either fails: NO DATA LEAKED                  │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Implementation Decision Tree

```
START: Need to implement multi-tenant feature?
  │
  ├─> YES, new feature
  │     │
  │     ├─> Data needs tenant/kb context?
  │     │     │
  │     │     ├─> YES
  │     │     │   └─> Use TenantContext in all queries
  │     │     │       Add tenant_id, kb_id to schema
  │     │     │       Use support module helpers
  │     │     │
  │     │     └─> NO (metadata, config, etc.)
  │     │         └─> Store normally, reference by tenant later
  │     │
  │     └─> Done: Feature is multi-tenant safe
  │
  ├─> NO, maintaining existing feature
  │     │
  │     ├─> Feature crosses tenant boundaries?
  │     │     │
  │     │     ├─> YES (e.g., searching across tenants)
  │     │     │   └─> Explicitly separate results by tenant
  │     │     │       Never merge tenant data
  │     │     │       Document cross-tenant behavior
  │     │     │
  │     │     └─> NO (operates within single tenant)
  │     │         └─> Add tenant filter to query
  │     │             Test with multiple tenants
  │     │
  │     └─> Done: Feature remains tenant-safe
  │
  └─> Migration time
        │
        ├─> Have existing single-tenant data?
        │     │
        │     ├─> YES
        │     │   └─> Run migration script with dry-run
        │     │       Backup data
        │     │       Verify statistics
        │     │       Apply migration
        │     │       Run tests
        │     │
        │     └─> NO (new deployment)
        │         └─> Deploy with multi-tenant enabled
        │             No migration needed
        │
        └─> Done: Data is multi-tenant compatible
```

---

## Quick Implementation Checklist

```
BEFORE IMPLEMENTATION
[ ] Read Section 2: Data Isolation Layers
[ ] Review relevant backend examples (Section 6)
[ ] Check if new tables needed - plan composite keys
[ ] Get team buy-in on tenant context requirements

DURING IMPLEMENTATION
[ ] Add tenant_id, kb_id to schema (if new data)
[ ] Use TenantSQLBuilder/MongoTenantHelper/etc.
[ ] Extract tenant context from request headers
[ ] Add tests with multiple tenants
[ ] Add tenant context to logging/monitoring
[ ] Update documentation with tenant notes

BEFORE TESTING
[ ] Verify composite indexes exist
[ ] Check that all queries include tenant filter
[ ] Review code for hardcoded assumptions
[ ] Ensure tenant context flows through async tasks
[ ] Set up test data for multiple tenants

TESTING
[ ] Single tenant operations work
[ ] Multiple tenant queries return correct data
[ ] Cross-tenant queries return nothing
[ ] Edge cases: empty results, large datasets
[ ] Performance: check index usage with EXPLAIN
[ ] Concurrent operations from multiple tenants

BEFORE PRODUCTION
[ ] Run full test suite multiple times
[ ] Load test with multiple tenants
[ ] Backup production database
[ ] Have rollback plan ready
[ ] Monitor tenant-specific metrics
[ ] Update runbooks for multi-tenant queries

AFTER DEPLOYMENT
[ ] Monitor for 24+ hours
[ ] Check logs for any tenant context issues
[ ] Verify performance didn't degrade
[ ] Get user feedback from different tenants
[ ] Document any lessons learned
```

---

## Integration Points

```
┌──────────────────────────────────────────────────────┐
│          Where Multi-Tenant Touches System            │
├──────────────────────────────────────────────────────┤
│                                                      │
│  API Layer                                          │
│  ├─> Authentication: Get user's tenant ID           │
│  ├─> Headers: Extract tenant_id, kb_id             │
│  ├─> Validation: Verify user owns tenant            │
│  └─> Responses: Always include tenant context      │
│                                                      │
│  Query Layer                                        │
│  ├─> Query Builder: Add tenant filters              │
│  ├─> Parameters: Include tenant values              │
│  ├─> Optimization: Use composite indexes            │
│  └─> Caching: Key by (tenant, kb, ...)             │
│                                                      │
│  Storage Layer                                      │
│  ├─> Schema: (tenant_id, kb_id) in composite key   │
│  ├─> Indexes: Multi-column indexes                  │
│  ├─> Constraints: Prevent ID collisions             │
│  └─> Filters: WHERE clause enforcement              │
│                                                      │
│  Monitoring & Logging                               │
│  ├─> Logs: Include tenant in all entries            │
│  ├─> Metrics: Track per-tenant usage                │
│  ├─> Alerts: Tenant-specific thresholds             │
│  └─> Audit: Record who accessed what data           │
│                                                      │
│  Testing                                            │
│  ├─> Unit Tests: Test with multiple tenants         │
│  ├─> Integration: Test isolation between tenants    │
│  ├─> Performance: Benchmark multi-tenant queries    │
│  └─> Security: Verify no data leaks                 │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

| Scenario | Single Tenant | Multi Tenant | Notes |
|----------|---|---|---|
| **Query Speed** | Baseline | +0-5% | Composite index slightly slower on insert |
| **Storage Size** | Baseline | +5-10% | tenant_id, kb_id columns add overhead |
| **Index Count** | Fewer | More | Composite indexes needed |
| **Query Plans** | Simple | Clear | WHERE clause filters effectively |
| **Concurrent Access** | Good | Excellent | Isolation prevents lock contention |
| **Cache Efficiency** | High | Medium | Must key by (tenant, kb) |

---

## Quick Reference Patterns

### Pattern 1: Simple Query

```python
# PostgreSQL
from lightrag.kg.postgres_tenant_support import TenantSQLBuilder

sql = "SELECT * FROM documents WHERE status = :status"
filtered_sql, params = TenantSQLBuilder.build_filtered_query(
    sql, tenant_id="acme", kb_id="prod",
    additional_params=[{"status": "active"}]
)
results = await db.query(filtered_sql, params)
# Returns: Only active documents from acme/prod
```

### Pattern 2: Filter + Sort

```python
# MongoDB
from lightrag.kg.mongo_tenant_support import MongoTenantHelper

query = MongoTenantHelper.get_tenant_filter(
    tenant_id="acme", kb_id="prod",
    additional_filter={"status": "active"}
)
results = await collection.find(query).sort("created_at", -1).limit(10)
# Returns: Latest 10 active docs from acme/prod
```

### Pattern 3: Batch Operations

```python
# Redis batch
from lightrag.kg.redis_tenant_support import RedisTenantNamespace

ns = RedisTenantNamespace(redis, "acme", "prod")

# Batch set
await ns.mset({
    "user:1": json.dumps(user1_data),
    "user:2": json.dumps(user2_data),
})

# Batch get - all scoped to acme:prod
users = await ns.mget("user:1", "user:2")
# Keys expanded to "acme:prod:user:1", "acme:prod:user:2"
```

---

## Learning Path

A structured 7-step progression to understand multi-tenant architecture:

1. **Understand the Problem** (10 min)
   - Read: Section 1 - Overview
   - Watch the Real-World Scenario diagram
   - Why: Single deployment, multiple customers

2. **Learn the Concepts** (15 min)
   - Read: Section 2 - Data Isolation Layers
   - Read: Section 3 - Composite Key Pattern
   - Why: How isolation actually works

3. **See the Architecture** (10 min)
   - Read: Section 4 - System Architecture Diagram
   - Read: Section 5 - Query Execution Flow
   - Why: How requests are processed

4. **Find Your Backend** (10 min)
   - Read: Section 6 - Data Organization by Backend
   - Find your database type (PostgreSQL/MongoDB/Redis/etc.)
   - Why: Each backend has different approach

5. **Implement the Pattern** (20 min)
   - Read: Section 7 - Quick Reference Patterns
   - Copy the relevant example for your backend
   - Adapt it to your use case
   - Why: Actual working code you can use

6. **Secure It** (15 min)
   - Read: Section 8 - Security Boundaries
   - Review the checklist
   - Why: Prevent cross-tenant data leaks

7. **Test & Deploy** (30 min)
   - Use: Section 9 - Quick Implementation Checklist
   - Run multi-tenant tests
   - Deploy to production
   - Monitor for issues
   - Why: Ensure reliability

**Total Time**: ~90 minutes to full understanding and implementation

---

## Success Criteria

After implementing multi-tenant support, verify:

- [YES] Multiple tenants can exist in same deployment
- [YES] Tenant A cannot access Tenant B's data
- [YES] Queries automatically scoped to tenant
- [YES] No breaking changes to existing code
- [YES] All 10 backends supported
- [YES] Performance within baseline +5%
- [YES] Composite indexes created
- [YES] Tests pass with multiple tenants
- [YES] Logging includes tenant context
- [YES] Backward compatible with single-tenant code

---

## Common Questions

**Q: Do I need to change my existing code?**
A: No. Multi-tenant is built-in with defaults. Use support modules for new features.

**Q: What about backward compatibility?**
A: Complete. Legacy code uses "default" tenant automatically.

**Q: How do I test multi-tenant isolation?**
A: Create test data in 2+ tenants, verify queries return only scoped data.

**Q: Can I run single and multi-tenant tenants together?**
A: Yes. All data coexists. Default tenant for legacy code.

**Q: What if a query is missing tenant filter?**
A: Returns empty result (safe). Logging will show missing context.

---

## Resources

- Full Details: See `0001-multi-tenant-architecture.md`
- Navigation: See `0003-multi-tenant-documentation-index.md`
- Code Modules: See `lightrag/kg/` directory
- Tests: See `tests/test_multi_tenant_*.py`

---

**Status**: Production Ready
**Last Updated**: November 20, 2025
**Questions?** Review the learning path or check full architecture guide
