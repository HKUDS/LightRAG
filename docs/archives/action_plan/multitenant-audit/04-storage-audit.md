# Storage Layer Multi-Tenant Audit

**Date:** November 29, 2025
**Status:** In Progress

---

## Overview

This document audits the multi-tenant implementation in the LightRAG storage layer, including PostgreSQL, Redis, and Vector databases.

## Components Under Audit

### 1. PostgreSQL Multi-Tenant Support

#### Table Schema (`kg/postgres_tenant_support.py`)

**Table DDL Pattern:**
```sql
CREATE TABLE LIGHTRAG_DOC_STATUS (
    tenant_id VARCHAR(255) NOT NULL,
    kb_id VARCHAR(255) NOT NULL,
    workspace VARCHAR(255),
    id VARCHAR(255) NOT NULL,
    ...
    CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (tenant_id, kb_id, id)
)
```

**✅ Strengths:**
- All tables have `tenant_id` and `kb_id` columns
- Composite primary keys enforce uniqueness per tenant/KB
- Indexes designed for tenant-scoped queries

**Tables with Multi-Tenant Support:**
| Table | tenant_id | kb_id | Composite PK |
|-------|-----------|-------|--------------|
| LIGHTRAG_DOC_FULL | ✅ | ✅ | ✅ |
| LIGHTRAG_DOC_CHUNKS | ✅ | ✅ | ✅ |
| LIGHTRAG_VDB_CHUNKS | ✅ | ✅ | ✅ |
| LIGHTRAG_VDB_ENTITY | ✅ | ✅ | ✅ |
| LIGHTRAG_VDB_RELATION | ✅ | ✅ | ✅ |
| LIGHTRAG_LLM_CACHE | ✅ | ✅ | ✅ |
| LIGHTRAG_DOC_STATUS | ✅ | ✅ | ✅ |
| LIGHTRAG_FULL_ENTITIES | ✅ | ✅ | ✅ |
| LIGHTRAG_FULL_RELATIONS | ✅ | ✅ | ✅ |

#### SQL Builder (`TenantSQLBuilder`)

```python
@staticmethod
def add_tenant_filter(sql: str, table_alias: str = "", param_index: int = 1) -> Tuple[str, int]:
    tenant_filter = f"{prefix}tenant_id=${param_index} AND {prefix}kb_id=${param_index + 1}"
    if "WHERE" in sql:
        sql = sql.replace("WHERE", f"WHERE {tenant_filter} AND", 1)
    else:
        sql += f" WHERE {tenant_filter}"
    return sql, param_index + 2
```

**✅ Strengths:**
- Automatic injection of tenant filters
- Parameterized queries (SQL injection safe)
- Handles both existing WHERE and new WHERE clauses

**⚠️ Potential Issues:**
- Simple string replacement - could fail on complex queries
- No validation of sql input

#### Context Variable (`utils_context.py`)

```python
tenant_id_var: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)

def get_current_tenant_id() -> Optional[str]:
    return tenant_id_var.get()
```

**✅ Strengths:**
- Thread-safe and async-safe via ContextVar
- Can be accessed deep in the call stack

**⚠️ Potential Issues:**
- Returns None by default (needs checking by callers)
- No kb_id context variable observed

#### PostgreSQL RLS (`postgres_rls.sql`)

**Purpose:** Row-Level Security for additional protection.

```sql
-- Tenant RLS policy
CREATE POLICY tenant_isolation ON LIGHTRAG_DOC_STATUS
    USING (tenant_id = current_setting('app.current_tenant', true));
```

**✅ Strengths:**
- Defense-in-depth security
- Database-level enforcement
- Even if application bypasses, RLS blocks access

**⚠️ Potential Issues:**
- Requires setting `app.current_tenant` before each query
- May impact performance

### 2. Redis Multi-Tenant Support (`kg/redis_tenant_support.py`)

#### Key Pattern

```python
@staticmethod
def make_tenant_key(tenant_id: str, kb_id: str, original_key: str) -> str:
    return f"{tenant_id}:{kb_id}:{original_key}"
```

**Format:** `tenant_id:kb_id:original_key`

**Examples:**
- `acme:kb-prod:doc-123`
- `techstart:kb-main:entity-456`

**✅ Strengths:**
- Consistent namespace prefixing
- Easy to scan for tenant-specific keys
- Clear separation of concerns

**⚠️ Potential Issues:**
- Keys with `:` in original_key could cause parsing issues
- No encryption of tenant data

#### Namespace Manager (`RedisTenantNamespace`)

```python
class RedisTenantNamespace:
    async def get(self, key: str) -> Optional[Any]:
        tenant_key = RedisTenantHelper.make_tenant_key(self.tenant_id, self.kb_id, key)
        return await self.redis.get(tenant_key)
```

**✅ Strengths:**
- Encapsulates tenant logic
- Prevents accidental access to other tenants
- Batch operations supported

### 3. Vector Database Multi-Tenant Support (`kg/vector_tenant_support.py`)

#### Metadata Injection

```python
@staticmethod
def add_tenant_metadata(payload: Dict[str, Any], tenant_id: str, kb_id: str) -> Dict[str, Any]:
    payload["tenant_id"] = tenant_id
    payload["kb_id"] = kb_id
    return payload
```

#### Query Filtering

**Qdrant Filter:**
```python
def build_qdrant_filter(tenant_id: str, kb_id: str, additional_filter: Dict = None) -> Dict[str, Any]:
    must_conditions = [
        {"key": "tenant_id", "match": {"value": tenant_id}},
        {"key": "kb_id", "match": {"value": kb_id}}
    ]
    return {"must": must_conditions}
```

**Milvus Expression:**
```python
def build_milvus_expr(tenant_id: str, kb_id: str, additional_expr: str = None) -> str:
    expr = f'tenant_id == "{tenant_id}" && kb_id == "{kb_id}"'
```

**✅ Strengths:**
- Supports multiple vector DB backends
- Filter-based isolation (no collection per tenant needed)
- Efficient for large number of tenants

**⚠️ Potential Issues:**
- Filter overhead on every query
- No index on tenant_id/kb_id in some backends

#### Collection Naming (Alternative Approach)

```python
@staticmethod
def create_tenant_collection_name(base_name: str, tenant_id: str, kb_id: str) -> str:
    return f"{base_name}_{tenant_id}_{kb_id}".replace("-", "_")
```

**Use Case:** Separate collections per tenant for:
- Stronger isolation
- Easier tenant deletion
- Independent scaling

---

## Detailed Findings

### Finding STG-001: No kb_id in ContextVar
**Severity:** Medium
**Location:** `utils_context.py`

**Description:**
Only `tenant_id` is stored in ContextVar. The `kb_id` must be passed explicitly, which could lead to inconsistencies.

**Recommendation:**
Add `kb_id_var: ContextVar[Optional[str]]` for complete context propagation.

### Finding STG-002: Simple SQL String Replacement
**Severity:** Low
**Location:** `postgres_tenant_support.py`

**Description:**
The `add_tenant_filter` function uses simple string replacement:
```python
sql = sql.replace("WHERE", f"WHERE {tenant_filter} AND", 1)
```

This could fail on:
- CTEs with nested WHERE clauses
- Complex subqueries
- Case variations (where vs WHERE)

**Recommendation:**
Use proper SQL parsing or ORM-based filtering.

### Finding STG-003: Redis Key Collision Risk
**Severity:** Low
**Location:** `redis_tenant_support.py`

**Description:**
If `original_key` contains `:`, parsing could return incorrect results:
```python
parts = tenant_key.split(":", 2)
# With key "acme:kb-prod:my:special:key"
# Returns: tenant_id="acme", kb_id="kb-prod", original_key="my:special:key" ✅
```

The `split(2)` handles this correctly, but there's no validation preventing `:` in tenant_id or kb_id.

**Recommendation:**
Validate that tenant_id and kb_id don't contain the separator character.

### Finding STG-004: RLS Setting Not Always Applied
**Severity:** Medium
**Location:** `postgres_impl.py`

**Description:**
The tenant context is set in specific places:
```python
tenant_id = get_current_tenant_id()
if tenant_id:
    await connection.execute(f"SET app.current_tenant = '{tenant_id}'")
```

If `get_current_tenant_id()` returns None, RLS may block all access.

**Recommendation:**
Ensure tenant context is always set before any database operation.

### Finding STG-005: Vector Metadata Not Indexed
**Severity:** Low
**Location:** Vector DB implementations

**Description:**
Tenant filtering adds overhead to every vector query. Without proper indexing on `tenant_id`/`kb_id`, queries may be slow with many tenants.

**Recommendation:**
- Create index on `tenant_id`, `kb_id` metadata fields
- Consider partition collection by tenant for high-volume deployments

---

## Data Isolation Verification

### Test: PostgreSQL Isolation

```sql
-- Verify tenant_id is always set
SELECT COUNT(*) FROM lightrag_doc_status WHERE tenant_id IS NULL;
-- Expected: 0

-- Verify no cross-tenant data
SELECT tenant_id, kb_id, COUNT(*)
FROM lightrag_doc_status
GROUP BY tenant_id, kb_id;
-- Each row should show isolated counts

-- Test RLS (should return empty without setting tenant)
SELECT * FROM lightrag_doc_status LIMIT 5;
-- With RLS enabled and no app.current_tenant set: 0 rows
```

### Test: Redis Isolation

```bash
# List all keys for a tenant
redis-cli KEYS "tenant_a:*"

# Verify no keys without tenant prefix
redis-cli KEYS "*" | grep -v ":"
# Should be empty (all keys should be tenant-prefixed)
```

### Test: Vector DB Isolation

```python
# Query without tenant filter (should fail or return nothing)
results = collection.search(query_vector)
# Expected: Empty or error

# Query with correct tenant filter
results = collection.search(
    query_vector,
    filter={"tenant_id": "tenant_a", "kb_id": "kb_1"}
)
# Expected: Only tenant_a data
```

---

## Composite Key Pattern

The multi-tenant system uses composite keys throughout:

| Layer | Key Format |
|-------|------------|
| PostgreSQL PK | `(tenant_id, kb_id, id)` |
| Redis Key | `tenant_id:kb_id:original_key` |
| Vector ID | `tenant_id:kb_id:original_id` |
| Vector Metadata | `{tenant_id, kb_id, ...}` |

**Benefits:**
- Consistent isolation pattern
- Easy to identify tenant ownership
- Natural grouping for batch operations

**Drawbacks:**
- Longer keys/IDs
- Parsing overhead
- Can't use simple auto-increment IDs

---

## Migration Support

### Adding Tenant Columns

```python
async def add_tenant_columns_migration(db, table_name: str, tenant_id: str = "default", kb_id: str = "default"):
    # Adds tenant_id and kb_id columns
    # Populates with default values for existing data
```

**✅ Strengths:**
- Safe migration for existing deployments
- Default values prevent null issues

**⚠️ Caution:**
Existing data in a "default" tenant should be migrated to proper tenants.

---

## Conclusion

The storage layer has comprehensive multi-tenant support:

1. **PostgreSQL:** Composite PKs, parameterized queries, RLS support
2. **Redis:** Namespace prefixes, helper classes
3. **Vector DBs:** Metadata filtering, collection naming

Key concerns:
- **Medium:** No kb_id in ContextVar
- **Medium:** RLS not always applied if context missing
- **Low:** Simple SQL string replacement
- **Low:** Potential key parsing edge cases

Recommendations:
1. Add `kb_id` to ContextVar for complete context
2. Validate tenant context is set before all DB operations
3. Add index on tenant metadata in vector DBs
4. Consider SQL parsing library for complex queries
