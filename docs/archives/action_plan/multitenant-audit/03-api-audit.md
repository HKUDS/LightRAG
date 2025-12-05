# REST API Multi-Tenant Audit

**Date:** November 29, 2025
**Status:** In Progress

---

## Overview

This document audits the multi-tenant implementation in the LightRAG REST API (FastAPI backend).

## Components Under Audit

### 1. Dependency Injection (`api/dependencies.py`)

**Purpose:** Extract and validate tenant context from request headers.

#### `get_tenant_context()` - Required Tenant Context

```python
async def get_tenant_context(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_kb_id: Optional[str] = Header(None, alias="X-KB-ID"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> TenantContext:
```

**✅ Strengths:**
- Raises HTTPException if tenant_id is missing
- Validates authorization header format
- Priority system for tenant_id sources (middleware > token > header)
- Automatic resolution of "default" tenant/KB

**⚠️ Potential Issues:**
- Falls back to `kb_id = "default"` if not provided
- No validation that tenant_id/kb_id are valid UUIDs before use

#### `get_tenant_context_optional()` - Optional Tenant Context

```python
async def get_tenant_context_optional(...) -> Optional[TenantContext]:
    if x_tenant_id:
        # If X-Tenant-ID provided, full validation required
        return await get_tenant_context(...)
    try:
        return await get_tenant_context(...)
    except HTTPException:
        return None  # Falls back to global RAG
```

**⚠️ Critical Finding:**
This function allows requests to proceed without tenant context, falling back to global RAG. This could cause:
- Data leakage if global RAG contains data from multiple tenants
- Confusion about which data is being accessed

### 2. Tenant RAG Manager (`tenant_rag_manager.py`)

**Purpose:** Manages per-tenant LightRAG instances with caching.

```python
async def get_rag_instance(
    self,
    tenant_id: str,
    kb_id: str,
    user_id: Optional[str] = None,
) -> LightRAG:
```

**✅ Strengths:**
- LRU caching for memory efficiency
- Security validation of identifiers
- User access verification
- Separate working directories per tenant/KB
- Double-check locking for thread safety

**⚠️ Potential Issues:**
- `user_id` parameter is optional, allowing bypass of access control
- Warning logged but no error when `user_id` is None:
  ```python
  logger.warning(
      f"No user_id provided for tenant access - allowing for backward compatibility"
  )
  ```

### 3. Document Routes (`routers/document_routes.py`)

**Purpose:** CRUD operations for documents within tenant/KB context.

**Analysis:**

```python
async def get_tenant_rag(
    tenant_context: Optional[TenantContext] = Depends(get_tenant_context_optional)
) -> LightRAG:
    if rag_manager and tenant_context and tenant_context.tenant_id and tenant_context.kb_id:
        return await rag_manager.get_rag_instance(...)
    return rag  # Falls back to global RAG!
```

**⚠️ Critical Finding:**
All document routes use `get_tenant_context_optional`, meaning:
- If no tenant headers provided, uses global RAG
- Could allow document operations on wrong tenant
- Upload/delete could affect global data

**Affected Endpoints:**
- `POST /documents/scan` - Scans input directory
- `POST /documents/upload` - Uploads file
- `POST /documents/text` - Inserts text
- `DELETE /documents` - Deletes documents
- `GET /documents` - Lists documents
- `GET /documents/pipeline-status` - Gets pipeline status

### 4. Query Routes (`routers/query_routes.py`)

**Purpose:** Query operations against the knowledge base.

**Same pattern as document routes:**

```python
async def get_tenant_rag(
    tenant_context: Optional[TenantContext] = Depends(get_tenant_context_optional)
) -> LightRAG:
    if rag_manager and tenant_context and tenant_context.tenant_id and tenant_context.kb_id:
        return await rag_manager.get_rag_instance(...)
    return rag  # Falls back to global RAG
```

**⚠️ Same concern:** Queries without tenant context go to global RAG.

### 5. Tenant Routes (`routers/tenant_routes.py`)

**Purpose:** CRUD for tenants and knowledge bases.

**✅ Strengths:**
- Tenant list is public (for tenant selection on login)
- Admin operations require authentication
- KB operations are tenant-scoped via headers

**Analysis of Key Endpoints:**

```python
@router.get("/tenants")
async def list_tenants(...):
    # Public endpoint - no auth required
    # Intentional for login page tenant selection
```

```python
@router.post("/tenants")
async def create_tenant(
    ...,
    admin_context: dict = Depends(get_admin_context)
):
    # Requires admin context
```

```python
@router.get("/knowledge-bases")
async def list_knowledge_bases(
    context: TenantContext = Depends(get_tenant_context_no_kb)
):
    # Requires tenant context, but not KB
```

### 6. Tenant Service (`services/tenant_service.py`)

**Purpose:** Business logic for tenant/KB operations.

**Key Security Function:**

```python
async def verify_user_access(
    self,
    user_id: str,
    tenant_id: str,
    required_role: str = "viewer"
) -> bool:
    # TEMPORARY: Admin bypass
    if user_id.lower() == "admin":
        return True  # ⚠️ Security concern!

    # Check PostgreSQL membership table
    result = await self.kv_storage.db.query(
        "SELECT has_tenant_access($1, $2, $3) as has_access",
        [user_id, tenant_id, required_role]
    )
```

**⚠️ Security Concern:**
The admin bypass (`if user_id.lower() == "admin": return True`) is marked as temporary but could be exploited.

---

## Detailed Findings

### Finding API-001: Optional Tenant Context Allows Global RAG Access
**Severity:** High
**Location:** `document_routes.py`, `query_routes.py`

**Description:**
Using `get_tenant_context_optional` allows requests without `X-Tenant-ID` header to fall back to global RAG instance. This could:
- Expose data from all tenants if global RAG is shared
- Allow document operations on unintended data
- Create confusion about data scope

**Recommendation:**
- Use `get_tenant_context` (required) for multi-tenant deployments
- Add configuration flag to enforce tenant context
- Add global RAG deprecation warning

### Finding API-002: User ID Optional in RAG Manager
**Severity:** Medium
**Location:** `tenant_rag_manager.py`

**Description:**
The `user_id` parameter in `get_rag_instance()` is optional:
```python
if user_id:
    has_access = await self.tenant_service.verify_user_access(...)
else:
    logger.warning("No user_id provided - allowing for backward compatibility")
```

This allows bypassing access control for backward compatibility.

**Recommendation:**
- Deprecate the no-user-id path
- Add configuration to require user_id
- Audit all callers to ensure user_id is passed

### Finding API-003: Admin User Bypass
**Severity:** High
**Location:** `services/tenant_service.py`

**Description:**
Any user with username "admin" (case-insensitive) can access any tenant:
```python
if user_id.lower() == "admin":
    return True
```

**Recommendation:**
- Remove this bypass or make it configurable
- Use proper role-based admin access
- Log admin access attempts

### Finding API-004: Default KB Fallback
**Severity:** Low
**Location:** `dependencies.py`

**Description:**
If `kb_id` is not provided but `tenant_id` is, the code defaults to "default":
```python
if not kb_id:
    if tenant_id:
        kb_id = "default"
```

This could lead to unintended operations on a default KB.

**Recommendation:**
- Make KB ID required for data operations
- Only use default KB for tenant-level operations
- Document this behavior clearly

---

## Request Flow Analysis

### Flow 1: Authenticated Request with Tenant Context

```
Client Request
    │
    ├─ Headers: Authorization, X-Tenant-ID, X-KB-ID
    │
    ▼
get_tenant_context()
    │
    ├─ Validate Authorization → Extract username, role
    ├─ Extract tenant_id (middleware > token > header)
    ├─ Resolve "default" tenant → first accessible tenant
    ├─ Extract kb_id (token > header > "default")
    ├─ Resolve "default" kb → first KB in tenant
    │
    ▼
TenantContext(tenant_id, kb_id, user_id, role)
    │
    ▼
get_tenant_rag()
    │
    ├─ rag_manager.get_rag_instance(tenant_id, kb_id, user_id)
    │     ├─ Validate identifiers
    │     ├─ Verify user access
    │     ├─ Create/cache tenant-specific RAG
    │
    ▼
Tenant-Specific LightRAG Instance
```

### Flow 2: Request Without Tenant Context (Fallback)

```
Client Request
    │
    ├─ Headers: Authorization only (no tenant headers)
    │
    ▼
get_tenant_context_optional()
    │
    ├─ No X-Tenant-ID → Try get_tenant_context()
    ├─ Fails → Return None
    │
    ▼
get_tenant_rag()
    │
    ├─ tenant_context is None → return global rag
    │
    ▼
Global RAG Instance (⚠️ Not tenant-isolated!)
```

---

## Test Scenarios

### Scenario API-T1: Required Tenant Context
```bash
# Request without tenant headers - should succeed with optional, fail with required
curl -X GET http://localhost:9621/documents \
  -H "Authorization: Bearer <token>"
```

**Expected with get_tenant_context:** HTTP 400 (Missing tenant_id)
**Expected with get_tenant_context_optional:** Uses global RAG (⚠️)

### Scenario API-T2: Cross-Tenant Access Prevention
```bash
# User from Tenant A tries to access Tenant B
curl -X GET http://localhost:9621/documents \
  -H "Authorization: Bearer <token_tenant_a>" \
  -H "X-Tenant-ID: tenant_b_id" \
  -H "X-KB-ID: kb_b_id"
```

**Expected:** HTTP 403 (Access denied)

### Scenario API-T3: Document Isolation
```bash
# Upload to Tenant A
curl -X POST http://localhost:9621/documents/upload \
  -H "Authorization: Bearer <token>" \
  -H "X-Tenant-ID: tenant_a_id" \
  -H "X-KB-ID: kb_a_id" \
  -F "file=@test.txt"

# List from Tenant B - should not see Tenant A's document
curl -X GET http://localhost:9621/documents \
  -H "Authorization: Bearer <token>" \
  -H "X-Tenant-ID: tenant_b_id" \
  -H "X-KB-ID: kb_b_id"
```

**Expected:** Empty document list for Tenant B

### Scenario API-T4: Query Isolation
```bash
# Query in Tenant A context
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -H "X-Tenant-ID: tenant_a_id" \
  -H "X-KB-ID: kb_a_id" \
  -d '{"query": "test query"}'

# Same query in Tenant B context - should get different result
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -H "X-Tenant-ID: tenant_b_id" \
  -H "X-KB-ID: kb_b_id" \
  -d '{"query": "test query"}'
```

**Expected:** Different responses based on tenant data

---

## Security Audit Checklist

| Check | Status | Notes |
|-------|--------|-------|
| Tenant ID validated before use | ⬜ | Need to verify |
| KB ID validated before use | ⬜ | Need to verify |
| User access verified for tenant | ⬜ | Has admin bypass |
| SQL injection prevented | ⬜ | Using parameterized queries |
| Path traversal prevented | ⬜ | Has validation functions |
| Cross-tenant data access blocked | ⬜ | Need to test |
| Rate limiting per tenant | ⬜ | Not observed |

---

## Conclusion

The REST API has a functional multi-tenant implementation with proper:
- Header extraction and validation
- Tenant-specific RAG instance caching
- Security validation for identifiers

Key concerns:
1. **Critical:** Optional tenant context allows global RAG fallback
2. **High:** Admin user bypass in access control
3. **Medium:** No user_id requirement in RAG manager
4. **Low:** Default KB fallback behavior

Recommendations:
1. Make tenant context required for multi-tenant mode
2. Remove or properly secure admin bypass
3. Add strict mode configuration flag
4. Document expected deployment modes
