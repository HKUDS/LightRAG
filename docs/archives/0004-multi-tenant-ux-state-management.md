# Multi-Tenant UX & State Management

This document describes the multi-tenant state management architecture implemented in LightRAG, covering tenant switching, URL handling, state persistence, and security considerations.

## Overview

LightRAG implements a header-based multi-tenant architecture where:
- **Tenant context** is provided via `X-Tenant-ID` and `X-KB-ID` HTTP headers
- **URLs are tenant-agnostic** - they contain only UI state (page, filters, sort)
- **State is persisted** per-tenant in sessionStorage for quick restores
- **Security** is enforced server-side with token validation

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (WebUI)                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ TenantStateManager │←→│  sessionStorage  │    │    URL       │ │
│  │                   │    │ (tenant-scoped)  │    │ (no tenant)  │ │
│  └────────┬──────────┘    └─────────────────┘    └──────────────┘ │
│           │                                                       │
│  ┌────────▼──────────┐                                           │
│  │  Axios Interceptor │ ──── Adds X-Tenant-ID / X-KB-ID headers  │
│  └────────┬──────────┘                                           │
└───────────┼─────────────────────────────────────────────────────┘
            │
            ▼ HTTP Requests with headers
┌───────────────────────────────────────────────────────────────────┐
│                         Backend (API)                              │
├───────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐                                          │
│  │  dependencies.py     │ ─── Extracts & validates tenant context  │
│  │  get_tenant_context  │                                          │
│  └──────────┬──────────┘                                          │
│             │                                                      │
│  ┌──────────▼──────────┐    ┌─────────────────┐                   │
│  │  TenantRAGManager    │───→│ Tenant-scoped   │                   │
│  │                      │    │ LightRAG inst.  │                   │
│  └──────────────────────┘    └─────────────────┘                   │
└───────────────────────────────────────────────────────────────────┘
```

## Frontend State Management

### TenantStateManager

The `tenantStateManager` is a centralized module for managing tenant+route state:

```typescript
import { tenantStateManager } from '@/services/tenantStateManager'

// Get state for current tenant and route
const state = tenantStateManager.getState(tenantId, 'documents')

// Update state (persists to sessionStorage)
tenantStateManager.setState(tenantId, 'documents', { page: 5 })

// Sync to URL (debounced, tenant-agnostic)
tenantStateManager.syncToURL('documents', state)

// Handle tenant switch
tenantStateManager.onTenantSwitch(oldTenantId, newTenantId)
```

### State Storage Strategy

| Priority | Storage | Purpose |
|----------|---------|---------|
| Primary | URL query params | Route-level UI settings (page, filters, sort) |
| Secondary | sessionStorage | Per-tenant state for quick restores |
| Tertiary | In-memory | Fast runtime access |

**Key Format for sessionStorage:**
```
lightrag:tenant:<tenantId>:route:<routeName>
```

### useRouteState Hook

React hook for easy integration:

```typescript
function DocumentManager() {
  const {
    page,
    pageSize,
    sort,
    sortDirection,
    filters,
    setPage,
    setFilters,
    resetState,
  } = useRouteState('documents')

  // State changes automatically sync to URL and sessionStorage
}
```

## URL Format

URLs are **tenant-agnostic** for security. Examples:

```
/documents?kb=backup&page=3&pageSize=25&filters=status:active
/graph?kb=master&view=graph&filters=entityType:company
/retrieval?q=search+query
```

**Security Note:** Tenant identifiers are NEVER included in URLs. Tenant context comes from:
1. `X-Tenant-ID` header (required)
2. `X-KB-ID` header (optional, defaults to first KB)
3. Authorization token claims (for validation)

## Backend Tenant Resolution

The backend resolves tenant context in `dependencies.py`:

```python
async def get_tenant_context(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_kb_id: Optional[str] = Header(None, alias="X-KB-ID"),
) -> TenantContext:
    """
    Priority for tenant_id resolution:
    1. Middleware state (subdomain/JWT extracted early)
    2. Token metadata
    3. X-Tenant-ID header (fallback)
    """
```

## Ingestion Idempotency

The ingestion API supports idempotency via `external_id`:

```python
# Request
POST /documents/text
X-Tenant-ID: tenant-123
X-KB-ID: kb-456

{
  "text": "Document content...",
  "external_id": "my-unique-doc-id"
}

# Response (first time)
{"status": "success", "track_id": "insert_xxx"}

# Response (same external_id again)
{"status": "duplicated", "message": "Document with external_id 'my-unique-doc-id' already exists"}
```

## Database Indexes

For optimal performance, the following indexes are created:

```sql
-- Pagination indexes
CREATE INDEX idx_doc_status_workspace_status_updated_at
  ON LIGHTRAG_DOC_STATUS (workspace, status, updated_at DESC);

CREATE INDEX idx_doc_status_workspace_status_created_at
  ON LIGHTRAG_DOC_STATUS (workspace, status, created_at DESC);

-- Idempotency index
CREATE INDEX idx_doc_status_workspace_external_id
  ON LIGHTRAG_DOC_STATUS (workspace, (metadata->>'external_id'))
  WHERE metadata->>'external_id' IS NOT NULL;
```

## Security Considerations

1. **Never expose tenant IDs in URLs** - Use headers only
2. **Server-side validation** - Always validate tenant context from token
3. **Tenant isolation** - Each tenant's data is stored with workspace prefix
4. **Strict mode** - Set `LIGHTRAG_MULTI_TENANT_STRICT=true` to require tenant context

## Testing

### Unit Tests
```bash
# Frontend tests
cd lightrag_webui
npm run test -- src/__tests__/tenantStateManager.test.ts

# Backend tests
pytest tests/test_idempotency.py -v
```

### E2E Tests
```bash
# Requires running server
RUN_E2E_TESTS=1 pytest tests/e2e_multi_tenant_state.py -v
```

## Rollout Checklist

- [ ] Deploy backend with new indexes
- [ ] Enable `LIGHTRAG_MULTI_TENANT_STRICT` in staging
- [ ] Run e2e tests against staging
- [ ] Deploy frontend with tenantStateManager
- [ ] Monitor per-tenant request latency
- [ ] Monitor ingestion failure rates

## Related Documentation

- [0001-multi-tenant-architecture.md](archives/0001-multi-tenant-architecture.md) - Core architecture
- [0002-multi-tenant-visual-reference.md](archives/0002-multi-tenant-visual-reference.md) - Visual diagrams
- [LOCAL_DEVELOPMENT.md](archives/LOCAL_DEVELOPMENT.md) - Local testing setup
