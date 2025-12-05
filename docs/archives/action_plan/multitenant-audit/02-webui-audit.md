# Web UI Multi-Tenant Audit

**Date:** November 29, 2025
**Status:** In Progress

---

## Overview

This document audits the multi-tenant implementation in the LightRAG Web UI (React/TypeScript frontend).

## Components Under Audit

### 1. Tenant State Store (`stores/tenant.ts`)

**Purpose:** Zustand-based state management for tenant and KB selection.

**Analysis:**

```typescript
// Key State Properties
interface TenantState {
  selectedTenant: Tenant | null;
  selectedKB: KnowledgeBase | null;
  tenants: Tenant[];
  knowledgeBases: KnowledgeBase[];
}
```

**✅ Strengths:**
- State is persisted to localStorage for page refresh resilience
- State is initialized on module load (not lazy)
- Clear separation between tenant and KB selection

**⚠️ Potential Issues:**
- localStorage is domain-scoped, not tenant-scoped
- If two users access same browser, localStorage could leak tenant context
- No encryption of stored tenant data

**🔍 Test Point:**
- Verify localStorage is cleared on logout
- Check if tenant IDs are validated before use

### 2. API Client (`api/client.ts`)

**Purpose:** Axios instance with interceptors for adding tenant headers.

**Analysis:**

```typescript
// Header injection in interceptor
if (selectedTenant?.tenant_id) {
  config.headers['X-Tenant-ID'] = selectedTenant.tenant_id;
}
if (selectedKB?.kb_id) {
  config.headers['X-KB-ID'] = selectedKB.kb_id;
}
```

**✅ Strengths:**
- Headers automatically added to ALL requests
- Console logging enabled for debugging
- Reads directly from localStorage to avoid circular dependencies

**⚠️ Potential Issues:**
- If localStorage is empty/corrupted, requests proceed without tenant headers
- No validation that tenant_id/kb_id are valid UUIDs
- Logging may expose sensitive IDs in production

**🔍 Test Points:**
1. What happens if localStorage has invalid JSON?
2. What happens if tenant_id is malformed?
3. Are headers properly propagated for streaming requests?

### 3. Tenant API Functions (`api/tenant.ts`)

**Purpose:** API functions for tenant/KB CRUD operations.

**Analysis:**

**✅ Strengths:**
- Paginated API calls for efficiency
- Fallback to default tenant if API fails
- Headers explicitly added for tenant-scoped calls

**⚠️ Potential Issues:**
- Fallback to "default" tenant could hide errors
- No retry logic for failed API calls
- Error handling may swallow important context

### 4. Document Manager (`features/DocumentManager.tsx`)

**Purpose:** Component for managing documents within a tenant/KB.

**Analysis from recent fix (2025-02-25):**

```typescript
// Reset document state when tenant or KB changes
useEffect(() => {
  setCurrentPageDocs([]);
  setDocs(null);
  setStatusCounts({ all: 0 });
  setPagination(prev => ({...prev, page: 1, total_count: 0, ...}));
  setSelectedDocIds([]);
  setPageByStatus({ all: 1, processed: 1, processing: 1, pending: 1, failed: 1 });
}, [selectedTenant?.tenant_id, selectedKB?.kb_id]);
```

**✅ Strengths:**
- State is cleared immediately on tenant/KB change
- Prevents showing stale data from previous context
- Logging added for debugging

**⚠️ Potential Issues:**
- Brief moment where old data could be visible during transition
- No loading indicator during context switch
- Dependency array may not catch all cases

### 5. Query/Chat Panel

**Purpose:** Component for running queries against the knowledge base.

**🔍 Test Points:**
1. Are queries properly scoped to selected tenant/KB?
2. Does streaming work correctly with tenant headers?
3. Are conversation histories properly isolated?

---

## Detailed Findings

### Finding WUI-001: localStorage Security Concern
**Severity:** Medium
**Location:** `stores/tenant.ts`, `api/client.ts`

**Description:**
Tenant context is stored in localStorage as plain JSON. This could:
- Be read by any JavaScript on the same domain
- Persist after logout if not properly cleared
- Be shared between browser tabs/windows unintentionally

**Recommendation:**
- Clear localStorage on logout
- Consider sessionStorage for tenant context
- Validate tenant context on each API call

### Finding WUI-002: Fallback to Default Tenant
**Severity:** Low
**Location:** `api/tenant.ts`

**Description:**
When API fails, the code returns a default tenant:
```typescript
return {
  items: [{
    tenant_id: 'default',
    tenant_name: 'Default Tenant',
    ...
  }],
  ...
}
```

This could mask API errors and lead to unexpected behavior.

**Recommendation:**
- Distinguish between API errors and empty results
- Show error state to user instead of fallback
- Log API failures for debugging

### Finding WUI-003: Missing Header Validation
**Severity:** Low
**Location:** `api/client.ts`

**Description:**
The interceptor logs warnings but doesn't prevent requests without tenant context:
```typescript
if (!selectedTenantJson) {
  console.warn('[Axios Interceptor] No SELECTED_TENANT in localStorage');
}
// Request still proceeds
```

**Recommendation:**
- For tenant-required endpoints, block request if no context
- Add middleware to validate tenant context presence

---

## Test Scenarios

### Scenario WUI-T1: Tenant Selection Persistence
1. Select Tenant A
2. Refresh page
3. Verify Tenant A is still selected
4. Check localStorage for correct data

**Expected:** Tenant A persisted and restored correctly

### Scenario WUI-T2: KB Selection Isolation
1. Select Tenant A, KB Alpha
2. Verify document list shows only KB Alpha docs
3. Switch to KB Beta
4. Verify document list clears and shows KB Beta docs

**Expected:** Documents properly filtered by KB

### Scenario WUI-T3: Cross-Tenant Query Isolation
1. Select Tenant A, add document about "apples"
2. Query "what do you know about apples?"
3. Switch to Tenant B
4. Query same question
5. Verify response is empty/different

**Expected:** Query results isolated to selected tenant

### Scenario WUI-T4: Header Propagation
1. Open browser DevTools Network tab
2. Select Tenant A, KB Alpha
3. Trigger document upload
4. Verify request headers include:
   - `X-Tenant-ID: <tenant_a_id>`
   - `X-KB-ID: <kb_alpha_id>`

**Expected:** Headers correctly set on all requests

---

## Conclusion

The Web UI implementation has a solid foundation for multi-tenant support with:
- State management via Zustand
- Automatic header injection via Axios interceptors
- State clearing on context change

Key areas for improvement:
1. localStorage security (session vs persistent storage)
2. Error handling for API failures
3. Validation of tenant context before API calls
4. Loading states during context transitions

Next step: Verify these findings through manual testing.
