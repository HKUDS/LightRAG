# Session Log: Fix Retrieval State Clearing and Tenant Card Statistics

## Date: 2025-12-04

## Summary
Fixed two issues:
1. Retrieval history not being cleared when tenant or KB changes
2. Tenant selection cards showing "0" for KBs, Docs, and GB instead of real figures

## Problem 1: Retrieval State Not Clearing

### Issue
When switching between tenants or knowledge bases, the retrieval chat history persisted, potentially showing results from a different tenant/KB context.

### Root Cause
The `RetrievalTesting.tsx` component did not have an effect hook to clear the messages state when `selectedTenant` or `selectedKB` changed.

### Solution
Added a `useEffect` hook in `RetrievalTesting.tsx` that:
1. Gets both `selectedTenant` and `selectedKB` from `useTenantState`
2. Clears messages state and retrieval history when tenant or KB changes
3. Logs the clearing action for debugging

### Code Changes
**File: `/lightrag_webui/src/features/RetrievalTesting.tsx`**

Added import for `selectedTenant`:
```typescript
const selectedTenant = useTenantState.use.selectedTenant()
const selectedKB = useTenantState.use.selectedKB()
```

Added effect to clear history:
```typescript
// Clear retrieval history when tenant or KB changes to prevent showing stale data
useEffect(() => {
  setMessages([])
  useSettingsStore.getState().setRetrievalHistory([])
  console.log('[RetrievalTesting] Cleared retrieval history due to tenant/KB change:', {
    tenant_id: selectedTenant?.tenant_id,
    kb_id: selectedKB?.kb_id
  })
}, [selectedTenant?.tenant_id, selectedKB?.kb_id])
```

## Problem 2: Tenant Cards Showing Zero Statistics

### Issue
The tenant selection cards displayed "0 KBs", "0 Docs", "0 GB" even when tenants had knowledge bases and documents.

### Root Cause
The `list_tenants` method in `tenant_service.py` was querying the PostgreSQL database but creating `Tenant` objects with default statistics (0, 0, 0.0) instead of computing them from the `knowledge_bases` and `documents` tables.

### Solution
Updated the SQL query in `list_tenants` to use LEFT JOINs to compute:
- `kb_count`: Count of knowledge bases per tenant
- `total_documents`: Count of documents per tenant
- `total_size_bytes`: Sum of file sizes for storage calculation

### Code Changes
**File: `/lightrag/services/tenant_service.py`**

Updated SQL query:
```sql
SELECT
    t.tenant_id,
    t.name,
    t.description,
    t.created_at,
    t.updated_at,
    COALESCE(kb_stats.kb_count, 0) as kb_count,
    COALESCE(doc_stats.doc_count, 0) as total_documents,
    COALESCE(doc_stats.total_size_bytes, 0) as total_size_bytes
FROM tenants t
LEFT JOIN (
    SELECT tenant_id, COUNT(*) as kb_count
    FROM knowledge_bases
    GROUP BY tenant_id
) kb_stats ON t.tenant_id = kb_stats.tenant_id
LEFT JOIN (
    SELECT tenant_id, COUNT(*) as doc_count, COALESCE(SUM(file_size), 0) as total_size_bytes
    FROM documents
    GROUP BY tenant_id
) doc_stats ON t.tenant_id = doc_stats.tenant_id
ORDER BY t.created_at DESC
```

Updated Tenant object creation to include computed statistics:
```python
tenant = Tenant(
    tenant_id=row['tenant_id'],
    tenant_name=row['name'],
    description=row.get('description', ''),
    created_by=None,
    metadata={},
    kb_count=row.get('kb_count', 0) or 0,
    total_documents=row.get('total_documents', 0) or 0,
    total_storage_mb=total_storage_mb,
)
```

## Additional Fix: fetchTenants API Compatibility

### Issue
After the tenant service changes, the frontend `fetchTenants` function broke because it expected an array but received a paginated response object.

### Solution
Updated `fetchTenants` in `/lightrag_webui/src/api/tenant.ts` to handle both formats:
```typescript
export async function fetchTenants(): Promise<Tenant[]> {
  try {
    const response = await apiClient.get('/api/v1/tenants')
    const data = response.data
    if (Array.isArray(data)) {
      return data
    }
    // New paginated format returns { items: [...], total: N, ... }
    return data?.items || []
  } catch (error) {
    console.error('Failed to fetch tenants:', error)
    throw error
  }
}
```

## Verification
- ✅ Frontend build passes (TypeScript, Vite)
- ✅ Python syntax check passes
- ✅ Tenant cards now show correct KB count (e.g., "2 KBs" for both tenants)
- ✅ Retrieval history clears when tenant/KB changes (verified via console logs)
- ✅ Empty KB shows proper empty state (no stale retrieval data)

---

## Task Logs

### Actions:
- Added useEffect in RetrievalTesting.tsx to clear messages on tenant/KB change
- Updated SQL query in tenant_service.py to compute real statistics with LEFT JOINs
- Fixed fetchTenants API to handle paginated response format

### Decisions:
- Used LEFT JOINs for statistics to ensure tenants with no KBs or documents still appear
- Used COALESCE for null safety on aggregates

### Next Steps:
- Consider adding refresh button on tenant selection page to update stats
- Monitor for performance impact of JOIN queries on large deployments

### Lessons/Insights:
- When adding new API response formats, ensure backward compatibility in frontend consumers
- Tenant statistics should be computed dynamically from source tables rather than stored as cached values
