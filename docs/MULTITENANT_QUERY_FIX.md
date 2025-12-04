# Multi-Tenant Query Context Fix

## Problem
The Retrieval/Query page was not respecting the selected tenant and knowledge base context. While documents were properly isolated by tenant on the backend, the query interface was not sending tenant context headers for streaming queries.

## Root Cause
The `queryTextStream` function in `lightrag_webui/src/api/lightrag.ts` was using the raw `fetch()` API instead of `axiosInstance`, which meant:
1. It wasn't benefiting from the axios interceptor that adds `X-Tenant-ID` and `X-KB-ID` headers
2. It was manually constructing headers but missing the tenant/KB context
3. This caused queries to default to the global RAG instance instead of the tenant-specific one

## Solution

### Frontend Fix: `lightrag_webui/src/api/lightrag.ts`

Updated the `queryTextStream` function to read and include tenant context from localStorage:

```typescript
export const queryTextStream = async (
  request: QueryRequest,
  onChunk: (chunk: string) => void,
  onError?: (error: string) => void
) => {
  const apiKey = useSettingsStore.getState().apiKey;
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN');
  
  // Get tenant context from localStorage
  const selectedTenantJson = localStorage.getItem('SELECTED_TENANT');
  const selectedKBJson = localStorage.getItem('SELECTED_KB');
  
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    'Accept': 'application/x-ndjson',
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  
  // Add tenant context headers
  if (selectedTenantJson) {
    try {
      const selectedTenant = JSON.parse(selectedTenantJson);
      if (selectedTenant?.tenant_id) {
        headers['X-Tenant-ID'] = selectedTenant.tenant_id;
      }
    } catch (e) {
      console.error('[queryTextStream] Failed to parse selected tenant:', e);
    }
  }
  
  if (selectedKBJson) {
    try {
      const selectedKB = JSON.parse(selectedKBJson);
      if (selectedKB?.kb_id) {
        headers['X-KB-ID'] = selectedKB.kb_id;
      }
    } catch (e) {
      console.error('[queryTextStream] Failed to parse selected KB:', e);
    }
  }

  try {
    const response = await fetch(`${backendBaseUrl}/query/stream`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(request),
    });
    
    // ... rest of the function
  }
};
```

### Key Changes:
1. Added `selectedTenantJson` and `selectedKBJson` extraction from localStorage
2. Parsed JSON safely with try-catch error handling
3. Added `X-Tenant-ID` and `X-KB-ID` headers to the fetch request if tenant/KB context is available
4. Logs errors but doesn't fail - allows queries to proceed even if headers are missing

## Backend Verification

The backend was already correctly configured:

### `lightrag/api/dependencies.py`
- `get_tenant_context_optional`: Extracts tenant context from `X-Tenant-ID` and `X-KB-ID` headers
- Propagates authentication errors instead of silently falling back to global RAG

### `lightrag/api/routers/query_routes.py`
All query endpoints use the `get_tenant_rag` dependency:

```python
async def get_tenant_rag(tenant_context: Optional[TenantContext] = Depends(get_tenant_context_optional)) -> LightRAG:
    """Dependency to get tenant-specific RAG instance for query operations"""
    if rag_manager and tenant_context and tenant_context.tenant_id and tenant_context.kb_id:
        return await rag_manager.get_rag_instance(
            tenant_context.tenant_id, 
            tenant_context.kb_id,
            tenant_context.user_id  # Pass user_id for security validation
        )
    return rag
```

This ensures:
- `/query` endpoint uses tenant-specific RAG
- `/query/stream` endpoint uses tenant-specific RAG  
- `/query/data` endpoint uses tenant-specific RAG

## How Tenant Context Flows

### For Axios-based calls (e.g., `queryText`):
1. Request is made via `axiosInstance.post('/query', request)`
2. Axios interceptor in `client.ts` automatically reads `SELECTED_TENANT` and `SELECTED_KB` from localStorage
3. Interceptor adds `X-Tenant-ID` and `X-KB-ID` headers
4. Backend receives headers and routes to tenant-specific RAG

### For Fetch-based calls (e.g., `queryTextStream`):
1. Request is made via `fetch()` to `/query/stream`
2. **NEW**: Function now reads `SELECTED_TENANT` and `SELECTED_KB` from localStorage
3. **NEW**: Function manually adds `X-Tenant-ID` and `X-KB-ID` headers to fetch request
4. Backend receives headers and routes to tenant-specific RAG

## Testing the Fix

### Prerequisites:
1. User must be authenticated and have an active tenant/KB selected
2. Documents must exist in the selected tenant/KB

### Test Steps:
1. Select a tenant from the tenant dropdown
2. Select a knowledge base from the KB dropdown
3. Navigate to the Retrieval tab
4. Enter a query that should match documents in the selected KB
5. Verify the query returns results from that tenant/KB only

### Expected Behavior:
- Query respects the selected tenant/KB context
- Results are filtered to the selected tenant/KB
- No data from other tenants appears in results
- "No relevant context found" only appears if no matching documents in selected KB

## Multi-Tenant Architecture Summary

### Data Isolation Layers:
1. **HTTP Headers**: `X-Tenant-ID` and `X-KB-ID` sent by frontend
2. **Dependency Injection**: `get_tenant_context_optional` extracts and validates headers
3. **RAG Instance Selection**: `get_tenant_rag` returns tenant-specific RAG instance
4. **Query Execution**: Tenant-specific RAG only searches that tenant's knowledge graph

### Frontend Storage:
- `SELECTED_TENANT`: JSON object with `tenant_id` stored in localStorage
- `SELECTED_KB`: JSON object with `kb_id` stored in localStorage
- Updated whenever user changes tenant/KB selection

### Query Endpoints:
- **`/query`**: Non-streaming query, uses axios (automatically includes headers)
- **`/query/stream`**: Streaming query, uses fetch (now manually includes headers)
- **`/query/data`**: Structured data retrieval, uses axios (automatically includes headers)

## Files Modified

1. **`lightrag_webui/src/api/lightrag.ts`**
   - Modified `queryTextStream` function (lines 317-365+)
   - Added tenant context extraction from localStorage
   - Added `X-Tenant-ID` and `X-KB-ID` headers to fetch request

## Verification Checklist

- [x] Frontend code compiles without errors
- [x] Backend query endpoints properly use `get_tenant_rag` dependency
- [x] Tenant context is extracted from `X-Tenant-ID` header
- [x] KB context is extracted from `X-KB-ID` header
- [x] `queryTextStream` now includes tenant/KB headers
- [x] Error handling includes proper logging
- [x] Build succeeds with no TypeScript errors

## Backward Compatibility

This fix maintains backward compatibility:
- Non-authenticated requests (no tenant headers) still work with global RAG
- Existing API clients that don't send headers still function
- Multi-tenant isolation is optional based on header presence
- No breaking changes to API contracts

## Performance Impact

- Minimal: Only adds header reading from localStorage during query
- localStorage.getItem() and JSON.parse() are fast operations
- Negligible impact on query latency

## Security Implications

✅ **Improved Security**
- Query operations now respect tenant isolation
- Headers are validated on backend
- Each query is scoped to selected tenant/KB
- Prevents accidental cross-tenant data leakage

## Related Documentation

See also:
- `docs/0001-multi-tenant-architecture.md` - Overall multi-tenant design
- `docs/0002-multi-tenant-visual-reference.md` - Visual architecture guide
- `lightrag/api/dependencies.py` - Tenant context injection
- `lightrag/tenant_rag_manager.py` - Tenant RAG instance management
