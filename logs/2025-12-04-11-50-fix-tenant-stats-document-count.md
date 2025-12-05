# Session Log: Fix Tenant Statistics - Document Count Issue

## Date: 2025-12-04

## Summary
Fixed the tenant card statistics to correctly display document count and storage usage. The issue was that documents were stored in `lightrag_doc_full` table (with workspace column) instead of the `documents` table.

## Problem
The tenant selection cards were showing "0 Docs" for TechStart tenant even though there was 1 document in the Main KB. Additionally, storage_used_gb was always 0.

## Root Cause Analysis
The SQL query in `list_tenants` was looking for documents in the `documents` table, which was empty. The actual documents are stored in the `lightrag_doc_full` table with a `workspace` column in format `{tenant_id}:{kb_id}`.

Database investigation showed:
- `documents` table: Empty (0 rows)
- `lightrag_doc_full` table: Contains actual documents with workspace field
- Example: workspace = `techstart:kb-main` for TechStart tenant's Main KB

## Solution
Updated the SQL query to:
1. Count documents from `lightrag_doc_full` table
2. Extract tenant_id from the workspace column using `SPLIT_PART(workspace, ':', 1)`
3. Calculate storage from content length instead of file_size
4. Join this data with tenant and KB statistics

### Updated SQL Query
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
    SELECT
        SPLIT_PART(workspace, ':', 1) as tenant_id,
        COUNT(*) as doc_count,
        COALESCE(SUM(LENGTH(content)), 0) as total_size_bytes
    FROM lightrag_doc_full
    GROUP BY SPLIT_PART(workspace, ':', 1)
) doc_stats ON t.tenant_id = doc_stats.tenant_id
ORDER BY t.created_at DESC
```

## Results
Before Fix:
```json
{
    "tenant_id": "techstart",
    "num_knowledge_bases": 2,
    "num_documents": 0,
    "storage_used_gb": 0.0
}
```

After Fix:
```json
{
    "tenant_id": "techstart",
    "num_knowledge_bases": 2,
    "num_documents": 1,
    "storage_used_gb": 0.00010807719081640244
}
```

## Verification
✅ Tenant cards now show correct document count:
- TechStart Inc: "1 Doc" (was "0 Docs")
- Acme Corporation: "0 Docs" (correct, no documents)

✅ Storage usage now calculated from actual content:
- TechStart Inc: 0.00010807719081640244 GB (was 0.0)

✅ KB count remains correct:
- Both tenants: "2 KBs"

## Code Changes
**File: `/lightrag/services/tenant_service.py` (lines 612-644)**
- Updated SQL query to use `lightrag_doc_full` table
- Used `SPLIT_PART(workspace, ':', 1)` to extract tenant_id
- Changed storage calculation to use `SUM(LENGTH(content))`

## Files Modified
1. `/lightrag/services/tenant_service.py` - Updated list_tenants SQL query

## Testing
- ✅ Restarted backend server
- ✅ Verified API returns correct stats via curl
- ✅ Verified tenant selection cards display correct values
- ✅ Document count matches actual data in database

---

## Task Logs

### Actions:
- Investigated database schema to find actual document storage location
- Updated SQL query to count from lightrag_doc_full instead of documents table
- Extracted tenant_id from workspace column using SPLIT_PART
- Restarted backend and verified stats are now correct

### Decisions:
- Used LENGTH(content) for storage calculation since file_size was not available
- Used SPLIT_PART to parse workspace field rather than a JOIN on a computed column

### Insights:
- LightRAG uses two document storage modes: workspace mode (lightrag_doc_full) and tenant mode (documents)
- The system was using workspace mode which stores workspace as {tenant_id}:{kb_id}
- Statistics need to aggregate data across both storage modes or correctly identify which one is in use
