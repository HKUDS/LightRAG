# Task Log: Multi-Tenant Document Routes Fix

**Date:** 2025-01-25 17:00
**Mode:** beastmode

## Problem
Document upload, scan, and other document operations were using the **global** RAG instance instead of the **tenant-specific** RAG instance. This caused:
1. Documents uploaded in one tenant's KB being stored in the global namespace
2. Pipeline status being read from global namespace instead of tenant-specific
3. Document count in header showing wrong number ("0 docs")

## Root Cause
In `lightrag/api/routers/document_routes.py`, many endpoints were using the closure-captured `rag` variable (global) instead of the `get_tenant_rag` dependency that provides tenant-specific RAG instances.

## Actions
Fixed the following endpoints to use `tenant_rag: LightRAG = Depends(get_tenant_rag)`:

1. **POST /documents/upload** - File upload now uses tenant-specific RAG
2. **POST /documents/scan** - Directory scan now uses tenant-specific RAG
3. **POST /documents/text** - Text insertion now uses tenant-specific RAG
4. **POST /documents/texts** - Batch text insertion now uses tenant-specific RAG
5. **DELETE /documents** (clear_documents) - Clear all now uses tenant-specific RAG
6. **DELETE /documents/delete** - Delete by ID now uses tenant-specific RAG
7. **POST /documents/clear_cache** - Cache clearing now uses tenant-specific RAG
8. **DELETE /documents/delete_entity** - Entity deletion now uses tenant-specific RAG
9. **DELETE /documents/delete_relation** - Relation deletion now uses tenant-specific RAG
10. **POST /documents/reprocess_failed** - Reprocess now uses tenant-specific RAG
11. **POST /documents/cancel_pipeline** - Pipeline cancel now uses tenant-specific RAG

## Decisions
- Used existing `get_tenant_rag` dependency pattern for consistency
- All document operations now properly scope to tenant/KB context from headers

## Next Steps
1. Restart backend server to apply changes
2. Test document upload in multi-tenant mode
3. Verify pipeline processes documents in correct tenant namespace
4. Verify document count in header updates correctly

## Lessons/Insights
- Multi-tenant systems require careful auditing of ALL endpoints to ensure proper tenant isolation
- Using dependency injection patterns (like `Depends(get_tenant_rag)`) makes it cleaner to manage tenant context
- Background tasks must receive the correct RAG instance at task creation time
