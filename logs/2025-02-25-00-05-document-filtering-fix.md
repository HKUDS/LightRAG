# Document Filtering by Tenant/KB Fix - Task Log

**Date**: 2025-02-25 00:05

## Summary

Fixed the document list not being properly filtered when changing the Knowledge Base (KB) selection. Documents were showing stale data from the previous KB.

## Problem

When changing the KB selection:
- The "0 docs" indicator in the header was correct (from KB metadata)
- But the document list still showed documents from the previous KB
- This created a confusing UX where the count didn't match the displayed list

## Root Cause

The `DocumentManager.tsx` component was missing a useEffect to clear the document state when tenant/KB changed. While there was an effect to trigger a data fetch when KB changed, the old document list remained visible until the new data arrived (which might be empty for a different KB).

## Solution

Added a new useEffect in `lightrag_webui/src/features/DocumentManager.tsx` that:

1. **Clears all document state immediately** when tenant/KB changes:
   - `setCurrentPageDocs([])` - Clear displayed documents
   - `setDocs(null)` - Clear legacy docs state
   - `setStatusCounts({ all: 0 })` - Reset status counts
   - `setPagination(...)` - Reset pagination to page 1
   - `setSelectedDocIds([])` - Clear selection
   - `setPageByStatus(...)` - Reset page memory for all status filters

2. **Added logging** for debugging tenant/KB changes

3. **Updated central fetch effect** to also include `selectedTenant?.tenant_id` in dependencies

## Changes Made

### File: `lightrag_webui/src/features/DocumentManager.tsx`

Added new useEffect (lines 1062-1080):
```tsx
// Reset document state when tenant or KB changes - clear old data immediately
useEffect(() => {
  // Clear current documents to prevent showing stale data
  setCurrentPageDocs([]);
  setDocs(null);
  setStatusCounts({ all: 0 });
  setPagination(prev => ({
    ...prev,
    page: 1,
    total_count: 0,
    total_pages: 0,
    has_next: false,
    has_prev: false
  }));
  setSelectedDocIds([]);
  // Reset page memory for all status filters
  setPageByStatus({ all: 1, processed: 1, processing: 1, pending: 1, failed: 1 });
  console.log('[DocumentManager] Reset document state due to tenant/KB change:', {
    tenant_id: selectedTenant?.tenant_id,
    kb_id: selectedKB?.kb_id
  });
}, [selectedTenant?.tenant_id, selectedKB?.kb_id]);
```

Updated central fetch effect to include tenant dependency:
```tsx
useEffect(() => {
  if (currentTab === 'documents' && selectedTenant && selectedKB) {
    fetchPaginatedDocuments(pagination.page, pagination.page_size, statusFilter);
  }
}, [
  currentTab,
  pagination.page,
  pagination.page_size,
  statusFilter,
  sortField,
  sortDirection,
  fetchPaginatedDocuments,
  selectedTenant?.tenant_id, // NEW: Trigger fetch when tenant changes
  selectedKB?.kb_id
]);
```

## Expected Behavior After Fix

When user changes KB:
1. Document list is immediately cleared (shows empty/loading)
2. New fetch is triggered with correct tenant/KB headers
3. API returns documents for the selected KB only
4. Document list updates with fresh data (or shows empty if no docs in that KB)

## Verification

- ✅ TypeScript compilation successful
- ✅ Build completed in 3.70s
- ✅ Console logging added for debugging
- ✅ Pagination reset to page 1 on change
- ✅ Page memory reset for all status filters

## Related Changes

This fix complements the earlier queryTextStream fix which added tenant/KB headers to streaming queries. Together, they ensure:
- Documents are filtered by tenant/KB
- Queries are scoped to tenant/KB
- State is properly cleared when context changes

## Task Logs

**Actions:**
- Added useEffect to clear document state on tenant/KB change
- Added selectedTenant?.tenant_id to central fetch effect deps
- Added condition to check selectedTenant && selectedKB before fetching

**Decisions:**
- Clear all document-related state immediately (not just currentPageDocs)
- Reset pagination to page 1 when context changes
- Reset page memory for all status filters to avoid stale page references

**Next steps:**
- Test in browser to verify document list clears and refetches correctly
- Verify console logs show correct tenant/KB IDs
- Test switching between KBs with different document counts

**Lessons/insights:**
- Stale data can persist in React state even when dependency triggers a refetch
- Always clear state before fetching new data when context changes completely
- Console logging helps debug multi-tenant context propagation
