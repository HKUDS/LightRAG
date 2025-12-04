# Session Log: Fix Loading State Race Condition

## Date: 2025-12-04

## Summary
Fixed the infinite loading spinner issue when selecting an empty Knowledge Base in the DocumentManager component.

## Problem Description
When a user selected an empty Knowledge Base from the KB selector dropdown, the loading spinner would spin forever instead of showing the "No Documents Yet" empty state.

## Root Cause Analysis
**Race condition between localStorage and Zustand state updates:**

1. When `setSelectedKB` is called, it updates:
   - localStorage (synchronously)
   - Zustand store state (asynchronously via React batch updates)

2. The central data-fetching effect in DocumentManager was using Zustand state (`selectedKB`) in its dependency array and comparisons

3. When the effect triggered, it captured the OLD closure value of `selectedKB` (which was `undefined` or the previous KB)

4. The `isTenantContextReady()` check compared localStorage values against Zustand values:
   ```typescript
   if (parsedKB?.kb_id !== selectedKB?.kb_id) return false;  // FAILS!
   ```

5. This check failed because localStorage had the new KB but Zustand still had the old value

6. Result: The effect returned early, never fetched documents, spinner never stopped

## Solution
Made DocumentManager rely on localStorage as the authoritative source instead of comparing against Zustand state:

### 1. Fixed `isTenantContextReady()` function (lines 683-715)
- Changed from checking Zustand+localStorage match to only checking localStorage
- Now only verifies that valid tenant and KB exist in localStorage
- Removed dependency on Zustand state values

### 2. Fixed Central Effect (lines 1183-1237)
- Removed `selectedKB` from closure dependencies
- Now reads directly from localStorage instead of `selectedKB?.kb_id`
- Uses `getStoredTenantContext()` helper for localStorage access

### 3. Fixed Render Guard (lines 1253-1268)
- Added fallback to check localStorage when Zustand state is undefined
- Prevents flash of "Please select a KB" message during state sync

## Key Code Changes

### isTenantContextReady() - Before:
```typescript
if (parsedKB?.kb_id !== selectedKB?.kb_id) return false;
if (parsedTenant?.id !== selectedTenant?.id) return false;
```

### isTenantContextReady() - After:
```typescript
// Only check localStorage - don't compare against Zustand state
// The axios interceptor reads from localStorage anyway
if (!parsedTenant?.id || !parsedKB?.kb_id) return false;
return true;
```

## Verification
- Tested with Playwright MCP browser automation
- Empty KB "Backup KB" now shows "No Documents Yet" empty state
- KB "Main KB" with 1 document loads and displays correctly
- Switching between KBs works without infinite spinner
- TypeScript build passes with no errors
- Production build successful

## Files Modified
- `/lightrag_webui/src/features/DocumentManager.tsx`

## Lessons Learned
1. When using both localStorage and Zustand state, be aware of timing differences
2. Effects capture closure values at time of creation, not current values
3. For components that need sync with localStorage (like axios interceptors), read from localStorage directly rather than comparing against async state
4. Playwright MCP is excellent for debugging UX issues with real browser interactions

---

## Task Logs

### Actions:
- Fixed isTenantContextReady() to only validate localStorage
- Fixed central effect to read KB from localStorage
- Fixed render guard to check localStorage fallback
- Verified with Playwright testing
- Ran TypeScript and production builds

### Decisions:
- Chose localStorage as authority since axios interceptor reads from it
- Removed Zustand state comparison to avoid race condition

### Next Steps:
- Monitor for any edge cases with KB switching
- Consider adding localStorage change listener for more robust sync

### Lessons/Insights:
- Zustand updates are async; localStorage is sync; effects capture closures at trigger time
