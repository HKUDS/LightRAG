# Task Log: Multi-Tenant UX Improvements

**Date:** 2025-12-04 10:51
**Mode:** beastmode

## Summary

Fixed race condition issues in multi-tenant DocumentManager that caused false error toasts and improved UX during tenant/KB transitions. Also fixed Knowledge Graph not refreshing when KB changes.

## Actions

### Session 1 (Initial Fix)
1. Added `isTenantContextReady()` guard to `handleManualRefresh` callback
2. Added `classifyError()` usage in `handleManualRefresh` error handling
3. Introduced `hasLoadedOnce` state to track initial load completion
4. Split empty state rendering into "loading" and "truly empty" cases
5. Added `Loader2` icon import from lucide-react
6. Added "loading" translation key to 5 locale files (en, zh, zh_TW, fr, ar)

### Session 2 (Follow-up Fixes)
7. Fixed infinite loading spinner by setting `hasLoadedOnce=true` on non-context errors
8. Replaced simple 50ms timeout with retry mechanism (5 attempts, 30ms interval) for localStorage sync checking
9. Removed `isTenantContextReady` from effect dependencies to avoid stale closures
10. Fixed Knowledge Graph not refreshing when KB changes by:
    - Calling `state.reset()` immediately on KB change
    - Resetting all fetch refs (`fetchInProgressRef`, `dataLoadedRef`, etc.)
    - Resetting `emptyDataHandledRef` to allow empty graph handling

## Decisions

- Used retry mechanism instead of single timeout for more robust localStorage sync checking
- Set `hasLoadedOnce=true` on errors (except context-missing) to prevent infinite spinner
- Clear Knowledge Graph immediately on KB change rather than waiting for API response
- Reset all graph refs on KB change to ensure clean state for new fetch

## Files Modified

### DocumentManager.tsx
- Added `hasLoadedOnce=true` in error catch block for non-context errors
- Replaced 50ms timeout with retry mechanism (5 attempts, 30ms each)
- Removed `isTenantContextReady` from dependency array
- Added max attempts fallback to show empty state instead of infinite spinner

### useLightragGraph.tsx
- Enhanced KB change effect to fully reset graph state
- Call `state.reset()` to clear current graph immediately
- Reset all refs: `dataLoadedRef`, `initialLoadRef`, `fetchInProgressRef`, `emptyDataHandledRef`
- Added logging for debugging

### Locale files (en.json, zh.json, zh_TW.json, fr.json, ar.json)
- Added "loading" translation key

## Next Steps

1. Test the fixes by:
   - Selecting a tenant and verifying loading spinner stops
   - Switching KB and verifying Knowledge Graph refreshes
   - Checking no error toasts appear during transitions

## Lessons/Insights

- Closures in React effects can capture stale state when using memoized callbacks
- Simple timeouts are not reliable for async state synchronization - use retries
- When changing context (tenant/KB), must reset all related refs and state to prevent stale data
- Setting `hasLoadedOnce` on error prevents infinite loading states
