# Task Log: Fix Pipeline Status Multi-Tenant Mismatch

**Date:** 2025-01-25 16:45
**Mode:** beastmode

## Problem
The Documents panel was stuck showing "Processing in Progress" even when the tenant-specific pipeline was not busy. This happened because:
1. The global `/health` endpoint returns `pipeline_busy` from global namespace (not tenant-aware)
2. The `/documents/pipeline_status` endpoint returns tenant-specific pipeline status
3. The UI used the global `pipelineBusy` state from health endpoint to decide which empty state to show

## Actions
1. Added `getPipelineStatus` import to DocumentManager.tsx
2. Added `setPipelineBusy` hook to update global pipeline state
3. Modified `handleIntelligentRefresh` to fetch tenant-specific pipeline status after documents load
4. Updated dependency array to include `setPipelineBusy` and `isTenantContextReady`

## Decisions
- Chose to update global `pipelineBusy` state from DocumentManager rather than modifying the health endpoint
- This approach keeps the health endpoint simple and makes the Documents panel the source of truth for tenant-specific pipeline state

## Next Steps
- Refresh browser and verify Documents panel shows correct empty state
- When switching tenants/KBs, the pipeline status should update correctly

## Lessons/Insights
- In multi-tenant mode, global status endpoints (like /health) don't reflect tenant-specific state
- Components that need tenant-specific data should fetch it directly from tenant-aware endpoints
