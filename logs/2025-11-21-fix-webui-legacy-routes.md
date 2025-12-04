# Task Log: Fix WebUI Default Tenant Access

## Issue

User reported that the WebUI documents list becomes empty after navigating to the Knowledge Graph and back.
Logs showed HTTP 404 errors for `GET /api/v1/tenants/default/knowledge-bases` and HTTP 405 for `GET /api/v1/tenants`.

## Root Cause

The WebUI (running in a container) uses legacy API routes that were removed during the recent API refactoring for multi-tenancy.

- `GET /api/v1/tenants` was missing (only `POST` existed).
- `GET /api/v1/tenants/{tenant_id}/knowledge-bases` was missing (replaced by `/api/v1/knowledge-bases` with headers).

## Fix

1. **Backend Routes**: Added legacy support routes to `lightrag/api/routers/tenant_routes.py`:
   - `GET /tenants`: Lists tenants (aliases to `tenant_service.list_tenants`).
   - `GET /tenants/{tenant_id}/knowledge-bases`: Lists KBs for a specific tenant path (uses `resolve_default_tenant` to handle "default").

2. **Docker Configuration**: Updated `docker-compose.yml` to mount the local `lightrag` source code into the container (`./lightrag:/app/lightrag`). This ensures that code changes are reflected in the running service without rebuilding the image.

3. **Verification**:
   - Updated `test_multitenant.sh` with Test 9 to verify the legacy route.
   - Ran tests: All passed, including the new legacy route test.

## Result

The backend now supports the API calls made by the WebUI. The "default" tenant alias is correctly resolved, and the Knowledge Base list is returned successfully. This prevents the WebUI from clearing the selected KB/Tenant state, resolving the empty documents issue.
