Actions:
- Ran tests for tenant API routes and captured failures.
- Inspected router and dependencies (tenant_routes.py and dependencies.py).
- Fixed get_current_tenant to re-raise HTTPException so 404 isn't converted to 500.
- Updated tests to set mock service return values and add permissions to TenantContext where needed.

Decisions:
- Tests are relevant and should be kept (they exercise important multi-tenant behaviours).
- Failures stemmed from two issues: incorrect error handling in route and insufficient test setup for permission checks.

Next steps:
- Re-run whole test suite to find other issues (optional).
- Consider standardizing how tests override permission checks (e.g. helper fixture to set TenantContext.permissions).

Lessons/insights:
- Route code should re-raise HTTPException to preserve intended client-facing error codes.
- check_permission uses TenantContext.has_permission, so tests should set permissions in the context (not try to override check_permission by creating new function instances).

Files changed:
- lightrag/api/routers/tenant_routes.py: fix error handling in get_current_tenant
- tests/test_tenant_api_routes.py: added mock return values and explicit permission entries

Timestamp: 2025-12-04T12:00:00Z
