
# Migration & deployment guide — required steps (concise)

This document lists the EXACT deployment steps to move from upstream/main (HKUDS) to this multi-tenant version.

Prerequisites

- Back up database and take a snapshot. (Always.)

- Ensure staging environment that mirrors production with Postgres + identical config.

Database migrations (required)

1) Create the following core objects/tables if not present:

   - tenants (tenant_id PK, name, description, metadata jsonb, created_at, updated_at)

   - knowledge_bases (kb_id, tenant_id FK, name, description, created_at, updated_at)

   - user_tenant_memberships (id, user_id, tenant_id, role, created_at, created_by, updated_at)

2) Create function has_tenant_access(user_id text, tenant_id text, required_role text) RETURNS boolean. (This function is used by TenantService.verify_user_access). Implement role hierarchy checks and super-admin bypass.

3) Install RLS policies from lightrag/kg/postgres_rls.sql and confirm session variable set_config('app.current_tenant', tenant_id, false) is used by application before running queries.

Application config

- Required environment variables (ensure no defaults used in production):

  - TOKEN_SECRET

  - LIGHTRAG_MULTI_TENANT_STRICT (recommended true)

  - LIGHTRAG_REQUIRE_USER_AUTH (recommended true)

  - LIGHTRAG_SUPER_ADMIN_USERS (explicit comma separated admins)

  - LIGHTRAG_API_KEY (if API key flows used)

Runtime wiring

- Ensure database client sets session var on every connection or at least at the start of each request. Example (asyncpg):

  connection.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)

- Middleware: TenantMiddleware will set request.state.tenant_id from token/subdomain — ensure it’s added to the FastAPI middleware list before routers.

- Inject TenantService and TenantRAGManager into app.state or DI container so dependencies.py can access rag_manager. The code expects request.app.state.rag_manager/tenant_service.

Testing (staging) — essential checks before rollout

1) RLS check: create two tenants, insert a doc under tenant A, set session var to tenant B, verify SELECT does not return tenant A rows.

2) Token vs subdomain: test mismatch handling (token tenant != subdomain tenant) — behavior: dependencies.py raises 400 Tenant ID mismatch.

3) Membership: test add/update/remove membership flows and verify has_tenant_access returns expected results.

4) Eviction: test TenantRAGManager cache eviction under concurrent load.

Rollback plan

- If any test fails either disable multi-tenant features (set LIGHTRAG_MULTI_TENANT_STRICT=false and run in single-tenant fallback) or restore DB snapshot and rollback application to previous tag.
