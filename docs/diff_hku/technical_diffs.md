# Technical diffs — concise file-level audit

Below are the most impactful changes and why they matter. Each entry has: file, what changed, immediate risk/impact and recommended next action.


- lightrag/api/dependencies.py (NEW)
  - Adds tenant extraction logic, token+API-key handling, default resolution and a set of dependencies: get_tenant_context, get_tenant_context_optional, get_tenant_context_no_kb, check_permission, get_admin_context.
  - Impact: central piece enabling tenant isolation. Critical path for auth and RBAC.
  - Note: heavy use of request.app.state.rag_manager and tenant_service — the presence and shape must match runtime wiring.
  - Action: add unit tests for each dependency (happy and failure paths), and document API headers (X-Tenant-ID, X-KB-ID, X-API-Key).

- lightrag/api/middleware/tenant.py (NEW)
  - Sets request.state.tenant_id early (subdomain or JWT reading). Non-blocking on invalid token (delegates to dependencies).
  - Impact: helpful early-set context; but missing domain allowlist and safe checks for public domain names; risks if subdomain parsing is naive.
  - Action: add config-driven allowed domains and test for IP/localhost cases.

- lightrag/models/tenant.py (NEW)
  - Domain models for Tenant, KnowledgeBase, configs, role/permission mapping.
  - Impact: canonicalizes multi-tenant metadata and default configuration; important for governance.
  - Action: add JSON schema/pydantic view and unit tests for to_dict and default values.

- lightrag/services/tenant_service.py (NEW)
  - Implements tenant/KB lifecycle, membership, RBAC checks, and Postgres-backed queries when available.
  - Impact: critical authorization and multi-tenant CRUD. Calls to DB functions (has_tenant_access) must exist in DB.
  - Risk: error handling assumes query shapes; must test against asyncpg row shapes (Record vs arrays).
  - Action: add DB migrations and tests verifying has_tenant_access and membership queries; add strong integration tests for permissions.

- lightrag/tenant_rag_manager.py (NEW)
  - Tenant-scoped LightRAG instance manager with LRU caching, creation, initialization and eviction.
  - Impact: performance & isolation — per-tenant storage paths, careful eviction required to avoid resource leaks.
  - Action: add unit tests for LRU eviction and concurrency (race conditions), measure memory and file handle usage under load.

- lightrag/kg/postgres_rls.sql (NEW)
  - Adds RLS policies to core tables and helper to set tenant context using set_config('app.current_tenant',..).
  - Impact: strong DB-level defense-in-depth preventing cross-tenant reads if used end-to-end.
  - Risk: RLS will break existing queries unless session variable is set for each DB connection; migrations and DB driver instrumentation required.
  - Action: create a migration and instrument DB connection setup to set session variable immediately per request. Add integration tests.

- lightrag/api/config.py (modified)
  - New flags: MULTI_TENANT_STRICT_MODE, REQUIRE_USER_AUTH, SUPER_ADMIN_USERS; default LLM/embedding binding changed to openai.
  - Impact: default behaviour changes, risk of breaking pre-existing deployments expecting ollama; security features toggled by env.
  - Action: update README and examples; emphasize env vars and warn about default model changes.

- lightrag/api/auth.py (modified)
  - Adds logging for token validation problems, but still uses default token secret if not changed.
  - Impact: better debugging but still risky defaults.
  - Action: require secure token secrets in prod and document default is temporary only for local dev.

- Other: many docs, e2e tests, routers and API surface expanded (tenant_routes.py, membership_routes.py, admin_routes.py), asset removals from webui
  - Impact: many new surfaces — need a test matrix for public vs tenant-limited endpoints.

Summary: The local branch adds a complete multi-tenant subsystem with DB-level RLS and RBAC. The changes are substantive and beneficial for enterprise deployment, but require database migrations, configuration, runtime wiring and robust tests before release.
