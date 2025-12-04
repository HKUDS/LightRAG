
# Security audit — prioritized, concise (1 page)

Priority mapping: P0 (must fix before release) | P1 (fix ASAP) | P2 (recommended)

P0 — Critical blockers

- DB migrations & session: The RLS policies (lightrag/kg/postgres_rls.sql) require DB session variable set for EVERY DB connection. Without application instrumentation to set app.current_tenant per request/connection, RLS will either not be applied (if not set) or break queries. Add DB migrations to create membership and access-check functions (has_tenant_access, has_tenant_membership) used by tenant_service. Test RLS end-to-end before release.

- Secrets: token_secret default is 'lightrag-jwt-default-secret' — this is insecure for production. Require production deployments to set TOKEN_SECRET env var or fail-start. Rotate/replace default secret in examples to a placeholder that forces change.

P1 — High risk, serious but not immediately blocking

- Sensitive fallback modes: get_tenant_context_optional will allow fallback to global RAG unless MULTI_TENANT_STRICT_MODE is true. Be explicit in config and document the leakage risk. Default is to allow fallback; recommend setting default to strict=true for enterprise builds and document consequences.

- Super-admin handling: SUPER_ADMIN_USERS is read from env/config and defaults to "admin". Empty env var means no super-admins — but code treats absence as allowing a default 'admin'. Clarify and make explicit: empty => no super-admins; set to '*' for wide admin (explicit opt-in).

P2 — Improvements and hygiene

- Authorization logging: dependencies.py and auth.py add debug/warning logs. Avoid logging full JWTs or long token prefixes; ensure logs are safe and don't leak PII.

- Middleware subdomain parsing is naive and can accidentally interpret hosts. Add allowlist of trusted domains and thorough unit tests for host parsing edge-cases (IP, localhost, proxies, forwarded headers).

- validate_identifier / validate_working_directory — good protections. Ensure storage systems that accept slugs vs UUIDs are consistent and documented.

Testing required (minimum):

- RLS + session binding tests (P0)

- Permission/role matrix tests (get_tenant_context, check_permission, get_admin_context) for admin/editor/viewer/owner (P1)

- Tenant lifecycle and membership flows (create_tenant add_user_to_tenant remove/update membership) (P1)

Deployment notes:

- During deployment, provision a migrations step to install tenants/knowledge_bases tables and membership function. Tie the DB connection pool to a per-request session variable setter.

- Run an RLS smoke test immediately after deploying (create two tenants, insert records with different tenant_id, verify queries cannot cross-access when app.current_tenant is set accordingly).
