
# Tests & acceptance checklist (minimal but actionable)

Security & isolation

- [ ] RLS enforcement smoke-test (PG): set session app.current_tenant -> verify cross-tenant reads are blocked

- [ ] DB function has_tenant_access unit tests + integration tests using expected query shapes

- [ ] get_tenant_context / get_tenant_context_optional tests verifying header precedence, missing token behavior, and strict-mode behavior

Functional

- [ ] Tenant lifecycle: create tenant, add user as owner, create KB, add documents, delete KB, delete tenant

- [ ] Membership actions: add/remove/update membership and verify effects on access

- [ ] TenantRAGManager: concurrency test that spawns multiple tasks requesting same tenant/kb and verifies a single instance created then evicted properly.

E2E

- [ ] Run provided e2e tests under e2e/ with a Postgres instance that includes the migrations and RLS applied. Verify all pass.

- [ ] Negative tests: attempt to access KB from different tenant using crafted token and assert HTTP 403.

Operational / CI

- [ ] Add CI jobs that run the above tests using a disposable Postgres service in GitHub Actions or local Docker Compose.

If tests above pass -> safe to roll out to staging. If any P0 test fails, block release and revert until fixed.
