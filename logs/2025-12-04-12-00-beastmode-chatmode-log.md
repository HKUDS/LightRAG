Task logs - 2025-12-04 12:00

- Actions:
  - Updated `specs/001-spec-improvements` with detailed UX, URL schema, frontend state strategy, backend recommendations, ingestion pipeline changes, tests, acceptance criteria and rollout notes.
  - Created a todo list to track implementation steps using the project todo tool.

- Decisions:
  - URL will be canonical source-of-truth: routes begin with `/t/:tenantId` and encode KB and UI state as query params.
  - Frontend state saved to URL and sessionStorage (URL takes precedence).
  - Backend must validate tenant identity on every request; tenant_id stored on tenant-scoped tables.

- Next steps:
  - Implement `tenantStateManager` frontend module and update Documents routing.
  - Add DB indexes and update ingestion API to include `tenant_id` and `external_id` for idempotency.
  - Add unit/integration/e2e tests described in spec.

- Lessons/insights:
  - Treat URL + server validation as single source of truth to avoid cross-tenant leaks.
