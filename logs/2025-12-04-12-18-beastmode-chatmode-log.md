Task logs - 2025-12-04 12:18

- Actions:
  - Updated `specs/001-spec-improvements` to require tenant context via `X-Tenant` header and to remove route-based tenant IDs.
  - Adjusted frontend state rules so URLs remain tenant-agnostic and sessionStorage keys must be tenant-prefixed using header-provided tenant id.
  - Updated API contract, ingestion pipeline notes, tests, acceptance criteria, and documentation guidance to reflect header-based tenant handling and secure snapshot/share-token approach.
  - Advanced the project TODOs: marked spec and URL schema as completed and set `Frontend state design` to in-progress.

- Decisions:
  - Tenant identifiers must not appear in URLs for security; always derive tenant context from `X-Tenant` header or validated auth token claims.
  - URLs remain tenant-agnostic; for cross-user sharing in the same tenant, implement server-side snapshot tokens validated with `X-Tenant`.

- Next steps:
  - Implement `tenantStateManager` frontend module (in-progress) and integrate per-tenant sessionStorage handling.
  - After frontend work, start backend validation and ingestion changes.

- Lessons/insights:
  - Hiding tenant in headers improves security but requires explicit design for sharing/bookmark features (server-side snapshots) and careful sessionStorage scoping.
