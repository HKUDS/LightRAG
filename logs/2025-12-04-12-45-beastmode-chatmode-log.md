Task logs - 2025-12-04 12:45

- Actions:
  - Audited codebase for tenant handling, header usage, webui state, ingestion endpoints and tests.
  - Updated `specs/001-spec-improvements` with a Reality Check section and normalized header names to `X-Tenant-ID` and `X-KB-ID` throughout.
  - Added pragmatic next steps and concrete file pointers for frontend and backend changes (no implementation performed).

- Key findings:
  - Frontend already injects `X-Tenant-ID` / `X-KB-ID` headers using an axios interceptor (`lightrag_webui/src/api/client.ts`), driven by `localStorage` keys `SELECTED_TENANT` and `SELECTED_KB`.
  - Backend has existing tenant dependency helpers (`lightrag/api/dependencies.py`) that read `X-Tenant-ID` and provide fallback logic.
  - e2e tests and scripts already exercise multi-tenant behavior with header usage.

- Next steps:
  - (If asked) Implement `tenantStateManager` in the WebUI and integrate with the existing interceptor.
  - (If asked) Add server-side snapshot/share-token endpoints and tests.

- Notes and recommendations:
  - Keep tenant IDs out of URLs and ensure server-side validation.
  - Keep tenant identifiers opaque (UUID/numeric internal) and avoid storing secrets in client storage.

