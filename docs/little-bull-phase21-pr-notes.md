# Little Bull Premium Phase 21 PR Notes

Status date: 2026-05-01

This note summarizes commit `40779b98 Fix Little Bull permissions and tri-bank smoke`.
It contains no secrets, tokens, passwords, `.env` values or connection strings.

## Summary

- Hardened Little Bull Premium document upload so classified uploads require group and subgroup selection.
- Added typed frontend API contracts for knowledge groups, subgroups and classified document upload responses.
- Added pure workspace helpers for Premium navigation permissions, upload readiness, subgroup filtering and stale selection cleanup.
- Added Bun coverage for classified upload, permission fallbacks and frontend API upload parameters.
- Added Playwright visual smoke coverage for the Premium shell across mobile, tablet and desktop viewports.
- Fixed macOS Bash 3.2 drift by making direct setup execution and interactive setup tests prefer Bash 4+ when available.
- Closed least-privilege UI permission bugs: document listing works with `little_bull.documents.read`, scoped workspace choices can be derived from the principal, and taxonomy for classified upload remains gated by `little_bull.areas.read`.
- Recorded additive live-smoke and strict tri-bank smoke evidence in the Little Bull risk register.

## Backend / Control Plane

- Little Bull service/router/models/admin store were extended for the Phase 20/21 hardening path already covered by enterprise contract tests.
- Classified upload contracts now preserve workspace, group, subgroup and registry document identifiers through backend/frontend boundaries.
- No global `NEO4J_WORKSPACE`, `QDRANT_WORKSPACE` or `POSTGRES_WORKSPACE` override was set.
- No Neo4j/Qdrant data was deleted or reset.

## Frontend

- `LittleBullPreview` now loads groups/subgroups for uploads and blocks file selection until classification is complete.
- Document refresh no longer fails for users who can read documents but cannot read areas.
- Users without area-listing permission can still select scoped workspaces from their principal `workspace_ids`.
- `littleBullWorkspace.ts` centralizes permission and upload state rules for direct unit testing.
- Playwright smoke uses route-mocked local API responses and a fake non-secret JWT.
- Playwright reports and test-results are ignored by git.

## Validation Evidence

- `uv run ruff check lightrag_enterprise tests_enterprise lightrag/api/lightrag_server.py tests/test_interactive_setup/_helpers.py`
- `python -m lightrag_enterprise.system.migrate`
- `./scripts/test.sh tests_enterprise -q`: 184 passed, 4 skipped.
- `./scripts/test.sh tests -q`: 794 passed, 32 skipped.
- `cd lightrag_webui && bun test`: 21 passed.
- `cd lightrag_webui && bunx tsc --noEmit`
- `cd lightrag_webui && bun run lint`
- `cd lightrag_webui && bun run test:visual`: 3 passed.
- `cd lightrag_webui && bun run build`
- `node /Users/joao_tourinho/Documents/specops-tooling-os/packages/cli/dist/index.js validate`: 0 issues.
- `node /Users/joao_tourinho/Documents/specops-tooling-os/packages/cli/dist/index.js eval`: 10 passed, 0 failed.
- `git diff --check`

## Smoke Evidence

- Additive current-worktree smoke on temporary server `127.0.0.1:9631`:
  - workspace `phase21_smoke_820a9f72e3b4`
  - document `doc-1588f7b35aed62f7a69230c53c9b711a`
  - naive query returned 1 retrieval reference
  - server stopped after validation

- Strict tri-bank smoke on temporary server `127.0.0.1:9632`:
  - target storage: PGKV/PGDocStatus + Neo4j + Qdrant
  - workspace `phase21_tribank_44657da3ed86`
  - document `doc-974a377549c90fa3d2434ec663c8b1f4`
  - Qdrant counts for this workspace: 1 chunk, 5 entities, 0 relationships
  - naive query returned 1 retrieval reference
  - server stopped after validation

## Rollback

- Disable feature flags when applicable:
  - `LITTLE_BULL_GRAPH_V2_ENABLED`
  - `LITTLE_BULL_QDRANT_DATA_PLANE_ENABLED`
  - `LITTLE_BULL_OBSIDIAN_WORKSPACE_ENABLED`
- PostgreSQL remains the control-plane source of truth.
- Neo4j and Qdrant artifacts are rebuildable from control-plane state plus reindexing.
- Do not run cleanup commands without explicit approval and exact target list.

## READY Status

Do not declare READY from this note alone.
READY still requires final release review, explicit cleanup decision if cleanup is desired, and all production release gates.
