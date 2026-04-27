# Little Bull Functional Layer

## Objective

Little Bull is the governed local-first facade for using LightRAG from a home/workspace-oriented UI. It turns the previous fixture-only preview into authenticated behavior backed by the LightRAG document/query pipeline, system authorization, approval requests, and durable audit events.

## Runtime Flags

- `LITTLE_BULL_FUNCTIONAL_ENABLED=true`: enables `/little-bull/*`, `/auth/me`, `/system/*`, `/approvals`, and `/audit/events`.
- `LITTLE_BULL_PRIVATE_STRICT=true`: blocks sensitive/private queries unless the UI selects the `privado` model profile.
- `LITTLE_BULL_APPROVALS_ENFORCED=true`: destructive actions become approval requests instead of executing immediately.
- `LIGHTRAG_SYSTEM_DATABASE_URL` or `DATABASE_URL`: enables PostgreSQL storage for users, tenants, workspaces, roles, approvals, and audit.
- `LITTLE_BULL_BOOTSTRAP_TOKEN`: optional header gate for `POST /system/bootstrap-master`.

If no database URL is set, the system layer uses an in-memory repository for local tests and development. PostgreSQL is required for durable local-first operation.

## Bootstrap

Run the schema migration:

```bash
python -m lightrag_enterprise.system.migrate
```

Bootstrap the global MASTER:

```bash
python -m lightrag_enterprise.system.bootstrap_master --username master --password '<password>'
```

The script creates the default tenant/workspace and assigns the global MASTER to that workspace.

## Permission Matrix

| Role | Permissions |
| --- | --- |
| `operador` | read areas, read/upload documents, query, read assistants, read activity |
| `gerente` | operador permissions plus workspace management, document deletion request, approval read/decide, audit read |
| `master` | global `*` permission |

Important activity IDs:

- `little_bull.query`
- `little_bull.documents.read`
- `little_bull.documents.upload`
- `little_bull.documents.delete`
- `little_bull.approvals.decide`
- `little_bull.audit.read`

## API Surface

- `POST /auth/login`
- `GET /auth/me`
- `POST /system/bootstrap-master`
- `GET /system/tenants`
- `GET /system/workspaces`
- `POST /system/users`
- `GET /little-bull/areas`
- `GET /little-bull/documents?workspace_id=...`
- `POST /little-bull/documents/upload?workspace_id=...`
- `DELETE /little-bull/documents/{document_id}?workspace_id=...`
- `POST /little-bull/query`
- `GET /little-bull/activity?workspace_id=...`
- `GET /little-bull/assistants?workspace_id=...`
- `GET /approvals`
- `POST /approvals/{approval_id}/approve`
- `POST /approvals/{approval_id}/reject`
- `GET /audit/events`

## Frontend Routes

- `#/little-bull`
- `#/little-bull-preview`

Both routes require a non-guest authenticated session. The preview route name is kept as a compatibility alias, but the UI now calls real APIs.

## Validation

Recommended checks for this layer:

```bash
./scripts/test.sh tests_enterprise
uv run ruff check lightrag_enterprise tests_enterprise
cd lightrag_webui && bunx tsc --noEmit
cd lightrag_webui && bun test src/api/lightrag.test.ts src/fixtures/littleBullKnowledge.test.ts
cd lightrag_webui && bun run build
```

## Known Limits

- The facade authorizes workspace access before calling LightRAG, but the current LightRAG instance is still initialized with a startup workspace. Full dynamic multi-workspace RAG instances remain a later production hardening item.
- The first UI slice supports real areas, documents, upload, query, activity, approvals, assistants, and audit. Full tenant/user/policy administration is available through backend endpoints first.
