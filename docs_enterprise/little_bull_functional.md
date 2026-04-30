# Little Bull Functional Layer

## Objective

Little Bull is the governed local-first facade for using LightRAG from a home/workspace-oriented UI. It turns the previous fixture-only preview into authenticated behavior backed by the LightRAG document/query pipeline, system authorization, approval requests, and durable audit events.

## Runtime Flags

- `LITTLE_BULL_FUNCTIONAL_ENABLED=true`: enables `/little-bull/*`, `/auth/me`, `/system/*`, `/approvals`, and `/audit/events`.
- `LITTLE_BULL_PRIVATE_STRICT=true`: blocks sensitive/private queries unless the UI selects the `privado` model profile.
- `LLM_BINDING=openai`, `LLM_BINDING_HOST=https://openrouter.ai/api/v1`, `LLM_BINDING_API_KEY`: supported configuration for OpenRouter as the hosted LLM API.
- `LLM_MODEL`: OpenRouter model id used by the hosted profile, for example `<provider>/<model>`.
- `LITTLE_BULL_PRIVATE_LOCAL_MODEL`: optional explicit local model for `privado` queries when the active LightRAG model is hosted, for example `qwen-local` or `ollama/qwen-local`.
- `LITTLE_BULL_PRIVATE_LOCAL_BINDING=ollama`: binding used by the explicit private/local model.
- `LITTLE_BULL_PRIVATE_LOCAL_HOST`, `LITTLE_BULL_PRIVATE_LOCAL_API_KEY`, `LITTLE_BULL_PRIVATE_LOCAL_TIMEOUT`: connection options for the private/local model.
- `LITTLE_BULL_APPROVALS_ENFORCED=true`: destructive actions become approval requests instead of executing immediately.
- `LIGHTRAG_SYSTEM_DATABASE_URL` or `DATABASE_URL`: required when the functional layer is enabled; stores users, tenants, workspaces, roles, approvals, and audit in PostgreSQL.
- `LIGHTRAG_SYSTEM_TOKEN_SECRET` or `TOKEN_SECRET`: required before enterprise tokens can be issued or validated.
- `LIGHTRAG_SYSTEM_ALLOW_INSECURE_DEV_SECRET=true`: allows the built-in development token secret only for local throwaway environments.
- `LIGHTRAG_SYSTEM_ALLOW_IN_MEMORY_REPOSITORY=true`: allows the in-memory system repository only for local tests or throwaway development.
- `LITTLE_BULL_BOOTSTRAP_TOKEN`: required header gate for `POST /system/bootstrap-master`.

If the functional layer is enabled and no database URL is set, the system layer fails closed instead of issuing guest tokens or falling back to legacy auth. The in-memory repository must be opted into explicitly and should not be used for durable local-first operation.

## Bootstrap

Provision or link the Little Bull system PostgreSQL database first. If an existing/offline database is already available, validate and bind it:

```bash
LIGHTRAG_SYSTEM_DATABASE_URL='postgresql://app_user:<password>@localhost:5432/lightrag_little_bull_e2e' \
python -m lightrag_enterprise.system.provision_postgres --write-env .env
```

If only an administrator connection is available, create or link a dedicated database and optional application user:

```bash
LIGHTRAG_SYSTEM_POSTGRES_ADMIN_URL='postgresql://postgres:<admin-password>@localhost:5432/postgres' \
python -m lightrag_enterprise.system.provision_postgres \
  --database lightrag_little_bull_e2e \
  --app-user lightrag_system \
  --app-password '<app-password>' \
  --write-env .env
```

The provisioner validates an existing database when `LIGHTRAG_SYSTEM_DATABASE_URL`/`DATABASE_URL` is set. When no system database URL exists, it uses the admin connection to create the dedicated database if missing, creates the app role only when `--app-password` is supplied, grants the app role access, runs the schema, and optionally writes `LIGHTRAG_SYSTEM_DATABASE_URL` plus `LITTLE_BULL_FUNCTIONAL_ENABLED=true` to `.env`. It does not drop, truncate, or reset existing data.

Run the schema migration:

```bash
python -m lightrag_enterprise.system.migrate
```

Bootstrap the global MASTER:

```bash
python -m lightrag_enterprise.system.bootstrap_master --username master --password '<password>'
```

The script creates the default tenant/workspace and assigns the global MASTER to that workspace.
Set `LIGHTRAG_SYSTEM_TOKEN_SECRET` or `TOKEN_SECRET` before logging in. HTTP bootstrap is intentionally closed unless `LITTLE_BULL_BOOTSTRAP_TOKEN` is configured and sent as `X-Little-Bull-Bootstrap-Token`; the CLI bootstrap is the preferred local-first path.

OpenRouter is supported as the normal hosted LLM API for Little Bull. Configure it through the OpenAI-compatible binding:

```bash
LLM_BINDING=openai
LLM_BINDING_HOST=https://openrouter.ai/api/v1
LLM_BINDING_API_KEY='<openrouter-api-key>'
LLM_MODEL='<openrouter-model-id>'
```

Hosted providers can be used for non-private profiles such as `rapido`, `equilibrado`, and `inteligente`. Private/sensitive flows are a separate policy surface: with `LITTLE_BULL_PRIVATE_STRICT=true`, requests marked `sensivel`/`privado` or using `model_profile=privado` require either a local/private runtime or a MASTER-managed hosted exception. Do not describe OpenRouter-backed processing as `Privado/local`; it is hosted processing with explicit approval and audit.

MASTER can allow OpenRouter for private/sensitive data in a specific tenant/workspace:

```bash
curl -X POST http://127.0.0.1:9621/system/policies/private-data/hosted-llm-exception \
  -H "Authorization: Bearer <master-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "tenant_id": "default",
    "workspace_id": "default",
    "provider": "openrouter",
    "binding": "openai",
    "binding_host": "https://openrouter.ai/api/v1",
    "allowed_model_ids": ["openai/gpt-4o-mini"],
    "allowed_confidentiality": ["sensivel", "privado"],
    "expires_at": "2026-05-28T00:00:00Z",
    "approval_id": "apr_optional",
    "reason": "MASTER approved hosted processing for this workspace.",
    "ticket_id": "LB-42"
  }'
```

The policy key is `little_bull.private_data.hosted_llm_exception`. It is fail-closed: missing, disabled, expired, malformed, wrong provider, wrong binding host, wrong model, wrong confidentiality, or wrong workspace still blocks hosted private processing. The policy is audited on update, and each query using it records `hosted_private_exception=true`, `hosted_private_policy_hash`, selected provider/model, and `cache_disabled=true`. The `privado` model profile remains local-only; use hosted exceptions with hosted profiles such as `equilibrado`.

## Permission Matrix

| Role | Permissions |
| --- | --- |
| `operador` | read areas, read/upload documents, query, read assistants, read activity, save/read/export own conversations, suggest correlations |
| `gerente` | operador permissions plus workspace management, document deletion request, approval read/decide, audit read, decide correlation suggestions |
| `master` | global `*` permission |

Important activity IDs:

- `little_bull.query`
- `little_bull.documents.read`
- `little_bull.documents.upload`
- `little_bull.documents.delete`
- `little_bull.documents.reindex`
- `little_bull.core.cache.clear`
- `little_bull.core.graph.create`
- `little_bull.core.graph.mutate`
- `little_bull.core.ollama.use`
- `little_bull.core.pipeline.manage`
- `little_bull.core.query.data`
- `little_bull.approvals.decide`
- `little_bull.audit.read`
- `little_bull.models.manage`
- `little_bull.agents.manage`
- `little_bull.conversations.read`
- `little_bull.conversations.save`
- `little_bull.conversations.export`
- `little_bull.correlations.suggest`
- `little_bull.correlations.decide`

Core LightRAG routes use the same enterprise principal when Little Bull functional mode is active. Ingestion, pipeline operations, graph create/mutate, cache clear, query/data, and Ollama-compatible chat/generate/status routes are audited before the handler runs. Unauthorized enterprise principals receive `403`, and destructive actions that require approval receive `409` with a pending approval payload instead of executing.

Approval decisions are intentionally action-allowlisted. Approving a `little_bull.documents.delete` request transitions the request through execution and calls `LightRAG.adelete_by_doc_id` once, then stores `executed` with an audit event tied to the `approval_id`. Approving a `little_bull.documents.reindex` request executes the Little Bull reindex handler for that workspace and records the queued/no-files result with the same `approval_id`. Re-approving an already executed request is idempotent and does not execute the action again. Reindex approvals compare the execution payload with the approved payload, including `include_archived`, `include_input_root`, and `destructive_rebuild`.

When `destructive_rebuild=true`, the reindex handler always requires approval even if the global approvals flag is off. After approval, Little Bull first verifies supported source files, creates a filesystem snapshot under the LightRAG working directory, drops the current workspace storages through the LightRAG storage APIs, then queues the workspace sources for indexing. If no supported source files are available, the destructive rebuild is not executed. Rollback is exposed as a MASTER-only audited endpoint for attached non-startup workspaces; rollback of the active startup workspace is intentionally blocked online and must be performed offline while the server is stopped.

Private/local queries are routed through a Little Bull gateway before calling LightRAG. If the request or workspace contains private data, or if the user explicitly selects the `privado` profile, the gateway requires a local/private runtime. When the active LightRAG binding is local, the normal `rag.aquery_llm` path is used. When the active binding is hosted, `LITTLE_BULL_PRIVATE_LOCAL_MODEL` must be configured so the query can use `QueryParam.model_func` with the local model. If no local runtime is available, the query fails closed and is audited before any RAG call. Private/local queries temporarily disable the LLM response cache for that call to avoid reusing hosted responses across privacy profiles.

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
- `GET /little-bull/graph?workspace_id=...&label=...&max_depth=...&max_nodes=...`
- `GET /little-bull/graph/label/list?workspace_id=...`
- `GET /little-bull/graph/label/popular?workspace_id=...&limit=300`
- `GET /little-bull/graph/label/search?workspace_id=...&q=...`
- `GET /little-bull/admin/models?workspace_id=...`
- `POST /little-bull/admin/models?workspace_id=...`
- `GET /little-bull/admin/embedding-models`
- `GET /little-bull/admin/knowledge-bases`
- `POST /little-bull/admin/knowledge-bases`
- `POST /little-bull/admin/knowledge-bases/{workspace_id}/attach-data-plane`
- `POST /little-bull/admin/knowledge-bases/{workspace_id}/reindex`
- `POST /little-bull/admin/knowledge-bases/{workspace_id}/rollback`
- `POST /little-bull/admin/embedding-cost-estimate`
- `GET /little-bull/admin/agents?workspace_id=...`
- `POST /little-bull/admin/agents?workspace_id=...`
- `POST /little-bull/admin/agents/preview`
- `GET /little-bull/conversations?workspace_id=...`
- `POST /little-bull/conversations`
- `GET /little-bull/conversations/{conversation_id}`
- `GET /little-bull/conversations/{conversation_id}/export?format=md|txt|docx`
- `GET /little-bull/correlation-suggestions?workspace_id=...`
- `POST /little-bull/correlation-suggestions`
- `POST /little-bull/correlation-suggestions/{suggestion_id}/approve`
- `POST /little-bull/correlation-suggestions/{suggestion_id}/reject`
- `GET /enterprise/model-catalog`
- `POST /enterprise/model-catalog/sync`
- `POST /enterprise/model-route`
- `GET /approvals`
- `POST /approvals/{approval_id}/approve`
- `POST /approvals/{approval_id}/reject`
- `GET /audit/events`

## Frontend Routes

- `#/little-bull`
- `#/little-bull-preview`

Both routes require a non-guest authenticated session. The preview route name is kept as a compatibility alias, but the UI now calls real APIs.

## Agent Studio v1

Admin > Agentes is now an Agent Studio surface instead of a flat prompt editor. Agent configs remain stored in PostgreSQL as versioned JSONB (`config.schema_version=1`) so no additional migration is required for the first iteration.

The v1 config covers identity, model/profile, knowledge retrieval preferences, personality, ethics, vocabulary, tools policy, memory declaration, output format, and saved test cases. `POST /little-bull/admin/agents/preview` returns the normalized agent, issues, readiness score, and compiled prompt. Publishing an agent blocks `severity=error` findings, including unsupported schema versions, prompt-injection patterns, raw secret-like values, invalid memory scope, disabled active tools, and vocabulary conflicts.

Queries using `agent_id` now execute the same compiled Agent Studio prompt returned by preview. The agent model profile is resolved before the private/local gateway, so an agent configured as `privado` is evaluated as private even if the request default profile is `equilibrado`.

## Validation

Recommended checks for this layer:

```bash
./scripts/test.sh tests_enterprise/test_little_bull_agent_studio.py tests_enterprise/test_little_bull_service.py tests_enterprise/test_system_approval_execution.py tests_enterprise/test_little_bull_router_contract.py -q
./scripts/test.sh tests_enterprise
uv run ruff check lightrag_enterprise tests_enterprise lightrag/api/lightrag_server.py
cd lightrag_webui && bunx tsc --noEmit
cd lightrag_webui && bun test src/api/lightrag.test.ts src/fixtures/littleBullKnowledge.test.ts
cd lightrag_webui && bun run build
```

Real smoke against a running API, local PostgreSQL, and OpenRouter/LLM API is opt-in:

```bash
LITTLE_BULL_E2E=1 \
LITTLE_BULL_E2E_BOOTSTRAP=1 \
LIGHTRAG_API_BASE_URL=http://127.0.0.1:9621 \
LIGHTRAG_SYSTEM_DATABASE_URL=postgresql://localhost:5432/lightrag_little_bull_e2e \
LIGHTRAG_SYSTEM_TOKEN_SECRET='<token-secret>' \
LITTLE_BULL_BOOTSTRAP_TOKEN='<bootstrap-token>' \
LITTLE_BULL_E2E_MASTER_USERNAME=master \
LITTLE_BULL_E2E_MASTER_PASSWORD='<password>' \
LLM_BINDING=openai \
LLM_BINDING_HOST=https://openrouter.ai/api/v1 \
LLM_BINDING_API_KEY='<openrouter-api-key>' \
LLM_MODEL='<openrouter-model-id>' \
LITTLE_BULL_E2E_CONFIDENTIALITY=normal \
LITTLE_BULL_E2E_MODEL_PROFILE=equilibrado \
./scripts/test.sh tests_enterprise/test_little_bull_real_api_smoke.py -q
```

The smoke refuses to run unless `LITTLE_BULL_E2E=1` is set, a PostgreSQL URL is configured, an LLM API signal is present, and the database URL looks local and dedicated to `test`, `e2e`, or `smoke`. It does not reset or truncate the database. Use `LITTLE_BULL_E2E_ALLOW_NON_TEST_DB=1` only for an intentionally isolated environment. The current smoke covers bootstrap/login, authenticated query through the Little Bull API, anonymous denial, activity, and durable audit evidence. By default it uses the hosted path (`normal` + `equilibrado`) so OpenRouter can be tested. To test a hosted private exception, first create the MASTER policy above, then set `LITTLE_BULL_E2E_CONFIDENTIALITY=privado`, `LITTLE_BULL_E2E_MODEL_PROFILE=equilibrado`, and `LITTLE_BULL_E2E_HOSTED_PRIVATE_EXCEPTION=1`. To test strict local-only private routing separately, set `LITTLE_BULL_E2E_CONFIDENTIALITY=privado`, `LITTLE_BULL_E2E_MODEL_PROFILE=privado`, and configure a local/private runtime. Upload, indexing, and executable delete approval remain a heavier real-integration gate because they also depend on embedding configuration and background indexing latency.

Upload/index/query smoke is available as an explicit heavier gate:

```bash
LITTLE_BULL_E2E=1 \
LITTLE_BULL_E2E_UPLOAD=1 \
./scripts/test.sh tests_enterprise/test_little_bull_real_api_smoke.py::test_little_bull_real_api_upload_index_query_smoke -q
```

The workspace graph view uses `/little-bull/graph/*` rather than the legacy core graph endpoints, so the UI cannot accidentally read a different startup workspace. The popular-labels contract accepts the frontend default of `limit=300`.
In Little Bull workspace context, the graph properties panel is read-only for persistent entity/relation fields until mutation endpoints are workspace-aware and approval-gated. The graph still supports viewing, search, expansion, pruning, zoom, layout, and legend interactions.

## Known Limits

- The facade authorizes workspace access before calling LightRAG and blocks authorized workspaces that are not attached to a Little Bull data plane. Attached workspaces get dedicated LightRAG instances cached at runtime; production hardening still needs lifecycle controls for cache eviction, rebuild windows, and long-running indexing observability.
- Private/local enforcement now has an explicit gateway for Little Bull queries. It blocks hosted profiles when a workspace/request is private and can override the query model function to an explicit local Ollama runtime. Automatic content classification remains a later hardening item.
- The Admin UI now persists model settings, Agent Studio configs, conversations, DOCX/MD/TXT exports, correlation suggestions, approvals, and audit. Chat model overrides can be used through stored OpenAI-compatible profiles. Embedding settings are persisted and audited. Changing embeddings marks the base as requiring reindex; the normal reindex queues sources, while `destructive_rebuild=true` performs an approval-backed snapshot/drop/requeue flow.
- Agent Studio v1 enforces validation at publish and prompt/profile at query time. Tool execution policy, memory retention automation, and redaction for saved/exported private conversations are still declared capabilities; action-specific enforcement beyond query/model routing remains a later hardening item.
- Core route governance currently protects document ingestion/deletion, graph create/mutate/delete/merge, cache clear, pipeline reprocess/cancel, query/data, and Ollama-compatible routes. `little_bull.documents.delete` and `little_bull.documents.reindex` have executable approval handlers in this slice; other core approvals remain decision-only until their action-specific executors are added.
- Little Bull graph editing is intentionally disabled in the workspace UI until entity/relation edit, merge, and existence checks are exposed through `/little-bull/graph/*` or another governed workspace-aware facade.
