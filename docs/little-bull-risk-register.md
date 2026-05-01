# Little Bull Risk Register

Status date: 2026-05-01

This register tracks residual risks that must be reduced or closed before a later READY gate. It intentionally contains no secrets, tokens, passwords, `.env` values, or connection strings.

## Active Residual Risks

### R1: Diagnostic tri-db stack is not production-shaped

- Status: contained.
- Current exposure: the local `trag-phase2` stack remains diagnostic only because it was created before the Qdrant template pin and uses a Neo4j no-auth setup on loopback.
- Current control: data-plane product flags remain disabled; Phase 3 pilot now fails closed on Qdrant client/server mismatch and Neo4j no-auth unless explicit diagnostic overrides are set.
- Correction plan:
  1. Create a new non-destructive stack name with new volumes and pinned Qdrant image.
  2. Enable Neo4j auth in the new stack and validate HTTP/Bolt health without printing credentials.
  3. Run Phase 2 health checks against the new stack.
  4. Mark the old `trag-phase2` stack as superseded and stop it without `-v` only when no active process depends on it.
- Stop condition: do not remove old containers or volumes without explicit confirmation.
- Exit gate: Postgres, Neo4j and Qdrant health checks pass on the new stack; no global workspace override env vars are set.

### R2: Pilot artifacts still exist

- Status: contained.
- Current exposure: Phase 3 pilot artifacts remain in local Postgres/Qdrant/Neo4j.
- Current control: `scripts/little_bull_phase3_inventory.py` is read-only and reports `destructive_actions: []`; no cleanup has been run.
- Correction plan:
  1. Keep using fresh workspaces for subsequent pilots.
  2. Before cleanup, present the exact database schemas, Qdrant collections, Neo4j labels/indexes and Docker volumes that would be affected.
  3. Ask for confirmation before any destructive operation.
  4. Prefer abandonment/rebuild over mutation because Postgres is the control-plane source of truth.
- Stop condition: no `docker compose down -v`, `docker volume rm`, `DROP`, `DELETE`, Qdrant collection deletion, or Neo4j delete without confirmation.
- Exit gate: either cleanup is explicitly approved and validated, or artifacts remain inventoried and isolated from product flows.

### R3: SpecOps Tooling OS patch is outside this repository

- Status: contained.
- Current exposure: the local SpecOps walker patch lives in `/Users/joao_tourinho/Documents/specops-tooling-os`, outside TRAG.
- Current control: TRAG `SpecOps validate` and `SpecOps eval` pass with the current local tooling; no TRAG secret or data depends on the external patch.
- Correction plan:
  1. In a dedicated SpecOps Tooling OS branch, commit the walker ignore update for `.venv`, `venv` and local caches.
  2. Run that project’s own tests/build.
  3. Return to TRAG and re-run `SpecOps validate` and `SpecOps eval`.
- Stop condition: do not vendor or copy SpecOps internals into TRAG.
- Exit gate: external patch is committed or released in its own repo, and TRAG gates still pass.

## Phase 5 Risk Controls Added

- Markdown notes require `group_id` and `subgroup_id` at service level.
- Source document links must stay in the same workspace, group and subgroup as the note.
- Markdown note registry has an idempotent `NOT VALID` check to protect new markdown notes without validating or deleting historical rows.
- Wiki links and tags are derived from markdown and stored in PostgreSQL control-plane tables only; Neo4j/Qdrant are not activated in this phase.
- Note reads and writes are audited through existing Little Bull activity gates.

## Phase 6 Risk Controls Added

- Backlinks are stored in PostgreSQL control-plane tables and remain tenant/workspace scoped.
- Manual backlinks validate source and target references before persistence.
- Backlinks across different group/subgroup scopes are rejected.
- Wikilink-derived backlinks only resolve notes within the same group/subgroup; cross-subgroup labels stay unresolved instead of creating cross-scope edges.
- Provenance panels are limited to canonical note/document targets to avoid workspace-wide fallback queries.
- Source provenance validates document/note references and rejects mixed group/subgroup references.
- `graph_edge_origin_id`, `agent_id` and `usage_ledger_id` are blocked until scoped validation is implemented for those IDs.

## Phase 7 Risk Controls Added

- Canvas boards, nodes and edges are stored in PostgreSQL control-plane tables only; Neo4j/Qdrant remain inactive for this phase.
- Canvas boards require group/subgroup and cannot be moved across group/subgroup by reposting an existing slug or id.
- Canvas nodes with `ref_id` currently support only scoped note/document references.
- Canvas node references must share the board group/subgroup scope.
- Canvas edges require both endpoint nodes to belong to the route board.
- Client-supplied canvas node/edge ids are rejected unless the existing id already belongs to the route board.
- Canvas-to-dossier export creates a draft dossier with `requires_lgpd_review=true`.

## Phase 8 Risk Controls Added

- Content maps and knowledge trails are stored in PostgreSQL control-plane tables only; Neo4j/Qdrant remain inactive for this phase.
- Content maps and knowledge trails require group/subgroup and cannot be moved across group/subgroup by reposting an existing slug or id.
- Real Postgres upserts now have explicit update-by-id paths plus atomic group/subgroup guards on slug conflicts for canvas boards, content maps and knowledge trails.
- Content map root notes must exist in the same workspace, group and subgroup.
- Knowledge trail steps validate note, document and canvas references against the trail group/subgroup.
- Client-supplied trail step ids are rejected unless the id already belongs to the route trail.
- List endpoints are covered for subgroup isolation so MOCs/trails do not leak across subgroup filters.

## Phase 9 Risk Controls Added

- Inbox and daily notes are stored in PostgreSQL control-plane tables only; Neo4j/Qdrant remain inactive for this phase.
- Inbox items with scoped sources require group/subgroup and validate note, document, canvas, trail and content map sources before persistence.
- Inbox `conversation` and `suggestion` sources validate tenant, workspace and user scope before persistence.
- Existing inbox items cannot be moved or descope-muted by reusing `inbox_item_id` with missing or different group/subgroup.
- Inbox open-list filters are covered for subgroup isolation.
- Daily notes are created through classified Markdown notes and cannot be moved across group/subgroup after creation.
- Daily note slugs are checked before markdown creation so a daily note cannot hijack or move an existing non-daily note.
- Daily note auto-pending collection only includes open inbox items from the same group/subgroup.

## Phase 10 Risk Controls Added

- Agent Builder and Agent Studio remain PostgreSQL control-plane only; Neo4j/Qdrant are not activated in this phase.
- Builder sessions keep `agent_builder` model settings separate from runtime agent model settings.
- Published/runtime agents only accept enabled model settings with `usage` `chat` or `agent`.
- Model settings, agent configs, builder sessions and context budgets cannot be moved across tenant/workspace scope by reusing ids.
- Agent Builder publish requires explicit human approval and readiness validation before creating or enabling an agent.
- Context budgets validate agent/model scope, token windows and cost caps before persistence.
- Query runtime enforces prompt/context ceilings before RAG and applies `QueryParam.max_total_tokens`/entity/relation caps.
- Cost-limited query budgets require `max_context_tokens` and pricing metadata so retrieved context cannot be underpriced.
- Cost-limited agent queries reserve/debit the LLM usage ledger before RAG through a Postgres transaction plus advisory lock.
- Successful non-cost budgeted agent queries append scoped ledger rows; blocked queries do not create extra debits.
- Concurrency coverage proves two same-agent queries under a one-request daily limit produce one RAG call and one ledger row.
- Subagents Hegel, Peirce and Russell reaudited Fase 10 with no P0/P1 blockers after fixes.

## Phase 11 Risk Controls Added

- Context calculator is backend/control-plane only; Neo4j/Qdrant are not activated in this phase.
- Estimates validate workspace, agent, group, subgroup and explicit document scope before computing token windows.
- Estimates report query, history, agent prompt, document, chunk and reserved response token slices plus overflow and available capacity.
- Budget caps override larger model windows in calculator output and are covered by exact overflow invariants.
- Runtime query now keeps `top_k` and `chunk_top_k` aligned with calculator assumptions.
- Runtime query accepts the same group/subgroup/document contract, validates it, and fails closed until the data plane explicitly supports scoped retrieval filters.
- Agent `reserved_response_tokens` is enforced through model function limits or the query is blocked before RAG.
- Ollama local/private completion maps `max_tokens` to `options.num_predict`, preserving any lower existing limit.
- OpenAPI contracts pin query scope fields and context calculator request/response fields.
- Subagents Hegel, Peirce and Russell reaudited Fase 11 with no P0/P1 blockers after fixes.

## Phase 12 Risk Controls Added

- Ledger and cost summaries remain PostgreSQL control-plane only; Neo4j/Qdrant are not activated in this phase.
- `/little-bull/costs/summary` reports total, month, last 7 days and today plus breakdowns by user, agent, model, group/subgroup and operation.
- The endpoint requires audit-read permission and scopes all reads by tenant/workspace before aggregation.
- `little_bull_llm_usage_ledger` now has nullable `group_id` and `subgroup_id` columns plus a group-scope index.
- Scoped summaries include both new scoped columns and legacy metadata-only scoped ledger rows to avoid under-reporting after migration.
- Agent query ledger payloads carry group/subgroup/document scope when scoped retrieval becomes data-plane enabled.
- Non-reservation ledger appends now use a transaction and per-workspace advisory lock before computing the previous hash.
- Cost-budget reservations also acquire the ledger-chain lock before computing the previous hash, keeping the append-only chain linear.
- Summary tests cover workspace decoys, period windows, actual-vs-estimated fallback, by-agent/model/user/group/operation and legacy metadata-only scope.
- Subagents Hegel, Peirce and Russell reaudited Fase 12 with no P0/P1 blockers after fixes.

## Phase 13 Risk Controls Added

- The Obsidian-like graph endpoint is a dedicated PostgreSQL control-plane route and does not call the LightRAG data plane.
- Graph reads are workspace-bounded even when the requested graph scope is `global`.
- Group and subgroup scopes are validated before graph assembly; subgroup scope requires both `group_id` and `subgroup_id`.
- Nodes are composed from note, document and trail control-plane registries; backlinks and trail steps become typed edges with origin filters.
- Central-node focus returns a one-hop graph snapshot plus chat-context metadata without opening an LLM/chat session.
- In-memory clusters are derived from the filtered graph response, not from Neo4j/Qdrant.
- Backlink creation now validates canvas, trail, content map, conversation and agent references instead of treating them as implicitly scope-compatible.
- Conversation saves now fail closed when a supplied conversation id already belongs to another tenant/workspace/user before messages are rewritten.
- Tests cover no-data-plane graph assembly, subgroup isolation, group/workspace views, central-node focus, origin filters, missing graph refs and scoped conversation upsert guards.
- Subagents Hegel, Peirce and Russell audited Fase 13; P0/P1 findings were either fixed or reduced to already-documented non-blocking legacy-route risk.

## Phase 14 Risk Controls Added

- Operational Chat now has a server-owned envelope at `/little-bull/operational-chat` and `/little-bull/chat/operational`.
- The response returns visible context, token/cost estimate metadata, sources, optional saved conversation, optional note and optional suggestion.
- Agent selection is validated server-side before conversation persistence; missing or disabled agents cannot be saved into chat history.
- Conversation saves no longer require a data-plane attachment and now persist an immutable `scope_snapshot` with group, subgroup and document scope.
- Conversation upserts fail closed when tenant, workspace, user or scope snapshot differs before any messages are deleted/reinserted.
- Transform-to-note requires group/subgroup before RAG and creates Markdown notes through the scoped note service with conversation provenance.
- Transform-to-suggestion creates a pending suggestion with conversation/source metadata and does not send anything externally.
- Cross-subgroup document scope fails before RAG and before conversation/note mutation.
- Operational chat still uses the existing query data plane for answer generation; no new Neo4j/Qdrant services are started or globally configured in this phase.
- Tests cover OpenAPI contracts, operational chat context/cost/sources, conversation save, note/suggestion transforms, invalid agent saves, scope snapshot guards and failed-scope no-mutation behavior.

## Phase 15 Risk Controls Added

- The Curator Agent is exposed as pending review suggestions through `/little-bull/curator/suggestions`.
- Curator outputs are stored as `curator_suggestion` inbox items, not as direct graph/document mutations.
- Supported suggestion kinds are backlink, content map/MOC, subgroup, conversation-to-note and canvas-to-dossier.
- Backlink suggestions validate source/target references and reject cross-group/subgroup edges before persistence.
- Conversation-to-note suggestions require a scoped saved conversation and inherit its group/subgroup scope.
- Canvas-to-dossier suggestions validate the source canvas and inherit board group/subgroup scope.
- Apply is explicitly blocked with `409` while human-review application is not implemented, so no graph-critical mutation can run silently.
- Tests cover route contracts, pending suggestions, no direct backlink/MOC/dossier mutation, apply blocking and cross-scope rejection.

## Phase 16 Risk Controls Added

- Legal extraction runs are PostgreSQL control-plane only; this phase does not call DataJud, TPU, Neo4j, Qdrant or external enrichment.
- `/little-bull/legal/extractions` supports create/list plus scoped get and human review endpoints.
- Create validates workspace, group, subgroup and document registry scope before persistence.
- Cross-subgroup documents and source references pointing at a different document are rejected before any run is stored.
- Source references are mandatory and must include a locator, page, chunk, span or paragraph marker for provenance.
- `schema_version` is pinned to `legal-matter/v1`; empty extraction payloads are rejected.
- Runs always start with `review_status=pending` and `requires_human_review=true`; callers cannot pre-approve a run on create.
- Review records reviewer, status, timestamp and error message while preserving the human-review requirement.
- Schema contract advertises legal entities, review policy and no-external-enrichment placeholders for future DataJud/TPU work.
- Tests cover route contracts, create/list/get/review, provenance requirements, cross-scope rejection, unsupported schema, empty payload rejection and no RAG/data-plane calls.

## Phase 17 Risk Controls Added

- Dossier exports remain PostgreSQL/control-plane based and do not send files to external systems.
- `/little-bull/dossiers`, `/little-bull/dossiers/{id}` and `/little-bull/dossiers/{id}/export` expose dossier listing, fetch and file export.
- Supported export formats are TXT, Markdown, DOCX and XLSX.
- Internal dossier exports apply PII redaction before producing response bodies.
- External dossier exports return `pending_approval` and no file body until a matching human LGPD/export approval is approved.
- Approval metadata binds dossier id, format, destination, tenant, workspace, group, subgroup, content refs, redaction policy and LGPD requirement to prevent payload drift.
- Approved external exports transition the approval to executed and audit the export with `approval_id`.
- Legal extraction `review_status=approved` is not treated as export approval; external dossier export still requires separate LGPD/export approval.
- Legal extraction approval now requires approval-decision permission in addition to document upload authority.
- Tests cover route contracts, PII masking, pending external approval, approved XLSX export, approval drift rejection and legal-review-not-export-approval behavior.

## Phase 18 Risk Controls Added

- UI Premium stays inside `LittleBullPreview` and existing `/little-bull` routing; no auth, env or data-plane behavior was changed.
- The sidebar exposes the product surface for Dashboard, Workspaces, Groups, Subgroups, Documents, Notes, Inbox, Daily Notes, Canvas, MOCs, Trails, Graph, Chat, Agent Builder, Assistants, Models, Costs, Jobs, Legal, Reports, Activity, Audit, Approvals and Admin.
- Premium pages reuse real backend-backed state where endpoints exist: documents, activity, assistants, approvals, audit, knowledge bases, models, agents, conversations, costs, dossiers and legal extractions.
- Dossier download actions call the Phase 17 export endpoint and only request internal exports from the UI slice.
- Frontend API client now has typed wrappers for cost summary, dossiers, dossier export and legal extractions.
- Sidebar overflow is scrollable to avoid hidden navigation on smaller desktop heights.
- Validation passed for TypeScript, Bun tests, ESLint and production build.

## Phase 18 Residual Risks / Correction Plan

- CLOSED in Phase 20: classified upload now requires selected group/subgroup in the UI and sends backend-required `group_id` and `subgroup_id` query params.
- REDUCED in Phase 21: Premium navigation permissions and classified-upload state are now covered by pure Bun tests, and a Playwright route-mocked visual smoke validates the premium shell on mobile, tablet and desktop.
- Remaining workflow depth is tracked as product scope rather than blocker: several Premium pages are first-surface operational panels over shared real data rather than full dedicated CRUD workflows.

## Phase 19 QA Evidence

- Backend/enterprise lint passed with `uv run ruff check lightrag/llm/ollama.py lightrag_enterprise tests_enterprise lightrag/api/lightrag_server.py scripts/little_bull_phase3_pilot.py scripts/little_bull_phase3_inventory.py`.
- PostgreSQL migration passed with `.env` loaded without printing secret values.
- `./scripts/test.sh tests_enterprise -q` passed with 184 passed and 4 skipped.
- `./scripts/test.sh tests -q` passed with 794 passed and 32 skipped when rerun with Bash 5 on PATH; first run failed only because `/bin/bash` is Bash 3.2 on macOS.
- Frontend gates passed: `bunx tsc --noEmit`, `bun test`, `bun run lint`, and `bun run build`.
- SpecOps validate passed with 0 issues and SpecOps eval passed 10/10.
- `git diff --check` passed.
- Local tri-bank containers were observed healthy for Postgres, Neo4j and Qdrant.
- No global `NEO4J_WORKSPACE`, `QDRANT_WORKSPACE` or `POSTGRES_WORKSPACE` variables were present in the process environment.
- Secret scan over touched code/docs found only test placeholders, environment-variable references and localStorage token reads; no real `.env` or provider secret value was exposed.

## Phase 19 Residual Risks / Correction Plan

- Do not declare READY yet. Remaining release-hardening item: live upload/index/query smoke against the intended clean workspace.
- CLOSED in Phase 20: direct setup execution now re-execs Bash 4+ when available, and the interactive setup test helper prepends the resolved Bash 4+ directory for tests that invoke `bash` directly.
- Base cleanup still requires explicit user confirmation with exact delete targets; no data or volumes were deleted in this run.

## Phase 20 Risk Controls Added

- Document upload in the Premium UI now loads workspace knowledge groups and subgroups with documents.
- Upload is disabled until the operator has permission and selects both a group and a subgroup.
- Subgroup choices are filtered by the selected group, and changing workspace or group clears stale subgroup selection.
- `uploadLittleBullDocument` now requires `workspace_id`, `group_id` and `subgroup_id` at the TypeScript API boundary and sends all three to `/little-bull/documents/upload`.
- Frontend document/upload response types now include `group_id`, `subgroup_id` and `registry_document_id`, matching backend contracts.
- A Bun API unit test verifies classified upload params, multipart file payload and progress mapping.
- `scripts/setup/setup.sh` now re-execs through Bash 4+ for direct invocation on macOS when available.
- Interactive setup tests resolve the same Bash 4+ family and make direct `bash` subprocesses prefer it.
- Validation passed for ruff, migration, enterprise tests, full backend tests, frontend TypeScript/tests/lint/build, SpecOps validate/eval, `git diff --check`, tri-bank health and workspace override env absence.

## Phase 20 Residual Risks / Correction Plan

- CLOSED in Phase 21: dedicated no-new-dependency UI logic tests and Playwright route-mocked screenshots now cover the Premium UI risk without requiring a separate MSW stack.
- Do not declare READY yet. Remaining release-hardening item is a live upload/index/query smoke against the intended clean workspace.
- Base cleanup still requires explicit user confirmation with exact delete targets; no knowledge-base data, volumes or legacy indexes were deleted in this run.

## Phase 21 Risk Controls Added

- Premium UI state moved into pure helpers for classified upload readiness, subgroup filtering, stale selection cleanup, permission checks, page access and fallback page selection.
- `LittleBullPreview` now consumes those helpers instead of duplicating permission/navigation logic inside the component.
- Bun tests cover classified upload gating, subgroup filtering, stale selection cleanup, master/wildcard access, blocked premium pages and fallback page behavior.
- Document listing remains available with `little_bull.documents.read` even when the user lacks `little_bull.areas.read`; classified-upload taxonomy loading is separately gated by `areas.read` to match backend permissions.
- Playwright is now a declared frontend dev dependency with `bun run test:visual`.
- Playwright visual smoke mocks only local API routes, uses a fake non-secret JWT, does not call backend services and writes screenshots to `/tmp`.
- Visual smoke passed for mobile, tablet and desktop and produced `/tmp/trag-little-bull-premium-mobile.png`, `/tmp/trag-little-bull-premium-tablet.png` and `/tmp/trag-little-bull-premium-desktop.png`.
- `bun test` remains scoped to Bun-compatible tests by naming the Playwright file `*.e2e.ts` and configuring Playwright `testMatch`.

## Phase 21 Residual Risks / Correction Plan

- CLOSED in Phase 21: live upload/index/query smoke passed in an additive workspace on a temporary local server using the current worktree.
- CLOSED in Phase 21: strict tri-bank data-plane smoke passed with PGKV/PGDocStatus, Neo4j and Qdrant storage enabled end-to-end.
- Do not declare READY yet. Remaining release policy is final human review of the accumulated gates and any cleanup decision with explicit target confirmation.
- Live smoke created additive diagnostic workspaces/documents, including `phase21_smoke_820a9f72e3b4` and `phase21_tribank_44657da3ed86`; do not clean these without a separate explicit target list and confirmation.

## Phase 21 Live Smoke Evidence

- Temporary server ran on `127.0.0.1:9631` with current worktree routes, then was stopped.
- Additive workspace `phase21_smoke_820a9f72e3b4` was created; no existing data was deleted or reset.
- Classified upload used a new group/subgroup and queued a text document.
- Indexing reached processed status for document `doc-1588f7b35aed62f7a69230c53c9b711a`.
- Naive query returned one retrieval reference proving the uploaded document was used.

## Phase 21 Strict Tri-bank Smoke Evidence

- Temporary server ran on `127.0.0.1:9632` with current worktree routes and target storage overrides, then was stopped.
- Storage configuration used PGKV/PGDocStatus for PostgreSQL, Neo4j graph storage and Qdrant vector storage without setting global workspace override variables.
- Additive workspace `phase21_tribank_44657da3ed86` was created; no existing data was deleted or reset.
- Classified upload used a new group/subgroup and queued a text document.
- Indexing reached processed status for document `doc-974a377549c90fa3d2434ec663c8b1f4`.
- Naive query returned one retrieval reference, proving the uploaded document was used through the tri-bank data-plane path.
