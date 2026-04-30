# Little Bull Risk Register

Status date: 2026-04-30

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
