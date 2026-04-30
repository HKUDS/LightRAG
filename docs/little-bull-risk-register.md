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
