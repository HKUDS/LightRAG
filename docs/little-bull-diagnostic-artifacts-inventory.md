# Little Bull Diagnostic Artifacts Inventory

Status date: 2026-05-01

This inventory is read-only documentation for possible future cleanup. No cleanup has been executed.
It contains no secrets, tokens, passwords, `.env` values or connection strings.

## Rule

Do not delete, drop, truncate, reset, prune, or remove any item in this file without explicit confirmation that names the exact targets.

## Local Services Observed

- `trag-phase2-postgres-1`: healthy, loopback Postgres service.
- `trag-phase2-neo4j-1`: healthy, loopback Neo4j service.
- `trag-phase2-qdrant-1`: healthy, loopback Qdrant service.

These containers are diagnostic. Stopping them without volume deletion is lower risk than removing containers or volumes, but still should be coordinated if active tests depend on them.

## Filesystem Artifacts

Candidate directories under `rag_storage/`:

- `rag_storage/phase21_smoke_18311564788c`: empty directory observed.
- `rag_storage/phase21_smoke_820a9f72e3b4`: 10 files observed.
- `rag_storage/reindex_smoke_933a0228`: empty directory observed.
- `rag_storage/reindex_smoke_ac241841`: empty directory observed.

Candidate directories under `inputs/`:

- `inputs/phase21_smoke_18311564788c`
- `inputs/phase21_smoke_820a9f72e3b4`
- `inputs/phase21_smoke_820a9f72e3b4/__enqueued__`
- `inputs/phase21_tribank_44657da3ed86`
- `inputs/phase21_tribank_44657da3ed86/__enqueued__`
- `inputs/reindex_smoke_933a0228`
- `inputs/reindex_smoke_ac241841`

Known document in `rag_storage/phase21_smoke_820a9f72e3b4`:

- `doc-1588f7b35aed62f7a69230c53c9b711a`

Visual smoke screenshots are temporary and may or may not exist depending on `/tmp` retention:

- `/tmp/trag-little-bull-premium-mobile.png`
- `/tmp/trag-little-bull-premium-tablet.png`
- `/tmp/trag-little-bull-premium-desktop.png`

## Qdrant Artifacts

Collections observed on the local diagnostic Qdrant service:

- `lightrag_vdb_chunks_openai_text_embedding_3_small_1536d`
- `lightrag_vdb_entities_openai_text_embedding_3_small_1536d`
- `lightrag_vdb_relationships_openai_text_embedding_3_small_1536d`
- `lightrag_vdb_chunks_phase3_fake_local_phase3_pilot_20260430130751_16d`
- `lightrag_vdb_entities_phase3_fake_local_phase3_pilot_20260430130751_16d`
- `lightrag_vdb_relationships_phase3_fake_local_phase3_pilot_20260430130751_16d`
- `lightrag_vdb_chunks_phase3_fake_local_phase3_pilot_20260430130805_16d`
- `lightrag_vdb_entities_phase3_fake_local_phase3_pilot_20260430130805_16d`
- `lightrag_vdb_relationships_phase3_fake_local_phase3_pilot_20260430130805_16d`
- `lightrag_vdb_chunks_phase3_fake_local_phase3_pilot_20260430130847_16d`
- `lightrag_vdb_entities_phase3_fake_local_phase3_pilot_20260430130847_16d`
- `lightrag_vdb_relationships_phase3_fake_local_phase3_pilot_20260430130847_16d`

Workspace-specific counts observed for `phase21_tribank_44657da3ed86`:

- `lightrag_vdb_chunks_openai_text_embedding_3_small_1536d`: 1 point with `workspace_id=phase21_tribank_44657da3ed86`.
- `lightrag_vdb_entities_openai_text_embedding_3_small_1536d`: 5 points with `workspace_id=phase21_tribank_44657da3ed86`.
- `lightrag_vdb_relationships_openai_text_embedding_3_small_1536d`: 0 points with `workspace_id=phase21_tribank_44657da3ed86`.

Do not delete whole shared OpenAI collections just to remove one workspace. A future cleanup should use workspace-filtered point deletion only after explicit approval.

## Neo4j Artifacts

Indexes observed in the local diagnostic Neo4j service:

- `entity_id_fulltext_idx_phase21_smoke_18311564788c`
- `entity_id_fulltext_idx_phase21_smoke_820a9f72e3b4`
- `entity_id_fulltext_idx_phase21_tribank_44657da3ed86`

Workspace-property counts for `phase21_tribank_44657da3ed86` returned 0 nodes and 0 relationships. The index remains a cleanup candidate.

## PostgreSQL / Control Plane Artifacts

Known diagnostic workspaces/documents created during live smokes:

- workspace `phase21_smoke_820a9f72e3b4`
  - document `doc-1588f7b35aed62f7a69230c53c9b711a`
- workspace `phase21_tribank_44657da3ed86`
  - document `doc-974a377549c90fa3d2434ec663c8b1f4`

PostgreSQL cleanup must be planned from table metadata and foreign-key order before any `DELETE`, `DROP`, or `TRUNCATE`.
Prefer a read-only SQL inventory first, then a dry-run transaction script that rolls back, then an approved cleanup transaction.

## Safe Next Step For Cleanup

If cleanup is approved later, first generate a read-only cleanup packet containing:

- exact filesystem paths;
- exact Qdrant collection names and workspace-filter predicates;
- exact Neo4j index names and any node/relationship predicates;
- exact PostgreSQL tables and row counts by workspace/document id;
- rollback notes or rebuild path for each plane.

No destructive command should be run before that packet is reviewed and approved.
