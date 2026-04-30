# TRAG System Spec

## Architecture

- PostgreSQL stores tenants, workspaces, users, permissions, documents, notes, jobs, approvals, audit, costs, versions, legal extractions, backlinks, trails, MOCs, canvas, inbox, and dossiês.
- Neo4j stores graph entities, relationships, clusters, paths, and content maps.
- Qdrant stores embeddings for chunks, notes, entities, relations, conversations, canvas cards, and MOCs.
- LightRAG coordinates ingestion and query.

## Runtime Rules

- `NEO4J_WORKSPACE`, `QDRANT_WORKSPACE`, and `POSTGRES_WORKSPACE` must not be set globally for dynamic workspace routing.
- Feature flags gate graph v2, Qdrant data plane, Obsidian workspace, and clean-base operations.

## Acceptance Criteria

- System contracts exist before UI expansion.
- Tests verify safe defaults, no raw secrets, append-only ledgers, and workspace scoping.
- Local pilots use isolated workspaces and non-destructive rollback.
