# TRAG Architecture Plan

## Phases

1. Contracts and PostgreSQL schema.
2. Local tri-database infrastructure.
3. Isolated data-plane pilot.
4. Group/subgroup-classified upload.
5. Markdown notes and wikilinks.
6. Backlinks, provenance, canvas, MOCs, trails, agents, costs, legal, exports, UI, QA, and release.

## Boundaries

- Control plane is authoritative.
- Neo4j and Qdrant are reconstructible from governed sources.
- Data-plane pilots must not activate product-wide defaults.

## Acceptance Criteria

- Each phase has tests and rollback notes.
- Destructive cleanup, rebuild, reindex, and release require explicit confirmation or approval policy.
- Graph/vector pilots use workspace-specific identifiers.
