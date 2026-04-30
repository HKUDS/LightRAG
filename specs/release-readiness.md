# TRAG Release Readiness

## Readiness Policy

Do not declare READY until product gates pass for control plane, graph plane, vector plane, LightRAG ingestion/query, UI, costs, agents, legal review, exports, and rollback.

## Rollback

- Disable graph v2, Qdrant data plane, and Obsidian workspace flags.
- Rebuild Neo4j and Qdrant from PostgreSQL/governed sources when needed.
- Preserve snapshots before destructive rebuild.

## Acceptance Criteria

- Release notes include validation evidence.
- Rollback is documented and tested for the changed surface.
- No cross-workspace leakage, secret exposure, or unreviewed legal output is accepted.
