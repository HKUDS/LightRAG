# NebulaGraph Support Design

- Date: 2026-03-24
- Status: Approved in conversation
- Scope: Production-grade graph storage support for LightRAG using NebulaGraph

## Summary

This design adds a new `GRAPH_STORAGE` backend named `NebulaGraphStorage` for LightRAG. The first production-facing version targets:

- Manual `.env` configuration
- External self-managed NebulaGraph cluster
- High-quality `search_labels` support via NebulaGraph full-text search
- Native multi-workspace support by mapping each LightRAG workspace to a dedicated Nebula `SPACE`

This version does not include setup wizard integration or bundled Docker templates.

## Goals

- Add `NebulaGraphStorage` as a first-class `GRAPH_STORAGE` implementation.
- Preserve current `BaseGraphStorage` behavior and API compatibility.
- Support production-safe workspace isolation.
- Support high-quality label search with Nebula full-text indexing.
- Keep LightRAG query, graph browsing, and editing flows working without upstream API changes.

## Non-Goals

- No `scripts/setup/` integration in the first version.
- No bundled NebulaGraph Docker Compose templates.
- No attempt to unify Nebula deployment lifecycle with LightRAG deployment.
- No migration tooling from existing graph backends in the first version.

## Current Constraints

LightRAG graph storage implementations must satisfy the full `BaseGraphStorage` contract, not only node and edge upserts. The backend must support:

- Node and edge CRUD
- Degree queries
- Batch fetch helpers
- Label listing and search
- Knowledge-graph subgraph retrieval
- Drop and cleanup behavior

This is required by the runtime, graph API routes, and integration tests.

## Proposed Backend

- Class name: `NebulaGraphStorage`
- Module path: `lightrag/kg/nebula_impl.py`
- Storage type: `GRAPH_STORAGE`

Registration updates are required in:

- `lightrag/kg/__init__.py`
- `env.example`
- `README.md`
- `README-zh.md`
- Optional example script(s)

## Workspace Model

### Decision

Use `space-per-workspace`.

Each LightRAG workspace maps to a dedicated Nebula `SPACE`. This is the preferred production model because it aligns with NebulaGraph's native isolation semantics and minimizes accidental cross-workspace reads.

### Mapping

Recommended mapping:

`space_name = <NEBULA_SPACE_PREFIX>__<sanitized_workspace>`

Rules:

- Empty workspace maps to `base`
- Invalid characters are normalized
- If normalization causes collisions, append a short hash suffix

Examples:

- `""` -> `lightrag__base`
- `"hr-prod"` -> `lightrag__hr_prod`
- `"book/cn"` -> `lightrag__book_cn`

## Environment Contract

### Required

- `LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage`
- `NEBULA_HOSTS`
- `NEBULA_USER`
- `NEBULA_PASSWORD`

### Recommended Defaults

- `NEBULA_SPACE_PREFIX=lightrag`
- `NEBULA_PARTS_NUM=10`
- `NEBULA_REPLICA_FACTOR=1`
- `NEBULA_VID_TYPE=FIXED_STRING(256)`
- `NEBULA_ENABLE_SSL=false`
- `NEBULA_FULLTEXT_INDEX_NAME=entity_id_ft`
- `NEBULA_FULLTEXT_ANALYZER=standard`
- `NEBULA_WORKSPACE_STRATEGY=space_per_workspace`

## Schema Design

Each workspace `SPACE` owns a fixed schema.

### Tag

`entity`

Fields:

- `entity_id string`
- `entity_type string`
- `description string`
- `keywords string`
- `source_id string`
- `file_path string`
- `created_at int64`
- `updated_at int64`
- `truncate string`

### Edge

`relation`

Fields:

- `relation_id string`
- `source_id string`
- `target_id string`
- `relationship string`
- `description string`
- `keywords string`
- `weight double`
- `file_path string`
- `created_at int64`
- `updated_at int64`

### Indexes

- Tag/property index on `entity.entity_id`
- Edge/property index suitable for relation lookup
- Full-text tag index on `entity.entity_id`

## Full-Text Search

### Decision

Use NebulaGraph full-text search for production-quality `search_labels`.

### Operational Requirement

NebulaGraph full-text search depends on Elasticsearch and Listener. This must be documented as a required external dependency for high-quality label search.

### Runtime Strategy

- Primary path: full-text search through Nebula full-text index
- Fallback path: degraded `CONTAINS`-style search with explicit warning logs

### Ranking

Return ranking should mimic current user expectations:

- Exact match first
- Prefix match second
- Contains match third
- Stable tie-break by label length and label ascending

## Initialization and Schema Management

`initialize()` is responsible for ensuring the active workspace space is ready for use.

Recommended sequence:

1. Create connection pool or session pool
2. Resolve workspace to target `SPACE`
3. `CREATE SPACE IF NOT EXISTS`
4. Wait for metadata visibility
5. `USE <space>`
6. `CREATE TAG IF NOT EXISTS entity(...)`
7. `CREATE EDGE IF NOT EXISTS relation(...)`
8. Create indexes and full-text index definitions
9. Trigger rebuild if needed
10. Wait for index readiness

### Design Notes

- Initialization must be idempotent.
- Schema readiness must be explicitly checked.
- The backend should not assume space, tag, edge, or index availability.
- Per-process caching may skip repeated schema checks only after first successful initialization.

## Data Model Semantics

### Nodes

- `VID` is the `entity_id`
- `entity_type` remains a regular property
- Dynamic tag creation per entity type is explicitly not used

This avoids schema explosion and matches how the LightRAG core primarily consumes node data.

### Undirected Edge Semantics

LightRAG requires graph edges to behave as undirected at the abstraction level.

NebulaGraph uses directed edges, so the backend will enforce a canonical ordering:

- Normalize `(src, tgt)` to `tuple(sorted([src, tgt]))`
- Persist one canonical edge record
- Resolve reads and deletes through the canonical pair

This preserves:

- `get_edge(A, B) == get_edge(B, A)`
- reverse-edge deletion behavior
- stable batch edge lookup behavior

## BaseGraphStorage Method Mapping

### Must Implement

- `has_node`
- `has_edge`
- `node_degree`
- `edge_degree`
- `get_node`
- `get_edge`
- `get_node_edges`
- `upsert_node`
- `upsert_edge`
- `delete_node`
- `remove_nodes`
- `remove_edges`
- `get_all_labels`
- `get_knowledge_graph`
- `get_all_nodes`
- `get_all_edges`
- `get_popular_labels`
- `search_labels`

### Batch Methods to Optimize

Although defaults exist in `BaseGraphStorage`, the following should be overridden for production performance:

- `get_nodes_batch`
- `node_degrees_batch`
- `get_edges_batch`
- `get_nodes_edges_batch`

## Query Strategy

### get_all_labels

Return all `entity.entity_id` values in ascending order.

### get_popular_labels

Return `entity_id` sorted by node degree descending.

### search_labels

- Use full-text search when available
- Use degraded matching when full-text is unavailable

### get_knowledge_graph

Two modes must match current LightRAG semantics:

- `node_label == "*"`: return a truncated global graph view
- specific label: return a bounded-depth subgraph centered on that entity

Initial implementation priority:

- Correctness
- Stable truncation behavior
- Compatibility with existing graph API expectations

Query-plan optimization can be deferred.

## API Compatibility

No API contract change is expected.

The existing routes should continue working:

- graph label list
- popular labels
- label search
- graph retrieval
- entity existence checks

The backend change should remain internal to storage selection.

## Testing Strategy

### Unit Tests

- Workspace to space-name normalization
- Collision-safe space naming
- Canonical undirected edge normalization
- Result conversion into `KnowledgeGraphNode` and `KnowledgeGraphEdge`
- Full-text enabled vs degraded search branching

### Integration Tests

Reuse and extend the generic graph storage integration path:

- `tests/test_graph_storage.py`

Run with:

- `LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage`

Coverage expectations:

- Node CRUD
- Edge CRUD
- Reverse-edge consistency
- Degree queries
- Batch operations
- Label listing
- Popular labels
- Knowledge graph retrieval
- Drop behavior

### Nebula-Specific Tests

Add:

- `tests/test_nebula_graph_storage.py`

Key cases:

- Auto-create space per workspace
- Cross-workspace isolation
- Index readiness wait behavior
- Full-text success path
- Full-text dependency missing or unavailable
- Drop current workspace space

## Documentation Changes

Update:

- `env.example`
- `README.md`
- `README-zh.md`

Document clearly:

- Required Nebula credentials
- Multi-workspace equals multi-space
- Full-text search requires Elasticsearch + Listener
- Manual deployment only in first release

## Risks

### High

- Full-text search availability depends on Elasticsearch + Listener
- Schema/index readiness can lag after creation
- Multi-workspace initialization can race under concurrent first access

### Medium

- Undirected semantics need careful canonicalization
- Batch query performance may degrade if implemented as repeated point lookups

### Low

- Treating `entity_type` as a property instead of a dynamic label/tag

## Rollout Plan

### Phase 1

- Add backend registration
- Implement connection and initialization skeleton
- Implement space and schema creation

### Phase 2

- Implement node and edge CRUD
- Implement batch helpers
- Implement graph browsing methods

### Phase 3

- Implement full-text search path
- Implement degraded fallback behavior
- Add Nebula-specific tests

### Phase 4

- Update docs and examples
- Run integration verification

## Acceptance Criteria

- `LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage` works with manual `.env` configuration
- Multiple LightRAG workspaces do not share graph data
- `search_labels` uses Nebula full-text search when configured and available
- Generic graph storage tests pass for NebulaGraph
- Graph API behavior remains compatible without route changes

## Deferred Work

- Setup wizard integration
- Docker templates for NebulaGraph, Elasticsearch, and Listener
- Migration utilities from Neo4j or other graph backends
- Performance tuning beyond baseline production correctness
