# NebulaGraph Review Follow-up Design

- Date: 2026-03-25
- Status: Approved in conversation
- Scope: NebulaGraph review follow-up for performance, correctness, and search contract alignment

## Summary

This follow-up design addresses the highest-priority NebulaGraph review findings without reopening the full March 24 delivery scope.

The implemented focus areas are:

- remove request-time full scans from hot Nebula batch paths
- stop wildcard graph retrieval from materializing the full graph in Python
- persist the relation fields that current LightRAG merge/rebuild flows actually consume
- replace heavy existence checks with lightweight bounded probes
- implement the approved search contract:
  - canonical return value is always `entity_id`
  - full-text search is built on `entity_id`
  - `name` is used as a supplemental recall path

## In Scope

- Review item `#1`: batch query full scans
- Review item `#2`: repeated full scans in graph browsing
- Review item `#3` in its runtime-critical subset:
  - relation `keywords`
  - relation `file_path`
- Review items `#4` and `#10` as search-contract alignment
- Review item `#8`: heavy `has_node` / `has_edge`

## Out of Scope

- Full cleanup of all review items `#1-#10`
- `env.example` and setup-wizard naming cleanup
- dead-code cleanup in `_wait_for_space_ready`
- dataclass cleanup
- full parity for deferred schema fields such as:
  - `relation_id`
  - `updated_at`
  - node `updated_at`

## Root Causes

### Batch methods lost database-side boundedness

`get_nodes_batch`, `node_degrees_batch`, `get_edges_batch`, and `get_nodes_edges_batch` were adapted to live-cluster-friendly `MATCH` queries, but the adaptation removed caller-bounded filtering and replaced it with Python-side filtering.

### Tests encoded the wrong success condition

Nebula unit tests verified:

- one query was issued
- returned payload shape was correct

They did not verify:

- requested IDs or canonical pairs were present in the query
- unbounded `MATCH ... RETURN ...` forms were rejected

### Search semantics drifted away from identity semantics

The original intent was to search canonical entity identifiers. The implementation instead built full-text on `name`, while LightRAG edit and rename flows are centered on `entity_id`.

## Design Decisions

### 1. Hot batch methods must be bounded at the database layer

For hot runtime paths, "single query" is not enough. Queries must be scoped by:

- requested node IDs
- requested frontier IDs
- requested canonical edge pairs

Full-scan-then-filter is explicitly forbidden.

### 2. Wildcard global graph retrieval must scale with `max_nodes`

`get_knowledge_graph("*")` must not depend on:

- `get_all_nodes()`
- `get_all_edges()`

for its primary truncation path.

Instead, it should:

1. obtain top candidate labels
2. fetch only the selected node payloads
3. fetch only adjacency for selected nodes
4. fetch only induced edge payloads

### 3. Relation persistence only needs to cover runtime-critical fields in this round

This follow-up persists the relation fields current LightRAG logic actively reads:

- `source_id`
- `target_id`
- `relationship`
- `description`
- `keywords`
- `weight`
- `file_path`

Relation schema changes must be additive so existing Nebula spaces remain usable.

### 4. `has_node` and `has_edge` must be lightweight probes

These methods must answer boolean existence using bounded queries and must not call:

- `get_node()`
- `get_edge()`

### 5. Search contract uses `entity_id` as the canonical identity

The approved search behavior is:

- return `entity_id`
- rank `entity_id` matches ahead of `name`-only matches
- keep `name` as supplemental recall only

### 6. Full-text index lives on `entity_id`

The implementation uses Nebula full-text on `entity_id`, not `name`.

Reason:

- it aligns with LightRAG identity semantics
- it avoids returning results that look unrelated to the canonical label

`name` is handled through a supplemental query path and merged into the final ranked candidate set.

### 7. Rename defaults must keep `name` aligned unless caller explicitly overrides it

When entity identity is renamed and the caller does not intentionally provide a separate display `name`, the persisted node should keep:

- `entity_id == new_entity_name`
- `name == new_entity_name`

This prevents Nebula-specific search drift after rename operations.

## Search Ranking Rules

Final ranking tiers are:

1. exact `entity_id`
2. prefix `entity_id`
3. contains `entity_id`
4. exact `name`
5. prefix `name`
6. contains `name`

Stable tie-break:

- shorter `entity_id`
- then `entity_id` ascending

## Acceptance Criteria

- No hot Nebula batch path uses full-graph scan plus Python filtering.
- Wildcard graph retrieval no longer depends on `get_all_nodes/get_all_edges`.
- Nebula relation reads/writes preserve `keywords` and `file_path`.
- `has_node` and `has_edge` are bounded existence probes.
- `search_labels` returns `entity_id`, prefers `entity_id` matches, and still surfaces `name`-only hits.
- Rename paths keep `name` aligned by default.

## References

- Original design: [2026-03-24-nebulagraph-support-design.md](/root/project/LightRAG/.worktrees/nebulagraph-review-followup/docs/superpowers/specs/2026-03-24-nebulagraph-support-design.md)
- Original plan: [2026-03-24-nebulagraph-support.md](/root/project/LightRAG/.worktrees/nebulagraph-review-followup/docs/superpowers/plans/2026-03-24-nebulagraph-support.md)

