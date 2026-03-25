# NebulaGraph Review Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair NebulaGraphStorage review issues by eliminating hot-path full scans, fixing wildcard graph retrieval, preserving runtime-critical relation fields, and aligning search behavior with canonical `entity_id` semantics.

**Architecture:** Keep the existing Nebula backend shape, but replace unbounded query implementations with caller-bounded variants, make wildcard graph retrieval scale with `max_nodes`, and switch search to `entity_id` full-text plus `name` supplemental recall. Preserve public API return shapes while tightening unit tests so regressions fail early.

**Tech Stack:** Python 3.10+, NebulaGraph, `nebula3-python`, LightRAG storage abstractions, pytest, ruff

---

## File Map

### Modify

- `lightrag/kg/nebula_impl.py`
- `lightrag/utils_graph.py`
- `tests/test_nebula_graph_storage.py`
- `tests/test_description_api_validation.py`

### Verify Against

- `tests/test_graph_storage.py`
- `lightrag/operate.py`
- `lightrag/api/routers/graph_routes.py`
- `lightrag_webui/src/components/graph/GraphLabels.tsx`
- `lightrag_webui/src/hooks/useLightragGraph.tsx`

## Task 1: Lock In Bounded Batch Query Contracts

**Files:**
- Modify: `tests/test_nebula_graph_storage.py`
- Modify: `lightrag/kg/nebula_impl.py`

- [ ] Add helper assertions that reject unbounded `MATCH ... RETURN ...` SQL in hot batch methods.
- [ ] Strengthen tests for:
  - `get_nodes_batch`
  - `node_degrees_batch`
  - `get_edges_batch`
  - `get_nodes_edges_batch`
- [ ] Reimplement these methods so the generated SQL includes caller-bounded filters.
- [ ] Run:

```bash
./scripts/test.sh tests/test_nebula_graph_storage.py -k "get_nodes_batch or node_degrees_batch or get_edges_batch or get_nodes_edges_batch" -v
```

Expected: PASS

## Task 2: Replace Heavy Existence Queries

**Files:**
- Modify: `tests/test_nebula_graph_storage.py`
- Modify: `lightrag/kg/nebula_impl.py`

- [ ] Add failing tests proving `has_node` does not call `get_node`.
- [ ] Add failing tests proving `has_edge` does not call `get_edge`.
- [ ] Reimplement both methods as bounded existence probes with `LIMIT 1`.
- [ ] Run:

```bash
./scripts/test.sh tests/test_nebula_graph_storage.py -k "has_node or has_edge" -v
```

Expected: PASS

## Task 3: Rework Wildcard Graph Retrieval

**Files:**
- Modify: `tests/test_nebula_graph_storage.py`
- Modify: `lightrag/kg/nebula_impl.py`

- [ ] Add failing tests proving wildcard retrieval does not depend on `get_all_nodes/get_all_edges`.
- [ ] Reimplement `_build_global_knowledge_graph()` to use:
  1. `get_popular_labels`
  2. `get_nodes_batch`
  3. `get_nodes_edges_batch`
  4. `get_edges_batch`
- [ ] Run:

```bash
./scripts/test.sh tests/test_nebula_graph_storage.py -k "wildcard or popular_labels" -v
```

Expected: PASS

## Task 4: Persist Runtime-Critical Relation Fields

**Files:**
- Modify: `tests/test_nebula_graph_storage.py`
- Modify: `lightrag/kg/nebula_impl.py`

- [ ] Add failing tests requiring relation `keywords` and `file_path` to survive:
  - `upsert_edge` -> `get_edge`
  - `get_edges_batch`
  - `get_all_edges`
- [ ] Extend Nebula relation schema additively with `ALTER EDGE relation ADD (...)`.
- [ ] Update edge write/read helpers and `_EDGE_FIELDS`.
- [ ] Run:

```bash
./scripts/test.sh tests/test_nebula_graph_storage.py -k "upsert_edge or get_edges_batch or get_all_edges or initialize_creates_space_and_schema" -v
```

Expected: PASS

## Task 5: Align Search With Canonical `entity_id`

**Files:**
- Modify: `tests/test_nebula_graph_storage.py`
- Modify: `tests/test_description_api_validation.py`
- Modify: `lightrag/kg/nebula_impl.py`
- Modify: `lightrag/utils_graph.py`

- [ ] Add failing unit tests for candidate ranking:
  - `entity_id` exact/prefix/contains outrank `name` matches
  - name-only hits still return canonical `entity_id`
  - duplicate candidates collapse by `entity_id`
- [ ] Change the managed Nebula full-text tag index from `entity(name)` to `entity(entity_id)`.
- [ ] Add a supplemental `name` match query and merge it with full-text `entity_id` candidates.
- [ ] Re-rank merged candidates using explicit entity-first tiers.
- [ ] Add a rename test proving `name` syncs to the new `entity_id` by default.
- [ ] Update rename logic in `lightrag/utils_graph.py`.
- [ ] Run:

```bash
./scripts/test.sh tests/test_nebula_graph_storage.py tests/test_description_api_validation.py -k "search_labels or rank_search_candidate_rows or rename_syncs_name_to_new_entity_id" -v
```

Expected: PASS

## Task 6: Final Verification

**Files:**
- Verify: `lightrag/kg/nebula_impl.py`
- Verify: `lightrag/utils_graph.py`
- Verify: `tests/test_nebula_graph_storage.py`
- Verify: `tests/test_description_api_validation.py`

- [ ] Run:

```bash
./scripts/test.sh tests/test_nebula_graph_storage.py -v
./scripts/test.sh tests/test_description_api_validation.py -v
uv run ruff check lightrag/kg/nebula_impl.py lightrag/utils_graph.py tests/test_nebula_graph_storage.py tests/test_description_api_validation.py
```

Expected: PASS

- [ ] If a live Nebula environment is available, additionally run:

```bash
LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage ./scripts/test.sh tests/test_graph_storage.py -v --run-integration
```

Expected: PASS against a real Nebula cluster.

