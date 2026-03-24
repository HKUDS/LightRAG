# NebulaGraph Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-grade `NebulaGraphStorage` backend for LightRAG with manual `.env` configuration, native multi-workspace isolation via Nebula `SPACE`, and high-quality `search_labels` via Nebula full-text search.

**Architecture:** Implement a new `BaseGraphStorage` backend in `lightrag/kg/nebula_impl.py` and register it as `NebulaGraphStorage`. Each LightRAG workspace maps to a dedicated Nebula `SPACE`; the backend owns schema initialization, undirected-edge canonicalization, batch graph queries, and full-text search fallback handling.

**Tech Stack:** Python 3.10+, NebulaGraph, `nebula3-python`, LightRAG storage abstractions, pytest, ruff

---

## File Map

### Create

- `lightrag/kg/nebula_impl.py`
- `tests/test_nebula_graph_storage.py`
- `examples/lightrag_openai_nebula_demo.py`

### Modify

- `lightrag/kg/__init__.py`
- `pyproject.toml`
- `requirements-offline.txt`
- `requirements-offline-storage.txt`
- `env.example`
- `README.md`
- `README-zh.md`
- `tests/test_graph_storage.py`

### Existing Code To Follow

- `lightrag/base.py`
- `lightrag/kg/neo4j_impl.py`
- `lightrag/kg/memgraph_impl.py`
- `lightrag/kg/opensearch_impl.py`
- `lightrag/types.py`
- `scripts/test.sh`

### Testing Targets

- `tests/test_nebula_graph_storage.py`
- `tests/test_graph_storage.py`

## Task 1: Register the Backend and Dependency Surface

**Files:**
- Create: `tests/test_nebula_graph_storage.py`
- Modify: `lightrag/kg/__init__.py`
- Modify: `pyproject.toml`
- Modify: `requirements-offline.txt`
- Modify: `requirements-offline-storage.txt`
- Modify: `tests/test_graph_storage.py`

- [ ] **Step 1: Write failing registration/config tests**

```python
from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGE_ENV_REQUIREMENTS, STORAGES


def test_nebula_graph_storage_is_registered():
    assert "NebulaGraphStorage" in STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"]
    assert STORAGES["NebulaGraphStorage"] == ".kg.nebula_impl"


def test_nebula_graph_storage_env_requirements():
    assert STORAGE_ENV_REQUIREMENTS["NebulaGraphStorage"] == [
        "NEBULA_HOSTS",
        "NEBULA_USER",
        "NEBULA_PASSWORD",
    ]
```

- [ ] **Step 2: Run the registration tests to verify they fail**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "registered or env_requirements" -v`
Expected: FAIL with missing `NebulaGraphStorage` registration or missing env requirements.

- [ ] **Step 3: Register the backend and declare dependencies**

Implement these minimal changes:

- Add `"NebulaGraphStorage"` to `GRAPH_STORAGE` implementations in `lightrag/kg/__init__.py`
- Add `STORAGE_ENV_REQUIREMENTS["NebulaGraphStorage"]`
- Add `STORAGES["NebulaGraphStorage"] = ".kg.nebula_impl"`
- Add `nebula3-python` to `offline-storage` in `pyproject.toml`
- Add `nebula3-python` to `requirements-offline.txt` and `requirements-offline-storage.txt`
- Update the supported storage list comment/docstring in `tests/test_graph_storage.py`

- [ ] **Step 4: Re-run the registration tests**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "registered or env_requirements" -v`
Expected: PASS

- [ ] **Step 5: Lint the touched metadata and test files**

Run: `uv run ruff check lightrag/kg/__init__.py tests/test_nebula_graph_storage.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add lightrag/kg/__init__.py pyproject.toml requirements-offline.txt requirements-offline-storage.txt tests/test_graph_storage.py tests/test_nebula_graph_storage.py
git commit -m "feat: register NebulaGraph storage backend"
```

## Task 2: Build the Backend Skeleton and Pure Helpers

**Files:**
- Create: `lightrag/kg/nebula_impl.py`
- Test: `tests/test_nebula_graph_storage.py`

- [ ] **Step 1: Write failing unit tests for space naming and canonical edge behavior**

```python
from lightrag.kg.nebula_impl import _canonical_edge_pair, _normalize_space_name


def test_normalize_space_name_uses_prefix_and_workspace():
    assert _normalize_space_name("lightrag", "hr-prod") == "lightrag__hr_prod"


def test_normalize_space_name_uses_base_for_empty_workspace():
    assert _normalize_space_name("lightrag", "") == "lightrag__base"


def test_canonical_edge_pair_is_undirected():
    assert _canonical_edge_pair("B", "A") == ("A", "B")
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "normalize_space_name or canonical_edge_pair" -v`
Expected: FAIL with import error because `lightrag/kg/nebula_impl.py` does not exist yet.

- [ ] **Step 3: Create `lightrag/kg/nebula_impl.py` with a minimal backend skeleton**

Start with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final

import pipmaster as pm

if not pm.is_installed("nebula3-python"):
    pm.install("nebula3-python")

from ..base import BaseGraphStorage


def _canonical_edge_pair(src: str, tgt: str) -> tuple[str, str]:
    return tuple(sorted((src, tgt)))


def _normalize_space_name(prefix: str, workspace: str) -> str:
    ...


@final
@dataclass
class NebulaGraphStorage(BaseGraphStorage):
    ...
```

- [ ] **Step 4: Implement pure helpers only**

Implement:

- workspace normalization
- collision-safe hash suffix helper
- canonical edge pair helper
- env parsing helpers

Do not implement Nebula I/O yet.

- [ ] **Step 5: Re-run the helper tests**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "normalize_space_name or canonical_edge_pair" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add lightrag/kg/nebula_impl.py tests/test_nebula_graph_storage.py
git commit -m "feat: add NebulaGraph storage skeleton helpers"
```

## Task 3: Implement Connection and Schema Initialization

**Files:**
- Modify: `lightrag/kg/nebula_impl.py`
- Test: `tests/test_nebula_graph_storage.py`

- [ ] **Step 1: Write failing tests for initialization SQL generation and readiness checks**

Use mocking so the test does not require a live Nebula cluster:

```python
async def test_initialize_creates_space_and_schema(mocker):
    storage = build_storage(workspace="finance")
    exec_mock = mocker.AsyncMock()
    mocker.patch.object(storage, "_execute", exec_mock)
    mocker.patch.object(storage, "_wait_for_schema_ready", mocker.AsyncMock())

    await storage._ensure_space_ready()

    assert any("CREATE SPACE IF NOT EXISTS" in call.args[0] for call in exec_mock.await_args_list)
    assert any("CREATE TAG IF NOT EXISTS entity" in call.args[0] for call in exec_mock.await_args_list)
    assert any("CREATE EDGE IF NOT EXISTS relation" in call.args[0] for call in exec_mock.await_args_list)
```

- [ ] **Step 2: Run the initialization tests to verify they fail**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "initialize_creates_space_and_schema" -v`
Expected: FAIL because initialization helpers are missing.

- [ ] **Step 3: Implement Nebula client bootstrapping**

Add:

- connection/session pool construction
- host parsing from `NEBULA_HOSTS`
- user/password loading
- `finalize()` cleanup
- lightweight `_execute()` / `_execute_in_space()` helpers

- [ ] **Step 4: Implement schema initialization**

Implement these methods:

- `_ensure_space_ready()`
- `_create_space_if_needed()`
- `_use_space()`
- `_create_schema_if_needed()`
- `_create_indexes_if_needed()`
- `_wait_for_schema_ready()`
- `_wait_for_index_ready()`

Commands should cover:

- `CREATE SPACE IF NOT EXISTS`
- `USE <space>`
- `CREATE TAG IF NOT EXISTS entity(...)`
- `CREATE EDGE IF NOT EXISTS relation(...)`
- full-text index setup

- [ ] **Step 5: Re-run the mocked initialization tests**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "initialize_creates_space_and_schema" -v`
Expected: PASS

- [ ] **Step 6: Lint the backend**

Run: `uv run ruff check lightrag/kg/nebula_impl.py tests/test_nebula_graph_storage.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add lightrag/kg/nebula_impl.py tests/test_nebula_graph_storage.py
git commit -m "feat: initialize NebulaGraph spaces and schema"
```

## Task 4: Implement Node and Edge CRUD with Undirected Semantics

**Files:**
- Modify: `lightrag/kg/nebula_impl.py`
- Test: `tests/test_nebula_graph_storage.py`
- Test: `tests/test_graph_storage.py`

- [ ] **Step 1: Add failing CRUD-focused Nebula tests**

Cover:

- `upsert_node` / `get_node`
- `upsert_edge` / `get_edge`
- reverse edge lookup returns the same properties
- `delete_node`
- `remove_edges`

Minimal example:

```python
@pytest.mark.integration
@pytest.mark.requires_db
async def test_nebula_edge_reads_are_undirected(storage):
    await storage.upsert_node("A", {"entity_id": "A", "entity_type": "X"})
    await storage.upsert_node("B", {"entity_id": "B", "entity_type": "X"})
    await storage.upsert_edge("B", "A", {"relationship": "rel", "weight": 1.0})
    assert await storage.get_edge("A", "B") == await storage.get_edge("B", "A")
```

- [ ] **Step 2: Run the CRUD tests to verify they fail**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "undirected or get_node or upsert" -v`
Expected: FAIL because CRUD methods still raise or return empty data.

- [ ] **Step 3: Implement node CRUD**

Implement:

- `has_node`
- `get_node`
- `upsert_node`
- `delete_node`

Use `entity_id` as VID and keep `entity_type` as a regular property.

- [ ] **Step 4: Implement canonical undirected edge CRUD**

Implement:

- `has_edge`
- `get_edge`
- `upsert_edge`
- `remove_edges`

Always canonicalize `(src, tgt)` before writing or reading.

- [ ] **Step 5: Run targeted Nebula tests and generic graph CRUD checks**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "undirected or get_node or upsert or remove_edges" -v`
Expected: PASS

Run: `./scripts/test.sh tests/test_graph_storage.py -k "graph_basic or graph_advanced" -v --run-integration`
Expected: PASS for Nebula when `LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage` and Nebula env vars are set.

- [ ] **Step 6: Commit**

```bash
git add lightrag/kg/nebula_impl.py tests/test_nebula_graph_storage.py
git commit -m "feat: implement NebulaGraph node and edge CRUD"
```

## Task 5: Implement Batch Methods and Graph Browsing

**Files:**
- Modify: `lightrag/kg/nebula_impl.py`
- Test: `tests/test_nebula_graph_storage.py`

- [ ] **Step 1: Write failing batch and graph browsing tests**

Cover:

- `get_nodes_batch`
- `node_degrees_batch`
- `get_edges_batch`
- `get_nodes_edges_batch`
- `get_all_labels`
- `get_all_nodes`
- `get_all_edges`
- `get_popular_labels`

- [ ] **Step 2: Run the batch tests to verify they fail**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "batch or popular_labels or get_all_" -v`
Expected: FAIL

- [ ] **Step 3: Implement batch methods with Nebula queries**

Avoid one-by-one fallback for production paths. Implement dedicated query helpers for:

- batch node lookup
- batch edge lookup
- batch degree aggregation
- node-edge adjacency lookup

- [ ] **Step 4: Implement graph listing methods**

Implement:

- `get_node_edges`
- `get_all_labels`
- `get_all_nodes`
- `get_all_edges`
- `get_popular_labels`

Return formats must match existing graph storage expectations.

- [ ] **Step 5: Re-run batch and browsing tests**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "batch or popular_labels or get_all_" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add lightrag/kg/nebula_impl.py tests/test_nebula_graph_storage.py
git commit -m "feat: add NebulaGraph batch and browsing queries"
```

## Task 6: Implement `get_knowledge_graph` and Full-Text Search

**Files:**
- Modify: `lightrag/kg/nebula_impl.py`
- Test: `tests/test_nebula_graph_storage.py`
- Test: `tests/test_graph_storage.py`

- [ ] **Step 1: Write failing tests for subgraph retrieval and label search**

Cover:

- `get_knowledge_graph("*", ...)`
- `get_knowledge_graph("entity", ...)`
- `search_labels` full-text happy path
- `search_labels` fallback path when full-text is unavailable

Example:

```python
async def test_search_labels_falls_back_when_fulltext_unavailable(mocker):
    storage = build_storage()
    mocker.patch.object(storage, "_search_labels_fulltext", side_effect=RuntimeError("no ft"))
    mocker.patch.object(storage, "_search_labels_contains", mocker.AsyncMock(return_value=["Machine Learning"]))
    assert await storage.search_labels("learn", limit=10) == ["Machine Learning"]
```

- [ ] **Step 2: Run the graph retrieval and search tests to verify they fail**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "knowledge_graph or search_labels" -v`
Expected: FAIL

- [ ] **Step 3: Implement `get_knowledge_graph`**

Implement both modes:

- `node_label == "*"`: global truncated graph
- explicit label: bounded-depth subgraph

Convert results into `KnowledgeGraph`, `KnowledgeGraphNode`, and `KnowledgeGraphEdge` using the existing types in `lightrag/types.py`.

- [ ] **Step 4: Implement full-text search with fallback**

Implement:

- `_search_labels_fulltext()`
- `_search_labels_contains()`
- `search_labels()`

Requirements:

- use Nebula full-text search when available
- log explicit warning on fallback
- preserve ranking order: exact > prefix > contains > stable tie-break

- [ ] **Step 5: Re-run targeted tests**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "knowledge_graph or search_labels" -v`
Expected: PASS

Run: `./scripts/test.sh tests/test_graph_storage.py -k "graph_advanced or graph_batch_operations" -v --run-integration`
Expected: PASS for Nebula when integration env is configured.

- [ ] **Step 6: Commit**

```bash
git add lightrag/kg/nebula_impl.py tests/test_nebula_graph_storage.py
git commit -m "feat: add NebulaGraph graph retrieval and label search"
```

## Task 7: Document the Manual Configuration Flow

**Files:**
- Create: `examples/lightrag_openai_nebula_demo.py`
- Modify: `env.example`
- Modify: `README.md`
- Modify: `README-zh.md`

- [ ] **Step 1: Write failing documentation checks**

Add tests or assertions in `tests/test_nebula_graph_storage.py` that verify the new env keys are documented in `env.example` and that README mentions:

- `NebulaGraphStorage`
- `NEBULA_HOSTS`
- full-text dependency on Elasticsearch + Listener
- workspace-to-space mapping

- [ ] **Step 2: Run the documentation checks to verify they fail**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "env_example or readme" -v`
Expected: FAIL because docs are not updated yet.

- [ ] **Step 3: Update docs and example**

Add:

- NebulaGraph env section in `env.example`
- NebulaGraph storage description in both READMEs
- warning that high-quality `search_labels` requires Elasticsearch + Listener
- example showing `graph_storage="NebulaGraphStorage"`

- [ ] **Step 4: Re-run the documentation checks**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -k "env_example or readme" -v`
Expected: PASS

- [ ] **Step 5: Lint or sanity-check docs-adjacent Python example**

Run: `uv run python -m py_compile examples/lightrag_openai_nebula_demo.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add env.example README.md README-zh.md examples/lightrag_openai_nebula_demo.py tests/test_nebula_graph_storage.py
git commit -m "docs: add NebulaGraph configuration guide"
```

## Task 8: Full Verification and Cleanup

**Files:**
- Modify: `lightrag/kg/nebula_impl.py` as needed
- Modify: `tests/test_nebula_graph_storage.py` as needed

- [ ] **Step 1: Run the focused Nebula test suite**

Run: `./scripts/test.sh tests/test_nebula_graph_storage.py -v`
Expected: PASS

- [ ] **Step 2: Run the generic graph storage integration suite against Nebula**

Run: `LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage ./scripts/test.sh tests/test_graph_storage.py -v --run-integration`
Expected: PASS when Nebula, Elasticsearch, and Listener are correctly configured.

- [ ] **Step 3: Run lint on all touched Python files**

Run: `uv run ruff check lightrag/kg/nebula_impl.py tests/test_nebula_graph_storage.py examples/lightrag_openai_nebula_demo.py`
Expected: PASS

- [ ] **Step 4: Perform a manual smoke test with a real workspace**

Suggested env:

```bash
export LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage
export NEBULA_HOSTS=127.0.0.1:9669
export NEBULA_USER=root
export NEBULA_PASSWORD=nebula
export WORKSPACE=smoke_test
```

Smoke-test goals:

- first startup auto-creates the target space
- document insertion creates entities and relations
- `search_labels` returns expected entities
- switching `WORKSPACE` uses a different Nebula `SPACE`

- [ ] **Step 5: Commit final fixes**

```bash
git add lightrag/kg/nebula_impl.py tests/test_nebula_graph_storage.py examples/lightrag_openai_nebula_demo.py env.example README.md README-zh.md
git commit -m "test: verify NebulaGraph backend end to end"
```

- [ ] **Step 6: Prepare merge notes**

Document in the final handoff:

- required external services
- env vars used
- tests that were run
- tests not run and why
- known operational limitations for full-text search

## Notes for Implementation Workers

- Use `@superpowers:test-driven-development` for each task slice.
- Use `@superpowers:verification-before-completion` before claiming the backend is complete.
- Keep `NebulaGraphStorage` focused on graph storage only; do not expand scope into setup-wizard integration.
- Do not model `entity_type` as dynamic Nebula tags.
- Treat undirected edge semantics as a hard compatibility requirement.
