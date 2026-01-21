# Research: Hybrid Cross-Document Entity Resolution

**Phase**: 0 - Research
**Date**: 2025-01-21

## Research Tasks Completed

### 1. Entity VDB Availability

**Question**: Does an entity VDB already exist for semantic search?

**Decision**: Yes, `entities_vdb` already exists and is populated during indexing.

**Rationale**:
- Located at `lightrag/lightrag.py:674`
- Uses `BaseVectorStorage` implementation (configurable backend)
- Already workspace-scoped
- Entities are indexed with their names as embeddings

**Alternatives Considered**:
- Create separate VDB for entity resolution → Rejected (unnecessary duplication)
- Use graph-based similarity → Rejected (no semantic understanding)

**Code Reference**:
```python
# lightrag/lightrag.py:674
self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(
    namespace=NameSpace.VECTOR_STORE_ENTITIES,
    workspace=self.workspace,
    ...
)
```

---

### 2. Node Count Implementation

**Question**: How to efficiently count entities in each storage backend?

**Decision**: Add `get_node_count()` abstract method to `BaseGraphStorage` with O(1) implementations.

**Rationale**:
- Called once per document indexing
- Must not add overhead proportional to graph size
- All backends support efficient counting

**Implementation per Backend**:

| Backend | Implementation | Complexity |
|---------|---------------|------------|
| PostgreSQL | `SELECT count(*) FROM nodes WHERE workspace = $1` | O(1) with index |
| MongoDB | `collection.count_documents({})` | O(1) |
| NetworkX | `len(self._graph.nodes())` | O(1) |
| Neo4j | `MATCH (n) RETURN count(n)` | O(1) with count store |

**Alternatives Considered**:
- Maintain separate counter → Rejected (requires sync logic, error-prone)
- Count on demand via `len(get_all_nodes())` → Rejected (O(n) complexity)

---

### 3. VDB Query Interface

**Question**: Does the VDB support type-filtered queries?

**Decision**: The existing VDB interface supports filtering, but entity_type may need to be added to metadata.

**Rationale**:
- `BaseVectorStorage.query()` supports metadata filtering
- Current entity metadata may not include `entity_type` for filtering
- Need to verify and potentially add entity_type to VDB metadata

**Action Required**:
- Verify entity metadata structure in VDB
- Add `entity_type` to metadata if missing (backward compatible)

**Code Reference**:
```python
# Expected query pattern
candidates = await entity_vdb.query(
    query_embedding=entity_embedding[0],
    top_k=top_k,
    filter={"entity_type": entity_type},  # Requires entity_type in metadata
)
```

---

### 4. Similarity Computation Reuse

**Question**: Can we reuse existing similarity computation?

**Decision**: Yes, `compute_entity_similarity()` from `entity_resolution.py` is reusable.

**Rationale**:
- Already implements proper fuzzy matching with rapidfuzz
- Handles edge cases (short names, punctuation)
- Well-tested in current cross-doc resolution

**Code Reference**:
```python
# lightrag/entity_resolution.py
def compute_entity_similarity(name1: str, name2: str) -> float:
    # Uses rapidfuzz for fuzzy matching
    ...
```

---

### 5. Configuration Pattern

**Question**: How should configuration be structured?

**Decision**: Environment variables with constants.py defaults, matching existing patterns.

**Rationale**:
- Consistent with existing configuration style (e.g., `MAX_GRAPH_NODES`)
- No code changes needed for different deployments
- Supports runtime configuration via .env

**Constants to Add**:
```python
# lightrag/constants.py
DEFAULT_CROSS_DOC_RESOLUTION_MODE = "hybrid"
DEFAULT_CROSS_DOC_THRESHOLD_ENTITIES = 5000
DEFAULT_CROSS_DOC_VDB_TOP_K = 10
```

**Environment Variables**:
- `CROSS_DOC_RESOLUTION_MODE` - "full" | "vdb" | "hybrid" | "disabled"
- `CROSS_DOC_THRESHOLD_ENTITIES` - Integer threshold
- `CROSS_DOC_VDB_TOP_K` - Integer top-k candidates

---

### 6. Metrics Logging Pattern

**Question**: How to implement resolution metrics logging?

**Decision**: Use existing PERF logging pattern with optional token_tracker integration.

**Rationale**:
- Consistent with recent performance logging additions (PR #34)
- No additional dependencies
- Supports both debugging and billing analysis

**Implementation**:
```python
logger.info(
    f"PERF cross_doc_resolution mode={mode} "
    f"entities={entities_checked} duplicates={duplicates_found} "
    f"time_ms={elapsed_ms:.1f}"
)
```

---

## Dependencies Identified

| Dependency | Purpose | Status |
|------------|---------|--------|
| rapidfuzz | Fuzzy string matching | Already installed |
| pytest-asyncio | Async test support | Already installed |
| entities_vdb | Semantic similarity search | Already exists |
| BaseGraphStorage | Storage abstraction | Extend with get_node_count() |

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| VDB entity_type metadata missing | Medium | Medium | Verify and add if needed (backward compatible) |
| Threshold too high/low for production | Medium | Low | Configurable via env var; can tune post-deployment |
| VDB precision lower than expected | Low | Medium | Hybrid mode ensures full mode for critical early growth |

## Open Questions Resolved

All clarifications from design.md have been addressed:

1. ~~Entity VDB existence~~ → Confirmed exists
2. ~~get_node_count() performance~~ → O(1) implementations defined
3. ~~Concurrency behavior~~ → Documented as acceptable (eventual switch)
4. ~~Configuration approach~~ → Environment variables chosen
5. ~~Option C (gradual)~~ → Deferred; start with Option A
