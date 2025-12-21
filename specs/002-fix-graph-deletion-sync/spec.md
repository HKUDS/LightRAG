# Feature Specification: Fix Graph Not Updating After Document Deletion

**Feature Branch**: `002-fix-graph-deletion-sync`
**Created**: 2025-12-21
**Status**: Draft
**Input**: User description: "Fix graph not updating after document deletion. After DELETE /documents/delete_document, the knowledge graph retains same entities/relations."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Document Deletion Reflects in Graph (Priority: P1)

As a user, when I delete a document from the knowledge base, I expect the knowledge graph to immediately reflect the removal of entities and relationships that were exclusively derived from that document.

**Why this priority**: This is the core bug fix. Without it, the knowledge graph becomes inconsistent with the actual document corpus, leading to incorrect query results and user confusion.

**Independent Test**: Can be tested by uploading a document, verifying entity count, deleting the document, and confirming entity count decreases proportionally.

**Acceptance Scenarios**:

1. **Given** a knowledge graph with 1244 nodes derived from multiple documents, **When** I delete a document that contributed 112 unique entities, **Then** the graph should show approximately 1132 nodes (1244 - 112) after deletion completes.

2. **Given** a document with entities that are also referenced by other documents, **When** I delete that document, **Then** only entities exclusive to the deleted document should be removed; shared entities should remain.

3. **Given** a document deletion in progress, **When** I query the `/graphs` endpoint after deletion completes, **Then** the response should reflect the updated node and edge counts.

---

### User Story 2 - Cache Invalidation Before Rebuild (Priority: P1)

As a system operator, when a document is deleted, the LLM extraction cache for that document's chunks must be invalidated before any rebuild operation occurs, preventing stale data from restoring deleted entities.

**Why this priority**: This is the root cause of the bug. Stale cache entries cause deleted entities to be recreated during the rebuild phase.

**Independent Test**: Can be tested by deleting a document with `delete_llm_cache: true` and verifying that rebuild does not restore any entities from the deleted document.

**Acceptance Scenarios**:

1. **Given** a document with cached LLM extraction results, **When** I delete the document, **Then** the cache entries for all chunks of that document must be invalidated before the rebuild phase begins.

2. **Given** 29 cached chunk extractions where 27 belong to a deleted document, **When** the rebuild process runs, **Then** only the 2 valid (non-deleted) chunk extractions should be processed.

3. **Given** a rebuild operation reading from cache, **When** it encounters a cache entry for a deleted chunk, **Then** that cache entry must be skipped or return empty results.

---

### User Story 3 - Atomic Graph Operations (Priority: P2)

As a system, during document deletion, graph modification operations (remove_nodes, remove_edges) must complete atomically without being interrupted by disk reloads that could restore deleted data.

**Why this priority**: This prevents race conditions where the graph is reloaded from disk mid-deletion, causing in-memory changes to be lost.

**Independent Test**: Can be tested by monitoring graph state during deletion and verifying no intermediate reload occurs between remove operations and persistence.

**Acceptance Scenarios**:

1. **Given** a deletion operation in progress with remove_nodes() called, **When** another process triggers a graph update flag, **Then** the current deletion operation must complete its in-memory changes before any reload occurs.

2. **Given** remove_nodes() deletes 112 entities from the in-memory graph, **When** index_done_callback() is called, **Then** the persisted graph file must contain 112 fewer nodes than before deletion.

3. **Given** concurrent graph operations during deletion, **When** the deletion completes, **Then** logs must show consistent node counts (before > after).

---

### Edge Cases

- What happens when a document has no associated entities or chunks?
  - Deletion should complete successfully with appropriate messaging.

- What happens when all chunks of a document fail cache lookup?
  - Rebuild should skip gracefully without errors; entities should remain deleted.

- What happens when deletion is interrupted mid-operation?
  - System should either complete atomically or rollback to consistent state.

- What happens when the same entity exists across multiple documents?
  - Entity should only be deleted when ALL source documents are removed.

- What happens when the graph file is locked by another process during persistence?
  - System should retry or queue the persistence operation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST update the in-memory graph immediately when remove_nodes() or remove_edges() is called, without requiring explicit persistence.

- **FR-002**: System MUST invalidate LLM cache entries for all chunks belonging to a deleted document BEFORE the rebuild_knowledge_from_chunks() function is invoked.

- **FR-003**: System MUST prevent graph reload from disk (via _get_graph()) during an active deletion operation sequence.

- **FR-004**: System MUST persist the updated graph to disk after all deletion operations complete, reflecting the actual in-memory state.

- **FR-005**: System MUST log the graph node/edge count before and after deletion operations, with counts accurately reflecting the changes.

- **FR-006**: System MUST ensure that rebuild operations only process cache entries for chunks that still exist in the system.

- **FR-007**: System MUST handle the `delete_llm_cache: true` parameter to remove cached extraction results for deleted document chunks.

- **FR-008**: System MUST maintain referential integrity between entities, relationships, and their source chunks during deletion.

### Key Entities

- **Document**: The primary content unit being deleted; contains metadata and references to chunks.

- **Chunk**: Text segments derived from documents; each chunk has associated LLM cache entries.

- **Entity (Node)**: Knowledge graph nodes extracted from chunks; linked to source_ids indicating origin chunks.

- **Relationship (Edge)**: Knowledge graph edges connecting entities; also linked to source_ids.

- **LLM Cache Entry**: Cached extraction results keyed by chunk_id; contains extracted entities and relationships.

- **Graph Storage**: In-memory graph structure with disk persistence; subject to reload from disk.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After deleting a document with N unique entities, the graph node count must decrease by exactly N (within 5% tolerance for shared entities).

- **SC-002**: The `/graphs` endpoint must return updated counts within 5 seconds of deletion completion.

- **SC-003**: Zero entities from deleted documents should be restored during rebuild operations.

- **SC-004**: Graph persistence logs must show "Writing graph with X nodes" where X < previous count after any document deletion.

- **SC-005**: All existing document deletion tests must continue to pass (backward compatibility).

- **SC-006**: New regression tests must verify graph count decreases after deletion with 100% pass rate.

## Assumptions

- The current deletion logic correctly identifies which entities are exclusive to a document vs. shared.
- Graph operations (remove_node, remove_edge) are synchronous and immediately affect the in-memory graph object.
- The `storage_updated` flag mechanism is the primary cause of unintended graph reloads.
- Cache invalidation can be performed before rebuild without affecting other concurrent operations.

## Out of Scope

- Performance optimization of deletion operations.
- Bulk deletion improvements beyond single document deletion.
- Changes to non-NetworkX graph storage implementations.
- UI/WebUI changes for deletion feedback.
