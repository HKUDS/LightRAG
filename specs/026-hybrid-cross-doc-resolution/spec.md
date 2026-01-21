# Feature Specification: Hybrid Cross-Document Entity Resolution

**Feature Branch**: `026-hybrid-cross-doc-resolution`
**Created**: 2025-01-21
**Status**: Draft
**Input**: User description: "Hybrid cross-document entity resolution with VDB-assisted matching"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Fast Document Indexing at Scale (Priority: P1)

As a system operator managing a large knowledge graph with thousands of entities, I want document indexing to remain fast regardless of graph size, so that I can continue adding documents without experiencing exponential slowdowns.

**Why this priority**: This is the core problem. Current indexing takes ~10 hours for 300 documents with 11K entities, making the system unusable for production workloads at scale.

**Independent Test**: Can be fully tested by indexing a batch of documents into a graph with >5000 entities and measuring processing time per document.

**Acceptance Scenarios**:

1. **Given** a knowledge graph with 10,000 existing entities, **When** I index a new document with 50 entities, **Then** cross-document resolution completes in under 5 seconds (vs current ~7.6 seconds).
2. **Given** a knowledge graph with 50,000 existing entities, **When** I index a new document, **Then** cross-document resolution time does not increase proportionally to graph size.
3. **Given** a large graph, **When** the system processes cross-document resolution, **Then** entity deduplication quality remains above 90% compared to full matching.

---

### User Story 2 - Maximum Quality for Small Graphs (Priority: P2)

As a user with a new or small knowledge graph, I want the system to use the most precise entity matching algorithm, so that I get maximum quality deduplication during the initial growth phase.

**Why this priority**: New users with small graphs should not sacrifice quality for performance they don't need yet.

**Independent Test**: Can be fully tested by indexing documents into a graph with <5000 entities and verifying full fuzzy matching is used.

**Acceptance Scenarios**:

1. **Given** a knowledge graph with fewer than the configured threshold entities, **When** I index a new document, **Then** the system uses full fuzzy matching for maximum precision.
2. **Given** a knowledge graph that grows past the threshold, **When** I index the next document, **Then** the system automatically switches to the faster matching mode.

---

### User Story 3 - Configurable Resolution Behavior (Priority: P3)

As a system administrator, I want to configure the entity resolution mode and thresholds via environment variables, so that I can tune the system for my specific use case without code changes.

**Why this priority**: Different deployments have different quality/performance trade-offs.

**Independent Test**: Can be fully tested by setting environment variables and verifying the system respects them.

**Acceptance Scenarios**:

1. **Given** environment variable `CROSS_DOC_RESOLUTION_MODE=full`, **When** I index documents, **Then** the system always uses full matching regardless of graph size.
2. **Given** environment variable `CROSS_DOC_RESOLUTION_MODE=vdb`, **When** I index documents, **Then** the system always uses fast matching regardless of graph size.
3. **Given** environment variable `CROSS_DOC_THRESHOLD_ENTITIES=2000`, **When** the graph has 2001 entities, **Then** the system switches to fast matching.
4. **Given** environment variable `CROSS_DOC_RESOLUTION_MODE=disabled`, **When** I index documents, **Then** no cross-document resolution is performed.

---

### User Story 4 - Observability and Metrics (Priority: P4)

As a system operator, I want visibility into which resolution mode is being used and its performance metrics, so that I can monitor system behavior and tune configuration.

**Why this priority**: Operators need to understand and tune system behavior in production.

**Independent Test**: Can be fully tested by indexing documents and checking logs for resolution metrics.

**Acceptance Scenarios**:

1. **Given** any indexing operation, **When** cross-document resolution runs, **Then** logs include: mode used, entities checked, duplicates found, and processing time.
2. **Given** billing is enabled, **When** resolution triggers entity summarization, **Then** token usage is tracked for billing.

---

### Edge Cases

- What happens when the graph is exactly at the threshold (e.g., 5000 entities)?
  - System uses full mode (threshold is exclusive: `< threshold` = full, `>= threshold` = VDB)
- What happens when multiple documents are indexed in parallel near the threshold?
  - Both may use full mode; switch is "eventual" not instant (acceptable behavior)
- What happens when entity embedding fails during VDB lookup?
  - Fall back to including entity without deduplication; log warning
- What happens when VDB returns no candidates for an entity?
  - Entity is kept as-is (no match found)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support four resolution modes: `full`, `vdb`, `hybrid`, and `disabled`
- **FR-002**: In `hybrid` mode, system MUST automatically switch from full to VDB matching when entity count exceeds threshold
- **FR-003**: System MUST allow configuration of resolution mode via `CROSS_DOC_RESOLUTION_MODE` environment variable
- **FR-004**: System MUST allow configuration of entity threshold via `CROSS_DOC_THRESHOLD_ENTITIES` environment variable (default: 5000)
- **FR-005**: System MUST allow configuration of VDB candidate count via `CROSS_DOC_VDB_TOP_K` environment variable (default: 10)
- **FR-006**: In VDB mode, system MUST only compare new entities against top-K semantically similar existing entities
- **FR-007**: System MUST maintain entity type filtering (only match entities of the same type)
- **FR-008**: System MUST log resolution metrics (mode, entities checked, duplicates found, time) for each indexing operation
- **FR-009**: System MUST provide a method to count existing entities with O(1) complexity

### Key Entities

- **Entity Resolution Mode**: Configuration determining which algorithm to use (full/vdb/hybrid/disabled)
- **Entity Count Threshold**: Number of entities at which hybrid mode switches from full to VDB matching
- **Resolution Metrics**: Performance and quality data captured during resolution (mode, count, time, duplicates)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Document indexing time for cross-document resolution is under 1 second per document for graphs with up to 50,000 entities
- **SC-002**: Entity deduplication precision in VDB mode is at least 90% compared to full matching
- **SC-003**: System correctly switches modes at threshold boundary in 100% of test cases
- **SC-004**: All configuration options are respected when set via environment variables
- **SC-005**: Resolution metrics are logged for every indexing operation
- **SC-006**: No regression in deduplication quality for graphs under the threshold
