# Implementation Plan: Hybrid Cross-Document Entity Resolution

**Branch**: `026-hybrid-cross-doc-resolution` | **Date**: 2025-01-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/026-hybrid-cross-doc-resolution/spec.md`

## Summary

Optimize cross-document entity resolution to scale with large knowledge graphs by implementing a hybrid approach: use full fuzzy matching for small graphs (maximum precision) and VDB-assisted matching for large graphs (O(log m) vs O(m) complexity). Configuration is environment-driven with automatic mode switching at a configurable threshold.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: rapidfuzz (existing), asyncio, existing LightRAG storage abstractions
**Storage**: PostgreSQL, MongoDB, NetworkX, Neo4j (via existing `BaseGraphStorage` implementations)
**Testing**: pytest with async support (`pytest-asyncio`)
**Target Platform**: Linux server (existing deployment)
**Project Type**: Single Python package (lightrag/)
**Performance Goals**: <1 second per document for cross-doc resolution with up to 50K entities
**Constraints**: <5% performance regression for single-workspace mode, workspace isolation maintained
**Scale/Scope**: 50,000+ entities per workspace

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. API Backward Compatibility** | PASS | New parameters have defaults maintaining current behavior (hybrid mode = current full mode for small graphs) |
| **II. Workspace and Tenant Isolation** | PASS | VDB queries are already workspace-scoped; `get_node_count()` will be workspace-scoped |
| **III. Explicit Server Configuration** | PASS | All settings via environment variables with documented defaults |
| **IV. Multi-Workspace Test Coverage** | PASS | Tests will verify workspace isolation in VDB queries |

**Security Requirements**:
- `get_node_count()` uses workspace-scoped queries - PASS
- No cross-workspace data exposure in VDB queries - PASS

**Performance Standards**:
- Multi-workspace operation must not degrade single-workspace by >5% - PASS (hybrid defaults to full mode behavior)
- Workspace resolution adds <1ms latency - PASS (existing workspace scoping)

## Project Structure

### Documentation (this feature)

```text
specs/026-hybrid-cross-doc-resolution/
├── spec.md              # Feature specification
├── design.md            # Technical design (already exists)
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (internal contracts)
└── tasks.md             # Phase 2 output (from /speckit.tasks)
```

### Source Code (repository root)

```text
lightrag/
├── base.py              # Add get_node_count() abstract method
├── constants.py         # Add CROSS_DOC_* constants
├── entity_resolution.py # Existing resolver (reference)
├── lightrag.py          # Add config parameters
├── operate.py           # Implement hybrid resolution logic
└── kg/
    ├── postgres_impl.py   # Implement get_node_count()
    ├── mongo_impl.py      # Implement get_node_count()
    ├── networkx_impl.py   # Implement get_node_count()
    ├── neo4j_impl.py      # Implement get_node_count()
    └── ...

tests/
├── test_cross_doc_resolution.py  # NEW: hybrid mode tests
└── test_entity_resolution.py     # Existing tests (verify no regression)
```

**Structure Decision**: Single Python package structure matches existing codebase. No new directories needed - changes integrate into existing modules.

## Complexity Tracking

> No violations - all changes align with existing patterns and constitution principles.

| Aspect | Approach | Justification |
|--------|----------|---------------|
| Storage abstraction | Add method to existing `BaseGraphStorage` | Follows established pattern |
| Configuration | Environment variables in `constants.py` | Matches existing config style |
| Mode switching | Simple threshold comparison | Option A (fixed threshold) per design decision |
