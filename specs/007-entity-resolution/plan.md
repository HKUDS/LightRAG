# Implementation Plan: Entity Resolution (Linking & Conflict Detection)

**Branch**: `007-entity-resolution` | **Date**: 2026-01-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/007-entity-resolution/spec.md`

## Summary

Implement entity deduplication via fuzzy string matching and conflict detection for contradictory entity descriptions. The system will consolidate entities with similar names (e.g., "Apple Inc" vs "Apple") during ingestion and detect/log conflicts in entity descriptions (e.g., different founding dates).

**Technical Approach**:
- New module `lightrag/entity_resolution.py` for fuzzy matching using `rapidfuzz` library
- New module `lightrag/conflict_detection.py` for pattern-based conflict detection
- Integration point: `operate.py` after entity extraction, before graph storage
- Configuration via environment variables (ENABLE_ENTITY_RESOLUTION, ENTITY_SIMILARITY_THRESHOLD, etc.)

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: rapidfuzz (new), existing: networkx, asyncpg, pydantic
**Storage**: PostgreSQL (via postgres_impl.py), existing entity/graph storage
**Testing**: pytest, pytest-asyncio
**Target Platform**: Linux server (Docker), Windows dev
**Project Type**: Single Python package (lightrag)
**Performance Goals**: Ingestion throughput within 10% of current (SC-004)
**Constraints**: No breaking changes to public API, workspace isolation maintained
**Scale/Scope**: Typical entity counts: 1K-100K per workspace

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. API Backward Compatibility | ✅ PASS | New feature is additive; no public API changes. Configuration via env vars with backward-compatible defaults. |
| II. Workspace/Tenant Isolation | ✅ PASS | Entity resolution operates within workspace scope only. No cross-workspace data access. |
| III. Explicit Server Configuration | ✅ PASS | New config in env.example: ENABLE_ENTITY_RESOLUTION, ENTITY_SIMILARITY_THRESHOLD, ENABLE_CONFLICT_DETECTION |
| IV. Multi-Workspace Test Coverage | ✅ PASS | Tests will verify isolation: entity matching within workspace only |

**Gate Result**: PASS - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/007-entity-resolution/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # N/A (no new API endpoints)
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
lightrag/
├── entity_resolution.py    # NEW: EntityResolver class
├── conflict_detection.py   # NEW: ConflictDetector class
├── operate.py              # MODIFY: Inject resolution after entity extraction
├── prompt.py               # MODIFY: Add conflict-aware summarization prompt
├── base.py                 # MODIFY: Add aliases field to entity schema (optional)
└── constants.py            # MODIFY: New configuration constants

tests/
├── test_entity_resolution.py   # NEW: Unit tests for EntityResolver
├── test_conflict_detection.py  # NEW: Unit tests for ConflictDetector
└── test_entity_integration.py  # NEW: Integration tests

env.example                 # MODIFY: Add new configuration variables
pyproject.toml              # MODIFY: Add rapidfuzz dependency
```

**Structure Decision**: Single project structure maintained. Two new modules added to existing lightrag package.

## Complexity Tracking

No constitutional violations - no complexity justification needed.

---

## Phase 0: Research Summary

See [research.md](research.md) for detailed findings.

### Key Decisions

1. **Fuzzy Matching Library**: `rapidfuzz` - Fast C++ implementation, MIT license, no heavy dependencies
2. **Matching Algorithm**: Token Set Ratio - handles word order variations and partial matches
3. **Conflict Detection**: Regex-based pattern matching for dates, numbers, and attributions
4. **Integration Point**: After `_merge_nodes_then_upsert` in operate.py, before graph storage

---

## Phase 1: Design Summary

### Data Model

See [data-model.md](data-model.md) for full entity schemas.

**Key Changes**:
- Entity gains optional `aliases: list[str]` field
- New `ConflictInfo` dataclass for detected conflicts
- Resolution logs via standard Python logging

### API Contracts

No new REST API endpoints. Feature is internal to ingestion pipeline.

### Quickstart

See [quickstart.md](quickstart.md) for configuration and usage guide.

---

## Implementation Phases

### Phase A: Entity Resolution Module (P1)

1. Create `lightrag/entity_resolution.py`
   - `EntityResolver` class with configurable threshold
   - `resolve()` method for single entity lookup
   - `consolidate_entities()` for batch processing
   - Token Set Ratio matching via rapidfuzz

2. Add `rapidfuzz>=3.0.0` to pyproject.toml dependencies

3. Create unit tests `tests/test_entity_resolution.py`
   - Test fuzzy matching accuracy
   - Test threshold behavior
   - Test case-insensitive matching
   - Test short name exclusion (≤2 chars)
   - Test canonical name selection (longest)

### Phase B: Integration with operate.py (P1)

1. Modify `operate.py` around line 2460
   - After entity collection, before graph storage
   - Call `EntityResolver.consolidate_entities(all_nodes)`
   - Preserve entity type checking

2. Add configuration loading from global_config
   - `enable_entity_resolution` (default: True)
   - `entity_similarity_threshold` (default: 0.85)

3. Add logging for entity resolutions

### Phase C: Conflict Detection Module (P2)

1. Create `lightrag/conflict_detection.py`
   - `ConflictDetector` class
   - `ConflictInfo` dataclass
   - Pattern detection: dates, numbers, attributions
   - Confidence scoring

2. Create unit tests `tests/test_conflict_detection.py`
   - Test date conflict detection
   - Test number conflict detection
   - Test attribution conflict detection
   - Test non-conflict cases (extensions)

### Phase D: Conflict Integration (P2)

1. Modify `operate.py` around line 1720
   - Before `_handle_entity_relation_summary`
   - Detect conflicts in description list
   - Log conflicts with details

2. Modify `prompt.py`
   - Add `summarize_with_conflicts` prompt template
   - Include uncertainty indication format

3. Modify summary call to use conflict-aware prompt when conflicts detected

### Phase E: Configuration & Documentation (P3)

1. Add environment variables to `env.example`
2. Add constants to `lightrag/constants.py`
3. Create integration tests
4. Update CLAUDE.md if needed

---

## Verification Plan

### Automated Tests

```bash
cd c:\Users\cleme\...\LightRAG-MT
pytest tests/test_entity_resolution.py tests/test_conflict_detection.py -v
pytest tests/ -v  # Full test suite
```

### Manual Tests

1. **Entity Resolution Test**:
   - Ingest document with: "Apple Inc", "Apple Inc.", "Apple", "APPLE INC"
   - Query knowledge graph
   - Verify: single entity with all descriptions merged

2. **Conflict Detection Test**:
   - Ingest: "Tesla founded in 2003" then "Tesla founded in 2004"
   - Check logs for conflict warning
   - Query entity description
   - Verify: uncertainty indication in response

3. **Configuration Test**:
   - Set ENTITY_SIMILARITY_THRESHOLD=0.95
   - Verify stricter matching behavior
   - Set ENABLE_ENTITY_RESOLUTION=false
   - Verify no deduplication occurs

---

## Dependencies

| Dependency | Version | Purpose | License |
|------------|---------|---------|---------|
| rapidfuzz | >=3.0.0 | Fuzzy string matching | MIT |

No other new dependencies required.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| False positive matches | Entity data corruption | Conservative default threshold (0.85), entity type checking, logging |
| Performance degradation | Slower ingestion | Lazy initialization, batch processing, profile during tests |
| Conflict detection false positives | Noise in logs | Confidence threshold, precision-focused patterns |
