# Feature Specification: Entity Resolution (Linking & Conflict Detection)

**Feature Branch**: `007-entity-resolution`
**Created**: 2026-01-16
**Status**: Draft
**Input**: User description: "Entity Linking (fuzzy deduplication of similar entity names like 'Apple Inc' vs 'Apple Inc.' vs 'Apple') and Conflict Detection (detect contradictions in entity descriptions like conflicting dates or attributions)"

## Clarifications

### Session 2026-01-16

- Q: Comment déterminer le nom canonique lors de la fusion d'entités? → A: Nom le plus long; en cas d'égalité, premier rencontré
- Q: Comment traiter les entités à nom très court (1-2 chars)? → A: Exclure du matching fuzzy (jamais fusionnées automatiquement)
- Q: Comment présenter les conflits dans les descriptions résumées? → A: Mentionner les deux versions avec indication d'incertitude (ex: "Founded in 2003 or 2004 according to different sources")
- Q: Comment traiter les conflits multi-sources (n-way)? → A: Lister toutes les valeurs distinctes (ex: "Founded in 2003, 2004, or 2005 according to different sources")

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Entity Deduplication During Ingestion (Priority: P1)

As a knowledge base administrator, when I ingest documents that mention the same entity with slight naming variations (e.g., "Apple Inc", "Apple Inc.", "Apple"), I want the system to automatically recognize these as the same entity and merge them, so that my knowledge graph remains clean and queries return complete information.

**Why this priority**: Entity fragmentation is the most common cause of incomplete query responses. Without deduplication, the same real-world entity appears multiple times in the graph, splitting its relationships and descriptions across fragments. This directly impacts the quality of context returned to Cleo.

**Independent Test**: Can be fully tested by ingesting a document containing entity name variations and verifying that only one consolidated entity appears in the knowledge graph with all relationships preserved.

**Acceptance Scenarios**:

1. **Given** a document mentioning "Apple Inc" multiple times with variations ("Apple Inc.", "Apple", "APPLE INC"), **When** the document is ingested, **Then** only one entity "Apple Inc" exists in the graph with all descriptions merged
2. **Given** an existing entity "Microsoft Corporation" in the graph, **When** a new document mentions "Microsoft Corp.", **Then** the new information is merged into the existing entity rather than creating a duplicate
3. **Given** two genuinely different entities with similar names (e.g., "Apple Inc" the tech company and "Apple Records" the music label), **When** documents about both are ingested, **Then** they remain as separate entities because their types differ

---

### User Story 2 - Conflict Detection in Entity Descriptions (Priority: P2)

As a knowledge base administrator, when multiple documents provide contradictory information about the same entity (e.g., different founding dates), I want the system to detect and log these conflicts so that I can be aware of data quality issues and the system can handle them appropriately during summarization.

**Why this priority**: Conflicting information silently merged into entity descriptions leads to unreliable responses. Detecting conflicts enables transparency about data quality and allows for more nuanced handling during summarization.

**Independent Test**: Can be fully tested by ingesting documents with contradictory facts about the same entity and verifying that conflicts are logged with details about the contradiction.

**Acceptance Scenarios**:

1. **Given** a document stating "Tesla was founded in 2003" and another stating "Tesla was founded in 2004", **When** both are ingested, **Then** a conflict is detected and logged with both dates and their sources
2. **Given** a document stating "SpaceX was founded by Elon Musk" and another stating "SpaceX was founded by Elon Musk and others", **When** both are ingested, **Then** no conflict is detected (the second is an extension, not a contradiction)
3. **Given** conflicting monetary values ("revenue of $100M" vs "revenue of $150M" for the same period), **When** detected, **Then** the conflict is logged with the specific values and their sources

---

### User Story 3 - Configurable Resolution Behavior (Priority: P3)

As a system operator, I want to configure the sensitivity of entity matching and conflict detection thresholds, so that I can tune the system behavior based on my specific data characteristics.

**Why this priority**: Different knowledge bases have different characteristics. A legal document corpus needs stricter matching than a news aggregation system. Configuration allows operators to optimize for their use case.

**Independent Test**: Can be fully tested by changing configuration values and verifying that matching/detection behavior changes accordingly.

**Acceptance Scenarios**:

1. **Given** entity similarity threshold set to 0.95 (strict), **When** "Apple Inc" and "Apple" are compared, **Then** they are NOT matched (below threshold)
2. **Given** entity similarity threshold set to 0.75 (lenient), **When** "Apple Inc" and "Apple" are compared, **Then** they ARE matched (above threshold)
3. **Given** entity resolution is disabled via configuration, **When** documents are ingested, **Then** no deduplication occurs and entities are stored as-is

---

### Edge Cases

- Short entity names (≤2 chars): Excluded from fuzzy matching, stored as-is
- How does the system handle entities in different languages but referring to the same real-world entity (e.g., "Germany" vs "Deutschland")?
- What happens when conflicting information comes from the same source document?
- N-way conflicts: List all distinct values with uncertainty indication
- What happens when entity matching is ambiguous (similarity score exactly at threshold)?

## Requirements *(mandatory)*

### Functional Requirements

**Entity Linking**

- **FR-001**: System MUST compare newly extracted entity names against existing entities using fuzzy string matching
- **FR-002**: System MUST merge entities when their similarity score exceeds the configured threshold (default: 0.85)
- **FR-003**: System MUST preserve the longest entity name as the canonical name (e.g., "Apple Inc" over "Apple"); in case of equal length, use the first encountered
- **FR-004**: System MUST track all name variants as aliases of the canonical entity
- **FR-005**: System MUST log all entity resolutions with the merged names and similarity scores
- **FR-006**: System MUST consider entity type when determining matches (same-type entities only)
- **FR-007**: System MUST handle case-insensitive matching ("APPLE" matches "Apple")
- **FR-018**: System MUST exclude entity names with 2 or fewer characters from fuzzy matching (stored as-is, never auto-merged)

**Conflict Detection**

- **FR-008**: System MUST detect temporal conflicts (different dates for the same event)
- **FR-009**: System MUST detect attribution conflicts (different actors for the same action)
- **FR-010**: System MUST detect numerical conflicts (different values for the same metric)
- **FR-011**: System MUST log detected conflicts with conflict type, both values, and source references
- **FR-012**: System MUST include both conflicting values with uncertainty indication when summarizing entity descriptions (e.g., "Founded in 2003 or 2004 according to different sources")
- **FR-013**: System MUST assign a confidence score to each detected conflict
- **FR-019**: System MUST handle n-way conflicts (3+ sources) by listing all distinct conflicting values in the summary

**Configuration**

- **FR-014**: System MUST allow enabling/disabling entity resolution via configuration
- **FR-015**: System MUST allow configuring the similarity threshold for entity matching
- **FR-016**: System MUST allow enabling/disabling conflict detection via configuration
- **FR-017**: System MUST allow configuring the conflict confidence threshold

### Key Entities

- **Entity**: A node in the knowledge graph representing a real-world concept (person, organization, location, etc.) with a canonical name, entity type, description, and optional aliases
- **Alias**: An alternative name for an entity that was resolved to the canonical name during deduplication
- **Conflict**: A detected contradiction between two or more descriptions of the same entity, with conflict type, the conflicting values, source references, and confidence score

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Entity fragmentation rate (duplicate entities for same real-world concept) reduced by 80% or more
- **SC-002**: All entity resolutions are logged and traceable to source documents
- **SC-003**: Conflicts are detected with at least 70% precision (detected conflicts are actual contradictions)
- **SC-004**: System maintains ingestion throughput within 10% of current performance
- **SC-005**: Configuration changes take effect without requiring system restart
- **SC-006**: Query responses for deduplicated entities include information from all merged sources

## Assumptions

- Entity type information is available and reliable from the extraction phase
- Fuzzy matching based on token similarity is sufficient for most entity deduplication cases
- Conflicts are logged only (no persistent audit storage) per user preference
- The default similarity threshold of 0.85 provides a good balance between precision and recall
- Entity names in different languages are out of scope for this iteration
