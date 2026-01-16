# Data Model: Entity Resolution

**Feature**: 007-entity-resolution
**Date**: 2026-01-16

## Entity Schema Changes

### Current Entity Structure

```python
# From base.py - TextChunkSchema and entity storage
{
    "entity_name": str,           # Primary identifier
    "entity_type": str,           # PERSON, ORGANIZATION, LOCATION, etc.
    "description": str,           # Merged descriptions from all sources
    "source_id": str,             # Source chunk reference
    # ... other fields
}
```

### Enhanced Entity Structure

```python
{
    "entity_name": str,           # Canonical name (longest variant)
    "entity_type": str,           # Entity type for matching constraint
    "description": str,           # Merged descriptions
    "source_id": str,             # Source chunk reference
    "aliases": list[str],         # NEW: Alternative names that resolved to this entity
    # ... other fields unchanged
}
```

**Note**: The `aliases` field is optional. Implementation may choose to:
1. Store aliases in entity metadata (preferred)
2. Log aliases without persistence (simpler)

---

## New Data Structures

### EntityResolver Class

```python
@dataclass
class EntityResolver:
    """Resolves and consolidates similar entity names."""

    similarity_threshold: float = 0.85
    min_name_length: int = 3

    # Internal state
    canonical_names: dict[str, str] = field(default_factory=dict)
    alias_groups: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def resolve(self, entity_name: str, entity_type: str) -> str:
        """
        Returns canonical name for the given entity.

        Args:
            entity_name: Name to resolve
            entity_type: Entity type (used for matching constraint)

        Returns:
            Canonical name (may be same as input if no match found)
        """
        ...

    def consolidate_entities(
        self,
        all_nodes: dict[str, list[dict]]
    ) -> dict[str, list[dict]]:
        """
        Consolidates entities with similar names.

        Args:
            all_nodes: Dict mapping entity_name -> list of entity records

        Returns:
            Consolidated dict with merged entities under canonical names
        """
        ...
```

### ConflictInfo Dataclass

```python
@dataclass
class ConflictInfo:
    """Represents a detected conflict between entity descriptions."""

    entity_name: str              # Entity with conflicting info
    conflict_type: str            # "temporal", "attribution", "numerical"
    value_a: str                  # First conflicting value
    value_b: str                  # Second conflicting value
    source_a: str                 # Source of first value (chunk_id or doc_id)
    source_b: str                 # Source of second value
    confidence: float             # Detection confidence (0.0-1.0)
    context_a: str                # Sentence containing value_a
    context_b: str                # Sentence containing value_b

    def to_log_message(self) -> str:
        """Format for logging."""
        return (
            f"Conflict[{self.conflict_type}] in '{self.entity_name}': "
            f"'{self.value_a}' vs '{self.value_b}' "
            f"(confidence: {self.confidence:.2f})"
        )

    def to_prompt_context(self) -> str:
        """Format for LLM prompt injection."""
        return (
            f"CONFLICT DETECTED: {self.conflict_type}\n"
            f"  Value 1: {self.value_a} (from: {self.source_a})\n"
            f"  Value 2: {self.value_b} (from: {self.source_b})"
        )
```

### ConflictDetector Class

```python
@dataclass
class ConflictDetector:
    """Detects contradictions in entity descriptions."""

    confidence_threshold: float = 0.7

    # Pattern definitions
    DATE_PATTERNS: ClassVar[list[str]] = [
        r'\b(19|20)\d{2}\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    ]

    ATTRIBUTION_PATTERNS: ClassVar[list[str]] = [
        r'(founded|created|established)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]

    NUMBER_PATTERNS: ClassVar[list[str]] = [
        r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?',
    ]

    def detect_conflicts(
        self,
        entity_name: str,
        descriptions: list[tuple[str, str]]  # (description, source_id)
    ) -> list[ConflictInfo]:
        """
        Detect conflicts in a list of descriptions.

        Args:
            entity_name: Name of the entity
            descriptions: List of (description_text, source_id) tuples

        Returns:
            List of detected conflicts
        """
        ...
```

---

## Configuration Schema

### Environment Variables

```python
# Entity Resolution
ENABLE_ENTITY_RESOLUTION: bool = True
ENTITY_SIMILARITY_THRESHOLD: float = 0.85  # Range: 0.0-1.0
ENTITY_MIN_NAME_LENGTH: int = 3            # Minimum chars for fuzzy matching

# Conflict Detection
ENABLE_CONFLICT_DETECTION: bool = True
CONFLICT_CONFIDENCE_THRESHOLD: float = 0.7  # Range: 0.0-1.0
```

### Global Config Integration

```python
# Added to global_config dict in operate.py
{
    "enable_entity_resolution": True,
    "entity_similarity_threshold": 0.85,
    "entity_min_name_length": 3,
    "enable_conflict_detection": True,
    "conflict_confidence_threshold": 0.7,
}
```

---

## Prompt Templates

### Conflict-Aware Summary Prompt

```python
PROMPTS["summarize_with_conflicts"] = """---Role---
You are a Knowledge Graph Specialist with expertise in data reconciliation.

---IMPORTANT: Conflicts Detected---
The following conflicting information was found from different sources:

{conflict_details}

---Instructions---
1. Create a unified summary that integrates all available information
2. For each conflict: mention BOTH versions with uncertainty language
   Example: "Founded in 2003 or 2004 according to different sources"
3. Do NOT arbitrarily pick one version over another
4. Preserve all non-conflicting information normally

---Entity to Summarize---
Entity: {entity_name}
Type: {entity_type}

---Descriptions to Merge---
{description_list}

---Output---
Provide a single coherent description that acknowledges the conflicts:
"""
```

---

## Storage Considerations

### Aliases Storage Options

**Option A: In-Memory Only (Recommended for v1)**
- Aliases computed during ingestion
- Logged for traceability
- Not persisted to storage
- Simpler implementation

**Option B: Persistent Storage (Future)**
- Add `aliases` column to entity table
- Requires schema migration
- Enables alias-based querying
- More complex

### Conflict Storage Options

**Selected: Logs Only (per user preference)**
- Conflicts logged via standard logging
- No persistent audit table
- Sufficient for monitoring and debugging
- Can be upgraded later if needed

---

## Validation Rules

### Entity Name Validation

1. **Minimum Length**: Names ≤2 characters excluded from fuzzy matching
2. **Case Normalization**: Matching is case-insensitive
3. **Type Constraint**: Only entities of same type can be merged

### Conflict Detection Validation

1. **Same Entity**: Only compare descriptions of the same entity
2. **Pattern Match**: Values must match defined patterns
3. **Different Values**: Values must differ to be a conflict
4. **Confidence Threshold**: Only report conflicts above threshold

---

## Examples

### Entity Resolution Example

**Input**:
```python
all_nodes = {
    "Apple Inc": [{"description": "Tech company founded in 1976", ...}],
    "Apple Inc.": [{"description": "Makes iPhones and Macs", ...}],
    "Apple": [{"description": "Based in Cupertino", ...}],
    "APPLE INC": [{"description": "Market cap over $2T", ...}],
}
```

**After Resolution**:
```python
all_nodes = {
    "Apple Inc": [
        {"description": "Tech company founded in 1976", ...},
        {"description": "Makes iPhones and Macs", ...},
        {"description": "Based in Cupertino", ...},
        {"description": "Market cap over $2T", ...},
    ],
    # aliases: ["Apple Inc.", "Apple", "APPLE INC"]
}
```

### Conflict Detection Example

**Input**:
```python
descriptions = [
    ("Tesla was founded in 2003 by Martin Eberhard", "doc_001"),
    ("Tesla was founded in 2004", "doc_002"),
    ("Tesla is an electric vehicle manufacturer", "doc_003"),
]
```

**Output**:
```python
conflicts = [
    ConflictInfo(
        entity_name="Tesla",
        conflict_type="temporal",
        value_a="2003",
        value_b="2004",
        source_a="doc_001",
        source_b="doc_002",
        confidence=0.95,
        context_a="Tesla was founded in 2003 by Martin Eberhard",
        context_b="Tesla was founded in 2004",
    )
]
```
