"""
Entity Resolution module for LightRAG.

This module provides fuzzy matching and deduplication of similar entity names
during document ingestion. Entities like "Apple Inc", "Apple Inc.", and "Apple"
are consolidated under a single canonical name.

Enhanced with French legal entity support:
- Removes legal forms (SAS, SARL, etc.)
- Normalizes accents (é → e)
- Coalesces acronyms (S F J B → SFJB)
- Removes articles (La, Le, Société)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from lightrag.constants import (
    DEFAULT_ENABLE_ENTITY_RESOLUTION,
    DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
    DEFAULT_ENTITY_MIN_NAME_LENGTH,
)

logger = logging.getLogger("lightrag.entity_resolution")


# French legal forms to remove during normalization (case-insensitive)
FRENCH_LEGAL_FORMS = {
    "sas", "sarl", "sa", "sasu", "eurl", "sci", "snc", "sca", "scop",
    "saem", "sem", "eirl", "ei", "gie", "gmbh", "ag", "ltd", "llc",
    "inc", "corp", "corporation", "incorporated", "limited",
}

# French articles and common prefixes to remove
# Note: "compagnie" is NOT included as it's often the main name
FRENCH_ARTICLES = {
    "la", "le", "les", "l", "un", "une", "des", "du", "de", "d",
    "société", "societe", "ste", "entreprise", "ets", "etablissements",
    "groupe", "holding", "cie",
}


def _remove_accents(text: str) -> str:
    """Remove accents from text (é → e, è → e, etc.)."""
    # NFD normalization separates base characters from combining marks
    normalized = unicodedata.normalize("NFD", text)
    # Remove combining marks (accents)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def _coalesce_single_chars(text: str) -> str:
    """
    Coalesce single characters separated by spaces into acronyms.

    Examples:
        "2 C B SAS" → "2CB SAS"
        "S F J B" → "SFJB"
        "A B C Company" → "ABC Company"
    """
    # Match patterns like "X Y Z" or "2 C B" where each part is 1 char
    # We look for sequences of single characters separated by spaces

    def coalesce_match(match):
        # Remove all spaces between single characters
        return match.group(0).replace(" ", "")

    # Pattern: single char + (space + single char)+
    # This matches "A B C" but not "AB CD"
    pattern = r'\b([A-Za-z0-9])((?:\s+[A-Za-z0-9])+)\b'

    result = text
    # Keep applying until no more changes (handles overlapping patterns)
    while True:
        new_result = re.sub(pattern, coalesce_match, result)
        if new_result == result:
            break
        result = new_result

    return result


def _normalize_for_matching(name: str) -> str:
    """
    Normalize entity name for fuzzy matching comparison.

    This is more aggressive than display normalization:
    1. Lowercase
    2. Remove accents
    3. Remove punctuation
    4. Coalesce single characters (acronyms)
    5. Remove French legal forms
    6. Remove French articles
    7. Normalize whitespace

    Args:
        name: The entity name to normalize.

    Returns:
        Normalized string for comparison.
    """
    # Step 1: Basic normalization
    result = name.lower().strip()

    # Step 2: Remove accents
    result = _remove_accents(result)

    # Step 3: Remove punctuation (keep alphanumeric and spaces)
    result = re.sub(r"[^\w\s]", " ", result)

    # Step 4: Coalesce single characters into acronyms
    result = _coalesce_single_chars(result)

    # Step 5: Tokenize (split on whitespace, removes empty tokens)
    tokens = result.split()

    # Step 6: Remove legal forms and articles
    filtered_tokens = [
        token for token in tokens
        if token not in FRENCH_LEGAL_FORMS and token not in FRENCH_ARTICLES
    ]

    # Step 7: If all tokens were removed, keep original tokens
    if not filtered_tokens:
        filtered_tokens = tokens

    # Step 8: Rejoin with single spaces
    return " ".join(filtered_tokens)


@dataclass
class EntityResolver:
    """
    Resolves and consolidates similar entity names using fuzzy matching.

    This class uses the rapidfuzz library to identify entities with similar names
    and merge them under a canonical name (the longest variant).

    Attributes:
        similarity_threshold: Minimum similarity score (0.0-1.0) for matching.
        min_name_length: Minimum character length for fuzzy matching eligibility.

    Example:
        >>> resolver = EntityResolver(similarity_threshold=0.85)
        >>> all_nodes = {
        ...     "Apple Inc": [{"description": "Tech company"}],
        ...     "Apple Inc.": [{"description": "Makes iPhones"}],
        ...     "Apple": [{"description": "Based in Cupertino"}],
        ... }
        >>> resolved = resolver.consolidate_entities(all_nodes)
        >>> # Result: {"Apple Inc": [all three descriptions merged]}
    """

    similarity_threshold: float = DEFAULT_ENTITY_SIMILARITY_THRESHOLD
    min_name_length: int = DEFAULT_ENTITY_MIN_NAME_LENGTH

    # Internal state - maps original names to their canonical form
    canonical_names: dict[str, str] = field(default_factory=dict)
    # Maps canonical names to all their aliases
    alias_groups: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )
        if self.min_name_length < 1:
            raise ValueError(
                f"min_name_length must be at least 1, got {self.min_name_length}"
            )

    def _normalize_name(self, name: str) -> str:
        """
        Normalize entity name for fuzzy matching comparison.

        Applies comprehensive normalization for better matching:
        - Case-insensitive comparison
        - Accent normalization (é → e)
        - Single character coalescence (S F J B → SFJB)
        - French legal form removal (SAS, SARL, etc.)
        - French article removal (La, Le, Société)

        Args:
            name: The entity name to normalize.

        Returns:
            Normalized version of the name for comparison.
        """
        return _normalize_for_matching(name)

    def _compute_similarity(self, name1: str, name2: str) -> float:
        """
        Compute similarity score between two entity names.

        Uses rapidfuzz's token_set_ratio for robust matching that handles:
        - Word order variations ("Apple Inc" vs "Inc Apple")
        - Partial matches ("Apple" vs "Apple Inc")
        - Punctuation differences ("Apple Inc." vs "Apple Inc")

        Args:
            name1: First entity name.
            name2: Second entity name.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Import here to avoid loading rapidfuzz if entity resolution is disabled
        from rapidfuzz import fuzz

        normalized1 = self._normalize_name(name1)
        normalized2 = self._normalize_name(name2)

        # token_set_ratio handles word order and partial matches well
        score = fuzz.token_set_ratio(normalized1, normalized2) / 100.0
        return score

    def _select_canonical_name(self, names: set[str]) -> str:
        """
        Select the canonical name from a set of similar names.

        Selection criteria (per FR-003):
        1. Longest name is preferred (more specific/complete)
        2. If equal length, first alphabetically (deterministic)
        3. Ensure first letter is capitalized for display

        Args:
            names: Set of similar entity names.

        Returns:
            The selected canonical name with proper capitalization.
        """
        if not names:
            raise ValueError("Cannot select canonical name from empty set")

        # Sort by length (descending), then alphabetically for ties
        sorted_names = sorted(names, key=lambda n: (-len(n), n))
        canonical = sorted_names[0]

        # Ensure first letter is capitalized for proper display in graph
        if canonical and canonical[0].islower():
            canonical = canonical[0].upper() + canonical[1:]

        return canonical

    def resolve(self, entity_name: str, entity_type: str) -> str:
        """
        Resolve an entity name to its canonical form.

        If this name has already been seen and matched to a canonical name,
        returns that canonical name. Otherwise, returns the input name.

        Args:
            entity_name: The entity name to resolve.
            entity_type: The entity type (used for logging, matching constraint
                        is applied in consolidate_entities).

        Returns:
            The canonical name for this entity, or the input name if no match.
        """
        if entity_name in self.canonical_names:
            return self.canonical_names[entity_name]
        return entity_name

    def consolidate_entities(
        self, all_nodes: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Consolidate entities with similar names.

        This method:
        1. Groups entities by type (only same-type entities can be merged)
        2. For each type group, identifies clusters of similar names
        3. Merges entity records under canonical names
        4. Logs all merge operations

        Args:
            all_nodes: Dict mapping entity_name -> list of entity records.
                      Each record should have an 'entity_type' field.

        Returns:
            Consolidated dict with merged entities under canonical names.
        """
        if not all_nodes:
            return all_nodes

        # Group entities by type first (FR-006: same-type constraint)
        entities_by_type: dict[str, dict[str, list[dict]]] = defaultdict(dict)

        for entity_name, records in all_nodes.items():
            if not records:
                continue
            # Get entity type from first record
            entity_type = records[0].get("entity_type", "UNKNOWN")
            entities_by_type[entity_type][entity_name] = records

        # Process each type group separately
        consolidated: dict[str, list[dict[str, Any]]] = {}

        for entity_type, type_entities in entities_by_type.items():
            type_consolidated = self._consolidate_same_type_entities(
                type_entities, entity_type
            )
            consolidated.update(type_consolidated)

        return consolidated

    def _consolidate_same_type_entities(
        self, entities: dict[str, list[dict[str, Any]]], entity_type: str
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Consolidate entities of the same type.

        Args:
            entities: Dict mapping entity_name -> list of entity records.
            entity_type: The type of all entities in this dict.

        Returns:
            Consolidated dict with merged entities.
        """
        if not entities:
            return entities

        entity_names = list(entities.keys())

        # Skip short names from fuzzy matching (FR-018)
        matchable_names = [
            name for name in entity_names if len(name) > self.min_name_length
        ]
        short_names = [
            name for name in entity_names if len(name) <= self.min_name_length
        ]

        # Build clusters of similar names
        clusters: list[set[str]] = []
        used_names: set[str] = set()

        for name in matchable_names:
            if name in used_names:
                continue

            # Start a new cluster with this name
            cluster = {name}
            used_names.add(name)

            # Find all similar names
            for other_name in matchable_names:
                if other_name in used_names:
                    continue

                similarity = self._compute_similarity(name, other_name)
                if similarity >= self.similarity_threshold:
                    cluster.add(other_name)
                    used_names.add(other_name)

            clusters.append(cluster)

        # Build consolidated result
        consolidated: dict[str, list[dict[str, Any]]] = {}

        # Process clusters
        for cluster in clusters:
            canonical = self._select_canonical_name(cluster)
            merged_records: list[dict[str, Any]] = []

            for name in cluster:
                merged_records.extend(entities[name])
                # Track mapping
                self.canonical_names[name] = canonical
                if name != canonical:
                    self.alias_groups[canonical].add(name)

            consolidated[canonical] = merged_records

            # Log merge operation if there were aliases
            if len(cluster) > 1:
                aliases = cluster - {canonical}
                logger.info(
                    f"Entity resolution: {aliases} → '{canonical}' (type: {entity_type})"
                )

        # Add short names as-is (no fuzzy matching)
        for name in short_names:
            consolidated[name] = entities[name]

        return consolidated
