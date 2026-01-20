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

import asyncio
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
    DEFAULT_PREFER_SHORTER_CANONICAL_NAME,
    DEFAULT_CPU_YIELD_INTERVAL,
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
        "2 CB" → "2CB"
    """
    # Match patterns like "X Y Z" or "2 C B" where each part is 1 char
    # We look for sequences of single characters separated by spaces

    def coalesce_match(match):
        # Remove all spaces between single characters
        return match.group(0).replace(" ", "")

    # Pattern 1: single char + (space + single char)+
    # This matches "A B C" but not "AB CD"
    pattern1 = r'\b([A-Za-z0-9])((?:\s+[A-Za-z0-9])+)\b'

    # Pattern 2: single DIGIT + space + short acronym (2-3 letters)
    # This matches "2 CB", "2 cb" but excludes French articles (la, le, un, du, de, les, des)
    # Uses negative lookahead to avoid breaking article removal
    pattern2 = r'\b([0-9])\s+(?!la\b|le\b|un\b|du\b|de\b|les\b|des\b)([A-Za-z]{2,3})\b'

    result = text
    # Keep applying until no more changes (handles overlapping patterns)
    while True:
        new_result = re.sub(pattern1, coalesce_match, result)
        # Also apply pattern 2 for short acronym fragments
        new_result = re.sub(pattern2, r'\1\2', new_result)
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


def compute_entity_similarity(name1: str, name2: str) -> float:
    """
    Compute similarity score between two entity names.

    This is a standalone function that can be used for cross-document entity
    resolution and graph consolidation. It uses a CONSERVATIVE matching approach
    to minimize false positives:
    - Uses fuzz.ratio (not token_set_ratio) for strict character-level matching
    - Requires numeric tokens to match exactly
    - Rejects first-name mismatches for person names

    This approach prioritizes precision over recall - it's better to miss
    a valid merge than to incorrectly merge different entities.

    Args:
        name1: First entity name.
        name2: Second entity name.

    Returns:
        Similarity score between 0.0 and 1.0.

    Examples:
        >>> compute_entity_similarity("Apple Inc", "Apple Inc.") > 0.9
        True
        >>> compute_entity_similarity("Senozan", "617 Impasse, 71260 Senozan") < 0.5
        True
        >>> compute_entity_similarity("J. Bondoux", "Sylvie Bondoux") < 0.5
        True
    """
    # Import here to avoid loading rapidfuzz if entity resolution is disabled
    from rapidfuzz import fuzz

    normalized1 = _normalize_for_matching(name1)
    normalized2 = _normalize_for_matching(name2)

    # Exact normalized match - this IS valid (e.g., "2 C B SAS" matches "2CB")
    if normalized1 == normalized2:
        return 1.0

    # Protection 1: Check numeric tokens - they must match exactly
    # This prevents "Facture 24012823" matching "Facture 24012815"
    nums1 = set(re.findall(r"\d+", normalized1))
    nums2 = set(re.findall(r"\d+", normalized2))
    if nums1 and nums2 and nums1 != nums2:
        # Both have numbers but they're different
        return 0.0

    # Protection 2: For 2-token names (likely person names), check first token similarity
    # This prevents "J. Bondoux" matching "Sylvie Bondoux"
    tokens1 = normalized1.split()
    tokens2 = normalized2.split()
    if len(tokens1) == 2 and len(tokens2) == 2:
        # If first tokens are very different, it's likely different people
        first_sim = fuzz.ratio(tokens1[0], tokens2[0]) / 100.0
        if first_sim < 0.6:
            return 0.0

    # Protection 3: Reject if one name is embedded in a much longer unrelated string
    # This prevents "Senozan" matching "617 Impasse du Pré denfer, 71260 Senozan"
    # But allows "Acme" to match "Acme Ingenierie" (proper prefix/subset)
    len_ratio = len(normalized1) / len(normalized2) if len(normalized2) > 0 else 0
    is_prefix_match = False

    if len_ratio < 0.4 or len_ratio > 2.5:
        # One is much shorter - check if it's a valid prefix/subset
        shorter_norm = normalized1 if len(normalized1) < len(normalized2) else normalized2
        longer_norm = normalized2 if len(normalized1) < len(normalized2) else normalized1
        shorter_tokens = set(tokens1) if len(normalized1) < len(normalized2) else set(tokens2)
        longer_tokens = set(tokens2) if len(normalized1) < len(normalized2) else set(tokens1)

        # Check if shorter name is a PREFIX of longer name (starts with it)
        # "acme" is prefix of "acme ingenierie" → valid
        # "senozan" is NOT prefix of "617 impasse..." → invalid
        if longer_norm.startswith(shorter_norm):
            is_prefix_match = True
        elif not shorter_tokens.issubset(longer_tokens):
            # Not a prefix and not a subset → reject
            return 0.0
        else:
            # Subset but not prefix - check if embedded in unrelated context
            # If the longer name has many more tokens, it's likely an address/description
            if len(longer_tokens) > len(shorter_tokens) * 3:
                return 0.0

    # For valid prefix matches, give high score
    # This handles "Acme" → "Acme Ingenierie" where fuzz.ratio would be low
    if is_prefix_match:
        return 0.95

    # Use fuzz.ratio for STRICT character-level matching
    # This gives lower scores for partial/subset matches, avoiding false positives
    score = fuzz.ratio(normalized1, normalized2) / 100.0

    return score


@dataclass
class EntityResolver:
    """
    Resolves and consolidates similar entity names using fuzzy matching.

    This class uses the rapidfuzz library to identify entities with similar names
    and merge them under a canonical name (the longest variant).

    Attributes:
        similarity_threshold: Minimum similarity score (0.0-1.0) for matching.
        min_name_length: Minimum character length for fuzzy matching eligibility.
        cpu_yield_interval: Yield control to event loop every N comparisons.

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
    prefer_shorter_canonical_name: bool = DEFAULT_PREFER_SHORTER_CANONICAL_NAME
    cpu_yield_interval: int = DEFAULT_CPU_YIELD_INTERVAL

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
        if self.cpu_yield_interval < 1:
            raise ValueError(
                f"cpu_yield_interval must be at least 1, got {self.cpu_yield_interval}"
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

        Uses a CONSERVATIVE matching approach to minimize false positives:
        - Uses fuzz.ratio (not token_set_ratio) for strict character-level matching
        - Requires numeric tokens to match exactly
        - Rejects first-name mismatches for person names

        This approach prioritizes precision over recall - it's better to miss
        a valid merge than to incorrectly merge different entities.

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

        # Protection 1: Check numeric tokens - they must match exactly
        # This prevents "Facture 24012823" matching "Facture 24012815"
        nums1 = set(re.findall(r"\d+", normalized1))
        nums2 = set(re.findall(r"\d+", normalized2))
        if nums1 and nums2 and nums1 != nums2:
            # Both have numbers but they're different
            return 0.0

        # Protection 2: For 2-token names (likely person names), check first token similarity
        # This prevents "J. Bondoux" matching "Sylvie Bondoux"
        tokens1 = normalized1.split()
        tokens2 = normalized2.split()
        if len(tokens1) == 2 and len(tokens2) == 2:
            # If first tokens are very different, it's likely different people
            first_sim = fuzz.ratio(tokens1[0], tokens2[0]) / 100.0
            if first_sim < 0.6:
                return 0.0

        # Use fuzz.ratio for STRICT character-level matching
        # This gives lower scores for partial/subset matches, avoiding false positives:
        # - "Jacques" vs "Jacques Bondoux" → 0.64 (won't merge at 0.92 threshold)
        # - "THALIE" vs "Thalie" → 1.00 (will merge)
        # - "SAS SFJB" vs "SFJB" → 1.00 after normalization (will merge)
        score = fuzz.ratio(normalized1, normalized2) / 100.0

        return score

    def _select_canonical_name(self, names: set[str]) -> str:
        """
        Select the canonical name from a set of similar names.

        Selection criteria:
        1. By default, longest name is preferred (more specific/complete)
        2. If prefer_shorter_canonical_name is True, shortest name is preferred
        3. If equal length, first alphabetically (deterministic)
        4. Ensure first letter is capitalized for display

        Args:
            names: Set of similar entity names.

        Returns:
            The selected canonical name with proper capitalization.
        """
        if not names:
            raise ValueError("Cannot select canonical name from empty set")

        # Sort by length (ascending if prefer_shorter, descending otherwise)
        # Then alphabetically for ties
        if self.prefer_shorter_canonical_name:
            sorted_names = sorted(names, key=lambda n: (len(n), n))
        else:
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

    async def consolidate_entities(
        self, all_nodes: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Consolidate entities with similar names.

        This method:
        1. Groups entities by type (only same-type entities can be merged)
        2. For each type group, identifies clusters of similar names
        3. Merges entity records under canonical names
        4. Logs all merge operations

        Note: This method is async to allow CPU yielding during intensive
        fuzzy matching operations, preventing event loop blocking.

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
            # Get entity type from first record (normalize to uppercase for consistent grouping)
            entity_type = records[0].get("entity_type", "UNKNOWN").upper()
            entities_by_type[entity_type][entity_name] = records

        # Process each type group separately
        consolidated: dict[str, list[dict[str, Any]]] = {}

        for entity_type, type_entities in entities_by_type.items():
            type_consolidated = await self._consolidate_same_type_entities(
                type_entities, entity_type
            )
            consolidated.update(type_consolidated)

        return consolidated

    async def _consolidate_same_type_entities(
        self, entities: dict[str, list[dict[str, Any]]], entity_type: str
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Consolidate entities of the same type.

        This method performs O(n²) fuzzy matching comparisons which can be
        CPU-intensive. It yields control back to the event loop periodically
        to prevent blocking other async operations.

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

        # CPU yielding configuration: yield every N comparisons
        comparison_count = 0
        yield_interval = self.cpu_yield_interval

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
                comparison_count += 1

                # CPU yielding: allow other async tasks to run
                if comparison_count % yield_interval == 0:
                    await asyncio.sleep(0)

                if similarity >= self.similarity_threshold:
                    cluster.add(other_name)
                    used_names.add(other_name)

            clusters.append(cluster)

        # Log total comparisons for debugging
        if comparison_count > 1000:
            logger.debug(
                f"Entity resolution ({entity_type}): {comparison_count} comparisons, "
                f"{len(matchable_names)} entities"
            )

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
