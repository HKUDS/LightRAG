"""
Conflict Detection module for LightRAG.

This module detects contradictions in entity descriptions during summarization.
It identifies conflicts in dates, numbers, and attributions across different
source documents and surfaces them in entity summaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import ClassVar

from lightrag.constants import (
    DEFAULT_ENABLE_CONFLICT_DETECTION,
    DEFAULT_CONFLICT_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger("lightrag.conflict_detection")


@dataclass
class ConflictInfo:
    """
    Represents a detected conflict between entity descriptions.

    Attributes:
        entity_name: Name of the entity with conflicting information.
        conflict_type: Type of conflict ("temporal", "attribution", "numerical").
        value_a: First conflicting value.
        value_b: Second conflicting value.
        source_a: Source identifier for value_a (chunk_id or doc_id).
        source_b: Source identifier for value_b.
        confidence: Detection confidence score (0.0-1.0).
        context_a: Sentence containing value_a.
        context_b: Sentence containing value_b.
    """

    entity_name: str
    conflict_type: str
    value_a: str
    value_b: str
    source_a: str
    source_b: str
    confidence: float
    context_a: str
    context_b: str

    def to_log_message(self) -> str:
        """
        Format conflict for logging.

        Returns:
            Human-readable log message describing the conflict.
        """
        return (
            f"Conflict[{self.conflict_type}] in '{self.entity_name}': "
            f"'{self.value_a}' vs '{self.value_b}' "
            f"(confidence: {self.confidence:.2f})"
        )

    def to_prompt_context(self) -> str:
        """
        Format conflict for LLM prompt injection.

        Returns:
            Formatted string suitable for including in LLM prompts.
        """
        return (
            f"CONFLICT DETECTED: {self.conflict_type}\n"
            f"  Value 1: {self.value_a} (from: {self.source_a})\n"
            f"  Value 2: {self.value_b} (from: {self.source_b})"
        )


@dataclass
class ConflictDetector:
    """
    Detects contradictions in entity descriptions.

    This class uses pattern-based detection to identify conflicts in:
    - Temporal data (years, dates)
    - Attribution data (founders, creators)
    - Numerical data (amounts, quantities)

    Attributes:
        confidence_threshold: Minimum confidence score for reporting conflicts.

    Example:
        >>> detector = ConflictDetector(confidence_threshold=0.7)
        >>> descriptions = [
        ...     ("Tesla was founded in 2003", "doc_001"),
        ...     ("Tesla was founded in 2004", "doc_002"),
        ... ]
        >>> conflicts = detector.detect_conflicts("Tesla", descriptions)
        >>> # Returns: [ConflictInfo(conflict_type="temporal", value_a="2003", value_b="2004", ...)]
    """

    confidence_threshold: float = DEFAULT_CONFLICT_CONFIDENCE_THRESHOLD

    # Pattern definitions for different conflict types
    DATE_PATTERNS: ClassVar[list[str]] = [
        r"\b(19|20)\d{2}\b",  # Years: 1900-2099
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates: MM/DD/YY or MM/DD/YYYY
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",  # Dates: MM-DD-YY or MM-DD-YYYY
    ]

    ATTRIBUTION_PATTERNS: ClassVar[list[str]] = [
        r"(founded|created|established|started|invented)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"(founder|creator|inventor)(?:\s+is|\s+was)?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]

    NUMBER_PATTERNS: ClassVar[list[str]] = [
        r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?",  # Currency
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:percent|%)",  # Percentages
        r"\b\d+(?:,\d{3})*\s+(?:employees|workers|people|users|customers)",  # Counts
    ]

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )

    def _extract_values(
        self, text: str, patterns: list[str]
    ) -> list[tuple[str, int, int]]:
        """
        Extract values matching patterns from text.

        Args:
            text: Text to search for patterns.
            patterns: List of regex patterns to match.

        Returns:
            List of tuples (matched_value, start_pos, end_pos).
        """
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # For attribution patterns, capture the name (group 2)
                if match.lastindex and match.lastindex >= 2:
                    value = match.group(2)
                else:
                    value = match.group(0)
                matches.append((value, match.start(), match.end()))
        return matches

    def _compare_values(
        self,
        values_a: list[tuple[str, int, int]],
        values_b: list[tuple[str, int, int]],
        conflict_type: str,
    ) -> list[tuple[str, str, float]]:
        """
        Compare two sets of extracted values for conflicts.

        Args:
            values_a: Values from first description.
            values_b: Values from second description.
            conflict_type: Type of conflict being checked.

        Returns:
            List of tuples (value_a, value_b, confidence) for detected conflicts.
        """
        conflicts = []

        for val_a, _, _ in values_a:
            for val_b, _, _ in values_b:
                # Normalize values for comparison
                norm_a = val_a.lower().strip()
                norm_b = val_b.lower().strip()

                # Skip if values are the same
                if norm_a == norm_b:
                    continue

                # For numerical/date values, check if they're actually different numbers
                if conflict_type in ("temporal", "numerical"):
                    # Extract just the numbers for comparison
                    nums_a = re.findall(r"\d+", val_a)
                    nums_b = re.findall(r"\d+", val_b)
                    if nums_a == nums_b:
                        continue

                # Calculate confidence based on conflict type
                confidence = self._calculate_confidence(val_a, val_b, conflict_type)
                if confidence >= self.confidence_threshold:
                    conflicts.append((val_a, val_b, confidence))

        return conflicts

    def _calculate_confidence(
        self, value_a: str, value_b: str, conflict_type: str
    ) -> float:
        """
        Calculate confidence score for a potential conflict.

        Args:
            value_a: First value.
            value_b: Second value.
            conflict_type: Type of conflict.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Base confidence varies by type
        base_confidence = {
            "temporal": 0.95,  # Years/dates are usually clear conflicts
            "attribution": 0.85,  # Names might have variations
            "numerical": 0.90,  # Numbers are usually clear
        }.get(conflict_type, 0.80)

        return base_confidence

    def _get_context_sentence(self, text: str, start: int, end: int) -> str:
        """
        Extract the sentence containing a matched value.

        Args:
            text: Full text.
            start: Start position of matched value.
            end: End position of matched value.

        Returns:
            The sentence containing the matched value.
        """
        # Find sentence boundaries
        sentence_start = text.rfind(".", 0, start)
        sentence_start = 0 if sentence_start == -1 else sentence_start + 1

        sentence_end = text.find(".", end)
        sentence_end = len(text) if sentence_end == -1 else sentence_end + 1

        return text[sentence_start:sentence_end].strip()

    def detect_conflicts(
        self, entity_name: str, descriptions: list[tuple[str, str]]
    ) -> list[ConflictInfo]:
        """
        Detect conflicts in a list of descriptions.

        This method compares all pairs of descriptions to find contradictions
        in temporal, attribution, and numerical data.

        Args:
            entity_name: Name of the entity being analyzed.
            descriptions: List of (description_text, source_id) tuples.

        Returns:
            List of detected ConflictInfo objects.
        """
        if len(descriptions) < 2:
            return []

        conflicts: list[ConflictInfo] = []

        # Define what to check for each conflict type
        conflict_checks = [
            ("temporal", self.DATE_PATTERNS),
            ("attribution", self.ATTRIBUTION_PATTERNS),
            ("numerical", self.NUMBER_PATTERNS),
        ]

        # Compare all pairs of descriptions
        for i, (desc_a, source_a) in enumerate(descriptions):
            for desc_b, source_b in descriptions[i + 1 :]:
                for conflict_type, patterns in conflict_checks:
                    # Extract values from both descriptions
                    values_a = self._extract_values(desc_a, patterns)
                    values_b = self._extract_values(desc_b, patterns)

                    if not values_a or not values_b:
                        continue

                    # Compare values for conflicts
                    detected = self._compare_values(values_a, values_b, conflict_type)

                    for val_a, val_b, confidence in detected:
                        # Find context for the values
                        match_a = next(
                            (v for v in values_a if v[0] == val_a),
                            (val_a, 0, len(val_a)),
                        )
                        match_b = next(
                            (v for v in values_b if v[0] == val_b),
                            (val_b, 0, len(val_b)),
                        )

                        conflict = ConflictInfo(
                            entity_name=entity_name,
                            conflict_type=conflict_type,
                            value_a=val_a,
                            value_b=val_b,
                            source_a=source_a,
                            source_b=source_b,
                            confidence=confidence,
                            context_a=self._get_context_sentence(
                                desc_a, match_a[1], match_a[2]
                            ),
                            context_b=self._get_context_sentence(
                                desc_b, match_b[1], match_b[2]
                            ),
                        )
                        conflicts.append(conflict)

                        # Log the conflict
                        logger.warning(conflict.to_log_message())

        return conflicts
