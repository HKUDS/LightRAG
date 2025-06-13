"""
Relationship registry for Neo4j graph storage.
This module provides a registry for all supported relationship types for tech/development domains.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from difflib import SequenceMatcher

from fuzzywuzzy import fuzz

from ...utils import logger


def standardize_relationship_type(rel_type: str) -> str:
    """
    Convert a relationship type string to Neo4j standard format with comprehensive standardization.
    This function handles various input formats including CamelCase, spaces, hyphens, and more.

    Args:
        rel_type: Original relationship type string (e.g., "integrates with", "createdBy", "IntegratesWith")

    Returns:
        Standardized relationship type (e.g., "INTEGRATES_WITH", "CREATED_BY", "INTEGRATES_WITH")
    """
    # Handle None or empty input
    if not rel_type or not isinstance(rel_type, str):
        return "RELATED"

    # Convert to lowercase for processing
    processed_type = rel_type.strip().lower()

    # Handle empty after stripping
    if not processed_type:
        return "RELATED"

    # Step 1: Handle CamelCase/PascalCase to snake_case
    # Only apply camelCase conversion if:
    # 1. No spaces or underscores are present
    # 2. There are uppercase letters in the original string
    # 3. The string is not already all uppercase
    if (
        "_" not in rel_type
        and " " not in rel_type
        and any(c.isupper() for c in rel_type)
        and not rel_type.isupper()
    ):
        # Use regex to handle camelCase/PascalCase properly
        # This handles consecutive capitals better (e.g., APICall -> API_Call -> api_call)
        processed_type = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", rel_type)
        processed_type = re.sub(
            r"([A-Z])([A-Z][a-z])", r"\1_\2", processed_type
        ).lower()

    # Step 2: Replace spaces and hyphens with underscores
    processed_type = re.sub(r"[\s-]+", "_", processed_type)

    # Step 3: Remove any characters that are not alphanumeric or underscore
    processed_type = re.sub(r"[^a-z0-9_]", "", processed_type)

    # Step 4: Collapse multiple underscores into one
    processed_type = re.sub(r"_+", "_", processed_type)

    # Step 5: Remove leading/trailing underscores
    processed_type = processed_type.strip("_")

    # Step 6: Check if result is empty after transformations
    if not processed_type:
        return "RELATED"

    # Step 7: Convert to uppercase
    result = processed_type.upper()

    # Step 8: Truncate if too long for Neo4j (limit to 50 characters)
    if len(result) > 50:
        result = result[:50]
        # Remove trailing underscore if truncation created one
        result = result.rstrip("_")

    return result


class RelationshipTypeRegistry:
    """Registry of valid relationship types with their metadata for tech/development domain."""

    def __init__(self):
        """Initialize with the supported relationship types."""
        self.registry = {}
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize with tech/development focused relationship types."""
        # Tech/Development focused relationship types
        relationship_types = [
            # AI/ML Relationships
            "TRAINS_MODEL",
            "USES_EMBEDDINGS",
            "GENERATES_RESPONSE",
            "QUERIES_LLM",
            "FINE_TUNES",
            "PROMPTS",
            "VECTORIZES",
            "SEMANTIC_SEARCH",
            "AUGMENTS_DATA",
            # API & Integration
            "CALLS_API",
            "EXPOSES_ENDPOINT",
            "INTEGRATES_WITH",
            "WEBHOOKS_TO",
            "SUBSCRIBES_TO",
            "PUBLISHES_TO",
            "AUTHENTICATES_WITH",
            "RATE_LIMITS",
            "PROXIES_TO",
            # Frontend Development
            "RENDERS_COMPONENT",
            "MANAGES_STATE",
            "ROUTES_TO",
            "DISPLAYS_DATA",
            "HANDLES_EVENT",
            "STYLES_WITH",
            "ANIMATES",
            "RESPONSIVE_TO",
            "LAZY_LOADS",
            # Backend Development
            "PROCESSES_REQUEST",
            "QUERIES_DATABASE",
            "CACHES_IN",
            "VALIDATES_DATA",
            "TRANSFORMS_DATA",
            "SCHEDULES_JOB",
            "QUEUES_TASK",
            "LOGS_TO",
            "MONITORS",
            # Data Flow
            "READS_FROM",
            "WRITES_TO",
            "STREAMS_DATA",
            "BATCHES_DATA",
            "AGGREGATES",
            "FILTERS",
            "MAPS_TO",
            "REDUCES_TO",
            "PIPELINES_THROUGH",
            # Architecture & Infrastructure
            "DEPLOYS_TO",
            "CONTAINERIZED_IN",
            "ORCHESTRATES",
            "LOAD_BALANCES",
            "SCALES_WITH",
            "HOSTED_ON",
            "BACKED_BY",
            "REPLICATES_TO",
            "FAILOVER_TO",
            # Development Process
            "DEPENDS_ON",
            "EXTENDS",
            "IMPLEMENTS",
            "INHERITS_FROM",
            "COMPOSES",
            "DECORATES",
            "MOCKS",
            "TESTS",
            "DEBUGS",
            # Automation
            "AUTOMATES",
            "TRIGGERS",
            "SCHEDULES",
            "ORCHESTRATES_WORKFLOW",
            "CHAINS_TO",
            "BRANCHES_TO",
            "RETRIES_ON",
            "ROLLBACK_TO",
            "NOTIFIES",
            # Security & Auth
            "AUTHORIZES",
            "ENCRYPTS",
            "SIGNS_WITH",
            "VALIDATES_TOKEN",
            "GRANTS_ACCESS",
            "REVOKES_ACCESS",
            "AUDITS",
            "SECURES",
            # Version Control & CI/CD
            "COMMITS_TO",
            "MERGES_INTO",
            "BRANCHES_FROM",
            "TAGS_AS",
            "BUILDS_FROM",
            "PACKAGES_AS",
            "RELEASES_TO",
            "ROLLBACKS_FROM",
            # Documentation & Knowledge
            "DOCUMENTS",
            "REFERENCES",
            "ANNOTATES",
            "EXAMPLES_OF",
            "TUTORIALS_FOR",
            "MIGRATES_FROM",
            "DEPRECATED_BY",
            "REPLACES",
            # Performance & Optimization
            "OPTIMIZES",
            "INDEXES",
            "COMPRESSES",
            "MINIFIES",
            "BUNDLES_WITH",
            "LAZY_EVALUATES",
            "MEMOIZES",
            "PARALLELIZES",
            # Generic
            "RELATED",
            "CONNECTED_TO",
            "ASSOCIATED_WITH",
            "PART_OF",
            "CONTAINS",
            "BELONGS_TO",
            "USED_FOR",
        ]

        # Common relationship variants that need normalization
        common_variants = {
            # AI/ML variants
            "uses ai": "USES_EMBEDDINGS",
            "ai powered": "USES_EMBEDDINGS",
            "machine learning": "TRAINS_MODEL",
            "calls gpt": "QUERIES_LLM",
            "uses llm": "QUERIES_LLM",
            "generates with ai": "GENERATES_RESPONSE",
            # API variants
            "api call": "CALLS_API",
            "makes request to": "CALLS_API",
            "posts to": "CALLS_API",
            "gets from": "CALLS_API",
            "fetches from": "CALLS_API",
            "sends webhook": "WEBHOOKS_TO",
            "listens to": "SUBSCRIBES_TO",
            "broadcasts to": "PUBLISHES_TO",
            # Frontend variants
            "displays": "DISPLAYS_DATA",
            "shows": "DISPLAYS_DATA",
            "renders": "RENDERS_COMPONENT",
            "handles click": "HANDLES_EVENT",
            "on click": "HANDLES_EVENT",
            "styled with": "STYLES_WITH",
            "uses css": "STYLES_WITH",
            "navigates to": "ROUTES_TO",
            # Backend variants
            "processes": "PROCESSES_REQUEST",
            "handles request": "PROCESSES_REQUEST",
            "queries": "QUERIES_DATABASE",
            "selects from": "QUERIES_DATABASE",
            "inserts into": "WRITES_TO",
            "updates": "WRITES_TO",
            "deletes from": "WRITES_TO",
            "caches": "CACHES_IN",
            # Data flow variants
            "reads": "READS_FROM",
            "writes": "WRITES_TO",
            "saves to": "WRITES_TO",
            "loads from": "READS_FROM",
            "imports from": "READS_FROM",
            "exports to": "WRITES_TO",
            "transforms": "TRANSFORMS_DATA",
            "converts": "TRANSFORMS_DATA",
            # Architecture variants
            "deployed on": "DEPLOYS_TO",
            "runs on": "HOSTED_ON",
            "hosted by": "HOSTED_ON",
            "uses database": "BACKED_BY",
            "backed by": "BACKED_BY",
            "dockerized": "CONTAINERIZED_IN",
            "in container": "CONTAINERIZED_IN",
            "manages traffic for": "LOAD_BALANCES",
            "load balances": "LOAD_BALANCES",
            "balances traffic": "LOAD_BALANCES",
            # Development variants
            "requires": "DEPENDS_ON",
            "needs": "DEPENDS_ON",
            "uses": "DEPENDS_ON",
            "based on": "EXTENDS",
            "built on": "EXTENDS",
            "implements interface": "IMPLEMENTS",
            "tested by": "TESTS",
            "unit test": "TESTS",
            # Automation variants
            "automated by": "AUTOMATES",
            "triggers when": "TRIGGERS",
            "runs after": "CHAINS_TO",
            "follows": "CHAINS_TO",
            "notifies via": "NOTIFIES",
            "alerts": "NOTIFIES",
            # Classification variants
            "is a": "PART_OF",
            "is a type of": "PART_OF",
            "type of": "PART_OF",
            "instance of": "PART_OF",
            "kind of": "PART_OF",
            "category of": "PART_OF",
            # Usage variants
            "used for": "USED_FOR",
            "utilized by": "USED_FOR",
            "employed by": "USED_FOR",
            "leveraged by": "USED_FOR",
            "applied to": "USED_FOR",
            "used in": "USED_FOR",
            "utilized in": "USED_FOR",
            # Traffic/Network variants
            "directs traffic to": "ROUTES_TO",
            "routes traffic to": "ROUTES_TO",
            "forwards to": "ROUTES_TO",
            "redirects to": "ROUTES_TO",
            # Enhancement/Improvement variants
            "improves": "OPTIMIZES",
            "enhances": "OPTIMIZES",
            "boosts": "OPTIMIZES",
            "speeds up": "OPTIMIZES",
            # Error/Issue variants
            "causes": "TRIGGERS",
            "leads to": "TRIGGERS",
            "results in": "TRIGGERS",
            "manifests as": "PART_OF",
            # Monitoring/Tracking variants
            "tracks": "MONITORS",
            "observes": "MONITORS",
            "watches": "MONITORS",
            "measures": "MONITORS",
            # Protection/Security variants
            "protects against": "SECURES",
            "defends from": "SECURES",
            "guards against": "SECURES",
            # Generic variants
            "related to": "RELATED",
            "is related to": "RELATED",
            "connected with": "CONNECTED_TO",
            "is part of": "PART_OF",
            "includes": "CONTAINS",
            "has": "CONTAINS",
        }

        # Populate the registry with Neo4j types and additional metadata
        for rel_type in relationship_types:
            # Convert to lowercase for registry keys (human-readable version)
            original_type = rel_type.lower().replace("_", " ")

            # Generate Neo4j type using the enhanced standardization function
            neo4j_type = standardize_relationship_type(original_type)

            # Add to registry with metadata
            self.registry[original_type] = {
                "neo4j_type": neo4j_type,
                "description": f"Relationship type: {original_type}",
                "bidirectional": original_type
                in ["related", "connected to", "associated with"],
                "inverse": None,
            }

        # Add common variants
        for variant, standard in common_variants.items():
            if variant not in self.registry:
                # Generate standardized Neo4j type for the variant using enhanced function
                variant_neo4j_type = standardize_relationship_type(variant)

                # Find the standardized form in our registry for metadata consistency
                standard_key = standard.lower().replace("_", " ")
                if standard_key in self.registry:
                    # Use consistent metadata structure
                    self.registry[variant] = {
                        "neo4j_type": variant_neo4j_type,
                        "description": f"Variant of {standard_key}: {variant}",
                        "bidirectional": self.registry[standard_key].get(
                            "bidirectional", False
                        ),
                        "inverse": self.registry[standard_key].get("inverse", None),
                    }
                else:
                    # Fallback if standard key is not found
                    self.registry[variant] = {
                        "neo4j_type": variant_neo4j_type,
                        "description": f"Relationship variant: {variant}",
                        "bidirectional": False,
                        "inverse": None,
                    }

        # Define explicit bidirectional relationships and inverses
        inverse_pairs = [
            # API/Integration
            ("calls api", "exposes endpoint"),
            ("subscribes to", "publishes to"),
            ("webhooks to", "receives webhook from"),
            # Data flow
            ("reads from", "read by"),
            ("writes to", "written by"),
            ("streams data", "receives stream"),
            # Architecture
            ("deploys to", "hosts"),
            ("depends on", "required by"),
            ("extends", "extended by"),
            ("implements", "implemented by"),
            # Frontend/Backend
            ("renders component", "rendered by"),
            ("displays data", "displayed by"),
            ("processes request", "processed by"),
            ("queries database", "queried by"),
            # Automation
            ("triggers", "triggered by"),
            ("automates", "automated by"),
            ("chains to", "chained from"),
            # Development
            ("tests", "tested by"),
            ("documents", "documented by"),
            ("replaces", "replaced by"),
            ("migrates from", "migrates to"),
            # Structure
            ("contains", "part of"),
            ("composes", "composed by"),
            ("inherits from", "inherited by"),
        ]

        # Set up the inverse relationships
        for rel1, rel2 in inverse_pairs:
            if rel1 in self.registry and rel2 in self.registry:
                self.registry[rel1]["inverse"] = rel2
                self.registry[rel2]["inverse"] = rel1

        logger.info(
            f"Initialized tech/development relationship registry with {len(self.registry)} types"
        )

    def get_neo4j_type(self, original_type: str) -> str:
        """
        Get standardized Neo4j type from original relationship description.

        Args:
            original_type: Original relationship type string

        Returns:
            Neo4j-compatible relationship type string
        """
        if not original_type:
            return "RELATED"

        # Convert to lowercase for case-insensitive lookup
        original_type_lower = original_type.lower()

        # Direct lookup
        if original_type_lower in self.registry:
            return self.registry[original_type_lower]["neo4j_type"]

        # Find closest match
        closest_match = self._find_closest_match(original_type_lower)
        if closest_match:
            return self.registry[closest_match]["neo4j_type"]

        # If no match found, standardize the original
        return standardize_relationship_type(original_type)

    def get_relationship_metadata(self, original_type: str) -> Dict[str, Any]:
        """
        Get metadata for a relationship type.

        Args:
            original_type: Original relationship type string

        Returns:
            Dictionary of metadata for the relationship type
        """
        original_type_lower = original_type.lower()

        # Direct lookup
        if original_type_lower in self.registry:
            return self.registry[original_type_lower]

        # Find closest match
        closest_match = self._find_closest_match(original_type_lower)
        if closest_match:
            return self.registry[closest_match]

        # If no match found, return default metadata
        neo4j_type = standardize_relationship_type(original_type)
        return {
            "neo4j_type": neo4j_type,
            "description": f"Custom relationship type: {original_type}",
            "bidirectional": False,
            "inverse": None,
        }

    def get_all_relationship_types(self) -> List[str]:
        """
        Get all registered relationship types.

        Returns:
            List of all registered relationship types
        """
        return list(self.registry.keys())

    def get_bidirectional_types(self) -> List[str]:
        """
        Get all bidirectional relationship types.

        Returns:
            List of bidirectional relationship types
        """
        return [
            rel
            for rel, metadata in self.registry.items()
            if metadata.get("bidirectional", False)
        ]

    def get_relationship_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all relationship types with their inverses.

        Returns:
            List of (relationship_type, inverse_type) tuples
        """
        return [
            (rel, metadata["inverse"])
            for rel, metadata in self.registry.items()
            if metadata.get("inverse") is not None
        ]

    def _find_closest_match(self, rel_type: str) -> Optional[str]:
        """
        Find the closest matching registered relationship type.

        This method uses fuzzy matching to find similar relationship types,
        which is crucial for handling LLM-generated variations.

        Args:
            rel_type: Relationship type to find match for

        Returns:
            The closest matching relationship type or None if no good match found
        """
        if not rel_type:
            return None

        # Check for substring matches first (more precise)
        for reg_type in self.registry:
            # Check if one is a substring of the other
            if rel_type in reg_type or reg_type in rel_type:
                logger.debug(f"Found substring match for '{rel_type}': '{reg_type}'")
                return reg_type

        # If no substring match, use fuzzy matching
        best_match = None
        best_score = 70  # Minimum similarity threshold (0-100)

        for reg_type in self.registry:
            # Calculate similarity score (0-100)
            score = fuzz.ratio(rel_type, reg_type)
            if score > best_score:
                best_score = score
                best_match = reg_type

        if best_match:
            logger.debug(
                f"Found fuzzy match for '{rel_type}': '{best_match}' (score: {best_score})"
            )

        return best_match

    def find_best_match_with_confidence(
        self, rel_type: str
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching relationship type with confidence score.

        Args:
            rel_type: Relationship type to find match for

        Returns:
            Tuple of (best_match, confidence_score) where confidence is 0.0-1.0
        """
        if not rel_type:
            return None, 0.0

        rel_type_lower = rel_type.lower()

        # Direct match gets highest confidence
        if rel_type_lower in self.registry:
            return rel_type_lower, 1.0

        # Check for substring matches (high confidence)
        for reg_type in self.registry:
            if rel_type_lower in reg_type or reg_type in rel_type_lower:
                confidence = 0.9 + (
                    0.1
                    * max(len(rel_type_lower), len(reg_type))
                    / max(len(rel_type_lower) + len(reg_type), 1)
                )
                return reg_type, min(
                    confidence, 0.99
                )  # Cap at 0.99 for substring matches

        # Fuzzy matching with normalized confidence
        best_match = None
        best_score = 0

        for reg_type in self.registry:
            # Use multiple similarity metrics for better accuracy
            ratio_score = fuzz.ratio(rel_type_lower, reg_type)
            token_sort_score = fuzz.token_sort_ratio(rel_type_lower, reg_type)
            token_set_score = fuzz.token_set_ratio(rel_type_lower, reg_type)

            # Weighted average with emphasis on ratio score
            combined_score = (
                0.5 * ratio_score + 0.3 * token_sort_score + 0.2 * token_set_score
            )

            if combined_score > best_score:
                best_score = combined_score
                best_match = reg_type

        # Convert score to confidence (0-1) with minimum threshold
        if best_score >= 70:  # Minimum threshold
            confidence = (best_score - 70) / 30.0  # Map 70-100 to 0.0-1.0
            return best_match, min(confidence, 0.85)  # Cap fuzzy matches at 0.85

        return None, 0.0

    def get_relationship_suggestions(
        self, rel_type: str, max_suggestions: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get multiple relationship type suggestions with confidence scores.

        Args:
            rel_type: Relationship type to find suggestions for
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of (relationship_type, confidence) tuples, sorted by confidence
        """
        if not rel_type:
            return []

        rel_type_lower = rel_type.lower()

        # Direct match
        if rel_type_lower in self.registry:
            return [(rel_type_lower, 1.0)]

        suggestions = []

        for reg_type in self.registry:
            # Use multiple similarity metrics
            ratio_score = fuzz.ratio(rel_type_lower, reg_type)
            token_sort_score = fuzz.token_sort_ratio(rel_type_lower, reg_type)
            token_set_score = fuzz.token_set_ratio(rel_type_lower, reg_type)
            partial_score = fuzz.partial_ratio(rel_type_lower, reg_type)

            # Weighted average with domain-specific emphasis
            combined_score = (
                0.4 * ratio_score
                + 0.25 * token_sort_score
                + 0.25 * token_set_score
                + 0.1 * partial_score
            )

            # Apply domain-specific boosts
            domain_boost = self._calculate_domain_boost(rel_type_lower, reg_type)
            combined_score = min(combined_score + domain_boost, 100)

            # Convert to confidence score
            if combined_score >= 50:  # Lower threshold for suggestions
                confidence = (combined_score - 50) / 50.0
                suggestions.append((reg_type, min(confidence, 0.95)))

        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]

    def _calculate_domain_boost(self, input_type: str, registry_type: str) -> float:
        """
        Calculate domain-specific boost for relationship type matching.

        Args:
            input_type: Input relationship type
            registry_type: Registry relationship type to compare

        Returns:
            Boost score (0-10) to add to similarity score
        """
        boost = 0.0

        # Domain keyword mappings
        domain_keywords = {
            "api": ["call", "request", "endpoint", "webhook", "subscribe", "publish"],
            "ai": ["train", "model", "embed", "generate", "llm", "prompt", "vector"],
            "data": ["read", "write", "stream", "batch", "aggregate", "filter", "map"],
            "frontend": ["render", "display", "route", "event", "style", "animate"],
            "backend": [
                "process",
                "query",
                "cache",
                "validate",
                "transform",
                "schedule",
            ],
            "infra": ["deploy", "container", "scale", "host", "replicate", "monitor"],
        }

        # Check for domain keyword matches
        for domain, keywords in domain_keywords.items():
            input_has_keyword = any(keyword in input_type for keyword in keywords)
            registry_has_keyword = any(keyword in registry_type for keyword in keywords)

            if input_has_keyword and registry_has_keyword:
                boost += 5.0  # Strong domain match
            elif input_has_keyword or registry_has_keyword:
                boost += 2.0  # Partial domain match

        # Length similarity boost (prefer similar lengths)
        length_diff = abs(len(input_type) - len(registry_type))
        if length_diff <= 2:
            boost += 3.0
        elif length_diff <= 5:
            boost += 1.0

        return boost

    def validate_relationship_type(self, rel_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a relationship type is registered or similar to a registered type.

        Args:
            rel_type: Relationship type to validate

        Returns:
            (is_valid, suggested_alternative) tuple
        """
        if not rel_type:
            return False, "related"

        rel_type_lower = rel_type.lower()

        # Direct match
        if rel_type_lower in self.registry:
            return True, None

        # Use enhanced confidence-based matching
        best_match, confidence = self.find_best_match_with_confidence(rel_type)

        if best_match and confidence >= 0.7:  # High confidence threshold for validation
            logger.info(
                f"Relationship type '{rel_type}' validated with match '{best_match}' (confidence: {confidence:.2f})"
            )
            return True, best_match
        elif (
            best_match and confidence >= 0.4
        ):  # Suggest alternative for medium confidence
            logger.warning(
                f"Relationship type '{rel_type}' has low confidence match '{best_match}' (confidence: {confidence:.2f})"
            )
            return False, best_match

        # No good match found, suggest generic type
        logger.warning(
            f"No suitable match found for relationship type '{rel_type}', suggesting 'related'"
        )
        return False, "related"

    def validate_relationship_type_detailed(self, rel_type: str) -> Dict[str, Any]:
        """
        Enhanced validation with detailed feedback and confidence scores.

        Args:
            rel_type: Relationship type to validate

        Returns:
            Dictionary with validation results and suggestions
        """
        if not rel_type:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "best_match": "related",
                "suggestions": [("related", 1.0)],
                "feedback": "Empty relationship type provided",
            }

        rel_type_lower = rel_type.lower()

        # Direct match
        if rel_type_lower in self.registry:
            return {
                "is_valid": True,
                "confidence": 1.0,
                "best_match": rel_type_lower,
                "suggestions": [(rel_type_lower, 1.0)],
                "feedback": "Exact match found",
            }

        # Get best match with confidence
        best_match, confidence = self.find_best_match_with_confidence(rel_type)

        # Get multiple suggestions
        suggestions = self.get_relationship_suggestions(rel_type, max_suggestions=5)

        # Determine validity based on confidence
        is_valid = confidence >= 0.7

        # Generate feedback message
        feedback = self._generate_validation_feedback(
            rel_type, best_match, confidence, is_valid
        )

        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "best_match": best_match or "related",
            "suggestions": suggestions if suggestions else [("related", 1.0)],
            "feedback": feedback,
        }

    def _generate_validation_feedback(
        self,
        input_type: str,
        best_match: Optional[str],
        confidence: float,
        is_valid: bool,
    ) -> str:
        """
        Generate human-readable feedback for relationship type validation.

        Args:
            input_type: Original input relationship type
            best_match: Best matching registry type
            confidence: Confidence score (0.0-1.0)
            is_valid: Whether the type is considered valid

        Returns:
            Human-readable feedback message
        """
        if is_valid and best_match:
            if confidence == 1.0:
                return f"'{input_type}' is a registered relationship type"
            else:
                return f"'{input_type}' closely matches '{best_match}' (confidence: {confidence:.1%})"
        elif best_match and confidence >= 0.4:
            return f"'{input_type}' has moderate similarity to '{best_match}' (confidence: {confidence:.1%}). Consider using the suggested type."
        else:
            return f"'{input_type}' is not recognized. Using generic 'related' type. Consider using a more specific relationship type."

    def get_category_relationships(self, category: str) -> List[str]:
        """
        Get all relationships belonging to a specific category.

        Categories: ai_ml, api, frontend, backend, data, architecture,
                   development, automation, security, vcs, docs, performance

        Args:
            category: Category name

        Returns:
            List of relationship types in that category
        """
        categories = {
            "ai_ml": [
                "trains model",
                "uses embeddings",
                "generates response",
                "queries llm",
                "fine tunes",
                "prompts",
                "vectorizes",
                "semantic search",
                "augments data",
            ],
            "api": [
                "calls api",
                "exposes endpoint",
                "integrates with",
                "webhooks to",
                "subscribes to",
                "publishes to",
                "authenticates with",
                "rate limits",
                "proxies to",
            ],
            "frontend": [
                "renders component",
                "manages state",
                "routes to",
                "displays data",
                "handles event",
                "styles with",
                "animates",
                "responsive to",
                "lazy loads",
            ],
            "backend": [
                "processes request",
                "queries database",
                "caches in",
                "validates data",
                "transforms data",
                "schedules job",
                "queues task",
                "logs to",
                "monitors",
            ],
            "data": [
                "reads from",
                "writes to",
                "streams data",
                "batches data",
                "aggregates",
                "filters",
                "maps to",
                "reduces to",
                "pipelines through",
            ],
            "architecture": [
                "deploys to",
                "containerized in",
                "orchestrates",
                "load balances",
                "scales with",
                "hosted on",
                "backed by",
                "replicates to",
                "failover to",
            ],
            "development": [
                "depends on",
                "extends",
                "implements",
                "inherits from",
                "composes",
                "decorates",
                "mocks",
                "tests",
                "debugs",
            ],
            "automation": [
                "automates",
                "triggers",
                "schedules",
                "orchestrates workflow",
                "chains to",
                "branches to",
                "retries on",
                "rollback to",
                "notifies",
            ],
            "security": [
                "authorizes",
                "encrypts",
                "signs with",
                "validates token",
                "grants access",
                "revokes access",
                "audits",
                "secures",
            ],
            "vcs": [
                "commits to",
                "merges into",
                "branches from",
                "tags as",
                "builds from",
                "packages as",
                "releases to",
                "rollbacks from",
            ],
            "docs": [
                "documents",
                "references",
                "annotates",
                "examples of",
                "tutorials for",
                "migrates from",
                "deprecated by",
                "replaces",
            ],
            "performance": [
                "optimizes",
                "indexes",
                "compresses",
                "minifies",
                "bundles with",
                "lazy evaluates",
                "memoizes",
                "parallelizes",
            ],
        }

        return categories.get(category.lower(), [])


def _test_standardize_relationship_type():
    """
    Unit tests for the enhanced standardize_relationship_type function.
    These tests validate the comprehensive standardization logic for various input formats.
    """
    test_cases = [
        # Test cases from PRD section 9
        ("created by", "CREATED_BY"),
        ("integrates_with", "INTEGRATES_WITH"),
        ("createdby", "CREATEDBY"),  # Single word, no camelCase
        ("IntegratesWith", "INTEGRATES_WITH"),  # PascalCase
        ("calls-api", "CALLS_API"),  # Hyphenated
        ("  leading and trailing spaces  ", "LEADING_AND_TRAILING_SPACES"),
        ("special!@#chars", "SPECIALCHARS"),  # Special characters removed
        ("RELATED", "RELATED"),  # Already uppercase
        ("", "RELATED"),  # Empty string
        (None, "RELATED"),  # None input
        # Additional test cases for comprehensive coverage
        ("camelCaseExample", "CAMEL_CASE_EXAMPLE"),  # camelCase
        ("API_Call", "API_CALL"),  # Mixed case with underscore
        ("multiple   spaces", "MULTIPLE_SPACES"),  # Multiple spaces
        ("hyphens-and-underscores_mixed", "HYPHENS_AND_UNDERSCORES_MIXED"),
        ("123numbers456", "123NUMBERS456"),  # Numbers
        ("api-call_with-mixed_separators", "API_CALL_WITH_MIXED_SEPARATORS"),
        (
            "___multiple___underscores___",
            "MULTIPLE_UNDERSCORES",
        ),  # Multiple underscores
        ("CamelCaseWithNumbers123", "CAMEL_CASE_WITH_NUMBERS123"),
        ("a" * 60, "A" * 50),  # Truncation test (should be limited to 50 chars)
        ("   ", "RELATED"),  # Only whitespace
        ("APICall", "API_CALL"),  # Consecutive capitals
    ]

    print("Testing standardize_relationship_type function...")

    passed = 0
    failed = 0

    for input_val, expected in test_cases:
        try:
            result = standardize_relationship_type(input_val)
            if result == expected:
                print(f"✓ PASS: '{input_val}' -> '{result}'")
                passed += 1
            else:
                print(
                    f"✗ FAIL: '{input_val}' -> Expected: '{expected}', Got: '{result}'"
                )
                failed += 1
        except Exception as e:
            print(f"✗ ERROR: '{input_val}' -> Exception: {str(e)}")
            failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    # Run tests when the module is executed directly
    _test_standardize_relationship_type()
