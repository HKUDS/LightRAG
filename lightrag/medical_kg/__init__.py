"""Deterministic medical knowledge graph normalization helpers."""

from .graph_projection import project_medical_graph
from .normalizer import NormalizedExtraction, normalize_medical_extraction
from .ontology import (
    CANONICAL_ALIASES,
    RELATION_KEYWORD_ALIASES,
    VALUE_ENTITY_TYPES,
    VALUE_LIKE_PATTERNS,
    DroppedNode,
    canonical_name,
    is_value_like_entity,
    normalize_relation_keyword,
)

__all__ = [
    "CANONICAL_ALIASES",
    "RELATION_KEYWORD_ALIASES",
    "VALUE_ENTITY_TYPES",
    "VALUE_LIKE_PATTERNS",
    "DroppedNode",
    "NormalizedExtraction",
    "canonical_name",
    "is_value_like_entity",
    "normalize_medical_extraction",
    "normalize_relation_keyword",
    "project_medical_graph",
]
