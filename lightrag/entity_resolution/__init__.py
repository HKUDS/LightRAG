"""
Entity Resolution Module for LightRAG

Provides automatic entity deduplication using a 3-layer approach:
1. Case normalization (exact match)
2. Fuzzy string matching (typos)
3. Vector similarity + LLM verification (semantic matches)
"""

from .resolver import (
    resolve_entity,
    resolve_entity_with_vdb,
    ResolutionResult,
    get_cached_alias,
    store_alias,
    fuzzy_similarity,
)
from .config import EntityResolutionConfig, DEFAULT_CONFIG

__all__ = [
    "resolve_entity",
    "resolve_entity_with_vdb",
    "ResolutionResult",
    "EntityResolutionConfig",
    "DEFAULT_CONFIG",
    "get_cached_alias",
    "store_alias",
    "fuzzy_similarity",
]
