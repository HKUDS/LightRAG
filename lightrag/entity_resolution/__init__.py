"""
Entity Resolution Module for LightRAG

Provides automatic entity deduplication using a 3-layer approach:
1. Case normalization (exact match)
2. Fuzzy string matching (typos)
3. Vector similarity + LLM verification (semantic matches)
"""

from .config import DEFAULT_CONFIG, EntityResolutionConfig
from .resolver import (
    ResolutionResult,
    fuzzy_similarity,
    get_cached_alias,
    resolve_entity,
    resolve_entity_with_vdb,
    store_alias,
)

__all__ = [
    'DEFAULT_CONFIG',
    'EntityResolutionConfig',
    'ResolutionResult',
    'fuzzy_similarity',
    'get_cached_alias',
    'resolve_entity',
    'resolve_entity_with_vdb',
    'store_alias',
]
