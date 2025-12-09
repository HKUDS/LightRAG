"""
LightRAG caching infrastructure.

This module provides caching implementations for various LightRAG components:
- fts_cache: Full-text search result caching
"""

from .fts_cache import (
    get_cached_fts_results,
    store_fts_results,
    invalidate_fts_cache_for_workspace,
    FTS_CACHE_ENABLED,
    FTS_CACHE_TTL,
    FTS_CACHE_MAX_SIZE,
)

__all__ = [
    'get_cached_fts_results',
    'store_fts_results',
    'invalidate_fts_cache_for_workspace',
    'FTS_CACHE_ENABLED',
    'FTS_CACHE_TTL',
    'FTS_CACHE_MAX_SIZE',
]
