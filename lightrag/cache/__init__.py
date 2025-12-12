"""
LightRAG caching infrastructure.

This module provides caching implementations for various LightRAG components:
- fts_cache: Full-text search result caching
"""

from .fts_cache import (
    FTS_CACHE_ENABLED,
    FTS_CACHE_MAX_SIZE,
    FTS_CACHE_TTL,
    get_cached_fts_results,
    invalidate_fts_cache_for_workspace,
    store_fts_results,
)

__all__ = [
    'FTS_CACHE_ENABLED',
    'FTS_CACHE_MAX_SIZE',
    'FTS_CACHE_TTL',
    'get_cached_fts_results',
    'invalidate_fts_cache_for_workspace',
    'store_fts_results',
]
