"""
Unit tests for Full-Text Search caching.
"""

import asyncio
import pytest
import time

from lightrag.cache.fts_cache import (
    get_cached_fts_results,
    store_fts_results,
    invalidate_fts_cache_for_workspace,
    get_fts_cache_stats,
    _fts_cache,
    _compute_cache_key,
    FTS_CACHE_TTL,
)


class TestFTSCacheKey:
    """Test cache key computation."""

    def test_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        key1 = _compute_cache_key('test query', 'default', 10, 'english')
        key2 = _compute_cache_key('test query', 'default', 10, 'english')
        assert key1 == key2

    def test_cache_key_differs_by_query(self):
        """Different queries produce different keys."""
        key1 = _compute_cache_key('query A', 'default', 10, 'english')
        key2 = _compute_cache_key('query B', 'default', 10, 'english')
        assert key1 != key2

    def test_cache_key_differs_by_workspace(self):
        """Different workspaces produce different keys."""
        key1 = _compute_cache_key('test', 'workspace1', 10, 'english')
        key2 = _compute_cache_key('test', 'workspace2', 10, 'english')
        assert key1 != key2

    def test_cache_key_differs_by_limit(self):
        """Different limits produce different keys."""
        key1 = _compute_cache_key('test', 'default', 10, 'english')
        key2 = _compute_cache_key('test', 'default', 20, 'english')
        assert key1 != key2

    def test_cache_key_differs_by_language(self):
        """Different languages produce different keys."""
        key1 = _compute_cache_key('test', 'default', 10, 'english')
        key2 = _compute_cache_key('test', 'default', 10, 'french')
        assert key1 != key2

    def test_cache_key_length(self):
        """Cache key is 16 characters (SHA256[:16])."""
        key = _compute_cache_key('test', 'default', 10, 'english')
        assert len(key) == 16


class TestFTSCacheOperations:
    """Test cache operations."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear cache before each test."""
        _fts_cache.clear()
        yield
        _fts_cache.clear()

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        result = await get_cached_fts_results('new query', 'default', 10, 'english')
        assert result is None

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        """Stored results can be retrieved."""
        test_results = [{'id': 'chunk1', 'content': 'test', 'score': 0.9}]

        await store_fts_results('test query', 'default', 10, 'english', test_results)
        cached = await get_cached_fts_results('test query', 'default', 10, 'english')

        assert cached == test_results

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Expired entries are not returned."""
        test_results = [{'id': 'chunk1', 'content': 'test', 'score': 0.9}]

        await store_fts_results('test query', 'default', 10, 'english', test_results)

        # Manually expire the entry
        cache_key = _compute_cache_key('test query', 'default', 10, 'english')
        _fts_cache[cache_key] = (test_results, 0)  # timestamp = 0 (expired)

        cached = await get_cached_fts_results('test query', 'default', 10, 'english')
        assert cached is None

    @pytest.mark.asyncio
    async def test_invalidation_clears_cache(self):
        """Invalidation clears cache entries."""
        test_results = [{'id': 'chunk1', 'content': 'test', 'score': 0.9}]

        await store_fts_results('test query', 'default', 10, 'english', test_results)
        assert len(_fts_cache) > 0

        await invalidate_fts_cache_for_workspace('default')
        assert len(_fts_cache) == 0

    @pytest.mark.asyncio
    async def test_empty_results_not_cached(self):
        """Empty results are not cached (by design of caller)."""
        # The store function should handle empty results
        # but the main function only calls store if results exist
        await store_fts_results('test query', 'default', 10, 'english', [])

        # Empty results are still stored (for negative caching)
        cached = await get_cached_fts_results('test query', 'default', 10, 'english')
        assert cached == []


class TestFTSCacheStats:
    """Test cache statistics."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear cache before each test."""
        _fts_cache.clear()
        yield
        _fts_cache.clear()

    @pytest.mark.asyncio
    async def test_stats_empty_cache(self):
        """Stats for empty cache."""
        stats = get_fts_cache_stats()
        assert stats['total_entries'] == 0
        assert stats['valid_entries'] == 0

    @pytest.mark.asyncio
    async def test_stats_with_entries(self):
        """Stats reflect cache entries."""
        await store_fts_results('q1', 'ws', 10, 'en', [{'id': '1'}])
        await store_fts_results('q2', 'ws', 10, 'en', [{'id': '2'}])

        stats = get_fts_cache_stats()
        assert stats['total_entries'] == 2
        assert stats['valid_entries'] == 2

    @pytest.mark.asyncio
    async def test_stats_expired_entries(self):
        """Stats correctly count expired entries."""
        await store_fts_results('q1', 'ws', 10, 'en', [{'id': '1'}])

        # Manually expire one entry
        cache_key = _compute_cache_key('q1', 'ws', 10, 'en')
        _fts_cache[cache_key] = ([{'id': '1'}], 0)

        await store_fts_results('q2', 'ws', 10, 'en', [{'id': '2'}])

        stats = get_fts_cache_stats()
        assert stats['total_entries'] == 2
        assert stats['valid_entries'] == 1  # Only q2 is valid
