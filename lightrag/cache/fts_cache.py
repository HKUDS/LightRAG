"""
Full-text search result caching for LightRAG.

Follows the same dual-tier pattern as operate.py's query embedding cache:
- Local LRU cache for single-process performance
- Optional Redis for cross-worker sharing

Cache invalidation strategy:
- TTL-based expiration (default 5 minutes - shorter than embedding cache since content changes)
- Manual invalidation on document changes via invalidate_fts_cache_for_workspace()

Environment Variables:
    FTS_CACHE_ENABLED: Enable/disable FTS caching (default: true)
    FTS_CACHE_TTL: Cache TTL in seconds (default: 300 = 5 minutes)
    FTS_CACHE_MAX_SIZE: Maximum local cache entries (default: 5000)
    REDIS_FTS_CACHE: Enable Redis for cross-worker caching (default: false)
    REDIS_URI: Redis connection URI (default: redis://localhost:6379)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any

from lightrag.utils import logger

# Configuration from environment (matching operate.py pattern)
FTS_CACHE_TTL = int(os.getenv('FTS_CACHE_TTL', '300'))  # 5 minutes
FTS_CACHE_MAX_SIZE = int(os.getenv('FTS_CACHE_MAX_SIZE', '5000'))
FTS_CACHE_ENABLED = os.getenv('FTS_CACHE_ENABLED', 'true').lower() == 'true'
REDIS_FTS_CACHE_ENABLED = os.getenv('REDIS_FTS_CACHE', 'false').lower() == 'true'
REDIS_URI = os.getenv('REDIS_URI', 'redis://localhost:6379')

# Local in-memory cache: {cache_key: (results, timestamp)}
_fts_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_fts_cache_lock = asyncio.Lock()

# Redis client (lazy initialized, can be shared with embedding cache)
_redis_client = None


def _compute_cache_key(query: str, workspace: str, limit: int, language: str) -> str:
    """Compute deterministic cache key from search parameters.

    Uses SHA256 truncated to 16 characters for compact keys while
    maintaining very low collision probability.
    """
    key_data = f'{query}|{workspace}|{limit}|{language}'
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


async def _get_redis_client():
    """Lazy initialize Redis client for FTS cache."""
    global _redis_client
    if _redis_client is None and REDIS_FTS_CACHE_ENABLED:
        try:
            import redis.asyncio as redis

            _redis_client = redis.from_url(REDIS_URI, decode_responses=True)
            await _redis_client.ping()
            logger.info(f'Redis FTS cache connected: {REDIS_URI}')
        except ImportError:
            logger.warning('Redis package not installed for FTS cache')
            return None
        except Exception as e:
            logger.warning(f'Failed to connect to Redis for FTS cache: {e}')
            return None
    return _redis_client


async def get_cached_fts_results(
    query: str,
    workspace: str,
    limit: int,
    language: str = 'english',
) -> list[dict[str, Any]] | None:
    """Get cached FTS results if available and not expired.

    Args:
        query: Search query string
        workspace: Workspace identifier
        limit: Maximum results limit
        language: Language for text search (default: english)

    Returns:
        Cached results list if cache hit, None if cache miss or disabled
    """
    if not FTS_CACHE_ENABLED:
        return None

    cache_key = _compute_cache_key(query, workspace, limit, language)
    redis_key = f'lightrag:fts:{cache_key}'
    current_time = time.time()

    # Try Redis first (if enabled)
    if REDIS_FTS_CACHE_ENABLED:
        try:
            redis_client = await _get_redis_client()
            if redis_client:
                cached_json = await redis_client.get(redis_key)
                if cached_json:
                    results = json.loads(cached_json)
                    logger.debug(f'Redis FTS cache hit for hash {cache_key[:8]}')
                    # Update local cache for faster subsequent hits
                    _fts_cache[cache_key] = (results, current_time)
                    return results
        except Exception as e:
            logger.debug(f'Redis FTS cache read error: {e}')

    # Check local cache
    cached = _fts_cache.get(cache_key)
    if cached and (current_time - cached[1]) < FTS_CACHE_TTL:
        logger.debug(f'Local FTS cache hit for hash {cache_key[:8]}')
        return cached[0]

    return None


async def store_fts_results(
    query: str,
    workspace: str,
    limit: int,
    language: str,
    results: list[dict[str, Any]],
) -> None:
    """Store FTS results in cache.

    Args:
        query: Search query string
        workspace: Workspace identifier
        limit: Maximum results limit
        language: Language for text search
        results: Search results to cache
    """
    if not FTS_CACHE_ENABLED:
        return

    cache_key = _compute_cache_key(query, workspace, limit, language)
    redis_key = f'lightrag:fts:{cache_key}'
    current_time = time.time()

    # Manage local cache size - LRU eviction
    async with _fts_cache_lock:
        if len(_fts_cache) >= FTS_CACHE_MAX_SIZE:
            # Remove oldest 10% of entries
            sorted_entries = sorted(_fts_cache.items(), key=lambda x: x[1][1])
            for old_key, _ in sorted_entries[: FTS_CACHE_MAX_SIZE // 10]:
                del _fts_cache[old_key]

        _fts_cache[cache_key] = (results, current_time)

    # Store in Redis (if enabled)
    if REDIS_FTS_CACHE_ENABLED:
        try:
            redis_client = await _get_redis_client()
            if redis_client:
                await redis_client.setex(
                    redis_key,
                    FTS_CACHE_TTL,
                    json.dumps(results),
                )
                logger.debug(f'FTS results cached in Redis for hash {cache_key[:8]}')
        except Exception as e:
            logger.debug(f'Redis FTS cache write error: {e}')


async def invalidate_fts_cache_for_workspace(workspace: str) -> int:
    """Invalidate all FTS cache entries for a workspace.

    Call this when documents are added/modified/deleted in a workspace.

    Note: Since cache keys are hashes and don't encode workspace in a
    recoverable way, this clears all local cache entries. For Redis,
    it clears all FTS cache entries. In practice, this is acceptable
    because:
    1. Document changes are relatively infrequent
    2. FTS cache TTL is short (5 minutes)
    3. Cache rebuild cost is low

    Args:
        workspace: Workspace identifier (for logging)

    Returns:
        Number of entries invalidated
    """
    invalidated = 0

    # Clear local cache
    async with _fts_cache_lock:
        invalidated = len(_fts_cache)
        _fts_cache.clear()
        logger.info(f'FTS cache invalidated for workspace {workspace}: cleared {invalidated} local entries')

    # Clear Redis FTS cache (if enabled)
    if REDIS_FTS_CACHE_ENABLED:
        try:
            redis_client = await _get_redis_client()
            if redis_client:
                # Use SCAN to find and delete FTS cache keys
                cursor = 0
                redis_deleted = 0
                while True:
                    cursor, keys = await redis_client.scan(
                        cursor=cursor,
                        match='lightrag:fts:*',
                        count=100,
                    )
                    if keys:
                        await redis_client.delete(*keys)
                        redis_deleted += len(keys)
                    if cursor == 0:
                        break
                if redis_deleted > 0:
                    logger.info(f'FTS cache invalidated in Redis: cleared {redis_deleted} entries')
                invalidated += redis_deleted
        except Exception as e:
            logger.warning(f'Redis FTS cache invalidation error: {e}')

    return invalidated


def get_fts_cache_stats() -> dict[str, Any]:
    """Get current FTS cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    current_time = time.time()

    # Count valid (non-expired) entries
    valid_entries = sum(1 for _, (_, ts) in _fts_cache.items() if (current_time - ts) < FTS_CACHE_TTL)

    return {
        'enabled': FTS_CACHE_ENABLED,
        'total_entries': len(_fts_cache),
        'valid_entries': valid_entries,
        'max_size': FTS_CACHE_MAX_SIZE,
        'ttl_seconds': FTS_CACHE_TTL,
        'redis_enabled': REDIS_FTS_CACHE_ENABLED,
    }
