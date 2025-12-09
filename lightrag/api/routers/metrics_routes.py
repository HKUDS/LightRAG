"""
Metrics routes for operational monitoring.

This module provides endpoints for retrieving LightRAG operational metrics
including query latency percentiles, cache hit rates, and API call counts.

Endpoints:
- GET /metrics: Aggregated operational metrics
- GET /metrics/queries: Recent query details
"""

from typing import Any, ClassVar

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.cache.fts_cache import get_fts_cache_stats
from lightrag.metrics import get_metrics_collector
from lightrag.utils import logger, statistic_data


class LatencyPercentiles(BaseModel):
    """Query latency percentiles in milliseconds."""

    p50: float = Field(description='50th percentile (median) latency')
    p95: float = Field(description='95th percentile latency')
    p99: float = Field(description='99th percentile latency')
    avg: float = Field(description='Average latency')
    min: float = Field(description='Minimum latency')
    max: float = Field(description='Maximum latency')


class CacheStats(BaseModel):
    """LLM cache statistics."""

    total_llm_calls: int = Field(description='Total LLM API calls')
    total_cache_hits: int = Field(description='Total cache hits')
    hit_rate: float = Field(description='Cache hit rate (0.0 to 1.0)')


class EmbedStats(BaseModel):
    """Embedding statistics."""

    total_calls: int = Field(description='Total embedding API calls')


class FTSCacheStats(BaseModel):
    """Full-text search cache statistics."""

    enabled: bool = Field(description='Whether FTS caching is enabled')
    total_entries: int = Field(description='Total entries in cache')
    valid_entries: int = Field(description='Non-expired entries')
    max_size: int = Field(description='Maximum cache size')
    ttl_seconds: int = Field(description='Cache TTL in seconds')
    redis_enabled: bool = Field(description='Whether Redis caching is enabled')


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    uptime_seconds: float = Field(description='Server uptime in seconds')
    total_queries: int = Field(description='Total queries processed')
    queries_in_window: int = Field(description='Queries in the time window')
    window_seconds: float = Field(description='Time window for percentile calculations')
    latency_percentiles: LatencyPercentiles | None = Field(
        default=None, description='Query latency percentiles (null if no queries in window)'
    )
    cache_stats: CacheStats
    embed_stats: EmbedStats
    fts_cache_stats: FTSCacheStats
    mode_distribution: dict[str, int] = Field(
        description='Query count by mode in the time window'
    )
    legacy_stats: dict[str, int] = Field(
        description='Legacy statistics from utils.statistic_data'
    )

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            'example': {
                'uptime_seconds': 3600.5,
                'total_queries': 150,
                'queries_in_window': 45,
                'window_seconds': 3600,
                'latency_percentiles': {
                    'p50': 234.5,
                    'p95': 890.2,
                    'p99': 1200.0,
                    'avg': 345.6,
                    'min': 100.0,
                    'max': 2500.0,
                },
                'cache_stats': {
                    'total_llm_calls': 100,
                    'total_cache_hits': 50,
                    'hit_rate': 0.33,
                },
                'embed_stats': {'total_calls': 200},
                'fts_cache_stats': {
                    'enabled': True,
                    'total_entries': 150,
                    'valid_entries': 140,
                    'max_size': 5000,
                    'ttl_seconds': 300,
                    'redis_enabled': False,
                },
                'mode_distribution': {'mix': 30, 'local': 10, 'global': 5},
                'legacy_stats': {'llm_call': 100, 'llm_cache': 50, 'embed_call': 200},
            }
        }


class QueryMetricItem(BaseModel):
    """A single query metric entry."""

    timestamp: float = Field(description='Unix timestamp of query')
    duration_ms: float = Field(description='Query duration in milliseconds')
    mode: str = Field(description='Query mode used')
    cache_hit: bool = Field(description='Whether LLM cache was hit')
    entities_count: int = Field(description='Number of entities retrieved')
    relations_count: int = Field(description='Number of relations retrieved')
    chunks_count: int = Field(description='Number of chunks retrieved')
    tokens_used: int = Field(description='Total tokens used')


class RecentQueriesResponse(BaseModel):
    """Response model for recent queries endpoint."""

    queries: list[QueryMetricItem] = Field(description='Recent query metrics')
    count: int = Field(description='Number of queries returned')


def create_metrics_routes(api_key: str | None = None) -> APIRouter:
    """Create metrics routes for operational monitoring.

    Args:
        api_key: Optional API key for authentication

    Returns:
        FastAPI router with metrics endpoints
    """
    router = APIRouter(tags=['metrics'])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        '/metrics',
        response_model=MetricsResponse,
        dependencies=[Depends(combined_auth)],
        summary='Get operational metrics',
        description=(
            'Returns lightweight operational metrics including latency percentiles, '
            'cache hit rates, and API call counts. Metrics are computed on-demand '
            'from a circular buffer of recent queries.'
        ),
    )
    async def get_metrics(
        window: float = Query(
            default=3600.0,
            ge=60.0,
            le=86400.0,
            description='Time window in seconds for percentile calculations (60s to 24h)',
        ),
    ) -> MetricsResponse:
        """Get operational metrics."""
        try:
            collector = await get_metrics_collector()
            stats = collector.compute_stats(window_seconds=window)
            fts_stats = get_fts_cache_stats()

            return MetricsResponse(
                uptime_seconds=stats['uptime_seconds'],
                total_queries=stats['total_queries'],
                queries_in_window=stats['queries_in_window'],
                window_seconds=stats['window_seconds'],
                latency_percentiles=(
                    LatencyPercentiles(**stats['latency_percentiles'])
                    if stats['latency_percentiles']
                    else None
                ),
                cache_stats=CacheStats(**stats['cache_stats']),
                embed_stats=EmbedStats(**stats['embed_stats']),
                fts_cache_stats=FTSCacheStats(**fts_stats),
                mode_distribution=stats['mode_distribution'],
                legacy_stats=statistic_data.copy(),
            )
        except Exception as e:
            logger.error(f'Error getting metrics: {e}', exc_info=True)
            raise

    @router.get(
        '/metrics/queries',
        response_model=RecentQueriesResponse,
        dependencies=[Depends(combined_auth)],
        summary='Get recent query metrics',
        description='Returns detailed metrics for recent queries.',
    )
    async def get_recent_queries(
        limit: int = Query(
            default=10,
            ge=1,
            le=100,
            description='Maximum number of recent queries to return',
        ),
    ) -> RecentQueriesResponse:
        """Get recent query metrics."""
        try:
            collector = await get_metrics_collector()
            queries = collector.get_recent_queries(limit=limit)

            return RecentQueriesResponse(
                queries=[QueryMetricItem(**q) for q in queries],
                count=len(queries),
            )
        except Exception as e:
            logger.error(f'Error getting recent queries: {e}', exc_info=True)
            raise

    return router
