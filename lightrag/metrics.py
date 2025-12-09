"""
Metrics collection and aggregation for LightRAG.

This module provides lightweight metrics collection without heavy computation
on each request. Metrics are collected incrementally and aggregated on demand.

Key features:
- Circular buffer for query history (bounded memory)
- Percentile calculations (p50, p95, p99)
- Cache hit rate tracking
- Thread-safe async operations

Environment Variables:
    ENABLE_METRICS: Enable/disable metrics collection (default: true)
    METRICS_HISTORY_SIZE: Number of queries to keep in buffer (default: 1000)
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from lightrag.constants import (
    DEFAULT_METRICS_ENABLED,
    DEFAULT_METRICS_HISTORY_SIZE,
    DEFAULT_METRICS_WINDOW_SECONDS,
)

# Configuration
METRICS_ENABLED = os.getenv('ENABLE_METRICS', str(DEFAULT_METRICS_ENABLED)).lower() == 'true'
METRICS_HISTORY_SIZE = int(os.getenv('METRICS_HISTORY_SIZE', str(DEFAULT_METRICS_HISTORY_SIZE)))

# Global metrics instance (singleton pattern)
_metrics_instance: MetricsCollector | None = None
_metrics_lock = asyncio.Lock()


@dataclass
class QueryMetric:
    """Single query metric entry."""

    timestamp: float
    duration_ms: float
    mode: str
    cache_hit: bool
    entities_count: int
    relations_count: int
    chunks_count: int
    tokens_used: int


@dataclass
class MetricsCollector:
    """Collects and aggregates LightRAG metrics.

    Uses a circular buffer to store recent query metrics, preventing
    unbounded memory growth. Aggregations are computed on-demand.
    """

    # Circular buffer for recent queries
    query_history: deque[QueryMetric] = field(
        default_factory=lambda: deque(maxlen=METRICS_HISTORY_SIZE)
    )

    # Counters (monotonically increasing, never reset)
    total_queries: int = 0
    total_llm_calls: int = 0
    total_llm_cache_hits: int = 0
    total_embed_calls: int = 0

    # Server start timestamp
    started_at: float = field(default_factory=time.time)

    def record_query(self, metric: QueryMetric) -> None:
        """Record a query metric.

        Args:
            metric: QueryMetric instance with query details
        """
        if not METRICS_ENABLED:
            return

        self.query_history.append(metric)
        self.total_queries += 1
        if metric.cache_hit:
            self.total_llm_cache_hits += 1
        else:
            self.total_llm_calls += 1

    def record_llm_call(self, cache_hit: bool = False) -> None:
        """Record an LLM API call.

        Args:
            cache_hit: Whether this was a cache hit
        """
        if cache_hit:
            self.total_llm_cache_hits += 1
        else:
            self.total_llm_calls += 1

    def record_embed_call(self) -> None:
        """Record an embedding API call."""
        self.total_embed_calls += 1

    def _get_percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile from a list of values.

        Uses linear interpolation between nearest ranks.

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Calculated percentile value, or 0.0 if empty
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n == 1:
            return sorted_values[0]

        # Calculate the rank
        k = (n - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f < n - 1 else f

        # Linear interpolation
        return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

    def compute_stats(
        self,
        window_seconds: float = DEFAULT_METRICS_WINDOW_SECONDS,
    ) -> dict[str, Any]:
        """Compute aggregated statistics.

        Args:
            window_seconds: Time window for percentile calculations (default 1 hour)

        Returns:
            Dictionary with computed metrics
        """
        now = time.time()
        cutoff = now - window_seconds

        # Filter recent queries within window
        recent = [q for q in self.query_history if q.timestamp >= cutoff]

        # Extract latencies
        latencies = [q.duration_ms for q in recent] if recent else []

        # Calculate percentiles
        latency_percentiles = None
        if latencies:
            latency_percentiles = {
                'p50': round(self._get_percentile(latencies, 50), 2),
                'p95': round(self._get_percentile(latencies, 95), 2),
                'p99': round(self._get_percentile(latencies, 99), 2),
                'avg': round(sum(latencies) / len(latencies), 2),
                'min': round(min(latencies), 2),
                'max': round(max(latencies), 2),
            }

        # Calculate cache hit rate
        total_llm_ops = self.total_llm_calls + self.total_llm_cache_hits
        cache_hit_rate = (
            self.total_llm_cache_hits / total_llm_ops if total_llm_ops > 0 else 0.0
        )

        # Mode distribution in window
        mode_counts: dict[str, int] = {}
        for q in recent:
            mode_counts[q.mode] = mode_counts.get(q.mode, 0) + 1

        return {
            'uptime_seconds': round(now - self.started_at, 2),
            'total_queries': self.total_queries,
            'queries_in_window': len(recent),
            'window_seconds': window_seconds,
            'latency_percentiles': latency_percentiles,
            'cache_stats': {
                'total_llm_calls': self.total_llm_calls,
                'total_cache_hits': self.total_llm_cache_hits,
                'hit_rate': round(cache_hit_rate, 4),
            },
            'embed_stats': {
                'total_calls': self.total_embed_calls,
            },
            'mode_distribution': mode_counts,
        }

    def get_recent_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent query metrics.

        Args:
            limit: Maximum number of recent queries to return

        Returns:
            List of recent query metrics as dictionaries
        """
        recent = list(self.query_history)[-limit:]
        return [
            {
                'timestamp': q.timestamp,
                'duration_ms': q.duration_ms,
                'mode': q.mode,
                'cache_hit': q.cache_hit,
                'entities_count': q.entities_count,
                'relations_count': q.relations_count,
                'chunks_count': q.chunks_count,
                'tokens_used': q.tokens_used,
            }
            for q in reversed(recent)  # Most recent first
        ]


async def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance.

    Thread-safe singleton pattern using async lock.

    Returns:
        The global MetricsCollector instance
    """
    global _metrics_instance
    async with _metrics_lock:
        if _metrics_instance is None:
            _metrics_instance = MetricsCollector()
        return _metrics_instance


def get_metrics_collector_sync() -> MetricsCollector:
    """Get or create the global metrics collector instance (sync version).

    Note: This should only be used in contexts where async is not available.
    Prefer get_metrics_collector() for async code.

    Returns:
        The global MetricsCollector instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


async def record_query_metric(
    duration_ms: float,
    mode: str,
    cache_hit: bool = False,
    entities_count: int = 0,
    relations_count: int = 0,
    chunks_count: int = 0,
    tokens_used: int = 0,
) -> None:
    """Convenience function to record a query metric.

    Args:
        duration_ms: Query duration in milliseconds
        mode: Query mode (local, global, hybrid, mix, naive)
        cache_hit: Whether LLM cache was hit
        entities_count: Number of entities retrieved
        relations_count: Number of relations retrieved
        chunks_count: Number of chunks retrieved
        tokens_used: Total tokens used in context
    """
    collector = await get_metrics_collector()
    collector.record_query(
        QueryMetric(
            timestamp=time.time(),
            duration_ms=duration_ms,
            mode=mode,
            cache_hit=cache_hit,
            entities_count=entities_count,
            relations_count=relations_count,
            chunks_count=chunks_count,
            tokens_used=tokens_used,
        )
    )
