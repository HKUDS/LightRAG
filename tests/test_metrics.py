"""
Unit tests for metrics collection infrastructure.
"""

import pytest
import time

from lightrag.metrics import (
    MetricsCollector,
    QueryMetric,
    get_metrics_collector,
    record_query_metric,
)


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_initialization(self):
        """Collector initializes with empty history."""
        collector = MetricsCollector()
        assert collector.total_queries == 0
        assert collector.total_llm_calls == 0
        assert collector.total_llm_cache_hits == 0
        assert collector.total_embed_calls == 0
        assert len(collector.query_history) == 0

    def test_record_query(self):
        """Query metrics are recorded correctly."""
        collector = MetricsCollector()

        metric = QueryMetric(
            timestamp=time.time(),
            duration_ms=100.0,
            mode='mix',
            cache_hit=False,
            entities_count=10,
            relations_count=20,
            chunks_count=5,
            tokens_used=1000,
        )
        collector.record_query(metric)

        assert collector.total_queries == 1
        assert collector.total_llm_calls == 1
        assert len(collector.query_history) == 1

    def test_record_query_cache_hit(self):
        """Cache hits are recorded correctly."""
        collector = MetricsCollector()

        metric = QueryMetric(
            timestamp=time.time(),
            duration_ms=50.0,
            mode='mix',
            cache_hit=True,
            entities_count=10,
            relations_count=20,
            chunks_count=5,
            tokens_used=1000,
        )
        collector.record_query(metric)

        assert collector.total_queries == 1
        assert collector.total_llm_cache_hits == 1
        assert collector.total_llm_calls == 0

    def test_record_llm_call(self):
        """LLM calls are tracked."""
        collector = MetricsCollector()

        collector.record_llm_call(cache_hit=False)
        assert collector.total_llm_calls == 1
        assert collector.total_llm_cache_hits == 0

        collector.record_llm_call(cache_hit=True)
        assert collector.total_llm_calls == 1
        assert collector.total_llm_cache_hits == 1

    def test_record_embed_call(self):
        """Embedding calls are tracked."""
        collector = MetricsCollector()

        collector.record_embed_call()
        collector.record_embed_call()
        assert collector.total_embed_calls == 2


class TestPercentileCalculation:
    """Test percentile calculation."""

    def test_empty_values(self):
        """Empty values return 0.0."""
        collector = MetricsCollector()
        assert collector._get_percentile([], 50) == 0.0

    def test_single_value(self):
        """Single value returns that value for any percentile."""
        collector = MetricsCollector()
        assert collector._get_percentile([100.0], 50) == 100.0
        assert collector._get_percentile([100.0], 99) == 100.0

    def test_p50_median(self):
        """P50 returns median."""
        collector = MetricsCollector()
        # Sorted: [10, 20, 30, 40, 50]
        values = [30, 10, 50, 20, 40]
        p50 = collector._get_percentile(values, 50)
        assert p50 == 30.0

    def test_p99_high(self):
        """P99 returns high value."""
        collector = MetricsCollector()
        values = list(range(1, 101))  # 1-100
        p99 = collector._get_percentile(values, 99)
        assert p99 >= 99.0


class TestComputeStats:
    """Test statistics computation."""

    def test_empty_stats(self):
        """Stats for empty collector."""
        collector = MetricsCollector()
        stats = collector.compute_stats()

        assert stats['total_queries'] == 0
        assert stats['queries_in_window'] == 0
        assert stats['latency_percentiles'] is None
        assert stats['cache_stats']['hit_rate'] == 0.0

    def test_stats_with_queries(self):
        """Stats with recorded queries."""
        collector = MetricsCollector()

        # Record some queries
        for i in range(5):
            collector.record_query(
                QueryMetric(
                    timestamp=time.time(),
                    duration_ms=100.0 + i * 10,
                    mode='mix',
                    cache_hit=False,
                    entities_count=10,
                    relations_count=20,
                    chunks_count=5,
                    tokens_used=1000,
                )
            )

        stats = collector.compute_stats()

        assert stats['total_queries'] == 5
        assert stats['queries_in_window'] == 5
        assert stats['latency_percentiles'] is not None
        assert stats['latency_percentiles']['p50'] >= 100.0

    def test_stats_window_filtering(self):
        """Stats filter by time window."""
        collector = MetricsCollector()

        # Record old query (outside window)
        old_metric = QueryMetric(
            timestamp=time.time() - 7200,  # 2 hours ago
            duration_ms=100.0,
            mode='mix',
            cache_hit=False,
            entities_count=10,
            relations_count=20,
            chunks_count=5,
            tokens_used=1000,
        )
        collector.record_query(old_metric)

        # Record recent query
        recent_metric = QueryMetric(
            timestamp=time.time(),
            duration_ms=200.0,
            mode='mix',
            cache_hit=False,
            entities_count=10,
            relations_count=20,
            chunks_count=5,
            tokens_used=1000,
        )
        collector.record_query(recent_metric)

        stats = collector.compute_stats(window_seconds=3600)  # 1 hour window

        assert stats['total_queries'] == 2
        assert stats['queries_in_window'] == 1  # Only recent one

    def test_cache_hit_rate(self):
        """Cache hit rate calculation."""
        collector = MetricsCollector()

        # 2 cache hits, 3 cache misses
        collector.total_llm_cache_hits = 2
        collector.total_llm_calls = 3

        stats = collector.compute_stats()

        expected_rate = 2 / 5  # 2 hits out of 5 total
        assert abs(stats['cache_stats']['hit_rate'] - expected_rate) < 0.01

    def test_mode_distribution(self):
        """Mode distribution tracking."""
        collector = MetricsCollector()

        for mode, count in [('mix', 3), ('local', 2), ('global', 1)]:
            for _ in range(count):
                collector.record_query(
                    QueryMetric(
                        timestamp=time.time(),
                        duration_ms=100.0,
                        mode=mode,
                        cache_hit=False,
                        entities_count=10,
                        relations_count=20,
                        chunks_count=5,
                        tokens_used=1000,
                    )
                )

        stats = collector.compute_stats()

        assert stats['mode_distribution']['mix'] == 3
        assert stats['mode_distribution']['local'] == 2
        assert stats['mode_distribution']['global'] == 1


class TestRecentQueries:
    """Test recent queries retrieval."""

    def test_recent_queries_empty(self):
        """Empty collector returns empty list."""
        collector = MetricsCollector()
        recent = collector.get_recent_queries()
        assert recent == []

    def test_recent_queries_limited(self):
        """Limit parameter is respected."""
        collector = MetricsCollector()

        for i in range(10):
            collector.record_query(
                QueryMetric(
                    timestamp=time.time(),
                    duration_ms=100.0,
                    mode='mix',
                    cache_hit=False,
                    entities_count=i,
                    relations_count=20,
                    chunks_count=5,
                    tokens_used=1000,
                )
            )

        recent = collector.get_recent_queries(limit=3)
        assert len(recent) == 3

    def test_recent_queries_order(self):
        """Most recent queries are first."""
        collector = MetricsCollector()

        for i in range(3):
            collector.record_query(
                QueryMetric(
                    timestamp=time.time() + i,
                    duration_ms=100.0 + i,
                    mode='mix',
                    cache_hit=False,
                    entities_count=i,
                    relations_count=20,
                    chunks_count=5,
                    tokens_used=1000,
                )
            )

        recent = collector.get_recent_queries()

        # Most recent (i=2) should be first
        assert recent[0]['entities_count'] == 2
        assert recent[1]['entities_count'] == 1
        assert recent[2]['entities_count'] == 0


class TestGlobalMetricsCollector:
    """Test global metrics collector singleton."""

    @pytest.mark.asyncio
    async def test_get_collector_singleton(self):
        """Same collector returned on multiple calls."""
        c1 = await get_metrics_collector()
        c2 = await get_metrics_collector()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_record_query_metric_helper(self):
        """Convenience function records metrics."""
        collector = await get_metrics_collector()
        initial_count = collector.total_queries

        await record_query_metric(
            duration_ms=100.0,
            mode='mix',
            cache_hit=False,
            entities_count=10,
            relations_count=20,
            chunks_count=5,
            tokens_used=1000,
        )

        assert collector.total_queries == initial_count + 1
