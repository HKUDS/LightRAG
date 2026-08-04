"""Unit tests for the daily query activation cache used by 3D viewer idle replay."""

import importlib
import sys
import time

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline


def _make_activation(query: str) -> dict:
    return {
        "timestamp": 1000.0,
        "query": query,
        "entities": ["A", "B"],
        "relationships": [{"src": "A", "tgt": "B"}],
    }


class TestRecordDailyActivation:
    def setup_method(self):
        query_routes._daily_query_cache.clear()

    def test_appends_under_today(self):
        query_routes._record_daily_activation(_make_activation("q1"))
        query_routes._record_daily_activation(_make_activation("q2"))

        day_key = time.strftime("%Y-%m-%d")
        assert list(query_routes._daily_query_cache.keys()) == [day_key]
        assert [a["query"] for a in query_routes._daily_query_cache[day_key]] == [
            "q1",
            "q2",
        ]

    def test_day_rollover_drops_previous_days(self, monkeypatch):
        monkeypatch.setattr(query_routes.time, "strftime", lambda fmt: "2026-08-04")
        query_routes._record_daily_activation(_make_activation("old"))

        monkeypatch.setattr(query_routes.time, "strftime", lambda fmt: "2026-08-05")
        query_routes._record_daily_activation(_make_activation("new"))

        assert list(query_routes._daily_query_cache.keys()) == ["2026-08-05"]
        assert [a["query"] for a in query_routes._daily_query_cache["2026-08-05"]] == [
            "new"
        ]

    def test_cache_is_capped(self, monkeypatch):
        monkeypatch.setattr(query_routes, "_MAX_DAILY_CACHE", 3)
        for i in range(5):
            query_routes._record_daily_activation(_make_activation(f"q{i}"))

        day_key = time.strftime("%Y-%m-%d")
        bucket = query_routes._daily_query_cache[day_key]
        assert len(bucket) == 3
        assert [a["query"] for a in bucket] == ["q2", "q3", "q4"]
