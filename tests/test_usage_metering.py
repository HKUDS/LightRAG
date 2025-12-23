"""
Test suite for API usage metering feature.
Verifies token tracking, usage reporting, and cost estimation.

Feature: 003-api-usage-metering
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from lightrag.utils import TokenTracker
from lightrag.api.models.usage import (
    LLMUsageInfo,
    EmbeddingUsageInfo,
    UsageInfo,
    UsageAggregateResponse,
)


# ============================================================================
# Phase 3: User Story 1 - Per-Request Usage Visibility Tests
# ============================================================================


class TestTokenTracker:
    """Tests for extended TokenTracker functionality."""

    def test_llm_usage_tracking(self):
        """Test that LLM usage is tracked correctly with model name."""
        tracker = TokenTracker()

        # Simulate LLM call
        tracker.add_usage(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="gpt-4o-mini",
        )

        usage = tracker.get_llm_usage()
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["call_count"] == 1
        assert usage["model"] == "gpt-4o-mini"

    def test_multiple_llm_calls(self):
        """Test that multiple LLM calls accumulate correctly."""
        tracker = TokenTracker()

        tracker.add_usage(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="gpt-4o-mini",
        )
        tracker.add_usage(
            {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
            model="gpt-4o-mini",
        )

        usage = tracker.get_llm_usage()
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 150
        assert usage["total_tokens"] == 450
        assert usage["call_count"] == 2

    def test_embedding_usage_tracking(self):
        """Test that embedding usage is tracked separately from LLM."""
        tracker = TokenTracker()

        # Simulate embedding call
        tracker.add_embedding_usage(
            {"prompt_tokens": 50, "total_tokens": 50},
            model="text-embedding-3-small",
        )

        embedding = tracker.get_embedding_usage()
        assert embedding["total_tokens"] == 50
        assert embedding["call_count"] == 1
        assert embedding["model"] == "text-embedding-3-small"

        # LLM should be unaffected
        llm = tracker.get_llm_usage()
        assert llm["call_count"] == 0

    def test_mixed_llm_and_embedding_tracking(self):
        """Test that LLM and embedding usage are tracked independently."""
        tracker = TokenTracker()

        # Simulate mixed calls
        tracker.add_usage(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="gpt-4o-mini",
        )
        tracker.add_embedding_usage(
            {"total_tokens": 25},
            model="text-embedding-3-small",
        )
        tracker.add_usage(
            {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        )
        tracker.add_embedding_usage({"total_tokens": 30})

        llm = tracker.get_llm_usage()
        assert llm["prompt_tokens"] == 150
        assert llm["completion_tokens"] == 75
        assert llm["call_count"] == 2
        assert llm["model"] == "gpt-4o-mini"

        embedding = tracker.get_embedding_usage()
        assert embedding["total_tokens"] == 55
        assert embedding["call_count"] == 2
        assert embedding["model"] == "text-embedding-3-small"

    def test_reset_clears_all_tracking(self):
        """Test that reset clears both LLM and embedding tracking."""
        tracker = TokenTracker()

        tracker.add_usage({"prompt_tokens": 100, "completion_tokens": 50})
        tracker.add_embedding_usage({"total_tokens": 25})

        tracker.reset()

        llm = tracker.get_llm_usage()
        assert llm["call_count"] == 0
        assert llm["model"] is None

        embedding = tracker.get_embedding_usage()
        assert embedding["call_count"] == 0
        assert embedding["model"] is None

    def test_backward_compatible_get_usage(self):
        """Test that get_usage() still works for backward compatibility."""
        tracker = TokenTracker()

        tracker.add_usage({"prompt_tokens": 100, "completion_tokens": 50})

        usage = tracker.get_usage()
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert "call_count" in usage


class TestUsageInfoModel:
    """Tests for UsageInfo Pydantic model."""

    def test_usage_info_from_tracker(self):
        """T011: Test that UsageInfo can be built from TokenTracker."""
        tracker = TokenTracker()

        tracker.add_usage(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="gpt-4o-mini",
        )
        tracker.add_embedding_usage(
            {"total_tokens": 25},
            model="text-embedding-3-small",
        )

        usage_info = UsageInfo.from_token_tracker(tracker)

        assert usage_info.llm is not None
        assert usage_info.llm.prompt_tokens == 100
        assert usage_info.llm.completion_tokens == 50
        assert usage_info.llm.total_tokens == 150
        assert usage_info.llm.calls == 1
        assert usage_info.llm.model == "gpt-4o-mini"

        assert usage_info.embedding is not None
        assert usage_info.embedding.tokens == 25
        assert usage_info.embedding.calls == 1
        assert usage_info.embedding.model == "text-embedding-3-small"

    def test_usage_info_with_cost(self):
        """Test that UsageInfo can include cost estimation."""
        tracker = TokenTracker()
        tracker.add_usage({"prompt_tokens": 1000, "completion_tokens": 500})

        usage_info = UsageInfo.from_token_tracker(tracker, estimated_cost=0.0025)

        assert usage_info.estimated_cost_usd == 0.0025

    def test_usage_info_without_calls(self):
        """T012: Test backward compatibility - empty tracker returns null usage sections."""
        tracker = TokenTracker()

        usage_info = UsageInfo.from_token_tracker(tracker)

        # No calls made, so both should be None (backward compatible)
        assert usage_info.llm is None
        assert usage_info.embedding is None
        assert usage_info.estimated_cost_usd is None

    def test_usage_info_serialization(self):
        """Test that UsageInfo serializes correctly to JSON."""
        usage = UsageInfo(
            llm=LLMUsageInfo(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                calls=1,
                model="gpt-4o-mini",
            ),
            estimated_cost_usd=0.001,
        )

        data = usage.model_dump()
        assert data["llm"]["prompt_tokens"] == 100
        assert data["llm"]["model"] == "gpt-4o-mini"
        assert data["embedding"] is None
        assert data["estimated_cost_usd"] == 0.001


# ============================================================================
# Phase 4: User Story 2 - Embedding Usage Tracking Tests
# ============================================================================


class TestEmbeddingUsageTracking:
    """Tests for separate embedding usage tracking."""

    def test_document_upload_tracks_embedding(self):
        """T018: Test that document upload tracks embedding usage separately."""
        tracker = TokenTracker()

        # Simulate document processing with multiple embedding calls
        tracker.add_embedding_usage({"total_tokens": 100}, model="text-embedding-3-small")
        tracker.add_embedding_usage({"total_tokens": 150})
        tracker.add_embedding_usage({"total_tokens": 200})

        # Also an LLM call for entity extraction
        tracker.add_usage(
            {"prompt_tokens": 500, "completion_tokens": 100},
            model="gpt-4o-mini",
        )

        usage_info = UsageInfo.from_token_tracker(tracker)

        # Embedding should be tracked separately
        assert usage_info.embedding is not None
        assert usage_info.embedding.tokens == 450
        assert usage_info.embedding.calls == 3
        assert usage_info.embedding.model == "text-embedding-3-small"

        # LLM should also be tracked
        assert usage_info.llm is not None
        assert usage_info.llm.prompt_tokens == 500
        assert usage_info.llm.completion_tokens == 100
        assert usage_info.llm.calls == 1

    def test_embedding_only_operation(self):
        """T019: Test operation with only embedding calls (no LLM)."""
        tracker = TokenTracker()

        # Only embedding calls
        tracker.add_embedding_usage({"total_tokens": 50}, model="text-embedding-3-large")

        usage_info = UsageInfo.from_token_tracker(tracker)

        assert usage_info.embedding is not None
        assert usage_info.embedding.tokens == 50
        assert usage_info.llm is None  # No LLM calls made


# ============================================================================
# Phase 5: User Story 3 - Cost Estimation Tests
# ============================================================================


class TestCostEstimation:
    """Tests for cost estimation functionality."""

    def test_cost_included_when_configured(self):
        """T025: Test that cost is included when pricing is configured."""
        tracker = TokenTracker()
        tracker.add_usage({"prompt_tokens": 1000, "completion_tokens": 500})

        # Simulate pricing calculation: 1000 * 0.0015/1000 + 500 * 0.002/1000
        expected_cost = 0.0015 + 0.001  # = 0.0025

        usage_info = UsageInfo.from_token_tracker(tracker, estimated_cost=expected_cost)

        assert usage_info.estimated_cost_usd == expected_cost

    def test_cost_omitted_when_not_configured(self):
        """T026: Test that cost is omitted (null) when pricing not configured."""
        tracker = TokenTracker()
        tracker.add_usage({"prompt_tokens": 1000, "completion_tokens": 500})

        usage_info = UsageInfo.from_token_tracker(tracker)

        assert usage_info.estimated_cost_usd is None

    def test_cost_serialization_omits_null(self):
        """Test that null cost is excluded from JSON when using exclude_none."""
        usage = UsageInfo(
            llm=LLMUsageInfo(prompt_tokens=100, completion_tokens=50, calls=1),
        )

        data = usage.model_dump(exclude_none=True)
        assert "estimated_cost_usd" not in data


# ============================================================================
# Phase 6: User Story 4 - Usage Aggregation Tests
# ============================================================================


class TestUsageAggregation:
    """Tests for workspace usage aggregation."""

    def test_aggregate_response_model(self):
        """T032: Test UsageAggregateResponse model structure."""
        from datetime import date

        response = UsageAggregateResponse(
            workspace="test-workspace",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            llm={
                "total_prompt_tokens": 10000,
                "total_completion_tokens": 5000,
                "total_calls": 100,
            },
            embedding={
                "total_tokens": 50000,
                "total_calls": 500,
            },
            total_estimated_cost_usd=2.50,
            request_count=150,
        )

        assert response.workspace == "test-workspace"
        assert response.request_count == 150
        assert response.llm["total_prompt_tokens"] == 10000
        assert response.embedding["total_tokens"] == 50000

    def test_aggregate_empty_period(self):
        """Test aggregate response for period with no activity."""
        from datetime import date

        response = UsageAggregateResponse(
            workspace="empty-workspace",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            request_count=0,
        )

        assert response.request_count == 0
        assert response.llm is None
        assert response.embedding is None


# ============================================================================
# Multi-workspace Isolation Tests
# ============================================================================


class TestWorkspaceIsolation:
    """T033, T047: Tests for multi-workspace usage isolation."""

    def test_independent_trackers_per_workspace(self):
        """Test that each workspace has independent usage tracking."""
        # Simulate separate trackers for different workspaces
        tracker_ws1 = TokenTracker()
        tracker_ws2 = TokenTracker()

        tracker_ws1.add_usage({"prompt_tokens": 100, "completion_tokens": 50})
        tracker_ws2.add_usage({"prompt_tokens": 200, "completion_tokens": 100})

        usage_ws1 = UsageInfo.from_token_tracker(tracker_ws1)
        usage_ws2 = UsageInfo.from_token_tracker(tracker_ws2)

        assert usage_ws1.llm.prompt_tokens == 100
        assert usage_ws2.llm.prompt_tokens == 200

        # Workspaces are independent
        assert usage_ws1.llm.prompt_tokens != usage_ws2.llm.prompt_tokens
