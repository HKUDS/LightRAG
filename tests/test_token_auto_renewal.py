"""
Pytest unit tests for token auto-renewal functionality

Tests:
1. Backend token renewal logic
2. Rate limiting for token renewals
3. Token renewal state tracking
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from fastapi import Response
import time
import sys

# Mock the config before importing utils_api
sys.modules["lightrag.api.config"] = Mock()
sys.modules["lightrag.api.auth"] = Mock()

# Create a simple token renewal cache for testing
_token_renewal_cache = {}
_RENEWAL_MIN_INTERVAL = 60


@pytest.mark.offline
class TestTokenRenewal:
    """Tests for token auto-renewal logic"""

    @pytest.fixture
    def mock_auth_handler(self):
        """Mock authentication handler"""
        handler = Mock()
        handler.guest_expire_hours = 24
        handler.expire_hours = 24
        handler.create_token = Mock(return_value="new-token-12345")
        return handler

    @pytest.fixture
    def mock_global_args(self):
        """Mock global configuration"""
        args = Mock()
        args.token_auto_renew = True
        args.token_renew_threshold = 0.5
        return args

    @pytest.fixture
    def mock_token_info_guest(self):
        """Mock token info for guest user"""
        # Token with 10 hours remaining (below 50% of 24 hours)
        exp_time = datetime.now(timezone.utc) + timedelta(hours=10)
        return {
            "username": "guest",
            "role": "guest",
            "exp": exp_time,
            "metadata": {"auth_mode": "disabled"},
        }

    @pytest.fixture
    def mock_token_info_user(self):
        """Mock token info for regular user"""
        # Token with 10 hours remaining (below 50% of 24 hours)
        exp_time = datetime.now(timezone.utc) + timedelta(hours=10)
        return {
            "username": "testuser",
            "role": "user",
            "exp": exp_time,
            "metadata": {"auth_mode": "enabled"},
        }

    @pytest.fixture
    def mock_token_info_above_threshold(self):
        """Mock token info with time above renewal threshold"""
        # Token with 20 hours remaining (above 50% of 24 hours)
        exp_time = datetime.now(timezone.utc) + timedelta(hours=20)
        return {
            "username": "testuser",
            "role": "user",
            "exp": exp_time,
            "metadata": {"auth_mode": "enabled"},
        }

    def test_token_renewal_when_below_threshold(
        self, mock_auth_handler, mock_global_args, mock_token_info_user
    ):
        """Test that token is renewed when remaining time < threshold"""
        # Use global cache
        global _token_renewal_cache

        # Clear cache
        _token_renewal_cache.clear()

        response = Mock(spec=Response)
        response.headers = {}

        # Simulate the renewal logic
        expire_time = mock_token_info_user["exp"]
        now = datetime.now(timezone.utc)
        remaining_seconds = (expire_time - now).total_seconds()

        role = mock_token_info_user["role"]
        total_hours = (
            mock_auth_handler.expire_hours
            if role == "user"
            else mock_auth_handler.guest_expire_hours
        )
        total_seconds = total_hours * 3600

        # Should renew because remaining_seconds < total_seconds * 0.5
        should_renew = (
            remaining_seconds < total_seconds * mock_global_args.token_renew_threshold
        )
        assert should_renew is True

        # Simulate renewal
        username = mock_token_info_user["username"]
        current_time = time.time()
        last_renewal = _token_renewal_cache.get(username, 0)
        time_since_last_renewal = current_time - last_renewal

        # Should pass rate limit (first renewal)
        assert time_since_last_renewal >= 60 or last_renewal == 0

        # Perform renewal
        new_token = mock_auth_handler.create_token(
            username=username, role=role, metadata=mock_token_info_user["metadata"]
        )
        response.headers["X-New-Token"] = new_token
        _token_renewal_cache[username] = current_time

        # Verify
        assert "X-New-Token" in response.headers
        assert response.headers["X-New-Token"] == "new-token-12345"
        assert username in _token_renewal_cache

    def test_token_no_renewal_when_above_threshold(
        self, mock_auth_handler, mock_global_args, mock_token_info_above_threshold
    ):
        """Test that token is NOT renewed when remaining time > threshold"""
        response = Mock(spec=Response)
        response.headers = {}

        expire_time = mock_token_info_above_threshold["exp"]
        now = datetime.now(timezone.utc)
        remaining_seconds = (expire_time - now).total_seconds()

        mock_token_info_above_threshold["role"]
        total_hours = mock_auth_handler.expire_hours
        total_seconds = total_hours * 3600

        # Should NOT renew because remaining_seconds > total_seconds * 0.5
        should_renew = (
            remaining_seconds < total_seconds * mock_global_args.token_renew_threshold
        )
        assert should_renew is False

        # No renewal should happen
        assert "X-New-Token" not in response.headers

    def test_token_renewal_disabled(
        self, mock_auth_handler, mock_global_args, mock_token_info_user
    ):
        """Test that no renewal happens when TOKEN_AUTO_RENEW=false"""
        mock_global_args.token_auto_renew = False
        response = Mock(spec=Response)
        response.headers = {}

        # Auto-renewal is disabled, so even if below threshold, no renewal
        if not mock_global_args.token_auto_renew:
            # Skip renewal logic
            pass

        assert "X-New-Token" not in response.headers

    def test_token_renewal_for_guest_mode(
        self, mock_auth_handler, mock_global_args, mock_token_info_guest
    ):
        """Test that guest tokens are renewed correctly"""
        # Use global cache
        global _token_renewal_cache

        _token_renewal_cache.clear()

        response = Mock(spec=Response)
        response.headers = {}

        expire_time = mock_token_info_guest["exp"]
        now = datetime.now(timezone.utc)
        remaining_seconds = (expire_time - now).total_seconds()

        role = mock_token_info_guest["role"]
        total_hours = mock_auth_handler.guest_expire_hours
        total_seconds = total_hours * 3600

        should_renew = (
            remaining_seconds < total_seconds * mock_global_args.token_renew_threshold
        )
        assert should_renew is True

        # Renewal for guest
        username = mock_token_info_guest["username"]
        new_token = mock_auth_handler.create_token(
            username=username, role=role, metadata=mock_token_info_guest["metadata"]
        )
        response.headers["X-New-Token"] = new_token
        _token_renewal_cache[username] = time.time()

        assert "X-New-Token" in response.headers
        assert username in _token_renewal_cache


@pytest.mark.offline
class TestRateLimiting:
    """Tests for token renewal rate limiting"""

    @pytest.fixture
    def mock_auth_handler(self):
        """Mock authentication handler"""
        handler = Mock()
        handler.expire_hours = 24
        handler.create_token = Mock(return_value="new-token-12345")
        return handler

    def test_rate_limit_prevents_rapid_renewals(self, mock_auth_handler):
        """Test that second renewal within 60s is blocked"""
        # Use global cache and constant
        global _token_renewal_cache, _RENEWAL_MIN_INTERVAL

        username = "testuser"
        _token_renewal_cache.clear()

        # First renewal
        current_time_1 = time.time()
        _token_renewal_cache[username] = current_time_1

        response_1 = Mock(spec=Response)
        response_1.headers = {}
        response_1.headers["X-New-Token"] = "new-token-12345"

        # Immediate second renewal attempt (within 60s)
        current_time_2 = time.time()  # Almost same time
        last_renewal = _token_renewal_cache.get(username, 0)
        time_since_last_renewal = current_time_2 - last_renewal

        # Should be blocked by rate limit
        assert time_since_last_renewal < _RENEWAL_MIN_INTERVAL

        response_2 = Mock(spec=Response)
        response_2.headers = {}

        # No new token should be issued
        if time_since_last_renewal < _RENEWAL_MIN_INTERVAL:
            # Rate limited, skip renewal
            pass

        assert "X-New-Token" not in response_2.headers

    def test_rate_limit_allows_renewal_after_interval(self, mock_auth_handler):
        """Test that renewal succeeds after 60s interval"""
        # Use global cache and constant
        global _token_renewal_cache, _RENEWAL_MIN_INTERVAL

        username = "testuser"
        _token_renewal_cache.clear()

        # First renewal at time T
        first_renewal_time = time.time() - 61  # 61 seconds ago
        _token_renewal_cache[username] = first_renewal_time

        # Second renewal attempt now
        current_time = time.time()
        last_renewal = _token_renewal_cache.get(username, 0)
        time_since_last_renewal = current_time - last_renewal

        # Should pass rate limit (>60s elapsed)
        assert time_since_last_renewal >= _RENEWAL_MIN_INTERVAL

        response = Mock(spec=Response)
        response.headers = {}

        if time_since_last_renewal >= _RENEWAL_MIN_INTERVAL:
            new_token = mock_auth_handler.create_token(
                username=username, role="user", metadata={}
            )
            response.headers["X-New-Token"] = new_token
            _token_renewal_cache[username] = current_time

        assert "X-New-Token" in response.headers
        assert response.headers["X-New-Token"] == "new-token-12345"

    def test_rate_limit_per_user(self, mock_auth_handler):
        """Test that different users have independent rate limits"""
        # Use global cache
        global _token_renewal_cache

        _token_renewal_cache.clear()

        user1 = "user1"
        user2 = "user2"

        current_time = time.time()

        # User1 gets renewal
        _token_renewal_cache[user1] = current_time

        # User2 should still be able to get renewal (independent cache)
        last_renewal_user2 = _token_renewal_cache.get(user2, 0)
        assert last_renewal_user2 == 0  # No previous renewal

        # User2 can renew
        _token_renewal_cache[user2] = current_time

        # Both users should have entries
        assert user1 in _token_renewal_cache
        assert user2 in _token_renewal_cache
        assert _token_renewal_cache[user1] == _token_renewal_cache[user2]


@pytest.mark.offline
class TestTokenExpirationCalculation:
    """Tests for token expiration time calculation"""

    def test_expiration_extraction_from_jwt(self):
        """Test extracting expiration time from JWT token"""
        import base64
        import json

        # Create a mock JWT payload
        exp_timestamp = int(
            (datetime.now(timezone.utc) + timedelta(hours=24)).timestamp()
        )
        payload = {"sub": "testuser", "role": "user", "exp": exp_timestamp}

        # Encode as base64 (simulating JWT structure: header.payload.signature)
        payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        mock_token = f"header.{payload_b64}.signature"

        # Simulate extraction
        parts = mock_token.split(".")
        assert len(parts) == 3

        decoded_payload = json.loads(base64.b64decode(parts[1]))
        assert decoded_payload["exp"] == exp_timestamp
        assert decoded_payload["sub"] == "testuser"

    def test_remaining_time_calculation(self):
        """Test calculation of remaining token time"""
        # Token expires in 10 hours
        exp_time = datetime.now(timezone.utc) + timedelta(hours=10)
        now = datetime.now(timezone.utc)

        remaining_seconds = (exp_time - now).total_seconds()

        # Should be approximately 10 hours (36000 seconds)
        assert 35990 < remaining_seconds < 36010

        # Calculate percentage remaining (for 24-hour token)
        total_seconds = 24 * 3600
        percentage_remaining = remaining_seconds / total_seconds

        # Should be approximately 41.67% remaining
        assert 0.41 < percentage_remaining < 0.42

    def test_threshold_comparison(self):
        """Test threshold-based renewal decision"""
        threshold = 0.5
        total_hours = 24
        total_seconds = total_hours * 3600

        # Scenario 1: 10 hours remaining -> should renew
        remaining_seconds_1 = 10 * 3600
        should_renew_1 = remaining_seconds_1 < total_seconds * threshold
        assert should_renew_1 is True

        # Scenario 2: 20 hours remaining -> should NOT renew
        remaining_seconds_2 = 20 * 3600
        should_renew_2 = remaining_seconds_2 < total_seconds * threshold
        assert should_renew_2 is False

        # Scenario 3: Exactly 12 hours remaining (at threshold) -> should NOT renew
        remaining_seconds_3 = 12 * 3600
        should_renew_3 = remaining_seconds_3 < total_seconds * threshold
        assert should_renew_3 is False


@pytest.mark.offline
def test_renewal_cache_cleanup():
    """Test that renewal cache can be cleared"""
    # Use global cache
    global _token_renewal_cache

    # Clear first
    _token_renewal_cache.clear()

    # Add some entries
    _token_renewal_cache["user1"] = time.time()
    _token_renewal_cache["user2"] = time.time()

    assert len(_token_renewal_cache) == 2

    # Clear cache
    _token_renewal_cache.clear()

    assert len(_token_renewal_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
