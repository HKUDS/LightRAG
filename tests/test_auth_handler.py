"""Tests for authentication handler module"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from lightrag.api.auth.handler import (
    AuthHandler,
    TokenPayload,
)


class TestAuthHandler:
    """Test AuthHandler class"""

    @pytest.fixture
    def auth_handler(self):
        """Create an AuthHandler instance for testing"""
        return AuthHandler()

    def test_auth_handler_initialization(self, auth_handler):
        """Test AuthHandler initialization"""
        assert auth_handler is not None
        assert hasattr(auth_handler, "secret")
        assert hasattr(auth_handler, "algorithm")
        assert hasattr(auth_handler, "expire_hours")

    def test_create_token_basic(self, auth_handler):
        """Test basic token creation"""
        token = auth_handler.create_token("testuser")
        assert isinstance(token, str)
        assert len(token) > 0

        # Token should have three parts (header.payload.signature)
        parts = token.split(".")
        assert len(parts) == 3

    def test_create_token_with_role(self, auth_handler):
        """Test token creation with role"""
        token = auth_handler.create_token("testuser", role="admin")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_token_with_metadata(self, auth_handler):
        """Test token creation with metadata"""
        metadata = {"user_id": "123", "email": "test@example.com"}
        token = auth_handler.create_token("testuser", metadata=metadata)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_token_valid(self, auth_handler):
        """Test validating a valid token"""
        # Create a token
        token = auth_handler.create_token("testuser")

        # Validate the token
        payload = auth_handler.validate_token(token)
        assert payload["username"] == "testuser"
        assert payload["role"] == "user"
        assert "metadata" in payload

    def test_validate_token_invalid(self, auth_handler):
        """Test validating an invalid token"""
        invalid_token = "invalid.token.here"

        with pytest.raises(HTTPException) as exc_info:
            auth_handler.validate_token(invalid_token)
        assert exc_info.value.status_code == 401

    def test_auth_accounts_initialization(self):
        """Test initialization with auth accounts"""
        # Mock global_args with accounts
        mock_global_args = MagicMock()
        mock_global_args.token_secret = "test_secret"
        mock_global_args.jwt_algorithm = "HS256"
        mock_global_args.token_expire_hours = 24
        mock_global_args.guest_token_expire_hours = 1
        mock_global_args.auth_accounts = "user1:pass1,user2:pass2"

        with patch("lightrag.api.auth.handler.global_args", mock_global_args):
            handler = AuthHandler()
            assert "user1" in handler.accounts
            assert "user2" in handler.accounts
            assert handler.accounts["user1"] == "pass1"
            assert handler.accounts["user2"] == "pass2"


class TestTokenPayload:
    """Test TokenPayload model"""

    def test_token_payload_basic(self):
        """Test basic TokenPayload creation"""
        payload = TokenPayload(
            sub="testuser", exp=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        assert payload.sub == "testuser"
        assert payload.role == "user"  # default
        assert payload.metadata == {}  # default

    def test_token_payload_with_custom_fields(self):
        """Test TokenPayload with custom fields"""
        custom_metadata = {"user_id": "123", "email": "test@example.com"}
        payload = TokenPayload(
            sub="admin_user",
            exp=datetime.now(timezone.utc) + timedelta(hours=24),
            role="admin",
            metadata=custom_metadata,
        )
        assert payload.sub == "admin_user"
        assert payload.role == "admin"
        assert payload.metadata == custom_metadata


class TestAuthIntegration:
    """Integration tests for authentication system"""

    def test_complete_token_workflow(self):
        """Test complete token creation and validation workflow"""
        # Create auth handler
        auth_handler = AuthHandler()

        # Create a token
        username = "testuser"
        token = auth_handler.create_token(username, role="admin")

        # Validate the token
        payload = auth_handler.validate_token(token)

        # Check the payload
        assert payload["username"] == username
        assert payload["role"] == "admin"
        assert "metadata" in payload
