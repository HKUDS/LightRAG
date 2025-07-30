"""
Comprehensive test suite for LightRAG Authentication Phase 1.

Tests enhanced password security, rate limiting, security headers,
and audit logging functionality.
"""

import pytest
import asyncio
import os
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightrag.api.auth.password_manager import (
    PasswordManager, PasswordPolicy, PasswordStrength
)
from lightrag.api.middleware.rate_limiter import (
    AdvancedRateLimiter, RateLimitConfig, RateLimitType
)
from lightrag.api.middleware.security_headers import (
    SecurityHeadersMiddleware, SecurityHeadersConfig, SecurityAnalyzer
)
from lightrag.api.logging.audit_logger import (
    AuditLogger, AuditEvent, AuditEventType, AuditSeverity, AuditLogConfig
)


class TestPasswordManager:
    """Test enhanced password security."""
    
    def setup_method(self):
        """Setup test environment."""
        self.password_manager = PasswordManager()
        self.test_passwords = [
            "SimplePass123!",
            "VerySecurePassword2024#",
            "weak",
            "NoNumbers!",
            "nonumbersorspecial",
            "NOLOWERCASE123!",
            "A1!" + "x" * 100  # Too long
        ]
    
    def test_password_policy_creation(self):
        """Test password policy creation and configuration."""
        # Test default policy
        policy = PasswordPolicy()
        assert policy.min_length == 8
        assert policy.require_uppercase is True
        assert policy.require_numbers is True
        
        # Test custom policy
        custom_policy = PasswordPolicy(
            min_length=12,
            require_special_chars=False,
            history_count=10
        )
        assert custom_policy.min_length == 12
        assert custom_policy.require_special_chars is False
        assert custom_policy.history_count == 10
    
    def test_password_policy_from_env(self):
        """Test password policy creation from environment variables."""
        with patch.dict(os.environ, {
            'PASSWORD_MIN_LENGTH': '10',
            'PASSWORD_REQUIRE_UPPERCASE': 'false',
            'PASSWORD_HISTORY_COUNT': '3'
        }):
            policy = PasswordPolicy.from_env()
            assert policy.min_length == 10
            assert policy.require_uppercase is False
            assert policy.history_count == 3
    
    def test_password_validation(self):
        """Test password validation against policy."""
        test_cases = [
            ("SimplePass123!", True, PasswordStrength.STRONG),
            ("weak", False, PasswordStrength.WEAK),
            ("NoNumbers!", False, PasswordStrength.FAIR),
            ("nonumbersorspecial", False, PasswordStrength.WEAK),
            ("NOLOWERCASE123!", False, PasswordStrength.GOOD),
            ("VerySecurePassword2024#", True, PasswordStrength.VERY_STRONG)
        ]
        
        for password, expected_valid, expected_strength in test_cases:
            is_valid, errors, strength = self.password_manager.validate_password(password)
            assert is_valid == expected_valid, f"Password '{password}' validation failed"
            assert strength == expected_strength, f"Password '{password}' strength mismatch"
            
            if not expected_valid:
                assert len(errors) > 0, f"Expected errors for invalid password '{password}'"
    
    def test_password_hashing_and_verification(self):
        """Test password hashing and verification."""
        password = "TestPassword123!"
        
        # Test hashing
        hashed = self.password_manager.hash_password(password)
        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long
        
        # Test verification
        assert self.password_manager.verify_password(password, hashed) is True
        assert self.password_manager.verify_password("wrong", hashed) is False
        assert self.password_manager.verify_password("", hashed) is False
        assert self.password_manager.verify_password(password, "") is False
    
    def test_secure_password_generation(self):
        """Test secure password generation."""
        # Test default length
        password = self.password_manager.generate_secure_password()
        assert len(password) == 16
        
        # Test custom length
        password = self.password_manager.generate_secure_password(length=20)
        assert len(password) == 20
        
        # Test minimum length enforcement
        password = self.password_manager.generate_secure_password(length=4)
        assert len(password) == 8  # Should be enforced to minimum
        
        # Test password meets policy requirements
        generated_password = self.password_manager.generate_secure_password()
        is_valid, errors, strength = self.password_manager.validate_password(generated_password)
        assert is_valid is True, f"Generated password failed validation: {errors}"
    
    def test_token_generation(self):
        """Test secure token generation."""
        # Test URL-safe token
        token = self.password_manager.generate_secure_token(32)
        assert len(token) > 40  # Base64 encoding makes it longer
        assert all(c.isalnum() or c in '-_' for c in token)  # URL-safe characters
        
        # Test numeric code
        code = self.password_manager.generate_numeric_code(6)
        assert len(code) == 6
        assert code.isdigit()
        
        # Test token hashing
        token_hash = self.password_manager.hash_token(token)
        assert len(token_hash) == 64  # SHA-256 hex string
        assert token_hash != token
    
    def test_common_password_detection(self):
        """Test common password detection."""
        common_passwords = ["password", "123456", "qwerty", "admin"]
        for password in common_passwords:
            is_valid, errors, _ = self.password_manager.validate_password(password)
            assert is_valid is False
            assert any("common" in error.lower() for error in errors)
    
    @pytest.mark.asyncio
    async def test_password_history_checking(self):
        """Test password history functionality."""
        # Mock database connection
        mock_db = AsyncMock()
        mock_db.fetch.return_value = [
            {"password_hash": self.password_manager.hash_password("oldpass1")},
            {"password_hash": self.password_manager.hash_password("oldpass2")}
        ]
        
        # Test password reuse detection
        allowed = await self.password_manager.check_password_history(
            "user123", "oldpass1", mock_db
        )
        assert allowed is False
        
        # Test new password acceptance
        allowed = await self.password_manager.check_password_history(
            "user123", "newpass123", mock_db
        )
        assert allowed is True


class TestRateLimiter:
    """Test advanced rate limiting functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = RateLimitConfig(
            enabled=True,
            warning_mode=False,
            authentication_limit="3/minute",
            general_api_limit="10/minute"
        )
        self.rate_limiter = AdvancedRateLimiter(self.config)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        await self.rate_limiter.initialize()
        assert self.rate_limiter._store is not None
    
    @pytest.mark.asyncio
    async def test_in_memory_rate_limiting(self):
        """Test in-memory rate limiting store."""
        await self.rate_limiter.initialize()
        
        # Mock request
        mock_request = Mock()
        mock_request.state.user_id = "test_user"
        mock_request.headers = {}
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/login"
        
        # Test multiple requests within limit
        for i in range(2):
            is_allowed, info = await self.rate_limiter.check_rate_limit(
                mock_request, RateLimitType.AUTHENTICATION
            )
            assert is_allowed is True
            assert info.get("remaining", 0) >= 0
        
        # Test request that exceeds limit
        is_allowed, info = await self.rate_limiter.check_rate_limit(
            mock_request, RateLimitType.AUTHENTICATION
        )
        assert is_allowed is False
        assert "limit" in info
        assert info["current"] > info["limit"]
    
    def test_rate_limit_parsing(self):
        """Test rate limit string parsing."""
        test_cases = [
            ("100/minute", 100, 60),
            ("50/hour", 50, 3600),
            ("10/second", 10, 1),
            ("invalid", 100, 60)  # Default fallback
        ]
        
        for rate_str, expected_limit, expected_window in test_cases:
            limit, window = self.rate_limiter._parse_rate_limit(rate_str)
            assert limit == expected_limit
            assert window == expected_window
    
    def test_client_identifier_generation(self):
        """Test client identifier generation logic."""
        # Test user ID priority
        mock_request = Mock()
        mock_request.state.user_id = "user123"
        mock_request.headers = {"X-API-Key": "api_key_123"}
        mock_request.client.host = "192.168.1.1"
        
        identifier = self.rate_limiter._get_client_identifier(mock_request)
        assert identifier.startswith("user:")
        
        # Test API key fallback
        mock_request.state.user_id = None
        identifier = self.rate_limiter._get_client_identifier(mock_request)
        assert identifier.startswith("api_key:")
        
        # Test IP fallback
        mock_request.headers = {}
        identifier = self.rate_limiter._get_client_identifier(mock_request)
        assert identifier.startswith("ip:")
    
    @pytest.mark.asyncio
    async def test_warning_mode(self):
        """Test rate limiter warning mode."""
        warning_config = RateLimitConfig(
            enabled=True,
            warning_mode=True,
            authentication_limit="1/minute"
        )
        warning_limiter = AdvancedRateLimiter(warning_config)
        await warning_limiter.initialize()
        
        mock_request = Mock()
        mock_request.state.user_id = "test_user"
        mock_request.headers = {}
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/login"
        
        # First request should be allowed
        is_allowed, info = await warning_limiter.check_rate_limit(
            mock_request, RateLimitType.AUTHENTICATION
        )
        assert is_allowed is True
        
        # Second request should exceed limit but be allowed in warning mode
        is_allowed, info = await warning_limiter.check_rate_limit(
            mock_request, RateLimitType.AUTHENTICATION
        )
        assert is_allowed is True  # Allowed due to warning mode
        assert info.get("warning_mode") is True


class TestSecurityHeaders:
    """Test security headers middleware."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = SecurityHeadersConfig()
        self.middleware = SecurityHeadersMiddleware(None, self.config)
    
    def test_security_headers_config(self):
        """Test security headers configuration."""
        assert self.config.enable_csp is True
        assert self.config.enable_hsts is True
        assert self.config.hsts_max_age == 31536000
        
        # Test CSP header building
        csp_header = self.config.build_csp_header()
        assert "default-src 'self'" in csp_header
        assert "script-src" in csp_header
        assert "object-src 'none'" in csp_header
        
        # Test HSTS header building
        hsts_header = self.config.build_hsts_header()
        assert "max-age=31536000" in hsts_header
        assert "includeSubDomains" in hsts_header
        assert "preload" in hsts_header
    
    def test_security_headers_from_env(self):
        """Test security headers configuration from environment."""
        with patch.dict(os.environ, {
            'SECURITY_ENABLE_CSP': 'false',
            'HSTS_MAX_AGE': '7776000',
            'CSP_DEFAULT_SRC': "'self' https:",
            'X_FRAME_OPTIONS': 'SAMEORIGIN'
        }):
            config = SecurityHeadersConfig.from_env()
            assert config.enable_csp is False
            assert config.hsts_max_age == 7776000
            assert config.csp_default_src == "'self' https:"
            assert config.x_frame_options == "SAMEORIGIN"
    
    @pytest.mark.asyncio
    async def test_security_headers_middleware(self):
        """Test security headers middleware functionality."""
        # Mock request and response
        mock_request = Mock()
        mock_request.url.scheme = "https"
        mock_request.url.path = "/api/test"
        mock_request.headers = {}
        
        mock_response = Mock()
        mock_response.headers = {}
        
        async def mock_call_next(request):
            return mock_response
        
        # Test middleware processing
        response = await self.middleware.dispatch(mock_request, mock_call_next)
        
        # Verify security headers were added
        assert "Content-Security-Policy" in response.headers
        assert "Strict-Transport-Security" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "Referrer-Policy" in response.headers
    
    def test_security_analyzer(self):
        """Test security configuration analyzer."""
        analyzer = SecurityAnalyzer(self.config)
        analysis = analyzer.analyze_security_posture()
        
        assert "score" in analysis
        assert "security_level" in analysis
        assert "recommendations" in analysis
        assert "good_practices" in analysis
        assert analysis["score"] > 0
        
        # Test security report generation
        report = analyzer.generate_security_report()
        assert "Security Score" in report
        assert "Good Practices" in report
        assert "Recommendations" in report


class TestAuditLogger:
    """Test comprehensive audit logging."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_audit.log")
        
        self.config = AuditLogConfig(
            log_file=self.log_file,
            async_logging=False,  # Disable for testing
            structured_logging=True,
            enable_analytics=True
        )
        self.audit_logger = AuditLogger(self.config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_audit_event_creation(self):
        """Test audit event creation and serialization."""
        event = AuditEvent(
            event_id="test-123",
            event_type=AuditEventType.LOGIN_SUCCESS,
            timestamp=datetime.now(),
            severity=AuditSeverity.LOW,
            user_id="user123",
            ip_address="192.168.1.1",
            success=True,
            message="User logged in successfully"
        )
        
        assert event.event_id == "test-123"
        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.user_id == "user123"
        assert event.success is True
        
        # Test serialization
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "auth.login.success"
        assert event_dict["severity"] == "low"
        
        # Test JSON serialization
        event_json = event.to_json()
        parsed = json.loads(event_json)
        assert parsed["user_id"] == "user123"
    
    @pytest.mark.asyncio
    async def test_audit_logging_functionality(self):
        """Test audit logging functionality."""
        await self.audit_logger.start()
        
        # Test authentication event logging
        await self.audit_logger.log_auth_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="test_user",
            success=True,
            ip_address="127.0.0.1",
            details={"auth_method": "password"}
        )
        
        # Test API access logging
        await self.audit_logger.log_api_access(
            endpoint="/api/test",
            method="GET",
            status_code=200,
            user_id="test_user",
            response_time=0.5
        )
        
        # Test security event logging
        await self.audit_logger.log_security_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            severity=AuditSeverity.HIGH,
            message="Multiple failed login attempts",
            details={"attempts": 5, "window": "5 minutes"}
        )
        
        await self.audit_logger.stop()
        
        # Verify log file was created and contains events
        assert os.path.exists(self.log_file)
        
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            assert "auth.login.success" in log_content
            assert "api.access" in log_content
            assert "security.suspicious" in log_content
    
    def test_audit_analytics(self):
        """Test audit analytics functionality."""
        analytics = self.audit_logger.get_analytics()
        
        # Create test events
        events = [
            AuditEvent(
                event_id=f"test-{i}",
                event_type=AuditEventType.LOGIN_SUCCESS if i % 2 == 0 else AuditEventType.LOGIN_FAILURE,
                timestamp=datetime.now(),
                severity=AuditSeverity.LOW,
                user_id=f"user{i}",
                success=i % 2 == 0
            )
            for i in range(10)
        ]
        
        # Process events
        for event in events:
            analytics.process_event(event)
        
        # Test summary generation
        summary = analytics.get_event_summary(hours=1)
        assert summary["total_events"] == 10
        assert "auth.login.success" in summary["event_types"]
        assert "auth.login.failure" in summary["event_types"]
        assert summary["success_rate"] == 0.5  # 50% success rate
        
        # Test failure counting
        failure_count = analytics.get_recent_failures("user1", minutes=60)
        assert failure_count >= 0
    
    def test_audit_config_from_env(self):
        """Test audit configuration from environment."""
        with patch.dict(os.environ, {
            'AUDIT_LOG_FILE': '/tmp/test_audit.log',
            'AUDIT_ASYNC_LOGGING': 'false',
            'AUDIT_ENABLE_ANALYTICS': 'true',
            'AUDIT_RETENTION_DAYS': '30'
        }):
            config = AuditLogConfig.from_env()
            assert config.log_file == '/tmp/test_audit.log'
            assert config.async_logging is False
            assert config.enable_analytics is True
            assert config.retention_days == 30


class TestIntegration:
    """Integration tests for Phase 1 components."""
    
    @pytest.mark.asyncio
    async def test_password_and_audit_integration(self):
        """Test integration between password manager and audit logging."""
        # Setup components
        password_manager = PasswordManager()
        audit_logger = AuditLogger()
        await audit_logger.start()
        
        # Test password validation with audit logging
        password = "TestPassword123!"
        is_valid, errors, strength = password_manager.validate_password(password)
        
        if is_valid:
            # Log successful password validation
            await audit_logger.log_auth_event(
                event_type=AuditEventType.PASSWORD_CHANGE,
                user_id="test_user",
                success=True,
                details={"strength": strength.value}
            )
        
        await audit_logger.stop()
        
        assert is_valid is True
        assert strength == PasswordStrength.STRONG
    
    @pytest.mark.asyncio
    async def test_rate_limiter_and_audit_integration(self):
        """Test integration between rate limiter and audit logging."""
        # Setup components
        rate_limiter = AdvancedRateLimiter()
        audit_logger = AuditLogger()
        await rate_limiter.initialize()
        await audit_logger.start()
        
        # Mock request
        mock_request = Mock()
        mock_request.state.user_id = "test_user"
        mock_request.headers = {}
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/api/test"
        
        # Test rate limit check
        is_allowed, info = await rate_limiter.check_rate_limit(
            mock_request, RateLimitType.GENERAL_API
        )
        
        # Log the rate limit check
        if not is_allowed:
            await audit_logger.log_security_event(
                event_type=AuditEventType.API_RATE_LIMITED,
                severity=AuditSeverity.MEDIUM,
                message="Rate limit exceeded",
                ip_address=mock_request.client.host,
                details=info
            )
        
        await audit_logger.stop()
        await rate_limiter.close()
        
        # First request should be allowed
        assert is_allowed is True


@pytest.mark.asyncio
async def test_full_phase1_workflow():
    """Test complete Phase 1 authentication workflow."""
    # This test simulates a complete authentication flow with all Phase 1 components
    
    # Setup all components
    password_manager = PasswordManager()
    rate_limiter = AdvancedRateLimiter()
    audit_logger = AuditLogger()
    
    await rate_limiter.initialize()
    await audit_logger.start()
    
    try:
        # Step 1: User registration with password validation
        new_password = "SecurePassword2024!"
        is_valid, errors, strength = password_manager.validate_password(new_password)
        
        assert is_valid is True, f"Password validation failed: {errors}"
        
        # Step 2: Hash password for storage
        password_hash = password_manager.hash_password(new_password)
        assert password_hash is not None
        
        # Step 3: Log user registration
        await audit_logger.log_auth_event(
            event_type=AuditEventType.USER_CREATED,
            user_id="new_user_123",
            success=True,
            details={"password_strength": strength.value}
        )
        
        # Step 4: Simulate login attempt with rate limiting
        mock_request = Mock()
        mock_request.state.user_id = "new_user_123"
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.100"
        mock_request.url.path = "/login"
        
        is_allowed, rate_info = await rate_limiter.check_rate_limit(
            mock_request, RateLimitType.AUTHENTICATION
        )
        
        assert is_allowed is True
        
        # Step 5: Verify password
        password_valid = password_manager.verify_password(new_password, password_hash)
        assert password_valid is True
        
        # Step 6: Log successful login
        await audit_logger.log_auth_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="new_user_123",
            success=True,
            ip_address="192.168.1.100"
        )
        
        # Step 7: Test security headers would be applied (in actual middleware)
        security_config = SecurityHeadersConfig()
        assert security_config.enable_csp is True
        assert security_config.enable_hsts is True
        
        print("✅ Complete Phase 1 workflow test passed!")
        
    finally:
        await audit_logger.stop()
        await rate_limiter.close()


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])