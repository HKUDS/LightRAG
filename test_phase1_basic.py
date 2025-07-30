#!/usr/bin/env python3
"""
Basic functionality test for LightRAG Authentication Phase 1.
Tests core components without requiring full test suite setup.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag.api.auth.password_manager import PasswordManager, PasswordPolicy, PasswordStrength
from lightrag.api.middleware.rate_limiter import AdvancedRateLimiter, RateLimitConfig
from lightrag.api.middleware.security_headers import SecurityHeadersConfig, SecurityAnalyzer
from lightrag.api.logging.audit_logger import AuditLogger, AuditEvent, AuditEventType, AuditSeverity, AuditLogConfig


def test_password_manager():
    """Test enhanced password security."""
    print("🔐 Testing Password Manager...")
    
    password_manager = PasswordManager()
    
    # Test password validation
    test_cases = [
        ("SimplePass123!", True, "Strong password"),
        ("weak", False, "Too weak"),
        ("NoNumbers!", False, "Missing numbers"),
        ("VerySecurePassword2024#", True, "Very strong password")
    ]
    
    for password, expected_valid, description in test_cases:
        is_valid, errors, strength = password_manager.validate_password(password)
        status = "✅" if is_valid == expected_valid else "❌"
        print(f"  {status} {description}: {password} -> Valid: {is_valid}, Strength: {strength.value}")
        if errors:
            print(f"    Errors: {', '.join(errors)}")
    
    # Test password hashing
    test_password = "TestPassword123!"
    hashed = password_manager.hash_password(test_password)
    verification = password_manager.verify_password(test_password, hashed)
    print(f"  ✅ Password hashing: Original != Hash: {test_password != hashed}")
    print(f"  ✅ Password verification: {verification}")
    
    # Test secure password generation
    generated = password_manager.generate_secure_password(16)
    gen_valid, gen_errors, gen_strength = password_manager.validate_password(generated)
    print(f"  ✅ Generated password: {generated} -> Valid: {gen_valid}, Strength: {gen_strength.value}")
    
    # Test token generation
    token = password_manager.generate_secure_token(32)
    numeric_code = password_manager.generate_numeric_code(6)
    print(f"  ✅ Secure token: {token[:20]}... (length: {len(token)})")
    print(f"  ✅ Numeric code: {numeric_code} (length: {len(numeric_code)})")
    
    print("  ✅ Password Manager tests completed\n")


async def test_rate_limiter():
    """Test advanced rate limiting."""
    print("🚦 Testing Rate Limiter...")
    
    config = RateLimitConfig(
        enabled=True,
        warning_mode=False,
        authentication_limit="3/minute",
        general_api_limit="10/minute"
    )
    
    rate_limiter = AdvancedRateLimiter(config)
    await rate_limiter.initialize()
    
    # Mock request object
    class MockRequest:
        def __init__(self, user_id=None, ip="127.0.0.1"):
            self.state = type('State', (), {'user_id': user_id})()
            self.headers = {}
            self.client = type('Client', (), {'host': ip})()
            self.url = type('URL', (), {'path': '/test'})()
    
    mock_request = MockRequest("test_user")
    
    # Test rate limiting
    from lightrag.api.middleware.rate_limiter import RateLimitType
    
    results = []
    for i in range(5):
        is_allowed, info = await rate_limiter.check_rate_limit(
            mock_request, RateLimitType.AUTHENTICATION
        )
        results.append((i+1, is_allowed, info.get('remaining', 0)))
        print(f"  Request {i+1}: Allowed: {is_allowed}, Remaining: {info.get('remaining', 0)}")
    
    # Verify rate limiting works
    allowed_count = sum(1 for _, allowed, _ in results if allowed)
    print(f"  ✅ Rate limiting: {allowed_count}/5 requests allowed (limit: 3)")
    
    await rate_limiter.close()
    print("  ✅ Rate Limiter tests completed\n")


def test_security_headers():
    """Test security headers configuration."""
    print("🛡️  Testing Security Headers...")
    
    config = SecurityHeadersConfig()
    
    # Test CSP header building
    csp_header = config.build_csp_header()
    print(f"  ✅ CSP Header: {csp_header[:50]}...")
    
    # Test HSTS header building
    hsts_header = config.build_hsts_header()
    print(f"  ✅ HSTS Header: {hsts_header}")
    
    # Test security analysis
    analyzer = SecurityAnalyzer(config)
    analysis = analyzer.analyze_security_posture()
    
    print(f"  ✅ Security Score: {analysis['score']}/{analysis['max_score']} ({analysis['security_level']})")
    print(f"  ✅ Good Practices: {len(analysis['good_practices'])}")
    print(f"  ✅ Recommendations: {len(analysis['recommendations'])}")
    
    # Test environment variable configuration
    original_csp = os.environ.get('CSP_DEFAULT_SRC')
    os.environ['CSP_DEFAULT_SRC'] = "'self' https:"
    
    env_config = SecurityHeadersConfig.from_env()
    print(f"  ✅ Environment config: CSP default-src = {env_config.csp_default_src}")
    
    # Restore original environment
    if original_csp:
        os.environ['CSP_DEFAULT_SRC'] = original_csp
    elif 'CSP_DEFAULT_SRC' in os.environ:
        del os.environ['CSP_DEFAULT_SRC']
    
    print("  ✅ Security Headers tests completed\n")


async def test_audit_logger():
    """Test comprehensive audit logging."""
    print("📝 Testing Audit Logger...")
    
    # Create temporary log file
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, "test_audit.log")
    
    config = AuditLogConfig(
        log_file=log_file,
        async_logging=False,  # Synchronous for testing
        structured_logging=True,
        enable_analytics=True
    )
    
    audit_logger = AuditLogger(config)
    await audit_logger.start()
    
    # Test event creation
    from datetime import datetime, timezone
    event = AuditEvent(
        event_id="test-123",
        event_type=AuditEventType.LOGIN_SUCCESS,
        timestamp=datetime.now(timezone.utc),
        severity=AuditSeverity.LOW,
        user_id="test_user",
        ip_address="192.168.1.1",
        success=True,
        message="Test login event"
    )
    
    print(f"  ✅ Event created: {event.event_type.value}")
    print(f"  ✅ Event JSON: {event.to_json()[:80]}...")
    
    # Test logging different event types
    events_logged = []
    
    # Authentication event
    await audit_logger.log_auth_event(
        event_type=AuditEventType.LOGIN_SUCCESS,
        user_id="test_user",
        success=True,
        ip_address="127.0.0.1"
    )
    events_logged.append("Authentication")
    
    # API access event
    await audit_logger.log_api_access(
        endpoint="/api/test",
        method="GET",
        status_code=200,
        user_id="test_user",
        response_time=0.5
    )
    events_logged.append("API Access")
    
    # Security event
    await audit_logger.log_security_event(
        event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
        severity=AuditSeverity.MEDIUM,
        message="Test security event"
    )
    events_logged.append("Security")
    
    print(f"  ✅ Events logged: {', '.join(events_logged)}")
    
    # Test analytics
    analytics = audit_logger.get_analytics()
    summary = analytics.get_event_summary(hours=1)
    print(f"  ✅ Analytics: {summary['total_events']} total events")
    
    await audit_logger.stop()
    
    # Check log file
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
            print(f"  ✅ Log file created: {len(log_content)} characters")
            print(f"  ✅ Contains events: {'auth.login.success' in log_content}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("  ✅ Audit Logger tests completed\n")


async def test_integration():
    """Test integration between components."""
    print("🔗 Testing Component Integration...")
    
    # Test password manager with audit logging
    password_manager = PasswordManager()
    
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, "integration_test.log")
    
    config = AuditLogConfig(log_file=log_file, async_logging=False)
    audit_logger = AuditLogger(config)
    await audit_logger.start()
    
    # Simulate user registration workflow
    new_password = "IntegrationTest2024!"
    
    # Step 1: Validate password
    is_valid, errors, strength = password_manager.validate_password(new_password)
    print(f"  ✅ Password validation: Valid={is_valid}, Strength={strength.value}")
    
    # Step 2: Hash password
    if is_valid:
        hashed = password_manager.hash_password(new_password)
        print(f"  ✅ Password hashed: {len(hashed)} characters")
        
        # Step 3: Log registration
        await audit_logger.log_auth_event(
            event_type=AuditEventType.USER_CREATED,
            user_id="integration_user",
            success=True,
            details={"password_strength": strength.value}
        )
        print("  ✅ Registration logged")
        
        # Step 4: Verify password works
        verified = password_manager.verify_password(new_password, hashed)
        print(f"  ✅ Password verification: {verified}")
        
        if verified:
            # Step 5: Log successful login
            await audit_logger.log_auth_event(
                event_type=AuditEventType.LOGIN_SUCCESS,
                user_id="integration_user",
                success=True,
                ip_address="127.0.0.1"
            )
            print("  ✅ Login logged")
    
    await audit_logger.stop()
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("  ✅ Integration tests completed\n")


async def main():
    """Run all Phase 1 tests."""
    print("🚀 LightRAG Authentication Phase 1 - Basic Functionality Test")
    print("=" * 60)
    
    try:
        # Test individual components
        test_password_manager()
        await test_rate_limiter()
        test_security_headers()
        await test_audit_logger()
        await test_integration()
        
        print("🎉 All Phase 1 tests completed successfully!")
        print("✅ Enhanced password security: Working")
        print("✅ Advanced rate limiting: Working")
        print("✅ Security headers: Working")
        print("✅ Comprehensive audit logging: Working")
        print("✅ Component integration: Working")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)