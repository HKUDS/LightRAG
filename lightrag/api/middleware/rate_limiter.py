"""
Advanced rate limiting middleware for LightRAG.

Provides multi-tier rate limiting with Redis backend, IP blocking,
and comprehensive monitoring capabilities.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger("lightrag.auth.rate_limiter")


class RateLimitType(Enum):
    """Rate limit types for different endpoint categories."""

    AUTHENTICATION = "authentication"
    GENERAL_API = "general_api"
    DOCUMENT_UPLOAD = "document_upload"
    QUERY_OPERATIONS = "query_operations"
    GRAPH_OPERATIONS = "graph_operations"
    ADMIN_OPERATIONS = "admin_operations"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    # Rate limits (requests/time_window)
    authentication_limit: str = "5/minute"
    general_api_limit: str = "100/minute"
    document_upload_limit: str = "10/minute"
    query_operations_limit: str = "50/minute"
    graph_operations_limit: str = "30/minute"
    admin_operations_limit: str = "200/minute"

    # IP blocking settings
    enable_ip_blocking: bool = True
    block_threshold: int = 50  # Failed requests to trigger block
    block_window_minutes: int = 60  # Time window for counting violations
    block_duration_minutes: int = 120  # Block duration

    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_key_prefix: str = "lightrag:rate_limit:"

    # General settings
    enabled: bool = True
    warning_mode: bool = False  # Log but don't block
    per_user_limits: bool = True  # Different limits per user

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Create configuration from environment variables."""
        import os

        def get_env_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, "").lower()
            return value in ("true", "1", "yes", "on") if value else default

        return cls(
            authentication_limit=os.getenv("RATE_LIMIT_AUTH", "5/minute"),
            general_api_limit=os.getenv("RATE_LIMIT_GENERAL", "100/minute"),
            document_upload_limit=os.getenv("RATE_LIMIT_UPLOAD", "10/minute"),
            query_operations_limit=os.getenv("RATE_LIMIT_QUERY", "50/minute"),
            graph_operations_limit=os.getenv("RATE_LIMIT_GRAPH", "30/minute"),
            admin_operations_limit=os.getenv("RATE_LIMIT_ADMIN", "200/minute"),
            enable_ip_blocking=get_env_bool("RATE_LIMIT_IP_BLOCKING", True),
            block_threshold=int(os.getenv("RATE_LIMIT_BLOCK_THRESHOLD", "50")),
            block_window_minutes=int(os.getenv("RATE_LIMIT_BLOCK_WINDOW", "60")),
            block_duration_minutes=int(os.getenv("RATE_LIMIT_BLOCK_DURATION", "120")),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            redis_key_prefix=os.getenv("RATE_LIMIT_PREFIX", "lightrag:rate_limit:"),
            enabled=get_env_bool("RATE_LIMITING_ENABLED", True),
            warning_mode=get_env_bool("RATE_LIMITING_WARNING_MODE", False),
            per_user_limits=get_env_bool("RATE_LIMIT_PER_USER", True),
        )


class RedisRateLimitStore:
    """Redis-based rate limiting storage."""

    def __init__(self, redis_url: str, key_prefix: str = "lightrag:rate_limit:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None
        self._connection_pool = None

    async def initialize(self):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis not available. Install with: pip install redis")

        try:
            self._connection_pool = redis.ConnectionPool.from_url(self.redis_url)
            self._redis = redis.Redis(connection_pool=self._connection_pool)

            # Test connection
            await self._redis.ping()
            logger.info("Redis rate limiting store initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()

    async def check_rate_limit(
        self, key: str, limit: int, window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit.

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        if not self._redis:
            return True, {}

        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds

            # Use sliding window counter
            pipe = self._redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current requests
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(current_time): current_time})

            # Set expiration
            pipe.expire(key, window_seconds + 1)

            results = await pipe.execute()
            current_count = results[1]

            is_allowed = current_count < limit

            rate_limit_info = {
                "limit": limit,
                "current": current_count + 1,  # Include current request
                "remaining": max(0, limit - current_count - 1),
                "reset_time": current_time + window_seconds,
                "retry_after": 1 if not is_allowed else 0,
            }

            return is_allowed, rate_limit_info

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Allow request if Redis fails
            return True, {}

    async def record_violation(
        self, ip_address: str, endpoint: str, user_agent: str = ""
    ) -> None:
        """Record rate limit violation for IP blocking."""
        if not self._redis:
            return

        try:
            violation_key = f"{self.key_prefix}violations:{ip_address}"
            current_time = time.time()

            violation_data = {
                "timestamp": current_time,
                "endpoint": endpoint,
                "user_agent": user_agent[:200],  # Truncate user agent
            }

            # Add violation to sorted set
            await self._redis.zadd(
                violation_key, {json.dumps(violation_data): current_time}
            )

            # Set expiration
            await self._redis.expire(violation_key, 3600)  # 1 hour

        except Exception as e:
            logger.error(f"Error recording violation: {e}")

    async def check_ip_block(
        self, ip_address: str, threshold: int, window_minutes: int
    ) -> Tuple[bool, Optional[datetime]]:
        """
        Check if IP should be blocked due to violations.

        Returns:
            Tuple of (is_blocked, unblock_time)
        """
        if not self._redis:
            return False, None

        try:
            # Check if IP is already blocked
            block_key = f"{self.key_prefix}blocked:{ip_address}"
            blocked_until = await self._redis.get(block_key)

            if blocked_until:
                unblock_time = datetime.fromtimestamp(float(blocked_until))
                if unblock_time > datetime.utcnow():
                    return True, unblock_time
                else:
                    # Block expired, remove it
                    await self._redis.delete(block_key)

            # Count recent violations
            violation_key = f"{self.key_prefix}violations:{ip_address}"
            window_start = time.time() - (window_minutes * 60)

            violation_count = await self._redis.zcount(
                violation_key, window_start, time.time()
            )

            if violation_count >= threshold:
                # Block IP
                block_until = datetime.utcnow() + timedelta(minutes=120)  # 2 hours
                await self._redis.setex(
                    block_key,
                    7200,  # 2 hours in seconds
                    block_until.timestamp(),
                )

                logger.warning(
                    f"IP {ip_address} blocked due to {violation_count} violations"
                )
                return True, block_until

            return False, None

        except Exception as e:
            logger.error(f"Error checking IP block: {e}")
            return False, None


class InMemoryRateLimitStore:
    """In-memory rate limiting store (fallback when Redis unavailable)."""

    def __init__(self):
        self._store: Dict[str, List[float]] = {}
        self._blocked_ips: Dict[str, float] = {}
        self._violations: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize in-memory store."""
        logger.warning("Using in-memory rate limiting (not recommended for production)")

    async def close(self):
        """Close in-memory store."""
        pass

    async def check_rate_limit(
        self, key: str, limit: int, window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using in-memory storage."""
        async with self._lock:
            current_time = time.time()
            window_start = current_time - window_seconds

            # Clean old entries
            if key in self._store:
                self._store[key] = [t for t in self._store[key] if t > window_start]
            else:
                self._store[key] = []

            # Check limit
            current_count = len(self._store[key])
            is_allowed = current_count < limit

            if is_allowed:
                self._store[key].append(current_time)

            rate_limit_info = {
                "limit": limit,
                "current": current_count + (1 if is_allowed else 0),
                "remaining": max(0, limit - current_count - (1 if is_allowed else 0)),
                "reset_time": int(current_time + window_seconds),
                "retry_after": 1 if not is_allowed else 0,
            }

            return is_allowed, rate_limit_info

    async def record_violation(
        self, ip_address: str, endpoint: str, user_agent: str = ""
    ) -> None:
        """Record violation in memory."""
        async with self._lock:
            if ip_address not in self._violations:
                self._violations[ip_address] = []
            self._violations[ip_address].append(time.time())

    async def check_ip_block(
        self, ip_address: str, threshold: int, window_minutes: int
    ) -> Tuple[bool, Optional[datetime]]:
        """Check IP block using in-memory storage."""
        async with self._lock:
            current_time = time.time()

            # Check existing block
            if ip_address in self._blocked_ips:
                if self._blocked_ips[ip_address] > current_time:
                    return True, datetime.fromtimestamp(self._blocked_ips[ip_address])
                else:
                    del self._blocked_ips[ip_address]

            # Count recent violations
            if ip_address in self._violations:
                window_start = current_time - (window_minutes * 60)
                recent_violations = [
                    t for t in self._violations[ip_address] if t > window_start
                ]
                self._violations[ip_address] = recent_violations

                if len(recent_violations) >= threshold:
                    # Block IP
                    block_until = current_time + (120 * 60)  # 2 hours
                    self._blocked_ips[ip_address] = block_until

                    logger.warning(
                        f"IP {ip_address} blocked due to {len(recent_violations)} violations"
                    )
                    return True, datetime.fromtimestamp(block_until)

            return False, None


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple tiers and IP blocking."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig.from_env()
        self._store = None
        self._limiter = None

        # Rate limit mappings
        self._rate_limits = {
            RateLimitType.AUTHENTICATION: self.config.authentication_limit,
            RateLimitType.GENERAL_API: self.config.general_api_limit,
            RateLimitType.DOCUMENT_UPLOAD: self.config.document_upload_limit,
            RateLimitType.QUERY_OPERATIONS: self.config.query_operations_limit,
            RateLimitType.GRAPH_OPERATIONS: self.config.graph_operations_limit,
            RateLimitType.ADMIN_OPERATIONS: self.config.admin_operations_limit,
        }

    async def initialize(self):
        """Initialize rate limiter."""
        if not self.config.enabled:
            logger.info("Rate limiting disabled")
            return

        # Initialize storage
        if REDIS_AVAILABLE and self.config.redis_url:
            try:
                self._store = RedisRateLimitStore(
                    self.config.redis_url, self.config.redis_key_prefix
                )
                await self._store.initialize()
            except Exception as e:
                logger.warning(
                    f"Redis initialization failed, using in-memory store: {e}"
                )
                self._store = InMemoryRateLimitStore()
                await self._store.initialize()
        else:
            self._store = InMemoryRateLimitStore()
            await self._store.initialize()

        logger.info("Rate limiter initialized")

    async def close(self):
        """Close rate limiter."""
        if self._store:
            await self._store.close()

    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Priority: User ID > API Key > IP Address
        user_id = getattr(request.state, "user_id", None)
        if user_id and self.config.per_user_limits:
            return f"user:{user_id}"

        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Use hash of API key for privacy
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"api_key:{key_hash}"

        ip_address = get_remote_address(request)
        return f"ip:{ip_address}"

    def _parse_rate_limit(self, rate_limit_str: str) -> Tuple[int, int]:
        """
        Parse rate limit string like '100/minute' to (limit, window_seconds).

        Returns:
            Tuple of (limit, window_seconds)
        """
        try:
            limit_str, period = rate_limit_str.split("/")
            limit = int(limit_str)

            period_mapping = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}

            window_seconds = period_mapping.get(period, 60)
            return limit, window_seconds

        except (ValueError, KeyError):
            logger.error(f"Invalid rate limit format: {rate_limit_str}")
            return 100, 60  # Default: 100/minute

    async def check_rate_limit(
        self, request: Request, limit_type: RateLimitType
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        if not self.config.enabled or not self._store:
            return True, {}

        # Check IP blocking first
        ip_address = get_remote_address(request)
        if self.config.enable_ip_blocking:
            is_blocked, unblock_time = await self._store.check_ip_block(
                ip_address,
                self.config.block_threshold,
                self.config.block_window_minutes,
            )

            if is_blocked:
                return False, {
                    "error": "IP_BLOCKED",
                    "message": "IP address is temporarily blocked due to excessive violations",
                    "unblock_time": unblock_time.isoformat() if unblock_time else None,
                    "retry_after": int(
                        (unblock_time - datetime.utcnow()).total_seconds()
                    )
                    if unblock_time
                    else 3600,
                }

        # Get rate limit configuration
        rate_limit_str = self._rate_limits.get(
            limit_type, self.config.general_api_limit
        )
        limit, window_seconds = self._parse_rate_limit(rate_limit_str)

        # Generate rate limit key
        client_id = self._get_client_identifier(request)
        rate_limit_key = (
            f"{self.config.redis_key_prefix}limit:{limit_type.value}:{client_id}"
        )

        # Check rate limit
        is_allowed, rate_limit_info = await self._store.check_rate_limit(
            rate_limit_key, limit, window_seconds
        )

        # Record violation if rate limit exceeded
        if not is_allowed:
            await self._store.record_violation(
                ip_address, request.url.path, request.headers.get("user-agent", "")
            )

        # In warning mode, log but allow request
        if not is_allowed and self.config.warning_mode:
            logger.warning(f"Rate limit exceeded for {client_id} on {request.url.path}")
            is_allowed = True
            rate_limit_info["warning_mode"] = True

        return is_allowed, rate_limit_info

    def get_rate_limit_decorator(self, limit_type: RateLimitType):
        """Get rate limit decorator for specific endpoint type."""

        def decorator(func):
            func.__rate_limit_type__ = limit_type
            return func

        return decorator


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(self, app, rate_limiter: AdvancedRateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter."""
        # Determine rate limit type based on endpoint
        limit_type = self._determine_limit_type(request)

        # Check rate limit
        is_allowed, rate_limit_info = await self.rate_limiter.check_rate_limit(
            request, limit_type
        )

        if not is_allowed:
            return self._create_rate_limit_response(rate_limit_info)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        if rate_limit_info:
            response.headers["X-RateLimit-Limit"] = str(
                rate_limit_info.get("limit", "")
            )
            response.headers["X-RateLimit-Remaining"] = str(
                rate_limit_info.get("remaining", "")
            )
            response.headers["X-RateLimit-Reset"] = str(
                rate_limit_info.get("reset_time", "")
            )

            if rate_limit_info.get("warning_mode"):
                response.headers["X-RateLimit-Warning"] = (
                    "Rate limit exceeded (warning mode)"
                )

        return response

    def _determine_limit_type(self, request: Request) -> RateLimitType:
        """Determine rate limit type based on request."""
        path = request.url.path.lower()

        # Authentication endpoints
        if any(
            auth_path in path for auth_path in ["/login", "/auth", "/token", "/logout"]
        ):
            return RateLimitType.AUTHENTICATION

        # Document operations
        if any(
            doc_path in path for doc_path in ["/documents/upload", "/documents/batch"]
        ):
            return RateLimitType.DOCUMENT_UPLOAD

        # Query operations
        if any(query_path in path for query_path in ["/query", "/search"]):
            return RateLimitType.QUERY_OPERATIONS

        # Graph operations
        if any(
            graph_path in path
            for graph_path in ["/graph", "/entities", "/relationships"]
        ):
            return RateLimitType.GRAPH_OPERATIONS

        # Admin operations
        if any(admin_path in path for admin_path in ["/admin", "/system", "/config"]):
            return RateLimitType.ADMIN_OPERATIONS

        # Default to general API
        return RateLimitType.GENERAL_API

    def _create_rate_limit_response(
        self, rate_limit_info: Dict[str, Any]
    ) -> JSONResponse:
        """Create rate limit exceeded response."""
        retry_after = rate_limit_info.get("retry_after", 60)

        error_data = {
            "error": rate_limit_info.get("error", "RATE_LIMIT_EXCEEDED"),
            "message": rate_limit_info.get("message", "Rate limit exceeded"),
            "retry_after": retry_after,
        }

        # Add additional info if available
        if "unblock_time" in rate_limit_info:
            error_data["unblock_time"] = rate_limit_info["unblock_time"]

        headers = {"Retry-After": str(retry_after)}

        if "limit" in rate_limit_info:
            headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
            headers["X-RateLimit-Remaining"] = "0"
            headers["X-RateLimit-Reset"] = str(rate_limit_info.get("reset_time", ""))

        return JSONResponse(status_code=429, content=error_data, headers=headers)


# Global rate limiter instance
rate_limiter = AdvancedRateLimiter()
