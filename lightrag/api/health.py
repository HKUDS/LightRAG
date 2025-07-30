"""
Health Check Module for LightRAG Production Deployment
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

import redis
import psycopg2
from psycopg2 import OperationalError as PgOperationalError

from lightrag.api.config import config

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status data structure"""

    healthy: bool
    status: str
    timestamp: str
    version: str = "1.0.0"
    environment: str = "production"
    uptime_seconds: float = 0.0
    checks: Dict[str, Any] = None

    def __post_init__(self):
        if self.checks is None:
            self.checks = {}


class HealthChecker:
    """Comprehensive health checker for LightRAG production deployment"""

    def __init__(self):
        self.start_time = time.time()
        self.redis_client: Optional[redis.Redis] = None
        self.postgres_conn: Optional[Any] = None

    async def check_health(self) -> HealthStatus:
        """Perform comprehensive health check"""
        try:
            uptime = time.time() - self.start_time
            timestamp = datetime.now(timezone.utc).isoformat()

            checks = {}
            overall_healthy = True

            # Basic application health
            checks["application"] = await self._check_application()
            if not checks["application"]["healthy"]:
                overall_healthy = False

            # Database connectivity
            if config.get("POSTGRES_HOST"):
                checks["database"] = await self._check_database()
                if not checks["database"]["healthy"]:
                    overall_healthy = False

            # Redis connectivity
            if config.get("REDIS_URI"):
                checks["redis"] = await self._check_redis()
                if not checks["redis"]["healthy"]:
                    overall_healthy = False

            # System resources
            checks["system"] = await self._check_system_resources()
            if not checks["system"]["healthy"]:
                overall_healthy = False

            # LLM connectivity (optional)
            if config.get("LLM_API_KEY"):
                checks["llm"] = await self._check_llm()
                # Don't fail overall health for LLM issues

            status = "healthy" if overall_healthy else "unhealthy"

            return HealthStatus(
                healthy=overall_healthy,
                status=status,
                timestamp=timestamp,
                uptime_seconds=uptime,
                checks=checks,
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthStatus(
                healthy=False,
                status="error",
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=time.time() - self.start_time,
                checks={"error": {"healthy": False, "message": str(e)}},
            )

    async def _check_application(self) -> Dict[str, Any]:
        """Check basic application health"""
        try:
            return {
                "healthy": True,
                "status": "running",
                "pid": psutil.Process().pid,
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.Process().cpu_percent(),
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity"""
        try:
            if not self.postgres_conn:
                self.postgres_conn = psycopg2.connect(
                    host=config.get("POSTGRES_HOST", "localhost"),
                    port=config.get("POSTGRES_PORT", 5432),
                    user=config.get("POSTGRES_USER", "postgres"),
                    password=config.get("POSTGRES_PASSWORD", ""),
                    database=config.get("POSTGRES_DATABASE", "lightrag"),
                    connect_timeout=5,
                )

            # Test connection with a simple query
            with self.postgres_conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()

            return {
                "healthy": True,
                "status": "connected",
                "host": config.get("POSTGRES_HOST"),
                "database": config.get("POSTGRES_DATABASE"),
                "test_query_result": result[0] if result else None,
            }

        except PgOperationalError as e:
            # Reset connection on error
            self.postgres_conn = None
            return {"healthy": False, "status": "connection_failed", "error": str(e)}
        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}

    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            if not self.redis_client:
                redis_uri = config.get("REDIS_URI", "redis://localhost:6379/0")
                self.redis_client = redis.from_url(
                    redis_uri,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    decode_responses=True,
                )

            # Test connection with ping
            response = self.redis_client.ping()
            info = self.redis_client.info()

            return {
                "healthy": True,
                "status": "connected",
                "ping": response,
                "version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
            }

        except redis.ConnectionError as e:
            # Reset client on error
            self.redis_client = None
            return {"healthy": False, "status": "connection_failed", "error": str(e)}
        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}

    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_healthy = memory.percent < 90

            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_healthy = cpu_percent < 90

            # Disk check
            disk = psutil.disk_usage("/")
            disk_healthy = disk.percent < 90

            overall_healthy = memory_healthy and cpu_healthy and disk_healthy

            return {
                "healthy": overall_healthy,
                "memory": {
                    "percent": memory.percent,
                    "available_gb": memory.available / 1024 / 1024 / 1024,
                    "healthy": memory_healthy,
                },
                "cpu": {"percent": cpu_percent, "healthy": cpu_healthy},
                "disk": {
                    "percent": disk.percent,
                    "free_gb": disk.free / 1024 / 1024 / 1024,
                    "healthy": disk_healthy,
                },
            }

        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_llm(self) -> Dict[str, Any]:
        """Check LLM connectivity (optional)"""
        try:
            # This is a basic connectivity check
            # In a real implementation, you might want to make a simple API call
            return {
                "healthy": True,
                "status": "configured",
                "provider": config.get("LLM_BINDING", "unknown"),
                "model": config.get("LLM_MODEL", "unknown"),
            }

        except Exception as e:
            return {"healthy": False, "status": "error", "error": str(e)}

    def close(self):
        """Clean up connections"""
        if self.postgres_conn:
            try:
                self.postgres_conn.close()
            except Exception:
                pass

        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass


# Global health checker instance
health_checker = HealthChecker()


async def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    status = await health_checker.check_health()
    return asdict(status)


def get_readiness_status() -> Dict[str, Any]:
    """Simple readiness check"""
    return {
        "ready": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
    }


def get_liveness_status() -> Dict[str, Any]:
    """Simple liveness check"""
    return {
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "alive",
    }
