import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import psycopg2
import redis

from lightrag.api.health import (
    HealthChecker,
    HealthStatus,
    get_health_status,
    get_readiness_status,
    get_liveness_status,
)


# Mock global_args for configuration
@pytest.fixture
def mock_global_args():
    with patch("lightrag.api.health.global_args") as mock_args:
        mock_args.POSTGRES_HOST = "test_db_host"
        mock_args.POSTGRES_PORT = 5432
        mock_args.POSTGRES_USER = "test_user"
        mock_args.POSTGRES_PASSWORD = "test_password"
        mock_args.POSTGRES_DATABASE = "test_db"
        mock_args.REDIS_URI = "redis://test_redis:6379/0"
        mock_args.LLM_API_KEY = "test_llm_key"
        yield mock_args


# Fixture for HealthChecker instance
@pytest.fixture
def health_checker_instance():
    checker = HealthChecker()
    yield checker
    checker.close()


@pytest.mark.asyncio
async def test_health_status_dataclass():
    status = HealthStatus(
        healthy=True, status="healthy", timestamp="2025-01-01T00:00:00Z"
    )
    assert status.healthy is True
    assert status.status == "healthy"
    assert status.timestamp == "2025-01-01T00:00:00Z"
    assert status.version == "1.0.0"
    assert status.environment == "production"
    assert status.uptime_seconds == 0.0
    assert status.checks == {}


@pytest.mark.asyncio
async def test_check_application_healthy(health_checker_instance):
    with patch("psutil.Process") as mock_process:
        mock_process.return_value.pid = 1234
        mock_process.return_value.memory_info.return_value.rss = (
            1024 * 1024 * 100
        )  # 100 MB
        mock_process.return_value.cpu_percent.return_value = 5.0
        result = await health_checker_instance._check_application()
        assert result["healthy"] is True
        assert result["pid"] == 1234
        assert result["memory_usage_mb"] == 100.0
        assert result["cpu_percent"] == 5.0


@pytest.mark.asyncio
async def test_check_application_error(health_checker_instance):
    with patch("psutil.Process", side_effect=Exception("Test error")):
        result = await health_checker_instance._check_application()
        assert result["healthy"] is False
        assert "error" in result


@pytest.mark.asyncio
async def test_check_database_healthy(health_checker_instance):
    with (
        patch("psycopg2.connect") as mock_connect,
        patch("lightrag.api.health.get_config") as mock_get_config,
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "POSTGRES_HOST": "test_db_host",
            "POSTGRES_PORT": 5432,
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
            "POSTGRES_DATABASE": "test_db",
        }.get(key, default)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = await health_checker_instance._check_database()
        assert result["healthy"] is True
        assert result["status"] == "connected"
        assert result["host"] == "test_db_host"
        assert result["database"] == "test_db"
        assert result["test_query_result"] == 1
        mock_connect.assert_called_once_with(
            host="test_db_host",
            port=5432,
            user="test_user",
            password="test_password",
            database="test_db",
            connect_timeout=5,
        )


@pytest.mark.asyncio
async def test_check_database_connection_error(health_checker_instance):
    with (
        patch(
            "psycopg2.connect",
            side_effect=psycopg2.OperationalError("Connection failed"),
        ),
        patch("lightrag.api.health.get_config") as mock_get_config,
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "POSTGRES_HOST": "test_db_host",
            "POSTGRES_PORT": 5432,
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
            "POSTGRES_DATABASE": "test_db",
        }.get(key, default)
        result = await health_checker_instance._check_database()
        assert result["healthy"] is False
        assert result["status"] == "connection_failed"
        assert "error" in result
        assert health_checker_instance.postgres_conn is None  # Should reset connection


@pytest.mark.asyncio
async def test_check_database_other_error(health_checker_instance):
    with (
        patch("psycopg2.connect", side_effect=Exception("Other DB error")),
        patch("lightrag.api.health.get_config") as mock_get_config,
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "POSTGRES_HOST": "test_db_host",
            "POSTGRES_PORT": 5432,
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
            "POSTGRES_DATABASE": "test_db",
        }.get(key, default)
        result = await health_checker_instance._check_database()
        assert result["healthy"] is False
        assert result["status"] == "error"
        assert "error" in result


@pytest.mark.asyncio
async def test_check_redis_healthy(health_checker_instance):
    with (
        patch("redis.from_url") as mock_from_url,
        patch("lightrag.api.health.get_config") as mock_get_config,
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "REDIS_URI": "redis://test_redis:6379/0",
        }.get(key, default)

        mock_redis_client = MagicMock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.info.return_value = {
            "redis_version": "6.0.0",
            "connected_clients": 10,
            "used_memory_human": "10M",
        }
        mock_from_url.return_value = mock_redis_client

        result = await health_checker_instance._check_redis()
        assert result["healthy"] is True
        assert result["status"] == "connected"
        assert result["ping"] is True
        assert result["version"] == "6.0.0"
        assert result["connected_clients"] == 10
        assert result["used_memory_human"] == "10M"
        mock_from_url.assert_called_once_with(
            "redis://test_redis:6379/0",
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            decode_responses=True,
        )


@pytest.mark.asyncio
async def test_check_redis_connection_error(health_checker_instance):
    with (
        patch(
            "redis.from_url",
            side_effect=redis.ConnectionError("Redis connection failed"),
        ),
        patch("lightrag.api.health.get_config") as mock_get_config,
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "REDIS_URI": "redis://test_redis:6379/0",
        }.get(key, default)
        result = await health_checker_instance._check_redis()
        assert result["healthy"] is False
        assert result["status"] == "connection_failed"
        assert "error" in result
        assert health_checker_instance.redis_client is None  # Should reset client


@pytest.mark.asyncio
async def test_check_redis_other_error(health_checker_instance):
    with (
        patch("redis.from_url", side_effect=Exception("Other Redis error")),
        patch("lightrag.api.health.get_config") as mock_get_config,
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "REDIS_URI": "redis://test_redis:6379/0",
        }.get(key, default)
        result = await health_checker_instance._check_redis()
        assert result["healthy"] is False
        assert result["status"] == "error"
        assert "error" in result


@pytest.mark.asyncio
async def test_check_system_resources_healthy(health_checker_instance):
    with (
        patch("psutil.virtual_memory") as mock_virtual_memory,
        patch("psutil.cpu_percent") as mock_cpu_percent,
        patch("psutil.disk_usage") as mock_disk_usage,
    ):
        mock_virtual_memory.return_value.percent = 50
        mock_virtual_memory.return_value.available = 100 * 1024 * 1024 * 1024  # 100 GB
        mock_cpu_percent.return_value = 50
        mock_disk_usage.return_value.percent = 50
        mock_disk_usage.return_value.free = 100 * 1024 * 1024 * 1024  # 100 GB

        result = await health_checker_instance._check_system_resources()
        assert result["healthy"] is True
        assert result["memory"]["healthy"] is True
        assert result["cpu"]["healthy"] is True
        assert result["disk"]["healthy"] is True


@pytest.mark.asyncio
async def test_check_system_resources_unhealthy(health_checker_instance):
    with (
        patch("psutil.virtual_memory") as mock_virtual_memory,
        patch("psutil.cpu_percent") as mock_cpu_percent,
        patch("psutil.disk_usage") as mock_disk_usage,
    ):
        mock_virtual_memory.return_value.percent = 95  # Unhealthy memory
        mock_virtual_memory.return_value.available = 1 * 1024 * 1024 * 1024
        mock_cpu_percent.return_value = 95  # Unhealthy CPU
        mock_disk_usage.return_value.percent = 95  # Unhealthy disk
        mock_disk_usage.return_value.free = 1 * 1024 * 1024 * 1024

        result = await health_checker_instance._check_system_resources()
        assert result["healthy"] is False
        assert result["memory"]["healthy"] is False
        assert result["cpu"]["healthy"] is False
        assert result["disk"]["healthy"] is False


@pytest.mark.asyncio
async def test_check_system_resources_error(health_checker_instance):
    with patch("psutil.virtual_memory", side_effect=Exception("System error")):
        result = await health_checker_instance._check_system_resources()
        assert result["healthy"] is False
        assert "error" in result


@pytest.mark.asyncio
async def test_check_llm_configured(health_checker_instance):
    with patch("lightrag.api.health.get_config") as mock_get_config:
        mock_get_config.side_effect = lambda key, default=None: {
            "LLM_API_KEY": "test_llm_key",
            "LLM_BINDING": "test_llm_binding",
            "LLM_MODEL": "test_llm_model",
        }.get(key, default)
        result = await health_checker_instance._check_llm()
        assert result["healthy"] is True
        assert result["status"] == "configured"
        assert result["provider"] == "test_llm_binding"
        assert result["model"] == "test_llm_model"


@pytest.mark.asyncio
async def test_check_llm_error(health_checker_instance):
    # Simulate an error during LLM check (e.g., if LLM_BINDING was set to something invalid)
    with patch(
        "lightrag.api.health.get_config", side_effect=Exception("LLM config error")
    ):
        result = await health_checker_instance._check_llm()
        assert result["healthy"] is False
        assert result["status"] == "error"
        assert "error" in result


@pytest.mark.asyncio
async def test_overall_health_healthy(health_checker_instance):
    with (
        patch("lightrag.api.health.get_config") as mock_get_config,
        patch.object(
            health_checker_instance,
            "_check_application",
            return_value={"healthy": True},
        ),
        patch.object(
            health_checker_instance, "_check_database", return_value={"healthy": True}
        ),
        patch.object(
            health_checker_instance, "_check_redis", return_value={"healthy": True}
        ),
        patch.object(
            health_checker_instance,
            "_check_system_resources",
            return_value={"healthy": True},
        ),
        patch.object(
            health_checker_instance, "_check_llm", return_value={"healthy": True}
        ),
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "POSTGRES_HOST": "test_db_host",
            "REDIS_URI": "redis://test_redis:6379/0",
            "LLM_API_KEY": "test_llm_key",
        }.get(key, default)

        status = await health_checker_instance.check_health()
        assert status.healthy is True
        assert status.status == "healthy"
        assert "application" in status.checks
        assert "database" in status.checks
        assert "redis" in status.checks
        assert "system" in status.checks
        assert "llm" in status.checks


@pytest.mark.asyncio
async def test_overall_health_unhealthy_database(health_checker_instance):
    with (
        patch("lightrag.api.health.get_config") as mock_get_config,
        patch.object(
            health_checker_instance,
            "_check_application",
            return_value={"healthy": True},
        ),
        patch.object(
            health_checker_instance,
            "_check_database",
            return_value={"healthy": False, "error": "DB down"},
        ),
        patch.object(
            health_checker_instance, "_check_redis", return_value={"healthy": True}
        ),
        patch.object(
            health_checker_instance,
            "_check_system_resources",
            return_value={"healthy": True},
        ),
        patch.object(
            health_checker_instance, "_check_llm", return_value={"healthy": True}
        ),
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "POSTGRES_HOST": "test_db_host",
            "REDIS_URI": "redis://test_redis:6379/0",
            "LLM_API_KEY": "test_llm_key",
        }.get(key, default)

        status = await health_checker_instance.check_health()
        assert status.healthy is False
        assert status.status == "unhealthy"
        assert status.checks["database"]["healthy"] is False
        assert status.checks["database"]["error"] == "DB down"


@pytest.mark.asyncio
async def test_overall_health_error(health_checker_instance):
    with patch.object(
        health_checker_instance,
        "_check_application",
        side_effect=Exception("Overall error"),
    ):
        status = await health_checker_instance.check_health()
        assert status.healthy is False
        assert status.status == "error"
        assert "error" in status.checks


@pytest.mark.asyncio
async def test_get_health_status(health_checker_instance):
    with (
        patch("lightrag.api.health.health_checker", health_checker_instance),
        patch.object(
            health_checker_instance,
            "check_health",
            return_value=HealthStatus(
                healthy=True,
                status="healthy",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        ),
    ):
        status = await get_health_status()
        assert status["healthy"] is True
        assert status["status"] == "healthy"


def test_get_readiness_status():
    """Test readiness status endpoint"""
    status = get_readiness_status()
    assert status["ready"] is True
    assert status["status"] == "ready"
    assert "timestamp" in status
    assert isinstance(status["timestamp"], str)


def test_get_liveness_status():
    """Test liveness status endpoint"""
    status = get_liveness_status()
    assert status["alive"] is True
    assert status["status"] == "alive"
    assert "timestamp" in status
    assert isinstance(status["timestamp"], str)


@pytest.mark.asyncio
async def test_health_checker_close_with_connections(health_checker_instance):
    """Test closing health checker with active connections"""
    # Mock connections
    mock_pg_conn = MagicMock()
    mock_redis_client = MagicMock()

    health_checker_instance.postgres_conn = mock_pg_conn
    health_checker_instance.redis_client = mock_redis_client

    # Call close
    health_checker_instance.close()

    # Verify connections were closed
    mock_pg_conn.close.assert_called_once()
    mock_redis_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_health_checker_close_no_connections(health_checker_instance):
    """Test closing health checker with no active connections"""
    # Ensure no connections are set
    health_checker_instance.postgres_conn = None
    health_checker_instance.redis_client = None

    # This should not raise any exceptions
    health_checker_instance.close()


@pytest.mark.asyncio
async def test_health_checker_close_connection_errors(health_checker_instance):
    """Test closing health checker when connection close methods raise exceptions"""
    # Mock connections that raise exceptions on close
    mock_pg_conn = MagicMock()
    mock_redis_client = MagicMock()
    mock_pg_conn.close.side_effect = Exception("PostgreSQL close error")
    mock_redis_client.close.side_effect = Exception("Redis close error")

    health_checker_instance.postgres_conn = mock_pg_conn
    health_checker_instance.redis_client = mock_redis_client

    # This should not raise any exceptions despite the errors
    health_checker_instance.close()

    # Verify close was attempted on both connections
    mock_pg_conn.close.assert_called_once()
    mock_redis_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_overall_health_multiple_failures(health_checker_instance):
    """Test overall health with multiple component failures"""
    with (
        patch("lightrag.api.health.get_config") as mock_get_config,
        patch.object(
            health_checker_instance,
            "_check_application",
            return_value={"healthy": False, "error": "App failure"},
        ),
        patch.object(
            health_checker_instance,
            "_check_database",
            return_value={"healthy": False, "error": "DB failure"},
        ),
        patch.object(
            health_checker_instance,
            "_check_redis",
            return_value={"healthy": False, "error": "Redis failure"},
        ),
        patch.object(
            health_checker_instance,
            "_check_system_resources",
            return_value={"healthy": False, "error": "System failure"},
        ),
        patch.object(
            health_checker_instance,
            "_check_llm",
            return_value={"healthy": False, "error": "LLM failure"},
        ),
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "POSTGRES_HOST": "test_db_host",
            "REDIS_URI": "redis://test_redis:6379/0",
            "LLM_API_KEY": "test_llm_key",
        }.get(key, default)

        status = await health_checker_instance.check_health()
        assert status.healthy is False
        assert status.status == "unhealthy"
        assert status.checks["application"]["healthy"] is False
        assert status.checks["database"]["healthy"] is False
        assert status.checks["redis"]["healthy"] is False
        assert status.checks["system"]["healthy"] is False
        assert status.checks["llm"]["healthy"] is False
