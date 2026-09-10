import pytest

from lightrag.kg.hologres.config import HologresConfig, HologresConfigError


BASE_ENV = {
    "HOLOGRES_HOST": "example.hologres.aliyuncs.com",
    "HOLOGRES_PORT": "80",
    "HOLOGRES_USER": "test_user",
    "HOLOGRES_PASSWORD": "top-secret-password",
    "HOLOGRES_DATABASE": "analytics",
}


def test_config_reads_valid_hologres_environment_and_redacts_password():
    config = HologresConfig.from_env(
        {
            **BASE_ENV,
            "HOLOGRES_SCHEMA": "LightRAG_1",
            "HOLOGRES_POOL_MIN_SIZE": "2",
            "HOLOGRES_POOL_MAX_SIZE": "7",
            "HOLOGRES_CONNECT_TIMEOUT": "12.5",
            "HOLOGRES_COMMAND_TIMEOUT": "45",
            "HOLOGRES_POOL_ACQUIRE_TIMEOUT": "6.5",
            "HOLOGRES_POOL_CLOSE_TIMEOUT": "4.5",
            "HOLOGRES_CONNECTION_RETRIES": "3",
            "HOLOGRES_RETRY_BACKOFF": "0.25",
            "HOLOGRES_STREAM_COPY_ENABLED": "yes",
            "HOLOGRES_AGE_SEARCH_PATH": "yes",
            "POSTGRES_PASSWORD": "must-not-be-read",
        }
    )

    assert config.host == "example.hologres.aliyuncs.com"
    assert config.port == 80
    assert config.user == "test_user"
    assert config.password == "top-secret-password"
    assert config.database == "analytics"
    assert config.schema == "LightRAG_1"
    assert config.pool_min_size == 2
    assert config.pool_max_size == 7
    assert config.connect_timeout == 12.5
    assert config.command_timeout == 45.0
    assert config.pool_acquire_timeout == 6.5
    assert config.pool_close_timeout == 4.5
    assert config.connection_retries == 3
    assert config.retry_backoff == 0.25
    assert config.stream_copy_enabled is True
    assert config.age_search_path is True
    assert "example.hologres.aliyuncs.com" not in repr(config)
    assert "test_user" not in repr(config)
    assert "top-secret-password" not in repr(config)
    assert "analytics" not in repr(config)
    assert "must-not-be-read" not in repr(config)
    assert "<redacted>" in repr(config)


@pytest.mark.parametrize(
    ("updates", "expected_fragment"),
    [
        ({"HOLOGRES_PORT": "0"}, "HOLOGRES_PORT"),
        ({"HOLOGRES_PORT": "not-an-int"}, "HOLOGRES_PORT"),
        ({"HOLOGRES_POOL_MIN_SIZE": "0"}, "HOLOGRES_POOL_MIN_SIZE"),
        (
            {"HOLOGRES_POOL_MIN_SIZE": "4", "HOLOGRES_POOL_MAX_SIZE": "3"},
            "HOLOGRES_POOL_MAX_SIZE",
        ),
        ({"HOLOGRES_CONNECT_TIMEOUT": "0"}, "HOLOGRES_CONNECT_TIMEOUT"),
        ({"HOLOGRES_COMMAND_TIMEOUT": "nan"}, "HOLOGRES_COMMAND_TIMEOUT"),
        ({"HOLOGRES_CONNECTION_RETRIES": "-1"}, "HOLOGRES_CONNECTION_RETRIES"),
        ({"HOLOGRES_RETRY_BACKOFF": "-0.1"}, "HOLOGRES_RETRY_BACKOFF"),
        ({"HOLOGRES_STREAM_COPY_ENABLED": "sometimes"}, "HOLOGRES_STREAM_COPY_ENABLED"),
        ({"HOLOGRES_SCHEMA": "public; DROP SCHEMA public"}, "HOLOGRES_SCHEMA"),
    ],
)
def test_config_rejects_invalid_values_without_echoing_them(updates, expected_fragment):
    environment = {**BASE_ENV, **updates}
    invalid_value = next(iter(updates.values()))

    with pytest.raises(HologresConfigError) as exc_info:
        HologresConfig.from_env(environment)

    assert expected_fragment in str(exc_info.value)
    assert repr(invalid_value) not in str(exc_info.value)


def test_config_requires_connection_fields_without_echoing_other_secrets():
    environment = {**BASE_ENV}
    del environment["HOLOGRES_HOST"]

    with pytest.raises(HologresConfigError) as exc_info:
        HologresConfig.from_env(environment)

    assert str(exc_info.value) == "HOLOGRES_HOST is required"
    assert BASE_ENV["HOLOGRES_PASSWORD"] not in str(exc_info.value)


def _direct_config(**updates):
    values = {
        "host": "example.hologres.aliyuncs.com",
        "port": 80,
        "user": "test_user",
        "password": "secret",
        "database": "analytics",
    }
    values.update(updates)
    return HologresConfig(**values)


@pytest.mark.parametrize(
    "updates",
    [
        {"host": ""},
        {"user": ""},
        {"password": ""},
        {"database": ""},
        {"port": 0},
        {"port": 80.0},
        {"schema": "public;drop"},
        {"pool_min_size": 0},
        {"pool_max_size": 1001},
        {"pool_min_size": 4, "pool_max_size": 3},
        {"connect_timeout": float("nan")},
        {"command_timeout": float("inf")},
        {"pool_acquire_timeout": 0},
        {"pool_close_timeout": 61},
        {"connection_retries": -1},
        {"connection_retries": 1.5},
        {"retry_backoff": float("nan")},
        {"statement_cache_size": -1},
        {"statement_cache_size": 1.5},
        {"ssl_mode": "sometimes"},
        {"ssl_mode": 1},
        {"stream_copy_enabled": "true"},
        {"stream_copy_enabled": 1},
        {"age_search_path": "true"},
        {"age_search_path": 1},
    ],
)
def test_direct_construction_cannot_bypass_configuration_validation(updates):
    with pytest.raises(HologresConfigError):
        _direct_config(**updates)


def test_direct_construction_normalizes_ssl_and_numeric_timeouts():
    config = _direct_config(ssl_mode="REQUIRE", connect_timeout=2)

    assert config.ssl_mode == "require"
    assert config.connect_timeout == 2.0
