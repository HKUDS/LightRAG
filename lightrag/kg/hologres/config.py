"""Environment-only configuration for the isolated Hologres backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import os
from typing import Mapping


class HologresConfigError(ValueError):
    """Raised when Hologres environment configuration is invalid."""


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_SSL_MODES = frozenset(
    {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
)


def _validate_required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HologresConfigError(f"{name} is required")
    return value


def _required(environment: Mapping[str, str], name: str) -> str:
    return _validate_required_text(environment.get(name), name)


def _validate_integer_value(
    value: object, name: str, *, minimum: int, maximum: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HologresConfigError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise HologresConfigError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise HologresConfigError(f"{name} must be an integer") from None
    return _validate_integer_value(
        value, name, minimum=minimum, maximum=maximum
    )


def _validate_number_value(
    value: object, name: str, *, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HologresConfigError(f"{name} must be a number")
    number = float(value)
    if not isfinite(number) or number < minimum or number > maximum:
        raise HologresConfigError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return number


def _number(
    environment: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise HologresConfigError(f"{name} must be a number") from None
    return _validate_number_value(
        value, name, minimum=minimum, maximum=maximum
    )


def _validate_boolean_value(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise HologresConfigError(f"{name} must be a boolean")
    return value


def _boolean(
    environment: Mapping[str, str], name: str, default: bool
) -> bool:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise HologresConfigError(f"{name} must be a boolean")


def _validate_ssl_mode(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise HologresConfigError(f"{name} is invalid")
    normalized = value.strip().lower()
    if normalized not in _SSL_MODES:
        raise HologresConfigError(f"{name} is invalid")
    return normalized


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise HologresConfigError(f"{name} is required")
    first = value[0]
    if not (first == "_" or first.isascii() and first.isalpha()):
        raise HologresConfigError(f"{name} must be a valid SQL identifier")
    if not all(
        character == "_" or character.isascii() and character.isalnum()
        for character in value
    ):
        raise HologresConfigError(f"{name} must be a valid SQL identifier")
    if len(value.encode("utf-8")) > 63:
        raise HologresConfigError(f"{name} must be a valid SQL identifier")
    return value


@dataclass(frozen=True, repr=False)
class HologresConfig:
    """Validated Hologres connection and client settings.

    Connection values deliberately have a custom representation so diagnostics
    cannot accidentally disclose credentials or a credential-bearing endpoint.
    """

    host: str
    port: int
    user: str
    password: str
    database: str
    schema: str = "public"
    pool_min_size: int = 1
    pool_max_size: int = 10
    connect_timeout: float = 10.0
    command_timeout: float = 60.0
    pool_acquire_timeout: float = 30.0
    pool_close_timeout: float = 5.0
    connection_retries: int = 2
    retry_backoff: float = 0.5
    statement_cache_size: int = 100
    ssl_mode: str = "prefer"
    stream_copy_enabled: bool = False
    age_search_path: bool = False

    def __post_init__(self) -> None:
        """Validate direct construction and normalize canonical field values."""

        validated = {
            "host": _validate_required_text(self.host, "HOLOGRES_HOST"),
            "port": _validate_integer_value(
                self.port, "HOLOGRES_PORT", minimum=1, maximum=65535
            ),
            "user": _validate_required_text(self.user, "HOLOGRES_USER"),
            "password": _validate_required_text(
                self.password, "HOLOGRES_PASSWORD"
            ),
            "database": _validate_required_text(
                self.database, "HOLOGRES_DATABASE"
            ),
            "schema": _identifier(self.schema, "HOLOGRES_SCHEMA"),
            "pool_min_size": _validate_integer_value(
                self.pool_min_size,
                "HOLOGRES_POOL_MIN_SIZE",
                minimum=1,
                maximum=1000,
            ),
            "pool_max_size": _validate_integer_value(
                self.pool_max_size,
                "HOLOGRES_POOL_MAX_SIZE",
                minimum=1,
                maximum=1000,
            ),
            "connect_timeout": _validate_number_value(
                self.connect_timeout,
                "HOLOGRES_CONNECT_TIMEOUT",
                minimum=0.001,
                maximum=600.0,
            ),
            "command_timeout": _validate_number_value(
                self.command_timeout,
                "HOLOGRES_COMMAND_TIMEOUT",
                minimum=0.001,
                maximum=3600.0,
            ),
            "pool_acquire_timeout": _validate_number_value(
                self.pool_acquire_timeout,
                "HOLOGRES_POOL_ACQUIRE_TIMEOUT",
                minimum=0.001,
                maximum=600.0,
            ),
            "pool_close_timeout": _validate_number_value(
                self.pool_close_timeout,
                "HOLOGRES_POOL_CLOSE_TIMEOUT",
                minimum=0.001,
                maximum=60.0,
            ),
            "connection_retries": _validate_integer_value(
                self.connection_retries,
                "HOLOGRES_CONNECTION_RETRIES",
                minimum=0,
                maximum=20,
            ),
            "retry_backoff": _validate_number_value(
                self.retry_backoff,
                "HOLOGRES_RETRY_BACKOFF",
                minimum=0.0,
                maximum=60.0,
            ),
            "statement_cache_size": _validate_integer_value(
                self.statement_cache_size,
                "HOLOGRES_STATEMENT_CACHE_SIZE",
                minimum=0,
                maximum=10000,
            ),
            "ssl_mode": _validate_ssl_mode(
                self.ssl_mode, "HOLOGRES_SSL_MODE"
            ),
            "stream_copy_enabled": _validate_boolean_value(
                self.stream_copy_enabled, "HOLOGRES_STREAM_COPY_ENABLED"
            ),
            "age_search_path": _validate_boolean_value(
                self.age_search_path, "HOLOGRES_AGE_SEARCH_PATH"
            ),
        }
        if validated["pool_max_size"] < validated["pool_min_size"]:
            raise HologresConfigError(
                "HOLOGRES_POOL_MAX_SIZE must be at least HOLOGRES_POOL_MIN_SIZE"
            )
        for field_name, value in validated.items():
            object.__setattr__(self, field_name, value)

    @classmethod
    def from_env(
        cls, environment: Mapping[str, str] | None = None
    ) -> "HologresConfig":
        """Build configuration exclusively from ``HOLOGRES_*`` variables."""

        source = os.environ if environment is None else environment
        pool_min_size = _integer(
            source,
            "HOLOGRES_POOL_MIN_SIZE",
            1,
            minimum=1,
            maximum=1000,
        )
        pool_max_size = _integer(
            source,
            "HOLOGRES_POOL_MAX_SIZE",
            10,
            minimum=1,
            maximum=1000,
        )
        if pool_max_size < pool_min_size:
            raise HologresConfigError(
                "HOLOGRES_POOL_MAX_SIZE must be at least HOLOGRES_POOL_MIN_SIZE"
            )

        ssl_mode = source.get("HOLOGRES_SSL_MODE", "prefer").strip().lower()
        if ssl_mode not in _SSL_MODES:
            raise HologresConfigError("HOLOGRES_SSL_MODE is invalid")

        return cls(
            host=_required(source, "HOLOGRES_HOST"),
            port=_integer(
                source, "HOLOGRES_PORT", 80, minimum=1, maximum=65535
            ),
            user=_required(source, "HOLOGRES_USER"),
            password=_required(source, "HOLOGRES_PASSWORD"),
            database=_required(source, "HOLOGRES_DATABASE"),
            schema=_identifier(source.get("HOLOGRES_SCHEMA", "public"), "HOLOGRES_SCHEMA"),
            pool_min_size=pool_min_size,
            pool_max_size=pool_max_size,
            connect_timeout=_number(
                source,
                "HOLOGRES_CONNECT_TIMEOUT",
                10.0,
                minimum=0.001,
                maximum=600.0,
            ),
            command_timeout=_number(
                source,
                "HOLOGRES_COMMAND_TIMEOUT",
                60.0,
                minimum=0.001,
                maximum=3600.0,
            ),
            pool_acquire_timeout=_number(
                source,
                "HOLOGRES_POOL_ACQUIRE_TIMEOUT",
                30.0,
                minimum=0.001,
                maximum=600.0,
            ),
            pool_close_timeout=_number(
                source,
                "HOLOGRES_POOL_CLOSE_TIMEOUT",
                5.0,
                minimum=0.001,
                maximum=60.0,
            ),
            connection_retries=_integer(
                source,
                "HOLOGRES_CONNECTION_RETRIES",
                2,
                minimum=0,
                maximum=20,
            ),
            retry_backoff=_number(
                source,
                "HOLOGRES_RETRY_BACKOFF",
                0.5,
                minimum=0.0,
                maximum=60.0,
            ),
            statement_cache_size=_integer(
                source,
                "HOLOGRES_STATEMENT_CACHE_SIZE",
                100,
                minimum=0,
                maximum=10000,
            ),
            ssl_mode=ssl_mode,
            stream_copy_enabled=_boolean(
                source, "HOLOGRES_STREAM_COPY_ENABLED", False
            ),
            age_search_path=_boolean(
                source, "HOLOGRES_AGE_SEARCH_PATH", False
            ),
        )

    def __repr__(self) -> str:
        return (
            "HologresConfig("
            "host='<redacted>', "
            f"port={self.port}, "
            "user='<redacted>', "
            "password='<redacted>', "
            "database='<redacted>', "
            f"schema={self.schema!r}, "
            f"pool_min_size={self.pool_min_size}, "
            f"pool_max_size={self.pool_max_size}, "
            f"connect_timeout={self.connect_timeout}, "
            f"command_timeout={self.command_timeout}, "
            f"pool_acquire_timeout={self.pool_acquire_timeout}, "
            f"pool_close_timeout={self.pool_close_timeout}, "
            f"connection_retries={self.connection_retries}, "
            f"retry_backoff={self.retry_backoff}, "
            f"statement_cache_size={self.statement_cache_size}, "
            f"ssl_mode={self.ssl_mode!r}, "
            f"stream_copy_enabled={self.stream_copy_enabled}, "
            f"age_search_path={self.age_search_path})"
        )
