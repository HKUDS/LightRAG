from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, final

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")
_DEFAULT_SPACE_PREFIX = "lightrag"
_DEFAULT_WORKSPACE = "base"
_MAX_SPACE_NAME_LEN = 127
_DEFAULT_NEBULA_PORT = 9669
_DEFAULT_PARTITION_NUM = 10
_DEFAULT_REPLICA_FACTOR = 1
# Live Nebula clusters can take >10s to surface new spaces/tags to graphd.
_DEFAULT_SCHEMA_RETRY_TIMES = 60
_DEFAULT_SCHEMA_RETRY_DELAY_MS = 200
_INDEX_STATUS_DONE_TOKENS = ("FINISHED", "SUCCEEDED", "SUCCESS", "DONE", "COMPLETED")
_INDEX_STATUS_PENDING_TOKENS = (
    "RUNNING",
    "QUEUE",
    "QUEUED",
    "QUEUING",
    "IN_PROGRESS",
    "BUILDING",
    "STARTING",
)
_INDEX_STATUS_FAILED_TOKENS = ("FAILED", "FAIL", "ERROR", "CANCELLED", "STOPPED")
_NODE_FIELDS = (
    "entity_id",
    "name",
    "entity_type",
    "description",
    "keywords",
    "source_id",
)
_EDGE_FIELDS = (
    "source_id",
    "target_id",
    "relationship",
    "description",
    "weight",
)


class NebulaIndexJobError(RuntimeError):
    """Raised when Nebula explicitly reports an index job failure."""


def _load_nebula_client_types() -> tuple[Any, Any]:
    try:
        from nebula3.Config import Config  # type: ignore
        from nebula3.gclient.net import ConnectionPool  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "nebula3-python is required for NebulaGraphStorage. Install with `uv add nebula3-python`."
        ) from exc
    return Config, ConnectionPool


def _canonical_edge_pair(src: str, tgt: str) -> tuple[str, str]:
    a, b = sorted((src, tgt))
    return a, b


def _sanitize_workspace_component(value: str, *, fallback: str) -> str:
    normalized = _SAFE_NAME_RE.sub("_", value.strip().lower()).strip("_")
    return normalized or fallback


def _short_hash_suffix(value: str, *, length: int = 8) -> str:
    if length <= 0:
        raise ValueError("hash suffix length must be positive")
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _normalize_space_name(prefix: str, workspace: str) -> str:
    safe_prefix = _sanitize_workspace_component(prefix, fallback=_DEFAULT_SPACE_PREFIX)
    safe_workspace = _sanitize_workspace_component(
        workspace, fallback=_DEFAULT_WORKSPACE
    )
    base_name = f"{safe_prefix}__{safe_workspace}"
    if len(base_name) <= _MAX_SPACE_NAME_LEN:
        return base_name

    suffix = _short_hash_suffix(base_name)
    head_max_len = _MAX_SPACE_NAME_LEN - len(suffix) - 2
    trimmed = base_name[: max(1, head_max_len)].rstrip("_")
    return f"{trimmed}__{suffix}"


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    cleaned = value.strip()
    if not cleaned:
        return default
    return cleaned


def _env_raw(name: str, default: str | None = None) -> str | None:
    """Return the raw env value without stripping empty strings to None."""
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer: {raw}") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _parse_nebula_hosts(hosts_value: str | None) -> list[tuple[str, int]]:
    if hosts_value is None:
        return []
    hosts: list[tuple[str, int]] = []
    for item in hosts_value.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        host = ""
        port = _DEFAULT_NEBULA_PORT

        if candidate.startswith("["):
            right = candidate.find("]")
            if right < 0:
                raise ValueError(
                    f"Invalid Nebula host entry '{candidate}': missing closing ']' for IPv6 host"
                )
            host = candidate[1:right].strip()
            if not host:
                raise ValueError(
                    f"Invalid Nebula host entry '{candidate}': empty IPv6 host"
                )
            tail = candidate[right + 1 :].strip()
            if tail:
                if not tail.startswith(":"):
                    raise ValueError(
                        f"Invalid Nebula host entry '{candidate}': expected ':port' after bracketed host"
                    )
                port_text = tail[1:].strip()
                if not port_text:
                    raise ValueError(
                        f"Invalid Nebula host entry '{candidate}': missing port after ':'"
                    )
                try:
                    port = int(port_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid Nebula port in host entry '{candidate}': '{port_text}' is not an integer"
                    ) from exc
        elif ":" in candidate:
            if candidate.count(":") > 1:
                host = candidate
            else:
                host, port_text = candidate.rsplit(":", 1)
                host = host.strip()
                port_text = port_text.strip()
                if not host:
                    raise ValueError(
                        f"Invalid Nebula host entry '{candidate}': host is empty"
                    )
                if not port_text:
                    raise ValueError(
                        f"Invalid Nebula host entry '{candidate}': port is empty"
                    )
                try:
                    port = int(port_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid Nebula port in host entry '{candidate}': '{port_text}' is not an integer"
                    ) from exc
        else:
            host = candidate
            if not host:
                raise ValueError(f"Invalid Nebula host entry '{candidate}': host is empty")

        if not (1 <= port <= 65535):
            raise ValueError(
                f"Invalid Nebula port in host entry '{candidate}': {port} must be in 1..65535"
            )
        hosts.append((host, port))
    return hosts


def _flatten_result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="ignore")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return " ".join(_flatten_result_text(v) for v in result.values())
    if isinstance(result, (list, tuple, set)):
        return " ".join(_flatten_result_text(v) for v in result)

    rows_attr = getattr(result, "rows", None)
    if callable(rows_attr):
        try:
            return _flatten_result_text(rows_attr())
        except Exception:
            pass
    elif rows_attr is not None:
        return _flatten_result_text(rows_attr)

    return str(result)


def _ngql_escape_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace('"', '\\"')
    )


def _ngql_quote(value: Any) -> str:
    return f'"{_ngql_escape_string(str(value))}"'


def _ngql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return _ngql_quote(value)


def _coerce_edge_weight(value: Any) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 1.0


def _decode_if_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def _unwrap_nebula_value(value: Any) -> Any:
    if value is None:
        return None

    cast_fn = getattr(value, "cast", None)
    if callable(cast_fn):
        try:
            return _decode_if_bytes(cast_fn())
        except Exception:
            pass

    for check_name, read_name in (
        ("is_null", None),
        ("is_string", "as_string"),
        ("is_int", "as_int"),
        ("is_double", "as_double"),
        ("is_bool", "as_bool"),
    ):
        check_fn = getattr(value, check_name, None)
        if not callable(check_fn):
            continue
        try:
            checked = check_fn()
        except Exception:
            continue
        if not checked:
            continue
        if read_name is None:
            return None
        read_fn = getattr(value, read_name, None)
        if callable(read_fn):
            try:
                return _decode_if_bytes(read_fn())
            except Exception:
                pass

    for getter_name in (
        "get_nVal",
        "get_sVal",
        "get_iVal",
        "get_bVal",
        "get_fVal",
        "get_dVal",
        "get_lVal",
    ):
        getter = getattr(value, getter_name, None)
        if callable(getter):
            try:
                if getter_name == "get_nVal":
                    getter()
                    return None
                return _decode_if_bytes(getter())
            except Exception:
                pass

    if isinstance(value, dict):
        return {str(k): _unwrap_nebula_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unwrap_nebula_value(v) for v in value]
    return _decode_if_bytes(value)


def _result_to_rows(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []

    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        rows: list[dict[str, Any]] = []
        for item in result:
            if isinstance(item, dict):
                rows.append(item)
            elif isinstance(item, (list, tuple)):
                rows.append(
                    {f"col_{idx}": _unwrap_nebula_value(value) for idx, value in enumerate(item)}
                )
        return rows

    keys: list[str] = []
    keys_fn = getattr(result, "keys", None)
    if callable(keys_fn):
        try:
            raw_keys = keys_fn()
            keys = [str(_decode_if_bytes(k)) for k in raw_keys]
        except Exception:
            keys = []

    rows_attr = getattr(result, "rows", None)
    if callable(rows_attr):
        try:
            raw_rows = rows_attr()
        except Exception:
            raw_rows = []
    else:
        raw_rows = rows_attr if isinstance(rows_attr, list) else []

    parsed_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            parsed_rows.append(row)
            continue
        values = None
        values_attr = getattr(row, "values", None)
        if callable(values_attr):
            try:
                values = values_attr()
            except Exception:
                values = None
        elif isinstance(values_attr, (list, tuple)):
            values = values_attr
        if values is None:
            continue
        row_values = [_unwrap_nebula_value(v) for v in values]
        if keys and len(keys) == len(row_values):
            parsed_rows.append(dict(zip(keys, row_values, strict=True)))
            continue
        parsed_rows.append(
            {f"col_{idx}": value for idx, value in enumerate(row_values)}
        )
    return parsed_rows


def _first_row(result: Any) -> dict[str, Any] | None:
    rows = _result_to_rows(result)
    if not rows:
        return None
    return rows[0]


def _normalize_listener_endpoint(endpoint: Any) -> str:
    value = str(endpoint).strip()
    value = value.replace('"', "").replace("'", "")
    return value


@final
@dataclass
class NebulaGraphStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        env_workspace = _env_str("NEBULA_WORKSPACE")
        resolved_workspace = env_workspace if env_workspace is not None else workspace
        if not resolved_workspace or not str(resolved_workspace).strip():
            resolved_workspace = _DEFAULT_WORKSPACE
        super().__init__(
            namespace=namespace,
            workspace=str(resolved_workspace),
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._space_prefix = _env_str("NEBULA_SPACE_PREFIX", _DEFAULT_SPACE_PREFIX)
        self._space_name = _normalize_space_name(
            self._space_prefix or _DEFAULT_SPACE_PREFIX, self.workspace
        )
        self._fulltext_tag_index_name = (
            f"nebula_entity_name_ft_{_short_hash_suffix(self._space_name, length=8)}"
        )
        self._fulltext_edge_index_name = (
            f"nebula_relation_rel_ft_{_short_hash_suffix(self._space_name, length=8)}"
        )
        self._hosts = _parse_nebula_hosts(_env_str("NEBULA_HOSTS"))
        self._user = _env_str("NEBULA_USER")
        self._password = _env_raw("NEBULA_PASSWORD")
        self._timeout_ms = _env_int("NEBULA_TIMEOUT_MS", 60_000)
        self._ssl_enabled = _env_bool("NEBULA_SSL", default=False)
        self._use_http2 = _env_bool("NEBULA_USE_HTTP2", default=False)
        self._listener_hosts = [
            _normalize_listener_endpoint(item)
            for item in (_env_str("NEBULA_LISTENER_HOSTS", "") or "").split(",")
            if item.strip()
        ]
        self._partition_num = _env_int("NEBULA_SPACE_PARTITIONS", _DEFAULT_PARTITION_NUM)
        self._replica_factor = _env_int(
            "NEBULA_SPACE_REPLICA_FACTOR", _DEFAULT_REPLICA_FACTOR
        )
        self._schema_retry_times = _env_int(
            "NEBULA_SCHEMA_RETRY_TIMES", _DEFAULT_SCHEMA_RETRY_TIMES
        )
        self._schema_retry_delay_ms = _env_int(
            "NEBULA_SCHEMA_RETRY_DELAY_MS", _DEFAULT_SCHEMA_RETRY_DELAY_MS
        )

        self._connection_pool: Any | None = None
        self._fulltext_init_error: str | None = None
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        async with self._initialize_lock:
            if self._initialized:
                return
            self._validate_required_env()
            try:
                await self._bootstrap_client()
                await self._ensure_space_ready()
            except Exception:
                await self._close_connection_pool()
                self._initialized = False
                raise
            self._initialized = True

    async def finalize(self):
        await self._close_connection_pool()
        self._initialized = False

    async def _close_connection_pool(self) -> None:
        if self._connection_pool is not None:
            await asyncio.to_thread(self._connection_pool.close)
            self._connection_pool = None

    def _validate_required_env(self) -> None:
        if not self._hosts:
            raise ValueError(
                "Environment variable NEBULA_HOSTS is required and must not be empty."
            )
        if self._user is None or not str(self._user).strip():
            raise ValueError(
                "Environment variable NEBULA_USER is required and must not be empty."
            )
        if self._password is None:
            raise ValueError("Environment variable NEBULA_PASSWORD is required.")

    async def _bootstrap_client(self) -> None:
        if self._connection_pool is not None:
            return

        Config, ConnectionPool = _load_nebula_client_types()
        config = Config()
        config.timeout = self._timeout_ms
        config.max_connection_pool_size = _env_int("NEBULA_MAX_CONNECTION_POOL_SIZE", 10)
        config.min_connection_pool_size = _env_int("NEBULA_MIN_CONNECTION_POOL_SIZE", 1)
        config.use_http2 = self._use_http2

        connection_pool = ConnectionPool()
        ok = await asyncio.to_thread(connection_pool.init, self._hosts, config)
        if not ok:
            raise RuntimeError("Failed to initialize Nebula connection pool.")

        self._connection_pool = connection_pool

    async def _acquire_session(self) -> Any:
        if self._connection_pool is None:
            raise RuntimeError("Nebula connection pool is not initialized.")
        return await asyncio.to_thread(
            self._connection_pool.get_session, self._user, self._password
        )

    async def _release_session(self, session: Any) -> None:
        if hasattr(session, "release"):
            await asyncio.to_thread(session.release)
            return
        if hasattr(session, "signout"):
            await asyncio.to_thread(session.signout)

    async def _execute(self, statement: str, *, session: Any | None = None) -> Any:
        owns_session = session is None
        active_session = session if session is not None else await self._acquire_session()
        if active_session is None:
            raise RuntimeError("Failed to acquire Nebula session from connection pool.")
        try:
            result = await asyncio.to_thread(active_session.execute, statement)
        finally:
            if owns_session:
                await self._release_session(active_session)
        if hasattr(result, "is_succeeded") and not result.is_succeeded():
            error_msg = "unknown error"
            if hasattr(result, "error_msg"):
                raw = result.error_msg()
                if isinstance(raw, bytes):
                    error_msg = raw.decode("utf-8", errors="ignore")
                else:
                    error_msg = str(raw)
            raise RuntimeError(f"Nebula query failed: {statement} ({error_msg})")
        return result

    async def _execute_in_space(self, statement: str) -> Any:
        session = await self._acquire_session()
        if session is None:
            raise RuntimeError("Failed to acquire Nebula session from connection pool.")
        try:
            await self._use_space(session)
            return await self._execute(statement, session=session)
        finally:
            await self._release_session(session)

    async def _ensure_space_ready(self) -> None:
        await self._create_space_if_needed()
        await self._wait_for_space_ready()
        await self._create_schema_if_needed()
        await self._wait_for_schema_ready()
        await self._create_indexes_if_needed()
        await self._wait_for_index_ready()

    async def _create_space_if_needed(self) -> None:
        sql = (
            f"CREATE SPACE IF NOT EXISTS `{self._space_name}` "
            f"(partition_num={self._partition_num}, "
            f"replica_factor={self._replica_factor}, vid_type=FIXED_STRING(256));"
        )
        await self._execute(sql)

    async def _use_space(self, session: Any) -> None:
        await self._execute(f"USE `{self._space_name}`;", session=session)

    async def _wait_for_space_ready(self) -> None:
        for attempt in range(1, self._schema_retry_times + 1):
            try:
                session = await self._acquire_session()
                try:
                    await self._use_space(session)
                    return
                finally:
                    await self._release_session(session)
            except RuntimeError:
                if attempt == self._schema_retry_times:
                    raise TimeoutError(
                        "Nebula space did not become ready in the expected time."
                    ) from None
            if attempt == self._schema_retry_times:
                raise TimeoutError(
                    "Nebula space did not become ready in the expected time."
                )
            await asyncio.sleep(self._schema_retry_delay_ms / 1000)

    async def _create_schema_if_needed(self) -> None:
        await self._execute_in_space(
            "CREATE TAG IF NOT EXISTS entity("
            "entity_id string, "
            "name string, "
            "entity_type string, "
            "description string, "
            "keywords string, "
            "source_id string"
            ");"
        )
        await self._execute_in_space(
            "CREATE EDGE IF NOT EXISTS relation("
            "source_id string, "
            "target_id string, "
            "relationship string, "
            "description string, "
            "weight double"
            ");"
        )

    async def _create_indexes_if_needed(self) -> None:
        await self._execute_in_space(
            "CREATE TAG INDEX IF NOT EXISTS entity_entity_id_idx ON entity(entity_id(256));"
        )
        await self._execute_in_space(
            "CREATE EDGE INDEX IF NOT EXISTS relation_pair_idx "
            "ON relation(source_id(256), target_id(256));"
        )
        for stmt in (
            "REBUILD TAG INDEX entity_entity_id_idx;",
            "REBUILD EDGE INDEX relation_pair_idx;",
        ):
            try:
                await self._execute_in_space(stmt)
            except RuntimeError as exc:
                if "index" in str(exc).lower() and "not found" in str(exc).lower():
                    logger.warning(
                        f"[{self.workspace}] Nebula rebuild skipped for missing index metadata: {exc}"
                    )
                    continue
                raise
        self._fulltext_init_error = None
        try:
            await self._ensure_fulltext_ready()
            await self._create_fulltext_index(
                f"CREATE FULLTEXT TAG INDEX IF NOT EXISTS {self._fulltext_tag_index_name} "
                "ON entity(name);",
                f"CREATE FULLTEXT TAG INDEX {self._fulltext_tag_index_name} ON entity(name);",
            )
            await self._create_fulltext_index(
                f"CREATE FULLTEXT EDGE INDEX IF NOT EXISTS {self._fulltext_edge_index_name} "
                "ON relation(relationship);",
                f"CREATE FULLTEXT EDGE INDEX {self._fulltext_edge_index_name} ON relation(relationship);",
            )
            try:
                await self._execute_in_space("REBUILD FULLTEXT INDEX;")
            except RuntimeError as exc:
                logger.warning(
                    f"[{self.workspace}] Nebula full-text rebuild skipped: {exc}"
                )
            await self._wait_for_fulltext_query_ready(self._fulltext_tag_index_name)
        except RuntimeError as exc:
            self._fulltext_init_error = str(exc)
            return

    async def _ensure_fulltext_ready(self) -> None:
        text_clients = await self._execute("SHOW TEXT SEARCH CLIENTS;")
        client_rows = _result_to_rows(text_clients)
        if not client_rows:
            raise RuntimeError(
                "Nebula text search client is not configured. Run SIGN IN TEXT SERVICE first."
            )

        listener_rows_result = await self._execute_in_space("SHOW LISTENER;")
        listener_rows = _result_to_rows(listener_rows_result)
        if not listener_rows:
            listener_hosts = (
                self._listener_hosts or await self._discover_listener_hosts()
            )
            if not listener_hosts:
                raise RuntimeError(
                    "Nebula listener is not configured for this space. Set NEBULA_LISTENER_HOSTS or configure listener manually."
                )
            await self._execute_in_space(
                "ADD LISTENER ELASTICSEARCH " + ",".join(listener_hosts) + ";"
            )
            listener_rows = _result_to_rows(
                await self._execute_in_space("SHOW LISTENER;")
            )

        for attempt in range(1, self._schema_retry_times + 1):
            listener_rows = _result_to_rows(
                await self._execute_in_space("SHOW LISTENER;")
            )
            if listener_rows and all(
                str(
                    row.get("Host Status", row.get("host status", row.get("col_3", "")))
                ).upper()
                == "ONLINE"
                for row in listener_rows
            ):
                return
            if attempt == self._schema_retry_times:
                raise RuntimeError(
                    "Nebula listener did not become ONLINE for the current space."
                )
            await asyncio.sleep(self._schema_retry_delay_ms / 1000)

    async def _discover_listener_hosts(self) -> list[str]:
        spaces_result = await self._execute("SHOW SPACES;")
        spaces_rows = _result_to_rows(spaces_result)
        space_names = [
            str(next(iter(row.values())))
            for row in spaces_rows
            if row and next(iter(row.values()), None) is not None
        ]
        for space in space_names:
            if space == self._space_name:
                continue
            session = await self._acquire_session()
            try:
                await self._execute(f"USE `{space}`;", session=session)
                result = await self._execute("SHOW LISTENER;", session=session)
            except Exception:
                await self._release_session(session)
                continue
            rows = _result_to_rows(result)
            await self._release_session(session)
            endpoints = [
                _normalize_listener_endpoint(row.get("col_2"))
                for row in rows
                if row.get("col_2")
            ]
            endpoints = [endpoint for endpoint in endpoints if endpoint]
            if endpoints:
                return self._unique_preserve_order(endpoints)
        return []

    async def _create_fulltext_index(
        self, stmt_if_not_exists: str, stmt_plain: str
    ) -> None:
        try:
            await self._execute_in_space(stmt_if_not_exists)
            return
        except RuntimeError as exc:
            if "syntax error" not in str(exc).lower() or "if" not in str(exc).lower():
                raise
        await self._execute_in_space(stmt_plain)

    async def _wait_for_fulltext_query_ready(self, index_name: str) -> None:
        probe = (
            "LOOKUP ON entity "
            f'WHERE ES_QUERY({index_name}, "a*") '
            "YIELD entity.entity_id AS entity_id "
            "| LIMIT 1;"
        )
        for attempt in range(1, self._schema_retry_times + 1):
            try:
                await self._execute_in_space(probe)
                return
            except RuntimeError:
                if attempt == self._schema_retry_times:
                    raise RuntimeError(
                        f"Nebula full-text index {index_name} did not become query-ready in time."
                    ) from None
                await asyncio.sleep(self._schema_retry_delay_ms / 1000)

    async def _wait_for_schema_ready(self) -> None:
        for attempt in range(1, self._schema_retry_times + 1):
            try:
                await self._execute_in_space(
                    "MATCH (v:entity) RETURN count(v) AS vertex_count LIMIT 1;"
                )
                await self._execute_in_space(
                    "MATCH ()-[e:relation]->() RETURN count(e) AS edge_count LIMIT 1;"
                )
                return
            except RuntimeError:
                if attempt == self._schema_retry_times:
                    raise TimeoutError(
                        "Nebula schema did not become ready in the expected time."
                    ) from None
                await asyncio.sleep(self._schema_retry_delay_ms / 1000)

    async def _wait_for_index_ready(self) -> None:
        for attempt in range(1, self._schema_retry_times + 1):
            try:
                tag_status = await self._execute_in_space("SHOW TAG INDEX STATUS;")
                edge_status = await self._execute_in_space("SHOW EDGE INDEX STATUS;")
                if self._is_index_status_ready(tag_status) and self._is_index_status_ready(
                    edge_status
                ):
                    return
            except NebulaIndexJobError:
                raise
            except RuntimeError:
                if attempt == self._schema_retry_times:
                    raise TimeoutError(
                        "Nebula indexes did not become ready in the expected time."
                    ) from None
                await asyncio.sleep(self._schema_retry_delay_ms / 1000)
                continue
            if attempt == self._schema_retry_times:
                raise TimeoutError(
                    "Nebula indexes did not become ready in the expected time."
                )
            await asyncio.sleep(self._schema_retry_delay_ms / 1000)

    def _is_index_status_ready(self, result: Any) -> bool:
        status_text = _flatten_result_text(result).upper()
        if not status_text.strip():
            return True
        if any(token in status_text for token in _INDEX_STATUS_FAILED_TOKENS):
            raise NebulaIndexJobError(f"Nebula index job failed: {status_text}")
        if any(token in status_text for token in _INDEX_STATUS_PENDING_TOKENS):
            return False
        return any(token in status_text for token in _INDEX_STATUS_DONE_TOKENS)

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        try:
            if self._connection_pool is None:
                self._validate_required_env()
                await self._bootstrap_client()

            await self._execute(f"DROP SPACE IF EXISTS `{self._space_name}`;")
            return {
                "status": "success",
                "message": f"workspace '{self._space_name}' dropped",
            }
        except Exception as exc:
            logger.error(
                f"[{self.workspace}] Error dropping Nebula space '{self._space_name}': {exc}"
            )
            return {"status": "error", "message": str(exc)}
        finally:
            self._fulltext_init_error = None
            self._initialized = False
            await self._close_connection_pool()

    async def has_node(self, node_id: str) -> bool:
        return await self.get_node(node_id) is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return await self.get_edge(source_node_id, target_node_id) is not None

    @staticmethod
    def _unique_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return unique

    @staticmethod
    def _build_or_equals_clause(field: str, values: list[str]) -> str:
        return " OR ".join(f"{field} == {_ngql_literal(value)}" for value in values)

    @staticmethod
    def _build_relation_endpoint_clause(node_ids: list[str]) -> str:
        conditions: list[str] = []
        for node_id in node_ids:
            literal = _ngql_literal(node_id)
            conditions.append(f"(src(edge) == {literal} OR dst(edge) == {literal})")
        return " OR ".join(conditions)

    @staticmethod
    def _extract_node_props(
        row: dict[str, Any], *, fallback_entity_id: str | None = None
    ) -> dict[str, str]:
        output: dict[str, str] = {}
        for field in _NODE_FIELDS:
            value = row.get(field)
            if value is None:
                continue
            output[field] = str(value)
        if "entity_id" not in output and fallback_entity_id is not None:
            output["entity_id"] = fallback_entity_id
        return output

    @staticmethod
    def _extract_edge_props(row: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for field in _EDGE_FIELDS:
            value = row.get(field)
            if value is None:
                continue
            output[field] = value
        return output

    @staticmethod
    def _resolve_max_nodes(global_config: dict[str, Any], max_nodes: int | None) -> int:
        config_limit = int(global_config.get("max_graph_nodes", 1000))
        if max_nodes is None:
            return max(0, config_limit)
        return max(0, min(int(max_nodes), config_limit))

    @staticmethod
    def _extract_neighbor(node_id: str, edge: tuple[str, str]) -> str | None:
        src, tgt = edge
        if src == node_id and tgt:
            return tgt
        if tgt == node_id and src:
            return src
        if src and src != node_id:
            return src
        if tgt and tgt != node_id:
            return tgt
        return None

    @staticmethod
    def _to_knowledge_graph_node(node_id: str, node_data: dict[str, Any]) -> KnowledgeGraphNode:
        props = dict(node_data)
        entity_id = str(props.get("entity_id", node_id))
        props.setdefault("entity_id", entity_id)
        return KnowledgeGraphNode(
            id=entity_id,
            labels=[entity_id],
            properties=props,
        )

    @staticmethod
    def _to_knowledge_graph_edge(
        source: str, target: str, edge_data: dict[str, Any] | None
    ) -> KnowledgeGraphEdge:
        props = dict(edge_data or {})
        edge_type = props.get("relationship")
        props.pop("source", None)
        props.pop("target", None)
        edge_id = f"{source}->{target}"
        return KnowledgeGraphEdge(
            id=edge_id,
            type=str(edge_type) if edge_type is not None else None,
            source=source,
            target=target,
            properties=props,
        )

    @staticmethod
    def _label_match_tier(label: str, query_lower: str) -> int:
        label_lower = label.lower()
        if label_lower == query_lower:
            return 0
        if label_lower.startswith(query_lower):
            return 1
        if query_lower in label_lower:
            return 2
        return 3

    @classmethod
    def _rank_labels(
        cls, labels: list[str], query: str, limit: int
    ) -> list[str]:
        if limit <= 0:
            return []
        query_lower = query.lower()
        deduplicated: dict[str, tuple[int, str, str]] = {}
        for raw_label in labels:
            label = str(raw_label)
            if query_lower not in label.lower():
                continue
            if label in deduplicated:
                continue
            deduplicated[label] = (
                cls._label_match_tier(label, query_lower),
                label.lower(),
                label,
            )
        ranked = sorted(
            deduplicated.items(),
            key=lambda item: (item[1][0], item[1][1], item[1][2]),
        )
        return [label for label, _ in ranked[:limit]]

    @staticmethod
    def _dedupe_labels_preserve_order(labels: list[str], limit: int) -> list[str]:
        if limit <= 0:
            return []
        seen: set[str] = set()
        output: list[str] = []
        for raw_label in labels:
            label = str(raw_label)
            if label in seen:
                continue
            seen.add(label)
            output.append(label)
            if len(output) >= limit:
                break
        return output

    @classmethod
    def _rank_fulltext_labels_stable(
        cls, labels: list[str], query: str, limit: int
    ) -> list[str]:
        if limit <= 0:
            return []
        query_lower = query.lower()
        deduped = cls._dedupe_labels_preserve_order(labels, len(labels))
        ranked = sorted(
            enumerate(deduped),
            key=lambda item: (cls._label_match_tier(item[1], query_lower), item[0]),
        )
        return [label for _, label in ranked[:limit]]

    async def _build_global_knowledge_graph(self, max_nodes: int) -> KnowledgeGraph:
        result = KnowledgeGraph()
        if max_nodes <= 0:
            all_nodes = await self.get_all_nodes()
            result.is_truncated = bool(all_nodes)
            return result

        all_nodes = await self.get_all_nodes()
        all_edges = await self.get_all_edges()

        node_map: dict[str, dict[str, Any]] = {}
        for node in all_nodes:
            entity_id = node.get("entity_id") or node.get("id")
            if entity_id is None:
                continue
            node_map[str(entity_id)] = dict(node)

        degree_map = {node_id: 0 for node_id in node_map}
        for edge in all_edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src is not None and str(src) in degree_map:
                degree_map[str(src)] += 1
            if tgt is not None and str(tgt) in degree_map:
                degree_map[str(tgt)] += 1

        sorted_nodes = sorted(
            node_map.keys(), key=lambda node_id: (-degree_map[node_id], node_id)
        )
        selected_ids = sorted_nodes[:max_nodes]
        selected_set = set(selected_ids)
        result.is_truncated = len(sorted_nodes) > len(selected_ids)

        for node_id in selected_ids:
            result.nodes.append(
                self._to_knowledge_graph_node(node_id, node_map[node_id])
            )

        seen_edges: set[tuple[str, str, str]] = set()
        for edge in all_edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src is None or tgt is None:
                continue
            source = str(src)
            target = str(tgt)
            if source not in selected_set or target not in selected_set:
                continue
            relation = str(edge.get("relationship", ""))
            edge_key = (source, target, relation)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            result.edges.append(self._to_knowledge_graph_edge(source, target, edge))

        return result

    async def _build_bounded_subgraph(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        result = KnowledgeGraph()
        if max_nodes <= 0:
            if await self.get_node(node_label) is not None:
                result.is_truncated = True
            return result

        start_nodes = await self.get_nodes_batch([node_label])
        if node_label not in start_nodes:
            return result

        visited: set[str] = set()
        bfs_order: list[str] = []
        frontier: deque[tuple[str, int]] = deque([(node_label, 0)])
        hit_node_limit = False
        hit_depth_limit = False

        while frontier and len(bfs_order) < max_nodes:
            current_layer: list[str] = []
            current_depth = frontier[0][1]
            while frontier and frontier[0][1] == current_depth:
                node_id, _depth = frontier.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)
                current_layer.append(node_id)
                bfs_order.append(node_id)
                if len(bfs_order) >= max_nodes:
                    break

            if len(bfs_order) >= max_nodes:
                hit_node_limit = bool(frontier)
                if not hit_node_limit:
                    adjacency = await self.get_nodes_edges_batch(current_layer)
                    for node_id in current_layer:
                        for edge in adjacency.get(node_id, []):
                            neighbor = self._extract_neighbor(node_id, edge)
                            if neighbor and neighbor not in visited:
                                hit_node_limit = True
                                break
                        if hit_node_limit:
                            break
                break

            if current_depth >= max_depth:
                adjacency = await self.get_nodes_edges_batch(current_layer)
                for node_id in current_layer:
                    for edge in adjacency.get(node_id, []):
                        neighbor = self._extract_neighbor(node_id, edge)
                        if neighbor and neighbor not in visited:
                            hit_depth_limit = True
                            break
                    if hit_depth_limit:
                        break
                continue

            adjacency = await self.get_nodes_edges_batch(current_layer)
            next_candidates: list[str] = []
            seen_next: set[str] = set()
            for node_id in current_layer:
                for edge in adjacency.get(node_id, []):
                    neighbor = self._extract_neighbor(node_id, edge)
                    if neighbor is None or neighbor in visited or neighbor in seen_next:
                        continue
                    seen_next.add(neighbor)
                    next_candidates.append(neighbor)
            next_candidates.sort()
            for neighbor in next_candidates:
                frontier.append((neighbor, current_depth + 1))

        selected_ids = bfs_order[:max_nodes]
        selected_set = set(selected_ids)
        result.is_truncated = hit_node_limit or hit_depth_limit or bool(frontier)
        if not selected_ids:
            return result

        node_payloads = await self.get_nodes_batch(selected_ids)
        for node_id in selected_ids:
            node_data = node_payloads.get(node_id)
            if node_data is None:
                node_data = {"entity_id": node_id}
            result.nodes.append(self._to_knowledge_graph_node(node_id, node_data))

        adjacency = await self.get_nodes_edges_batch(selected_ids)
        edge_pairs: list[dict[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for node_id in selected_ids:
            for edge in adjacency.get(node_id, []):
                src, tgt = edge
                source = str(src)
                target = str(tgt)
                if source not in selected_set or target not in selected_set:
                    continue
                canonical = _canonical_edge_pair(source, target)
                if canonical in seen_pairs:
                    continue
                seen_pairs.add(canonical)
                edge_pairs.append({"src": canonical[0], "tgt": canonical[1]})

        edge_payloads = (
            await self.get_edges_batch(edge_pairs)
            if edge_pairs
            else {}
        )
        for pair in edge_pairs:
            source = pair["src"]
            target = pair["tgt"]
            payload = edge_payloads.get((source, target), {"source": source, "target": target})
            payload.setdefault("source", source)
            payload.setdefault("target", target)
            result.edges.append(self._to_knowledge_graph_edge(source, target, payload))
        return result

    async def node_degree(self, node_id: str) -> int:
        degrees = await self.node_degrees_batch([node_id])
        return int(degrees.get(node_id, 0))

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        degrees = await self.node_degrees_batch([src_id, tgt_id])
        return int(degrees.get(src_id, 0)) + int(degrees.get(tgt_id, 0))

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        vid = _ngql_quote(node_id)
        result = await self._execute_in_space(
            "FETCH PROP ON entity "
            f"{vid} "
            "YIELD "
            "properties(vertex).entity_id AS entity_id, "
            "properties(vertex).name AS name, "
            "properties(vertex).entity_type AS entity_type, "
            "properties(vertex).description AS description, "
            "properties(vertex).keywords AS keywords, "
            "properties(vertex).source_id AS source_id;"
        )
        row = _first_row(result)
        if row is None:
            return None
        return self._extract_node_props(row, fallback_entity_id=node_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        src_id, tgt_id = _canonical_edge_pair(source_node_id, target_node_id)
        result = await self._execute_in_space(
            "MATCH (a:entity)-[e:relation]->(b:entity) "
            f"WHERE id(a) == {_ngql_literal(src_id)} AND id(b) == {_ngql_literal(tgt_id)} "
            "RETURN "
            "id(a) AS source, "
            "id(b) AS target, "
            "e.source_id AS source_id, "
            "e.target_id AS target_id, "
            "e.relationship AS relationship, "
            "e.description AS description, "
            "e.weight AS weight "
            "LIMIT 1;"
        )
        row = _first_row(result)
        if row is None:
            return None
        output = self._extract_edge_props(row)
        source = row.get("source")
        target = row.get("target")
        if source is not None:
            output["source"] = str(source)
        if target is not None:
            output["target"] = str(target)
        return output

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        requested_ids = [str(node_id) for node_id in node_ids]
        unique_ids = self._unique_preserve_order(
            [node_id for node_id in requested_ids if node_id]
        )
        if not unique_ids:
            return {}

        result = await self._execute_in_space(
            "MATCH (v:entity) "
            "RETURN "
            "id(v) AS entity_id, "
            "v.name AS name, "
            "v.entity_type AS entity_type, "
            "v.description AS description, "
            "v.keywords AS keywords, "
            "v.source_id AS source_id;"
        )
        rows = _result_to_rows(result)

        found_by_entity_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            entity_id = row.get("entity_id")
            if entity_id is None:
                continue
            found_by_entity_id[str(entity_id)] = row

        output: dict[str, dict] = {}
        for node_id in requested_ids:
            row = found_by_entity_id.get(node_id)
            if row is None:
                continue
            output[node_id] = self._extract_node_props(row, fallback_entity_id=node_id)
        return output

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        requested_ids = [str(node_id) for node_id in node_ids]
        output = {node_id: 0 for node_id in requested_ids}
        unique_ids = self._unique_preserve_order(
            [node_id for node_id in requested_ids if node_id]
        )
        if not unique_ids:
            return output

        result = await self._execute_in_space(
            "MATCH (a:entity)-[e:relation]->(b:entity) "
            "RETURN "
            "id(a) AS source, "
            "id(b) AS target;"
        )
        rows = _result_to_rows(result)
        requested_set = set(output)
        for row in rows:
            src = row.get("source")
            tgt = row.get("target")
            if src is not None:
                src_id = str(src)
                if src_id in requested_set:
                    output[src_id] += 1
            if tgt is not None:
                tgt_id = str(tgt)
                if tgt_id in requested_set:
                    output[tgt_id] += 1
        return output

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        requested_pairs: list[tuple[str, str]] = []
        canonical_to_requested: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for pair in pairs:
            src = str(pair.get("src", ""))
            tgt = str(pair.get("tgt", ""))
            if not src or not tgt:
                continue
            request_key = (src, tgt)
            requested_pairs.append(request_key)
            canonical = _canonical_edge_pair(src, tgt)
            canonical_to_requested.setdefault(canonical, []).append(request_key)

        if not requested_pairs:
            return {}

        result = await self._execute_in_space(
            "MATCH (a:entity)-[e:relation]->(b:entity) "
            "RETURN "
            "id(a) AS source, "
            "id(b) AS target, "
            "e.source_id AS source_id, "
            "e.target_id AS target_id, "
            "e.relationship AS relationship, "
            "e.description AS description, "
            "e.weight AS weight;"
        )
        rows = _result_to_rows(result)

        by_canonical: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            src = row.get("source")
            tgt = row.get("target")
            if src is None or tgt is None:
                continue
            canonical = _canonical_edge_pair(str(src), str(tgt))
            props = self._extract_edge_props(row)
            props["source"] = str(src)
            props["target"] = str(tgt)
            by_canonical[canonical] = props

        output: dict[tuple[str, str], dict] = {}
        for canonical, request_keys in canonical_to_requested.items():
            props = by_canonical.get(canonical)
            if props is None:
                continue
            for request_key in request_keys:
                output[request_key] = dict(props)
        return output

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        node = await self.get_node(source_node_id)
        if node is None:
            return None
        batched = await self.get_nodes_edges_batch([source_node_id])
        return batched.get(source_node_id, [])

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        requested_ids = [str(node_id) for node_id in node_ids]
        output = {node_id: [] for node_id in requested_ids}
        unique_ids = self._unique_preserve_order(
            [node_id for node_id in requested_ids if node_id]
        )
        if not unique_ids:
            return output

        result = await self._execute_in_space(
            "MATCH (a:entity)-[e:relation]->(b:entity) "
            "RETURN "
            "id(a) AS source, "
            "id(b) AS target;"
        )
        rows = _result_to_rows(result)
        requested_set = set(output)
        for row in rows:
            src = row.get("source")
            tgt = row.get("target")
            if src is None or tgt is None:
                continue
            src_id = str(src)
            tgt_id = str(tgt)
            edge = (src_id, tgt_id)
            if src_id in requested_set:
                output[src_id].append(edge)
            if tgt_id in requested_set and tgt_id != src_id:
                output[tgt_id].append(edge)
        return output

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        entity_id = str(node_id)
        name = str(node_data.get("name", entity_id))
        entity_type = str(node_data.get("entity_type", ""))
        description = str(node_data.get("description", ""))
        keywords = str(node_data.get("keywords", ""))
        source_id = str(node_data.get("source_id", ""))

        await self._execute_in_space(
            "INSERT VERTEX entity(entity_id, name, entity_type, description, keywords, source_id) "
            f"VALUES {_ngql_quote(entity_id)}:"
            "("
            f"{_ngql_literal(entity_id)}, "
            f"{_ngql_literal(name)}, "
            f"{_ngql_literal(entity_type)}, "
            f"{_ngql_literal(description)}, "
            f"{_ngql_literal(keywords)}, "
            f"{_ngql_literal(source_id)}"
            ");"
        )

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        src_id, tgt_id = _canonical_edge_pair(source_node_id, target_node_id)
        source_id = str(edge_data.get("source_id", ""))
        target_id = str(edge_data.get("target_id", ""))
        relationship = str(edge_data.get("relationship", ""))
        description = str(edge_data.get("description", ""))
        weight = _coerce_edge_weight(edge_data.get("weight"))

        await self._execute_in_space(
            "INSERT EDGE relation(source_id, target_id, relationship, description, weight) "
            f"VALUES {_ngql_quote(src_id)}->{_ngql_quote(tgt_id)}:"
            "("
            f"{_ngql_literal(source_id)}, "
            f"{_ngql_literal(target_id)}, "
            f"{_ngql_literal(relationship)}, "
            f"{_ngql_literal(description)}, "
            f"{_ngql_literal(weight)}"
            ");"
        )

    async def delete_node(self, node_id: str) -> None:
        await self._execute_in_space(f"DELETE VERTEX {_ngql_quote(node_id)} WITH EDGE;")

    async def remove_nodes(self, nodes: list[str]):
        unique_nodes = self._unique_preserve_order(
            [str(node_id) for node_id in nodes if str(node_id)]
        )
        for node_id in unique_nodes:
            await self._execute_in_space(
                f"DELETE VERTEX {_ngql_quote(node_id)} WITH EDGE;"
            )

    async def remove_edges(self, edges: list[tuple[str, str]]):
        unique_pairs: set[tuple[str, str]] = set()
        for src, tgt in edges:
            unique_pairs.add(_canonical_edge_pair(src, tgt))
        for src_id, tgt_id in unique_pairs:
            await self._execute_in_space(
                f"DELETE EDGE relation {_ngql_quote(src_id)}->{_ngql_quote(tgt_id)};"
            )

    async def get_all_labels(self) -> list[str]:
        result = await self._execute_in_space(
            "MATCH (v:entity) "
            "RETURN id(v) AS entity_id;"
        )
        rows = _result_to_rows(result)
        labels = {str(row["entity_id"]) for row in rows if row.get("entity_id") is not None}
        return sorted(labels)

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        resolved_max_nodes = self._resolve_max_nodes(self.global_config, max_nodes)
        resolved_max_depth = max(0, int(max_depth))
        label = str(node_label).strip()
        if not label:
            return KnowledgeGraph()
        if label == "*":
            return await self._build_global_knowledge_graph(resolved_max_nodes)
        return await self._build_bounded_subgraph(
            label, resolved_max_depth, resolved_max_nodes
        )

    async def get_all_nodes(self) -> list[dict]:
        result = await self._execute_in_space(
            "MATCH (v:entity) "
            "RETURN "
            "id(v) AS entity_id, "
            "v.name AS name, "
            "v.entity_type AS entity_type, "
            "v.description AS description, "
            "v.keywords AS keywords, "
            "v.source_id AS source_id;"
        )
        rows = _result_to_rows(result)
        output: list[dict] = []
        for row in rows:
            node = self._extract_node_props(row)
            entity_id = node.get("entity_id")
            if entity_id:
                node["id"] = entity_id
            if node:
                output.append(node)
        output.sort(key=lambda item: str(item.get("entity_id", "")))
        return output

    async def get_all_edges(self) -> list[dict]:
        result = await self._execute_in_space(
            "MATCH (a:entity)-[e:relation]->(b:entity) "
            "RETURN "
            "id(a) AS source, "
            "id(b) AS target, "
            "e.source_id AS source_id, "
            "e.target_id AS target_id, "
            "e.relationship AS relationship, "
            "e.description AS description, "
            "e.weight AS weight;"
        )
        rows = _result_to_rows(result)
        output: list[dict] = []
        for row in rows:
            edge = self._extract_edge_props(row)
            source = row.get("source")
            target = row.get("target")
            if source is None or target is None:
                continue
            edge["source"] = str(source)
            edge["target"] = str(target)
            output.append(edge)
        output.sort(
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("target", "")),
                str(item.get("relationship", "")),
            )
        )
        return output

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        if limit <= 0:
            return []
        result = await self._execute_in_space(
            "MATCH (a:entity)-[e:relation]->(b:entity) "
            "RETURN "
            "id(a) AS source, "
            "id(b) AS target;"
        )
        rows = _result_to_rows(result)
        degrees: dict[str, int] = {}
        for row in rows:
            source_id = row.get("source")
            target_id = row.get("target")
            if source_id is not None:
                src = str(source_id)
                degrees[src] = degrees.get(src, 0) + 1
            if target_id is not None:
                tgt = str(target_id)
                degrees[tgt] = degrees.get(tgt, 0) + 1
        ranking = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))
        return [label for label, _ in ranking[:limit]]

    async def _search_labels_fulltext(self, query: str, limit: int = 50) -> list[str]:
        if limit <= 0:
            return []
        query_strip = query.strip()
        if not query_strip:
            return []
        escaped_query = _ngql_escape_string(query_strip)
        result = await self._execute_in_space(
            "LOOKUP ON entity "
            f'WHERE ES_QUERY({self._fulltext_tag_index_name}, "{escaped_query}*") '
            "YIELD entity.entity_id AS entity_id "
            f"| LIMIT {int(limit)};"
        )
        rows = _result_to_rows(result)
        labels = [
            str(row["entity_id"])
            for row in rows
            if row.get("entity_id") is not None
        ]
        return self._rank_fulltext_labels_stable(labels, query_strip, limit)

    async def _search_labels_contains(self, query: str, limit: int = 50) -> list[str]:
        if limit <= 0:
            return []
        query_strip = query.strip()
        if not query_strip:
            return []
        labels = await self.get_all_labels()
        return self._rank_labels(labels, query_strip, limit)

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        if limit <= 0:
            return []
        query_strip = query.strip()
        if not query_strip:
            return []

        if self._fulltext_init_error:
            logger.warning(
                f"[{self.workspace}] Nebula full-text unavailable during init: "
                f"{self._fulltext_init_error}; falling back to contains search."
            )
            return await self._search_labels_contains(query_strip, limit=limit)

        try:
            labels = await self._search_labels_fulltext(query_strip, limit=limit)
            if labels:
                return labels
        except Exception as exc:
            logger.warning(
                f"[{self.workspace}] Nebula full-text search failed for "
                f"query '{query_strip}': {exc}; falling back to contains search."
            )
        return await self._search_labels_contains(query_strip, limit=limit)
