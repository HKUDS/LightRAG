from __future__ import annotations

import asyncio
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, final

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph

_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")
_DEFAULT_SPACE_PREFIX = "lightrag"
_DEFAULT_WORKSPACE = "base"
_MAX_SPACE_NAME_LEN = 127
_DEFAULT_NEBULA_PORT = 9669
_DEFAULT_PARTITION_NUM = 10
_DEFAULT_REPLICA_FACTOR = 1
_DEFAULT_SCHEMA_RETRY_TIMES = 30
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
        self._hosts = _parse_nebula_hosts(_env_str("NEBULA_HOSTS"))
        self._user = _env_str("NEBULA_USER")
        self._password = _env_str("NEBULA_PASSWORD")
        self._timeout_ms = _env_int("NEBULA_TIMEOUT_MS", 60_000)
        self._ssl_enabled = _env_bool("NEBULA_SSL", default=False)
        self._use_http2 = _env_bool("NEBULA_USE_HTTP2", default=False)
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
        if self._password is None or not str(self._password).strip():
            raise ValueError(
                "Environment variable NEBULA_PASSWORD is required and must not be empty."
            )

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
                result = await self._execute("SHOW SPACES;")
                if self._space_name.lower() in _flatten_result_text(result).lower():
                    return
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
            "description string, "
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
        await self._execute_in_space("REBUILD TAG INDEX entity_entity_id_idx;")
        await self._execute_in_space("REBUILD EDGE INDEX relation_pair_idx;")
        try:
            await self._execute_in_space(
                "CREATE FULLTEXT TAG INDEX IF NOT EXISTS entity_name_ft_idx "
                "ON entity(name);"
            )
            await self._execute_in_space(
                "CREATE FULLTEXT EDGE INDEX IF NOT EXISTS relation_rel_ft_idx "
                "ON relation(relationship);"
            )
        except RuntimeError as exc:
            # Full-text support depends on deployment wiring outside this backend.
            # Keep startup alive, but retain the original error for diagnostics.
            self._fulltext_init_error = str(exc)
            return

    async def _wait_for_schema_ready(self) -> None:
        for attempt in range(1, self._schema_retry_times + 1):
            try:
                await self._execute_in_space("DESCRIBE TAG entity;")
                await self._execute_in_space("DESCRIBE EDGE relation;")
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
        if any(token in status_text for token in _INDEX_STATUS_FAILED_TOKENS):
            raise NebulaIndexJobError(f"Nebula index job failed: {status_text}")
        if any(token in status_text for token in _INDEX_STATUS_PENDING_TOKENS):
            return False
        return any(token in status_text for token in _INDEX_STATUS_DONE_TOKENS)

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        return {"status": "error", "message": "unsupported"}

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def delete_node(self, node_id: str) -> None:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def remove_nodes(self, nodes: list[str]):
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def remove_edges(self, edges: list[tuple[str, str]]):
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def get_all_labels(self) -> list[str]:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def get_all_nodes(self) -> list[dict]:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def get_all_edges(self) -> list[dict]:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        raise NotImplementedError("Nebula I/O is not implemented in this task")

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        raise NotImplementedError("Nebula I/O is not implemented in this task")
