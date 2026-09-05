"""
NebulaGraph Storage Implementation for LightRAG

Uses ISO GQL (graph query language) syntax for all graph operations.
Connection via the async NebulaGraph Python SDK (nebula5_python).

Require NebulaGraph 5.0+ for ISO GQL support.
"""

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar

import configparser

from ..utils import logger, validate_workspace
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..kg.shared_storage import get_data_init_lock


from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


def _resolve_env_or_config(
    env_key: str, config_section: str, config_key: str, default: Any = None
) -> Any:
    """Resolve a value from environment variable, or config.ini fallback."""
    value = os.environ.get(env_key)
    if value is not None:
        return value
    try:
        return config.get(config_section, config_key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default


@dataclass
class NebulaStorage(BaseGraphStorage):
    """Graph storage backed by NebulaGraph using ISO GQL."""

    # Lucene query-syntax reserved characters for full-text search sanitization.
    _LUCENE_RESERVED: ClassVar[re.Pattern[str]] = re.compile(
        r'[+\-&|!(){}\[\]^"~*?:\\/]'
    )

    # Retry configuration for transient gRPC failures (socket closed, handshake
    # shutdown, etc.).  Overridable via env vars for tuning in deployment.
    _MAX_RETRIES: ClassVar[int] = int(os.environ.get("NEBULA_MAX_RETRIES", "5"))
    _RETRY_BASE_DELAY: ClassVar[float] = float(
        os.environ.get("NEBULA_RETRY_BASE_DELAY", "1.0")
    )
    _RETRY_MAX_DELAY: ClassVar[float] = float(
        os.environ.get("NEBULA_RETRY_MAX_DELAY", "30.0")
    )

    # gRPC status codes that are safe to retry (transient failures).
    _RETRYABLE_GRPC_CODES: ClassVar[set[int]] = {
        14,  # UNAVAILABLE
        4,  # DEADLINE_EXCEEDED
        8,  # RESOURCE_EXHAUSTED
        10,  # ABORTED
        13,  # INTERNAL
    }

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        nebula_workspace = os.environ.get("NEBULA_WORKSPACE")
        original_workspace = workspace
        if nebula_workspace and nebula_workspace.strip():
            workspace = nebula_workspace

        if not workspace or not str(workspace).strip():
            workspace = "base"

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        validate_workspace(self.workspace)

        if nebula_workspace and nebula_workspace.strip():
            logger.info(
                f"Using NEBULA_WORKSPACE env: '{nebula_workspace}' "
                f"(overriding '{original_workspace}')"
            )

        self._client: Any = None  # AsyncNebulaClient
        self._graph: str = ""
        self._entity_label: str = ""
        self._edge_label: str = ""

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------
    def _get_workspace_label(self) -> str:
        """Sanitized workspace identifier for use in element type names."""
        ws = re.sub(r"[^A-Za-z0-9_]+", "_", self.workspace.strip() or "base").strip("_")
        if not ws:
            ws = "base"
        if not re.match(r"[A-Za-z_]", ws[0]):
            ws = f"ws_{ws}"
        return ws

    def _build_label_names(self) -> tuple[str, str]:
        """Return (entity_element_type_name, edge_element_type_name) for this workspace."""
        ws = self._get_workspace_label()
        return f"entity_{ws}", f"directed_{ws}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self):
        async with get_data_init_lock():
            hosts = _resolve_env_or_config(
                "NEBULA_GRAPH_HOSTS", "nebula", "hosts", "127.0.0.1:9669"
            )
            user = _resolve_env_or_config("NEBULA_USER", "nebula", "user", "root")
            password = _resolve_env_or_config(
                "NEBULA_PASSWORD", "nebula", "password", "nebula"
            )
            self._graph = _resolve_env_or_config(
                "NEBULA_GRAPH", "nebula", "space", "lightrag"
            )
            connect_timeout_ms = int(
                _resolve_env_or_config(
                    "NEBULA_CONNECT_TIMEOUT", "nebula", "connect_timeout", "5000"
                )
            )
            request_timeout_ms = int(
                _resolve_env_or_config(
                    "NEBULA_REQUEST_TIMEOUT", "nebula", "request_timeout", "120000"
                )
            )

            self._entity_label, self._edge_label = self._build_label_names()

            from nebulagraph_python.client.nebula_client import AsyncNebulaClient

            self._client = AsyncNebulaClient(
                addresses=hosts,
                user_name=user,
                password=password if password else None,
                connect_timeout_ms=connect_timeout_ms,
                request_timeout_ms=request_timeout_ms,
            )
            await self._client._init_client()

            # Ensure the graph exists with proper element types.
            graph_needs_create = True

            try:
                await self._execute(f"SESSION SET GRAPH `{self._graph}`")
                # Graph exists — verify it has our element types.
                if await self._verify_element_types():
                    graph_needs_create = False
                else:
                    logger.info(
                        f"[{self.workspace}] Graph '{self._graph}' exists "
                        f"but missing element types, dropping and recreating"
                    )
                    await self._execute(f"DROP GRAPH IF EXISTS `{self._graph}`")
                    await self._execute(f"DROP GRAPH TYPE IF EXISTS `{self._graph}`")
            except Exception:
                pass  # Graph doesn't exist, will create below

            if graph_needs_create:
                logger.info(
                    f"[{self.workspace}] Creating graph type and graph '{self._graph}'"
                )
                try:
                    await self._ensure_graph_type()
                    await self._execute(
                        f"CREATE GRAPH IF NOT EXISTS `{self._graph}` TYPED `{self._graph}`"
                    )
                    await self._execute(f"SESSION SET GRAPH `{self._graph}`")
                    # Create full-text index for entity_type for faster text searches
                    await self._create_fulltext_index()
                except Exception as create_err:
                    logger.error(
                        f"[{self.workspace}] Failed to create graph "
                        f"'{self._graph}': {create_err}"
                    )
                    raise

            logger.info(
                f"[{self.workspace}] Connected to NebulaGraph space '{self._graph}' "
                f"at {hosts} (entity={self._entity_label}, edge={self._edge_label})"
            )

    async def _verify_element_types(self) -> bool:
        """Check that the expected element types exist in the current graph.

        Runs a lightweight MATCH query to confirm the entity label is
        recognized. Returns True if the element types are present.
        """
        try:
            await self._execute(f"MATCH (n:`{self._entity_label}`) RETURN n LIMIT 0")
            return True
        except Exception:
            return False

    async def _create_fulltext_index(self):
        """Create a full-text index on the entity_type property for faster text searches.

        Uses ISO GQL syntax to create a full-text index on the entity
        node type. The index accelerates prefix/contains lookups in search_labels.
        Falls back gracefully if the index already exists or if full-text indexes
        are not supported by the NebulaGraph deployment.
        """
        index_name = f"idx_ft_{self._entity_label}"
        try:
            await self._execute(
                f"USE `{self._graph}` CREATE FULLTEXT INDEX IF NOT EXISTS "
                f"`{index_name}` ON Node `{self._entity_label}` (entity_type)"
            )
            logger.info(
                f"[{self.workspace}] Created full-text index '{index_name}' "
                f"on `{self._entity_label}`(entity_type)"
            )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Could not create full-text index "
                f"'{index_name}': {e}. "
                "Search functionality will use slower non-indexed queries."
            )

    async def _ensure_graph_type(self):
        """Create the graph type with embedded element type definitions.

        ISO GQL requires a GRAPH TYPE to define the schema (node/edge element
        types) before a GRAPH instance can be created from it.
        """
        ddl = f"""
        CREATE GRAPH TYPE IF NOT EXISTS `{self._graph}` AS {{
            NODE `{self._entity_label}` (LABEL `{self._entity_label}` {{
                `entity_id` STRING NOT NULL,
                `entity_type` STRING NULL,
                `description` STRING NULL,
                `source_id` STRING NULL,
                `file_path` STRING NULL,
                `created_at` INT NULL,
                `truncate` STRING NULL,
                PRIMARY KEY (`entity_id`)
            }}),
            EDGE `{self._edge_label}` (`{self._entity_label}`) -[:`{self._edge_label}` {{
                `weight` DOUBLE NULL,
                `description` STRING NULL,
                `keywords` STRING NULL,
                `source_id` STRING NULL,
                `file_path` STRING NULL,
                `created_at` INT NULL,
                `truncate` STRING NULL
            }}]-> (`{self._entity_label}`)
        }}
        """
        try:
            await self._execute(ddl)
            logger.info(f"[{self.workspace}] Created graph type '{self._graph}'")
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Failed to create graph type '{self._graph}': {e}"
            )
            raise

    async def finalize(self):
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.debug(f"Error closing Nebula client: {e}")
            self._client = None

    async def index_done_callback(self) -> None:
        """NebulaGraph persists writes immediately; no-op."""
        return None

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    # ------------------------------------------------------------------
    # GQL execution helpers
    # ------------------------------------------------------------------
    async def _execute(self, gql: str, timeout_ms: int | None = None) -> Any:
        """Execute a GQL statement and return the ResultSet (error-checked).

        Automatically retries on transient gRPC failures (UNAVAILABLE,
        DEADLINE_EXCEEDED, etc.) with exponential backoff and client
        re-initialization between attempts.
        """
        last_exception: Exception | None = None

        for attempt in range(self._MAX_RETRIES + 1):
            if self._client is None:
                if attempt >= self._MAX_RETRIES:
                    raise RuntimeError(
                        "NebulaStorage client is None after reconnect attempts"
                    )
                delay = min(
                    self._RETRY_BASE_DELAY * (2**attempt),
                    self._RETRY_MAX_DELAY,
                )
                logger.warning(
                    f"[{self.workspace}] NebulaGraph client is None, "
                    f"attempting reconnect (attempt "
                    f"{attempt + 1}/{self._MAX_RETRIES}), "
                    f"waiting {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                try:
                    await self._reconnect()
                except Exception as reconnect_err:
                    last_exception = reconnect_err
                    logger.warning(
                        f"[{self.workspace}] Reconnect failed: {reconnect_err}"
                    )
                continue

            try:
                if timeout_ms is None:
                    result = await self._client.execute(gql)
                else:
                    result = await self._client.execute_with_timeout(gql, timeout_ms)
                result.raise_on_error()
                return result
            except Exception as e:
                last_exception = e

                # Determine if this error is retryable.
                retryable = self._is_retryable_error(e)
                if not retryable or attempt >= self._MAX_RETRIES:
                    raise

                delay = min(
                    self._RETRY_BASE_DELAY * (2**attempt),
                    self._RETRY_MAX_DELAY,
                )
                logger.warning(
                    f"[{self.workspace}] NebulaGraph transient error "
                    f"(attempt {attempt + 1}/{self._MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s: {e}"
                )

                await asyncio.sleep(delay)

                # Re-initialize the client to recover from broken connections.
                try:
                    await self._reconnect()
                except Exception as reconnect_err:
                    logger.warning(
                        f"[{self.workspace}] Reconnect failed: {reconnect_err}"
                    )
                    # Continue to next retry attempt even if reconnect fails;
                    # the client may still have a viable connection.

        # Should not reach here, but guard against it.
        if last_exception:
            raise last_exception
        raise RuntimeError("NebulaGraph _execute failed after retries")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check whether a gRPC/network error is safe to retry."""
        error_str = str(error).lower()
        # gRPC AioRpcError / ExecutingError with retryable status codes.
        if hasattr(error, "code") and callable(getattr(error, "code", None)):
            try:
                code = error.code()  # type: ignore[union-attr]
                if code is not None and int(code) in self._RETRYABLE_GRPC_CODES:
                    return True
            except Exception:
                pass
        # String-based heuristics for nebula5_python error wrappers.
        retryable_keywords = (
            "socket closed",
            "handshaker shutdown",
            "unavailable",
            "failed to connect to all addresses",
            "transport closed",
            "connection refused",
            "connection reset",
            "connection not established",
            "temporarily unavailable",
            "deadline exceeded",
            "connection timeout",
        )
        if any(kw in error_str for kw in retryable_keywords):
            return True
        return False

    async def _reconnect(self) -> None:
        """Re-initialize the gRPC client to recover from broken connections.

        Creates the new client first, then swaps it in atomically so that
        a failed reconnect never leaves self._client in a None/broken state.
        """
        from nebulagraph_python.client.nebula_client import AsyncNebulaClient

        hosts = _resolve_env_or_config(
            "NEBULA_GRAPH_HOSTS", "nebula", "hosts", "127.0.0.1:9669"
        )
        user = _resolve_env_or_config("NEBULA_USER", "nebula", "user", "root")
        password = _resolve_env_or_config(
            "NEBULA_PASSWORD", "nebula", "password", "nebula"
        )
        connect_timeout_ms = int(
            _resolve_env_or_config(
                "NEBULA_CONNECT_TIMEOUT", "nebula", "connect_timeout", "5000"
            )
        )
        request_timeout_ms = int(
            _resolve_env_or_config(
                "NEBULA_REQUEST_TIMEOUT", "nebula", "request_timeout", "120000"
            )
        )

        # Build the new client first.
        new_client = AsyncNebulaClient(
            addresses=hosts,
            user_name=user,
            password=password if password else None,
            connect_timeout_ms=connect_timeout_ms,
            request_timeout_ms=request_timeout_ms,
        )
        await new_client._init_client()

        # Verify the new connection works before swapping.
        result = await new_client.execute(f"SESSION SET GRAPH `{self._graph}`")
        result.raise_on_error()

        # Swap: close the old client, install the new one.
        old_client = self._client
        self._client = new_client

        if old_client is not None:
            try:
                await old_client.close()
            except Exception:
                pass

    async def _query(self, gql: str) -> list[dict[str, Any]]:
        """Execute a GQL query and return rows as a list of dicts."""
        result = await self._execute(gql)
        rows = list(result.as_primitive_by_row())
        return rows

    def _escape_iso_gql_string(self, value: str) -> str:
        """Escape a string for safe use inside ISO GQL single-quoted literals."""
        if not isinstance(value, str):
            return str(value)
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _format_iso_props(self, props: dict[str, Any]) -> str:
        """Format a dict as ISO GQL property map string: '{key1: value1, key2: value2}'."""
        parts = []
        for k, v in props.items():
            if v is None:
                parts.append(f"{k}: NULL")
            elif isinstance(v, bool):
                # bool MUST come before int/float — bool is a subclass of int
                parts.append(f"{k}: {str(v).lower()}")
            elif isinstance(v, (int, float)):
                parts.append(f"{k}: {v}")
            else:
                parts.append(f"{k}: '{self._escape_iso_gql_string(str(v))}'")
        return "{" + ", ".join(parts) + "}"

    # ------------------------------------------------------------------
    # Node / Vertex operations
    # ------------------------------------------------------------------
    async def has_node(self, node_id: str) -> bool:
        escaped = self._escape_iso_gql_string(node_id)
        gql = (
            f"MATCH (n:`{self._entity_label}` {{entity_id: '{escaped}'}}) "
            f"RETURN n LIMIT 1"
        )
        try:
            rows = await self._query(gql)
            return len(rows) > 0
        except Exception as e:
            logger.error(f"[{self.workspace}] has_node({node_id}) error: {e}")
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        escaped_src = self._escape_iso_gql_string(source_node_id)
        escaped_tgt = self._escape_iso_gql_string(target_node_id)
        gql = (
            f"MATCH (:`{self._entity_label}` {{entity_id: '{escaped_src}'}})"
            f"-[r:`{self._edge_label}`]-"
            f"(:`{self._entity_label}` {{entity_id: '{escaped_tgt}'}}) "
            f"RETURN r LIMIT 1"
        )
        try:
            rows = await self._query(gql)
            return len(rows) > 0
        except Exception as e:
            logger.error(
                f"[{self.workspace}] has_edge({source_node_id}, {target_node_id}) error: {e}"
            )
            return False

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        escaped = self._escape_iso_gql_string(node_id)
        gql = f"MATCH (n:`{self._entity_label}` {{entity_id: '{escaped}'}}) RETURN n"
        try:
            rows = await self._query(gql)
            if len(rows) > 1:
                logger.warning(
                    f"[{self.workspace}] Multiple vertices found for '{node_id}'. Using first."
                )
            if rows:
                node = rows[0].get("n", {})
                if isinstance(node, dict):
                    # ISO GQL nests properties under a 'properties' key
                    return dict(node.get("properties", node))
                return {}
            return None
        except Exception as e:
            logger.error(f"[{self.workspace}] get_node({node_id}) error: {e}")
            return None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        escaped = [self._escape_iso_gql_string(nid) for nid in node_ids]
        id_list = ", ".join(f"'{e}'" for e in escaped)
        gql = (
            f"MATCH (n:`{self._entity_label}`) "
            f"WHERE n.entity_id IN [{id_list}] "
            f"RETURN n"
        )
        try:
            rows = await self._query(gql)
            result: dict[str, dict] = {}
            for row in rows:
                node = row.get("n", {})
                if isinstance(node, dict):
                    props = node.get("properties", node)
                    if isinstance(props, dict):
                        entity_id = props.get("entity_id", "")
                        if entity_id:
                            result[str(entity_id)] = dict(props)
            return result
        except Exception as e:
            logger.error(f"[{self.workspace}] get_nodes_batch error: {e}")
            return {}

    async def node_degree(self, node_id: str) -> int:
        escaped = self._escape_iso_gql_string(node_id)
        gql = (
            f"MATCH (n:`{self._entity_label}` {{entity_id: '{escaped}'}})"
            f"-[r:`{self._edge_label}`]-() "
            f"RETURN count(r) AS degree"
        )
        try:
            rows = await self._query(gql)
            if rows:
                degree = rows[0].get("degree", 0)
                return int(degree) if degree is not None else 0
            return 0
        except Exception as e:
            logger.error(f"[{self.workspace}] node_degree({node_id}) error: {e}")
            return 0

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        escaped = [self._escape_iso_gql_string(nid) for nid in node_ids]
        id_list = ", ".join(f"'{e}'" for e in escaped)
        gql = (
            f"MATCH (n:`{self._entity_label}`)-[r:`{self._edge_label}`]-() "
            f"WHERE n.entity_id IN [{id_list}] "
            f"RETURN n.entity_id AS entity_id, count(r) AS degree"
        )
        try:
            rows = await self._query(gql)
            result: dict[str, int] = {nid: 0 for nid in node_ids}
            for row in rows:
                eid = str(row.get("entity_id", ""))
                if eid:
                    result[eid] = int(row.get("degree", 0))
            return result
        except Exception as e:
            logger.error(f"[{self.workspace}] node_degrees_batch error: {e}")
            return {nid: 0 for nid in node_ids}

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_deg = await self.node_degree(src_id)
        tgt_deg = await self.node_degree(tgt_id)
        return int(src_deg or 0) + int(tgt_deg or 0)

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        if not edge_pairs:
            return {}

        # Collect all unique node IDs.
        unique_ids: list[str] = list(
            {nid for src, tgt in edge_pairs for nid in (src, tgt)}
        )

        # Single bulk query for all node degrees.
        degrees: dict[str, int] = await self.node_degrees_batch(unique_ids)

        # Sum degrees for each edge pair.
        result: dict[tuple[str, str], int] = {}
        for src, tgt in edge_pairs:
            result[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)

        return result

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        if "entity_id" not in node_data:
            raise ValueError("node_data must contain 'entity_id'")

        # Ensure node_id is in node_data (as primary key)
        node_data.setdefault("entity_id", node_id)
        props = self._format_iso_props(node_data)
        gql = f"INSERT OR REPLACE (:`{self._entity_label}` {props})"
        try:
            await self._execute(gql)
        except Exception as e:
            logger.error(f"[{self.workspace}] upsert_node({node_id}) error: {e}")
            raise

    async def upsert_nodes_batch(
        self, nodes: list[tuple[str, dict[str, str]]], chunk_size: int = 50
    ) -> None:
        """Insert/update nodes in batched chunks using ISO GQL."""
        if not nodes:
            return
        for node_id, node_data in nodes:
            if "entity_id" not in node_data:
                raise ValueError("node_data must contain 'entity_id'")

        chunk_size = max(1, chunk_size)
        total = len(nodes)

        for offset in range(0, total, chunk_size):
            chunk = nodes[offset : offset + chunk_size]
            rows: list[str] = []
            for node_id, node_data in chunk:
                node_data.setdefault("entity_id", node_id)
                rows.append(self._format_iso_props(node_data))
            gql = "INSERT OR REPLACE " + ", ".join(
                f"(:`{self._entity_label}` {row})" for row in rows
            )
            try:
                await self._execute(gql)
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] upsert_nodes_batch error "
                    f"(chunk {offset // chunk_size + 1}, {len(chunk)} nodes): {e}"
                )
                raise

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        if not node_ids:
            return set()
        escaped = [self._escape_iso_gql_string(nid) for nid in node_ids]
        id_list = ", ".join(f"'{e}'" for e in escaped)
        gql = (
            f"MATCH (n:`{self._entity_label}`) "
            f"WHERE n.entity_id IN [{id_list}] "
            f"RETURN n.entity_id AS entity_id"
        )
        try:
            rows = await self._query(gql)
            return {str(row["entity_id"]) for row in rows if row.get("entity_id")}
        except Exception as e:
            logger.error(f"[{self.workspace}] has_nodes_batch error: {e}")
            return set()

    # ------------------------------------------------------------------
    # Edge / Relationship operations
    # ------------------------------------------------------------------
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        edge_props = self._format_iso_props(edge_data)
        escaped_src = self._escape_iso_gql_string(source_node_id)
        escaped_tgt = self._escape_iso_gql_string(target_node_id)
        gql = (
            f"MATCH (src:`{self._entity_label}` {{entity_id: '{escaped_src}'}}), "
            f"(tgt:`{self._entity_label}` {{entity_id: '{escaped_tgt}'}}) "
            f"INSERT OR REPLACE (src)-[:`{self._edge_label}` {edge_props}]->(tgt)"
        )
        try:
            await self._execute(gql)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] upsert_edge({source_node_id}, {target_node_id}) error: {e}"
            )
            raise

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]], chunk_size: int = 50
    ) -> None:
        """Insert/update edges in batched chunks using ISO GQL."""
        if not edges:
            return

        chunk_size = max(1, chunk_size)
        total = len(edges)

        for offset in range(0, total, chunk_size):
            chunk = edges[offset : offset + chunk_size]

            # Build MATCH ... INSERT OR REPLACE for each edge in the chunk.
            match_parts: list[str] = []
            insert_parts: list[str] = []
            for i, (src, tgt, edge_data) in enumerate(chunk):
                escaped_src = self._escape_iso_gql_string(src)
                escaped_tgt = self._escape_iso_gql_string(tgt)
                match_parts.append(
                    f"(s{i}:`{self._entity_label}` {{entity_id: '{escaped_src}'}}), "
                    f"(t{i}:`{self._entity_label}` {{entity_id: '{escaped_tgt}'}})"
                )
                edge_props = self._format_iso_props(edge_data)
                insert_parts.append(
                    f"(s{i})-[:`{self._edge_label}` {edge_props}]->(t{i})"
                )

            gql = (
                f"MATCH {', '.join(match_parts)} "
                f"INSERT OR REPLACE {', '.join(insert_parts)}"
            )
            try:
                await self._execute(gql)
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] upsert_edges_batch error "
                    f"(chunk {offset // chunk_size + 1}, {len(chunk)} edges): {e}"
                )
                raise

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        escaped_src = self._escape_iso_gql_string(source_node_id)
        escaped_tgt = self._escape_iso_gql_string(target_node_id)
        gql = (
            f"MATCH (:`{self._entity_label}` {{entity_id: '{escaped_src}'}})"
            f"-[r:`{self._edge_label}`]-"
            f"(:`{self._entity_label}` {{entity_id: '{escaped_tgt}'}}) "
            f"RETURN r"
        )
        try:
            rows = await self._query(gql)
            if len(rows) > 1:
                logger.warning(
                    f"[{self.workspace}] Multiple edges between '{source_node_id}' and '{target_node_id}'."
                )
            if rows:
                edge = rows[0].get("r", {})
                if isinstance(edge, dict):
                    # Edge result may wrap properties in a 'properties' key
                    edge_dict = dict(edge.get("properties", edge))
                else:
                    edge_dict = {}
                for key, default in {
                    "weight": 1.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }.items():
                    edge_dict.setdefault(key, default)
                return edge_dict
            return None
        except Exception as e:
            logger.error(
                f"[{self.workspace}] get_edge({source_node_id}, {target_node_id}) error: {e}"
            )
            return None

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}

        # Collect all unique node IDs across all pairs.
        unique_ids: set[str] = set()
        for pair in pairs:
            src = pair.get("src", "")
            tgt = pair.get("tgt", "")
            if src:
                unique_ids.add(src)
            if tgt:
                unique_ids.add(tgt)

        # Single bulk query: fetch all edges among the unique nodes.
        edge_lookup: dict[tuple[str, str], dict] = {}
        if unique_ids:
            escaped = [self._escape_iso_gql_string(nid) for nid in unique_ids]
            id_list = ", ".join(f"'{e}'" for e in escaped)
            gql = (
                f"MATCH (a:`{self._entity_label}`)-[r:`{self._edge_label}`]-"
                f"(b:`{self._entity_label}`) "
                f"WHERE a.entity_id IN [{id_list}] AND b.entity_id IN [{id_list}] "
                f"RETURN a.entity_id AS src, b.entity_id AS tgt, r"
            )
            try:
                rows = await self._query(gql)
                for row in rows:
                    src = str(row.get("src", ""))
                    tgt = str(row.get("tgt", ""))
                    if not src or not tgt:
                        continue
                    edge = row.get("r", {})
                    if isinstance(edge, dict):
                        edge_dict = dict(edge.get("properties", edge))
                    else:
                        edge_dict = {}
                    for key, default in {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }.items():
                        edge_dict.setdefault(key, default)
                    # Store both directions (edges are undirected).
                    edge_lookup[(src, tgt)] = edge_dict
                    edge_lookup[(tgt, src)] = edge_dict
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] get_edges_batch bulk query error: {e}"
                )

        # Assemble result for every requested pair.
        result: dict[tuple[str, str], dict] = {}
        for pair in pairs:
            src = pair.get("src", "")
            tgt = pair.get("tgt", "")
            key = (src, tgt)
            result[key] = edge_lookup.get(
                key,
                {
                    "weight": 1.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                },
            )

        return result

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        escaped = self._escape_iso_gql_string(source_node_id)
        gql = (
            f"MATCH (n:`{self._entity_label}` {{entity_id: '{escaped}'}})"
            f"-[r:`{self._edge_label}`]-"
            f"(connected:`{self._entity_label}`) "
            f"RETURN n.entity_id AS src_entity_id, connected.entity_id AS tgt_entity_id"
        )
        try:
            rows = await self._query(gql)
            if not rows:
                return []
            edges: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for row in rows:
                src_ent = row.get("src_entity_id")
                tgt_ent = row.get("tgt_entity_id")
                if src_ent and tgt_ent:
                    pair = (str(src_ent), str(tgt_ent))
                    if pair not in seen:
                        edges.append(pair)
                        seen.add(pair)
            return edges
        except Exception as e:
            logger.error(
                f"[{self.workspace}] get_node_edges({source_node_id}) error: {e}"
            )
            return None

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}
        escaped = [self._escape_iso_gql_string(nid) for nid in node_ids]
        id_list = ", ".join(f"'{e}'" for e in escaped)
        gql = (
            f"MATCH (n:`{self._entity_label}`)-[r:`{self._edge_label}`]-"
            f"(connected:`{self._entity_label}`) "
            f"WHERE n.entity_id IN [{id_list}] "
            f"RETURN n.entity_id AS src_entity_id, connected.entity_id AS tgt_entity_id"
        )
        result: dict[str, list[tuple[str, str]]] = {nid: [] for nid in node_ids}
        try:
            rows = await self._query(gql)
            for row in rows:
                src_ent = row.get("src_entity_id")
                tgt_ent = row.get("tgt_entity_id")
                if src_ent and tgt_ent:
                    src_str = str(src_ent)
                    if src_str in result:
                        result[src_str].append((src_str, str(tgt_ent)))
            return result
        except Exception as e:
            logger.error(f"[{self.workspace}] get_nodes_edges_batch error: {e}")
            return {nid: [] for nid in node_ids}

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------
    async def delete_node(self, node_id: str) -> None:
        escaped = self._escape_iso_gql_string(node_id)
        # Delete edges first (DETACH not available in ISO GQL), then the node
        try:
            await self._execute(
                f"MATCH (:`{self._entity_label}` {{entity_id: '{escaped}'}})"
                f"-[e]-() DELETE e"
            )
        except Exception:
            pass
        gql = f"MATCH (n:`{self._entity_label}` {{entity_id: '{escaped}'}}) DELETE n"
        try:
            await self._execute(gql)
        except Exception as e:
            logger.error(f"[{self.workspace}] delete_node({node_id}) error: {e}")
            raise

    async def remove_nodes(self, nodes: list[str]) -> None:
        for node_id in nodes:
            await self.delete_node(node_id)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        for src, tgt in edges:
            escaped_src = self._escape_iso_gql_string(src)
            escaped_tgt = self._escape_iso_gql_string(tgt)
            gql = (
                f"MATCH (:`{self._entity_label}` {{entity_id: '{escaped_src}'}})"
                f"-[r:`{self._edge_label}`]-"
                f"(:`{self._entity_label}` {{entity_id: '{escaped_tgt}'}}) "
                f"DELETE r"
            )
            try:
                await self._execute(gql)
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] remove_edges({src}, {tgt}) error: {e}"
                )

    # ------------------------------------------------------------------
    # Label / type operations
    # ------------------------------------------------------------------
    async def get_all_labels(self) -> list[str]:
        gql = (
            f"MATCH (n:`{self._entity_label}`) "
            f"RETURN DISTINCT n.entity_type AS entity_type"
        )
        try:
            rows = await self._query(gql)
            return [str(row["entity_type"]) for row in rows if row.get("entity_type")]
        except Exception as e:
            logger.error(f"[{self.workspace}] get_all_labels error: {e}")
            return []

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        gql = (
            f"MATCH (n:`{self._entity_label}`) "
            f"RETURN n.entity_type AS entity_type, count(*) AS cnt "
            f"ORDER BY cnt DESC "
            f"LIMIT {limit}"
        )
        try:
            rows = await self._query(gql)
            return [str(row["entity_type"]) for row in rows if row.get("entity_type")]
        except Exception as e:
            logger.error(f"[{self.workspace}] get_popular_labels error: {e}")
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search entity labels by entity_type with fuzzy matching.

        Uses LOOKUP ON with the full-text index for performance when the
        index is available, falling back to a MATCH … CONTAINS scan.
        """
        query_strip = query.strip()
        if not query_strip:
            return []

        escaped = self._escape_iso_gql_string(query_strip)

        # Attempt full-text index search first.
        try:
            gql = (
                f"MATCH (n:`{self._entity_label}`) "
                f"LET fsscore = ftscore(n.entity_type, '{escaped}') "
                f"FILTER WHERE fsscore > 0 "
                f"RETURN n.entity_type AS entity_type "
                f"ORDER BY fsscore DESC "
                f"LIMIT {limit}"
            )
            rows = await self._query(gql)
            labels = [str(row["entity_type"]) for row in rows if row.get("entity_type")]
            if labels:
                return labels
        except Exception:
            logger.warning(
                f"[{self.workspace}] Full-text index search failed or not available; falling back to CONTAINS scan. {gql}"
            )
            pass  # Full-text index not available; fall through to CONTAINS.

        # Fallback: MATCH with CONTAINS on entity_type.
        try:
            gql = (
                f"MATCH (n:`{self._entity_label}`) "
                f"WHERE contains(n.entity_type, '{escaped}') "
                f"RETURN DISTINCT n.entity_type AS entity_type "
                f"LIMIT {limit}"
            )
            rows = await self._query(gql)
            labels = [str(row["entity_type"]) for row in rows if row.get("entity_type")]
            # Score: exact match > starts-with > contains
            scored: list[tuple[int, str]] = []
            for label in labels:
                if label == query_strip:
                    scored.append((1000, label))
                elif label.startswith(query_strip):
                    scored.append((500, label))
                else:
                    scored.append((100 - min(len(label), 100), label))
            scored.sort(key=lambda x: x[0], reverse=True)
            seen: set[str] = set()
            result: list[str] = []
            for _, lbl in scored:
                if lbl not in seen:
                    seen.add(lbl)
                    result.append(lbl)
            return result[:limit]
        except Exception as e:
            logger.error(f"[{self.workspace}] search_labels error: {e}")
            return []

    # ------------------------------------------------------------------
    # Bulk retrieval
    # ------------------------------------------------------------------
    async def get_all_nodes(self) -> list[dict]:
        gql = f"MATCH (n:`{self._entity_label}`) RETURN n"
        try:
            rows = await self._query(gql)
            result: list[dict] = []
            for row in rows:
                node = row.get("n", {})
                if isinstance(node, dict):
                    result.append(dict(node.get("properties", node)))
            return result
        except Exception as e:
            logger.error(f"[{self.workspace}] get_all_nodes error: {e}")
            return []

    async def get_all_edges(self) -> list[dict]:
        gql = (
            f"MATCH (a:`{self._entity_label}`)-[r:`{self._edge_label}`]-(b:`{self._entity_label}`) "
            f"RETURN a.entity_id AS src, b.entity_id AS tgt, r"
        )
        try:
            rows = await self._query(gql)
            result: list[dict] = []
            for row in rows:
                edge = row.get("r", {})
                if isinstance(edge, dict):
                    edge_dict = dict(edge.get("properties", edge))
                else:
                    edge_dict = {}
                edge_dict["src"] = row.get("src", "")
                edge_dict["tgt"] = row.get("tgt", "")
                result.append(edge_dict)
            return result
        except Exception as e:
            logger.error(f"[{self.workspace}] get_all_edges error: {e}")
            return []

    # ------------------------------------------------------------------
    # Knowledge graph traversal (BFS)
    # ------------------------------------------------------------------
    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int | None = None,
    ) -> KnowledgeGraph:
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        result = KnowledgeGraph()
        seen_nodes: set[str] = set()
        seen_edges: set[tuple[str, str]] = set()

        if node_label == "*":
            # Get top nodes by degree
            count_gql = f"MATCH (n:`{self._entity_label}`) RETURN count(n) AS total"
            try:
                count_rows = await self._query(count_gql)
                total = int(count_rows[0]["total"]) if count_rows else 0
                if total > max_nodes:
                    result.is_truncated = True
                    logger.info(
                        f"[{self.workspace}] Graph truncated: {total} nodes, limited to {max_nodes}"
                    )
            except Exception as e:
                logger.error(f"[{self.workspace}] Error counting nodes: {e}")

            node_gql = (
                f"MATCH (n:`{self._entity_label}`) "
                f"OPTIONAL MATCH (n)-[r:`{self._edge_label}`]-() "
                f"RETURN n, count(r) AS degree "
                f"ORDER BY degree DESC "
                f"LIMIT {max_nodes}"
            )
            try:
                node_rows = await self._query(node_gql)
                kept_ids: list[str] = []
                for row in node_rows:
                    node = row.get("n", {})
                    if isinstance(node, dict):
                        props = node.get("properties", node)
                        if isinstance(props, dict):
                            nid = props.get("entity_id", "")
                        else:
                            nid = ""
                    else:
                        nid = ""
                    if nid and str(nid) not in seen_nodes:
                        seen_nodes.add(str(nid))
                        kept_ids.append(str(nid))
                        labels_list = [self._entity_label]
                        result.nodes.append(
                            KnowledgeGraphNode(
                                id=str(nid),
                                labels=labels_list,
                                properties=props if isinstance(props, dict) else {},
                            )
                        )
                # Get edges between kept nodes
                if len(kept_ids) >= 2:
                    id_list = ", ".join(
                        f"'{self._escape_iso_gql_string(nid)}'" for nid in kept_ids
                    )
                    edge_gql = (
                        f"MATCH (a:`{self._entity_label}`)-[r:`{self._edge_label}`]-(b:`{self._entity_label}`) "
                        f"WHERE a.entity_id IN [{id_list}] AND b.entity_id IN [{id_list}] "
                        f"RETURN a.entity_id AS src, b.entity_id AS tgt, r"
                    )
                    edge_rows = await self._query(edge_gql)
                    for row in edge_rows:
                        src = str(row.get("src", ""))
                        tgt = str(row.get("tgt", ""))
                        pair = (
                            (min(src, tgt), max(src, tgt)) if src <= tgt else (tgt, src)
                        )
                        if pair not in seen_edges:
                            seen_edges.add(pair)
                            edge = row.get("r", {})
                            if isinstance(edge, dict):
                                edge_dict = dict(edge.get("properties", edge))
                            else:
                                edge_dict = {}
                            result.edges.append(
                                KnowledgeGraphEdge(
                                    id=f"{pair[0]}-{pair[1]}",
                                    type=self._edge_label,
                                    source=pair[0],
                                    target=pair[1],
                                    properties=edge_dict,
                                )
                            )
            except Exception as e:
                logger.error(f"[{self.workspace}] get_knowledge_graph(*) error: {e}")
            return result

        # BFS from specific node
        current_level = {node_label}
        for depth in range(max_depth + 1):
            if not current_level:
                break
            next_level: set[str] = set()
            for nid in current_level:
                if nid in seen_nodes:
                    continue

                # If we already hit max_nodes, flag truncation
                if len(seen_nodes) >= max_nodes:
                    result.is_truncated = True
                    return result

                node = await self.get_node(nid)
                if node:
                    seen_nodes.add(nid)
                    labels_list = [self._entity_label]
                    result.nodes.append(
                        KnowledgeGraphNode(
                            id=nid,
                            labels=labels_list,
                            properties=node,
                        )
                    )
                else:
                    continue

                edges = await self.get_node_edges(nid)
                if edges:
                    for src, tgt in edges:
                        sorted_pair = (src, tgt) if src <= tgt else (tgt, src)
                        if sorted_pair not in seen_edges:
                            seen_edges.add(sorted_pair)
                            edge_data = await self.get_edge(src, tgt)
                            result.edges.append(
                                KnowledgeGraphEdge(
                                    id=f"{src}-{tgt}",
                                    type=self._edge_label,
                                    source=src,
                                    target=tgt,
                                    properties=edge_data or {},
                                )
                            )
                        if depth < max_depth:
                            if src != nid and src not in seen_nodes:
                                next_level.add(src)
                            if tgt != nid and tgt not in seen_nodes:
                                next_level.add(tgt)
            current_level = next_level

        if len(seen_nodes) >= max_nodes:
            result.is_truncated = True
        return result

    # ------------------------------------------------------------------
    # Drop all workspace data
    # ------------------------------------------------------------------
    async def drop(self) -> dict[str, str]:
        try:
            # Delete all edges first, then vertices
            try:
                await self._execute(f"MATCH ()-[r:`{self._edge_label}`]-() DELETE r")
            except Exception as e:
                logger.warning(f"[{self.workspace}] Error deleting edges: {e}")
            try:
                await self._execute(f"MATCH (n:`{self._entity_label}`) DELETE n")
            except Exception as e:
                logger.warning(f"[{self.workspace}] Error deleting vertices: {e}")

            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] drop error: {e}")
            return {"status": "error", "message": str(e)}
