"""Two-table graph storage for the isolated Hologres backend.

Semantics mirror ``PGTableGraphStorage`` (the plain-PostgreSQL reference):
canonical undirected edge order via Python ``min``/``max``, node property
MERGE on upsert versus edge property REPLACE, self-loops counting twice in
degrees but appearing once in adjacency lists, and degree-ranked knowledge
graph truncation. Every SQL shape used here was proven against a live
Hologres 5.x instance first; the differences from the reference are all
Hologres-imposed:

* no foreign keys — edge cleanup is explicit and always runs BEFORE node
  deletes, never via CASCADE;
* no multi-argument ``unnest`` — pair batches use ``generate_series`` with
  array subscripts;
* no ``COLLATE "C"`` clause — the capability probe instead proves the
  database collation IS ``C``, so plain ``ORDER BY`` already matches
  Python's code-point ordering.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any, final

from ...base import BaseGraphStorage
from ...namespace import NameSpace
from ...types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ...utils import validate_workspace
from .capabilities import (
    probe_production_capabilities,
    prove_stream_copy_capability,
)
from .client import quote_qualified_identifier
from .config import HologresConfig
from .kv import _SHARED_CLIENTS, _release_shared_client
from .schema import (
    GRAPH_EDGES_TABLE_NAME,
    GRAPH_NODES_TABLE_NAME,
    HologresSchemaManager,
    graph_schema_descriptors,
)


_ID_CHUNK_SIZE = 1000
_UPSERT_RECORD_LIMIT = 200
_UPSERT_BYTE_LIMIT = 4 * 1024 * 1024
_MISSING = object()
_ALLOWED_NAMESPACES = frozenset({NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION})


class HologresGraphError(RuntimeError):
    """Raised when a graph operation cannot return a trustworthy result."""


def _row_field(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return _MISSING


def _deterministic_json(value: Any, message: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        raise HologresGraphError(message) from None


def _decode_properties(value: Any) -> dict[str, Any]:
    try:
        if isinstance(value, str):
            decoded = json.loads(value)
        elif isinstance(value, Mapping):
            decoded = dict(value)
        else:
            raise TypeError
    except (TypeError, ValueError, json.JSONDecodeError):
        raise HologresGraphError("Hologres graph row is corrupt") from None
    if not isinstance(decoded, dict):
        raise HologresGraphError("Hologres graph row is corrupt")
    return decoded


def _materialize_rows(rows: Any, message: str) -> list[Any]:
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise HologresGraphError(message)
    try:
        return list(rows)
    except (TypeError, ValueError):
        raise HologresGraphError(message) from None


def _require_str(value: Any, message: str) -> str:
    if not isinstance(value, str):
        raise HologresGraphError(message)
    return value


def _require_int(value: Any, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HologresGraphError(message)
    return value


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _payload_batches(items: Sequence[tuple[Any, ...]], payload_index: int):
    """Split records so no statement exceeds the record or byte budget."""

    batch: list[tuple[Any, ...]] = []
    batch_bytes = 0
    for item in items:
        size = len(item[payload_index].encode("utf-8"))
        if size > _UPSERT_BYTE_LIMIT:
            raise HologresGraphError(
                "Hologres graph record exceeds the upsert batch limit"
            )
        if batch and (
            len(batch) >= _UPSERT_RECORD_LIMIT
            or batch_bytes + size > _UPSERT_BYTE_LIMIT
        ):
            yield batch
            batch = []
            batch_bytes = 0
        batch.append(item)
        batch_bytes += size
    if batch:
        yield batch


@final
@dataclass(repr=False)
class HologresGraphStorage(BaseGraphStorage):
    """Undirected knowledge graph over two fixed shared Hologres tables."""

    config: HologresConfig | None = field(default=None, repr=False)
    client: Any | None = field(default=None, repr=False)
    _active_client: Any | None = field(default=None, init=False, repr=False)
    _effective_config: HologresConfig | None = field(
        default=None, init=False, repr=False
    )
    _owns_shared_client: bool = field(default=False, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _lifecycle_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or self.namespace not in _ALLOWED_NAMESPACES
        ):
            raise ValueError("Unsupported Hologres graph namespace")
        try:
            self.workspace = validate_workspace(self.workspace)
        except (TypeError, ValueError):
            raise ValueError("Invalid Hologres graph workspace") from None
        self._lifecycle_lock = asyncio.Lock()

    def __repr__(self) -> str:
        return "HologresGraphStorage(<redacted>)"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Acquire one client and reconcile the two graph tables."""

        async with self._lifecycle_lock:
            if self._initialized:
                return

            actual_client = self.client
            config = self.config
            owns_shared = False
            try:
                if actual_client is None:
                    if config is None:
                        config = HologresConfig.from_env()
                        self.config = config
                    actual_client = await _SHARED_CLIENTS.acquire(config)
                    owns_shared = True
                elif config is None:
                    config = getattr(actual_client, "config", None)
                    if not isinstance(config, HologresConfig):
                        raise HologresGraphError(
                            "Hologres graph configuration is unavailable"
                        )

                capabilities = await probe_production_capabilities(actual_client)
                manager = HologresSchemaManager(actual_client, schema=config.schema)
                await manager.initialize(graph_schema_descriptors(config.schema))
                capabilities = await prove_stream_copy_capability(
                    actual_client, capabilities
                )
                apply_capabilities = getattr(
                    actual_client, "apply_capabilities", None
                )
                if apply_capabilities is not None:
                    apply_capabilities(capabilities)
            except BaseException as initialization_error:
                release_error = None
                if owns_shared and actual_client is not None and config is not None:
                    try:
                        await _release_shared_client(
                            _SHARED_CLIENTS.release, config, actual_client
                        )
                    except asyncio.CancelledError:
                        raise
                    except BaseException:
                        release_error = HologresGraphError(
                            "Hologres graph shared client release failed"
                        )
                if isinstance(initialization_error, asyncio.CancelledError):
                    if release_error is not None:
                        raise initialization_error from release_error
                    raise
                error = HologresGraphError(
                    "Hologres graph initialization failed"
                )
                if release_error is not None:
                    raise error from release_error
                raise error from None

            self._active_client = actual_client
            self._effective_config = config
            self._owns_shared_client = owns_shared
            self._initialized = True

    async def finalize(self) -> None:
        """Detach this storage and release only a manager-owned client."""

        async with self._lifecycle_lock:
            actual_client = self._active_client
            config = self._effective_config
            owns_shared = self._owns_shared_client
            self._active_client = None
            self._effective_config = None
            self._owns_shared_client = False
            self._initialized = False
            if owns_shared and actual_client is not None and config is not None:
                try:
                    await _release_shared_client(
                        _SHARED_CLIENTS.release, config, actual_client
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise HologresGraphError(
                        "Hologres graph finalization failed"
                    ) from None

    def _ready(self) -> tuple[Any, str, str]:
        if not self._initialized or self._active_client is None:
            raise HologresGraphError("Hologres graph storage is not initialized")
        if self._effective_config is None:
            raise HologresGraphError("Hologres graph configuration is unavailable")
        schema = self._effective_config.schema
        return (
            self._active_client,
            quote_qualified_identifier(schema, GRAPH_NODES_TABLE_NAME),
            quote_qualified_identifier(schema, GRAPH_EDGES_TABLE_NAME),
        )

    # ------------------------------------------------------------------
    # Row shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _node_props(node_id: str, properties: Any) -> dict[str, Any]:
        props = _decode_properties(properties)
        props["entity_id"] = node_id
        return props

    @staticmethod
    def _node_output(node_id: str, properties: Any) -> dict[str, Any]:
        props = HologresGraphStorage._node_props(node_id, properties)
        props["id"] = node_id
        return props

    async def index_done_callback(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        client, nodes, _edges = self._ready()
        try:
            row = await client.fetch_one(
                f"SELECT 1 AS present FROM {nodes} "
                "WHERE workspace = $1 AND namespace = $2 AND id = $3",
                self.workspace,
                self.namespace,
                node_id,
                descriptor="graph.node.exists",
            )
        except Exception:
            raise HologresGraphError("Hologres graph node read failed") from None
        return row is not None

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        client, nodes, _edges = self._ready()
        try:
            row = await client.fetch_one(
                f"SELECT properties FROM {nodes} "
                "WHERE workspace = $1 AND namespace = $2 AND id = $3",
                self.workspace,
                self.namespace,
                node_id,
                descriptor="graph.node.read",
            )
        except Exception:
            raise HologresGraphError("Hologres graph node read failed") from None
        if row is None:
            return None
        return self._node_props(node_id, _row_field(row, "properties"))

    def _prepare_node_record(
        self, node_id: str, node_data: dict[str, str]
    ) -> tuple[str, str]:
        # Match PGTableGraphStorage / PGGraphStorage: require entity_id to
        # surface malformed caller payloads early, then force it to node_id.
        if not isinstance(node_data, Mapping) or "entity_id" not in node_data:
            raise ValueError(
                "Hologres: node properties must contain an 'entity_id' field"
            )
        if not isinstance(node_id, str):
            raise HologresGraphError("Hologres graph input is invalid")
        props = dict(node_data)
        props["entity_id"] = node_id
        return node_id, _deterministic_json(
            props, "Hologres graph input is invalid"
        )

    async def _upsert_node_records(
        self, records: Sequence[tuple[str, str]]
    ) -> None:
        if not records:
            return
        client, nodes, _edges = self._ready()
        # Merge (not replace) so omitted keys survive, matching
        # NetworkXStorage.add_node(**data) and PGGraphStorage's SET n += .
        # EXCLUDED wins on shared keys and always carries entity_id = node_id.
        sql = (
            f"INSERT INTO {nodes} AS current "
            "(workspace, namespace, id, properties, updated_at) "
            "SELECT $1, $2, ($3::text[])[g.idx], ($4::jsonb[])[g.idx], "
            "CURRENT_TIMESTAMP "
            "FROM generate_series(1, $5::int) AS g(idx) "
            "ON CONFLICT (workspace, namespace, id) DO UPDATE SET "
            "properties = current.properties || EXCLUDED.properties, "
            "updated_at = CURRENT_TIMESTAMP"
        )
        for batch in _payload_batches(records, 1):
            ids = [record[0] for record in batch]
            payloads = [record[1] for record in batch]
            try:
                await client.execute_one(
                    sql,
                    self.workspace,
                    self.namespace,
                    ids,
                    payloads,
                    len(ids),
                    descriptor="graph.node.upsert",
                    replay_safe=True,
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph node upsert failed"
                ) from None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        await self._upsert_node_records([self._prepare_node_record(node_id, node_data)])

    async def upsert_nodes_batch(
        self, nodes: list[tuple[str, dict[str, str]]]
    ) -> None:
        if not nodes:
            return
        # Last write wins per id, and dedup also keeps one multi-row statement
        # from touching the same conflict target twice.
        deduped: dict[str, tuple[str, str]] = {}
        for node_id, node_data in nodes:
            deduped[node_id] = self._prepare_node_record(node_id, node_data)
        await self._upsert_node_records(list(deduped.values()))

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        if not node_ids:
            return set()
        client, nodes, _edges = self._ready()
        existing: set[str] = set()
        unique_ids = list(dict.fromkeys(node_ids))
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    f"SELECT id FROM {nodes} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "AND id = ANY($3::text[])",
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="graph.node.exists.batch",
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph node read failed"
                ) from None
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            ):
                existing.add(
                    _require_str(
                        _row_field(row, "id"),
                        "Hologres graph response is corrupt",
                    )
                )
        return existing

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        client, nodes, _edges = self._ready()
        result: dict[str, dict] = {}
        unique_ids = list(dict.fromkeys(node_ids))
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    f"SELECT id, properties FROM {nodes} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "AND id = ANY($3::text[])",
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="graph.node.read.batch",
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph node read failed"
                ) from None
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            ):
                node_id = _require_str(
                    _row_field(row, "id"), "Hologres graph response is corrupt"
                )
                result[node_id] = self._node_props(
                    node_id, _row_field(row, "properties")
                )
        return result

    async def delete_node(self, node_id: str) -> None:
        await self.remove_nodes([node_id])

    async def remove_nodes(self, nodes: list[str]) -> None:
        if not nodes:
            return
        if any(not isinstance(node_id, str) for node_id in nodes):
            raise HologresGraphError("Hologres graph input is invalid")
        client, nodes_table, edges = self._ready()
        unique_ids = list(dict.fromkeys(nodes))
        # No FK CASCADE on Hologres: incident edges are removed explicitly and
        # FIRST, so an interruption leaves an isolated node (benign) rather
        # than dangling edges.
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            try:
                await client.execute_one(
                    f"DELETE FROM {edges} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "AND (src_id = ANY($3::text[]) OR tgt_id = ANY($3::text[]))",
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="graph.node.delete.edges",
                    replay_safe=True,
                )
                await client.execute_one(
                    f"DELETE FROM {nodes_table} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "AND id = ANY($3::text[])",
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="graph.node.delete",
                    replay_safe=True,
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph node delete failed"
                ) from None

    # ------------------------------------------------------------------
    # Edge operations
    # All writes normalise to canonical order:
    # src_id = min(a, b), tgt_id = max(a, b).
    # ------------------------------------------------------------------

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        client, _nodes, edges = self._ready()
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        try:
            row = await client.fetch_one(
                f"SELECT 1 AS present FROM {edges} "
                "WHERE workspace = $1 AND namespace = $2 "
                "AND src_id = $3 AND tgt_id = $4",
                self.workspace,
                self.namespace,
                src,
                tgt,
                descriptor="graph.edge.exists",
            )
        except Exception:
            raise HologresGraphError("Hologres graph edge read failed") from None
        return row is not None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        client, _nodes, edges = self._ready()
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        try:
            row = await client.fetch_one(
                f"SELECT properties FROM {edges} "
                "WHERE workspace = $1 AND namespace = $2 "
                "AND src_id = $3 AND tgt_id = $4",
                self.workspace,
                self.namespace,
                src,
                tgt,
                descriptor="graph.edge.read",
            )
        except Exception:
            raise HologresGraphError("Hologres graph edge read failed") from None
        if row is None:
            return None
        return _decode_properties(_row_field(row, "properties"))

    def _prepare_edge_record(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> tuple[str, str, str]:
        if (
            not isinstance(source_node_id, str)
            or not isinstance(target_node_id, str)
            or not isinstance(edge_data, Mapping)
        ):
            raise HologresGraphError("Hologres graph input is invalid")
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        return src, tgt, _deterministic_json(
            dict(edge_data), "Hologres graph input is invalid"
        )

    async def _upsert_edge_records(
        self, records: Sequence[tuple[str, str, str]]
    ) -> None:
        if not records:
            return
        client, nodes, edges = self._ready()
        endpoints = list(
            dict.fromkeys(
                endpoint for record in records for endpoint in record[:2]
            )
        )
        # Auto-create missing endpoints first (DO NOTHING keeps existing
        # properties), then write the edges. Two statements because Hologres
        # rejects nothing here individually but the storage keeps every write
        # a single independently replay-safe statement; a crash in between
        # leaves stub nodes without edges, which a retry repairs.
        for chunk in _chunks(endpoints, _ID_CHUNK_SIZE):
            try:
                await client.execute_one(
                    f"INSERT INTO {nodes} "
                    "(workspace, namespace, id, properties, updated_at) "
                    "SELECT $1, $2, u.id, "
                    "jsonb_build_object('entity_id', u.id), CURRENT_TIMESTAMP "
                    "FROM unnest($3::text[]) AS u(id) "
                    "ON CONFLICT (workspace, namespace, id) DO NOTHING",
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="graph.edge.endpoints",
                    replay_safe=True,
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph edge upsert failed"
                ) from None
        # Edge properties REPLACE the stored value (unlike node merge),
        # matching PGTableGraphStorage and NetworkX add_edge semantics.
        sql = (
            f"INSERT INTO {edges} "
            "(workspace, namespace, src_id, tgt_id, properties, updated_at) "
            "SELECT $1, $2, ($3::text[])[g.idx], ($4::text[])[g.idx], "
            "($5::jsonb[])[g.idx], CURRENT_TIMESTAMP "
            "FROM generate_series(1, $6::int) AS g(idx) "
            "ON CONFLICT (workspace, namespace, src_id, tgt_id) DO UPDATE SET "
            "properties = EXCLUDED.properties, updated_at = CURRENT_TIMESTAMP"
        )
        for batch in _payload_batches(records, 2):
            try:
                await client.execute_one(
                    sql,
                    self.workspace,
                    self.namespace,
                    [record[0] for record in batch],
                    [record[1] for record in batch],
                    [record[2] for record in batch],
                    len(batch),
                    descriptor="graph.edge.upsert",
                    replay_safe=True,
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph edge upsert failed"
                ) from None

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        await self._upsert_edge_records(
            [self._prepare_edge_record(source_node_id, target_node_id, edge_data)]
        )

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        if not edges:
            return
        # Last write wins per canonical pair; dedup also keeps one multi-row
        # statement from touching the same conflict target twice.
        deduped: dict[tuple[str, str], tuple[str, str, str]] = {}
        for source_node_id, target_node_id, edge_data in edges:
            record = self._prepare_edge_record(
                source_node_id, target_node_id, edge_data
            )
            deduped[(record[0], record[1])] = record
        await self._upsert_edge_records(list(deduped.values()))

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return
        if not all(
            isinstance(edge[0], str) and isinstance(edge[1], str) for edge in edges
        ):
            raise HologresGraphError("Edge node IDs must be non-None strings")
        client, _nodes, edges_table = self._ready()
        pairs = list(
            dict.fromkeys(
                (min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges
            )
        )
        # Pair matching via generate_series array subscripts — the live-proven
        # substitute for the multi-argument unnest Hologres rejects.
        sql = (
            f"DELETE FROM {edges_table} AS e "
            "WHERE e.workspace = $1 AND e.namespace = $2 "
            "AND EXISTS ("
            "SELECT 1 FROM generate_series(1, $5::int) AS g(idx) "
            "WHERE ($3::text[])[g.idx] = e.src_id "
            "AND ($4::text[])[g.idx] = e.tgt_id)"
        )
        for chunk in _chunks(pairs, _ID_CHUNK_SIZE):
            try:
                await client.execute_one(
                    sql,
                    self.workspace,
                    self.namespace,
                    [pair[0] for pair in chunk],
                    [pair[1] for pair in chunk],
                    len(chunk),
                    descriptor="graph.edge.delete",
                    replay_safe=True,
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph edge delete failed"
                ) from None

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}
        requested: list[tuple[str, str]] = []
        for pair in pairs:
            src = pair.get("src") if isinstance(pair, Mapping) else None
            tgt = pair.get("tgt") if isinstance(pair, Mapping) else None
            if not isinstance(src, str) or not isinstance(tgt, str):
                raise HologresGraphError("Hologres graph input is invalid")
            requested.append((src, tgt))

        canonical = list(
            dict.fromkeys((min(s, t), max(s, t)) for s, t in requested)
        )
        client, _nodes, edges = self._ready()
        sql = (
            f"SELECT e.src_id, e.tgt_id, e.properties FROM {edges} e "
            "JOIN (SELECT ($3::text[])[g.idx] AS src, ($4::text[])[g.idx] AS tgt "
            "FROM generate_series(1, $5::int) AS g(idx)) p "
            "ON p.src = e.src_id AND p.tgt = e.tgt_id "
            "WHERE e.workspace = $1 AND e.namespace = $2"
        )
        found: dict[tuple[str, str], dict] = {}
        for chunk in _chunks(canonical, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    sql,
                    self.workspace,
                    self.namespace,
                    [pair[0] for pair in chunk],
                    [pair[1] for pair in chunk],
                    len(chunk),
                    descriptor="graph.edge.read.batch",
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph edge read failed"
                ) from None
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            ):
                src = _require_str(
                    _row_field(row, "src_id"),
                    "Hologres graph response is corrupt",
                )
                tgt = _require_str(
                    _row_field(row, "tgt_id"),
                    "Hologres graph response is corrupt",
                )
                found[(src, tgt)] = _decode_properties(
                    _row_field(row, "properties")
                )

        result: dict[tuple[str, str], dict] = {}
        for src, tgt in requested:
            properties = found.get((min(src, tgt), max(src, tgt)))
            if properties is not None:
                result[(src, tgt)] = properties
        return result

    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]] | None:
        # Three-outcome contract: [] = exists and isolated, None = confirmed
        # absent, raise = the backend could not answer (has_node raises too).
        if not await self.has_node(source_node_id):
            return None
        client, _nodes, edges = self._ready()
        try:
            rows = await client.fetch_all(
                f"SELECT src_id, tgt_id FROM {edges} "
                "WHERE workspace = $1 AND namespace = $2 "
                "AND (src_id = $3 OR tgt_id = $3)",
                self.workspace,
                self.namespace,
                source_node_id,
                descriptor="graph.edge.adjacency",
            )
        except Exception:
            raise HologresGraphError("Hologres graph edge read failed") from None
        result: list[tuple[str, str]] = []
        for row in _materialize_rows(rows, "Hologres graph response is corrupt"):
            src = _require_str(
                _row_field(row, "src_id"), "Hologres graph response is corrupt"
            )
            tgt = _require_str(
                _row_field(row, "tgt_id"), "Hologres graph response is corrupt"
            )
            result.append(
                (source_node_id, tgt if src == source_node_id else src)
            )
        result.sort(key=lambda edge: edge[1])
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}
        client, _nodes, edges = self._ready()
        unique_ids = list(dict.fromkeys(node_ids))
        result: dict[str, list[tuple[str, str]]] = {
            node_id: [] for node_id in unique_ids
        }
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    f"SELECT src_id, tgt_id FROM {edges} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "AND (src_id = ANY($3::text[]) OR tgt_id = ANY($3::text[]))",
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="graph.edge.adjacency.batch",
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph edge read failed"
                ) from None
            members = set(chunk)
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            ):
                src = _require_str(
                    _row_field(row, "src_id"),
                    "Hologres graph response is corrupt",
                )
                tgt = _require_str(
                    _row_field(row, "tgt_id"),
                    "Hologres graph response is corrupt",
                )
                if src in members:
                    result[src].append((src, tgt))
                # A self-loop appears ONCE in the adjacency list even though
                # it counts twice in the degree — same rule as the reference.
                if tgt in members and src != tgt:
                    result[tgt].append((tgt, src))
        for adjacency in result.values():
            adjacency.sort(key=lambda edge: edge[1])
        return result

    # ------------------------------------------------------------------
    # Degree queries — a self-loop counts twice, matching networkx.
    # ------------------------------------------------------------------

    async def node_degree(self, node_id: str) -> int:
        client, _nodes, edges = self._ready()
        try:
            value = await client.fetch_value(
                "SELECT count(*) FROM ("
                f"SELECT src_id AS id FROM {edges} "
                "WHERE workspace = $1 AND namespace = $2 AND src_id = $3 "
                "UNION ALL "
                f"SELECT tgt_id AS id FROM {edges} "
                "WHERE workspace = $1 AND namespace = $2 AND tgt_id = $3"
                ") sub",
                self.workspace,
                self.namespace,
                node_id,
                descriptor="graph.degree.node",
            )
        except Exception:
            raise HologresGraphError("Hologres graph degree read failed") from None
        return _require_int(value, "Hologres graph response is corrupt")

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        degrees = await self.node_degrees_batch([src_id, tgt_id])
        return degrees.get(src_id, 0) + degrees.get(tgt_id, 0)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        client, _nodes, edges = self._ready()
        unique_ids = list(dict.fromkeys(node_ids))
        result: dict[str, int] = {node_id: 0 for node_id in unique_ids}
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    "SELECT id, count(*) AS degree FROM ("
                    f"SELECT src_id AS id FROM {edges} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "AND src_id = ANY($3::text[]) "
                    "UNION ALL "
                    f"SELECT tgt_id AS id FROM {edges} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "AND tgt_id = ANY($3::text[])"
                    ") sub GROUP BY id",
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="graph.degree.batch",
                )
            except Exception:
                raise HologresGraphError(
                    "Hologres graph degree read failed"
                ) from None
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            ):
                node_id = _require_str(
                    _row_field(row, "id"), "Hologres graph response is corrupt"
                )
                if node_id in result:
                    result[node_id] = _require_int(
                        _row_field(row, "degree"),
                        "Hologres graph response is corrupt",
                    )
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        if not edge_pairs:
            return {}
        node_ids = list(
            dict.fromkeys(node for pair in edge_pairs for node in pair)
        )
        degrees = await self.node_degrees_batch(node_ids)
        return {
            (src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    # ------------------------------------------------------------------
    # Label queries
    # ------------------------------------------------------------------

    async def get_all_labels(self) -> list[str]:
        client, nodes, _edges = self._ready()
        try:
            rows = await client.fetch_all(
                f"SELECT id FROM {nodes} "
                "WHERE workspace = $1 AND namespace = $2",
                self.workspace,
                self.namespace,
                descriptor="graph.labels.all",
            )
        except Exception:
            raise HologresGraphError("Hologres graph label read failed") from None
        return sorted(
            _require_str(
                _row_field(row, "id"), "Hologres graph response is corrupt"
            )
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            )
        )

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise HologresGraphError("Hologres graph limit is invalid")
        client, nodes, edges = self._ready()
        # Rank ALL nodes including isolated (degree 0) ones, ties broken by
        # id ascending. No COLLATE clause: the probe proves datcollate='C',
        # so the default ORDER BY is already Python's code-point order.
        try:
            rows = await client.fetch_all(
                "SELECT n.id AS id, COALESCE(d.degree, 0) AS degree "
                f"FROM {nodes} n "
                "LEFT JOIN ("
                "SELECT id, count(*) AS degree FROM ("
                f"SELECT src_id AS id FROM {edges} "
                "WHERE workspace = $1 AND namespace = $2 "
                "UNION ALL "
                f"SELECT tgt_id AS id FROM {edges} "
                "WHERE workspace = $1 AND namespace = $2"
                ") sub GROUP BY id"
                ") d ON d.id = n.id "
                "WHERE n.workspace = $1 AND n.namespace = $2 "
                "ORDER BY degree DESC, n.id ASC "
                "LIMIT $3::int",
                self.workspace,
                self.namespace,
                limit,
                descriptor="graph.labels.popular",
            )
        except Exception:
            raise HologresGraphError("Hologres graph label read failed") from None
        return [
            _require_str(
                _row_field(row, "id"), "Hologres graph response is corrupt"
            )
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            )
        ]

    @staticmethod
    def _escape_like(value: str) -> str:
        return (
            value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        if not isinstance(query, str):
            raise HologresGraphError("Hologres graph input is invalid")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise HologresGraphError("Hologres graph limit is invalid")
        q = query.strip().lower()
        if not q:
            return []
        escaped = self._escape_like(q)
        client, nodes, _edges = self._ready()
        # Mirrors PGTableGraphStorage scoring: exact 1000, prefix 500, else
        # 100 - LENGTH(id) with a +50 word-boundary bonus nested INSIDE the
        # ELSE branch. Case folding is the database's LOWER(), same authority
        # as the rest of the PostgreSQL family.
        try:
            rows = await client.fetch_all(
                """
                SELECT id
                FROM (
                    SELECT id,
                           CASE
                               WHEN LOWER(id) = $3 THEN 1000
                               WHEN LOWER(id) LIKE $4 ESCAPE E'\\\\' THEN 500
                               ELSE 100 - LENGTH(id)
                                    + CASE
                                          WHEN LOWER(id) LIKE $5 ESCAPE E'\\\\'
                                            OR LOWER(id) LIKE $6 ESCAPE E'\\\\'
                                          THEN 50
                                          ELSE 0
                                      END
                           END AS score
                    FROM
                """
                + nodes
                + """
                    WHERE workspace = $1
                      AND namespace = $2
                      AND LOWER(id) LIKE $7 ESCAPE E'\\\\'
                ) scored
                ORDER BY score DESC, id ASC
                LIMIT $8::int
                """,
                self.workspace,
                self.namespace,
                q,
                f"{escaped}%",
                f"% {escaped}%",
                # '_' is a LIKE wildcard, so the word-boundary probe for
                # "_query" must escape it.
                rf"%\_{escaped}%",
                f"%{escaped}%",
                limit,
                descriptor="graph.labels.search",
            )
        except Exception:
            raise HologresGraphError("Hologres graph label search failed") from None
        return [
            _require_str(
                _row_field(row, "id"), "Hologres graph response is corrupt"
            )
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            )
        ]

    # ------------------------------------------------------------------
    # Whole-graph exports
    # ------------------------------------------------------------------

    async def get_all_nodes(self) -> list[dict]:
        client, nodes, _edges = self._ready()
        try:
            rows = await client.fetch_all(
                f"SELECT id, properties FROM {nodes} "
                "WHERE workspace = $1 AND namespace = $2",
                self.workspace,
                self.namespace,
                descriptor="graph.export.nodes",
            )
        except Exception:
            raise HologresGraphError("Hologres graph node read failed") from None
        result = [
            self._node_output(
                _require_str(
                    _row_field(row, "id"), "Hologres graph response is corrupt"
                ),
                _row_field(row, "properties"),
            )
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            )
        ]
        return sorted(result, key=lambda props: props["id"])

    async def get_all_edges(self) -> list[dict]:
        client, _nodes, edges = self._ready()
        try:
            rows = await client.fetch_all(
                f"SELECT src_id, tgt_id, properties FROM {edges} "
                "WHERE workspace = $1 AND namespace = $2",
                self.workspace,
                self.namespace,
                descriptor="graph.export.edges",
            )
        except Exception:
            raise HologresGraphError("Hologres graph edge read failed") from None
        result = [
            {
                **_decode_properties(_row_field(row, "properties")),
                "source": _require_str(
                    _row_field(row, "src_id"),
                    "Hologres graph response is corrupt",
                ),
                "target": _require_str(
                    _row_field(row, "tgt_id"),
                    "Hologres graph response is corrupt",
                ),
            }
            for row in _materialize_rows(
                rows, "Hologres graph response is corrupt"
            )
        ]
        return sorted(result, key=lambda edge: (edge["source"], edge["target"]))

    # ------------------------------------------------------------------
    # Knowledge graph — frontier-capped iterative BFS
    # ------------------------------------------------------------------

    async def _bfs_frontier(
        self, seed: str, max_depth: int, node_budget: int
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Frontier-capped BFS from ``seed``; see PGTableGraphStorage.

        Each hop runs one live-proven CTE that filters visited nodes, ranks
        the level's unvisited neighbours by full-graph degree and caps the
        admitted rows at the remaining budget plus one (the overfetch row
        that lets the caller distinguish "exactly full" from "truncated").
        Returns ``(rows, degrees)`` where degrees cover every collected node
        except the seed — the caller's sort pins the seed ahead of the degree
        term, so its degree is never compared.
        """

        client, nodes, edges = self._ready()
        try:
            seed_row = await client.fetch_one(
                f"SELECT id, properties FROM {nodes} "
                "WHERE workspace = $1 AND namespace = $2 AND id = $3",
                self.workspace,
                self.namespace,
                seed,
                descriptor="graph.kg.seed",
            )
        except Exception:
            raise HologresGraphError("Hologres graph read failed") from None
        if seed_row is None:
            return [], {}
        collected: dict[str, dict[str, Any]] = {
            seed: {
                "id": seed,
                "properties": _row_field(seed_row, "properties"),
                "depth": 0,
            }
        }
        degrees: dict[str, int] = {}
        frontier = [seed]
        depth = 0
        hop_sql = (
            "WITH nb AS ("
            f"SELECT tgt_id AS nid FROM {edges} "
            "WHERE workspace = $1 AND namespace = $2 AND src_id = ANY($3::text[]) "
            "UNION "
            f"SELECT src_id AS nid FROM {edges} "
            "WHERE workspace = $1 AND namespace = $2 AND tgt_id = ANY($3::text[])"
            "), "
            "visited AS (SELECT unnest($4::text[]) AS vid), "
            "candidates AS ("
            "SELECT nb.nid FROM nb "
            "WHERE NOT EXISTS (SELECT 1 FROM visited v WHERE v.vid = nb.nid)"
            "), "
            "candidate_degrees AS ("
            "SELECT id, count(*) AS degree FROM ("
            f"SELECT src_id AS id FROM {edges} "
            "WHERE workspace = $1 AND namespace = $2 "
            "AND src_id IN (SELECT nid FROM candidates) "
            "UNION ALL "
            f"SELECT tgt_id AS id FROM {edges} "
            "WHERE workspace = $1 AND namespace = $2 "
            "AND tgt_id IN (SELECT nid FROM candidates)"
            ") sub GROUP BY id"
            ") "
            "SELECT n.id AS id, n.properties AS properties, "
            "COALESCE(d.degree, 0) AS degree "
            "FROM candidates c "
            f"JOIN {nodes} n "
            "ON n.workspace = $1 AND n.namespace = $2 AND n.id = c.nid "
            "LEFT JOIN candidate_degrees d ON d.id = c.nid "
            "ORDER BY COALESCE(d.degree, 0) DESC, n.id ASC "
            "LIMIT $5::int"
        )
        while frontier and depth < max_depth and len(collected) <= node_budget:
            depth += 1
            level_cap = node_budget - len(collected) + 1
            try:
                rows = await client.fetch_all(
                    hop_sql,
                    self.workspace,
                    self.namespace,
                    frontier,
                    list(collected),
                    level_cap,
                    descriptor="graph.kg.hop",
                )
            except Exception:
                raise HologresGraphError("Hologres graph read failed") from None
            materialized = _materialize_rows(
                rows, "Hologres graph response is corrupt"
            )
            if not materialized:
                break
            next_frontier: list[str] = []
            for row in materialized:
                nid = _require_str(
                    _row_field(row, "id"), "Hologres graph response is corrupt"
                )
                degrees[nid] = _require_int(
                    _row_field(row, "degree"),
                    "Hologres graph response is corrupt",
                )
                collected[nid] = {
                    "id": nid,
                    "properties": _row_field(row, "properties"),
                    "depth": depth,
                }
                next_frontier.append(nid)
            frontier = next_frontier
        return list(collected.values()), degrees

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int | None = None,
    ) -> KnowledgeGraph:
        cap = self.global_config.get("max_graph_nodes", 1000)
        node_budget: int = cap if max_nodes is None else min(max_nodes, cap)

        if node_label == "*":
            client, nodes, edges = self._ready()
            try:
                node_rows_raw = await client.fetch_all(
                    "SELECT n.id AS id, n.properties AS properties, "
                    "COALESCE(d.degree, 0) AS degree "
                    f"FROM {nodes} n "
                    "LEFT JOIN ("
                    "SELECT id, count(*) AS degree FROM ("
                    f"SELECT src_id AS id FROM {edges} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "UNION ALL "
                    f"SELECT tgt_id AS id FROM {edges} "
                    "WHERE workspace = $1 AND namespace = $2"
                    ") sub GROUP BY id"
                    ") d ON d.id = n.id "
                    "WHERE n.workspace = $1 AND n.namespace = $2 "
                    "ORDER BY degree DESC, n.id ASC "
                    "LIMIT $3::int",
                    self.workspace,
                    self.namespace,
                    node_budget + 1,
                    descriptor="graph.kg.wildcard",
                )
            except Exception:
                raise HologresGraphError("Hologres graph read failed") from None
            node_rows = [
                {
                    "id": _require_str(
                        _row_field(row, "id"),
                        "Hologres graph response is corrupt",
                    ),
                    "properties": _row_field(row, "properties"),
                    "degree": _require_int(
                        _row_field(row, "degree"),
                        "Hologres graph response is corrupt",
                    ),
                }
                for row in _materialize_rows(
                    node_rows_raw, "Hologres graph response is corrupt"
                )
            ]
            degrees = {row["id"]: row["degree"] for row in node_rows}
        else:
            node_rows, degrees = await self._bfs_frontier(
                node_label, max_depth, node_budget
            )

        if not node_rows:
            return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

        # Same selection order as the reference: seed pinned first (exact
        # label only), then shallower BFS depth, then higher degree, then id.
        node_rows.sort(
            key=lambda row: (
                node_label != "*" and row["id"] != node_label,
                row.get("depth", 0),
                -degrees.get(row["id"], 0),
                row["id"],
            )
        )

        is_truncated = len(node_rows) > node_budget
        node_rows = node_rows[:node_budget]
        node_ids = {row["id"] for row in node_rows}

        kg_nodes = [
            KnowledgeGraphNode(
                id=row["id"],
                labels=[row["id"]],
                properties=self._node_props(row["id"], row["properties"]),
            )
            for row in node_rows
        ]

        kg_edges: list[KnowledgeGraphEdge] = []
        if node_ids:
            client, _nodes, edges = self._ready()
            try:
                edge_rows = await client.fetch_all(
                    f"SELECT src_id, tgt_id, properties FROM {edges} "
                    "WHERE workspace = $1 AND namespace = $2 "
                    "AND src_id = ANY($3::text[]) AND tgt_id = ANY($3::text[])",
                    self.workspace,
                    self.namespace,
                    list(node_ids),
                    descriptor="graph.kg.edges",
                )
            except Exception:
                raise HologresGraphError("Hologres graph read failed") from None
            decoded_edges = [
                (
                    _require_str(
                        _row_field(row, "src_id"),
                        "Hologres graph response is corrupt",
                    ),
                    _require_str(
                        _row_field(row, "tgt_id"),
                        "Hologres graph response is corrupt",
                    ),
                    _decode_properties(_row_field(row, "properties")),
                )
                for row in _materialize_rows(
                    edge_rows, "Hologres graph response is corrupt"
                )
            ]
            kg_edges = [
                KnowledgeGraphEdge(
                    id=f"{src}-{tgt}",
                    type="DIRECTED",
                    source=src,
                    target=tgt,
                    properties=properties,
                )
                for src, tgt, properties in sorted(
                    decoded_edges, key=lambda edge: (edge[0], edge[1])
                )
            ]

        return KnowledgeGraph(
            nodes=kg_nodes, edges=kg_edges, is_truncated=is_truncated
        )

    # ------------------------------------------------------------------
    # Namespace drop
    # ------------------------------------------------------------------

    async def drop(self) -> dict[str, str]:
        client, nodes, edges = self._ready()
        # Two autocommit statements (Hologres has no cross-statement
        # transactions here); edges first so an interruption leaves isolated
        # nodes, never dangling edges. Both deletes are replay-safe.
        try:
            await client.execute_one(
                f"DELETE FROM {edges} WHERE workspace = $1 AND namespace = $2",
                self.workspace,
                self.namespace,
                descriptor="graph.drop.edges",
                replay_safe=True,
            )
            await client.execute_one(
                f"DELETE FROM {nodes} WHERE workspace = $1 AND namespace = $2",
                self.workspace,
                self.namespace,
                descriptor="graph.drop.nodes",
                replay_safe=True,
            )
        except Exception:
            raise HologresGraphError("Hologres graph drop failed") from None
        return {"status": "success", "message": "data dropped"}
