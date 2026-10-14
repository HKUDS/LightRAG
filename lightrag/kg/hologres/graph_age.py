"""AGE-backed graph storage for the isolated Hologres backend.

Hologres 5.0 embeds Apache AGE 1.5.2 with a reworked surface (frozen against
a live instance): graph and label DDL runs only through the whitelisted
``CALL ag_catalog.hg_age_*`` procedures, labels must be pre-created, cypher
writes are single clauses without RETURN or edge variables and reject bound
parameters (property literals are escaped into the cypher text), reads accept
one protocol-bound agtype parameter map, and agtype values come back as JSON
text. agtype operators resolve only through an ``ag_catalog`` search_path, so
this backend runs on its own dedicated client instead of the shared pool.

The selected graph backend is authoritative. A missing AGE extension is
therefore a startup failure unless ``HOLOGRES_AGE_ALLOW_UNSUPPORTED`` is
explicitly enabled; only then does initialization fall back to the two-table
:class:`~lightrag.kg.hologres.graph.HologresGraphStorage`. Any other probe
failure is indeterminate and always fails initialization, because fallback
could send writes to a different physical store and hide an existing AGE
graph.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import json
import math
import re
from typing import Any, final

from ...base import BaseGraphStorage
from ...namespace import NameSpace
from ...types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ...utils import logger
from .capabilities import (
    ProbeStatus,
    probe_age_graph_capability,
    probe_production_capabilities,
)
from .client import HologresClient, OperationKind, validate_identifier
from .config import HologresConfig
from .graph import HologresGraphStorage
from .workspace import resolve_workspace


_GRAPH_NAME_PREFIX = "lightrag_age_"
_NODE_LABEL = "Entity"
_EDGE_LABEL = "DIRECTED"
_DOLLAR_TAG = "$lightrag_age$"
_ID_CHUNK_SIZE = 200
_PROPERTY_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_ALLOWED_NAMESPACES = frozenset({NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION})


class HologresAGEGraphError(RuntimeError):
    """Raised when an AGE graph operation cannot return a trustworthy result."""


def _agtype_value(value: Any) -> str:
    """Render one scalar property value as a cypher literal.

    Cypher writes reject bound parameters on this server, so values must be
    embedded as literals; anything that cannot be rendered exactly fails
    closed instead of being coerced.
    """

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HologresAGEGraphError("Hologres AGE property is invalid")
        return repr(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f"'{escaped}'"
    raise HologresAGEGraphError("Hologres AGE property is invalid")


def _agtype_map(properties: Mapping[str, Any]) -> str:
    parts = []
    for key in sorted(properties):
        if not isinstance(key, str) or _PROPERTY_KEY.fullmatch(key) is None:
            raise HologresAGEGraphError("Hologres AGE property key is invalid")
        parts.append(f"{key}: {_agtype_value(properties[key])}")
    return "{" + ", ".join(parts) + "}"


def _decode_agtype(value: Any) -> Any:
    if not isinstance(value, str):
        raise HologresAGEGraphError("Hologres AGE response is corrupt")
    try:
        return json.loads(value)
    except ValueError:
        raise HologresAGEGraphError("Hologres AGE response is corrupt") from None


def _decode_string(value: Any) -> str:
    decoded = _decode_agtype(value)
    if not isinstance(decoded, str):
        raise HologresAGEGraphError("Hologres AGE response is corrupt")
    return decoded


def _decode_int(value: Any) -> int:
    decoded = _decode_agtype(value)
    if isinstance(decoded, bool) or not isinstance(decoded, int):
        raise HologresAGEGraphError("Hologres AGE response is corrupt")
    return decoded


def _decode_map(value: Any) -> dict[str, Any]:
    decoded = _decode_agtype(value)
    if not isinstance(decoded, dict):
        raise HologresAGEGraphError("Hologres AGE response is corrupt")
    return decoded


def _materialize_rows(rows: Any) -> list[Any]:
    try:
        return list(rows)
    except (TypeError, ValueError):
        raise HologresAGEGraphError("Hologres AGE response is corrupt") from None


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _params_json(params: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            params, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        raise HologresAGEGraphError("Hologres AGE input is invalid") from None


@final
@dataclass
class HologresAGEGraphStorage(BaseGraphStorage):
    """Graph storage on the Hologres-embedded Apache AGE engine.

    One workspace maps to one AGE graph (``lightrag_age_<workspace>``), which
    is itself a schema on the server, so workspace isolation is physical. If
    the AGE capability probe fails at initialize time, every operation is
    delegated to a two-table ``HologresGraphStorage`` fallback instead.
    """

    config: HologresConfig | None = None
    client: Any | None = None
    _graph: str = field(default="", init=False, repr=False)
    _active_client: Any | None = field(default=None, init=False, repr=False)
    _owns_client: bool = field(default=False, init=False, repr=False)
    _delegate: HologresGraphStorage | None = field(
        default=None, init=False, repr=False
    )
    _initialized: bool = field(default=False, init=False, repr=False)
    _lifecycle_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or self.namespace not in _ALLOWED_NAMESPACES
        ):
            raise ValueError("Unsupported Hologres AGE graph namespace")
        self.workspace = resolve_workspace(
            self.workspace, role="AGE graph"
        )
        graph = f"{_GRAPH_NAME_PREFIX}{self.workspace}"
        try:
            self._graph = validate_identifier(graph)
        except Exception:
            raise ValueError(
                "Hologres AGE graph workspace must form a valid graph name"
            ) from None
        self._lifecycle_lock = asyncio.Lock()

    def __repr__(self) -> str:
        return "HologresAGEGraphStorage(<redacted>)"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Prove the AGE contract, then create the graph or fall back."""

        async with self._lifecycle_lock:
            if self._initialized:
                return

            config = self.config
            if self.client is None and config is None:
                config = HologresConfig.from_env()
                self.config = config
            if self.client is not None and config is None:
                config = getattr(self.client, "config", None)
                if not isinstance(config, HologresConfig):
                    raise HologresAGEGraphError(
                        "Hologres AGE graph configuration is unavailable"
                    )

            actual_client = self.client
            owns_client = False
            if actual_client is None:
                actual_client = HologresClient(
                    replace(config, age_search_path=True)
                )
                owns_client = True

            try:
                await probe_production_capabilities(actual_client)
                probe = await probe_age_graph_capability(actual_client)
                if probe.status is ProbeStatus.PASSED:
                    await self._ensure_graph(actual_client)
                    self._active_client = actual_client
                    self._owns_client = owns_client
                    self._initialized = True
                    return
            except BaseException:
                if owns_client:
                    try:
                        await actual_client.close()
                    except Exception:
                        pass
                raise

            # A FAILED probe splits into two outcomes. An absent AGE extension
            # is definitive, but it still changes the physical schema; make the
            # operator opt in before using the incompatible two-table store.
            # Anything else (transient connection loss, permission problems,
            # unexpected probe behavior) is indeterminate: falling back then
            # would send writes to a different physical store and turn an
            # existing AGE graph invisible, so initialization fails loudly.
            if probe.detail_code == "age_extension_missing":
                if not config.age_allow_unsupported:
                    raise HologresAGEGraphError(
                        "Hologres AGE extension is not available; refusing to "
                        "silently use the two-table backend. Set "
                        "HOLOGRES_AGE_ALLOW_UNSUPPORTED=true only to accept "
                        "the different physical graph schema."
                    )
                logger.warning(
                    "HOLOGRES_AGE_ALLOW_UNSUPPORTED is enabled; falling back "
                    "to the two-table graph backend"
                )
            else:
                raise HologresAGEGraphError(
                    "Hologres AGE graph capability probe failed "
                    f"({probe.detail_code}); refusing to fall back to the "
                    "two-table backend because writes would land in a "
                    "different physical store than an existing AGE graph"
                )
            if owns_client:
                try:
                    await actual_client.close()
                except Exception:
                    pass
            delegate = HologresGraphStorage(
                namespace=self.namespace,
                workspace=self.workspace,
                global_config=self.global_config,
                embedding_func=self.embedding_func,
                config=config,
            )
            await delegate.initialize()
            self._delegate = delegate
            self._initialized = True

    async def _ensure_graph(self, client: Any) -> None:
        graph = self._graph
        try:
            exists = await client.fetch_value(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)",
                graph,
                descriptor="age.graph.check",
            )
            if not exists:
                await client.call_age_procedure(
                    "create_graph",
                    graph,
                    descriptor="age.graph.create",
                    replay_safe=False,
                )
            for procedure, label, descriptor in (
                ("create_vlabel", _NODE_LABEL, "age.graph.vlabel"),
                ("create_elabel", _EDGE_LABEL, "age.graph.elabel"),
            ):
                label_exists = await client.fetch_value(
                    "SELECT EXISTS (SELECT 1 FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = $1 AND c.relname = $2)",
                    graph,
                    label,
                    descriptor="age.graph.label.check",
                )
                if not label_exists:
                    await client.call_age_procedure(
                        procedure,
                        graph,
                        label,
                        descriptor=descriptor,
                        replay_safe=False,
                    )
        except Exception:
            raise HologresAGEGraphError(
                "Hologres AGE graph initialization failed"
            ) from None

    async def finalize(self) -> None:
        async with self._lifecycle_lock:
            delegate = self._delegate
            actual_client = self._active_client
            owns_client = self._owns_client
            self._delegate = None
            self._active_client = None
            self._owns_client = False
            self._initialized = False
            if delegate is not None:
                await delegate.finalize()
                return
            if owns_client and actual_client is not None:
                try:
                    await actual_client.close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise HologresAGEGraphError(
                        "Hologres AGE graph client close failed"
                    ) from None

    def _ready(self) -> tuple[Any, str]:
        if not self._initialized or self._active_client is None:
            raise HologresAGEGraphError(
                "Hologres AGE graph storage is not initialized"
            )
        return self._active_client, self._graph

    # ------------------------------------------------------------------
    # Statement construction
    # ------------------------------------------------------------------

    def _statement(
        self, query: str, columns: str, *, bind_params: bool = False
    ) -> str:
        if _DOLLAR_TAG in query:
            raise HologresAGEGraphError("Hologres AGE input is invalid")
        params = ", $1" if bind_params else ""
        return (
            f"SELECT * FROM ag_catalog.cypher('{self._graph}', "
            f"$lightrag_age$ {query} $lightrag_age${params}) AS ({columns})"
        )

    async def _read(
        self,
        query: str,
        columns: str,
        *,
        descriptor: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        client, _graph = self._ready()
        statement = self._statement(query, columns, bind_params=params is not None)
        values = (_params_json(params),) if params is not None else ()
        try:
            rows = await client.fetch_all(
                statement, *values, descriptor=descriptor
            )
        except Exception:
            raise HologresAGEGraphError(
                "Hologres AGE graph read failed"
            ) from None
        return _materialize_rows(rows)

    async def _write(
        self, query: str, *, descriptor: str, replay_safe: bool
    ) -> None:
        client, _graph = self._ready()
        statement = self._statement(query, "result ag_catalog.agtype")
        try:
            await client.fetch_all(
                statement,
                descriptor=descriptor,
                operation_kind=OperationKind.WRITE,
                replay_safe=replay_safe,
            )
        except Exception:
            raise HologresAGEGraphError(
                "Hologres AGE graph write failed"
            ) from None

    async def index_done_callback(self) -> None:
        if self._delegate is not None:
            return await self._delegate.index_done_callback()
        return None

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        if self._delegate is not None:
            return await self._delegate.has_node(node_id)
        rows = await self._read(
            "MATCH (n:Entity) WHERE n.entity_id = $entity_id RETURN count(n)",
            "found ag_catalog.agtype",
            descriptor="age.node.exists",
            params={"entity_id": node_id},
        )
        return len(rows) == 1 and _decode_int(rows[0][0]) > 0

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        if self._delegate is not None:
            return await self._delegate.get_node(node_id)
        nodes = await self.get_nodes_batch([node_id])
        return nodes.get(node_id)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if self._delegate is not None:
            return await self._delegate.get_nodes_batch(node_ids)
        unique_ids = [
            node_id for node_id in dict.fromkeys(node_ids) if isinstance(node_id, str)
        ]
        if len(unique_ids) != len(dict.fromkeys(node_ids)):
            raise HologresAGEGraphError("Hologres AGE input is invalid")
        result: dict[str, dict] = {}
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            rows = await self._read(
                "MATCH (n:Entity) WHERE n.entity_id IN $entity_ids "
                "RETURN n.entity_id, properties(n)",
                "id ag_catalog.agtype, props ag_catalog.agtype",
                descriptor="age.node.read.batch",
                params={"entity_ids": list(chunk)},
            )
            for row in rows:
                node_id = _decode_string(row[0])
                properties = _decode_map(row[1])
                properties["entity_id"] = node_id
                result[node_id] = properties
        return result

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        if self._delegate is not None:
            return await self._delegate.has_nodes_batch(node_ids)
        found = await self.get_nodes_batch(node_ids)
        return set(found)

    def _node_write_queries(
        self, node_id: str, node_data: dict[str, str]
    ) -> tuple[str, str]:
        if not isinstance(node_data, Mapping) or "entity_id" not in node_data:
            raise ValueError(
                "Hologres: node properties must contain an 'entity_id' field"
            )
        if not isinstance(node_id, str):
            raise HologresAGEGraphError("Hologres AGE input is invalid")
        properties = dict(node_data)
        properties["entity_id"] = node_id
        anchor = _agtype_map({"entity_id": node_id})
        merge = f"MERGE (n:Entity {anchor})"
        # SET += merges like the two-table properties || update, and always
        # carries entity_id = node_id.
        update = (
            f"MATCH (n:Entity {anchor}) SET n += {_agtype_map(properties)}"
        )
        return merge, update

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        if self._delegate is not None:
            return await self._delegate.upsert_node(node_id, node_data)
        merge, update = self._node_write_queries(node_id, node_data)
        await self._write(merge, descriptor="age.node.merge", replay_safe=True)
        await self._write(update, descriptor="age.node.set", replay_safe=True)

    async def upsert_nodes_batch(
        self, nodes: list[tuple[str, dict[str, str]]]
    ) -> None:
        if self._delegate is not None:
            return await self._delegate.upsert_nodes_batch(nodes)
        for node_id, node_data in nodes:
            await self.upsert_node(node_id, node_data)

    async def delete_node(self, node_id: str) -> None:
        if self._delegate is not None:
            return await self._delegate.delete_node(node_id)
        await self.remove_nodes([node_id])

    async def remove_nodes(self, nodes: list[str]) -> None:
        if self._delegate is not None:
            return await self._delegate.remove_nodes(nodes)
        if not nodes:
            return
        if not all(isinstance(node_id, str) for node_id in nodes):
            raise HologresAGEGraphError("Hologres AGE input is invalid")
        unique_ids = list(dict.fromkeys(nodes))
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            # Write parameters are unsupported, so the id list is embedded as
            # escaped literals.
            id_list = ", ".join(_agtype_value(node_id) for node_id in chunk)
            await self._write(
                f"MATCH (n:Entity) WHERE n.entity_id IN [{id_list}] "
                "DETACH DELETE n",
                descriptor="age.node.delete",
                replay_safe=True,
            )

    # ------------------------------------------------------------------
    # Edge operations — undirected pairs stored once in canonical order
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_pair(
        source_node_id: str, target_node_id: str
    ) -> tuple[str, str]:
        if not isinstance(source_node_id, str) or not isinstance(
            target_node_id, str
        ):
            raise HologresAGEGraphError("Hologres AGE input is invalid")
        return (
            min(source_node_id, target_node_id),
            max(source_node_id, target_node_id),
        )

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        if self._delegate is not None:
            return await self._delegate.has_edge(source_node_id, target_node_id)
        edge = await self.get_edge(source_node_id, target_node_id)
        return edge is not None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        if self._delegate is not None:
            return await self._delegate.get_edge(source_node_id, target_node_id)
        src, tgt = self._canonical_pair(source_node_id, target_node_id)
        rows = await self._read(
            "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
            "WHERE a.entity_id = $src AND b.entity_id = $tgt "
            "RETURN properties(r)",
            "props ag_catalog.agtype",
            descriptor="age.edge.read",
            params={"src": src, "tgt": tgt},
        )
        if not rows:
            return None
        return _decode_map(rows[0][0])

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        if self._delegate is not None:
            return await self._delegate.upsert_edge(
                source_node_id, target_node_id, edge_data
            )
        if not isinstance(edge_data, Mapping):
            raise HologresAGEGraphError("Hologres AGE input is invalid")
        src, tgt = self._canonical_pair(source_node_id, target_node_id)
        src_anchor = _agtype_map({"entity_id": src})
        tgt_anchor = _agtype_map({"entity_id": tgt})
        # Auto-create missing endpoints as stubs (MERGE keeps existing
        # properties); a crash here leaves a stub node without an edge, which
        # a retry repairs — the same residue the two-table backend accepts.
        # The edge row itself is one statement: MERGE and its property
        # REPLACE run inside a single cypher call, so an interruption can
        # never expose an edge without its properties (a property-less edge
        # would carry no evidence and violate the relation weight contract).
        await self._write(
            f"MERGE (n:Entity {src_anchor})",
            descriptor="age.edge.endpoint",
            replay_safe=True,
        )
        if tgt != src:
            await self._write(
                f"MERGE (n:Entity {tgt_anchor})",
                descriptor="age.edge.endpoint",
                replay_safe=True,
            )
        await self._write(
            f"MATCH (a:Entity {src_anchor}), (b:Entity {tgt_anchor}) "
            "MERGE (a)-[r:DIRECTED]->(b) "
            f"SET r = {_agtype_map(dict(edge_data))}",
            descriptor="age.edge.upsert",
            replay_safe=True,
        )

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        if self._delegate is not None:
            return await self._delegate.upsert_edges_batch(edges)
        deduped: dict[tuple[str, str], tuple[str, str, dict[str, str]]] = {}
        for source_node_id, target_node_id, edge_data in edges:
            pair = self._canonical_pair(source_node_id, target_node_id)
            deduped[pair] = (source_node_id, target_node_id, edge_data)
        for source_node_id, target_node_id, edge_data in deduped.values():
            await self.upsert_edge(source_node_id, target_node_id, edge_data)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if self._delegate is not None:
            return await self._delegate.remove_edges(edges)
        if not edges:
            return
        if not all(
            isinstance(edge[0], str) and isinstance(edge[1], str) for edge in edges
        ):
            raise HologresAGEGraphError("Edge node IDs must be non-None strings")
        pairs = list(
            dict.fromkeys(self._canonical_pair(*edge) for edge in edges)
        )
        for src, tgt in pairs:
            src_anchor = _agtype_map({"entity_id": src})
            tgt_anchor = _agtype_map({"entity_id": tgt})
            await self._write(
                f"MATCH (a:Entity {src_anchor})-[r:DIRECTED]->"
                f"(b:Entity {tgt_anchor}) DELETE r",
                descriptor="age.edge.delete",
                replay_safe=True,
            )

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        if self._delegate is not None:
            return await self._delegate.get_edges_batch(pairs)
        if not pairs:
            return {}
        requested: list[tuple[str, str]] = []
        for pair in pairs:
            src = pair.get("src") if isinstance(pair, Mapping) else None
            tgt = pair.get("tgt") if isinstance(pair, Mapping) else None
            if not isinstance(src, str) or not isinstance(tgt, str):
                raise HologresAGEGraphError("Hologres AGE input is invalid")
            requested.append((src, tgt))
        canonical = list(
            dict.fromkeys(self._canonical_pair(s, t) for s, t in requested)
        )
        wanted = set(canonical)
        found: dict[tuple[str, str], dict] = {}
        endpoints = list(
            dict.fromkeys(endpoint for pair in canonical for endpoint in pair)
        )
        for chunk in _chunks(endpoints, _ID_CHUNK_SIZE):
            rows = await self._read(
                "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
                "WHERE a.entity_id IN $entity_ids "
                "RETURN a.entity_id, b.entity_id, properties(r)",
                "src ag_catalog.agtype, tgt ag_catalog.agtype, "
                "props ag_catalog.agtype",
                descriptor="age.edge.read.batch",
                params={"entity_ids": list(chunk)},
            )
            for row in rows:
                src = _decode_string(row[0])
                tgt = _decode_string(row[1])
                if (src, tgt) in wanted:
                    found[(src, tgt)] = _decode_map(row[2])
        result: dict[tuple[str, str], dict] = {}
        for src, tgt in requested:
            properties = found.get(self._canonical_pair(src, tgt))
            if properties is not None:
                result[(src, tgt)] = properties
        return result

    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]] | None:
        if self._delegate is not None:
            return await self._delegate.get_node_edges(source_node_id)
        adjacency = await self.get_nodes_edges_batch([source_node_id])
        edges = adjacency.get(source_node_id, [])
        if not edges and not await self.has_node(source_node_id):
            return None
        return edges

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if self._delegate is not None:
            return await self._delegate.get_nodes_edges_batch(node_ids)
        if not node_ids:
            return {}
        unique_ids = list(dict.fromkeys(node_ids))
        result: dict[str, list[tuple[str, str]]] = {
            node_id: [] for node_id in unique_ids
        }
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            # The undirected match yields a self-loop once, matching the
            # two-table adjacency rule (once in the list, twice in degree).
            rows = await self._read(
                "MATCH (n:Entity)-[r:DIRECTED]-(m:Entity) "
                "WHERE n.entity_id IN $entity_ids "
                "RETURN n.entity_id, m.entity_id",
                "id ag_catalog.agtype, peer ag_catalog.agtype",
                descriptor="age.edge.adjacency",
                params={"entity_ids": list(chunk)},
            )
            for row in rows:
                node_id = _decode_string(row[0])
                peer = _decode_string(row[1])
                if node_id in result:
                    result[node_id].append((node_id, peer))
        for adjacency in result.values():
            adjacency.sort(key=lambda edge: edge[1])
        return result

    # ------------------------------------------------------------------
    # Degrees — a self-loop counts twice, matching networkx; the undirected
    # cypher match sees it once, so a directed self-loop count compensates.
    # ------------------------------------------------------------------

    async def _degree_map(self, node_ids: list[str]) -> dict[str, int]:
        result: dict[str, int] = {node_id: 0 for node_id in node_ids}
        for chunk in _chunks(list(result), _ID_CHUNK_SIZE):
            base_rows = await self._read(
                "MATCH (n:Entity) WHERE n.entity_id IN $entity_ids "
                "OPTIONAL MATCH (n)-[r:DIRECTED]-() "
                "RETURN n.entity_id, count(r)",
                "id ag_catalog.agtype, degree ag_catalog.agtype",
                descriptor="age.degree.base",
                params={"entity_ids": list(chunk)},
            )
            for row in base_rows:
                node_id = _decode_string(row[0])
                if node_id in result:
                    result[node_id] = _decode_int(row[1])
            loop_rows = await self._read(
                "MATCH (n:Entity)-[r:DIRECTED]->(n) "
                "WHERE n.entity_id IN $entity_ids "
                "RETURN n.entity_id, count(r)",
                "id ag_catalog.agtype, loops ag_catalog.agtype",
                descriptor="age.degree.loops",
                params={"entity_ids": list(chunk)},
            )
            for row in loop_rows:
                node_id = _decode_string(row[0])
                if node_id in result:
                    result[node_id] += _decode_int(row[1])
        return result

    async def node_degree(self, node_id: str) -> int:
        if self._delegate is not None:
            return await self._delegate.node_degree(node_id)
        degrees = await self._degree_map([node_id])
        return degrees.get(node_id, 0)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        if self._delegate is not None:
            return await self._delegate.edge_degree(src_id, tgt_id)
        degrees = await self.node_degrees_batch([src_id, tgt_id])
        return degrees.get(src_id, 0) + degrees.get(tgt_id, 0)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if self._delegate is not None:
            return await self._delegate.node_degrees_batch(node_ids)
        if not node_ids:
            return {}
        return await self._degree_map(list(dict.fromkeys(node_ids)))

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        if self._delegate is not None:
            return await self._delegate.edge_degrees_batch(edge_pairs)
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

    async def _all_label_ids(self, descriptor: str) -> list[str]:
        rows = await self._read(
            "MATCH (n:Entity) RETURN n.entity_id",
            "id ag_catalog.agtype",
            descriptor=descriptor,
        )
        return [_decode_string(row[0]) for row in rows]

    async def get_all_labels(self) -> list[str]:
        if self._delegate is not None:
            return await self._delegate.get_all_labels()
        return sorted(await self._all_label_ids("age.labels.all"))

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        if self._delegate is not None:
            return await self._delegate.get_popular_labels(limit)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise HologresAGEGraphError("Hologres AGE graph limit is invalid")
        labels = await self._all_label_ids("age.labels.popular")
        degrees = await self._degree_map(list(dict.fromkeys(labels)))
        ranked = sorted(labels, key=lambda label: (-degrees.get(label, 0), label))
        return ranked[:limit]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        if self._delegate is not None:
            return await self._delegate.search_labels(query, limit)
        if not isinstance(query, str):
            raise HologresAGEGraphError("Hologres AGE input is invalid")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise HologresAGEGraphError("Hologres AGE graph limit is invalid")
        q = query.strip().lower()
        if not q:
            return []
        rows = await self._read(
            "MATCH (n:Entity) WHERE toLower(n.entity_id) CONTAINS $needle "
            "RETURN n.entity_id",
            "id ag_catalog.agtype",
            descriptor="age.labels.search",
            params={"needle": q},
        )
        candidates = [_decode_string(row[0]) for row in rows]

        # Mirrors the two-table scoring: exact 1000, prefix 500, else
        # 100 - len(id) with a +50 word-boundary bonus inside the ELSE branch.
        def _score(label: str) -> int:
            lowered = label.lower()
            if lowered == q:
                return 1000
            if lowered.startswith(q):
                return 500
            bonus = 50 if (f" {q}" in lowered or f"_{q}" in lowered) else 0
            return 100 - len(label) + bonus

        ranked = sorted(candidates, key=lambda label: (-_score(label), label))
        return ranked[:limit]

    # ------------------------------------------------------------------
    # Whole-graph exports
    # ------------------------------------------------------------------

    async def get_all_nodes(self) -> list[dict]:
        if self._delegate is not None:
            return await self._delegate.get_all_nodes()
        rows = await self._read(
            "MATCH (n:Entity) RETURN n.entity_id, properties(n)",
            "id ag_catalog.agtype, props ag_catalog.agtype",
            descriptor="age.export.nodes",
        )
        result = []
        for row in rows:
            node_id = _decode_string(row[0])
            properties = _decode_map(row[1])
            properties["entity_id"] = node_id
            properties["id"] = node_id
            result.append(properties)
        return sorted(result, key=lambda props: props["id"])

    async def get_all_edges(self) -> list[dict]:
        if self._delegate is not None:
            return await self._delegate.get_all_edges()
        rows = await self._read(
            "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
            "RETURN a.entity_id, b.entity_id, properties(r)",
            "src ag_catalog.agtype, tgt ag_catalog.agtype, "
            "props ag_catalog.agtype",
            descriptor="age.export.edges",
        )
        result = []
        for row in rows:
            properties = _decode_map(row[2])
            properties["source"] = _decode_string(row[0])
            properties["target"] = _decode_string(row[1])
            result.append(properties)
        return sorted(result, key=lambda edge: (edge["source"], edge["target"]))

    # ------------------------------------------------------------------
    # Knowledge graph — frontier-capped iterative BFS
    # ------------------------------------------------------------------

    async def _node_rows_for(
        self, node_ids: list[str], depth_map: dict[str, int]
    ) -> list[dict[str, Any]]:
        nodes = await self.get_nodes_batch(node_ids)
        return [
            {
                "id": node_id,
                "properties": properties,
                "depth": depth_map.get(node_id, 0),
            }
            for node_id, properties in nodes.items()
        ]

    async def _bfs_frontier(
        self, seed: str, max_depth: int, node_budget: int
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        if not await self.has_node(seed):
            return [], {}
        depth_map: dict[str, int] = {seed: 0}
        collected = [seed]
        frontier = [seed]
        depth = 0
        degrees: dict[str, int] = {}
        while frontier and depth < max_depth and len(collected) <= node_budget:
            depth += 1
            neighbours: list[str] = []
            for chunk in _chunks(frontier, _ID_CHUNK_SIZE):
                rows = await self._read(
                    "MATCH (n:Entity)-[r:DIRECTED]-(m:Entity) "
                    "WHERE n.entity_id IN $entity_ids "
                    "RETURN DISTINCT m.entity_id",
                    "peer ag_catalog.agtype",
                    descriptor="age.kg.hop",
                    params={"entity_ids": list(chunk)},
                )
                neighbours.extend(_decode_string(row[0]) for row in rows)
            candidates = [
                node_id
                for node_id in dict.fromkeys(neighbours)
                if node_id not in depth_map
            ]
            if not candidates:
                break
            candidate_degrees = await self._degree_map(candidates)
            level_cap = node_budget - len(collected) + 1
            admitted = sorted(
                candidates,
                key=lambda node_id: (-candidate_degrees.get(node_id, 0), node_id),
            )[:level_cap]
            for node_id in admitted:
                depth_map[node_id] = depth
                degrees[node_id] = candidate_degrees.get(node_id, 0)
                collected.append(node_id)
            frontier = admitted
        node_rows = await self._node_rows_for(collected, depth_map)
        return node_rows, degrees

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int | None = None,
    ) -> KnowledgeGraph:
        if self._delegate is not None:
            return await self._delegate.get_knowledge_graph(
                node_label, max_depth, max_nodes
            )
        cap = self.global_config.get("max_graph_nodes", 1000)
        node_budget: int = cap if max_nodes is None else min(max_nodes, cap)

        if node_label == "*":
            labels = await self._all_label_ids("age.kg.wildcard")
            unique_labels = list(dict.fromkeys(labels))
            degrees = await self._degree_map(unique_labels)
            ranked = sorted(
                unique_labels,
                key=lambda label: (-degrees.get(label, 0), label),
            )[: node_budget + 1]
            node_rows = await self._node_rows_for(ranked, {})
        else:
            node_rows, degrees = await self._bfs_frontier(
                node_label, max_depth, node_budget
            )

        if not node_rows:
            return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

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
                properties=row["properties"],
            )
            for row in node_rows
        ]

        kg_edges: list[KnowledgeGraphEdge] = []
        if node_ids:
            decoded_edges: list[tuple[str, str, dict[str, Any]]] = []
            selected_ids = sorted(node_ids)
            # Chunk only canonical sources so cross-chunk edges remain visible once.
            for chunk in _chunks(selected_ids, _ID_CHUNK_SIZE):
                rows = await self._read(
                    "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
                    "WHERE a.entity_id IN $source_ids "
                    "AND b.entity_id IN $target_ids "
                    "RETURN a.entity_id, b.entity_id, properties(r)",
                    "src ag_catalog.agtype, tgt ag_catalog.agtype, "
                    "props ag_catalog.agtype",
                    descriptor="age.kg.edges",
                    params={
                        "source_ids": list(chunk),
                        "target_ids": selected_ids,
                    },
                )
                decoded_edges.extend(
                    (
                        _decode_string(row[0]),
                        _decode_string(row[1]),
                        _decode_map(row[2]),
                    )
                    for row in rows
                )
            seen_pairs: set[tuple[str, str]] = set()
            kg_edges = []
            for src, tgt, properties in sorted(
                decoded_edges, key=lambda edge: (edge[0], edge[1])
            ):
                if (src, tgt) in seen_pairs:
                    continue
                if src not in node_ids or tgt not in node_ids:
                    continue
                seen_pairs.add((src, tgt))
                kg_edges.append(
                    KnowledgeGraphEdge(
                        id=f"{src}-{tgt}",
                        type="DIRECTED",
                        source=src,
                        target=tgt,
                        properties=properties,
                    )
                )

        return KnowledgeGraph(
            nodes=kg_nodes, edges=kg_edges, is_truncated=is_truncated
        )

    # ------------------------------------------------------------------
    # Workspace drop
    # ------------------------------------------------------------------

    async def drop(self) -> dict[str, str]:
        if self._delegate is not None:
            return await self._delegate.drop()
        await self._write(
            "MATCH (n:Entity) DETACH DELETE n",
            descriptor="age.drop",
            replay_safe=True,
        )
        return {"status": "success", "message": "data dropped"}
