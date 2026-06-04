"""
PostgreSQL + Recursive CTE graph storage for LightRAG.

Implements BaseGraphStorage using plain PostgreSQL tables and recursive
CTEs — no Apache AGE extension, no pgvector, no Cypher wrapper required.

Schema (created automatically on initialize):

    lightrag_graph_nodes(workspace TEXT, id TEXT, properties JSONB)
                                                  PK(workspace, id)
    lightrag_graph_edges(workspace TEXT, src_id TEXT, tgt_id TEXT, properties JSONB)
        Edges are stored in canonical order: src_id = LEAST(a, b), tgt_id = GREATEST(a, b)
                                                  PK(workspace, src_id, tgt_id)

Configuration: inherits all POSTGRES_* / POSTGRES_SSL_* / POSTGRES_WORKSPACE
environment variables via the shared ClientManager / PostgreSQLDB pool, identical
to PGKVStorage and PGVectorStorage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..base import BaseGraphStorage
from ..kg.shared_storage import get_data_init_lock
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger
from .postgres_impl import ClientManager, PostgreSQLDB

# ---------------------------------------------------------------------------
# DDL — executed once per process on first initialize()
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS lightrag_graph_nodes (
    workspace   TEXT        NOT NULL,
    id          TEXT        NOT NULL,
    properties  JSONB       NOT NULL DEFAULT '{}',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace, id)
);

CREATE TABLE IF NOT EXISTS lightrag_graph_edges (
    workspace   TEXT        NOT NULL,
    src_id      TEXT        NOT NULL,
    tgt_id      TEXT        NOT NULL,
    properties  JSONB       NOT NULL DEFAULT '{}',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace, src_id, tgt_id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_graph_edges_tgt
    ON lightrag_graph_edges (workspace, tgt_id);
"""

# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------


@dataclass
class PgRcteGraphStorage(BaseGraphStorage):
    """LightRAG graph storage backed by PostgreSQL + Recursive CTEs.

    Drop-in replacement for PGGraphStorage that requires no Apache AGE
    extension.  Uses two plain tables (lightrag_graph_nodes,
    lightrag_graph_edges) with JSONB properties and B-tree indexes.

    Edges are stored in canonical order: src_id = LEAST(a, b),
    tgt_id = GREATEST(a, b).  All write paths normalise before INSERT so
    upsert_edge(A, B) and upsert_edge(B, A) map to the same row.
    """

    db: PostgreSQLDB | None = field(default=None, init=False, repr=False)

    @property
    def _db(self) -> PostgreSQLDB:
        if self.db is None:
            raise RuntimeError(
                "PgRcteGraphStorage not initialized — call initialize() first"
            )
        return self.db

    # ------------------------------------------------------------------
    # Query helpers — thin wrappers over PostgreSQLDB.query / execute
    # ------------------------------------------------------------------

    async def _execute(self, sql: str, *args: Any) -> None:
        data = {str(i): v for i, v in enumerate(args)} if args else None
        await self._db.execute(sql, data=data)

    async def _fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        result = await self._db.query(sql, list(args) if args else None)
        return result if isinstance(result, dict) else None  # type: ignore[return-value]

    async def _fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        result = await self._db.query(sql, list(args) if args else None, multirows=True)
        return result if isinstance(result, list) else []  # type: ignore[return-value]

    async def _fetchval(self, sql: str, *args: Any) -> Any:
        result = await self._db.query(sql, list(args) if args else None)
        if not isinstance(result, dict):
            return None
        return next(iter(result.values()))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client(
                    vector_storage=self.global_config.get("vector_storage")
                )
                # Workspace priority: POSTGRES_WORKSPACE env > self.workspace > "default"
                if self.db.workspace:
                    self.workspace = self.db.workspace
                elif not getattr(self, "workspace", None):
                    self.workspace = "default"
                await self._db.execute(_DDL)
        logger.info(f"[{self.workspace}] PgRcteGraphStorage initialized")

    async def finalize(self) -> None:
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            logger.info(f"[{self.workspace}] PgRcteGraphStorage finalized")

    async def index_done_callback(self) -> None:
        pass  # PostgreSQL writes are immediately durable

    async def drop(self) -> dict[str, str]:
        try:
            await self._execute(
                "DELETE FROM lightrag_graph_edges WHERE workspace = $1", self.workspace
            )
            await self._execute(
                "DELETE FROM lightrag_graph_nodes WHERE workspace = $1", self.workspace
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        row = await self._fetchrow(
            "SELECT 1 FROM lightrag_graph_nodes WHERE workspace = $1 AND id = $2",
            self.workspace,
            node_id,
        )
        return row is not None

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        row = await self._fetchrow(
            "SELECT properties FROM lightrag_graph_nodes WHERE workspace = $1 AND id = $2",
            self.workspace,
            node_id,
        )
        return json.loads(row["properties"]) if row else None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        await self._execute(
            """
            INSERT INTO lightrag_graph_nodes (workspace, id, properties, updated_at)
            VALUES ($1, $2, $3, now())
            ON CONFLICT (workspace, id)
            DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
            """,
            self.workspace,
            node_id,
            json.dumps(node_data),
        )

    async def delete_node(self, node_id: str) -> None:
        await self._execute(
            "DELETE FROM lightrag_graph_edges WHERE workspace=$1 AND (src_id=$2 OR tgt_id=$2)",
            self.workspace,
            node_id,
        )
        await self._execute(
            "DELETE FROM lightrag_graph_nodes WHERE workspace = $1 AND id = $2",
            self.workspace,
            node_id,
        )

    async def remove_nodes(self, nodes: list[str]) -> None:
        if not nodes:
            return
        await self._execute(
            "DELETE FROM lightrag_graph_edges WHERE workspace=$1 AND (src_id=ANY($2) OR tgt_id=ANY($2))",
            self.workspace,
            nodes,
        )
        await self._execute(
            "DELETE FROM lightrag_graph_nodes WHERE workspace = $1 AND id = ANY($2)",
            self.workspace,
            nodes,
        )

    # ------------------------------------------------------------------
    # Edge operations
    # All writes normalise to canonical order: src_id = min(a, b), tgt_id = max(a, b).
    # ------------------------------------------------------------------

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        row = await self._fetchrow(
            "SELECT 1 FROM lightrag_graph_edges WHERE workspace=$1 AND src_id=$2 AND tgt_id=$3",
            self.workspace,
            src,
            tgt,
        )
        return row is not None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        row = await self._fetchrow(
            "SELECT properties FROM lightrag_graph_edges WHERE workspace=$1 AND src_id=$2 AND tgt_id=$3",
            self.workspace,
            src,
            tgt,
        )
        return json.loads(row["properties"]) if row else None

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        await self._execute(
            """
            INSERT INTO lightrag_graph_edges (workspace, src_id, tgt_id, properties, updated_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (workspace, src_id, tgt_id)
            DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
            """,
            self.workspace,
            src,
            tgt,
            json.dumps(edge_data),
        )

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return
        srcs = [min(e[0], e[1]) for e in edges]
        tgts = [max(e[0], e[1]) for e in edges]
        await self._execute(
            """
            DELETE FROM lightrag_graph_edges
            WHERE workspace = $1
              AND (src_id, tgt_id) IN (SELECT * FROM unnest($2::text[], $3::text[]))
            """,
            self.workspace,
            srcs,
            tgts,
        )

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if not await self.has_node(source_node_id):
            return None
        rows = await self._fetch(
            """
            SELECT src_id, tgt_id FROM lightrag_graph_edges
            WHERE workspace = $1 AND (src_id = $2 OR tgt_id = $2)
            """,
            self.workspace,
            source_node_id,
        )
        return [(r["src_id"], r["tgt_id"]) for r in rows]

    # ------------------------------------------------------------------
    # Degree queries
    # ------------------------------------------------------------------

    async def node_degree(self, node_id: str) -> int:
        val = await self._fetchval(
            "SELECT COUNT(*) FROM lightrag_graph_edges WHERE workspace=$1 AND (src_id=$2 OR tgt_id=$2)",
            self.workspace,
            node_id,
        )
        return int(val or 0)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        import asyncio

        s, t = await asyncio.gather(self.node_degree(src_id), self.node_degree(tgt_id))
        return s + t

    # ------------------------------------------------------------------
    # Label queries
    # ------------------------------------------------------------------

    async def get_all_labels(self) -> list[str]:
        rows = await self._fetch(
            "SELECT id FROM lightrag_graph_nodes WHERE workspace = $1 ORDER BY id",
            self.workspace,
        )
        return [r["id"] for r in rows]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        rows = await self._fetch(
            """
            SELECT id, COUNT(*) AS degree
            FROM (
                SELECT src_id AS id FROM lightrag_graph_edges WHERE workspace = $1
                UNION ALL
                SELECT tgt_id AS id FROM lightrag_graph_edges WHERE workspace = $1
            ) sub
            GROUP BY id
            ORDER BY degree DESC
            LIMIT $2
            """,
            self.workspace,
            limit,
        )
        return [r["id"] for r in rows]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        rows = await self._fetch(
            "SELECT id FROM lightrag_graph_nodes WHERE workspace=$1 AND id ILIKE $2 ORDER BY id LIMIT $3",
            self.workspace,
            f"%{query}%",
            limit,
        )
        return [r["id"] for r in rows]

    async def get_all_nodes(self) -> list[dict]:
        rows = await self._fetch(
            "SELECT id, properties FROM lightrag_graph_nodes WHERE workspace = $1",
            self.workspace,
        )
        return [{"id": r["id"], **json.loads(r["properties"])} for r in rows]

    async def get_all_edges(self) -> list[dict]:
        rows = await self._fetch(
            "SELECT src_id, tgt_id, properties FROM lightrag_graph_edges WHERE workspace = $1",
            self.workspace,
        )
        return [
            {
                "src_id": r["src_id"],
                "tgt_id": r["tgt_id"],
                **json.loads(r["properties"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Knowledge graph — Recursive CTE BFS
    # ------------------------------------------------------------------

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        # Fetch one extra row to detect true truncation without a separate COUNT.
        fetch_limit = max_nodes + 1
        if node_label == "*":
            # $1=workspace  $2=max_depth  $3=fetch_limit
            rcte_sql = """
            WITH RECURSIVE bfs(id, properties, depth) AS (
                SELECT id, properties, 0
                FROM lightrag_graph_nodes
                WHERE workspace = $1
                UNION
                SELECT next_n.id, next_n.properties, b.depth + 1
                FROM bfs b
                JOIN lightrag_graph_edges e
                  ON e.workspace = $1 AND (e.src_id = b.id OR e.tgt_id = b.id)
                JOIN lightrag_graph_nodes next_n
                  ON next_n.workspace = $1
                 AND next_n.id = CASE WHEN e.src_id = b.id THEN e.tgt_id ELSE e.src_id END
                WHERE b.depth < $2
            )
            SELECT DISTINCT id, properties FROM bfs LIMIT $3
            """
            params: list[Any] = [self.workspace, max_depth, fetch_limit]
        else:
            # $1=workspace  $2=label pattern  $3=max_depth  $4=fetch_limit
            rcte_sql = """
            WITH RECURSIVE bfs(id, properties, depth) AS (
                SELECT id, properties, 0
                FROM lightrag_graph_nodes
                WHERE workspace = $1 AND id ILIKE $2
                UNION
                SELECT next_n.id, next_n.properties, b.depth + 1
                FROM bfs b
                JOIN lightrag_graph_edges e
                  ON e.workspace = $1 AND (e.src_id = b.id OR e.tgt_id = b.id)
                JOIN lightrag_graph_nodes next_n
                  ON next_n.workspace = $1
                 AND next_n.id = CASE WHEN e.src_id = b.id THEN e.tgt_id ELSE e.src_id END
                WHERE b.depth < $3
            )
            SELECT DISTINCT id, properties FROM bfs LIMIT $4
            """
            params = [self.workspace, f"%{node_label}%", max_depth, fetch_limit]

        node_rows = await self._fetch(rcte_sql, *params)
        is_truncated = len(node_rows) > max_nodes
        node_rows = node_rows[:max_nodes]
        node_ids = {r["id"] for r in node_rows}

        # Sort by degree descending to match AGE backend behaviour under truncation.
        if node_ids:
            degrees = await self.node_degrees_batch(list(node_ids))
            node_rows.sort(key=lambda r: degrees.get(r["id"], 0), reverse=True)

        nodes = [
            KnowledgeGraphNode(
                id=r["id"], labels=[r["id"]], properties=json.loads(r["properties"])
            )
            for r in node_rows
        ]

        edges: list[KnowledgeGraphEdge] = []
        if node_ids:
            edge_rows = await self._fetch(
                """
                SELECT src_id, tgt_id, properties FROM lightrag_graph_edges
                WHERE workspace = $1 AND src_id = ANY($2) AND tgt_id = ANY($2)
                """,
                self.workspace,
                list(node_ids),
            )
            edges = [
                KnowledgeGraphEdge(
                    id=f"{r['src_id']}->{r['tgt_id']}",
                    type=None,
                    source=r["src_id"],
                    target=r["tgt_id"],
                    properties=json.loads(r["properties"]),
                )
                for r in edge_rows
            ]

        return KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=is_truncated)

    # ------------------------------------------------------------------
    # Read batch overrides — avoid N+1 serial calls
    # ------------------------------------------------------------------

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        rows = await self._fetch(
            "SELECT id, properties FROM lightrag_graph_nodes WHERE workspace=$1 AND id=ANY($2)",
            self.workspace,
            node_ids,
        )
        return {r["id"]: json.loads(r["properties"]) for r in rows}

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        rows = await self._fetch(
            """
            SELECT id, COUNT(*) AS degree
            FROM (
                SELECT src_id AS id FROM lightrag_graph_edges WHERE workspace=$1 AND src_id=ANY($2)
                UNION ALL
                SELECT tgt_id AS id FROM lightrag_graph_edges WHERE workspace=$1 AND tgt_id=ANY($2)
            ) sub
            GROUP BY id
            """,
            self.workspace,
            node_ids,
        )
        result = {nid: 0 for nid in node_ids}
        for r in rows:
            result[r["id"]] = int(r["degree"])
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        if not edge_pairs:
            return {}
        all_ids = list({nid for pair in edge_pairs for nid in pair})
        degrees = await self.node_degrees_batch(all_ids)
        return {(s, t): degrees.get(s, 0) + degrees.get(t, 0) for s, t in edge_pairs}

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}
        canonical = [(min(p["src"], p["tgt"]), max(p["src"], p["tgt"])) for p in pairs]
        srcs = [c[0] for c in canonical]
        tgts = [c[1] for c in canonical]
        rows = await self._fetch(
            """
            SELECT src_id, tgt_id, properties FROM lightrag_graph_edges
            WHERE workspace=$1
              AND (src_id, tgt_id) IN (SELECT * FROM unnest($2::text[], $3::text[]))
            """,
            self.workspace,
            srcs,
            tgts,
        )
        canonical_props = {
            (r["src_id"], r["tgt_id"]): json.loads(r["properties"]) for r in rows
        }
        return {
            (p["src"], p["tgt"]): canonical_props[c]
            for p, c in zip(pairs, canonical)
            if c in canonical_props
        }

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}
        rows = await self._fetch(
            """
            SELECT src_id, tgt_id FROM lightrag_graph_edges
            WHERE workspace=$1 AND (src_id=ANY($2) OR tgt_id=ANY($2))
            """,
            self.workspace,
            node_ids,
        )
        result: dict[str, list[tuple[str, str]]] = {nid: [] for nid in node_ids}
        for r in rows:
            src, tgt = r["src_id"], r["tgt_id"]
            if src in result:
                result[src].append((src, tgt))
            if tgt in result:
                result[tgt].append((src, tgt))
        return result

    # ------------------------------------------------------------------
    # Write batch overrides — single-roundtrip bulk upserts via unnest
    # ------------------------------------------------------------------

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        if not node_ids:
            return set()
        return set(await self.get_nodes_batch(node_ids))

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        if not nodes:
            return
        # Deduplicate: last write for each node_id wins.
        deduped: dict[str, dict[str, str]] = {}
        for node_id, node_data in nodes:
            deduped[node_id] = node_data
        ids = list(deduped.keys())
        props = [json.dumps(v) for v in deduped.values()]
        await self._execute(
            """
            INSERT INTO lightrag_graph_nodes (workspace, id, properties, updated_at)
            SELECT $1, u.id, u.props::jsonb, now()
            FROM unnest($2::text[], $3::text[]) AS u(id, props)
            ON CONFLICT (workspace, id)
            DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
            """,
            self.workspace,
            ids,
            props,
        )

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        if not edges:
            return
        # Normalise to canonical order and deduplicate: last write per edge pair wins.
        deduped: dict[tuple[str, str], dict[str, str]] = {}
        for src, tgt, edge_data in edges:
            key = (min(src, tgt), max(src, tgt))
            deduped[key] = edge_data
        srcs = [k[0] for k in deduped]
        tgts = [k[1] for k in deduped]
        props = [json.dumps(v) for v in deduped.values()]
        await self._execute(
            """
            INSERT INTO lightrag_graph_edges (workspace, src_id, tgt_id, properties, updated_at)
            SELECT $1, u.src, u.tgt, u.props::jsonb, now()
            FROM unnest($2::text[], $3::text[], $4::text[]) AS u(src, tgt, props)
            ON CONFLICT (workspace, src_id, tgt_id)
            DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
            """,
            self.workspace,
            srcs,
            tgts,
            props,
        )
