"""
PostgreSQL + Recursive CTE graph storage for LightRAG.

Implements BaseGraphStorage using plain PostgreSQL tables and recursive
CTEs — no Apache AGE extension, no pgvector, no Cypher wrapper required.

Schema (created automatically on initialize):

    graph_nodes(workspace TEXT, id TEXT, properties JSONB)  PK(workspace, id)
    graph_edges(workspace TEXT, src_id TEXT, tgt_id TEXT, properties JSONB)
                                                             PK(workspace, src_id, tgt_id)

All edge queries check both directions so the semantics are undirected,
matching the BaseGraphStorage contract.

Configuration (environment variables, same prefix as PGKVStorage):
    POSTGRES_HOST      default: localhost
    POSTGRES_PORT      default: 5432
    POSTGRES_USER      (required)
    POSTGRES_PASSWORD  (required)
    POSTGRES_DATABASE  (required)
    POSTGRES_MAX_CONNECTIONS  default: 10
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pipmaster as pm

if not pm.is_installed("asyncpg"):
    pm.install("asyncpg")

import asyncpg  # type: ignore

from dotenv import load_dotenv

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

load_dotenv(dotenv_path=".env", override=False)

# ---------------------------------------------------------------------------
# DDL — executed once per workspace on initialize()
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

CREATE INDEX IF NOT EXISTS idx_lightrag_graph_nodes_props
    ON lightrag_graph_nodes USING GIN (properties);
"""

# ---------------------------------------------------------------------------
# Thin asyncpg pool wrapper
# ---------------------------------------------------------------------------


class _PgPool:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def initialize(self, dsn: str, min_size: int = 1, max_size: int = 10) -> None:
        self._pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        async with self._pool.acquire() as conn:
            await conn.execute(_DDL)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PgRcteGraphStorage not initialized")
        return self._pool

    async def fetchrow(self, sql: str, *args: Any) -> asyncpg.Record | None:
        async with self._require().acquire() as conn:
            return await conn.fetchrow(sql, *args)

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        async with self._require().acquire() as conn:
            return await conn.fetch(sql, *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        async with self._require().acquire() as conn:
            return await conn.fetchval(sql, *args)

    async def execute(self, sql: str, *args: Any) -> None:
        async with self._require().acquire() as conn:
            await conn.execute(sql, *args)


# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------


@dataclass
class PgRcteGraphStorage(BaseGraphStorage):
    """LightRAG graph storage backed by PostgreSQL + Recursive CTEs.

    Drop-in replacement for PGGraphStorage that requires no Apache AGE
    extension.  Uses two plain tables (lightrag_graph_nodes,
    lightrag_graph_edges) with JSONB properties and B-tree indexes.
    """

    _pool: _PgPool | None = field(default=None, init=False, repr=False)

    @property
    def _db(self) -> _PgPool:
        if self._pool is None:
            raise RuntimeError("PgRcteGraphStorage not initialized — call initialize() first")
        return self._pool

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dsn(self) -> str:
        cfg = self.global_config
        host = cfg.get("postgres_host", os.environ.get("POSTGRES_HOST", "localhost"))
        port = cfg.get("postgres_port", os.environ.get("POSTGRES_PORT", "5432"))
        user = cfg.get("postgres_user", os.environ.get("POSTGRES_USER", ""))
        password = cfg.get("postgres_password", os.environ.get("POSTGRES_PASSWORD", ""))
        database = cfg.get("postgres_database", os.environ.get("POSTGRES_DATABASE", ""))
        if not user or not password or not database:
            raise ValueError(
                "PgRcteGraphStorage requires POSTGRES_USER, POSTGRES_PASSWORD, "
                "and POSTGRES_DATABASE to be set."
            )
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        max_conn = int(
            self.global_config.get(
                "postgres_max_connections",
                os.environ.get("POSTGRES_MAX_CONNECTIONS", 10),
            )
        )
        self._pool = _PgPool()
        await self._pool.initialize(self._dsn(), max_size=max_conn)
        logger.info(f"[{self.workspace}] PgRcteGraphStorage initialized")

    async def finalize(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            logger.info(f"[{self.workspace}] PgRcteGraphStorage finalized")

    async def index_done_callback(self) -> None:
        pass  # PostgreSQL writes are immediately durable

    async def drop(self) -> dict[str, str]:
        try:
            await self._db.execute(
                "DELETE FROM lightrag_graph_edges WHERE workspace = $1", self.workspace
            )
            await self._db.execute(
                "DELETE FROM lightrag_graph_nodes WHERE workspace = $1", self.workspace
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        row = await self._db.fetchrow(
            "SELECT 1 FROM lightrag_graph_nodes WHERE workspace = $1 AND id = $2",
            self.workspace, node_id,
        )
        return row is not None

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        row = await self._db.fetchrow(
            "SELECT properties FROM lightrag_graph_nodes WHERE workspace = $1 AND id = $2",
            self.workspace, node_id,
        )
        return json.loads(row["properties"]) if row else None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        await self._db.execute(
            """
            INSERT INTO lightrag_graph_nodes (workspace, id, properties, updated_at)
            VALUES ($1, $2, $3, now())
            ON CONFLICT (workspace, id)
            DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
            """,
            self.workspace, node_id, json.dumps(node_data),
        )

    async def delete_node(self, node_id: str) -> None:
        await self._db.execute(
            "DELETE FROM lightrag_graph_edges WHERE workspace=$1 AND (src_id=$2 OR tgt_id=$2)",
            self.workspace, node_id,
        )
        await self._db.execute(
            "DELETE FROM lightrag_graph_nodes WHERE workspace = $1 AND id = $2",
            self.workspace, node_id,
        )

    async def remove_nodes(self, nodes: list[str]) -> None:
        if not nodes:
            return
        await self._db.execute(
            "DELETE FROM lightrag_graph_edges WHERE workspace=$1 AND (src_id=ANY($2) OR tgt_id=ANY($2))",
            self.workspace, nodes,
        )
        await self._db.execute(
            "DELETE FROM lightrag_graph_nodes WHERE workspace = $1 AND id = ANY($2)",
            self.workspace, nodes,
        )

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        row = await self._db.fetchrow(
            """
            SELECT 1 FROM lightrag_graph_edges
            WHERE workspace = $1
              AND ((src_id=$2 AND tgt_id=$3) OR (src_id=$3 AND tgt_id=$2))
            """,
            self.workspace, source_node_id, target_node_id,
        )
        return row is not None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        row = await self._db.fetchrow(
            """
            SELECT properties FROM lightrag_graph_edges
            WHERE workspace = $1
              AND ((src_id=$2 AND tgt_id=$3) OR (src_id=$3 AND tgt_id=$2))
            """,
            self.workspace, source_node_id, target_node_id,
        )
        return json.loads(row["properties"]) if row else None

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        await self._db.execute(
            """
            INSERT INTO lightrag_graph_edges (workspace, src_id, tgt_id, properties, updated_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (workspace, src_id, tgt_id)
            DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
            """,
            self.workspace, source_node_id, target_node_id, json.dumps(edge_data),
        )

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return
        srcs = [e[0] for e in edges]
        tgts = [e[1] for e in edges]
        await self._db.execute(
            """
            DELETE FROM lightrag_graph_edges
            WHERE workspace = $1
              AND (src_id, tgt_id) IN (SELECT * FROM unnest($2::text[], $3::text[]))
            """,
            self.workspace, srcs, tgts,
        )
        await self._db.execute(
            """
            DELETE FROM lightrag_graph_edges
            WHERE workspace = $1
              AND (src_id, tgt_id) IN (SELECT * FROM unnest($2::text[], $3::text[]))
            """,
            self.workspace, tgts, srcs,
        )

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if not await self.has_node(source_node_id):
            return None
        rows = await self._db.fetch(
            """
            SELECT src_id, tgt_id FROM lightrag_graph_edges
            WHERE workspace = $1 AND (src_id = $2 OR tgt_id = $2)
            """,
            self.workspace, source_node_id,
        )
        return [(r["src_id"], r["tgt_id"]) for r in rows]

    # ------------------------------------------------------------------
    # Degree queries
    # ------------------------------------------------------------------

    async def node_degree(self, node_id: str) -> int:
        val = await self._db.fetchval(
            "SELECT COUNT(*) FROM lightrag_graph_edges WHERE workspace=$1 AND (src_id=$2 OR tgt_id=$2)",
            self.workspace, node_id,
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
        rows = await self._db.fetch(
            "SELECT id FROM lightrag_graph_nodes WHERE workspace = $1 ORDER BY id",
            self.workspace,
        )
        return [r["id"] for r in rows]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        rows = await self._db.fetch(
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
            self.workspace, limit,
        )
        return [r["id"] for r in rows]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        rows = await self._db.fetch(
            "SELECT id FROM lightrag_graph_nodes WHERE workspace=$1 AND id ILIKE $2 ORDER BY id LIMIT $3",
            self.workspace, f"%{query}%", limit,
        )
        return [r["id"] for r in rows]

    async def get_all_nodes(self) -> list[dict]:
        rows = await self._db.fetch(
            "SELECT id, properties FROM lightrag_graph_nodes WHERE workspace = $1",
            self.workspace,
        )
        return [{"id": r["id"], **json.loads(r["properties"])} for r in rows]

    async def get_all_edges(self) -> list[dict]:
        rows = await self._db.fetch(
            "SELECT src_id, tgt_id, properties FROM lightrag_graph_edges WHERE workspace = $1",
            self.workspace,
        )
        return [{"src_id": r["src_id"], "tgt_id": r["tgt_id"], **json.loads(r["properties"])} for r in rows]

    # ------------------------------------------------------------------
    # Knowledge graph — Recursive CTE BFS
    # ------------------------------------------------------------------

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        # RCTE with depth column so max_depth is enforced at the DB level.
        # UNION deduplicates (id, properties, depth) tuples; the final
        # SELECT DISTINCT deduplicates by id across depths.
        if node_label == "*":
            # $1=workspace  $2=max_depth  $3=max_nodes
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
            params: list[Any] = [self.workspace, max_depth, max_nodes]
        else:
            # $1=workspace  $2=label pattern  $3=max_depth  $4=max_nodes
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
            params = [self.workspace, f"%{node_label}%", max_depth, max_nodes]

        node_rows = await self._db.fetch(rcte_sql, *params)
        node_ids = {r["id"] for r in node_rows}
        is_truncated = len(node_ids) >= max_nodes

        nodes = [
            KnowledgeGraphNode(id=r["id"], labels=[r["id"]], properties=json.loads(r["properties"]))
            for r in node_rows
        ]

        edges: list[KnowledgeGraphEdge] = []
        if node_ids:
            edge_rows = await self._db.fetch(
                """
                SELECT src_id, tgt_id, properties FROM lightrag_graph_edges
                WHERE workspace = $1 AND src_id = ANY($2) AND tgt_id = ANY($2)
                """,
                self.workspace, list(node_ids),
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
    # Batch method overrides — avoid N+1 serial calls
    # ------------------------------------------------------------------

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        rows = await self._db.fetch(
            "SELECT id, properties FROM lightrag_graph_nodes WHERE workspace=$1 AND id=ANY($2)",
            self.workspace, node_ids,
        )
        return {r["id"]: json.loads(r["properties"]) for r in rows}

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        rows = await self._db.fetch(
            """
            SELECT id, COUNT(*) AS degree
            FROM (
                SELECT src_id AS id FROM lightrag_graph_edges WHERE workspace=$1 AND src_id=ANY($2)
                UNION ALL
                SELECT tgt_id AS id FROM lightrag_graph_edges WHERE workspace=$1 AND tgt_id=ANY($2)
            ) sub
            GROUP BY id
            """,
            self.workspace, node_ids,
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
        srcs = [p["src"] for p in pairs]
        tgts = [p["tgt"] for p in pairs]
        rows = await self._db.fetch(
            """
            SELECT src_id, tgt_id, properties FROM lightrag_graph_edges
            WHERE workspace=$1
              AND (src_id, tgt_id) IN (SELECT * FROM unnest($2::text[], $3::text[]))
            """,
            self.workspace, srcs, tgts,
        )
        result: dict[tuple[str, str], dict] = {}
        for r in rows:
            result[(r["src_id"], r["tgt_id"])] = json.loads(r["properties"])
        rev = await self._db.fetch(
            """
            SELECT src_id, tgt_id, properties FROM lightrag_graph_edges
            WHERE workspace=$1
              AND (src_id, tgt_id) IN (SELECT * FROM unnest($2::text[], $3::text[]))
            """,
            self.workspace, tgts, srcs,
        )
        for r in rev:
            key = (r["tgt_id"], r["src_id"])
            if key not in result:
                result[key] = json.loads(r["properties"])
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}
        rows = await self._db.fetch(
            """
            SELECT src_id, tgt_id FROM lightrag_graph_edges
            WHERE workspace=$1 AND (src_id=ANY($2) OR tgt_id=ANY($2))
            """,
            self.workspace, node_ids,
        )
        result: dict[str, list[tuple[str, str]]] = {nid: [] for nid in node_ids}
        for r in rows:
            src, tgt = r["src_id"], r["tgt_id"]
            if src in result:
                result[src].append((src, tgt))
            if tgt in result:
                result[tgt].append((src, tgt))
        return result
