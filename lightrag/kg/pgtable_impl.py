"""
PostgreSQL-native graph storage for LightRAG (plain tables, no extensions).

Implements BaseGraphStorage on ordinary indexed PostgreSQL tables — no Apache
AGE extension, no pgvector, no Cypher wrapper required. All operations are plain
indexed SQL; the visualization-only exact-label get_knowledge_graph traversal
uses a frontier-capped iterative BFS (bounded by max_depth / max_nodes), not a
recursive CTE.

Schema (created automatically on initialize):

    lightrag_graph_nodes(workspace TEXT, namespace TEXT, id TEXT, properties JSONB)
                                                  PK(workspace, namespace, id)
    lightrag_graph_edges(workspace TEXT, namespace TEXT, src_id TEXT, tgt_id TEXT, properties JSONB)
        Edges are stored in canonical order: src_id = min(a, b), tgt_id = max(a, b) — normalized via Python min/max (NOT SQL LEAST/GREATEST; mixing the two would diverge on non-ASCII under non-C collations and cause duplicate edges)
                                                  PK(workspace, namespace, src_id, tgt_id)

Configuration: inherits all POSTGRES_* / POSTGRES_SSL_* / POSTGRES_WORKSPACE
environment variables via the shared ClientManager / PostgreSQLDB pool, identical
to PGKVStorage and PGVectorStorage.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..base import BaseGraphStorage
from ..kg.shared_storage import get_data_init_lock
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger, validate_workspace


def _is_transient_error(exc: BaseException) -> bool:
    """True for query-level transient errors (read or write) NOT covered by
    PostgreSQLDB._run_with_retry (which retries connection-level failures only).

    Mirrors PGGraphStorage's graph-write retry surface: deadlock, serialization
    conflict, lock-acquisition timeout, and statement cancellation that can occur
    under concurrent document ingestion.

    asyncpg is imported lazily so importing this module does not pull in
    postgres_impl / asyncpg (preserves the no-import-side-effect contract). If
    asyncpg is somehow unavailable, treat the error as non-transient so the
    original exception propagates instead of being masked by ModuleNotFoundError.
    """
    try:
        import asyncpg
    except ImportError:
        return False

    return isinstance(
        exc,
        (
            asyncpg.exceptions.DeadlockDetectedError,
            asyncpg.exceptions.SerializationError,
            asyncpg.exceptions.LockNotAvailableError,
            asyncpg.exceptions.QueryCanceledError,
        ),
    )


# ---------------------------------------------------------------------------
# DDL — executed once per process on first initialize()
# ---------------------------------------------------------------------------

_DDL = """
SELECT pg_advisory_xact_lock(hashtext('lightrag_pgtable_schema'));

CREATE TABLE IF NOT EXISTS lightrag_graph_nodes (
    workspace   TEXT        NOT NULL,
    namespace   TEXT        NOT NULL DEFAULT 'chunk_entity_relation',
    id          TEXT        NOT NULL,
    properties  JSONB       NOT NULL DEFAULT '{}',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace, namespace, id)
);

CREATE TABLE IF NOT EXISTS lightrag_graph_edges (
    workspace   TEXT        NOT NULL,
    namespace   TEXT        NOT NULL DEFAULT 'chunk_entity_relation',
    src_id      TEXT        NOT NULL,
    tgt_id      TEXT        NOT NULL,
    properties  JSONB       NOT NULL DEFAULT '{}',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace, namespace, src_id, tgt_id)
);

ALTER TABLE lightrag_graph_nodes
    ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'chunk_entity_relation';

ALTER TABLE lightrag_graph_edges
    ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'chunk_entity_relation';

DO $$
DECLARE
    nodes_pk_has_namespace BOOLEAN;
    edges_pk_has_namespace BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_attribute a
          ON a.attrelid = c.conrelid
         AND a.attnum = ANY(c.conkey)
        WHERE c.conname = 'lightrag_graph_nodes_pkey'
          AND c.conrelid = 'lightrag_graph_nodes'::regclass
          AND a.attname = 'namespace'
    ) INTO nodes_pk_has_namespace;

    SELECT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_attribute a
          ON a.attrelid = c.conrelid
         AND a.attnum = ANY(c.conkey)
        WHERE c.conname = 'lightrag_graph_edges_pkey'
          AND c.conrelid = 'lightrag_graph_edges'::regclass
          AND a.attname = 'namespace'
    ) INTO edges_pk_has_namespace;

    IF NOT nodes_pk_has_namespace OR NOT edges_pk_has_namespace THEN
        ALTER TABLE lightrag_graph_edges
            DROP CONSTRAINT IF EXISTS fk_lightrag_graph_edges_src;
        ALTER TABLE lightrag_graph_edges
            DROP CONSTRAINT IF EXISTS fk_lightrag_graph_edges_tgt;

        IF NOT edges_pk_has_namespace THEN
            ALTER TABLE lightrag_graph_edges
                DROP CONSTRAINT IF EXISTS lightrag_graph_edges_pkey;
            -- Defensive: a legacy/corrupt table could hold duplicate rows for the
            -- new key, which would make ADD PRIMARY KEY fail mid-migration and
            -- abort startup with no self-heal. Drop dupes first, keeping the
            -- most-recently-updated row per key (ctid breaks updated_at ties).
            DELETE FROM lightrag_graph_edges a
            USING lightrag_graph_edges b
            WHERE a.workspace = b.workspace
              AND a.namespace = b.namespace
              AND a.src_id = b.src_id
              AND a.tgt_id = b.tgt_id
              AND (a.updated_at, a.ctid) < (b.updated_at, b.ctid);
            ALTER TABLE lightrag_graph_edges
                ADD PRIMARY KEY (workspace, namespace, src_id, tgt_id);
        END IF;

        IF NOT nodes_pk_has_namespace THEN
            ALTER TABLE lightrag_graph_nodes
                DROP CONSTRAINT IF EXISTS lightrag_graph_nodes_pkey;
            -- Same defensive dedup as edges above: keep the most-recently-updated
            -- row per (workspace, namespace, id) so ADD PRIMARY KEY cannot fail on
            -- a legacy/corrupt table that holds duplicates.
            DELETE FROM lightrag_graph_nodes a
            USING lightrag_graph_nodes b
            WHERE a.workspace = b.workspace
              AND a.namespace = b.namespace
              AND a.id = b.id
              AND (a.updated_at, a.ctid) < (b.updated_at, b.ctid);
            ALTER TABLE lightrag_graph_nodes
                ADD PRIMARY KEY (workspace, namespace, id);
        END IF;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_lightrag_graph_edges_namespace_tgt
    ON lightrag_graph_edges (workspace, namespace, tgt_id);

DELETE FROM lightrag_graph_edges e
WHERE NOT EXISTS (
        SELECT 1 FROM lightrag_graph_nodes n
        WHERE n.workspace = e.workspace
          AND n.namespace = e.namespace
          AND n.id = e.src_id
    )
   OR NOT EXISTS (
        SELECT 1 FROM lightrag_graph_nodes n
        WHERE n.workspace = e.workspace
          AND n.namespace = e.namespace
          AND n.id = e.tgt_id
    );

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_lightrag_graph_edges_src'
          AND conrelid = 'lightrag_graph_edges'::regclass
    ) THEN
        ALTER TABLE lightrag_graph_edges
            ADD CONSTRAINT fk_lightrag_graph_edges_src
            FOREIGN KEY (workspace, namespace, src_id)
            REFERENCES lightrag_graph_nodes (workspace, namespace, id)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_lightrag_graph_edges_tgt'
          AND conrelid = 'lightrag_graph_edges'::regclass
    ) THEN
        ALTER TABLE lightrag_graph_edges
            ADD CONSTRAINT fk_lightrag_graph_edges_tgt
            FOREIGN KEY (workspace, namespace, tgt_id)
            REFERENCES lightrag_graph_nodes (workspace, namespace, id)
            ON DELETE CASCADE;
    END IF;
END $$;
"""

# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------


@dataclass
class PGTableGraphStorage(BaseGraphStorage):
    """LightRAG BaseGraphStorage-compatible, PostgreSQL-native graph backend.

    This is NOT an AGE / PGGraphStorage clone. It implements the LightRAG
    BaseGraphStorage contract directly on two plain tables (lightrag_graph_nodes,
    lightrag_graph_edges) with JSONB properties and B-tree indexes — no Apache
    AGE extension, no Cypher. AGE / PGGraphStorage is only a reference
    implementation; behaviour that exists purely because of AGE's Cypher
    constraints is not copied.

    Edge storage / undirected semantics:
        Edges are stored in canonical order src_id = min(a, b), tgt_id = max(a, b)
        via Python min/max (NOT SQL LEAST/GREATEST; mixing the two diverges on
        non-ASCII under non-C collations and yields duplicate edges). All write
        paths normalise before INSERT, so upsert_edge(A, B) and upsert_edge(B, A)
        map to one row.

    Intentional contract differences from PGGraphStorage (documented on purpose):
      - Edge upsert CREATES missing endpoint nodes with a minimal
        {"entity_id": id} payload, matching NetworkX add_edge semantics.
        PGGraphStorage may silently skip such edges because its Cypher
        implementation MATCHes existing endpoints. Minimal endpoints are later
        enriched by node upsert (which merges, see below).
      - Node upsert REQUIRES entity_id (ValueError if absent, like
        PGGraphStorage), but the stored value is always canonicalised to node_id;
        node properties are MERGED (jsonb ||), not replaced, so omitted keys
        survive a partial update.
      - Batch node upsert dedupes duplicate node_ids by last-write-wins (like
        PGGraphStorage); edge properties are replaced, node properties merged.
      - get_knowledge_graph uses a frontier-capped iterative BFS (see
        _bfs_frontier) whose cost is bounded by max_nodes, not by the number of
        simple paths in the reachable subgraph.
    """

    db: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_workspace(self.workspace)

    @property
    def _db(self) -> Any:
        if self.db is None:
            raise RuntimeError(
                "PGTableGraphStorage not initialized — call initialize() first"
            )
        return self.db

    # ------------------------------------------------------------------
    # Query helpers — thin wrappers over PostgreSQLDB.query / execute
    # ------------------------------------------------------------------

    async def _with_retry(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        # Retry query-level transient errors (deadlock / serialization /
        # lock-timeout / cancel) that PostgreSQLDB._run_with_retry does not cover
        # (it retries connection-level failures only). Applies to reads AND writes:
        # writes hit deadlock / unique-index contention under concurrent workers,
        # and under a SERIALIZABLE / REPEATABLE READ pool a read racing concurrent
        # ingestion can surface a SerializationError. Backoff stays short so a
        # victim retries promptly rather than stalling throughput.
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
            retry=retry_if_exception(_is_transient_error),
            reraise=True,
        ):
            with attempt:
                return await fn()

    async def _execute(self, sql: str, *args: Any) -> None:
        data = {str(i): v for i, v in enumerate(args)} if args else None
        await self._with_retry(lambda: self._db.execute(sql, data=data))

    async def _fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        result = await self._with_retry(
            lambda: self._db.query(sql, list(args) if args else None)
        )
        return result if isinstance(result, dict) else None  # type: ignore[return-value]

    async def _fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        result = await self._with_retry(
            lambda: self._db.query(sql, list(args) if args else None, multirows=True)
        )
        return result if isinstance(result, list) else []  # type: ignore[return-value]

    async def _fetchval(self, sql: str, *args: Any) -> Any:
        result = await self._with_retry(
            lambda: self._db.query(sql, list(args) if args else None)
        )
        if not isinstance(result, dict):
            return None
        return next(iter(result.values()))

    @staticmethod
    def _json_loads(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        return dict(value or {})

    @staticmethod
    def _node_props(node_id: str, properties: Any) -> dict[str, Any]:
        props = PGTableGraphStorage._json_loads(properties)
        props["entity_id"] = node_id
        return props

    @staticmethod
    def _node_output(node_id: str, properties: Any) -> dict[str, Any]:
        props = PGTableGraphStorage._node_props(node_id, properties)
        props["id"] = node_id
        return props

    @staticmethod
    def _search_score(label: str, query: str) -> int:
        # Mirror NetworkXStorage.search_labels scoring exactly: the
        # word-boundary bonus applies ONLY to the contains branch, not to
        # exact/prefix matches.
        lowered = label.lower()
        if lowered == query:
            return 1000
        if lowered.startswith(query):
            return 500
        score = 100 - len(label)
        if f" {query}" in lowered or f"_{query}" in lowered:
            score += 50
        return score

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        async with get_data_init_lock():
            if self.db is None:
                from .postgres_impl import ClientManager

                vector_storage = (
                    self.global_config.get("vector_storage") or "PGTableGraphStorage"
                )
                self.db = await ClientManager.get_client(
                    vector_storage=vector_storage,
                )
                # Workspace priority: POSTGRES_WORKSPACE env > self.workspace > "default"
                if self.db.workspace:
                    self.workspace = self.db.workspace
                elif not getattr(self, "workspace", None):
                    self.workspace = "default"
                await self._db.execute(_DDL)
                await self._normalize_legacy_edges()
        logger.info(f"[{self.workspace}] PGTableGraphStorage initialized")

    async def finalize(self) -> None:
        if self.db is not None:
            from .postgres_impl import ClientManager

            await ClientManager.release_client(self.db)
            self.db = None
            logger.info(f"[{self.workspace}] PGTableGraphStorage finalized")

    async def _normalize_legacy_edges(self) -> None:
        # Fast path: if no edge is stored in non-canonical order, every edge is
        # already normalized AND free of reversed duplicates — a reversed
        # duplicate (B, A) of canonical (A, B) necessarily has src_id > tgt_id.
        # This guard avoids the full table scan + Python regrouping on every
        # initialize() once the one-time legacy migration has run.
        #
        # The canonical order is defined by Python min/max (code points), so the
        # guard compares with COLLATE "C" (byte order == code-point order for
        # UTF-8). Using the DB's default collation here could diverge on
        # non-ASCII ids and either skip rows that still need normalizing or
        # rescan already-canonical data forever.
        needs_normalize = await self._fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM lightrag_graph_edges
                WHERE workspace = $1 AND namespace = $2
                  AND src_id COLLATE "C" > tgt_id COLLATE "C"
            )
            """,
            self.workspace,
            self.namespace,
        )
        if not needs_normalize:
            return

        # Run the migration inside ONE transaction that holds the same advisory
        # lock as the schema DDL. The lock serializes concurrent initialize()
        # calls (ingestion writers do NOT take it) and FOR UPDATE locks the rows
        # the scan sees. A concurrent writer can still insert a fresh canonical
        # row the snapshot never saw, so the re-insert below guards its ON CONFLICT
        # update with `updated_at <= survivor` to avoid clobbering that newer row
        # with the older legacy survivor payload.
        async def _migrate(conn: Any) -> None:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('lightrag_pgtable_schema'))"
                )
                rows = await conn.fetch(
                    """
                    SELECT src_id, tgt_id, properties, updated_at
                    FROM lightrag_graph_edges
                    WHERE workspace = $1 AND namespace = $2
                    FOR UPDATE
                    """,
                    self.workspace,
                    self.namespace,
                )
                groups: dict[tuple[str, str], list[Any]] = {}
                for row in rows:
                    src, tgt = row["src_id"], row["tgt_id"]
                    groups.setdefault((min(src, tgt), max(src, tgt)), []).append(row)

                for (src, tgt), group in groups.items():
                    if (
                        len(group) == 1
                        and group[0]["src_id"] == src
                        and group[0]["tgt_id"] == tgt
                    ):
                        continue
                    survivor = max(
                        group,
                        key=lambda r: (
                            r["updated_at"] is not None,
                            r["updated_at"] or "",
                            # Deterministic tie-break: when reversed duplicates
                            # share updated_at, prefer the already-canonical row so
                            # which properties survive is reproducible across runs.
                            (r["src_id"], r["tgt_id"]) == (src, tgt),
                        ),
                    )
                    survivor_props = survivor["properties"]
                    if not isinstance(survivor_props, str):
                        survivor_props = json.dumps(survivor_props)
                    # Delete only the NON-canonical members of the group. Including
                    # the canonical (src, tgt) row would make this data-modifying
                    # CTE both DELETE and (via ON CONFLICT) UPDATE the same tuple in
                    # one statement, which Postgres rejects with "tuple to be updated
                    # was already modified by an operation triggered by the current
                    # command" — aborting the migration on every startup. Excluding
                    # it keeps the delete set and the INSERT target disjoint.
                    non_canonical = [
                        r for r in group if (r["src_id"], r["tgt_id"]) != (src, tgt)
                    ]
                    old_srcs = [r["src_id"] for r in non_canonical]
                    old_tgts = [r["tgt_id"] for r in non_canonical]
                    await conn.execute(
                        """
                        WITH deleted AS (
                            DELETE FROM lightrag_graph_edges
                            WHERE workspace = $1
                              AND namespace = $2
                              AND (src_id, tgt_id) IN (
                                  SELECT * FROM unnest($3::text[], $4::text[])
                              )
                        )
                        INSERT INTO lightrag_graph_edges
                            (workspace, namespace, src_id, tgt_id, properties, updated_at)
                        VALUES ($1, $2, $5, $6, $7::jsonb, now())
                        ON CONFLICT (workspace, namespace, src_id, tgt_id)
                        DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
                        -- Don't clobber a fresher row: a concurrent writer (which
                        -- does not take this migration's advisory lock) may have
                        -- inserted a canonical row the FOR UPDATE snapshot never
                        -- saw. Only overwrite when the existing row is not newer
                        -- than the legacy survivor being consolidated.
                        WHERE lightrag_graph_edges.updated_at <= $8
                        """,
                        self.workspace,
                        self.namespace,
                        old_srcs,
                        old_tgts,
                        src,
                        tgt,
                        survivor_props,
                        survivor["updated_at"],
                    )

        await self._db._run_with_retry(_migrate)

    async def index_done_callback(self) -> None:
        pass  # PostgreSQL writes are immediately durable

    async def drop(self) -> dict[str, str]:
        try:
            # Atomic drop: a single data-modifying CTE deletes edges and nodes in
            # one statement (hence one implicit transaction), so a mid-drop
            # failure can never leave nodes stripped of their edges or vice versa
            # — the two-statement version committed the edge delete independently.
            # Edges are deleted explicitly rather than relying solely on FK
            # CASCADE so the drop stays correct even on a legacy table whose FK
            # was never (re)created.
            await self._execute(
                """
                WITH deleted_edges AS (
                    DELETE FROM lightrag_graph_edges
                    WHERE workspace = $1 AND namespace = $2
                )
                DELETE FROM lightrag_graph_nodes
                WHERE workspace = $1 AND namespace = $2
                """,
                self.workspace,
                self.namespace,
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as exc:
            logger.error(f"drop() failed: {exc}")
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        row = await self._fetchrow(
            "SELECT 1 FROM lightrag_graph_nodes WHERE workspace = $1 AND namespace = $2 AND id = $3",
            self.workspace,
            self.namespace,
            node_id,
        )
        return row is not None

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        row = await self._fetchrow(
            "SELECT properties FROM lightrag_graph_nodes WHERE workspace = $1 AND namespace = $2 AND id = $3",
            self.workspace,
            self.namespace,
            node_id,
        )
        return self._node_props(node_id, row["properties"]) if row else None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        # Match PGGraphStorage: require entity_id to surface malformed caller
        # payloads early. The value is then forced to node_id (AGE uses the
        # node_id argument as the vertex label and ignores node_data's value).
        if "entity_id" not in node_data:
            raise ValueError(
                "PostgreSQL: node properties must contain an 'entity_id' field"
            )
        node_props = dict(node_data)
        node_props["entity_id"] = node_id
        await self._execute(
            """
            INSERT INTO lightrag_graph_nodes (workspace, namespace, id, properties, updated_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (workspace, namespace, id)
            DO UPDATE SET
                -- Merge (not replace) so omitted keys survive, matching
                -- NetworkXStorage.add_node(**data) and PGGraphStorage's SET n += .
                -- EXCLUDED wins on shared keys; EXCLUDED always carries
                -- entity_id = node_id, so the canonical entity_id is preserved.
                properties = lightrag_graph_nodes.properties || EXCLUDED.properties,
                updated_at = now()
            """,
            self.workspace,
            self.namespace,
            node_id,
            json.dumps(node_props),
        )

    async def delete_node(self, node_id: str) -> None:
        await self._execute(
            "DELETE FROM lightrag_graph_nodes WHERE workspace = $1 AND namespace = $2 AND id = $3",
            self.workspace,
            self.namespace,
            node_id,
        )

    async def remove_nodes(self, nodes: list[str]) -> None:
        if not nodes:
            return
        await self._execute(
            "DELETE FROM lightrag_graph_nodes WHERE workspace = $1 AND namespace = $2 AND id = ANY($3)",
            self.workspace,
            self.namespace,
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
            "SELECT 1 FROM lightrag_graph_edges WHERE workspace=$1 AND namespace=$2 AND src_id=$3 AND tgt_id=$4",
            self.workspace,
            self.namespace,
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
            "SELECT properties FROM lightrag_graph_edges WHERE workspace=$1 AND namespace=$2 AND src_id=$3 AND tgt_id=$4",
            self.workspace,
            self.namespace,
            src,
            tgt,
        )
        return self._json_loads(row["properties"]) if row else None

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        await self._execute(
            """
            WITH endpoints AS (
                INSERT INTO lightrag_graph_nodes (workspace, namespace, id, properties, updated_at)
                SELECT $1, $2, u.id, jsonb_build_object('entity_id', u.id), now()
                FROM unnest($6::text[]) AS u(id)
                ON CONFLICT (workspace, namespace, id) DO NOTHING
                RETURNING id
            ),
            endpoint_write AS (
                SELECT COUNT(*) AS inserted_count FROM endpoints
            )
            INSERT INTO lightrag_graph_edges (workspace, namespace, src_id, tgt_id, properties, updated_at)
            SELECT $1, $2, $3, $4, $5::jsonb, now()
            FROM endpoint_write
            ON CONFLICT (workspace, namespace, src_id, tgt_id)
            DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
            """,
            self.workspace,
            self.namespace,
            src,
            tgt,
            json.dumps(edge_data),
            [src, tgt],
        )

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return
        if not all(isinstance(e[0], str) and isinstance(e[1], str) for e in edges):
            raise ValueError("Edge node IDs must be non-None strings")
        srcs = [min(e[0], e[1]) for e in edges]
        tgts = [max(e[0], e[1]) for e in edges]
        await self._execute(
            """
            DELETE FROM lightrag_graph_edges
            WHERE workspace = $1
              AND namespace = $2
              AND (src_id, tgt_id) IN (SELECT * FROM unnest($3::text[], $4::text[]))
            """,
            self.workspace,
            self.namespace,
            srcs,
            tgts,
        )

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if not await self.has_node(source_node_id):
            return None
        rows = await self._fetch(
            """
            SELECT src_id, tgt_id FROM lightrag_graph_edges
            WHERE workspace = $1 AND namespace = $2 AND (src_id = $3 OR tgt_id = $3)
            """,
            self.workspace,
            self.namespace,
            source_node_id,
        )
        edges = [
            (
                source_node_id,
                r["tgt_id"] if r["src_id"] == source_node_id else r["src_id"],
            )
            for r in rows
        ]
        edges.sort(key=lambda e: e[1])
        return edges

    # ------------------------------------------------------------------
    # Degree queries
    # ------------------------------------------------------------------

    async def node_degree(self, node_id: str) -> int:
        val = await self._fetchval(
            """
            SELECT COALESCE(SUM(
                CASE WHEN src_id = $3 THEN 1 ELSE 0 END +
                CASE WHEN tgt_id = $3 THEN 1 ELSE 0 END
            ), 0)
            FROM lightrag_graph_edges
            WHERE workspace=$1 AND namespace=$2 AND (src_id=$3 OR tgt_id=$3)
            """,
            self.workspace,
            self.namespace,
            node_id,
        )
        return int(val or 0)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        s, t = await asyncio.gather(self.node_degree(src_id), self.node_degree(tgt_id))
        return s + t

    # ------------------------------------------------------------------
    # Label queries
    # ------------------------------------------------------------------

    async def get_all_labels(self) -> list[str]:
        rows = await self._fetch(
            "SELECT id FROM lightrag_graph_nodes WHERE workspace = $1 AND namespace = $2",
            self.workspace,
            self.namespace,
        )
        return sorted(r["id"] for r in rows)

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        # Rank ALL nodes by degree, including isolated (degree 0) nodes, to
        # match NetworkXStorage.get_popular_labels (dict(graph.degree()) covers
        # every node). Counting from the edge table alone would silently drop
        # isolated entities. Self-loops count twice (no src_id <> tgt_id guard),
        # consistent with node_degree.
        rows = await self._fetch(
            """
            SELECT n.id AS id, COALESCE(d.degree, 0) AS degree
            FROM lightrag_graph_nodes n
            LEFT JOIN (
                SELECT id, COUNT(*) AS degree
                FROM (
                    SELECT src_id AS id FROM lightrag_graph_edges
                    WHERE workspace = $1 AND namespace = $2
                    UNION ALL
                    SELECT tgt_id AS id FROM lightrag_graph_edges
                    WHERE workspace = $1 AND namespace = $2
                ) sub
                GROUP BY id
            ) d ON d.id = n.id
            WHERE n.workspace = $1 AND n.namespace = $2
            -- Rank + truncate in SQL instead of fetching every node and slicing
            -- in Python: the popular-label endpoint caps the result, so this
            -- avoids transferring the whole node set on large graphs. COLLATE "C"
            -- makes the id tie-break match Python's codepoint sort exactly
            -- (degree DESC, id ASC == the previous key=(-degree, id)).
            ORDER BY degree DESC, n.id COLLATE "C" ASC
            LIMIT $3
            """,
            self.workspace,
            self.namespace,
            limit,
        )
        return [r["id"] for r in rows]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        q = query.strip().lower()
        if not q:
            return []
        rows = await self._fetch(
            """
            SELECT id FROM lightrag_graph_nodes
            WHERE workspace=$1 AND namespace=$2 AND LOWER(id) LIKE $3 ESCAPE '\\'
            """,
            self.workspace,
            self.namespace,
            f"%{self._escape_like(q)}%",
        )
        labels = [r["id"] for r in rows]
        labels.sort(key=lambda label: (-self._search_score(label, q), label))
        return labels[:limit]

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    async def get_all_nodes(self) -> list[dict]:
        rows = await self._fetch(
            "SELECT id, properties FROM lightrag_graph_nodes WHERE workspace = $1 AND namespace = $2",
            self.workspace,
            self.namespace,
        )
        result = []
        for r in rows:
            result.append(self._node_output(r["id"], r["properties"]))
        return sorted(result, key=lambda props: props["id"])

    async def get_all_edges(self) -> list[dict]:
        rows = await self._fetch(
            "SELECT src_id, tgt_id, properties FROM lightrag_graph_edges WHERE workspace = $1 AND namespace = $2",
            self.workspace,
            self.namespace,
        )
        edges = [
            {
                **self._json_loads(r["properties"]),
                "source": r["src_id"],
                "target": r["tgt_id"],
            }
            for r in rows
        ]
        return sorted(edges, key=lambda edge: (edge["source"], edge["target"]))

    # ------------------------------------------------------------------
    # Knowledge graph — frontier-capped iterative BFS
    # ------------------------------------------------------------------

    async def _bfs_frontier(
        self, seed: str, max_depth: int, node_budget: int
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Frontier-capped breadth-first traversal from ``seed``.

        Expands one depth level per round, querying only the current frontier's
        neighbours and deduping against a process-side visited set so each node
        is fetched at most once no matter how many paths reach it. The *result*
        size is bounded by ``node_budget`` (traversal stops once it is exceeded),
        and the collected node count never explodes the way a UNION ALL recursive
        CTE with a path-local visited array does — that re-materializes shared
        nodes per path and blows up on dense/cyclic graphs.

        NOTE on cost: the per-hop *work* is bounded by the size of the frontier's
        neighbourhood, NOT by ``node_budget``. A single very high-degree node
        still pulls (and degree-ranks) its entire neighbour set on the hop that
        expands it, before the budget truncates the result. Pushing the
        degree-rank + cap down into SQL is tracked as a follow-up optimization.

        Within each depth level the highest-degree unvisited neighbours are
        admitted first, so when the budget cuts a level the retained nodes are
        the high-degree ones — matching NetworkX's degree-ordered BFS and the
        prior recursive CTE (which degree-sorted the full reachable set before
        truncating). Returns ``(rows, degrees)``: rows are shaped like the
        wildcard path ({"id", "properties", "depth"}) and degrees maps every
        collected node id to its full-graph degree, so the caller reuses them for
        the final seed-pinned ordering instead of recomputing node_degrees_batch.
        """
        seed_row = await self._fetchrow(
            "SELECT id, properties FROM lightrag_graph_nodes "
            "WHERE workspace = $1 AND namespace = $2 AND id = $3",
            self.workspace,
            self.namespace,
            seed,
        )
        if seed_row is None:
            return [], {}
        collected: dict[str, dict[str, Any]] = {
            seed: {"id": seed, "properties": seed_row["properties"], "depth": 0}
        }
        # Accumulate degrees as the BFS visits each level so the caller can reuse
        # them for the final ordering instead of issuing a second full
        # node_degrees_batch over the whole collected set.
        degrees: dict[str, int] = {}
        frontier = [seed]
        depth = 0
        while frontier and depth < max_depth and len(collected) <= node_budget:
            depth += 1
            # Split the undirected neighbour lookup into a UNION so each arm hits
            # an index: src_id = ANY uses the PK prefix (workspace, namespace,
            # src_id, ...), tgt_id = ANY uses idx_..._namespace_tgt. A single
            # "src_id = ANY OR tgt_id = ANY" predicate cannot use either index and
            # degrades to a seq scan of the whole edge table on every level —
            # which made traversal O(edges) per hop on large sparse graphs.
            rows = await self._fetch(
                """
                SELECT n.id AS id, n.properties AS properties
                FROM lightrag_graph_nodes n
                JOIN (
                    SELECT tgt_id AS nid FROM lightrag_graph_edges
                    WHERE workspace = $1 AND namespace = $2 AND src_id = ANY($3)
                  UNION
                    SELECT src_id AS nid FROM lightrag_graph_edges
                    WHERE workspace = $1 AND namespace = $2 AND tgt_id = ANY($3)
                ) nb ON n.id = nb.nid
                WHERE n.workspace = $1 AND n.namespace = $2
                """,
                self.workspace,
                self.namespace,
                frontier,
            )
            # Gather this level's unvisited neighbours, then admit the
            # highest-degree ones first so a budget cut keeps high-degree nodes
            # (degree-priority truncation), rather than whatever order the DB
            # happened to return rows in.
            candidates: dict[str, Any] = {}
            for r in rows:
                if r["id"] not in collected and r["id"] not in candidates:
                    candidates[r["id"]] = r["properties"]
            if not candidates:
                break
            level_degrees = await self.node_degrees_batch(list(candidates))
            degrees.update(level_degrees)
            ordered = sorted(
                candidates, key=lambda nid: (-level_degrees.get(nid, 0), nid)
            )
            next_frontier: list[str] = []
            for nid in ordered:
                collected[nid] = {
                    "id": nid,
                    "properties": candidates[nid],
                    "depth": depth,
                }
                next_frontier.append(nid)
                if len(collected) > node_budget:
                    break
            frontier = next_frontier
        # The seed is never a candidate (it seeds the frontier), so its degree
        # was never gathered in the loop — fetch it once for the ordering
        # tie-break. Isolated seeds (no neighbours) also land here.
        if seed not in degrees:
            degrees[seed] = (await self.node_degrees_batch([seed])).get(seed, 0)
        return list(collected.values()), degrees

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int | None = None,
    ) -> KnowledgeGraph:
        # Mirror AGE: respect global_config["max_graph_nodes"] cap.
        # Use a dedicated int local so the budget is never None below (the
        # max_nodes parameter is int | None and reassigning it keeps that type).
        cap = self.global_config.get("max_graph_nodes", 1000)
        node_budget: int = cap if max_nodes is None else min(max_nodes, cap)

        if node_label == "*":
            node_rows = await self._fetch(
                """
                SELECT n.id AS id,
                       n.properties AS properties,
                       COALESCE(d.degree, 0) AS degree
                FROM lightrag_graph_nodes n
                LEFT JOIN (
                    SELECT id, COUNT(*) AS degree
                    FROM (
                        SELECT src_id AS id FROM lightrag_graph_edges
                        WHERE workspace = $1 AND namespace = $2
                        UNION ALL
                        SELECT tgt_id AS id FROM lightrag_graph_edges
                        WHERE workspace = $1 AND namespace = $2
                    ) sub
                    GROUP BY id
                ) d ON d.id = n.id
                WHERE n.workspace = $1 AND n.namespace = $2
                ORDER BY degree DESC, n.id ASC
                LIMIT $3
                """,
                self.workspace,
                self.namespace,
                node_budget + 1,
            )
            # Wildcard degree comes straight from the SQL aggregate above.
            degrees = {r["id"]: int(r.get("degree", 0)) for r in node_rows}
        else:
            # Exact-match seed traversal via frontier-capped iterative BFS. The
            # blast radius is bounded by node_budget rather than by the number of
            # simple paths in the reachable subgraph (see _bfs_frontier). depth is
            # the shortest-hop distance from the seed (BFS visits each node once),
            # which the degree-aware truncation below orders on.
            # Degrees were gathered during the BFS (see _bfs_frontier) — reused
            # below instead of a second full node_degrees_batch over the set.
            node_rows, degrees = await self._bfs_frontier(
                node_label, max_depth, node_budget
            )

        if not node_rows:
            return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

        # Sort before truncation so the retained set is deterministic AND faithful
        # to the BFS-level semantics of the reference backends (networkx/AGE):
        #   1. seed pinned to position 0 (exact-label queries),
        #   2. shallower BFS depth first — never drop a near node for a distant one,
        #      which keeps the truncated subgraph connected to the seed,
        #   3. higher degree first within a depth level (matches networkx),
        #   4. stable id order as the final tie-breaker.
        # Wildcard rows carry no "depth" key (all default to 0), so the depth term is
        # inert there and selection collapses to the SQL's degree-desc ordering.
        node_rows.sort(
            key=lambda r: (
                # Pin the seed first — exact-label only. Under wildcard there is
                # no seed, so this term must stay constant (otherwise a real
                # entity whose id is literally "*" would be pinned).
                node_label != "*" and r["id"] != node_label,
                r.get("depth", 0),  # shallower BFS level first
                -degrees.get(r["id"], 0),  # higher degree first within a level
                r["id"],
            )
        )

        is_truncated = len(node_rows) > node_budget
        node_rows = node_rows[:node_budget]
        node_ids = {r["id"] for r in node_rows}

        nodes = [
            KnowledgeGraphNode(
                id=r["id"],
                labels=[r["id"]],
                properties=self._node_props(r["id"], r["properties"]),
            )
            for r in node_rows
        ]

        edges: list[KnowledgeGraphEdge] = []
        if node_ids:
            edge_rows = await self._fetch(
                """
                SELECT src_id, tgt_id, properties FROM lightrag_graph_edges
                WHERE workspace = $1
                  AND namespace = $2
                  AND src_id = ANY($3)
                  AND tgt_id = ANY($3)
                """,
                self.workspace,
                self.namespace,
                list(node_ids),
            )
            edges = [
                KnowledgeGraphEdge(
                    id=f"{r['src_id']}-{r['tgt_id']}",
                    type="DIRECTED",
                    source=r["src_id"],
                    target=r["tgt_id"],
                    properties=self._json_loads(r["properties"]),
                )
                for r in sorted(
                    edge_rows, key=lambda row: (row["src_id"], row["tgt_id"])
                )
            ]

        return KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=is_truncated)

    # ------------------------------------------------------------------
    # Read batch overrides — avoid N+1 serial calls
    # ------------------------------------------------------------------

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        rows = await self._fetch(
            "SELECT id, properties FROM lightrag_graph_nodes WHERE workspace=$1 AND namespace=$2 AND id=ANY($3)",
            self.workspace,
            self.namespace,
            node_ids,
        )
        return {r["id"]: self._node_props(r["id"], r["properties"]) for r in rows}

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        rows = await self._fetch(
            """
            SELECT id, COUNT(*) AS degree
            FROM (
                SELECT src_id AS id FROM lightrag_graph_edges
                WHERE workspace=$1 AND namespace=$2 AND src_id=ANY($3)
                UNION ALL
                SELECT tgt_id AS id FROM lightrag_graph_edges
                WHERE workspace=$1 AND namespace=$2 AND tgt_id=ANY($3)
            ) sub
            GROUP BY id
            """,
            self.workspace,
            self.namespace,
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
              AND namespace=$2
              AND (src_id, tgt_id) IN (SELECT * FROM unnest($3::text[], $4::text[]))
            """,
            self.workspace,
            self.namespace,
            srcs,
            tgts,
        )
        canonical_props = {
            (r["src_id"], r["tgt_id"]): self._json_loads(r["properties"]) for r in rows
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
            WHERE workspace=$1 AND namespace=$2 AND src_id=ANY($3)
            UNION
            SELECT src_id, tgt_id FROM lightrag_graph_edges
            WHERE workspace=$1 AND namespace=$2 AND tgt_id=ANY($3)
            """,
            self.workspace,
            self.namespace,
            node_ids,
        )
        result: dict[str, list[tuple[str, str]]] = {nid: [] for nid in node_ids}
        for r in rows:
            src, tgt = r["src_id"], r["tgt_id"]
            if src in result:
                result[src].append((src, tgt))
            if tgt in result and tgt != src:
                # self-loop (src == tgt) is one edge, not two — match
                # get_node_edges() and NetworkX.
                result[tgt].append((tgt, src))
        for edges in result.values():
            edges.sort(key=lambda edge: edge[1])
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
        # Deduplicate within the batch: last write per node_id wins, matching
        # PGGraphStorage.upsert_nodes_batch and the shared batch-ordering tests.
        # Cross-batch duplicates discard earlier payloads; the surviving payload
        # is still jsonb-merged with any pre-existing DB row by ON CONFLICT below.
        deduped: dict[str, dict[str, str]] = {}
        for node_id, node_data in nodes:
            if "entity_id" not in node_data:
                raise ValueError(
                    "PostgreSQL: node properties must contain an 'entity_id' field"
                )
            node_props = dict(node_data)
            node_props["entity_id"] = node_id
            deduped[node_id] = node_props
        sorted_ids = sorted(deduped)
        props = [json.dumps(deduped[node_id]) for node_id in sorted_ids]
        await self._execute(
            """
            INSERT INTO lightrag_graph_nodes (workspace, namespace, id, properties, updated_at)
            SELECT $1, $2, u.id, u.props::jsonb, now()
            FROM unnest($3::text[], $4::text[]) AS u(id, props)
            ORDER BY u.id
            ON CONFLICT (workspace, namespace, id)
            DO UPDATE SET
                -- Merge (not replace), same as upsert_node — see note there.
                properties = lightrag_graph_nodes.properties || EXCLUDED.properties,
                updated_at = now()
            """,
            self.workspace,
            self.namespace,
            sorted_ids,
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
        sorted_keys = sorted(deduped.keys())
        srcs = [k[0] for k in sorted_keys]
        tgts = [k[1] for k in sorted_keys]
        props = [json.dumps(deduped[k]) for k in sorted_keys]
        endpoint_ids = sorted({nid for key in sorted_keys for nid in key})
        await self._execute(
            """
            WITH endpoints AS (
                INSERT INTO lightrag_graph_nodes (workspace, namespace, id, properties, updated_at)
                SELECT $1, $2, u.id, jsonb_build_object('entity_id', u.id), now()
                FROM unnest($6::text[]) AS u(id)
                ON CONFLICT (workspace, namespace, id) DO NOTHING
                RETURNING id
            ),
            endpoint_write AS (
                SELECT COUNT(*) AS inserted_count FROM endpoints
            )
            INSERT INTO lightrag_graph_edges (workspace, namespace, src_id, tgt_id, properties, updated_at)
            SELECT $1, $2, u.src, u.tgt, u.props::jsonb, now()
            FROM unnest($3::text[], $4::text[], $5::text[]) AS u(src, tgt, props)
            CROSS JOIN endpoint_write
            ORDER BY u.src, u.tgt
            ON CONFLICT (workspace, namespace, src_id, tgt_id)
            DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()
            """,
            self.workspace,
            self.namespace,
            srcs,
            tgts,
            props,
            endpoint_ids,
        )
