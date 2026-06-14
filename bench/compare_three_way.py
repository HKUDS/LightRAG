"""
Three-way comparison: Neo4j vs PG+RCTE vs PG+AGE

Equivalence  — same data in, same results out
Performance  — RPS and latency for LightRAG query mix

Usage:
    python bench/compare_three_way.py [--mode equiv|perf|both]
                                      [--workers N] [--duration S] [--nodes N]

Container assumptions (all running locally):
    PG+RCTE  : localhost:5434  postgres/postgres/lightrag_test
    PG+AGE   : localhost:5433  bench/benchmark123/graphdb
    Neo4j    : bolt://localhost:7687  neo4j/password123
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import time

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# ---------------------------------------------------------------------------
# Backend factory helpers
# ---------------------------------------------------------------------------


async def make_rcte(workspace: str, pool_size: int = 20):
    from lightrag.kg.postgres_impl import ClientManager

    ClientManager._instances = {"db": None, "ref_count": 0, "vector_signature": None}
    os.environ.update(
        {
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5434",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "postgres",
            "POSTGRES_DATABASE": "lightrag_test",
            "POSTGRES_MAX_CONNECTIONS": str(pool_size),
        }
    )
    from lightrag.kg.pg_rcte_impl import PgRcteGraphStorage
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)
    store = PgRcteGraphStorage(
        namespace="compare",
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
    )
    await store.initialize()
    return store


async def make_neo4j(workspace: str):
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password123"
    os.environ["NEO4J_WORKSPACE"] = workspace
    from lightrag.kg.neo4j_impl import Neo4JStorage

    store = Neo4JStorage(
        namespace="compare",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace=workspace,
    )
    await store.initialize()
    return store


async def make_age(workspace: str, pool_size: int = 20):
    from lightrag.kg.postgres_impl import ClientManager

    ClientManager._instances = {"db": None, "ref_count": 0, "vector_signature": None}
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = "5435"
    os.environ["POSTGRES_USER"] = "bench"
    os.environ["POSTGRES_PASSWORD"] = "benchmark123"
    os.environ["POSTGRES_DATABASE"] = "lightrag_age"
    os.environ["POSTGRES_MAX_CONNECTIONS"] = str(pool_size)
    os.environ["PG_WORKSPACE"] = workspace
    from lightrag.kg.postgres_impl import PGGraphStorage

    store = PGGraphStorage(
        namespace="compare",
        global_config={"vector_storage": "PGVectorStorage", "max_graph_nodes": 1000},
        embedding_func=None,
        workspace=workspace,
    )
    await store.initialize()
    return store


class AgeDirectStore:
    """Minimal AGE store using asyncpg + cypher() directly."""

    GRAPH = "lightrag_compare"

    def __init__(self, workspace: str):
        self.workspace = workspace
        self._conn = None

    async def initialize(self):
        import asyncpg

        self._conn = await asyncpg.connect(
            "postgresql://bench:benchmark123@localhost:5433/graphdb"
        )
        await self._conn.execute("SET search_path = ag_catalog, '$user', public")
        try:
            await self._conn.execute(f"SELECT create_graph('{self.GRAPH}')")
        except Exception:
            pass  # already exists

    async def finalize(self):
        if self._conn:
            await self._conn.close()

    async def drop(self):
        try:
            await self._conn.execute(self._cypher("MATCH (n) DETACH DELETE n"))
        except Exception:
            pass

    def _cypher(self, q: str) -> str:
        return f"SELECT * FROM cypher('{self.GRAPH}', $${q}$$) AS (r agtype)"

    async def upsert_node(self, node_id: str, data: dict):
        eid = node_id.replace("'", "\\'")
        desc = data.get("description", "").replace("'", "\\'")
        await self._conn.execute(
            self._cypher(
                f"MERGE (n:Entity {{entity_id:'{eid}'}}) "
                f"SET n.description='{desc}' RETURN n"
            )
        )

    async def upsert_edge(self, src: str, tgt: str, data: dict):
        s = src.replace("'", "\\'")
        t = tgt.replace("'", "\\'")
        await self._conn.execute(
            self._cypher(
                f"MATCH (a:Entity {{entity_id:'{s}'}}), (b:Entity {{entity_id:'{t}'}}) "
                f"MERGE (a)-[:RELATED]->(b) RETURN a"
            )
        )

    async def has_node(self, node_id: str) -> bool:
        eid = node_id.replace("'", "\\'")
        rows = await self._conn.fetch(
            self._cypher(f"MATCH (n:Entity {{entity_id:'{eid}'}}) RETURN n")
        )
        return len(rows) > 0

    async def has_edge(self, src: str, tgt: str) -> bool:
        s = src.replace("'", "\\'")
        t = tgt.replace("'", "\\'")
        rows = await self._conn.fetch(
            self._cypher(
                f"MATCH (a:Entity {{entity_id:'{s}'}})-[r]-(b:Entity {{entity_id:'{t}'}}) RETURN r"
            )
        )
        return len(rows) > 0

    async def node_degree(self, node_id: str) -> int:
        eid = node_id.replace("'", "\\'")
        rows = await self._conn.fetch(
            self._cypher(
                f"MATCH (n:Entity {{entity_id:'{eid}'}})-[r]-() RETURN count(r) AS c"
            )
        )
        if not rows:
            return 0
        import json

        return int(json.loads(rows[0]["r"]))

    async def get_all_labels(self) -> list[str]:
        rows = await self._conn.fetch(
            self._cypher("MATCH (n:Entity) RETURN n.entity_id AS r")
        )
        import json

        return [json.loads(r["r"]) for r in rows]

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ):
        from lightrag.types import KnowledgeGraph, KnowledgeGraphNode
        import json

        if node_label == "*":
            seed_rows = await self._conn.fetch(
                self._cypher(
                    f"MATCH (n:Entity) RETURN n.entity_id AS r LIMIT {max_nodes}"
                )
            )
        else:
            seed_rows = await self._conn.fetch(
                self._cypher(
                    f"MATCH (n:Entity) WHERE n.entity_id CONTAINS '{node_label}' RETURN n.entity_id AS r"
                )
            )
        seeds = [json.loads(r["r"]) for r in seed_rows]
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(max_depth):
            if not frontier or len(visited) >= max_nodes:
                break
            ids_list = ", ".join(
                f"'{s.replace(chr(39), chr(92) + chr(39))}'" for s in frontier
            )
            if not ids_list:
                break
            rows = await self._conn.fetch(
                self._cypher(
                    f"MATCH (a:Entity)-[r]-(b:Entity) WHERE a.entity_id IN [{ids_list}] RETURN b.entity_id AS r"
                )
            )
            next_frontier = set()
            for row in rows:
                nid = json.loads(row["r"])
                if nid not in visited:
                    next_frontier.add(nid)
                    visited.add(nid)
            frontier = next_frontier
        nodes = [
            KnowledgeGraphNode(id=nid, labels=[nid], properties={}) for nid in visited
        ]
        return KnowledgeGraph(
            nodes=nodes, edges=[], is_truncated=len(visited) >= max_nodes
        )

    async def get_node(self, node_id: str):
        return None  # not needed for perf

    async def get_node_edges(self, node_id: str):
        eid = node_id.replace("'", "\\'")
        rows = await self._conn.fetch(
            self._cypher(
                f"MATCH (a:Entity {{entity_id:'{eid}'}})-[r]-(b:Entity) RETURN b.entity_id AS r"
            )
        )
        import json

        return [(node_id, json.loads(row["r"])) for row in rows]


# ---------------------------------------------------------------------------
# Small graph seed data
# ---------------------------------------------------------------------------

_NODES = {
    "alice": {"entity_id": "alice", "entity_type": "person", "description": "Alice"},
    "bob": {"entity_id": "bob", "entity_type": "person", "description": "Bob"},
    "carol": {"entity_id": "carol", "entity_type": "person", "description": "Carol"},
    "dave": {"entity_id": "dave", "entity_type": "person", "description": "Dave"},
    "eve": {"entity_id": "eve", "entity_type": "person", "description": "Eve"},
}
_EDGES = [
    ("alice", "bob", {"weight": "1.0", "description": "knows"}),
    ("alice", "carol", {"weight": "0.8", "description": "works_with"}),
    ("bob", "dave", {"weight": "0.5", "description": "follows"}),
    ("carol", "eve", {"weight": "0.6", "description": "friends"}),
]


async def seed(store) -> None:
    for nid, data in _NODES.items():
        await store.upsert_node(nid, data)
    for src, tgt, data in _EDGES:
        await store.upsert_edge(src, tgt, data)


# ---------------------------------------------------------------------------
# Equivalence check
# ---------------------------------------------------------------------------


async def check_equiv(label: tuple[str, str], a, b) -> list[str]:
    """Return list of mismatches between store a and store b (read ops)."""
    mismatches = []

    # has_node
    for nid in list(_NODES.keys()) + ["ghost"]:
        va = await a.has_node(nid)
        vb = await b.has_node(nid)
        if va != vb:
            mismatches.append(f"has_node({nid!r}): {label[0]}={va} {label[1]}={vb}")

    # get_node — compare shared keys
    for nid in _NODES:
        va = await a.get_node(nid)
        vb = await b.get_node(nid)
        if (va is None) != (vb is None):
            mismatches.append(f"get_node({nid!r}) None mismatch")
        elif va and vb:
            for key in ("entity_type", "description"):
                if va.get(key) != vb.get(key):
                    mismatches.append(
                        f"get_node({nid!r})[{key}]: "
                        f"{label[0]}={va.get(key)!r} {label[1]}={vb.get(key)!r}"
                    )

    # node_degree
    for nid in _NODES:
        va = await a.node_degree(nid)
        vb = await b.node_degree(nid)
        if va != vb:
            mismatches.append(f"node_degree({nid!r}): {label[0]}={va} {label[1]}={vb}")

    # has_edge / get_edge
    for src, tgt, edge_data in _EDGES:
        for s, t in [(src, tgt), (tgt, src)]:
            va = await a.has_edge(s, t)
            vb = await b.has_edge(s, t)
            if va != vb:
                mismatches.append(f"has_edge({s},{t}): {label[0]}={va} {label[1]}={vb}")
        ea = await a.get_edge(src, tgt)
        eb = await b.get_edge(src, tgt)
        if (ea is None) != (eb is None):
            mismatches.append(f"get_edge({src},{tgt}) None mismatch")
        elif ea and eb:
            for key in ("weight", "description"):
                if key in edge_data and ea.get(key) != eb.get(key):
                    mismatches.append(
                        f"get_edge({src},{tgt})[{key}]: "
                        f"{label[0]}={ea.get(key)!r} {label[1]}={eb.get(key)!r}"
                    )

    # get_node_edges
    for nid in _NODES:
        va = await a.get_node_edges(nid)
        vb = await b.get_node_edges(nid)
        if va is not None and vb is not None:
            va_set = {frozenset(e) for e in va}
            vb_set = {frozenset(e) for e in vb}
            if va_set != vb_set:
                mismatches.append(
                    f"get_node_edges({nid!r}): "
                    f"only_a={va_set - vb_set} only_b={vb_set - va_set}"
                )
        elif (va is None) != (vb is None):
            mismatches.append(
                f"get_node_edges({nid!r}): {label[0]}={va} {label[1]}={vb}"
            )

    # get_all_labels
    va = set(await a.get_all_labels())
    vb = set(await b.get_all_labels())
    if va != vb:
        mismatches.append(f"get_all_labels: only_a={va - vb} only_b={vb - va}")

    # get_knowledge_graph node sets.
    # Neo4JStorage's APOC path sets KnowledgeGraphNode.id to an internal
    # integer — use labels[0] (always entity_id) for portable comparison.
    def _entity_ids(kg) -> set:
        return {
            (n.labels[0] if n.labels else None) or n.properties.get("entity_id") or n.id
            for n in kg.nodes
        }

    for q_label in ["alice", "*"]:
        for depth in [1, 2]:
            kg_a = await a.get_knowledge_graph(q_label, max_depth=depth, max_nodes=100)
            kg_b = await b.get_knowledge_graph(q_label, max_depth=depth, max_nodes=100)
            ids_a = _entity_ids(kg_a)
            ids_b = _entity_ids(kg_b)
            if ids_a != ids_b:
                mismatches.append(
                    f"kg({q_label!r},d={depth}) nodes: "
                    f"only_a={ids_a - ids_b} only_b={ids_b - ids_a}"
                )

    return mismatches


async def check_write_equiv(label: tuple[str, str], a, b) -> list[str]:
    """Test write and delete operations produce equivalent state.

    Re-seeds both stores before each check to ensure a clean baseline.
    """
    mismatches = []

    # --- upsert_node update ---
    await a.drop()
    await b.drop()
    await seed(a)
    await seed(b)
    upd = {
        "entity_id": "alice",
        "entity_type": "updated_org",
        "description": "Alice v2",
    }
    await a.upsert_node("alice", upd)
    await b.upsert_node("alice", upd)
    va = await a.get_node("alice")
    vb = await b.get_node("alice")
    if va and vb:
        if va.get("entity_type") != vb.get("entity_type"):
            mismatches.append(
                f"upsert_node update entity_type: "
                f"{label[0]}={va.get('entity_type')!r} {label[1]}={vb.get('entity_type')!r}"
            )

    # --- upsert_edge update ---
    await a.upsert_edge("alice", "bob", {"weight": "9.9", "description": "upgraded"})
    await b.upsert_edge("alice", "bob", {"weight": "9.9", "description": "upgraded"})
    ea = await a.get_edge("alice", "bob")
    eb = await b.get_edge("alice", "bob")
    if (ea is None) != (eb is None):
        mismatches.append("upsert_edge update: None mismatch")
    elif ea and eb and ea.get("weight") != eb.get("weight"):
        mismatches.append(
            f"upsert_edge update weight: "
            f"{label[0]}={ea.get('weight')!r} {label[1]}={eb.get('weight')!r}"
        )

    # --- delete_node cascades to edges ---
    await a.drop()
    await b.drop()
    await seed(a)
    await seed(b)
    await a.delete_node("eve")
    await b.delete_node("eve")
    for nid in ("eve",):
        va_has = await a.has_node(nid)
        vb_has = await b.has_node(nid)
        if va_has != vb_has:
            mismatches.append(
                f"delete_node({nid!r}) has_node: {label[0]}={va_has} {label[1]}={vb_has}"
            )
    for s, t in [("carol", "eve"), ("eve", "carol")]:
        va_e = await a.has_edge(s, t)
        vb_e = await b.has_edge(s, t)
        if va_e != vb_e:
            mismatches.append(
                f"delete_node cascade has_edge({s},{t}): "
                f"{label[0]}={va_e} {label[1]}={vb_e}"
            )

    # --- remove_edges ---
    await a.drop()
    await b.drop()
    await seed(a)
    await seed(b)
    await a.remove_edges([("alice", "bob")])
    await b.remove_edges([("alice", "bob")])
    va_e = await a.has_edge("alice", "bob")
    vb_e = await b.has_edge("alice", "bob")
    if va_e != vb_e:
        mismatches.append(
            f"remove_edges([alice,bob]): {label[0]}={va_e} {label[1]}={vb_e}"
        )
    # Unrelated edge must survive
    va_e2 = await a.has_edge("alice", "carol")
    vb_e2 = await b.has_edge("alice", "carol")
    if va_e2 != vb_e2:
        mismatches.append(
            f"remove_edges survival alice-carol: {label[0]}={va_e2} {label[1]}={vb_e2}"
        )

    return mismatches


# ---------------------------------------------------------------------------
# Performance worker
# ---------------------------------------------------------------------------


async def _worker(store, duration: float, n_nodes: int, results: list):
    node_ids = [f"e_{i}" for i in range(n_nodes)]
    end = time.monotonic() + duration
    while time.monotonic() < end:
        roll = random.random()
        t0 = time.monotonic()
        try:
            if roll < 0.40:
                await store.has_node(random.choice(node_ids))
                op = "has_node"
            elif roll < 0.60:
                await store.get_node(random.choice(node_ids))
                op = "get_node"
            elif roll < 0.70:
                await store.get_node_edges(random.choice(node_ids))
                op = "get_node_edges"
            elif roll < 0.85:
                nid = random.choice(node_ids)
                await store.upsert_node(nid, {"entity_id": nid, "entity_type": "bench"})
                op = "upsert_node"
            elif roll < 0.90:
                src, tgt = random.sample(node_ids, 2)
                await store.upsert_edge(src, tgt, {"weight": "1.0"})
                op = "upsert_edge"
            else:
                label = f"e_{random.randint(0, n_nodes // 10)}"
                await store.get_knowledge_graph(label, max_depth=2, max_nodes=50)
                op = "get_knowledge_graph"
            results.append((op, time.monotonic() - t0))
        except Exception as exc:
            results.append((f"ERR:{exc.__class__.__name__}", time.monotonic() - t0))


async def seed_bench(store, n_nodes: int):
    for i in range(n_nodes):
        nid = f"e_{i}"
        await store.upsert_node(nid, {"entity_id": nid, "entity_type": "bench"})
    for i in range(n_nodes):
        src, tgt = f"e_{i}", f"e_{random.randint(0, n_nodes - 1)}"
        await store.upsert_edge(src, tgt, {"weight": "1.0"})


def _print_perf(name: str, results: list, elapsed: float):
    ok = [(op, lat) for op, lat in results if not op.startswith("ERR")]
    err = [op for op, _ in results if op.startswith("ERR")]
    lats = sorted(lat * 1000 for _, lat in ok)
    print(f"\n  [{name}]")
    print(f"    RPS      : {len(ok) / elapsed:.0f}")
    print(f"    Errors   : {len(err)} ({100 * len(err) / max(len(results), 1):.1f}%)")
    if lats:
        print(f"    p50      : {statistics.median(lats):.2f} ms")
        print(f"    p95      : {lats[int(len(lats) * 0.95)]:.2f} ms")
        print(f"    p99      : {lats[int(len(lats) * 0.99)]:.2f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_equiv(pool_size: int = 20):
    print(f"\n=== Equivalence Check (pool_size={pool_size}) ===")

    backends = {}
    errors = {}

    for name, factory, ws in [
        ("RCTE", lambda w: make_rcte(w, pool_size), "cmp_rcte"),
        ("Neo4j", make_neo4j, "cmp_neo4j"),
        ("AGE", lambda w: make_age(w, pool_size), "cmp_age"),
    ]:
        try:
            print(f"  Initializing {name}...", end=" ", flush=True)
            store = await factory(ws)
            await seed(store)
            backends[name] = store
            print("OK")
        except Exception as exc:
            errors[name] = str(exc)
            print(f"FAILED: {exc}")

    pairs = [
        ("RCTE", "Neo4j"),
        ("RCTE", "AGE"),
    ]
    for a_name, b_name in pairs:
        if a_name not in backends or b_name not in backends:
            print(f"  {a_name} vs {b_name}: skipped (init failed)")
            continue
        print(f"\n  --- {a_name} vs {b_name}: read ops ---")
        mismatches = await check_equiv(
            (a_name, b_name), backends[a_name], backends[b_name]
        )
        if mismatches:
            print(f"  {len(mismatches)} MISMATCH(ES)")
            for m in mismatches:
                print(f"    - {m}")
        else:
            print("  ALL MATCH")

        print(f"\n  --- {a_name} vs {b_name}: write/delete ops ---")
        write_mismatches = await check_write_equiv(
            (a_name, b_name), backends[a_name], backends[b_name]
        )
        if write_mismatches:
            print(f"  {len(write_mismatches)} MISMATCH(ES)")
            for m in write_mismatches:
                print(f"    - {m}")
        else:
            print("  ALL MATCH")

    # Cleanup
    for name, store in backends.items():
        try:
            await store.drop()
            await store.finalize()
        except Exception:
            pass


async def run_perf(workers: int, duration: float, n_nodes: int, pool_size: int = 20):
    print(
        f"\n=== Performance: {workers} workers, {duration}s, {n_nodes} nodes, pool={pool_size} ==="
    )

    for name, factory, ws in [
        ("RCTE", lambda w: make_rcte(w, pool_size), "perf_rcte"),
        ("Neo4j", make_neo4j, "perf_neo4j"),
        ("AGE", lambda w: make_age(w, pool_size), "perf_age"),
    ]:
        try:
            print(f"  Seeding {name}...", end=" ", flush=True)
            store = await factory(ws)
            await store.drop()
            await seed_bench(store, n_nodes)
            print("done")
            results = []
            t_start = time.monotonic()
            await asyncio.gather(
                *[_worker(store, duration, n_nodes, results) for _ in range(workers)]
            )
            elapsed = time.monotonic() - t_start
            _print_perf(name, results, elapsed)
            await store.drop()
            await store.finalize()
        except Exception as exc:
            print(f"  {name}: FAILED — {exc}")


async def main(mode: str, workers: int, duration: float, n_nodes: int, pool_size: int):
    if mode in ("equiv", "both"):
        await run_equiv(pool_size)
    if mode in ("perf", "both"):
        await run_perf(workers, duration, n_nodes, pool_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["equiv", "perf", "both"], default="both")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument(
        "--pool",
        type=int,
        default=20,
        help="Max connections per PG backend (same for all, for fair comparison)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.workers, args.duration, args.nodes, args.pool))
