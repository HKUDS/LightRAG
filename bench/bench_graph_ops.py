"""
Graph storage performance benchmark: 4 backends × synthetic dataset.

Backends : NetworkXStorage, PgRcteGraphStorage, PGGraphStorage(AGE), Neo4JStorage
Dataset  : Barabási-Albert scale-free graph (seed=42, fully reproducible)
           Default: 8,000 nodes / ~40,000 edges

Measures ONLY BaseGraphStorage method call latency — no LLM involved.

All PG backends use identical POSTGRES_MAX_CONNECTIONS so the RPS comparison
is apples-to-apples across pool configurations.

Workload mix (approximates LightRAG indexing + retrieval load):
  28%  get_node              — retrieval: fetch entity properties
  18%  get_node_edges        — retrieval: expand neighbours
  10%  node_degree           — retrieval: rank by connectivity
   8%  has_node              — indexing: existence check
   8%  upsert_node           — indexing: write/update entity
   8%  get_nodes_batch       — retrieval: batch property fetch (real call pattern)
   8%  get_knowledge_graph   — retrieval: recursive BFS subgraph (expensive)
   6%  upsert_nodes_batch    — indexing: batch entity write
   6%  upsert_edges_batch    — indexing: batch edge write

Usage:
    python bench/bench_graph_ops.py --workers 10 --duration 30 --output results.json
    python bench/bench_graph_ops.py --nodes 4000 --workers 20 --duration 60 --output results.json

Sample raw results (n2-standard-8):
    https://gist.github.com/ysys143/94de2e121282f9177613ec72e0100af1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

import networkx as nx
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# ---------------------------------------------------------------------------
# Pool size — identical across all PG backends for a fair comparison
# ---------------------------------------------------------------------------

BENCH_POOL_SIZE = 20

BACKENDS = [
    {
        "name": "NetworkX",
        "graph_storage": "NetworkXStorage",
        "env": {},
    },
    {
        "name": "RCTE",
        "graph_storage": "PgRcteGraphStorage",
        "env": {
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5434",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "postgres",
            "POSTGRES_DATABASE": "lightrag_test",
            "POSTGRES_MAX_CONNECTIONS": str(BENCH_POOL_SIZE),
        },
    },
    {
        "name": "AGE",
        "graph_storage": "PGGraphStorage",
        "env": {
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5435",
            "POSTGRES_USER": "bench",
            "POSTGRES_PASSWORD": "benchmark123",
            "POSTGRES_DATABASE": "lightrag_age",
            "POSTGRES_MAX_CONNECTIONS": str(BENCH_POOL_SIZE),
        },
    },
    {
        "name": "Neo4j",
        "graph_storage": "Neo4JStorage",
        "env": {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password123",
        },
    },
]

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def synthetic_dataset(n_nodes: int = 8000, m: int = 5) -> tuple[list, list]:
    """Barabási-Albert scale-free graph — realistic hub structure."""
    print(f"  Generating BA graph: {n_nodes} nodes, m={m}...")
    G = nx.barabasi_albert_graph(n_nodes, m, seed=42)
    nodes = []
    for i in G.nodes:
        nodes.append(
            (
                f"Entity_{i:05d}",
                {
                    "entity_id": f"Entity_{i:05d}",
                    "entity_type": random.choice(
                        ["concept", "process", "system", "metric"]
                    ),
                    "description": f"Synthetic entity number {i} in scale-free graph.",
                    "source_id": f"chunk-{i % 500:04d}",
                    "created_at": 1767000000 + i,
                },
            )
        )
    edges = []
    for src, tgt in G.edges:
        edges.append(
            (
                f"Entity_{src:05d}",
                f"Entity_{tgt:05d}",
                {
                    "weight": round(random.uniform(0.3, 2.0), 2),
                    "description": f"Relationship between entity {src} and {tgt}.",
                    "keywords": "synthetic,test",
                    "source_id": f"chunk-{src % 500:04d}",
                    "created_at": 1767000000 + src,
                },
            )
        )
    print(f"  Generated {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


# ---------------------------------------------------------------------------
# Storage factory
# ---------------------------------------------------------------------------


async def make_storage(backend: dict, workspace: str):
    import importlib
    from lightrag.kg import STORAGES
    from lightrag.kg.shared_storage import initialize_share_data

    for k, v in backend.get("env", {}).items():
        os.environ[k] = v

    name = backend["graph_storage"]

    # Reset shared ClientManager pool so each backend gets its own connection.
    if name in ("PGGraphStorage", "PgRcteGraphStorage"):
        from lightrag.kg.postgres_impl import ClientManager

        ClientManager._instances = {
            "db": None,
            "ref_count": 0,
            "vector_signature": None,
        }

    initialize_share_data()

    module = importlib.import_module(STORAGES[name], package="lightrag")
    cls = getattr(module, name)

    global_config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        "working_dir": f"/tmp/bench_{workspace}",
        "max_graph_nodes": 1000,
    }
    store = cls(
        namespace="bench",
        workspace=workspace,
        global_config=global_config,
        embedding_func=lambda texts: np.random.rand(len(texts), 10),
    )
    await store.initialize()
    return store


# ---------------------------------------------------------------------------
# Seed (batch write path — exercises upsert_nodes_batch / upsert_edges_batch)
# ---------------------------------------------------------------------------

_SEED_CHUNK = 500


async def seed(store, nodes: list, edges: list) -> float:
    t0 = time.monotonic()
    for i in range(0, len(nodes), _SEED_CHUNK):
        await store.upsert_nodes_batch(nodes[i : i + _SEED_CHUNK])
    for i in range(0, len(edges), _SEED_CHUNK):
        await store.upsert_edges_batch(edges[i : i + _SEED_CHUNK])
    return time.monotonic() - t0


# ---------------------------------------------------------------------------
# Benchmark worker
# ---------------------------------------------------------------------------


async def _worker(store, node_ids: list, duration: float, results: list):
    """Workload mirrors LightRAG's actual graph storage call pattern.

    Ref: lightrag/operate.py — extract_entities (indexing) + query paths (retrieval)
    Includes get_knowledge_graph and batch methods that the real paths use.
    """
    end = time.monotonic() + duration
    while time.monotonic() < end:
        nid = random.choice(node_ids)
        roll = random.random()
        t0 = time.monotonic()
        try:
            if roll < 0.28:
                await store.get_node(nid)
                op = "get_node"
            elif roll < 0.46:
                await store.get_node_edges(nid)
                op = "get_node_edges"
            elif roll < 0.56:
                await store.node_degree(nid)
                op = "node_degree"
            elif roll < 0.64:
                await store.has_node(nid)
                op = "has_node"
            elif roll < 0.72:
                await store.upsert_node(nid, {"entity_id": nid, "entity_type": "bench"})
                op = "upsert_node"
            elif roll < 0.80:
                batch_ids = random.choices(node_ids, k=5)
                await store.get_nodes_batch(batch_ids)
                op = "get_nodes_batch"
            elif roll < 0.88:
                await store.get_knowledge_graph(nid, max_depth=2, max_nodes=50)
                op = "get_knowledge_graph"
            elif roll < 0.94:
                batch = [
                    (n, {"entity_id": n, "entity_type": "bench"})
                    for n in random.choices(node_ids, k=3)
                ]
                await store.upsert_nodes_batch(batch)
                op = "upsert_nodes_batch"
            else:
                pairs = [
                    (
                        random.choice(node_ids),
                        random.choice(node_ids),
                        {"weight": "1.0"},
                    )
                    for _ in range(3)
                ]
                await store.upsert_edges_batch(pairs)
                op = "upsert_edges_batch"
            results.append((op, (time.monotonic() - t0) * 1000))
        except Exception as exc:
            results.append((f"ERR:{type(exc).__name__}", 0))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _collect_results(results: list, elapsed: float) -> dict:
    ok = [(op, lat) for op, lat in results if not op.startswith("ERR")]
    errs = [op for op, _ in results if op.startswith("ERR")]
    lats = sorted(lat for _, lat in ok)

    per_op: dict[str, list[float]] = {}
    for op, lat in ok:
        per_op.setdefault(op, []).append(lat)

    return {
        "rps": len(ok) / elapsed if elapsed > 0 else 0,
        "total_ops": len(ok),
        "errors": len(errs),
        "p50_ms": statistics.median(lats) if lats else 0,
        "p95_ms": lats[int(len(lats) * 0.95)] if lats else 0,
        "p99_ms": lats[int(len(lats) * 0.99)] if lats else 0,
        "per_op": {
            op: {
                "count": len(v),
                "p50_ms": round(statistics.median(v), 2),
                "p95_ms": round(sorted(v)[int(len(v) * 0.95)], 2),
            }
            for op, v in sorted(per_op.items())
        },
    }


def _print_results(backend: str, seed_s: float, stats: dict):
    print(
        f"\n  [{backend}]  seed={seed_s:.1f}s  "
        f"ops={stats['total_ops']}  err={stats['errors']}  "
        f"RPS={stats['rps']:.0f}  "
        f"p50={stats['p50_ms']:.1f}ms  "
        f"p95={stats['p95_ms']:.1f}ms  "
        f"p99={stats['p99_ms']:.1f}ms"
    )
    for op, s in stats["per_op"].items():
        print(
            f"    {op:<26}  n={s['count']:<6}  p50={s['p50_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_dataset(
    dataset_name: str, nodes: list, edges: list, workers: int, duration: float
) -> list[dict]:
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}  ({len(nodes)} nodes / {len(edges)} edges)")
    print(f"Pool size: {BENCH_POOL_SIZE} (all PG backends)")
    print(f"{'=' * 60}")
    node_ids = [n[0] for n in nodes]
    summary = []

    for backend in BACKENDS:
        ws = f"bench_{dataset_name}_{backend['name'].lower()}"
        print(f"\n  Backend: {backend['name']}")
        try:
            store = await make_storage(backend, ws)
            await store.drop()

            print("    Seeding (batch)...", end=" ", flush=True)
            seed_s = await seed(store, nodes, edges)
            print(f"{seed_s:.1f}s")

            print(
                f"    Benchmarking {workers} workers × {duration}s...",
                end=" ",
                flush=True,
            )
            results: list = []
            t0 = time.monotonic()
            await asyncio.gather(
                *[_worker(store, node_ids, duration, results) for _ in range(workers)]
            )
            elapsed = time.monotonic() - t0
            print("done")

            stats = _collect_results(results, elapsed)
            _print_results(backend["name"], seed_s, stats)
            summary.append(
                {
                    "dataset": dataset_name,
                    "backend": backend["name"],
                    "pool_size": BENCH_POOL_SIZE,
                    "workers": workers,
                    "duration_s": duration,
                    "seed_s": round(seed_s, 3),
                    **{
                        k: round(v, 3) if isinstance(v, float) else v
                        for k, v in stats.items()
                        if k != "per_op"
                    },
                    "per_op": stats["per_op"],
                }
            )
            await store.finalize()
        except Exception as exc:
            print(f"    FAILED: {exc}")
            summary.append(
                {
                    "dataset": dataset_name,
                    "backend": backend["name"],
                    "error": str(exc)[:120],
                }
            )

    # Summary table
    print(f"\n  --- {dataset_name} Summary ---")
    print(
        f"  {'Backend':<12} {'Seed(s)':<9} {'RPS':<8} {'p50(ms)':<10} {'p95(ms)':<10} {'Errors'}"
    )
    print(f"  {'-' * 56}")
    for r in summary:
        if "error" in r:
            print(f"  {r['backend']:<12} FAILED: {r['error']}")
        else:
            print(
                f"  {r['backend']:<12} {r['seed_s']:<9.1f} {r['rps']:<8.0f} "
                f"{r['p50_ms']:<10.1f} {r['p95_ms']:<10.1f} {r['errors']}"
            )

    return summary


async def main(nodes: int, workers: int, duration: float, output: str | None):
    node_list, edge_list = synthetic_dataset(n_nodes=nodes)
    results = await run_dataset("synthetic", node_list, edge_list, workers, duration)

    if output:
        out_path = Path(output)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nodes",
        type=int,
        default=8000,
        help="Number of nodes in the BA graph (default: 8000)",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--output", type=str, default=None, help="Write raw JSON results to this file"
    )
    args = parser.parse_args()
    asyncio.run(main(args.nodes, args.workers, args.duration, args.output))
