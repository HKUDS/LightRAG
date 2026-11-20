"""Retrieval and answer latency through the REAL HTTP retrieval surface.

The harness mounts the production query router, ``create_query_routes(rag)``,
on a minimal FastAPI app and drives it with ``httpx.ASGITransport``: requests
pass through ``QueryRequest`` validation and ``to_query_params`` exactly as
they do on a live server, minus the network hop. This is the documented
direct-call equivalent of POST /query/data and POST /query on a server using
the same LightRAG instance.

Protocol rules:
- Retrieval latency: /query/data with ``only_need_context=True``; answer
  latency: /query with ``only_need_context=False`` so the answer LLM runs.
  Both inject ``hl_keywords``/``ll_keywords`` equal to the query text,
  removing the keyword-extraction LLM from the measured path; the
  hybrid-vs-dense delta then includes the per-query index-validity probe
  cost.
- At least ``latency_queries`` query slots (cycling the dataset queries
  deterministically when fewer exist) times ``latency_rounds`` rounds,
  sequential, one shared client. Every request has a bounded timeout.
- Per-query statistic is the mean over rounds; p50/p95 use nearest-rank
  percentiles; the overhead CI resamples queries paired across arms.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
from dataclasses import dataclass

import httpx
from fastapi import FastAPI

from lightrag import LightRAG
from lightrag.evaluation.hybrid_bench.metrics import percentile


@dataclass(frozen=True, slots=True)
class LatencyResult:
    per_query: list[float]
    all_samples: list[float]
    p50: float
    p95: float
    rounds: int
    query_slots: int


def build_latency_client(rag: LightRAG) -> httpx.AsyncClient:
    # Importing query_routes instantiates the API auth handler, which parses
    # sys.argv on first touch; shield argv so a pytest or harness CLI argv
    # cannot kill the import. Same pattern as the repo's offline query-route
    # tests, test_query_validation_responses.py.
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    try:
        query_routes = importlib.import_module("lightrag.api.routers.query_routes")
    finally:
        sys.argv = original_argv
    app = FastAPI()
    app.include_router(query_routes.create_query_routes(rag))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://hybrid-bench.local",
    )


def select_query_slots(queries: list[str], minimum: int) -> list[str]:
    """Cycle the dataset queries deterministically to reach the minimum."""
    if not queries:
        raise ValueError("no queries available for the latency protocol")
    if minimum <= 0:
        raise ValueError("minimum query slots must be positive")
    return [queries[i % len(queries)] for i in range(max(minimum, len(queries)))]


async def measure_latency(
    client: httpx.AsyncClient,
    query_slots: list[str],
    rounds: int,
    enable_hybrid: bool,
    need_answer: bool,
    top_k: int,
    timeout: float,
) -> LatencyResult:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    path = "/query" if need_answer else "/query/data"
    samples: list[list[float]] = [[] for _ in query_slots]
    for _round in range(rounds):
        for slot, query in enumerate(query_slots):
            payload = {
                "query": query,
                "mode": "mix",
                "top_k": top_k,
                "chunk_top_k": top_k,
                "only_need_context": not need_answer,
                "enable_rerank": False,
                "enable_hybrid": enable_hybrid,
                "stream": False,
                "hl_keywords": [query],
                "ll_keywords": [query],
            }
            start = time.perf_counter()
            response = await asyncio.wait_for(
                client.post(path, json=payload), timeout=timeout
            )
            elapsed = time.perf_counter() - start
            response.raise_for_status()
            body = response.json()
            if need_answer:
                if not body.get("response"):
                    raise RuntimeError(f"answer query returned no response: {body}")
            elif body.get("status") != "success":
                raise RuntimeError(f"/query/data failed: {body}")
            samples[slot].append(elapsed)
    per_query = [sum(values) / len(values) for values in samples]
    flat = [value for values in samples for value in values]
    return LatencyResult(
        per_query=per_query,
        all_samples=flat,
        p50=percentile(flat, 50),
        p95=percentile(flat, 95),
        rounds=rounds,
        query_slots=len(query_slots),
    )
