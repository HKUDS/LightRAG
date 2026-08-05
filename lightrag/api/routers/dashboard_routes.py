"""
Dashboard and knowledge-base routes for the LightRAG API.

This module exposes read-only aggregation endpoints that back the WebUI
dashboard and the knowledge-base browser:

* ``GET /dashboard/stats``       — aggregated counters for the dashboard cards.
* ``GET /dashboard/system``      — runtime/system configuration summary.
* ``GET /knowledge_bases``       — knowledge bases visible to the caller.
* ``GET /knowledge_bases/{id}``  — per-knowledge-base detail.

Knowledge base concept
----------------------
LightRAG isolates data via ``workspace``. A "knowledge base" (KB) is the
user-facing name for a workspace: the same underlying isolation unit, exposed
with a stable id plus aggregated document/graph counters. The server's default
workspace is an internal singleton, not a user-facing KB — it is hidden from
the catalogue and excluded from the dashboard counters, which reflect only the
knowledge bases the user can actually see.

All endpoints are read-only aggregations over existing storages; they never
mutate pipeline state and never take the pipeline lock.
"""

from __future__ import annotations

import asyncio
import re
import time
import traceback
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from lightrag.kg.shared_storage import get_default_workspace
from lightrag.api.routers.knowledge_base_routes import (
    _discover_kbs,
    _KB_RAG_CACHE as _kb_cache,
)
from lightrag.utils import logger

from ..utils_api import get_combined_auth_dependency, internal_server_error

# Document status keys normalised across storage backends.
_STATUS_KEYS = ("pending", "processing", "parsing", "analyzing", "processed", "failed")

# Statuses that mean "work is currently in flight".
_IN_FLIGHT_STATUSES = ("processing", "parsing", "analyzing")

class KnowledgeBaseStats(BaseModel):
    """Aggregated counters for a single knowledge base."""

    documents: int = Field(0, description="Total number of documents")
    processed: int = Field(0, description="Documents fully processed")
    processing: int = Field(0, description="Documents currently in flight")
    pending: int = Field(0, description="Documents waiting to be processed")
    failed: int = Field(0, description="Documents that failed processing")
    entities: int = Field(0, description="Number of entities in the graph")
    relations: int = Field(0, description="Number of relations in the graph")
    chunks: int = Field(0, description="Number of text chunks")


class KnowledgeBase(BaseModel):
    """A knowledge base, i.e. the user-facing view of a workspace."""

    id: str = Field(..., description="Stable knowledge base identifier")
    name: str = Field(..., description="Display name")
    workspace: str = Field(..., description="Underlying workspace identifier")
    is_default: bool = Field(False, description="Whether this is the default KB")
    description: Optional[str] = Field(None, description="Optional description")
    stats: KnowledgeBaseStats = Field(default_factory=KnowledgeBaseStats)


class KnowledgeBaseListResponse(BaseModel):
    knowledge_bases: List[KnowledgeBase]
    total: int
    default_workspace: str


class DashboardStatsResponse(BaseModel):
    """Aggregated metrics rendered by the dashboard cards."""

    knowledge_base_count: int = Field(0, description="Number of knowledge bases")
    document_count: int = Field(0, description="Total documents")
    processed_count: int = Field(0, description="Documents fully processed")
    processing_count: int = Field(0, description="Documents in flight")
    pending_count: int = Field(0, description="Documents pending")
    failed_count: int = Field(0, description="Documents failed")
    entity_count: int = Field(0, description="Entities in the graph")
    relation_count: int = Field(0, description="Relations in the graph")
    chunk_count: int = Field(0, description="Text chunks")
    status_counts: Dict[str, int] = Field(
        default_factory=dict, description="Raw per-status counters"
    )
    pipeline_busy: bool = Field(False, description="Whether the pipeline is busy")
    workspace: str = Field("", description="Workspace these metrics describe")
    generated_at: float = Field(
        default_factory=time.time, description="Unix timestamp of this snapshot"
    )


class SystemInfoResponse(BaseModel):
    """Runtime configuration summary shown in the dashboard's config panel."""

    workspace: str = ""
    core_version: Optional[str] = None
    api_version: Optional[str] = None
    llm_binding: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_binding: Optional[str] = None
    embedding_model: Optional[str] = None
    kv_storage: Optional[str] = None
    vector_storage: Optional[str] = None
    graph_storage: Optional[str] = None
    doc_status_storage: Optional[str] = None
    max_parallel_insert: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap_size: Optional[int] = None


def _normalise_status_counts(raw: Any) -> Dict[str, int]:
    """
    Flatten a backend's status-count mapping into lowercase string keys.

    Backends may key by ``DocStatus`` enum members or by plain strings, so the
    key is coerced through ``.value`` when available before lowercasing. Keys
    outside ``_STATUS_KEYS`` (e.g. the ``all`` total the document routes add
    for the WebUI filter pills) are ignored — they are metadata, not a status,
    and would double-count the document total when summed.
    """
    counts: Dict[str, int] = {key: 0 for key in _STATUS_KEYS}
    if not isinstance(raw, dict):
        return counts

    for key, value in raw.items():
        name = getattr(key, "value", key)
        name = str(name).lower()
        if name not in _STATUS_KEYS:
            continue
        try:
            counts[name] = counts.get(name, 0) + int(value)
        except (TypeError, ValueError):
            continue
    return counts


async def _safe_count(coro_factory, label: str) -> int:
    """
    Await a counter coroutine, degrading to 0 on failure.

    Dashboard aggregation is best-effort: a single unavailable backend must not
    fail the whole response, otherwise the dashboard goes blank whenever an
    optional storage is misconfigured.
    """
    try:
        result = coro_factory()
        if asyncio.iscoroutine(result):
            result = await result
        return int(result or 0)
    except Exception as exc:  # noqa: BLE001 - best-effort aggregation
        logger.debug(f"Dashboard: failed to count {label}: {exc}")
        return 0


async def _count_graph_entities(rag: Any) -> int:
    storage = getattr(rag, "chunk_entity_relation_graph", None)
    if storage is None:
        return 0
    for method in ("get_all_nodes_count", "node_count", "get_node_count"):
        fn = getattr(storage, method, None)
        if callable(fn):
            return await _safe_count(fn, "entities")
    # BaseGraphStorage only guarantees get_all_labels / get_all_nodes, so fall
    # back to materialising them when the backend exposes no cheap counter.
    for method in ("get_all_labels", "get_all_nodes"):
        fn = getattr(storage, method, None)
        if callable(fn):
            try:
                items = await fn()
                return len(items or [])
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Dashboard: failed to count entities via {method}: {exc}")
    return 0


async def _count_graph_relations(rag: Any) -> int:
    storage = getattr(rag, "chunk_entity_relation_graph", None)
    if storage is None:
        return 0
    for method in ("get_all_edges_count", "edge_count", "get_edge_count"):
        fn = getattr(storage, method, None)
        if callable(fn):
            return await _safe_count(fn, "relations")
    edges_fn = getattr(storage, "get_all_edges", None)
    if callable(edges_fn):
        try:
            edges = await edges_fn()
            return len(edges or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Dashboard: failed to count relations via get_all_edges: {exc}")
    return 0


async def _count_chunks(rag: Any) -> int:
    # Prefer the chunk vector storage: PG backends expose a cheap SQL count()
    # there, while the chunk KV storage (``text_chunks``) only enumerates keys
    # on file-based backends and may not implement any counter at all.
    storage = getattr(rag, "chunks_vdb", None)
    if storage is None:
        storage = getattr(rag, "text_chunks", None)
    if storage is None:
        return 0
    for method in ("count", "get_count", "size"):
        fn = getattr(storage, method, None)
        if callable(fn):
            return await _safe_count(fn, "chunks")
    all_keys = getattr(storage, "all_keys", None)
    if callable(all_keys):
        try:
            keys = await all_keys()
            return len(keys or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Dashboard: failed to count chunks via all_keys: {exc}")
    return 0


async def _collect_status_counts(rag: Any) -> Dict[str, int]:
    doc_status = getattr(rag, "doc_status", None)
    if doc_status is None:
        return _normalise_status_counts(None)
    fn = getattr(doc_status, "get_all_status_counts", None)
    if not callable(fn):
        return _normalise_status_counts(None)
    try:
        return _normalise_status_counts(await fn())
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Dashboard: failed to read status counts: {exc}")
        return _normalise_status_counts(None)


async def _is_pipeline_busy(workspace: str) -> bool:
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        status = await get_namespace_data("pipeline_status", workspace=workspace)
        return bool(status.get("busy", False))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Dashboard: failed to read pipeline status: {exc}")
        return False


def _resolve_workspace(rag: Any) -> str:
    workspace = getattr(rag, "workspace", None)
    if workspace:
        return str(workspace)
    try:
        return get_default_workspace() or ""
    except Exception:  # noqa: BLE001
        return ""


def create_dashboard_routes(rag, api_key: Optional[str] = None, args: Any = None, build_rag_instance=None):
    """
    Build the dashboard/knowledge-base router.

    Args:
        rag: The initialised ``LightRAG`` instance.
        api_key: Optional API key enforced through the shared auth dependency.
        args: Parsed server arguments, used for the system-config summary.
        build_rag_instance: Callable(workspace) -> LightRAG for per-KB aggregation.
    """
    combined_auth = get_combined_auth_dependency(api_key)
    router = APIRouter(tags=["dashboard"])

    async def _build_stats(workspace: str) -> DashboardStatsResponse:
        kb_ids = set(_discover_kbs())
        # The server's default workspace is an internal singleton, not a
        # user-facing knowledge base: exclude it from the KB count even when a
        # ``default`` input subdirectory exists.
        kb_ids.discard("default")
        kb_count = len(kb_ids)

        # Aggregate across all knowledge bases
        total_documents = 0
        processed = 0
        processing = 0
        pending = 0
        failed = 0
        entities = 0
        relations = 0
        chunks = 0
        all_status_counts: Dict[str, int] = {}

        async def _accumulate(
            kb_rag, *, count_as_kb: bool = True
        ) -> None:
            nonlocal total_documents, processed, processing, pending, failed
            nonlocal entities, relations, chunks, kb_count
            try:
                status_counts, e, r, c = await asyncio.gather(
                    _collect_status_counts(kb_rag),
                    _count_graph_entities(kb_rag),
                    _count_graph_relations(kb_rag),
                    _count_chunks(kb_rag),
                )
                total_documents += sum(status_counts.values())
                processed += status_counts.get("processed", 0)
                processing += sum(
                    status_counts.get(key, 0) for key in _IN_FLIGHT_STATUSES
                )
                pending += status_counts.get("pending", 0)
                failed += status_counts.get("failed", 0)
                entities += e
                relations += r
                chunks += c
                for key, val in status_counts.items():
                    all_status_counts[key] = all_status_counts.get(key, 0) + val
                if count_as_kb:
                    kb_count += 1
            except Exception as exc:
                logger.debug(f"Dashboard: failed to aggregate KB: {exc}")

        for kb_id in kb_ids:
            try:
                if build_rag_instance is not None:
                    # Reuse cached instances from knowledge_base_routes to avoid
                    # redundant LightRAG construction and storage initialization.
                    cached = _kb_cache.get(kb_id)
                    if cached is not None:
                        kb_rag = cached
                    else:
                        kb_rag = build_rag_instance(kb_id)
                        await kb_rag.initialize_storages()
                        _kb_cache[kb_id] = kb_rag
                else:
                    kb_rag = rag

                await _accumulate(kb_rag, count_as_kb=False)
            except Exception as exc:
                logger.debug(f"Dashboard: failed to aggregate KB {kb_id}: {exc}")

        # The primary workspace is aggregated only when it is a real configured
        # workspace (non-empty): the unconfigured default is an internal
        # singleton, not a knowledge base, so its documents and graph are
        # excluded from the dashboard counters entirely — the cards reflect
        # only the knowledge bases the user can actually see. Skip when it is
        # already counted above as a discovered subdirectory.
        primary_workspace = workspace or ""
        if primary_workspace and primary_workspace not in kb_ids:
            await _accumulate(rag, count_as_kb=True)

        busy = await _is_pipeline_busy(workspace)

        return DashboardStatsResponse(
            knowledge_base_count=kb_count,
            document_count=total_documents,
            processed_count=processed,
            processing_count=processing,
            pending_count=pending,
            failed_count=failed,
            entity_count=entities,
            relation_count=relations,
            chunk_count=chunks,
            status_counts=all_status_counts,
            pipeline_busy=busy,
            workspace=workspace,
        )

    @router.get(
        "/dashboard/stats",
        response_model=DashboardStatsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def dashboard_stats() -> DashboardStatsResponse:
        """
        Aggregated counters for the dashboard summary cards.

        Every counter is best-effort: an unavailable backend contributes 0
        rather than failing the request, so the dashboard degrades gracefully.
        """
        try:
            return await _build_stats(_resolve_workspace(rag))
        except Exception as e:
            logger.error(f"Error building dashboard stats: {e}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    @router.get(
        "/dashboard/system",
        response_model=SystemInfoResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def dashboard_system() -> SystemInfoResponse:
        """Runtime configuration summary for the dashboard's config panel."""
        try:
            from lightrag import __version__ as core_version
            from lightrag.api import __api_version__ as api_version
        except Exception:  # noqa: BLE001
            core_version = None
            api_version = None

        def _arg(name: str, default: Any = None) -> Any:
            return getattr(args, name, default) if args is not None else default

        try:
            return SystemInfoResponse(
                workspace=_resolve_workspace(rag),
                core_version=core_version,
                api_version=api_version,
                llm_binding=_arg("llm_binding"),
                llm_model=_arg("llm_model"),
                embedding_binding=_arg("embedding_binding"),
                embedding_model=_arg("embedding_model"),
                kv_storage=_arg("kv_storage"),
                vector_storage=_arg("vector_storage"),
                graph_storage=_arg("graph_storage"),
                doc_status_storage=_arg("doc_status_storage"),
                max_parallel_insert=_arg("max_parallel_insert"),
                chunk_size=_arg("chunk_size"),
                chunk_overlap_size=_arg("chunk_overlap_size"),
            )
        except Exception as e:
            logger.error(f"Error building system info: {e}")
            logger.error(traceback.format_exc())
            raise internal_server_error(e)

    return router
