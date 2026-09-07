#!/usr/bin/env python3
"""Offline chunk-tracking repair tool (issue #3838, R4).

The tool rebuilds ``entity_chunks`` and ``relation_chunks`` from cached
extraction results. It is deliberately offline-only: a standalone process
cannot share the API server's in-memory pipeline reservations, and dropping a
whole namespace while any server, worker, or SDK writer is active can silently
lose concurrent updates.

Run without ``--apply`` to build and print a read-only repair plan. Before an
apply, stop every writer that uses the same backing storages and workspace.
The complete plan is computed before the first drop, so a source read failure
leaves tracking untouched. An interrupted apply is not atomic across the two
namespaces; keep the server stopped and re-run the tool until it succeeds.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from dotenv import load_dotenv

from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc, logger, make_relation_chunk_key, setup_logger

_CHUNK_SCAN_BATCH = 200
_UPSERT_BATCH = 500


@dataclass
class ChunkTrackingRepairReport:
    """Counts and warnings for a repair plan or completed apply."""

    scanned_documents: int = 0
    scanned_chunks: int = 0
    chunks_with_cache: int = 0
    chunks_without_cache: int = 0
    graph_entities: int = 0
    graph_relations: int = 0
    entity_rows_planned: int = 0
    relation_rows_planned: int = 0
    entity_rows_written: int = 0
    relation_rows_written: int = 0
    entities_without_evidence: int = 0
    relations_without_evidence: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkTrackingRepairPlan:
    """Stable replacement payload computed before any destructive mutation."""

    report: ChunkTrackingRepairReport
    entity_rows: dict[str, list[str]] = field(repr=False)
    relation_rows: dict[str, list[str]] = field(repr=False)

    @property
    def unsafe_empty_namespaces(self) -> list[str]:
        """Namespaces a server restart would repopulate from graph source_id."""
        unsafe: list[str] = []
        if self.report.graph_entities and not self.entity_rows:
            unsafe.append("entity_chunks")
        if self.report.graph_relations and not self.relation_rows:
            unsafe.append("relation_chunks")
        return unsafe


async def build_chunk_tracking_repair_plan(rag) -> ChunkTrackingRepairPlan:
    """Build a complete, read-only replacement plan for one workspace.

    The graph is an existence oracle only. Attribution is reconstructed at
    chunk granularity from ``text_chunks.llm_cache_list`` and the extraction
    cache; graph ``source_id`` and document-level anchors are never seeds.
    """
    _require_storages(rag)
    report = ChunkTrackingRepairReport()

    chunk_ids, report.scanned_documents = await _collect_corpus_chunk_ids(rag)
    report.scanned_chunks = len(chunk_ids)

    graph_entities = set(await rag.chunk_entity_relation_graph.get_all_labels())
    graph_relations = await _collect_graph_relation_keys(rag)
    report.graph_entities = len(graph_entities)
    report.graph_relations = len(graph_relations)

    entity_rows: dict[str, list[str]] = {}
    relation_rows: dict[str, list[str]] = {}
    if chunk_ids and rag.llm_response_cache is not None:
        (
            report.chunks_with_cache,
            report.chunks_without_cache,
        ) = await _accumulate_cached_attribution(
            rag,
            chunk_ids,
            graph_entities,
            graph_relations,
            entity_rows,
            relation_rows,
        )
    else:
        report.chunks_without_cache = len(chunk_ids)
        if rag.llm_response_cache is None:
            report.warnings.append(
                "LLM response cache is not configured, so no chunk-level "
                "attribution can be recovered."
            )

    report.entity_rows_planned = len(entity_rows)
    report.relation_rows_planned = len(relation_rows)
    report.entities_without_evidence = len(graph_entities - set(entity_rows))
    report.relations_without_evidence = len(graph_relations - set(relation_rows))
    _add_plan_warnings(report)

    plan = ChunkTrackingRepairPlan(report, entity_rows, relation_rows)
    if plan.unsafe_empty_namespaces:
        report.warnings.append(
            f"Refusing apply while {', '.join(plan.unsafe_empty_namespaces)} would "
            "be empty although the graph contains corresponding objects. A server "
            "restart would re-seed the empty namespace from KEEP-truncated graph "
            "source_id; restore the extraction cache or re-ingest first."
        )

    for warning in report.warnings:
        logger.warning(f"Chunk tracking repair plan: {warning}")
    return plan


async def apply_chunk_tracking_repair_plan(
    rag, plan: ChunkTrackingRepairPlan
) -> ChunkTrackingRepairReport:
    """Apply a precomputed plan, failing before drop for unsafe empty output."""
    _require_storages(rag)
    if plan.unsafe_empty_namespaces:
        raise ValueError(
            "Refusing to drop tracking: "
            f"{', '.join(plan.unsafe_empty_namespaces)} would be empty while the "
            "graph still contains corresponding objects. Restore the extraction "
            "cache or re-ingest, then build a new repair plan."
        )

    await _drop_tracking_namespace(rag.entity_chunks, "entity_chunks")
    await _drop_tracking_namespace(rag.relation_chunks, "relation_chunks")

    plan.report.entity_rows_written = await _write_tracking_rows(
        rag.entity_chunks, plan.entity_rows, "entity_chunks"
    )
    plan.report.relation_rows_written = await _write_tracking_rows(
        rag.relation_chunks, plan.relation_rows, "relation_chunks"
    )
    logger.info(
        "Chunk tracking repair completed: "
        f"{plan.report.entity_rows_written} entity row(s), "
        f"{plan.report.relation_rows_written} relation row(s) rebuilt from "
        f"{plan.report.chunks_with_cache}/{plan.report.scanned_chunks} cached chunk(s)"
    )
    return plan.report


def _require_storages(rag) -> None:
    required = (
        "doc_status",
        "text_chunks",
        "chunk_entity_relation_graph",
        "entity_chunks",
        "relation_chunks",
    )
    missing = [name for name in required if getattr(rag, name, None) is None]
    if missing:
        raise ValueError(
            f"Chunk tracking repair requires configured storages: {', '.join(missing)}"
        )


async def _collect_corpus_chunk_ids(rag) -> tuple[list[str], int]:
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus), strict=True)
    ordered: dict[str, None] = {}
    contributing_docs = 0
    for doc in docs.values():
        chunks_list = getattr(doc, "chunks_list", None) or []
        if not chunks_list:
            continue
        contributing_docs += 1
        for chunk_id in chunks_list:
            if chunk_id:
                ordered.setdefault(chunk_id, None)
    return list(ordered), contributing_docs


async def _collect_graph_relation_keys(rag) -> set[str]:
    keys: set[str] = set()
    for edge in await rag.chunk_entity_relation_graph.get_all_edges():
        src = edge.get("source") or edge.get("src_id") or edge.get("src")
        tgt = edge.get("target") or edge.get("tgt_id") or edge.get("tgt")
        if src and tgt:
            keys.add(make_relation_chunk_key(src, tgt))
    return keys


async def _accumulate_cached_attribution(
    rag,
    chunk_ids: list[str],
    graph_entities: set[str],
    graph_relations: set[str],
    entity_rows: dict[str, list[str]],
    relation_rows: dict[str, list[str]],
) -> tuple[int, int]:
    from lightrag.operate import (
        _get_cached_extraction_results,
        _rebuild_from_extraction_result,
    )

    with_cache = 0
    for start in range(0, len(chunk_ids), _CHUNK_SCAN_BATCH):
        batch = chunk_ids[start : start + _CHUNK_SCAN_BATCH]
        missing = await rag.text_chunks.filter_keys(set(batch))
        batch = [chunk_id for chunk_id in batch if chunk_id not in missing]
        if not batch:
            continue
        cached_results, chunk_data_by_id = await _get_cached_extraction_results(
            rag.llm_response_cache,
            set(batch),
            text_chunks_storage=rag.text_chunks,
        )
        for chunk_id, results in cached_results.items():
            parsed_any = False
            for extraction_result, timestamp in results:
                try:
                    entities, relationships = await _rebuild_from_extraction_result(
                        text_chunks_storage=rag.text_chunks,
                        chunk_id=chunk_id,
                        extraction_result=extraction_result,
                        timestamp=timestamp,
                        chunk_data=chunk_data_by_id.get(chunk_id),
                    )
                except Exception as exc:
                    logger.warning(
                        "Chunk tracking repair: unusable cached extraction "
                        f"result for chunk {chunk_id}: {exc}"
                    )
                    continue
                parsed_any = True
                for entity_name in entities:
                    if entity_name in graph_entities:
                        _append_chunk_id(entity_rows, entity_name, chunk_id)
                for src, tgt in relationships:
                    storage_key = make_relation_chunk_key(src, tgt)
                    if storage_key in graph_relations:
                        _append_chunk_id(relation_rows, storage_key, chunk_id)
            if parsed_any:
                with_cache += 1
    return with_cache, len(chunk_ids) - with_cache


def _add_plan_warnings(report: ChunkTrackingRepairReport) -> None:
    if report.chunks_without_cache:
        report.warnings.append(
            f"{report.chunks_without_cache} of {report.scanned_chunks} chunk(s) "
            "had no usable cached extraction result; objects supported only by "
            "them will have no tracking row."
        )
    if report.entities_without_evidence or report.relations_without_evidence:
        report.warnings.append(
            f"{report.entities_without_evidence} entity/entities and "
            f"{report.relations_without_evidence} relation(s) present in the "
            "graph have no cached chunk-level evidence."
        )


async def _drop_tracking_namespace(storage, label: str) -> None:
    result = await storage.drop()
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError(f"Failed to drop {label} during repair: {result}")


async def _write_tracking_rows(storage, rows: dict[str, list[str]], label: str) -> int:
    keys = list(rows)
    written = 0
    for start in range(0, len(keys), _UPSERT_BATCH):
        payload = {
            key: {"chunk_ids": rows[key], "count": len(rows[key])}
            for key in keys[start : start + _UPSERT_BATCH]
        }
        await storage.upsert(payload)
        written += len(payload)
    await storage.index_done_callback()
    logger.info(f"Chunk tracking repair: rebuilt {written} {label} row(s)")
    return written


def _append_chunk_id(rows: dict[str, list[str]], key: str, chunk_id: str) -> None:
    bucket = rows.setdefault(key, [])
    if chunk_id not in bucket:
        bucket.append(chunk_id)


def _print_report(report: ChunkTrackingRepairReport, *, applied: bool) -> None:
    phase = "Apply result" if applied else "Repair plan"
    print(f"\n{phase}:")
    print(
        f"  Corpus: {report.scanned_documents} document(s), "
        f"{report.scanned_chunks} chunk(s)"
    )
    print(
        f"  Cache: {report.chunks_with_cache} usable chunk(s), "
        f"{report.chunks_without_cache} unavailable"
    )
    print(
        f"  Graph: {report.graph_entities} entity/entities, "
        f"{report.graph_relations} relation(s)"
    )
    print(
        f"  Planned rows: {report.entity_rows_planned} entity, "
        f"{report.relation_rows_planned} relation"
    )
    if applied:
        print(
            f"  Written rows: {report.entity_rows_written} entity, "
            f"{report.relation_rows_written} relation"
        )
    for warning in report.warnings:
        print(f"  WARNING: {warning}")


def _confirm_offline(assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(
        "Have all LightRAG servers, workers, and SDK writers for this workspace "
        "been stopped? (yes/no): "
    )
    return answer.strip().lower() == "yes"


def _confirm_apply(assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(
        "Type REPAIR to drop and rebuild entity_chunks and relation_chunks: "
    )
    return answer.strip() == "REPAIR"


def _storage_description(storage) -> str:
    details: list[str] = []
    workspace = getattr(storage, "workspace", None)
    namespace = getattr(storage, "namespace", None)
    if workspace is not None:
        details.append(f"workspace={workspace!r}")
    if namespace is not None:
        details.append(f"namespace={namespace!r}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{type(storage).__name__}{suffix}"


async def _build_rag():
    from lightrag import LightRAG

    async def _noop_llm(*_args, **_kwargs) -> str:
        raise RuntimeError("chunk_tracking_repair never calls the LLM")

    async def _noop_embed(_texts):
        raise RuntimeError("chunk_tracking_repair never embeds")

    return LightRAG(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        workspace=os.getenv("WORKSPACE", ""),
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        doc_status_storage=os.getenv(
            "LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
        ),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=_noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=8192,
            func=_noop_embed,
            model_name=os.getenv("EMBEDDING_MODEL"),
        ),
    )


async def run(args: argparse.Namespace) -> bool:
    """Run the offline tool; return False for any unsafe or failed apply."""
    print("LightRAG Offline Chunk-Tracking Repair Tool")
    print("STOP every server, worker, and SDK writer for the target workspace.")
    if not _confirm_offline(args.yes):
        print("Operation cancelled. Stop all writers before running this tool.")
        return True

    rag = await _build_rag()
    await rag.initialize_storages()
    try:
        print(f"Working directory: {rag.working_dir}")
        print(f"Requested workspace: {rag.workspace or '(default)'}")
        print(f"Graph storage: {_storage_description(rag.chunk_entity_relation_graph)}")
        print(f"Tracking KV storage: {_storage_description(rag.entity_chunks)}")
        print(f"Doc-status storage: {_storage_description(rag.doc_status)}")

        plan = await build_chunk_tracking_repair_plan(rag)
        _print_report(plan.report, applied=False)
        if not args.apply:
            print("Dry run only. Re-run with --apply to rebuild tracking.")
            return True
        if plan.unsafe_empty_namespaces:
            print("Apply refused before drop; tracking was not modified.")
            return False
        if not _confirm_apply(args.yes):
            print("Apply cancelled; tracking was not modified.")
            return True

        report = await apply_chunk_tracking_repair_plan(rag, plan)
        _print_report(report, applied=True)
        print("Repair completed successfully. It is now safe to restart LightRAG.")
        return True
    except Exception as exc:
        logger.exception("Chunk tracking repair failed")
        print(f"Repair failed: {exc}")
        if args.apply:
            print(
                "Keep every LightRAG writer stopped. The tracking namespaces may "
                "be partially rebuilt; fix the cause and re-run until successful."
            )
        return False
    finally:
        await rag.finalize_storages()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline repair of entity/relation chunk tracking."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Drop and rebuild tracking; default is a read-only repair plan.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm all writers are stopped and accept the destructive apply.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(dotenv_path=".env", override=False)
    setup_logger("lightrag", level="INFO")
    if not asyncio.run(run(_parse_args())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
