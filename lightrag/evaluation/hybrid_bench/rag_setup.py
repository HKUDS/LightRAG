"""LightRAG construction, dataset ingest and scratch cleanup for the bench.

Rules:
- One LightRAG instance per dataset run; storages are initialized before any
  ingest and finalized by the caller's ``finally``.
- Passages ingest as documents with ``ids=corpus_id`` so chunk
  ``full_doc_id`` attribution maps back to qrels corpus ids.
- Extraction cap: the subset ALWAYS contains every qrels-positive passage,
  filled deterministically from the sorted remaining ids up to the cap. The
  subset runs the real pipeline with the answer-LLM binding; every other
  passage ingests chunks-only through ``ainsert_custom_kg`` with no LLM call.
  Cap 0 means chunks-only for the whole corpus.
- ``chunk_token_size`` defaults high so a passage stays one chunk; a longer
  passage splits, and the chunk-to-document map still resolves every chunk.
- ``max_gleaning`` overrides the LightRAG ``entity_extract_max_gleaning``
  constructor knob (0 disables the second extraction pass); None keeps the
  product default, including the MAX_GLEANING environment value.
- ``llm_num_predict`` caps output tokens per LLM call; None leaves the
  output uncapped. Small local models can ramble past the request timeout
  on some passages, which fails documents; a cap bounds the waste and the
  parser keeps the complete records before the cut. On the openai path the
  cap maps to ``max_tokens``.
- Backends: ``ollama`` (local), ``openai`` (any OpenAI-compatible API; see
  ``openai_backend`` for the host/key environment recipe) and the fakes.
  An empty host means unset: ollama falls back to the local ollama URL,
  openai to its binding environment variables.
- Concurrency knobs (``max_parallel_insert``, ``max_async_llm`` ->
  ``llm_model_max_async``, ``embedding_func_max_async``,
  ``embedding_batch_num``) always reach the LightRAG constructor. Defaults
  are aggressive-but-bounded for a network-bound API stack on a small host;
  resumability is the safety net that makes them safe.
- Resume: re-running ``ingest_dataset`` on the same workspace skips
  PROCESSED extraction documents (the product pipeline also auto-resumes
  interrupted PENDING/PARSING/ANALYZING/PROCESSING rows), re-ingests only
  chunks-only documents whose chunk row AND vector are not both present,
  and retries FAILED documents through the sanctioned recovery rounds.
  A fully-ingested dataset returns immediately. Progress checkpoints (loud
  log + ``ingest_progress.json`` under the workspace working dir) mark the
  phase and doc-status counts so a killed run can be resumed by re-invoking
  with the same workspace.
- ``ingest_timeout`` bounds the WHOLE dataset ingest (extraction subset +
  chunks-only remainder + recovery rounds) in seconds; the caller wraps
  ``ingest_dataset`` in ``asyncio.wait_for`` with it.
- ``ingest_recovery_rounds`` bounds FAILED-document recovery: each round
  issues ONE sanctioned sticky manual retry request (the
  ``/documents/reprocess_failed`` commit + drive sequence) granting one
  attempt per FAILED document, with the SAME parameters. 0 disables
  recovery. Any document still FAILED after the rounds raises; recovery
  never swallows a failure.
- Content-duplicate twins: the product pipeline demotes a subset document
  whose content hash already has a holder to a ``dup-<hash>`` FAILED
  pointer row, leaving the twin id without a status row. A twin is
  accepted ONLY when a content-hash pointer row names a PROCESSED
  original AND the product's content-hash lookup proves the twin's text
  hash is held by that same original; the twin's corpus id then joins the
  original's chunk attribution as an alias (``duplicate_aliased_doc_ids``
  in the report). Any other missing status is a silent skip and raises.
- Provider content moderation: a document still FAILED after the recovery
  rounds whose error carries the provider's output-inspection code
  ``data_inspection_failed`` (Aliyun MaaS BadRequestError 400) ingests
  chunks-only through ``ainsert_custom_kg`` instead of aborting
  (``moderation_fallback_doc_ids`` in the report). Its doc_status row
  stays FAILED (truthful: extraction failed; no state surgery), so every
  resume re-runs the bounded rounds and re-applies the idempotent
  fallback. Entities are NOT extracted for it: an honest entity-arm gap.
  Every OTHER permanent failure class still raises.
- PG scratch cleanup deletes ONLY the run's workspace rows and retains the
  shared model/dim vector tables: dropping them on an emptiness check races
  a concurrent run's commits. A workspace override (POSTGRES_WORKSPACE or
  config.ini) fails the run before storage init.
- The LLM cache is DISABLED for the whole run: repeated latency rounds must
  measure real retrieval and answer work, never a cached response, and arms
  must not share warm-cache state.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from uuid import uuid4

from lightrag import LightRAG
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset
from lightrag.evaluation.hybrid_bench.fakes import hash_embed_texts
from lightrag.kg.shared_storage import (
    commit_manual_retry_request,
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
)
from lightrag.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    logger,
    sanitize_text_for_encoding,
)
from lightrag.utils_pipeline import compute_text_content_hash


@dataclass(frozen=True, slots=True)
class BenchConfig:
    storage: str = "pg"
    working_dir: str = "bench_working"
    workspace: str = "hbench"
    embedding_backend: str = "ollama"
    embedding_host: str = ""
    embedding_model: str = "bge-m3:latest"
    embedding_dim: int = 1024
    llm_backend: str = "ollama"
    llm_host: str = ""
    llm_model: str = "qwen2.5:3b"
    llm_enable_thinking: bool | None = None
    max_parallel_insert: int = 8
    max_async_llm: int = 16
    embedding_func_max_async: int = 16
    embedding_batch_num: int = 32
    chunk_token_size: int = 8192
    cosine_threshold: float = 0.2
    extraction_cap: int = 0
    max_gleaning: int | None = None
    llm_num_predict: int | None = None
    ingest_timeout: int = 86400
    ingest_recovery_rounds: int = 2
    top_k: int = 10
    seed: int = 42
    resamples: int = 1000
    latency_queries: int = 100
    latency_rounds: int = 3
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_min_score: float = 0.5
    keep_workspace: bool = False
    request_timeout: float = 300.0

    @property
    def synthetic(self) -> bool:
        return self.embedding_backend == "fake-hash" or self.llm_backend == "fake"

    @property
    def embedding_model_name(self) -> str:
        if self.embedding_backend == "fake-hash":
            return "fakehash"
        return re.sub(r"[^a-zA-Z0-9_]", "_", self.embedding_model)


_OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"
_PROGRESS_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class IngestResult:
    # Chunk ids are content hashes: documents with identical text share one
    # chunk, so the map keeps EVERY corpus alias per chunk id, never the last
    # writer. Scoring expands all aliases of a retrieved chunk.
    chunk_doc_map: dict[str, list[str]]
    extracted_doc_ids: list[str]
    chunks_only_doc_ids: list[str]
    recovery_rounds_used: int = 0
    recovered_doc_ids: list[str] = field(default_factory=list)
    persistently_failed_doc_ids: list[str] = field(default_factory=list)
    # Subset ids demoted by the product's content-hash dedup and aliased to
    # their PROCESSED original's chunk attribution (verified duplicates).
    duplicate_aliased_doc_ids: list[str] = field(default_factory=list)
    # Subset ids whose extraction the provider's output moderation rejected
    # permanently; ingested chunks-only, NO entities (honest entity-arm gap).
    moderation_fallback_doc_ids: list[str] = field(default_factory=list)
    resumed: bool = False

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_doc_map)


def apply_pg_env(config: BenchConfig) -> None:
    """PG connection comes from the standard POSTGRES_* environment recipe;
    the CLI does not invent a second configuration channel.

    A workspace override (POSTGRES_WORKSPACE or config.ini [postgres]
    workspace) remaps every PG storage away from the run's minted workspace
    while cleanup still deletes the minted name, so the benchmark would
    write into the configured workspace and leave its own rows behind:
    fail fast BEFORE any storage is initialized.
    """
    if config.storage != "pg":
        return
    if not os.environ.get("POSTGRES_PASSWORD"):
        raise ValueError(
            "PG storage requires the POSTGRES_* environment recipe "
            "(POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE); export it before "
            "running the harness"
        )
    from lightrag.kg.postgres_impl import ClientManager

    override = ClientManager.get_config().get("workspace")
    if override is not None and str(override).strip():
        raise ValueError(
            f"PG workspace override {str(override).strip()!r} is set "
            "(POSTGRES_WORKSPACE or config.ini [postgres] workspace); it "
            "would remap storage away from the harness workspace while "
            "cleanup deletes the harness name. Unset it before running"
        )


def build_embedding_func(config: BenchConfig) -> EmbeddingFunc:
    if config.embedding_backend == "fake-hash":

        async def fake_embed(texts: list[str], **kwargs) -> object:
            return hash_embed_texts(texts, config.embedding_dim)

        return EmbeddingFunc(
            embedding_dim=config.embedding_dim,
            max_token_size=8192,
            func=fake_embed,
            model_name=config.embedding_model_name,
        )
    if config.embedding_backend == "openai":
        from lightrag.evaluation.hybrid_bench.openai_backend import (
            build_openai_embedding_func,
        )

        return build_openai_embedding_func(config)
    from lightrag.llm.ollama import ollama_embed

    return EmbeddingFunc(
        embedding_dim=config.embedding_dim,
        max_token_size=8192,
        func=partial(
            ollama_embed.func,
            embed_model=config.embedding_model,
            host=config.embedding_host or _OLLAMA_DEFAULT_HOST,
            embedding_dim=config.embedding_dim,
        ),
        model_name=config.embedding_model_name,
    )


def build_llm_func(config: BenchConfig, fake_llm=None):
    if config.llm_backend == "fake":
        if fake_llm is None:
            raise ValueError("fake llm_backend requires a FakeLLM instance")
        return fake_llm
    if config.llm_backend == "openai":
        from lightrag.evaluation.hybrid_bench.openai_backend import (
            build_openai_llm_func,
        )

        return build_openai_llm_func(config)
    from lightrag.llm.ollama import ollama_model_complete

    kwargs: dict = {
        "host": config.llm_host or _OLLAMA_DEFAULT_HOST,
        "timeout": config.request_timeout,
    }
    if config.llm_num_predict is not None:
        kwargs["options"] = {"num_predict": config.llm_num_predict}
    return partial(ollama_model_complete, **kwargs)


async def build_rag(config: BenchConfig, llm_func) -> LightRAG:
    """Construct and initialize one bench instance; caller finalizes."""
    apply_pg_env(config)
    kwargs = dict(
        working_dir=str(Path(config.working_dir) / config.workspace),
        workspace=config.workspace,
        llm_model_func=llm_func,
        llm_model_name=(
            "fake-llm" if config.llm_backend == "fake" else config.llm_model
        ),
        embedding_func=build_embedding_func(config),
        chunk_token_size=config.chunk_token_size,
        chunk_overlap_token_size=0,
        enable_llm_cache=False,
        max_parallel_insert=config.max_parallel_insert,
        llm_model_max_async=config.max_async_llm,
        embedding_func_max_async=config.embedding_func_max_async,
        embedding_batch_num=config.embedding_batch_num,
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": config.cosine_threshold
        },
    )
    if config.max_gleaning is not None:
        kwargs["entity_extract_max_gleaning"] = config.max_gleaning
    if config.storage == "pg":
        kwargs.update(
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGTableGraphStorage",
            doc_status_storage="PGDocStatusStorage",
        )
    rag = LightRAG(**kwargs)
    await rag.initialize_storages()
    return rag


def _record_chunk_alias(
    chunk_doc_map: dict[str, list[str]], chunk_id: str, doc_id: str
) -> None:
    aliases = chunk_doc_map.setdefault(chunk_id, [])
    if doc_id not in aliases:
        aliases.append(doc_id)


def extraction_subset(dataset: BenchDataset, cap: int) -> list[str]:
    """All qrels-positive passages first, then sorted fill up to the cap."""
    if cap <= 0:
        return []
    positives = sorted(dataset.positive_corpus_ids())
    if cap < len(positives):
        raise ValueError(
            f"extraction cap {cap} cannot cover all {len(positives)} "
            "qrels-positive passages; raise the cap"
        )
    rest = sorted(set(dataset.corpus) - set(positives))
    return positives + rest[: cap - len(positives)]


async def _drive_sanctioned_retry(rag: LightRAG) -> str:
    """One sticky manual retry request + pipeline drive, the exact sequence
    of POST /documents/reprocess_failed: ``commit_manual_retry_request``
    (fence recheck + sticky publish in one critical section) then
    ``apipeline_process_enqueue_documents``, which resets FAILED documents
    to PENDING for ONE attempt each and ACKs after the resets persist.
    Never flip doc_status fields directly. A fence refusal raises.
    """
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    ingress = await get_pipeline_ingress(rag.workspace)
    request_id = uuid4().hex
    refusal = await commit_manual_retry_request(
        pipeline_status,
        pipeline_status_lock,
        ingress,
        request_id,
        {"committed": False, "refusal": None},
    )
    if refusal is not None:
        raise RuntimeError(f"manual retry request refused: {refusal}")
    await rag.apipeline_process_enqueue_documents()
    return request_id


def _failed_subset_ids(
    subset: list[str], statuses: dict[str, DocProcessingStatus]
) -> list[str]:
    return [
        cid
        for cid in subset
        if (status := statuses.get(cid)) is not None
        and status.status == DocStatus.FAILED
    ]


# Aliyun MaaS output content inspection: BadRequestError 400 with this code
# rejects extraction output deterministically for flagged content. The code
# survives the pipeline's chunk/batch error prefixes into doc_status
# error_msg; tenacity treats 400 as permanent, so only the sanctioned
# recovery rounds retry it — and for flagged content they cannot succeed.
_PROVIDER_MODERATION_CODE = "data_inspection_failed"


def _is_provider_moderation_failure(error_msg: str | None) -> bool:
    return bool(error_msg) and _PROVIDER_MODERATION_CODE in error_msg


async def _ingest_chunks_only_doc(
    rag: LightRAG, dataset: BenchDataset, cid: str
) -> str:
    """One passage through ``ainsert_custom_kg`` (no LLM call); returns its
    content-hash chunk id. Shared by the chunks-only remainder and the
    moderation fallback so both write the identical payload shape."""
    await rag.ainsert_custom_kg(
        {
            "chunks": [
                {
                    "content": dataset.corpus[cid],
                    "source_id": cid,
                    "file_path": cid,
                    "chunk_order_index": 0,
                }
            ],
            "entities": [],
            "relationships": [],
        },
        full_doc_id=cid,
    )
    return compute_mdhash_id(
        sanitize_text_for_encoding(dataset.corpus[cid]), prefix="chunk-"
    )


async def _duplicate_twin_original(
    rag: LightRAG,
    dataset: BenchDataset,
    cid: str,
    statuses: dict[str, DocProcessingStatus],
) -> str | None:
    """The PROCESSED original for a content-dedup-demoted twin, else None.

    All three proofs are required: the product's content-hash lookup finds
    a holder of the twin's sanitized text hash; that holder is PROCESSED in
    the fetched snapshot; and a ``dup-<hash>`` FAILED pointer row
    (``is_duplicate`` + ``duplicate_kind=content_hash``) with the twin's
    corpus id as file_path names that same holder. None means the missing
    status is NOT a proven duplicate and the caller must fail loud.
    """
    content_hash = compute_text_content_hash(
        sanitize_text_for_encoding(dataset.corpus[cid])
    )
    match = await rag.doc_status.get_doc_by_content_hash(
        content_hash, exclude_doc_id=cid
    )
    if match is None:
        return None
    original = match[0]
    winner = statuses.get(original)
    if winner is None or winner.status != DocStatus.PROCESSED:
        return None
    pointer_found = any(
        status.status == DocStatus.FAILED
        and status.file_path == cid
        and status.metadata.get("is_duplicate")
        and status.metadata.get("duplicate_kind") == "content_hash"
        and status.metadata.get("original_doc_id") == original
        for status in statuses.values()
    )
    return original if pointer_found else None


def _progress_path(config: BenchConfig) -> Path:
    return Path(config.working_dir) / config.workspace / "ingest_progress.json"


def _write_progress(
    config: BenchConfig,
    dataset: BenchDataset,
    phase: str,
    counts: dict[str, int],
) -> None:
    payload = {
        "dataset": dataset.name,
        "workspace": config.workspace,
        "phase": phase,
        "doc_status_counts": counts,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = _progress_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "[hybrid-bench] ingest checkpoint (%s): doc_status counts %s",
        phase,
        counts,
    )


async def _doc_status_counts(rag: LightRAG) -> dict[str, int]:
    statuses = await rag.doc_status.get_docs_by_statuses(list(DocStatus), strict=True)
    counts: dict[str, int] = {}
    for status in statuses.values():
        key = DocStatus(status.status).value
        counts[key] = counts.get(key, 0) + 1
    return counts


async def _progress_monitor(
    rag: LightRAG,
    config: BenchConfig,
    dataset: BenchDataset,
    phase: dict[str, str],
) -> None:
    """Periodic checkpoints during a long ingest. Observability only: a
    failed checkpoint logs loudly and never kills the ingest itself."""
    while True:
        await asyncio.sleep(_PROGRESS_INTERVAL_SECONDS)
        try:
            _write_progress(
                config, dataset, phase["name"], await _doc_status_counts(rag)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[hybrid-bench] progress checkpoint failed: %s", exc)


async def _complete_chunk_doc_ids(
    rag: LightRAG, remainder_chunk_ids: dict[str, str]
) -> set[str]:
    """Chunks-only documents whose chunk row AND vector both exist.

    ``ainsert_custom_kg`` writes exactly those two stores for a chunks-only
    payload; a kill between the two upserts leaves one missing, so only the
    pair proves completeness. A read outage surfaces as absent and the
    document re-ingests, which heals it: repeated upserts are idempotent.
    """
    if not remainder_chunk_ids:
        return set()
    chunk_ids = set(remainder_chunk_ids.values())
    kv_present = chunk_ids - await rag.text_chunks.filter_keys(chunk_ids)
    vectors = await rag.chunks_vdb.get_vectors_by_ids(sorted(chunk_ids))
    return {
        cid
        for cid, chunk_id in remainder_chunk_ids.items()
        if chunk_id in kv_present and chunk_id in vectors
    }


async def ingest_dataset(
    rag: LightRAG, dataset: BenchDataset, config: BenchConfig
) -> IngestResult:
    """Ingest every passage; return the chunk-to-corpus-id attribution map.

    Resumable: on a reused workspace, PROCESSED extraction documents are
    skipped, interrupted rows resume through the product pipeline, and a
    chunks-only document re-ingests unless its chunk row AND vector both
    exist. A fully-ingested dataset returns immediately (``resumed=True``).
    FAILED extraction documents get at most ``config.ingest_recovery_rounds``
    sanctioned retry rounds before the final fail-loud check. Two demotion
    classes are accepted with proof and annotated instead of raising:
    content-duplicate twins (aliased to their PROCESSED original's chunks)
    and provider output-moderation failures (ingested chunks-only, no
    entities). Every other missing status or FAILED document still raises.
    """
    subset = extraction_subset(dataset, config.extraction_cap)
    subset_set = set(subset)
    remainder = [cid for cid in dataset.corpus if cid not in subset_set]
    remainder_chunk_ids = {
        cid: compute_mdhash_id(
            sanitize_text_for_encoding(dataset.corpus[cid]), prefix="chunk-"
        )
        for cid in remainder
    }
    chunk_doc_map: dict[str, list[str]] = {}
    for cid in remainder:
        _record_chunk_alias(chunk_doc_map, remainder_chunk_ids[cid], cid)

    processed = (
        await rag.doc_status.get_docs_by_statuses([DocStatus.PROCESSED], strict=True)
        if subset
        else {}
    )
    pending_subset = [cid for cid in subset if cid not in processed]
    complete_remainder = await _complete_chunk_doc_ids(rag, remainder_chunk_ids)
    pending_remainder = [cid for cid in remainder if cid not in complete_remainder]
    resumed = bool(processed) or bool(complete_remainder)

    phase = {"name": "extracting" if pending_subset else "chunks-only"}
    monitor = asyncio.create_task(_progress_monitor(rag, config, dataset, phase))
    try:
        _write_progress(config, dataset, phase["name"], await _doc_status_counts(rag))
        if not pending_subset and not pending_remainder:
            for cid in subset:
                for chunk_id in processed[cid].chunks_list or []:
                    _record_chunk_alias(chunk_doc_map, chunk_id, cid)
            logger.info(
                "[hybrid-bench] resume: workspace %s already holds the full "
                "dataset (%d extracted docs PROCESSED, %d chunks-only docs "
                "complete); skipping straight to retrieval",
                config.workspace,
                len(subset),
                len(remainder),
            )
            phase["name"] = "complete"
            return IngestResult(
                chunk_doc_map=chunk_doc_map,
                extracted_doc_ids=subset,
                chunks_only_doc_ids=remainder,
                resumed=True,
            )
        if pending_subset:
            if processed:
                logger.info(
                    "[hybrid-bench] resume: %d extraction docs already "
                    "PROCESSED, continuing with %d",
                    len(processed),
                    len(pending_subset),
                )
            await rag.ainsert(
                [dataset.corpus[cid] for cid in pending_subset],
                ids=pending_subset,
                file_paths=pending_subset,
            )
        phase["name"] = "chunks-only"
        for cid in pending_remainder:
            # Remainder aliases are pre-recorded above for the whole remainder;
            # this only writes the stores for the not-yet-complete documents.
            await _ingest_chunks_only_doc(rag, dataset, cid)

        recovery_rounds_used = 0
        recovered_doc_ids: list[str] = []
        persistently_failed_doc_ids: list[str] = []
        duplicate_aliased_doc_ids: list[str] = []
        moderation_fallback_doc_ids: list[str] = []
        if subset:
            statuses = await rag.doc_status.get_docs_by_statuses(
                [DocStatus.PROCESSED, DocStatus.FAILED], strict=True
            )
            failed = _failed_subset_ids(subset, statuses)
            while failed and recovery_rounds_used < config.ingest_recovery_rounds:
                recovery_rounds_used += 1
                logger.warning(
                    "[hybrid-bench] ingest recovery round %d/%d: retrying %d "
                    "FAILED document(s) via the sanctioned manual retry path "
                    "(same parameters): %s",
                    recovery_rounds_used,
                    config.ingest_recovery_rounds,
                    len(failed),
                    failed,
                )
                await asyncio.wait_for(
                    _drive_sanctioned_retry(rag), timeout=config.ingest_timeout
                )
                statuses = await rag.doc_status.get_docs_by_statuses(
                    [DocStatus.PROCESSED, DocStatus.FAILED], strict=True
                )
                still_failed = _failed_subset_ids(subset, statuses)
                recovered = [cid for cid in failed if cid not in set(still_failed)]
                recovered_doc_ids.extend(recovered)
                logger.warning(
                    "[hybrid-bench] ingest recovery round %d: recovered %s; "
                    "still FAILED %s",
                    recovery_rounds_used,
                    recovered,
                    still_failed,
                )
                failed = still_failed
            for cid in failed:
                if _is_provider_moderation_failure(statuses[cid].error_msg):
                    chunk_id = await _ingest_chunks_only_doc(rag, dataset, cid)
                    _record_chunk_alias(chunk_doc_map, chunk_id, cid)
                    moderation_fallback_doc_ids.append(cid)
                    logger.warning(
                        "[hybrid-bench] provider content-moderation fallback: "
                        "%s ingested chunks-only after %d recovery round(s); "
                        "NO entities extracted (entity-arm gap annotated in "
                        "the report); doc_status stays FAILED: %s",
                        cid,
                        recovery_rounds_used,
                        statuses[cid].error_msg,
                    )
            persistently_failed_doc_ids = [
                cid for cid in failed if cid not in moderation_fallback_doc_ids
            ]
            for cid in subset:
                status = statuses.get(cid)
                if status is None:
                    original = await _duplicate_twin_original(
                        rag, dataset, cid, statuses
                    )
                    if original is None:
                        raise RuntimeError(f"ingested document {cid!r} has no status")
                    duplicate_aliased_doc_ids.append(cid)
                    logger.info(
                        "[hybrid-bench] content-duplicate twin %s aliased to "
                        "PROCESSED original %s (batch_duplicate demotion "
                        "accepted; chunk attribution shared)",
                        cid,
                        original,
                    )
                    for chunk_id in statuses[original].chunks_list or []:
                        _record_chunk_alias(chunk_doc_map, chunk_id, cid)
                    continue
                if (
                    status.status == DocStatus.FAILED
                    and cid not in moderation_fallback_doc_ids
                ):
                    detail = f"document {cid!r} failed ingestion: {status.error_msg}"
                    if recovery_rounds_used:
                        detail += (
                            f" (recovery rounds used: {recovery_rounds_used}; "
                            f"recovered: {recovered_doc_ids}; "
                            f"persistently failed: {persistently_failed_doc_ids})"
                        )
                    raise RuntimeError(detail)
                # PROCESSED documents and moderation-fallback documents alike:
                # the fallback doc's failed-attempt chunks stay in the stores
                # (stage 1 writes them before extraction) and keep attribution.
                for chunk_id in status.chunks_list or []:
                    _record_chunk_alias(chunk_doc_map, chunk_id, cid)
        phase["name"] = "complete"
        return IngestResult(
            chunk_doc_map=chunk_doc_map,
            extracted_doc_ids=subset,
            chunks_only_doc_ids=remainder,
            recovery_rounds_used=recovery_rounds_used,
            recovered_doc_ids=recovered_doc_ids,
            persistently_failed_doc_ids=persistently_failed_doc_ids,
            duplicate_aliased_doc_ids=duplicate_aliased_doc_ids,
            moderation_fallback_doc_ids=moderation_fallback_doc_ids,
            resumed=resumed,
        )
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass
        try:
            _write_progress(
                config, dataset, phase["name"], await _doc_status_counts(rag)
            )
        except Exception as exc:
            logger.warning("[hybrid-bench] final progress checkpoint failed: %s", exc)


def _pg_connect_kwargs(pg: dict) -> dict:
    """asyncpg.connect kwargs mirroring the storage pool's connection setup,
    including the POSTGRES_SSL_* context/cert settings, so cleanup reaches
    SSL-required servers exactly like the measured path. write_amp_driver
    carries a self-contained mirror of this helper (baseline contract)."""
    from lightrag.kg.postgres_impl import PostgreSQLDB

    db = PostgreSQLDB(pg)
    kwargs = {
        "host": pg["host"],
        "port": int(pg["port"]),
        "user": pg["user"],
        "password": pg["password"],
        "database": pg["database"],
    }
    ssl_context = db._create_ssl_context()
    if ssl_context is not None:
        kwargs["ssl"] = ssl_context
    elif db.ssl_mode and db.ssl_mode.lower() in ("require", "prefer"):
        kwargs["ssl"] = True
    elif db.ssl_mode and db.ssl_mode.lower() == "disable":
        kwargs["ssl"] = False
    return kwargs


async def cleanup_pg_workspace(config: BenchConfig) -> None:
    """Delete ONLY the run's workspace rows; shared model/dim vector tables
    are RETAINED. An emptiness-check-then-DROP races a concurrent run that
    can commit rows between the check and the drop, losing them; leftover
    empty tables are harmless. Explicit per-table DELETEs: dynamic DO-block
    discovery has silently skipped tables before."""
    import asyncpg

    from lightrag.kg.postgres_impl import ClientManager

    pg = ClientManager.get_config()
    conn = await asyncpg.connect(**_pg_connect_kwargs(pg))
    try:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname='public' AND tablename LIKE 'lightrag_%'"
        )
        for row in tables:
            table = row["tablename"]
            has_workspace = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=$1 "
                "AND column_name='workspace')",
                table,
            )
            if has_workspace:
                await conn.execute(
                    f'DELETE FROM "{table}" WHERE workspace=$1',
                    config.workspace,
                )
    finally:
        await conn.close()
