#!/usr/bin/env python3
"""Offline GC for chunk-scoped LLM cache rows whose owning chunk is gone.

``delete_llm_cache=True`` removes a document's cache rows by following each
chunk's ``llm_cache_list``. A row whose key was never attached to its chunk
is invisible to that path, so the chunks are deleted, the deletion is
reported as ``success``, and the row stays behind. The row holds the whole
extraction prompt (chunk text included) and the extracted result, so this is
an incomplete-deletion residue, not just bytes.

Ordering closes the window on new data (a row is written only after its
reference is recorded), so what remains is historical rows plus the rows a
hard kill lost mid-flight. Reclaiming those requires listing the cache
namespace, which is an offline-only, deliberate expense:

    rag = LightRAG(...)
    await rag.initialize_storages()
    report = await gc_orphan_llm_cache(rag)              # report only
    report = await gc_orphan_llm_cache(rag, apply=True)  # delete the orphans

Usage — the CLI builds the storages the server's environment describes
(``WORKING_DIR``, ``WORKSPACE``, ``LIGHTRAG_KV_STORAGE`` /
``LIGHTRAG_GRAPH_STORAGE`` / ``LIGHTRAG_DOC_STATUS_STORAGE``, each backend's
own connection variables) and prints the cache storage it opened, so a sweep
against the wrong backend is visible instead of reported as an empty
workspace. The vector store is a no-op because nothing here reads it. A
deployment that builds its LightRAG another way should call
``gc_orphan_llm_cache`` from a script wired to that construction:

    python -m lightrag.tools.gc_llm_cache [--apply] [--batch-size N] [--verbose]

Scope rules
-----------
- Only rows carrying a non-empty ``chunk_id`` are candidates. Query and
  keywords rows carry ``chunk_id=None`` by construction and belong to
  ``lightrag-clean-llmqc``; the ``summary``, ``smartheading`` and ``analysis``
  artifacts carry no chunk reference either, so no liveness question can be
  asked about any of them. They are counted as ``unscoped_rows`` and left
  alone: deleting a row a sweep cannot justify is how it loses data the
  operator still wants. A document deletion still reclaims ``smartheading``
  and multimodal ``analysis`` rows through its own pool; only ``summary`` has
  no deletion path at all. Rows predating the field are the same case.
- A candidate is deleted only once its chunk's absence is *confirmed*:
  ``text_chunks.get_by_id_strict`` answers ``None`` for that chunk. The bulk
  ``filter_keys`` answer is only a candidate filter, because it fails open on
  a backend whose index is not ready (OpenSearch reports every key absent),
  which would empty the cache. A backend without
  ``supports_strict_point_reads`` is refused, and a strict read that cannot
  confirm absence aborts the sweep before the first delete.
- A row whose chunk still exists is left alone even when no chunk references
  it: its owning document is alive, and deleting the row would re-bill the
  next reprocess for nothing.
- Rows a document deletion deliberately retained count as dead here, because
  the chunk they name is gone. That is the operator's call, which is why the
  default is a report and ``--apply`` must be asked for.
- A sweep that fails part-way is safe to re-run: deleted batches stay deleted
  (nothing is restored), and the rows that remain are still orphans.

Run it while the server is stopped. The shared pipeline status that would let
a live process refuse the sweep is per-process for the file-backed KV stores,
so a CLI run cannot see an ingestion happening in another process.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, AsyncIterator

from lightrag.utils import logger

DEFAULT_BATCH_SIZE = 500

# Bounded sample carried in the report so a huge workspace cannot turn a dry
# run into an unbounded print. The counts are exact either way.
SAMPLE_LIMIT = 20


async def _iter_key_batches(kv, batch_size: int) -> AsyncIterator[list[str]]:
    """Yield the storage's namespace keys in batches.

    ``BaseKVStorage`` has no enumeration API today, so this delegates to the
    shared per-backend scan the other offline tools use; that scan materializes
    the key list, and the batches bound the row round trips that follow. The
    indirection is deliberate: when bounded KV iteration lands on
    ``BaseKVStorage``, this is the one function that has to change.

    A key the scan repeats is collapsed: it would otherwise be counted twice
    and "deleted" twice, so ``cache_rows`` and ``deleted_rows`` would stop
    describing the namespace. Key identity belongs to the namespace, not to the
    scan.
    """
    from lightrag.tools.rebuild_vdb import enumerate_kv_keys

    keys = await enumerate_kv_keys(kv)
    keys = list(dict.fromkeys(keys))
    for start in range(0, len(keys), batch_size):
        yield keys[start : start + batch_size]


async def gc_orphan_llm_cache(
    rag,
    *,
    apply: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Sweep chunk-scoped LLM cache rows whose owning chunk no longer exists.

    Args:
        rag: An initialized ``LightRAG`` (``initialize_storages()`` awaited).
        apply: ``False`` (default) reports; ``True`` deletes the orphans.
        batch_size: Keys per storage round trip.

    Returns:
        dict[str, Any]: counts and a bounded key sample:
        - ``cache_rows``: rows read from the cache namespace;
        - ``chunk_scoped_rows``: rows carrying a ``chunk_id`` (the candidates);
        - ``unscoped_rows``: rows without one — query/keywords answers
          (``lightrag-clean-llmqc`` owns those), the summary/smartheading/
          analysis artifacts that carry no chunk reference, and rows predating
          the field — counted and left alone;
        - ``orphan_rows``: candidates whose chunk is missing;
        - ``live_rows``: candidates whose chunk still exists;
        - ``deleted_rows``: rows removed (``apply=True`` only);
        - ``orphan_keys_sample`` / ``deleted_keys_sample``: up to
          ``SAMPLE_LIMIT`` keys, for the operator to eyeball.

    Raises:
        ValueError: ``batch_size`` is below 1, the cache / text-chunks storage
            is not configured, or the chunks backend cannot confirm absence
            (``supports_strict_point_reads`` is false).
        RuntimeError: A batch reported success but its rows are still present,
            so the sweep did not actually delete what it said it did. Where a
            backend stages deletes (OpenSearch) a tombstone that a retryable
            flush failure left buffered still reads as absent, so in that case
            the check proves acceptance rather than publication; the storage's
            own ``finalize()`` raises when operations remain buffered, but
            ``LightRAG.finalize_storages()`` logs a per-storage failure instead
            of re-raising, so the count can be printed with the CLI still
            exiting 0.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    cache_kv = rag.llm_response_cache
    chunks_kv = rag.text_chunks
    if cache_kv is None:
        raise ValueError("LLM response cache storage is not configured")
    if chunks_kv is None:
        raise ValueError("text chunks storage is not configured")

    report: dict[str, Any] = {
        "cache_rows": 0,
        "chunk_scoped_rows": 0,
        "unscoped_rows": 0,
        "orphan_rows": 0,
        "live_rows": 0,
        "deleted_rows": 0,
        "orphan_keys_sample": [],
        "deleted_keys_sample": [],
    }

    # chunk_id -> [cache key] for every row naming a chunk. Only keys are
    # held: retaining the rows would make the sweep's memory grow with the
    # whole cache payload (a stranded row stores its chunk text), which is the
    # cost the offline-only contract exists to bound.
    scoped: dict[str, list[str]] = {}

    async for batch in _iter_key_batches(cache_kv, batch_size):
        rows = await cache_kv.get_by_ids(batch)
        for key, row in zip(batch, rows):
            if not isinstance(row, dict):
                # Concurrently dropped, or a backend that never had it.
                continue
            report["cache_rows"] += 1
            chunk_id = row.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                report["unscoped_rows"] += 1
                continue
            report["chunk_scoped_rows"] += 1
            scoped.setdefault(chunk_id, []).append(key)

    # A "missing" answer from ``filter_keys`` is only a candidate: that read
    # fails open on a backend whose index is not ready (OpenSearch reports
    # every key absent), which would make every chunk-scoped row an orphan and
    # let ``--apply`` empty the cache. Absence is therefore confirmed one id at
    # a time, and a backend that cannot answer that way is not swept at all.
    if not getattr(chunks_kv, "supports_strict_point_reads", False):
        raise ValueError(
            f"{type(chunks_kv).__name__} cannot confirm that a chunk is absent "
            "(supports_strict_point_reads=False); refusing to delete cache "
            "rows on an unconfirmed miss"
        )

    candidates: list[str] = []
    chunk_ids = sorted(scoped)
    for start in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[start : start + batch_size]
        missing = await chunks_kv.filter_keys(set(batch_ids))
        candidates.extend(chunk_id for chunk_id in batch_ids if chunk_id in missing)

    orphan_ids: list[str] = []
    misread = 0
    for start in range(0, len(candidates), batch_size):
        batch_ids = candidates[start : start + batch_size]
        # ``None`` is confirmed absence. Anything else is a chunk that exists
        # (the bulk read was wrong about it), and an exception means absence
        # could not be confirmed: that aborts the sweep here, before the first
        # delete, with nothing removed.
        confirmed = await asyncio.gather(
            *(chunks_kv.get_by_id_strict(chunk_id) for chunk_id in batch_ids)
        )
        for chunk_id, row in zip(batch_ids, confirmed):
            if row is None:
                orphan_ids.append(chunk_id)
            else:
                misread += 1
    if misread:
        logger.warning(
            f"[gc-llm-cache] {type(chunks_kv).__name__} reported {misread} "
            "existing chunk(s) as absent; they were confirmed present and kept"
        )

    orphan_keys = [key for chunk_id in orphan_ids for key in scoped[chunk_id]]
    report["orphan_rows"] = len(orphan_keys)
    report["live_rows"] = report["chunk_scoped_rows"] - report["orphan_rows"]
    report["orphan_keys_sample"] = sorted(orphan_keys)[:SAMPLE_LIMIT]

    if not apply or not orphan_keys:
        return report

    for start in range(0, len(orphan_keys), batch_size):
        batch_keys = orphan_keys[start : start + batch_size]
        await cache_kv.delete(batch_keys)
        # The drop is only durable once the owning storage commits. ``delete``
        # itself reports nothing, so a backend whose delete is a no-op would
        # otherwise be indistinguishable from one that worked.
        await cache_kv.index_done_callback()
        # ``filter_keys`` answers with the keys that are ABSENT, so the keys
        # still present are what is left of the batch after subtracting them.
        absent = await cache_kv.filter_keys(set(batch_keys))
        still_present = set(batch_keys) - absent
        if still_present:
            raise RuntimeError(
                f"Cache GC deleted {len(batch_keys)} key(s) but "
                f"{len(still_present)} are still present; the sweep did not "
                "remove what it reported"
            )
        report["deleted_rows"] += len(batch_keys)

    report["deleted_keys_sample"] = sorted(orphan_keys)[:SAMPLE_LIMIT]
    logger.info(
        f"[gc-llm-cache] Deleted {report['deleted_rows']} orphaned cache row(s)"
    )
    return report


def _print_report(report: dict[str, Any], verbose: bool) -> None:
    print(
        f"Cache rows: {report['cache_rows']} "
        f"({report['chunk_scoped_rows']} chunk-scoped, "
        f"{report['unscoped_rows']} without a chunk_id)"
    )
    print(f"Chunk-scoped: {report['live_rows']} live, {report['orphan_rows']} orphaned")
    if report["deleted_rows"]:
        print(f"Deleted: {report['deleted_rows']}")
    if report["orphan_rows"] and not report["deleted_rows"]:
        print("Report only — re-run with --apply to delete the orphans.")
    if verbose:
        for key in report["orphan_keys_sample"]:
            print(f"  orphan: {key}")


async def _async_main(apply: bool, batch_size: int, verbose: bool) -> None:
    import numpy as np

    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def _noop_llm(*args, **kwargs) -> str:
        raise RuntimeError("gc_llm_cache never calls the LLM")

    async def _noop_embed(texts: list[str]) -> np.ndarray:
        raise RuntimeError("gc_llm_cache never embeds")

    rag = LightRAG(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        workspace=os.getenv("WORKSPACE", ""),
        # The constructor's defaults are file-backed, so a server deployment's
        # backends have to be named here or the sweep would open a fresh local
        # directory and report an empty namespace instead of looking at the
        # workspace the operator asked about. Vectors are only constructed,
        # never read.
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
        ),
    )
    await rag.initialize_storages()
    try:
        # Printed before the sweep: a run that aborts still has to say which
        # backend it opened.
        cache_kv = rag.llm_response_cache
        print(
            f"Cache storage: {type(cache_kv).__name__} "
            f"(workspace={getattr(cache_kv, 'workspace', None)!r})"
        )
        report = await gc_orphan_llm_cache(rag, apply=apply, batch_size=batch_size)
        _print_report(report, verbose)
    finally:
        await rag.finalize_storages()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)
    parser = argparse.ArgumentParser(
        description="Delete chunk-scoped LLM cache rows whose owning chunk no "
        "longer exists (rows a document deletion could not reach)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the orphaned rows (default: report only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Keys per storage round trip (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print a sample of orphaned keys"
    )
    args = parser.parse_args()
    asyncio.run(
        _async_main(apply=args.apply, batch_size=args.batch_size, verbose=args.verbose)
    )


if __name__ == "__main__":
    main()
