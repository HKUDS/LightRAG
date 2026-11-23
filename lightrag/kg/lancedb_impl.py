"""Cold-snapshot export job: LightRAG source-of-truth -> tagged Lance tables.

Rules for callers:
- Offline JOB, not a storage backend: no registry role, no queries; one job per destination.
- Pass an INITIALIZED, quiesced LightRAG (no writers, every ``index_done_callback``
  flush complete). Vectors are READ from vector storage (``get_vectors_by_ids``),
  never computed; on deferred-embedding backends an unflushed row can trigger a
  lazy embed, so install a fail-loud embedding function (the CLI does).
- Sources of truth: ``full_docs`` / ``text_chunks`` KV + the graph; vector rows join
  by the rebuild id scheme (``ent-`` / ``rel-`` hashes, chunk ids); rows with no
  source record are disposable. Every source must contribute >= 1 row and every
  vector-eligible record a stored vector of the configured dimension. Preflight covers
  only the FIRST batch of each table; a later failure aborts mid-write and leaves
  untagged versions (tiered-storage contract, interrupted-export residue), healed by a re-run.
- ``dest``: local directory, ``oss://`` or ``s3://`` URI (creds: ``storage_options_from_env``).
  Each export writes a NEW version of all four tables (fixed-size-list float32 embeddings); schema
  metadata ``lightrag.embedding_model`` records the model identity (empty when unknown) for restore.
- Tags are immutable: reusing a tag raises ``ExportError`` before any write. Tags land
  only after all four tables complete; a partial tag failure refuses a same-tag retry,
  so recover by publishing under a NEW tag.

CLI: ``python -m lightrag.kg.lancedb_impl --working-dir DIR --dest DEST --tag TAG``
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
from typing import TYPE_CHECKING, Any, AsyncIterator

import lancedb
import pyarrow as pa
from dotenv import load_dotenv

from lightrag.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    get_env_value,
    logger,
    make_relation_vdb_ids,
    setup_logger,
)

if TYPE_CHECKING:
    from lightrag import LightRAG
    from lightrag.base import BaseKVStorage, BaseVectorStorage

_TABLES = ("docs", "chunks", "entities", "relations")


class ExportError(RuntimeError):
    """A snapshot was refused or could not be published."""


def storage_options_from_env(dest: str = "") -> dict[str, str]:
    """object_store options for ``dest``; {} when no credentials are set.
    oss:// (or empty dest) + OSS key/secret -> native keys sourced ONLY from OSS env vars
    (the oss provider ignores aws_* keys); s3:// -> aws_* keys with AWS_* fallbacks, virtual-hosted.
    Creds: OSS_DEFAULT_SECRET_ID/KEY/TOKEN (STS), OSS_ACCESS_KEY_ID/SECRET, oss_access_ak/sk, AWS_*."""
    oss_key = os.getenv("OSS_DEFAULT_SECRET_ID") or os.getenv("OSS_ACCESS_KEY_ID") or os.getenv("oss_access_ak")
    oss_secret = os.getenv("OSS_DEFAULT_SECRET_KEY") or os.getenv("OSS_ACCESS_KEY_SECRET") or os.getenv("oss_access_sk")
    if oss_key and oss_secret and not dest.startswith("s3://"):
        token = os.getenv("OSS_DEFAULT_SECRET_TOKEN")
        endpoint = os.getenv("OSS_ENDPOINT") or os.getenv("oss_endpoint")
        options = {"access_key_id": oss_key, "access_key_secret": oss_secret, "region": os.getenv("OSS_REGION") or (endpoint or "").split("://")[-1].split(".")[0].removeprefix("oss-").removesuffix("-internal") or "cn-hangzhou"}
        if token:
            options["security_token"] = token
        if endpoint:
            options["endpoint"] = endpoint
        return options
    key, secret = oss_key or os.getenv("AWS_ACCESS_KEY_ID"), oss_secret or os.getenv("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        return {}
    token = os.getenv("OSS_DEFAULT_SECRET_TOKEN") or os.getenv("AWS_SESSION_TOKEN")
    endpoint = os.getenv("OSS_ENDPOINT") or os.getenv("oss_endpoint") or os.getenv("AWS_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT")
    options = {"aws_access_key_id": key, "aws_secret_access_key": secret, "virtual_hosted_style_request": "true", "aws_region": (os.getenv("OSS_REGION") or os.getenv("AWS_REGION")) or "us-east-1"}
    if token:
        options["aws_session_token"] = token
    if endpoint:
        options["aws_endpoint"] = endpoint
        options["aws_allow_http"] = "true" if endpoint.startswith("http://") else "false"
    return options


def _schemas(dim: int, model_name: str) -> dict[str, pa.Schema]:
    vec = pa.field("embedding", pa.list_(pa.float32(), dim), nullable=True)
    meta = {b"lightrag.embedding_model": model_name.encode()}
    return {
        "docs": pa.schema([("id", pa.string()), ("content", pa.string())], metadata=meta),
        "chunks": pa.schema(
            [("id", pa.string()), ("content", pa.string()), ("full_doc_id", pa.string()),
             ("chunk_order_index", pa.int64()), ("tokens", pa.int64()),
             ("file_path", pa.string()), vec], metadata=meta),
        "entities": pa.schema(
            [("id", pa.string()), ("entity_name", pa.string()), ("entity_type", pa.string()),
             ("description", pa.string()), ("source_id", pa.string()),
             ("file_path", pa.string()), vec], metadata=meta),
        "relations": pa.schema(
            [("id", pa.string()), ("src_id", pa.string()), ("tgt_id", pa.string()),
             ("keywords", pa.string()), ("description", pa.string()),
             ("source_id", pa.string()), ("weight", pa.float64()),
             ("file_path", pa.string()), vec], metadata=meta),
    }


async def _kv_batches(kv: BaseKVStorage, size: int) -> AsyncIterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    async for row in kv.iter_rows(page_size=size):
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


async def _raw_docs(rag: LightRAG, size: int) -> AsyncIterator[list[dict[str, Any]]]:
    async for batch in _kv_batches(rag.full_docs, size):
        yield [{"id": str(r["_id"]), "content": r.get("content") or ""} for r in batch]


async def _raw_chunks(rag: LightRAG, size: int) -> AsyncIterator[list[dict[str, Any]]]:
    async for batch in _kv_batches(rag.text_chunks, size):
        rows = []
        for r in batch:
            content = r.get("content") or ""
            rows.append(
                {"id": str(r["_id"]), "content": content,
                 "full_doc_id": r.get("full_doc_id") or "",
                 "chunk_order_index": int(r.get("chunk_order_index") or 0),
                 "tokens": int(r.get("tokens") or 0),
                 "file_path": r.get("file_path") or "",
                 # A content-less chunk has no vector (the rebuild skips it too).
                 "_candidates": [str(r["_id"])] if content else []})
        yield rows


async def _raw_entities(rag: LightRAG, size: int) -> AsyncIterator[list[dict[str, Any]]]:
    graph = rag.chunk_entity_relation_graph
    async for labels in graph.iter_labels(size):
        nodes = await graph.get_nodes_batch([str(label) for label in labels])
        rows = []
        for label in labels:
            node = nodes.get(str(label))
            if node is None:
                raise ExportError(f"graph node {label!r} was listed but could not be read")
            ent_id = compute_mdhash_id(str(label), prefix="ent-")
            rows.append(
                {"id": ent_id, "entity_name": str(label),
                 "entity_type": node.get("entity_type") or "",
                 "description": node.get("description") or "",
                 "source_id": node.get("source_id") or "",
                 "file_path": node.get("file_path") or "", "_candidates": [ent_id]})
        yield rows


async def _raw_relations(rag: LightRAG, size: int) -> AsyncIterator[list[dict[str, Any]]]:
    seen: set[tuple[str, str]] = set()
    async for edges in rag.chunk_entity_relation_graph.iter_edges(size):
        rows = []
        for edge in edges:
            src, tgt = sorted((str(edge["source"]), str(edge["target"])))
            norm = compute_mdhash_id(src + tgt, prefix="rel-")
            if (src, tgt) in seen:
                continue  # backends may list an undirected edge once per direction
            seen.add((src, tgt))
            rows.append(
                {"id": norm, "src_id": src, "tgt_id": tgt,
                 "keywords": edge.get("keywords") or "",
                 "description": edge.get("description") or "",
                 "source_id": edge.get("source_id") or "",
                 "weight": float(edge.get("weight", 1.0)),
                 "file_path": edge.get("file_path") or "",
                 "_candidates": make_relation_vdb_ids(src, tgt)})
        yield rows


async def _join_vectors(
    batch: list[dict[str, Any]], vdb: BaseVectorStorage | None, dim: int
) -> list[dict[str, Any]]:
    if vdb is None:
        return batch
    ids = list(dict.fromkeys(c for row in batch for c in row["_candidates"]))
    vectors = await vdb.get_vectors_by_ids(ids) if ids else {}
    for row in batch:
        candidates = row.pop("_candidates")
        found = next((c for c in candidates if c in vectors), None)
        if found is None:
            if candidates:
                raise ExportError(f"no stored vector for {row['id']!r} (candidates {candidates}); flush storages before export")
            row["embedding"] = None
            continue
        if len(vectors[found]) != dim:
            raise ExportError(f"stored vector for {found!r} has dimension {len(vectors[found])}, expected {dim}")
        row["id"] = found  # a legacy reverse-order relation id is kept as stored
        row["embedding"] = vectors[found]
    return batch


async def export_snapshot(
    rag: LightRAG, dest: str, tag: str, *,
    batch_size: int = 500, storage_options: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Publish one tagged cold snapshot; returns {"tag", "tables": {name: {"rows", "version"}}}.

    Caller rules (quiesced sources, immutable tags, fail-before-write) are in
    the module docstring.
    """
    if not isinstance(tag, str) or not tag.strip():
        raise ExportError("tag must be a non-empty string")
    if batch_size < 1:
        raise ExportError(f"batch_size must be >= 1, got {batch_size}")
    storages = (rag.full_docs, rag.text_chunks, rag.chunk_entity_relation_graph,
                rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb)
    if any(s is None for s in storages):
        raise ExportError("source storages are missing; call await rag.initialize_storages() first")
    db = lancedb.connect(dest, storage_options=storage_options or {})
    existing = set(db.list_tables())
    for name in _TABLES:
        if name in existing and tag in db.open_table(name).tags.list():
            raise ExportError(f"tag {tag!r} already exists on table {name!r}; tags are immutable, pick a new name")
    dim = rag.embedding_func.embedding_dim
    schemas = _schemas(dim, rag.embedding_func.model_name or "")
    raws = {"docs": _raw_docs(rag, batch_size), "chunks": _raw_chunks(rag, batch_size),
            "entities": _raw_entities(rag, batch_size), "relations": _raw_relations(rag, batch_size)}
    vdbs = {"docs": None, "chunks": rag.chunks_vdb,
            "entities": rag.entities_vdb, "relations": rag.relationships_vdb}
    # Read (and vector-join) the first batch of EVERY table before any write,
    # so an empty source or a missing vector destroys nothing.
    firsts: dict[str, list[dict[str, Any]]] = {}
    for name in _TABLES:
        batch = await anext(raws[name], None)
        if batch is None:
            raise ExportError(f"source for table {name!r} is empty; refusing to publish a partial snapshot")
        firsts[name] = await _join_vectors(batch, vdbs[name], dim)
    tables = []
    for name in _TABLES:
        table = db.create_table(name, data=pa.Table.from_pylist(firsts[name], schema=schemas[name]), mode="overwrite")
        async for batch in raws[name]:
            joined = await _join_vectors(batch, vdbs[name], dim)
            table.add(pa.Table.from_pylist(joined, schema=schemas[name]))
        tables.append(table)
    for table in tables:
        table.tags.create(tag, table.version)
    report = {"tag": tag, "tables": {n: {"rows": t.count_rows(), "version": t.version}
                                     for n, t in zip(_TABLES, tables)}}
    logger.info(f"Cold snapshot tagged {tag!r} at {dest}: {report['tables']}")
    return report


def _resolve_embedding_config() -> tuple[int, str | None]:
    """EMBEDDING_DIM/MODEL as the server resolves them: explicit env wins, else the binding's provider default."""
    dim = get_env_value("EMBEDDING_DIM", 0, int)
    model = get_env_value("EMBEDDING_MODEL", None, special_none=True)
    if dim:
        return dim, model
    binding = os.getenv("EMBEDDING_BINDING", "openai")
    try:
        provider = getattr(importlib.import_module(f"lightrag.llm.{binding}"), f"{binding}_embed")
    except (ImportError, AttributeError) as exc:
        raise ExportError(f"cannot import the {binding!r} embedding provider; set EMBEDDING_DIM explicitly") from exc
    default_model = provider.model_name
    effective = (os.getenv("AZURE_EMBEDDING_DEPLOYMENT") or model) if binding == "azure_openai" else model
    if effective and binding != "lollms" and (binding == "azure_openai" or effective.removesuffix(":latest") != (default_model or "").removesuffix(":latest")):
        raise ExportError(f"EMBEDDING_DIM must be set for model {effective!r}; the {binding!r} provider default only covers {default_model!r}")
    return provider.embedding_dim, model or default_model


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a LightRAG workspace as a tagged Lance cold snapshot.")
    p.add_argument("--working-dir", default=os.getenv("WORKING_DIR", "./rag_storage"))
    p.add_argument("--dest", required=True, help="local directory, oss:// or s3:// URI")
    p.add_argument("--tag", required=True, help="immutable tag name for this snapshot")
    p.add_argument("--workspace", default=os.getenv("WORKSPACE", ""))
    p.add_argument("--batch-size", type=int, default=500)
    return p


async def _run(args: argparse.Namespace) -> None:
    from lightrag import LightRAG

    async def _never_embed(*_args, **_kwargs):
        raise RuntimeError("cold export must not call the embedding function; flush first")

    async def _never_llm(*_args, **_kwargs):
        raise RuntimeError("cold export must not call the LLM")

    dim, model_name = _resolve_embedding_config()
    rag = LightRAG(
        working_dir=args.working_dir, workspace=args.workspace,
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        doc_status_storage=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
        embedding_func=EmbeddingFunc(embedding_dim=dim, func=_never_embed, model_name=model_name),
        llm_model_func=_never_llm,
    )
    await rag.initialize_storages()
    try:
        options = storage_options_from_env(args.dest) if args.dest.startswith(("s3://", "oss://")) else None
        await export_snapshot(rag, args.dest, args.tag, batch_size=args.batch_size, storage_options=options)
    finally:
        await rag.finalize_storages()


def main() -> None:
    load_dotenv(override=False)
    setup_logger("lightrag", level="INFO")
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()