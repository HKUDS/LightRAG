"""Self-contained upsert-latency driver for the write-amplification comparison.

Runs UNCHANGED on the feature branch and on a pristine origin/main baseline
checkout, each with its own interpreter and lightrag install: it imports only
lightrag plus numpy, never the harness package, and never touches hybrid
query parameters, which do not exist on the baseline. Synthetic hash
embeddings are legitimate here: this measures the write path, not quality.

Usage: python write_amp_driver.py CORPUS_JSONL WORKSPACE DIM BATCH_SIZE OUT_JSON
The PG connection comes from the POSTGRES_* environment. The driver deletes
its workspace rows and drops its scratch vector tables when empty.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from functools import lru_cache

import numpy as np

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, compute_mdhash_id

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
MODEL_NAME = "writeamp"


@lru_cache(maxsize=16384)
def _token_vector(token: str, dim: int) -> tuple[float, ...]:
    # One sha256 block covers 32 dims; larger dims concatenate numbered
    # blocks so the projection works at any embedding size.
    values: list[float] = []
    block = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{block}:{token}".encode("utf-8")).digest()
        values.extend(b / 127.5 - 1.0 for b in digest)
        block += 1
    return tuple(values[:dim])


def embed(texts: list[str], dim: int) -> np.ndarray:
    rows = []
    for text in texts:
        normalized = unicodedata.normalize("NFKC", text).lower()
        vec = np.zeros(dim, dtype=np.float64)
        for token in _TOKEN_RE.findall(normalized):
            vec += np.asarray(_token_vector(token, dim), dtype=np.float64)
        norm = np.linalg.norm(vec)
        rows.append(vec / norm if norm > 0 else vec)
    return np.asarray(rows, dtype=np.float32)


async def refuse_llm(prompt, *args, **kwargs):
    raise AssertionError("write-amp driver must not call the LLM")


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


async def run(corpus_path: str, workspace: str, dim: int, batch_size: int) -> dict:
    texts = [
        json.loads(line)["text"]
        for line in open(corpus_path, encoding="utf-8")
        if line.strip()
    ]
    corpus_sha = hashlib.sha256(
        "".join(texts).encode("utf-8")
    ).hexdigest()

    async def embed_func(texts: list[str], **kwargs) -> np.ndarray:
        return embed(texts, dim)

    rag = LightRAG(
        working_dir=os.path.join(os.getcwd(), f"writeamp_{workspace}"),
        workspace=workspace,
        llm_model_func=refuse_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=dim,
            max_token_size=8192,
            func=embed_func,
            model_name=MODEL_NAME,
        ),
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGTableGraphStorage",
        doc_status_storage="PGDocStatusStorage",
    )
    await rag.initialize_storages()
    latencies: list[float] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {}
            for offset, text in enumerate(batch):
                doc_id = f"wa-{start + offset:06d}"
                chunk_id = compute_mdhash_id(text, prefix="chunk-")
                payload[chunk_id] = {
                    "content": text,
                    "tokens": len(text),
                    "chunk_order_index": 0,
                    "full_doc_id": doc_id,
                    "file_path": doc_id,
                    "source_id": doc_id,
                }
            t0 = time.perf_counter()
            await rag.chunks_vdb.upsert(payload)
            await rag.chunks_vdb.index_done_callback()
            latencies.append(time.perf_counter() - t0)
        return {
            "p95_seconds": percentile95(latencies),
            "batches": len(latencies),
            "rows": len(texts),
            "batch_size": batch_size,
            "workspace": workspace,
            "corpus_sha256": corpus_sha,
            "dim": dim,
        }
    finally:
        await rag.finalize_storages()
        await cleanup(workspace, dim)


async def cleanup(workspace: str, dim: int) -> None:
    import asyncpg

    from lightrag.kg.postgres_impl import ClientManager

    pg = ClientManager.get_config()
    conn = await asyncpg.connect(
        host=pg["host"],
        port=int(pg["port"]),
        user=pg["user"],
        password=pg["password"],
        database=pg["database"],
    )
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
                    f'DELETE FROM "{table}" WHERE workspace=$1', workspace
                )
        for namespace in ("chunks", "entity", "relation"):
            table = f"lightrag_vdb_{namespace}_{MODEL_NAME}_{dim}d"
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_tables "
                "WHERE schemaname='public' AND tablename=$1)",
                table,
            )
            if exists and not await conn.fetchval(
                f'SELECT EXISTS(SELECT 1 FROM "{table}")'
            ):
                await conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    finally:
        await conn.close()


def main() -> int:
    corpus_path, workspace, dim, batch_size, out_json = sys.argv[1:6]
    result = asyncio.run(
        asyncio.wait_for(
            run(corpus_path, workspace, int(dim), int(batch_size)),
            timeout=1800,
        )
    )
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())