#!/usr/bin/env python
"""Wiring check for ``run_live.py``. NOT evidence for F01-F04.

Real storages, real resolver, real ``aquery_llm`` path -- but the chunk rows are
written by hand and both the embedding and the answering functions are local
deterministic stubs, so nothing here says anything about a model's answer.

Its only job is to catch plumbing defects (storage APIs, catalog injection, the
context recorder, record assembly) before a paid run, so that the paid run is
about the behaviour and not about a typo.

Run::

    .venv/bin/python \
        examples/evidence_policy/smoke_plumbing.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_3037 as corpus  # noqa: E402
import run_live  # noqa: E402

from lightrag import LightRAG  # noqa: E402
from lightrag.kg.shared_storage import initialize_pipeline_status  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

DIM = 64


def _embed(text: str) -> list[float]:
    """Deterministic bag-of-characters vector: no semantics, only plumbing."""
    vector = [0.0] * DIM
    for index, char in enumerate(text):
        vector[(ord(char) + index) % DIM] += 1.0
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


async def _embed_func(texts):
    import numpy as np

    return np.array([_embed(text) for text in texts], dtype=float)


async def _stub_llm(prompt, system_prompt=None, **_kwargs):
    # Echo the rendered context so the record shows what reached the model.
    return "STUB ANSWER\n" + (system_prompt or "")[-400:]


def _doc_status_row(chunk_ids, content, file_path):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "status": "processed",
        "content_summary": content[:80],
        "content_length": len(content),
        "file_path": file_path,
        "created_at": now,
        "updated_at": now,
        "track_id": "smoke",
        "chunks_count": len(chunk_ids),
        "chunks_list": chunk_ids,
        "metadata": {},
    }


async def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="3037-smoke-"))
    failures: list[str] = []
    try:
        for name in (run_live.WORKSPACE_MAIN, run_live.WORKSPACE_UNRECALLED):
            ws = workdir / name
            ws.mkdir(parents=True, exist_ok=True)
            llm_func, state = run_live.make_recorder(_stub_llm)
            rag = LightRAG(
                working_dir=str(ws),
                kv_storage="JsonKVStorage",
                vector_storage="NanoVectorDBStorage",
                graph_storage="NetworkXStorage",
                doc_status_storage="JsonDocStatusStorage",
                llm_model_func=llm_func,
                llm_model_name="stub",
                embedding_func=EmbeddingFunc(
                    embedding_dim=DIM, max_token_size=8192, func=_embed_func
                ),
                enable_llm_cache=False,
                evidence_policy_timezone="Asia/Shanghai",
            )
            await rag.initialize_storages()
            await initialize_pipeline_status()

            chunk_rows = {}
            status_rows = {}
            for spec in corpus.DOCUMENTS:
                chunk_id = f"chunk-{spec['doc_id']}"
                chunk_rows[chunk_id] = {
                    "content": spec["content"],
                    "full_doc_id": spec["doc_id"],
                    "file_path": spec["file_path"],
                    "chunk_order_index": 0,
                    "tokens": len(spec["content"]),
                }
                status_rows[spec["doc_id"]] = _doc_status_row(
                    [chunk_id], spec["content"], spec["file_path"]
                )
            await rag.text_chunks.upsert(chunk_rows)
            await rag.doc_status.upsert(status_rows)
            await rag.chunks_vdb.upsert(
                {
                    chunk_id: {
                        "content": row["content"],
                        "file_path": row["file_path"],
                    }
                    for chunk_id, row in chunk_rows.items()
                }
            )
            # A vector store publishes on index_done_callback; without this the
            # query leg sees an empty index.
            await rag.chunks_vdb.index_done_callback()

            catalog, detail = await run_live.build_catalog(
                rag, revision=f"smoke-{name}"
            )
            rag.evidence_catalog = catalog
            if detail["V-new"]["doc_id"] != corpus.DOC_NEW:
                failures.append(f"{name}: catalog doc id mismatch")

            if name == run_live.WORKSPACE_UNRECALLED:
                removed = list(detail["V-new"]["chunk_ids"])
                await rag.chunks_vdb.delete(removed)
                await rag.chunks_vdb.index_done_callback()

            for case in [c for c in run_live.CASES if c["workspace"] == name]:
                result = await run_live.run_case(case, rag, state, detail)
                structural = {
                    key: value
                    for key, value in result["checks"].items()
                    if key != "answer_contains_expected"
                }
                if not all(structural.values()):
                    failures.append(f"{case['id']}: {structural}")
                if result["policy_status"] != "resolved":
                    failures.append(f"{case['id']}: status {result['policy_status']}")
                if not result["actual_model_context"]:
                    failures.append(f"{case['id']}: empty model context")
                print(
                    json.dumps(
                        {
                            "case": result["id"],
                            "status": result["policy_status"],
                            "seeds": result["seed_versions"],
                            "expanded": result["expanded_versions"],
                            "selected": result["selected_versions"],
                            "context_chars": len(result["actual_model_context"]),
                        },
                        ensure_ascii=False,
                    )
                )
            await rag.finalize_storages()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print("WIRING FAILURES:")
        for item in failures:
            print(" -", item)
        return 1
    print("wiring ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
