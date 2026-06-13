#!/usr/bin/env python3
"""
Subprocess worker: index CS contexts into one LightRAG instance.
Called by eval_cs_pipeline.py with --mode none|llm_frames.

Must run as separate process so LIGHTRAG_FRAME_EXTRACTION_MODE is set
BEFORE operate.py is imported (module-level constant).
"""
import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["none", "llm_frames"])
parser.add_argument("--storage", required=True)
parser.add_argument("--contexts", required=True, help="JSON file with list of context strings")
args = parser.parse_args()

os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = args.mode
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT", "vertical-reason-476709-v8")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ["PYTHONIOENCODING"] = "utf-8"

import asyncio, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(".env", override=False)

from google import genai as _genai
from google.genai import types as _gt
from lightrag import LightRAG
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.gemini import gemini_complete_if_cache
from functools import partial

GCP_PROJECT  = os.environ["GOOGLE_CLOUD_PROJECT"]
GCP_LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
LLM_MODEL    = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBED_DIM    = int(os.getenv("EMBEDDING_DIM", "1536"))
MAX_ASYNC    = int(os.getenv("MAX_ASYNC", "2"))

# Rate limit: enforce minimum gap between LLM API calls to stay under Vertex AI quota.
# 8s gap → max ~7.5 RPM; safe even for large chunks (high TPM per call).
_RATE_INTERVAL = 8.0
_rate_last: float = 0.0
_rate_lock: asyncio.Lock | None = None

_base_llm = partial(
    gemini_complete_if_cache,
    LLM_MODEL,
    generation_config={"thinking_config": _gt.ThinkingConfig(thinking_budget=0)},
)

async def llm_func(*args, **kwargs):
    global _rate_last, _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    async with _rate_lock:
        gap = _RATE_INTERVAL - (asyncio.get_event_loop().time() - _rate_last)
        if gap > 0:
            await asyncio.sleep(gap)
        _rate_last = asyncio.get_event_loop().time()
    return await _base_llm(*args, **kwargs)

_gc = _genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

@wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=2048)
async def vertex_embed(texts: list[str]) -> np.ndarray:
    r = await _gc.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=_gt.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return np.array([e.values for e in r.embeddings], dtype=np.float32)


async def main():
    storage = Path(args.storage)
    storage.mkdir(parents=True, exist_ok=True)

    contexts = json.loads(Path(args.contexts).read_text(encoding="utf-8"))
    print(f"[{args.mode}] Storage: {storage}", flush=True)
    print(f"[{args.mode}] Contexts: {len(contexts)} docs | MAX_ASYNC={MAX_ASYNC} | rate_limit={_RATE_INTERVAL}s/call (~{60/_RATE_INTERVAL:.0f} RPM max)", flush=True)
    total_chars = sum(len(c) for c in contexts)
    print(f"[{args.mode}] Total chars: {total_chars:,} (~{total_chars//4:,} tokens)", flush=True)

    # Batch 3 docs at a time — enough parallelism, few enough to avoid quota cascade
    BATCH = 3
    rag = LightRAG(
        working_dir=str(storage),
        llm_model_func=llm_func,
        embedding_func=vertex_embed,
        llm_model_max_async=MAX_ASYNC,
        embedding_func_max_async=2,
        max_parallel_insert=BATCH,
    )
    await rag.initialize_storages()

    t0 = time.time()
    n_batches = (len(contexts) + BATCH - 1) // BATCH
    for i in range(0, len(contexts), BATCH):
        batch = contexts[i:i+BATCH]
        b = i // BATCH + 1
        print(f"[{args.mode}] Batch {b}/{n_batches}: docs {i+1}-{min(i+BATCH, len(contexts))}", flush=True)
        await rag.ainsert(batch)

    elapsed = time.time() - t0
    await rag.finalize_storages()

    # Report doc status
    status_file = storage / "kv_store_doc_status.json"
    if status_file.exists():
        statuses = json.loads(status_file.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for v in statuses.values():
            s = (v.get("status", "unknown") if isinstance(v, dict) else "unknown").lower()
            counts[s] = counts.get(s, 0) + 1
        print(f"[{args.mode}] Doc status: {counts}", flush=True)
        failed = sum(v for k, v in counts.items() if k in ("failed", "processing"))
        if failed > 0:
            print(f"[{args.mode}] WARNING: {failed} doc(s) failed — pipeline will retry.", flush=True)
            sys.exit(1)

    import re
    graphml = storage / "graph_chunk_entity_relation.graphml"
    gtext = graphml.read_text(encoding="utf-8") if graphml.exists() else ""
    nodes = len(re.findall(r"<node ", gtext))
    edges = len(re.findall(r"<edge ", gtext))
    print(f"[{args.mode}] DONE in {elapsed:.0f}s. Graph: {nodes} nodes, {edges} edges.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
