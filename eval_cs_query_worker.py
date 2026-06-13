#!/usr/bin/env python3
"""
Subprocess worker: run 125 hybrid queries on one LightRAG storage.
Called by eval_cs_pipeline.py. Uses subprocess isolation so
FRAME_EXTRACTION_MODE is set before operate.py is imported.
"""
import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode",     required=True, choices=["none", "llm_frames"])
parser.add_argument("--storage",  required=True)
parser.add_argument("--questions",required=True, help="path to questions .txt file")
parser.add_argument("--output",   required=True, help="output JSON file path")
args = parser.parse_args()

os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = args.mode
os.environ["GOOGLE_GENAI_USE_VERTEXAI"]  = "true"
os.environ["GOOGLE_CLOUD_PROJECT"]       = os.getenv("GOOGLE_CLOUD_PROJECT", "vertical-reason-476709-v8")
os.environ["GOOGLE_CLOUD_LOCATION"]      = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ["PYTHONIOENCODING"]           = "utf-8"

import asyncio, json, re, time
from functools import partial
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(".env", override=False)

from google import genai as _genai
from google.genai import types as _gt
from lightrag import LightRAG, QueryParam
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.gemini import gemini_complete_if_cache

GCP_PROJECT  = os.environ["GOOGLE_CLOUD_PROJECT"]
GCP_LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
LLM_MODEL    = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBED_DIM    = int(os.getenv("EMBEDDING_DIM", "1536"))
MAX_ASYNC    = int(os.getenv("MAX_ASYNC", "16"))

_gc = _genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

@wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=2048)
async def vertex_embed(texts: list[str]) -> np.ndarray:
    r = await _gc.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=_gt.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return np.array([e.values for e in r.embeddings], dtype=np.float32)

_RATE_INTERVAL = 6.0  # 10 RPM max for queries (shorter prompts than indexing)
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


async def main():
    storage = Path(args.storage)
    out_file = Path(args.output)
    qs_text  = Path(args.questions).read_text(encoding="utf-8")
    questions = re.findall(r"- Question \d+:\s*(.+)", qs_text)
    print(f"[{args.mode}] {len(questions)} questions, storage={storage}", flush=True)

    # Resume support
    if out_file.exists():
        results = json.loads(out_file.read_text(encoding="utf-8"))
        done = {r["query"] for r in results}
        remaining = [q for q in questions if q not in done]
        print(f"[{args.mode}] Resuming: {len(done)} done, {len(remaining)} remaining", flush=True)
    else:
        results, remaining = [], list(questions)

    if not remaining:
        print(f"[{args.mode}] All queries already done.", flush=True)
        return

    rag = LightRAG(
        working_dir=str(storage),
        llm_model_func=llm_func,
        embedding_func=vertex_embed,
        llm_model_max_async=MAX_ASYNC,
        embedding_func_max_async=MAX_ASYNC * 2,
    )
    await rag.initialize_storages()

    for i, q in enumerate(remaining):
        t0 = time.time()
        try:
            answer = await rag.aquery(q, param=QueryParam(mode="hybrid", top_k=60))
            results.append({"query": q, "result": str(answer) if answer else ""})
        except Exception as e:
            print(f"[{args.mode}] ERROR q{i+1}: {e}", flush=True)
            results.append({"query": q, "result": f"[ERROR: {e}]"})

        elapsed = time.time() - t0
        print(f"[{args.mode}] Q{i+1}/{len(remaining)}: {q[:60]}... ({elapsed:.1f}s)", flush=True)

        if (i + 1) % 10 == 0:
            out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    await rag.finalize_storages()
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.mode}] Done. {len(results)} results → {out_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
