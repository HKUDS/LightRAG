#!/usr/bin/env python3
"""
Worker process for one eval mode. Called by eval_postfix.py.
Must be run as a separate process so LIGHTRAG_FRAME_EXTRACTION_MODE
is set BEFORE operate.py is imported (module-level constant).

Usage:
    python eval_mode_worker.py --mode none --storage eval_tmp_none --output none.json
    python eval_mode_worker.py --mode llm_frames --storage eval_tmp_llm --output llm.json
"""
import argparse
import os
import sys

# ── CRITICAL: set env var before ANY lightrag import ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--storage", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = args.mode
os.environ["PYTHONIOENCODING"] = "utf-8"

# ─── now safe to import ───────────────────────────────────────────────────────
import asyncio
import json
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

import numpy as np
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import wrap_embedding_func_with_attrs

EVAL_DIR = Path("lightrag/evaluation")
SAMPLE_DOCS_DIR = EVAL_DIR / "sample_documents"
SAMPLE_DATASET = EVAL_DIR / "sample_dataset.json"

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
API_KEY = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1")


def build_rag(storage_dir: Path):
    _client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)

    async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        resp = await _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=0,
        )
        return resp.choices[0].message.content

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=8192)
    async def _embed(texts: list[str]) -> np.ndarray:
        resp = await _client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([d.embedding for d in resp.data])

    return LightRAG(
        working_dir=str(storage_dir),
        llm_model_func=_llm,
        embedding_func=_embed,
    )


async def main():
    storage_dir = Path(args.storage)

    rag = build_rag(storage_dir)
    await rag.initialize_storages()

    # Index
    texts = [
        f.read_text(encoding="utf-8")
        for f in sorted(SAMPLE_DOCS_DIR.glob("*.md"))
        if f.name != "README.md"
    ]
    print(f"[{args.mode}] Indexing {len(texts)} documents …", flush=True)
    t0 = time.time()
    await rag.ainsert(texts)
    index_time = time.time() - t0
    print(f"[{args.mode}] Indexing done in {index_time:.1f}s", flush=True)

    # Graph stats
    graphml = storage_dir / "graph_chunk_entity_relation.graphml"
    gtext = graphml.read_text(encoding="utf-8") if graphml.exists() else ""
    graph = {
        "nodes": len(re.findall(r"<node ", gtext)),
        "edges": len(re.findall(r"<edge ", gtext)),
    }
    print(f"[{args.mode}] Graph: {graph['nodes']} nodes, {graph['edges']} edges", flush=True)

    # Query
    test_cases = json.loads(SAMPLE_DATASET.read_text(encoding="utf-8"))["test_cases"]
    results = []
    print(f"[{args.mode}] Running {len(test_cases)} queries …", flush=True)
    t_q = time.time()
    for tc in test_cases:
        t1 = time.time()
        answer = await rag.aquery(tc["question"], param=QueryParam(mode="hybrid", top_k=20))
        elapsed = time.time() - t1
        results.append({
            "question": tc["question"],
            "answer": answer if isinstance(answer, str) else str(answer),
            "ground_truth": tc["ground_truth"],
            "elapsed": elapsed,
        })
        print(f"[{args.mode}]   Q: {tc['question'][:55]}… ({elapsed:.1f}s)", flush=True)
    query_time = time.time() - t_q

    await rag.finalize_storages()

    # RAGAS scoring
    print(f"[{args.mode}] Scoring with RAGAS …", flush=True)
    ragas_scores = {}
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
        from langchain_openai import ChatOpenAI

        dataset = Dataset.from_list([
            {
                "question": r["question"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "contexts": [r["answer"]],
            }
            for r in results
        ])

        base_url = API_BASE if API_BASE != "https://api.openai.com/v1" else None
        llm = ChatOpenAI(model=LLM_MODEL, api_key=API_KEY, base_url=base_url, temperature=0)

        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            raise_exceptions=False,
        )

        # RAGAS 0.4.x returns EvaluationResult; convert to DataFrame for safe extraction
        df = score.to_pandas()
        def _safe_mean(col):
            try:
                return float(df[col].dropna().mean())
            except Exception:
                return None

        ragas_scores = {
            "faithfulness": _safe_mean("faithfulness"),
            "answer_relevancy": _safe_mean("answer_relevancy"),
            "context_recall": _safe_mean("context_recall"),
            "context_precision": _safe_mean("context_precision"),
        }
        vals = [v for v in ragas_scores.values() if v is not None]
        ragas_scores["composite"] = sum(vals) / len(vals) if vals else None
        print(f"[{args.mode}] RAGAS: {ragas_scores}", flush=True)

    except Exception as e:
        print(f"[{args.mode}] RAGAS failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        ragas_scores = {"error": str(e)}

    output = {
        "mode": args.mode,
        "index_time_s": round(index_time, 1),
        "query_time_s": round(query_time, 1),
        "graph": graph,
        "ragas": ragas_scores,
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[{args.mode}] Output saved to {args.output}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
