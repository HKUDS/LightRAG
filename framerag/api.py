"""FrameRAG FastAPI server.

Provides REST endpoints analogous to LightRAG's API:
  POST /insert          — index a document
  POST /insert_batch    — index multiple documents
  POST /query           — answer a question (blocking)
  GET  /stream_query    — streaming answer (SSE)
  GET  /stats           — hypergraph statistics
  GET  /entity/{name}   — look up a canonical entity
  GET  /frames          — list all frames in the Frame DB
  GET  /causal_chain    — trace a causal chain from a trigger word
  DELETE /clear         — clear all indexed data (dangerous)

Usage:
    uvicorn framerag.api:create_app --factory --reload

    Or:
        from framerag.api import create_app
        app = create_app(
            working_dir="./storage",
            llm_func=my_llm,
            embed_func=my_embed,
        )

Environment variables (when using create_from_env()):
    FRAMERAG_WORKING_DIR     (default: ./framerag_storage)
    FRAMERAG_EMBEDDING_DIM   (default: 1536)
    OPENAI_API_KEY
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Awaitable, Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI not installed. Install with: pip install fastapi uvicorn"
    )

from lightrag.utils import logger

from .framerag import FrameRAG


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class InsertRequest(BaseModel):
    text: str
    source_doc: str = "unknown"


class InsertBatchRequest(BaseModel):
    texts: list[str]
    source_docs: Optional[list[str]] = None
    concurrency: int = 2


class QueryRequest(BaseModel):
    query: str
    top_chunks: int = 20
    top_frames: int = 10
    top_nodes: int = 15
    diffusion_steps: int = 3


class QueryResponse(BaseModel):
    query: str
    answer: str


class ContextResponse(BaseModel):
    query: str
    structured_facts: str
    text_passages: str
    frame_hits: list[dict]
    chunk_hits: list[dict]


class EntityResponse(BaseModel):
    name: str
    data: Optional[dict]


class StatsResponse(BaseModel):
    chunks: int
    entity_mentions: int
    canonical_entities: int
    events: int
    frame_instances: int
    info_nodes: int
    causal_edges: int
    frames_in_db: int
    top_frames_by_usage: list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

_rag: Optional[FrameRAG] = None


def create_app(
    working_dir: str = "./framerag_storage",
    llm_func: Optional[Callable[..., Awaitable[str]]] = None,
    embed_func: Optional[Callable[[list[str]], Awaitable[np.ndarray]]] = None,
    embedding_dim: int = 1536,
    **framerag_kwargs,
) -> FastAPI:
    """Create a configured FastAPI application wrapping a FrameRAG instance.

    Args:
        working_dir:  Directory for persisted storage.
        llm_func:     Async LLM callable; if None uses OPENAI_API_KEY.
        embed_func:   Async embed callable; if None uses OPENAI_API_KEY.
        embedding_dim: Embedding vector dimension.
        **framerag_kwargs: Extra kwargs forwarded to FrameRAG().
    """
    global _rag

    if llm_func is None or embed_func is None:
        llm_func, embed_func = _default_openai_funcs()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        global _rag
        _rag = FrameRAG(
            working_dir=working_dir,
            llm_func=llm_func,
            embed_func=embed_func,
            embedding_dim=embedding_dim,
            **framerag_kwargs,
        )
        await _rag.initialize()
        logger.info(f"[FrameRAG API] Ready. Storage: {working_dir}")
        yield
        await _rag.finalize()
        logger.info("[FrameRAG API] Shutdown complete")

    app = FastAPI(
        title="FrameRAG API",
        version="0.1.0",
        description="Frame-Semantic Event Hypergraph RAG — REST interface",
        lifespan=_lifespan,
    )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/insert", summary="Index a single document")
    async def insert(req: InsertRequest):
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        try:
            await _rag.ainsert(req.text, source_doc=req.source_doc)
        except Exception as e:
            raise HTTPException(500, str(e))
        return {"status": "ok", "source_doc": req.source_doc}

    @app.post("/insert_batch", summary="Index multiple documents")
    async def insert_batch(req: InsertBatchRequest):
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        try:
            await _rag.ainsert_batch(
                req.texts,
                source_docs=req.source_docs,
                concurrency=req.concurrency,
            )
        except Exception as e:
            raise HTTPException(500, str(e))
        return {"status": "ok", "inserted": len(req.texts)}

    @app.post("/query", response_model=QueryResponse, summary="Answer a question")
    async def query(req: QueryRequest):
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        try:
            answer = await _rag.aquery(
                req.query,
                top_chunks=req.top_chunks,
                top_frames=req.top_frames,
                top_nodes=req.top_nodes,
                diffusion_steps=req.diffusion_steps,
            )
        except Exception as e:
            raise HTTPException(500, str(e))
        return QueryResponse(query=req.query, answer=answer)

    @app.post("/context", response_model=ContextResponse,
              summary="Retrieve context without answer generation")
    async def context(req: QueryRequest):
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        try:
            ctx = await _rag.aretrieve_context(
                req.query,
                top_chunks=req.top_chunks,
                top_frames=req.top_frames,
                top_nodes=req.top_nodes,
                diffusion_steps=req.diffusion_steps,
            )
        except Exception as e:
            raise HTTPException(500, str(e))
        return ContextResponse(query=req.query, **ctx)

    @app.get("/stream_query", summary="Streaming SSE answer")
    async def stream_query(
        query: str = Query(..., description="Question to answer"),
        top_chunks: int = 20,
        top_frames: int = 10,
        top_nodes: int = 15,
        diffusion_steps: int = 3,
    ):
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")

        async def _event_stream() -> AsyncIterator[str]:
            try:
                async for token in _rag.aquery_stream(
                    query,
                    top_chunks=top_chunks,
                    top_frames=top_frames,
                    top_nodes=top_nodes,
                    diffusion_steps=diffusion_steps,
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/stats", response_model=StatsResponse, summary="Hypergraph statistics")
    async def stats():
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        return await _rag.get_stats()

    @app.get("/entity/{name}", response_model=EntityResponse,
             summary="Look up a canonical entity by name")
    async def get_entity(name: str):
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        data = await _rag.get_canonical_entity(name)
        return EntityResponse(name=name, data=data)

    @app.get("/entity/{name}/frames",
             summary="Frame instances involving an entity")
    async def entity_frames(name: str, top_k: int = 10):
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        fis = await _rag.get_frame_instances_for_entity(name, top_k=top_k)
        return {"entity": name, "frame_instances": fis}

    @app.get("/frames", summary="List all frames in the Frame DB")
    async def list_frames():
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        frames = await _rag._frame_db.all_frames()
        return {
            "count": len(frames),
            "frames": [
                {
                    "frame_name":       f.frame_name,
                    "lexical_units":    f.lexical_units,
                    "frame_definition": f.frame_definition,
                    "usage_count":      f.usage_count,
                    "core_fes":         [fe.fe_name for fe in f.core_fes],
                }
                for f in sorted(frames, key=lambda x: x.usage_count, reverse=True)
            ],
        }

    @app.get("/causal_chain", summary="Trace causal chain from a trigger word")
    async def causal_chain(
        trigger: str = Query(..., description="Trigger lemma to start from"),
        depth: int = Query(3, description="Max chain depth"),
    ):
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        chain = await _rag.get_causal_chain(trigger, depth=depth)
        return {"trigger": trigger, "chain": chain, "length": len(chain)}

    @app.delete("/clear", summary="Clear all indexed data")
    async def clear_data(confirm: str = Query(..., description="Type 'yes' to confirm")):
        if confirm.lower() != "yes":
            raise HTTPException(400, "Must pass confirm=yes to clear data")
        if _rag is None:
            raise HTTPException(503, "FrameRAG not initialized")
        import shutil
        await _rag.finalize()
        shutil.rmtree(_rag._working_dir, ignore_errors=True)
        os.makedirs(_rag._working_dir, exist_ok=True)
        await _rag.initialize()
        return {"status": "cleared"}

    return app


def create_from_env() -> FastAPI:
    """Create app from environment variables (for uvicorn --factory use)."""
    working_dir   = os.getenv("FRAMERAG_WORKING_DIR", "./framerag_storage")
    embedding_dim = int(os.getenv("FRAMERAG_EMBEDDING_DIM", "1536"))
    return create_app(working_dir=working_dir, embedding_dim=embedding_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Default OpenAI functions
# ─────────────────────────────────────────────────────────────────────────────

def _default_openai_funcs():
    """Return (llm_func, embed_func) using OpenAI defaults."""
    try:
        from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
        return gpt_4o_mini_complete, openai_embed
    except ImportError:
        pass

    import numpy as np
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    async def _llm(prompt: str, **kwargs) -> str:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=kwargs.get("stream", False),
        )
        if kwargs.get("stream"):
            return resp  # caller handles streaming
        return resp.choices[0].message.content or ""

    async def _embed(texts: list[str]) -> np.ndarray:
        resp = await client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    return _llm, _embed
