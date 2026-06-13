"""
Helper: LightRAG với Gemini 2.5 Flash qua Vertex AI.

Dùng Application Default Credentials (không cần API key).
Chạy một lần trước:  gcloud auth application-default login

Ví dụ sử dụng:
    from gemini_vertex import build_rag
    import asyncio

    async def main():
        rag = build_rag(working_dir="./my_storage")
        await rag.initialize_storages()
        await rag.ainsert("Your text here")
        answer = await rag.aquery("Your question")
        print(answer)
        await rag.finalize_storages()

    asyncio.run(main())
"""
from __future__ import annotations

import os
import numpy as np
from functools import partial, lru_cache

# Vertex AI config — đọc từ env hoặc dùng mặc định
GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT",  "vertical-reason-476709-v8")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
LLM_MODEL    = os.getenv("LLM_MODEL",             "gemini-2.5-flash")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL",       "gemini-embedding-001")
EMBED_DIM    = int(os.getenv("EMBEDDING_DIM",     "1536"))


@lru_cache(maxsize=1)
def _client():
    from google import genai
    return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)


def _build_embed_func():
    from lightrag.utils import wrap_embedding_func_with_attrs
    from google.genai import types as _gt

    @wrap_embedding_func_with_attrs(embedding_dim=EMBED_DIM, max_token_size=2048)
    async def vertex_embed(texts: list[str]) -> np.ndarray:
        r = await _client().aio.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=_gt.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBED_DIM,
            ),
        )
        return np.array([e.values for e in r.embeddings], dtype=np.float32)

    return vertex_embed


def _build_llm_func(model: str = LLM_MODEL):
    from lightrag.llm.gemini import gemini_complete_if_cache
    return partial(gemini_complete_if_cache, model)


def build_rag(
    working_dir: str = "./rag_storage",
    llm_model: str = LLM_MODEL,
    frame_mode: str | None = None,
) -> "LightRAG":
    """
    Tạo LightRAG instance dùng Gemini Vertex AI.

    Args:
        working_dir:  Thư mục lưu trữ graph + cache.
        llm_model:    Model Gemini cho LLM (mặc định gemini-2.5-flash).
        frame_mode:   'llm_frames' hoặc 'none'. None = đọc từ env.
    """
    if frame_mode:
        os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = frame_mode

    # Đặt Vertex AI env trước khi import LightRAG
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI",  "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT",       GCP_PROJECT)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION",      GCP_LOCATION)

    from lightrag import LightRAG

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=_build_llm_func(llm_model),
        embedding_func=_build_embed_func(),
    )
