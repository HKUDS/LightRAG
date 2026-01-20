"""
LightRAG Demo with vLLM (LLM, Embeddings, and Reranker)

This example demonstrates how to use LightRAG with:
- vLLM-served LLM (OpenAI-compatible API)
- vLLM-served embedding model
- Jina-compatible reranker (also vLLM-served)

Prerequisites:
    1. Create a .env file or export environment variables:
       - LLM_MODEL
       - LLM_BINDING_HOST
       - LLM_BINDING_API_KEY
       - EMBEDDING_MODEL
       - EMBEDDING_BINDING_HOST
       - EMBEDDING_BINDING_API_KEY
       - EMBEDDING_DIM
       - EMBEDDING_TOKEN_LIMIT
       - RERANK_MODEL
       - RERANK_BINDING_HOST
       - RERANK_BINDING_API_KEY

    2. Prepare a text file to index (default: Data/book-small.txt)

    3. Configure storage backends via environment variables or modify
       the storage parameters in initialize_rag() below.

Usage:
    python examples/lightrag_vllm_demo.py
"""

import os
import asyncio
from functools import partial
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.rerank import jina_rerank

load_dotenv()

# --------------------------------------------------
# Constants
# --------------------------------------------------

WORKING_DIR = "./LightRAG_Data"
BOOK_FILE = "Data/book-small.txt"

# --------------------------------------------------
# LLM function (vLLM, OpenAI-compatible)
# --------------------------------------------------


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=os.getenv("LLM_MODEL", "Qwen/Qwen3-14B-AWQ"),
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=os.getenv("LLM_BINDING_HOST", "http://0.0.0.0:4646/v1"),
        api_key=os.getenv("LLM_BINDING_API_KEY", "not_needed"),
        timeout=600,
        **kwargs,
    )


# --------------------------------------------------
# Embedding function (vLLM)
# --------------------------------------------------

vLLM_emb_func = EmbeddingFunc(
    model_name=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
    send_dimensions=False,
    embedding_dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", 4096)),
    func=partial(
        openai_embed.func,
        model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        base_url=os.getenv(
            "EMBEDDING_BINDING_HOST",
            "http://0.0.0.0:1234/v1",
        ),
        api_key=os.getenv("EMBEDDING_BINDING_API_KEY", "not_needed"),
    ),
)

# --------------------------------------------------
# Reranker (Jina-compatible, vLLM-served)
# --------------------------------------------------

jina_rerank_model_func = partial(
    jina_rerank,
    model=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
    api_key=os.getenv("RERANK_BINDING_API_KEY"),
    base_url=os.getenv(
        "RERANK_BINDING_HOST",
        "http://0.0.0.0:3535/v1/rerank",
    ),
)

# --------------------------------------------------
# Initialize RAG
# --------------------------------------------------


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=vLLM_emb_func,
        rerank_model_func=jina_rerank_model_func,
        # Storage backends (configurable via environment or modify here)
        kv_storage=os.getenv("KV_STORAGE", "PGKVStorage"),
        doc_status_storage=os.getenv("DOC_STATUS_STORAGE", "PGDocStatusStorage"),
        vector_storage=os.getenv("VECTOR_STORAGE", "PGVectorStorage"),
        graph_storage=os.getenv("GRAPH_STORAGE", "Neo4JStorage"),
    )

    await rag.initialize_storages()
    return rag


# --------------------------------------------------
# Main
# --------------------------------------------------


async def main():
    rag = None
    try:
        # Validate book file exists
        if not os.path.exists(BOOK_FILE):
            raise FileNotFoundError(
                f"'{BOOK_FILE}' not found. Please provide a text file to index."
            )

        rag = await initialize_rag()

        # --------------------------------------------------
        # Data Ingestion
        # --------------------------------------------------
        print(f"Indexing {BOOK_FILE}...")
        with open(BOOK_FILE, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())
        print("Indexing complete.")

        # --------------------------------------------------
        # Query
        # --------------------------------------------------
        query = (
            "What are the main themes of the book, and how do the key characters "
            "evolve throughout the story?"
        )

        print("\nHybrid Search with Reranking:")
        result = await rag.aquery(
            query,
            param=QueryParam(
                mode="hybrid",
                stream=False,
                enable_rerank=True,
            ),
        )

        print("\nResult:\n", result)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
    print("\nDone!")
