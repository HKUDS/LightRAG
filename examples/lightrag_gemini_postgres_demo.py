"""
LightRAG Demo with PostgreSQL + Google Gemini

This example demonstrates how to use LightRAG with:
- Google Gemini (LLM + Embeddings)
- PostgreSQL-backed storages for:
  - Vector storage
  - Graph storage
  - KV storage
  - Document status storage

Prerequisites:
1. PostgreSQL database running and accessible
2. Required tables will be auto-created by LightRAG
3. Set environment variables (example .env):

   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_USER=admin
   POSTGRES_PASSWORD=admin
   POSTGRES_DATABASE=ai

   LIGHTRAG_KV_STORAGE=PGKVStorage
   LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
   LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
   LIGHTRAG_VECTOR_STORAGE=PGVectorStorage

   GEMINI_API_KEY=your-api-key

4. Prepare a text file to index (default: Data/book-small.txt)

Usage:
    python examples/lightrag_postgres_demo.py
"""

import os
import asyncio
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import setup_logger, wrap_embedding_func_with_attrs


# --------------------------------------------------
# Logger
# --------------------------------------------------
setup_logger("lightrag", level="INFO")


# --------------------------------------------------
# Config
# --------------------------------------------------
WORKING_DIR = "./rag_storage"
BOOK_FILE = "Data/book.txt"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")


# --------------------------------------------------
# LLM function (Gemini)
# --------------------------------------------------
async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs,
) -> str:
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.0-flash",
        **kwargs,
    )


# --------------------------------------------------
# Embedding function (Gemini)
# --------------------------------------------------
@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=2048,
    model_name="models/text-embedding-004",
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await gemini_embed.func(
        texts,
        api_key=GEMINI_API_KEY,
        model="models/text-embedding-004",
    )


# --------------------------------------------------
# Initialize RAG with PostgreSQL storages
# --------------------------------------------------
async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_name="gemini-2.0-flash",
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        # Performance tuning
        embedding_func_max_async=4,
        embedding_batch_num=8,
        llm_model_max_async=2,
        # Chunking
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        # PostgreSQL-backed storages
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
        doc_status_storage="PGDocStatusStorage",
        kv_storage="PGKVStorage",
    )

    # REQUIRED: initialize all storage backends
    await rag.initialize_storages()
    return rag


# --------------------------------------------------
# Main
# --------------------------------------------------
async def main():
    rag = None
    try:
        print("Initializing LightRAG with PostgreSQL + Gemini...")
        rag = await initialize_rag()

        if not os.path.exists(BOOK_FILE):
            raise FileNotFoundError(
                f"'{BOOK_FILE}' not found. Please provide a text file to index."
            )

        print(f"\nReading document: {BOOK_FILE}")
        with open(BOOK_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"Loaded document ({len(content)} characters)")

        print("\nInserting document into LightRAG (this may take some time)...")
        await rag.ainsert(content)
        print("Document indexed successfully!")

        print("\n" + "=" * 60)
        print("Running sample queries")
        print("=" * 60)

        query = "What are the top themes in this document?"

        for mode in ["naive", "local", "global", "hybrid"]:
            print(f"\n[{mode.upper()} MODE]")
            result = await rag.aquery(query, param=QueryParam(mode=mode))
            print(result[:400] + "..." if len(result) > 400 else result)

        print("\nRAG system is ready for use!")

    except Exception as e:
        print("An error occurred:", e)
        import traceback

        traceback.print_exc()

    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
