"""
LightRAG Demo with OpenSearch + OpenAI

This example demonstrates how to use LightRAG with:
- OpenAI (LLM + Embeddings)
- OpenSearch-backed storages for:
  - KV storage
  - Vector storage (k-NN)
  - Graph storage (dual-index nodes + edges)
  - Document status storage

Prerequisites:
1. OpenSearch cluster running and accessible (3.x or higher with k-NN plugin)
2. Required indices will be auto-created by LightRAG
3. Set environment variables (example .env):

   OPENSEARCH_HOSTS=localhost:9200
   OPENSEARCH_USER=admin
   OPENSEARCH_PASSWORD=your-password
   OPENSEARCH_USE_SSL=false
   OPENSEARCH_VERIFY_CERTS=false

   OPENAI_API_KEY=your-api-key

4. Prepare a text file to index (default: ./book.txt)

Usage:
    python examples/lightrag_openai_opensearch_graph_demo.py
"""

import os
import asyncio
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger, EmbeddingFunc


# --------------------------------------------------
# Logger
# --------------------------------------------------
setup_logger("lightrag", level="INFO")


# --------------------------------------------------
# Config
# --------------------------------------------------
WORKING_DIR = "./opensearch_rag_storage"
BOOK_FILE = "./book.txt"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Replace with your API key, or set via environment variable
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))


# --------------------------------------------------
# Embedding function (OpenAI)
# --------------------------------------------------
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed.func(
        texts,
        model=EMBEDDING_MODEL,
    )


async def get_embedding_dimension():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    return embedding.shape[1]


async def create_embedding_function_instance():
    embedding_dimension = await get_embedding_dimension()
    return EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=embedding_func,
    )


# --------------------------------------------------
# Initialize RAG with OpenSearch storages
# --------------------------------------------------
async def initialize_rag() -> LightRAG:
    embedding_func_instance = await create_embedding_function_instance()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=embedding_func_instance,
        # OpenSearch-backed storages
        kv_storage="OpenSearchKVStorage",
        doc_status_storage="OpenSearchDocStatusStorage",
        graph_storage="OpenSearchGraphStorage",
        vector_storage="OpenSearchVectorDBStorage",
    )

    # REQUIRED: initialize all storage backends
    await rag.initialize_storages()

    # Clean previous data so the example is re-runnable
    for storage in [
        rag.full_docs,
        rag.text_chunks,
        rag.full_entities,
        rag.full_relations,
        rag.entity_chunks,
        rag.relation_chunks,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunks_vdb,
        rag.chunk_entity_relation_graph,
        rag.llm_response_cache,
        rag.doc_status,
    ]:
        await storage.drop()
    print("Cleared previous data.")

    return rag


# --------------------------------------------------
# Main
# --------------------------------------------------
async def main():
    rag = None
    try:
        print("Initializing LightRAG with OpenSearch + OpenAI...")
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
            print(result)

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
