"""
LightRAG demo with OpenAI + NebulaGraphStorage.

This demo focuses on manual configuration for an external NebulaGraph cluster.

Required environment variables:
    LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage
    NEBULA_HOSTS=127.0.0.1:9669
    NEBULA_USER=root
    NEBULA_PASSWORD=nebula
    OPENAI_API_KEY=your-openai-api-key

Notes:
    - NebulaGraphStorage maps each LightRAG workspace to one Nebula SPACE.
    - For higher-quality search_labels, deploy Nebula full-text search
      dependencies (Elasticsearch + Listener).
"""

import asyncio
import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed


WORKING_DIR = "./nebula_rag_storage"
BOOK_FILE = "./book.txt"


async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        graph_storage="NebulaGraphStorage",
    )
    await rag.initialize_storages()
    return rag


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    rag = None
    try:
        rag = await initialize_rag()
        with open(BOOK_FILE, "r", encoding="utf-8") as file:
            await rag.ainsert(file.read())

        answer = await rag.aquery(
            "What are the top themes in this document?",
            param=QueryParam(mode="hybrid"),
        )
        print(answer)
    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == "__main__":
    os.makedirs(WORKING_DIR, exist_ok=True)
    asyncio.run(main())
