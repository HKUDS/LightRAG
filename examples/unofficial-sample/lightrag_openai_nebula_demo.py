"""
LightRAG demo with OpenAI + NebulaGraphStorage.

This demo focuses on manual configuration for an external NebulaGraph cluster.

Required environment variables:
    NEBULA_HOSTS=127.0.0.1:9669
    NEBULA_USER=root
    NEBULA_PASSWORD=nebula
    OPENAI_API_KEY=your-openai-api-key

Optional environment variables:
    NEBULA_LISTENER_HOSTS=172.28.0.10:9789
        Enable auto-registration of Nebula listeners for newly created workspaces.
        Use this when you want new workspaces to automatically provision full-text support.

Required input file:
    BOOK_FILE=./book.txt (must exist before running this demo)

Notes:
    - This script already sets graph_storage="NebulaGraphStorage" in code.
      If you prefer env-based selection, you may set:
      LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage
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
    required_envs = ("OPENAI_API_KEY", "NEBULA_HOSTS", "NEBULA_USER")
    missing_envs = [name for name in required_envs if not os.getenv(name)]
    if "NEBULA_PASSWORD" not in os.environ:
        missing_envs.append("NEBULA_PASSWORD")
    if missing_envs:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing_envs)
            + ". Please configure Nebula and OpenAI credentials before running this demo. "
            + "NEBULA_PASSWORD may be an empty string, but it must still be set."
        )
    if not os.path.exists(BOOK_FILE):
        raise FileNotFoundError(
            f"Input file '{BOOK_FILE}' was not found. Prepare a text file to index first, "
            "or update BOOK_FILE in this script."
        )

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
