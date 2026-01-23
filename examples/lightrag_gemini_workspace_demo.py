"""
LightRAG Data Isolation Demo: Workspace Management

This example demonstrates how to maintain multiple isolated knowledge bases
within a single application using LightRAG's 'workspace' feature.

Key Concepts:
- Workspace Isolation: Each RAG instance is assigned a unique workspace name,
  which ensures that Knowledge Graphs, Vector Databases, and Chunks are
  stored in separate, non-conflicting directories.
- Independent Configuration: Different workspaces can utilize different
  ENTITY_TYPES and document sets simultaneously.

Prerequisites:
1. Set the following environment variables:
   - GEMINI_API_KEY: Your Google Gemini API key.
   - ENTITY_TYPES: A JSON string of entity categories (e.g., '["Person", "Organization"]').
2. Ensure your data directory contains:
   - Data/book-small.txt
   - Data/HR_policies.txt

Usage:
    python lightrag_workspace_demo.py
"""

import os
import asyncio
import json
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.constants import DEFAULT_ENTITY_TYPES


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """Wrapper for Gemini LLM completion."""
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash-exp",
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=768, max_token_size=2048, model_name="models/text-embedding-004"
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    """Wrapper for Gemini embedding model."""
    return await gemini_embed.func(
        texts, api_key=os.getenv("GEMINI_API_KEY"), model="models/text-embedding-004"
    )


async def initialize_rag(
    workspace: str = "default_workspace",
    entities=None,
) -> LightRAG:
    """
    Initializes a LightRAG instance with data isolation.

    - entities (if provided) overrides everything
    - else ENTITY_TYPES env var is used
    - else DEFAULT_ENTITY_TYPES is used
    """

    if entities is not None:
        entity_types = entities
    else:
        env_entities = os.getenv("ENTITY_TYPES")
        if env_entities:
            entity_types = json.loads(env_entities)
        else:
            entity_types = DEFAULT_ENTITY_TYPES

    rag = LightRAG(
        workspace=workspace,
        llm_model_name="gemini-2.0-flash",
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        embedding_func_max_async=4,
        embedding_batch_num=8,
        llm_model_max_async=2,
        addon_params={"entity_types": entity_types},
    )

    await rag.initialize_storages()
    return rag


async def main():
    rag_1 = None
    rag_2 = None
    try:
        # 1. Initialize Isolated Workspaces
        # Instance 1: Dedicated to literary analysis
        # Instance 2: Dedicated to corporate HR documentation
        print("Initializing isolated LightRAG workspaces...")
        rag_1 = await initialize_rag("rag_workspace_book")
        rag_2 = await initialize_rag("rag_workspace_hr")

        # 2. Populate Workspace 1 (Literature)
        book_path = "Data/book-small.txt"
        if os.path.exists(book_path):
            with open(book_path, "r", encoding="utf-8") as f:
                print(f"Indexing {book_path} into Literature Workspace...")
                await rag_1.ainsert(f.read())

        # 3. Populate Workspace 2 (Corporate)
        hr_path = "Data/HR_policies.txt"
        if os.path.exists(hr_path):
            with open(hr_path, "r", encoding="utf-8") as f:
                print(f"Indexing {hr_path} into HR Workspace...")
                await rag_2.ainsert(f.read())

        # 4. Context-Specific Querying
        print("\n--- Querying Literature Workspace ---")
        res1 = await rag_1.aquery(
            "What is the main theme?",
            param=QueryParam(mode="hybrid", stream=False),
        )
        print(f"Book Analysis: {res1[:200]}...")

        print("\n--- Querying HR Workspace ---")
        res2 = await rag_2.aquery(
            "What is the leave policy?", param=QueryParam(mode="hybrid")
        )
        print(f"HR Response: {res2[:200]}...")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Finalize storage to safely close DB connections and write buffers
        if rag_1:
            await rag_1.finalize_storages()
        if rag_2:
            await rag_2.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
