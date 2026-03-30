"""
LightRAG Data Isolation Demo: Workspace Management

This example demonstrates how to maintain multiple isolated knowledge bases
within a single application using LightRAG's 'workspace' feature.

Key Concepts:
- Workspace Isolation: Each RAG instance is assigned a unique workspace name,
  which ensures that Knowledge Graphs, Vector Databases, and Chunks are
  stored in separate, non-conflicting directories.
- Independent Configuration: Different workspaces can use different document
  sets simultaneously. Entity type guidance is customized via the prompt template.

Prerequisites:
1. Set the following environment variables:
   - GEMINI_API_KEY: Your Google Gemini API key.
2. Ensure your data directory contains:
   - Data/book-small.txt
   - Data/HR_policies.txt

Usage:
    python lightrag_workspace_demo.py
"""

import os
import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs


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
    entity_types_guidance: str | None = None,
) -> LightRAG:
    """
    Initializes a LightRAG instance with data isolation.

    Entity type guidance is controlled via addon_params['entity_types_guidance'].
    If not provided, the default entity types defined in the prompt template are used.

    Example — custom entity types for a medical knowledge base:
        entity_types_guidance = \"\"\"
        Use the following entity types to classify extracted entities:
        - Disease: Medical conditions or illnesses
        - Drug: Pharmaceutical substances or treatments
        - Symptom: Clinical signs or patient-reported complaints
        - Anatomy: Body parts, organs, or biological structures
        - Procedure: Medical or surgical procedures
        \"\"\"
        rag = await initialize_rag("rag_workspace_medical", entity_types_guidance)
    """
    addon_params = {"language": "English"}
    if entity_types_guidance is not None:
        addon_params["entity_types_guidance"] = entity_types_guidance

    rag = LightRAG(
        workspace=workspace,
        llm_model_name="gemini-2.0-flash",
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        embedding_func_max_async=4,
        embedding_batch_num=8,
        llm_model_max_async=2,
        addon_params=addon_params,
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
        rag_1 = await initialize_rag(
            "rag_workspace_book",
            entity_types_guidance="""Use the following entity types to classify extracted entities:
- Person: Individual human beings, real or fictional characters in the story
- Location: Physical places such as cities, countries, buildings, or fictional settings
- Event: Significant occurrences or plot points in the narrative
- Organization: Groups, factions, guilds, or institutions within the story
- Artifact: Important objects, weapons, tools, or items with narrative significance
- Concept: Abstract ideas, themes, beliefs, or philosophies explored in the text""",
        )
        rag_2 = await initialize_rag(
            "rag_workspace_hr",
            entity_types_guidance="""Use the following entity types to classify extracted entities:
- Policy: HR rules, regulations, guidelines, or procedures
- Role: Job titles, positions, or employee classifications
- Benefit: Compensation, insurance, leave entitlements, or perks
- Department: Organizational units or teams within the company
- Process: Workflows, approval chains, or operational procedures
- Regulation: Legal requirements, compliance standards, or labor laws""",
        )

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
