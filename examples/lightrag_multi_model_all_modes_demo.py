import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./lightrag_demo"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,  # Default model for queries
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Load the data
    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Query with naive mode (default model)
    print("--- NAIVE mode ---")
    print(
        rag.query(
            "What are the main themes in this story?", param=QueryParam(mode="naive")
        )
    )

    # Query with local mode (default model)
    print("\n--- LOCAL mode ---")
    print(
        rag.query(
            "What are the main themes in this story?", param=QueryParam(mode="local")
        )
    )

    # Query with global mode (default model)
    print("\n--- GLOBAL mode ---")
    print(
        rag.query(
            "What are the main themes in this story?", param=QueryParam(mode="global")
        )
    )

    # Query with hybrid mode (default model)
    print("\n--- HYBRID mode ---")
    print(
        rag.query(
            "What are the main themes in this story?", param=QueryParam(mode="hybrid")
        )
    )

    # Query with mix mode (default model)
    print("\n--- MIX mode ---")
    print(
        rag.query(
            "What are the main themes in this story?", param=QueryParam(mode="mix")
        )
    )

    # Query with a custom model (gpt-4o) for a more complex question
    print("\n--- Using custom model for complex analysis ---")
    print(
        rag.query(
            "How does the character development reflect Victorian-era attitudes?",
            param=QueryParam(
                mode="global",
                model_func=gpt_4o_complete,  # Override default model with more capable one
            ),
        )
    )


if __name__ == "__main__":
    main()
