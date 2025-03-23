import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./all_modes_demo"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    # Initialize LightRAG with a base model (gpt-4o-mini)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,  # Default model for most queries
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
    
    # Example query
    query_text = "What are the main themes in this story?"
    
    # Demonstrate using default model (gpt-4o-mini) for all modes
    print("\n===== Default Model (gpt-4o-mini) =====")
    
    for mode in ["local", "global", "hybrid", "naive", "mix"]:
        print(f"\n--- {mode.upper()} mode with default model ---")
        response = rag.query(
            query_text, 
            param=QueryParam(mode=mode)
        )
        print(response)
    
    # Demonstrate using custom model (gpt-4o) for all modes
    print("\n===== Custom Model (gpt-4o) =====")
    
    for mode in ["local", "global", "hybrid", "naive", "mix"]:
        print(f"\n--- {mode.upper()} mode with custom model ---")
        response = rag.query(
            query_text, 
            param=QueryParam(
                mode=mode,
                model_func=gpt_4o_complete  # Override with more capable model
            )
        )
        print(response)
    
    # Mixed approach - use different models for different modes
    print("\n===== Strategic Model Selection =====")
    
    # Complex analytical question
    complex_query = "How does the character development in the story reflect Victorian-era social values?"
    
    # Use default model for simpler modes
    print("\n--- NAIVE mode with default model (suitable for simple retrieval) ---")
    response1 = rag.query(
        complex_query,
        param=QueryParam(mode="naive")  # Use default model for basic retrieval
    )
    print(response1)
    
    # Use more capable model for complex modes
    print("\n--- HYBRID mode with more capable model (for complex analysis) ---")
    response2 = rag.query(
        complex_query,
        param=QueryParam(
            mode="hybrid",
            model_func=gpt_4o_complete  # Use more capable model for complex analysis
        )
    )
    print(response2)


if __name__ == "__main__":
    main() 