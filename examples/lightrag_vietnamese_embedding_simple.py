"""
Simple LightRAG Example with Vietnamese Embedding Model

This is a minimal example showing how to use the Vietnamese_Embedding model
with LightRAG for Vietnamese text processing.

Setup:
    export HUGGINGFACE_API_KEY="your_hf_token_here"
    export OPENAI_API_KEY="your_openai_key_here"
    
Usage:
    python examples/lightrag_vietnamese_embedding_simple.py
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.llm.vietnamese_embed import vietnamese_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./vietnamese_rag_storage"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def main():
    # Get HuggingFace token from environment
    hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    
    # Initialize LightRAG with Vietnamese embedding
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,  # Vietnamese_Embedding outputs 1024 dimensions
            max_token_size=2048,  # Model supports up to 2048 tokens
            func=lambda texts: vietnamese_embed(
                texts,
                model_name="AITeamVN/Vietnamese_Embedding111",
                token=hf_token
            )
        ),
    )
    
    # Initialize storage and pipeline
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Insert Vietnamese text
    vietnamese_text = """
    Việt Nam là một quốc gia nằm ở Đông Nam Á. 
    Thủ đô là Hà Nội và thành phố lớn nhất là Thành phố Hồ Chí Minh.
    Việt Nam có dân số khoảng 100 triệu người.
    """
    
    print("Inserting Vietnamese text...")
    await rag.ainsert(vietnamese_text)
    
    # Query the system
    query = "Thủ đô của Việt Nam là gì?"
    print(f"\nQuery: {query}")
    
    result = await rag.aquery(
        query,
        param=QueryParam(mode="hybrid")
    )
    
    print(f"Answer: {result}")
    
    # Clean up
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
