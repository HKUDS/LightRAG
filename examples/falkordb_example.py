#!/usr/bin/env python
"""
Example of using LightRAG with FalkorDB
"""

import asyncio
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc


async def main():
    """Example usage of LightRAG with FalkorDB"""
    
    # Set up environment for FalkorDB
    os.environ["LIGHTRAG_GRAPH_STORAGE"] = "FalkorDBStorage"
    os.environ["FALKORDB_HOST"] = "localhost"
    os.environ["FALKORDB_PORT"] = "6379"
    os.environ["FALKORDB_GRAPH_NAME"] = "lightrag_example"
    
    # Initialize LightRAG with FalkorDB
    rag = LightRAG(
        working_dir="./falkordb_example",
        llm_model_func=openai_complete_if_cache,  # Use your preferred LLM
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=openai_embedding  # Use your preferred embedding
        ),
    )
    
    # Example text to process
    sample_text = """
    FalkorDB is a high-performance graph database built on Redis. 
    It supports Cypher queries and provides excellent performance for graph operations.
    LightRAG can now use FalkorDB as its graph storage backend, enabling scalable
    knowledge graph operations with Redis-based persistence.
    """
    
    print("Inserting text into LightRAG with FalkorDB backend...")
    await rag.ainsert(sample_text)
    
    print("Querying the knowledge graph...")
    response = await rag.aquery(
        "What is FalkorDB and how does it relate to LightRAG?",
        param=QueryParam(mode="hybrid")
    )
    
    print("Response:", response)


if __name__ == "__main__":
    print("LightRAG with FalkorDB Example")
    print("==============================")
    print("Note: This requires FalkorDB running on localhost:6379")
    print("You can start FalkorDB with: docker run -p 6379:6379 falkordb/falkordb")
    print()
    
    asyncio.run(main())