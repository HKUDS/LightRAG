import asyncio
import os
import json
import logging  # Add logging import
from dotenv import load_dotenv
import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.infinity import infinity_embed, cleanup_infinity_models
from lightrag.llm.openai import openai_embed, openai_complete
from lightrag.base import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load environment variables
load_dotenv()

# Configure logging for debugging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_demo')

# Global variables
# MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-l-v2.0")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-l-v2.0")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Ensure we're using the correct environment variable for OPENAI_API_KEY
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable is not set.")

# Synthetic data - a small collection of programming language information
DOCUMENTS = [
    "Python is a high-level, interpreted programming language known for its readability and simplicity. Created by Guido van Rossum in 1991, it has become one of the most popular programming languages for web development, data science, AI, and automation.",
    "JavaScript is a high-level, just-in-time compiled programming language that is one of the core technologies of the World Wide Web. Initially created to make web pages interactive, it has evolved into a versatile language used for both front-end and back-end development.",
    "Rust is a systems programming language focused on safety, speed, and concurrency. Developed by Mozilla, Rust is syntactically similar to C++ but provides memory safety without using garbage collection. It's ideal for building reliable and efficient software.",
    "Go (or Golang) is a statically typed, compiled programming language designed at Google. It's known for its simplicity, efficiency, and built-in support for concurrent programming. Go is widely used for cloud services, server-side applications, and DevOps tools."
]

# Document IDs
DOC_IDS = ["python-doc", "javascript-doc", "rust-doc", "go-doc"]

# Example queries to test the RAG system with different modes
QUERIES = [
    "Compare Rust and Go programming languages",
    "Which programming language is best for web development?",
]

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await infinity_embed(
        texts,
        model_name=MODEL_NAME
    )


async def main():
    try:
        # Initialize the RAG with OpenAI embeddings and OpenAI for completions
        rag_params = {
            "llm_model_name": LLM_MODEL,
            "llm_model_func": openai_complete,
            "embedding_func": EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=embedding_func
            ),
            # Specify NetworkX as the knowledge graph implementation
            "graph_storage": "NetworkXStorage",
            # Configure better entity extraction
            "chunk_overlap_token_size": 25,   # Increase overlap for better entity linking
            # Add entity types to extract
            "addon_params": {
                "entity_types": ["language", "concept"]
            }
        }

        print(f"Using embedding model: {MODEL_NAME}")
        print(f"Using LLM model: {LLM_MODEL}")
        print("Using NetworkX for knowledge graph storage")

        # Initialize RAG with parameters
        rag = LightRAG(**rag_params)

        # Initialize storages and pipeline status
        await rag.initialize_storages()
        await initialize_pipeline_status()

        # Insert all documents together with their IDs
        print(f"Inserting {len(DOCUMENTS)} documents with IDs...")
        await rag.ainsert(DOCUMENTS, ids=DOC_IDS)

        kg = await rag.get_knowledge_graph("Python", max_depth=2, inclusive=True)
        print(f"Knowledge graph nodes: {len(kg.nodes)}")
        print(f"Knowledge graph edges: {len(kg.edges)}")


        # Test with regular queries first
        for i, query in enumerate(QUERIES):
            print(f"\n--- Query {i+1}: {query} ---")

            # Create query parameters for naive mode with only_need_context=True
            param = QueryParam(mode="naive", top_k=2, only_need_context=True)

            # Execute the query with naive mode to get only the context
            context = await rag.aquery(query, param)
            print(f"Context, top_k={param.top_k}: \n{context}")

        # Test with mix mode and json_response=True
        print("\n--- Testing mix mode with JSON response ---")
        mix_param = QueryParam(
            mode="mix",
            top_k=1,
            only_need_context=True,
            json_response=True
        )

        # Add debugging info before the call
        logger.info("DEBUG: About to call mix mode query with param: %s", mix_param)

        # Inspect available chunks before querying
        chunks_info = await rag.text_chunks.get_all()
        logger.info(f"DEBUG: Available text chunks before query: {chunks_info}")

        mix_context = await rag.aquery(QUERIES[0], mix_param)
        logger.info("DEBUG: Mix query completed successfully")
        print(f"Mix JSON Context: \n{json.dumps(mix_context, indent=2)}")  # Pretty print JSON
    finally:
        # Ensure cleanup of infinity models happens
        await cleanup_infinity_models()
        logger.info("Infinity models cleaned up")

if __name__ == "__main__":
    asyncio.run(main())