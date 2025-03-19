import asyncio
import os
from dotenv import load_dotenv
import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.infinity import infinity_embed

# Load environment variables
load_dotenv()

# Global variables
CHUNKS = 4
MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"  # Default model for embeddings

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Example embedding function using Infinity embeddings with SentenceTransformer"""
    return await infinity_embed(
        texts,
        model_name=MODEL_NAME
    )

async def query_embedding_func(texts: list[str]) -> np.ndarray:
    """Specific embedding function for queries that uses the query prefix"""
    return await infinity_embed(
        texts,
        model_name=MODEL_NAME,
        is_query=True  # This will add the "query: " prefix for Snowflake models
    )

async def main():
    # Initialize the RAG with Infinity embeddings
    rag = LightRAG(
        working_dir="lightrag_infinity",
        llm_model_name="gpt-3.5-turbo",  # This can be any model you want to use for completions
        llm_model_func=lambda prompt, **kwargs: "This is a mock response for the demo",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,  # Snowflake/snowflake-arctic-embed-l-v2.0 uses 1024 dimensions
            max_token_size=8192,
            func=embedding_func
        ),
        # Use a different embedding function for queries that adds the "query: " prefix
        query_embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=query_embedding_func
        )
    )

    # Add some documents
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language.",
        "Embeddings are vector representations of text."
    ]

    # Add each text as a document
    for i, text in enumerate(texts):
        doc_id = f"doc_{i}"
        await rag.add_doc(doc_id=doc_id, content=text, metadata={"source": "demo"})
        print(f"Added document {doc_id}")

    # Query the documents
    query = "Tell me about programming languages"
    result = await rag.gen(query, top_k=2)

    print(f"\nQuery: {query}")
    print(f"Top contexts: {result.contexts}")
    print(f"Response: {result.response}")

if __name__ == "__main__":
    asyncio.run(main())