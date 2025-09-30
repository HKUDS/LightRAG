"""
LightRAG with LiteLLM Integration Demo

LiteLLM provides a unified interface to 100+ LLM providers including:
- OpenAI, Anthropic, Cohere, Replicate
- Azure, AWS Bedrock, Google VertexAI
- HuggingFace, Ollama, Together AI
- And many more!

This example demonstrates how to use LightRAG with LiteLLM for maximum flexibility.

Setup:
1. Install dependencies:
   pip install lightrag-hku litellm

2. Set environment variables for your chosen provider:
   # For OpenAI (default)
   export OPENAI_API_KEY="sk-..."

   # For Anthropic
   export ANTHROPIC_API_KEY="sk-ant-..."

   # For Cohere
   export COHERE_API_KEY="..."

   # For custom LiteLLM proxy
   export LITELLM_API_BASE="http://localhost:4000"

3. Run the script:
   python examples/lightrag_litellm_demo.py
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.litellm import litellm_complete_if_cache, litellm_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./dickens"

# LiteLLM Model Configuration
# Format: "provider/model-name" or just "model-name" for OpenAI
# Examples:
#   OpenAI: "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
#   Anthropic: "anthropic/claude-3-sonnet-20240229", "anthropic/claude-3-opus-20240229"
#   Cohere: "cohere/command-r-plus", "cohere/command-r"
#   Ollama: "ollama/llama2", "ollama/mistral"
#   Azure: "azure/your-deployment-name"
#   Bedrock: "bedrock/anthropic.claude-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Embedding Model Configuration
# Examples:
#   OpenAI: "text-embedding-3-small", "text-embedding-3-large"
#   Cohere: "cohere/embed-english-v3.0", "cohere/embed-multilingual-v3.0"
#   Azure: "azure/your-embedding-deployment"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Optional: Custom API base (e.g., for LiteLLM proxy server)
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", None)

# Optional: Custom API key (if not using environment variables)
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", None)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
):
    """LLM function using LiteLLM."""
    return await litellm_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_base=LITELLM_API_BASE,
        api_key=LITELLM_API_KEY,
        **kwargs,
    )


async def embedding_func(texts: list[str]):
    """Embedding function using LiteLLM."""
    return await litellm_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        api_base=LITELLM_API_BASE,
        api_key=LITELLM_API_KEY,
    )


async def initialize_rag():
    """Initialize LightRAG instance."""
    # Get embedding dimension dynamically
    test_embedding = await embedding_func(["test"])
    embedding_dim = test_embedding.shape[1]
    print(f"Detected embedding dimension: {embedding_dim}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    print(f"Using LLM Model: {LLM_MODEL}")
    print(f"Using Embedding Model: {EMBEDDING_MODEL}")
    if LITELLM_API_BASE:
        print(f"Using API Base: {LITELLM_API_BASE}")
    print()

    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        # Insert example text
        print("Inserting document...")
        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())
        print("Document inserted successfully!\n")

        # Test different query modes
        query = "What are the top themes in this story?"

        print("=" * 50)
        print("Naive Search:")
        print("=" * 50)
        result = await rag.aquery(query, param=QueryParam(mode="naive"))
        print(result)
        print()

        print("=" * 50)
        print("Local Search:")
        print("=" * 50)
        result = await rag.aquery(query, param=QueryParam(mode="local"))
        print(result)
        print()

        print("=" * 50)
        print("Global Search:")
        print("=" * 50)
        result = await rag.aquery(query, param=QueryParam(mode="global"))
        print(result)
        print()

        print("=" * 50)
        print("Hybrid Search:")
        print("=" * 50)
        result = await rag.aquery(query, param=QueryParam(mode="hybrid"))
        print(result)
        print()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    print("LightRAG with LiteLLM Demo")
    print("=" * 50)
    asyncio.run(main())
    print("\nDone!")