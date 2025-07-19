"""
LightRAG Rerank Integration Example

This example demonstrates how to use rerank functionality with LightRAG
to improve retrieval quality across different query modes.

Configuration Required:
1. Set your LLM API key and base URL in llm_model_func()
2. Set your embedding API key and base URL in embedding_func()
3. Set your rerank API key and base URL in the rerank configuration
4. Or use environment variables (.env file):
   - RERANK_MODEL=your_rerank_model
   - RERANK_BINDING_HOST=your_rerank_endpoint
   - RERANK_BINDING_API_KEY=your_rerank_api_key

Note: Rerank is now controlled per query via the 'enable_rerank' parameter (default: True)
"""

import asyncio
import os
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.rerank import custom_rerank, RerankModel
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status

# Set up your working directory
WORKING_DIR = "./test_rerank"
setup_logger("test_rerank")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="your_llm_api_key_here",
        base_url="https://api.your-llm-provider.com/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="text-embedding-3-large",
        api_key="your_embedding_api_key_here",
        base_url="https://api.your-embedding-provider.com/v1",
    )


async def my_rerank_func(query: str, documents: list, top_n: int = None, **kwargs):
    """Custom rerank function with all settings included"""
    return await custom_rerank(
        query=query,
        documents=documents,
        model="BAAI/bge-reranker-v2-m3",
        base_url="https://api.your-rerank-provider.com/v1/rerank",
        api_key="your_rerank_api_key_here",
        top_n=top_n or 10,
        **kwargs,
    )


async def create_rag_with_rerank():
    """Create LightRAG instance with rerank configuration"""

    # Get embedding dimension
    test_embedding = await embedding_func(["test"])
    embedding_dim = test_embedding.shape[1]
    print(f"Detected embedding dimension: {embedding_dim}")

    # Method 1: Using custom rerank function
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embedding_func,
        ),
        # Rerank Configuration - provide the rerank function
        rerank_model_func=my_rerank_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def create_rag_with_rerank_model():
    """Alternative: Create LightRAG instance using RerankModel wrapper"""

    # Get embedding dimension
    test_embedding = await embedding_func(["test"])
    embedding_dim = test_embedding.shape[1]
    print(f"Detected embedding dimension: {embedding_dim}")

    # Method 2: Using RerankModel wrapper
    rerank_model = RerankModel(
        rerank_func=custom_rerank,
        kwargs={
            "model": "BAAI/bge-reranker-v2-m3",
            "base_url": "https://api.your-rerank-provider.com/v1/rerank",
            "api_key": "your_rerank_api_key_here",
        },
    )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embedding_func,
        ),
        rerank_model_func=rerank_model.rerank,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def test_rerank_with_different_settings():
    """
    Test rerank functionality with different enable_rerank settings
    """
    print("🚀 Setting up LightRAG with Rerank functionality...")

    rag = await create_rag_with_rerank()

    # Insert sample documents
    sample_docs = [
        "Reranking improves retrieval quality by re-ordering documents based on relevance.",
        "LightRAG is a powerful retrieval-augmented generation system with multiple query modes.",
        "Vector databases enable efficient similarity search in high-dimensional embedding spaces.",
        "Natural language processing has evolved with large language models and transformers.",
        "Machine learning algorithms can learn patterns from data without explicit programming.",
    ]

    print("📄 Inserting sample documents...")
    await rag.ainsert(sample_docs)

    query = "How does reranking improve retrieval quality?"
    print(f"\n🔍 Testing query: '{query}'")
    print("=" * 80)

    # Test with rerank enabled (default)
    print("\n📊 Testing with enable_rerank=True (default):")
    result_with_rerank = await rag.aquery(
        query,
        param=QueryParam(
            mode="naive",
            top_k=10,
            chunk_top_k=5,
            enable_rerank=True,  # Explicitly enable rerank
        ),
    )
    print(f"   Result length: {len(result_with_rerank)} characters")
    print(f"   Preview: {result_with_rerank[:100]}...")

    # Test with rerank disabled
    print("\n📊 Testing with enable_rerank=False:")
    result_without_rerank = await rag.aquery(
        query,
        param=QueryParam(
            mode="naive",
            top_k=10,
            chunk_top_k=5,
            enable_rerank=False,  # Disable rerank
        ),
    )
    print(f"   Result length: {len(result_without_rerank)} characters")
    print(f"   Preview: {result_without_rerank[:100]}...")

    # Test with default settings (enable_rerank defaults to True)
    print("\n📊 Testing with default settings (enable_rerank defaults to True):")
    result_default = await rag.aquery(
        query, param=QueryParam(mode="naive", top_k=10, chunk_top_k=5)
    )
    print(f"   Result length: {len(result_default)} characters")
    print(f"   Preview: {result_default[:100]}...")


async def test_direct_rerank():
    """Test rerank function directly"""
    print("\n🔧 Direct Rerank API Test")
    print("=" * 40)

    documents = [
        {"content": "Reranking significantly improves retrieval quality"},
        {"content": "LightRAG supports advanced reranking capabilities"},
        {"content": "Vector search finds semantically similar documents"},
        {"content": "Natural language processing with modern transformers"},
        {"content": "The quick brown fox jumps over the lazy dog"},
    ]

    query = "rerank improve quality"
    print(f"Query: '{query}'")
    print(f"Documents: {len(documents)}")

    try:
        reranked_docs = await custom_rerank(
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
            base_url="https://api.your-rerank-provider.com/v1/rerank",
            api_key="your_rerank_api_key_here",
            top_n=3,
        )

        print("\n✅ Rerank Results:")
        for i, doc in enumerate(reranked_docs):
            score = doc.get("rerank_score", "N/A")
            content = doc.get("content", "")[:60]
            print(f"  {i+1}. Score: {score:.4f} | {content}...")

    except Exception as e:
        print(f"❌ Rerank failed: {e}")


async def main():
    """Main example function"""
    print("🎯 LightRAG Rerank Integration Example")
    print("=" * 60)

    try:
        # Test rerank with different enable_rerank settings
        await test_rerank_with_different_settings()

        # Test direct rerank
        await test_direct_rerank()

        print("\n✅ Example completed successfully!")
        print("\n💡 Key Points:")
        print("   ✓ Rerank is now controlled per query via 'enable_rerank' parameter")
        print("   ✓ Default value for enable_rerank is True")
        print("   ✓ Rerank function is configured at LightRAG initialization")
        print("   ✓ Per-query enable_rerank setting overrides default behavior")
        print(
            "   ✓ If enable_rerank=True but no rerank model is configured, a warning is issued"
        )
        print("   ✓ Monitor API usage and costs when using rerank services")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
