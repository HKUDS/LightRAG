"""
LightRAG Rerank Integration Example

This example demonstrates how to use rerank functionality with LightRAG
to improve retrieval quality across different query modes.

Configuration Required:
1. Set your OpenAI LLM API key and base URL with env vars
    LLM_MODEL
    LLM_BINDING_HOST
    LLM_BINDING_API_KEY
2. Set your OpenAI embedding API key and base URL with env vars:
    EMBEDDING_MODEL
    EMBEDDING_DIM
    EMBEDDING_BINDING_HOST
    EMBEDDING_BINDING_API_KEY
3. Set your vLLM deployed AI rerank model setting with env vars:
    RERANK_BINDING=cohere
    RERANK_MODEL (e.g., answerai-colbert-small-v1 or rerank-v3.5)
    RERANK_BINDING_HOST (e.g., https://api.cohere.com/v2/rerank or LiteLLM proxy)
    RERANK_BINDING_API_KEY
    RERANK_ENABLE_CHUNKING=true (optional, for models with token limits)
    RERANK_MAX_TOKENS_PER_DOC=480 (optional, default 4096)

Note: Rerank is controlled per query via the 'enable_rerank' parameter (default: True)
"""

import asyncio
import os
from functools import partial

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.rerank import cohere_rerank
from lightrag.utils import EmbeddingFunc, setup_logger

# Set up your working directory
WORKING_DIR = './test_rerank'
setup_logger('test_rerank')

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        os.getenv('LLM_MODEL'),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv('LLM_BINDING_API_KEY'),
        base_url=os.getenv('LLM_BINDING_HOST'),
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=os.getenv('EMBEDDING_MODEL'),
        api_key=os.getenv('EMBEDDING_BINDING_API_KEY'),
        base_url=os.getenv('EMBEDDING_BINDING_HOST'),
    )


rerank_model_func = partial(
    cohere_rerank,
    model=os.getenv('RERANK_MODEL', 'rerank-v3.5'),
    api_key=os.getenv('RERANK_BINDING_API_KEY'),
    base_url=os.getenv('RERANK_BINDING_HOST', 'https://api.cohere.com/v2/rerank'),
    enable_chunking=os.getenv('RERANK_ENABLE_CHUNKING', 'false').lower() == 'true',
    max_tokens_per_doc=int(os.getenv('RERANK_MAX_TOKENS_PER_DOC', '4096')),
)


async def create_rag_with_rerank():
    """Create LightRAG instance with rerank configuration"""

    # Get embedding dimension
    test_embedding = await embedding_func(['test'])
    embedding_dim = test_embedding.shape[1]
    print(f'Detected embedding dimension: {embedding_dim}')

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
        rerank_model_func=rerank_model_func,
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


async def test_rerank_with_different_settings():
    """
    Test rerank functionality with different enable_rerank settings
    """
    print('\n\n🚀 Setting up LightRAG with Rerank functionality...')

    rag = await create_rag_with_rerank()

    # Insert sample documents
    sample_docs = [
        'Reranking improves retrieval quality by re-ordering documents based on relevance.',
        'LightRAG is a powerful retrieval-augmented generation system with multiple query modes.',
        'Vector databases enable efficient similarity search in high-dimensional embedding spaces.',
        'Natural language processing has evolved with large language models and transformers.',
        'Machine learning algorithms can learn patterns from data without explicit programming.',
    ]

    print('📄 Inserting sample documents...')
    await rag.ainsert(sample_docs)

    query = 'How does reranking improve retrieval quality?'
    print(f"\n🔍 Testing query: '{query}'")
    print('=' * 80)

    # Test with rerank enabled (default)
    print('\n📊 Testing with enable_rerank=True (default):')
    result_with_rerank = await rag.aquery(
        query,
        param=QueryParam(
            mode='naive',
            top_k=10,
            chunk_top_k=5,
            enable_rerank=True,  # Explicitly enable rerank
        ),
    )
    print(f'   Result length: {len(result_with_rerank)} characters')
    print(f'   Preview: {result_with_rerank[:100]}...')

    # Test with rerank disabled
    print('\n📊 Testing with enable_rerank=False:')
    result_without_rerank = await rag.aquery(
        query,
        param=QueryParam(
            mode='naive',
            top_k=10,
            chunk_top_k=5,
            enable_rerank=False,  # Disable rerank
        ),
    )
    print(f'   Result length: {len(result_without_rerank)} characters')
    print(f'   Preview: {result_without_rerank[:100]}...')

    # Test with default settings (enable_rerank defaults to True)
    print('\n📊 Testing with default settings (enable_rerank defaults to True):')
    result_default = await rag.aquery(query, param=QueryParam(mode='naive', top_k=10, chunk_top_k=5))
    print(f'   Result length: {len(result_default)} characters')
    print(f'   Preview: {result_default[:100]}...')


async def test_direct_rerank():
    """Test rerank function directly"""
    print('\n🔧 Direct Rerank API Test')
    print('=' * 40)

    documents = [
        'Vector search finds semantically similar documents',
        'LightRAG supports advanced reranking capabilities',
        'Reranking significantly improves retrieval quality',
        'Natural language processing with modern transformers',
        'The quick brown fox jumps over the lazy dog',
    ]

    query = 'rerank improve quality'
    print(f"Query: '{query}'")
    print(f'Documents: {len(documents)}')

    try:
        reranked_results = await rerank_model_func(
            query=query,
            documents=documents,
            top_n=4,
        )

        print('\n✅ Rerank Results:')
        for _i, result in enumerate(reranked_results):
            index = result['index']
            score = result['relevance_score']
            content = documents[index]
            print(f'  {index}. Score: {score:.4f} | {content}...')

    except Exception as e:
        print(f'❌ Rerank failed: {e}')


async def main():
    """Main example function"""
    print('🎯 LightRAG Rerank Integration Example')
    print('=' * 60)

    try:
        # Test direct rerank
        await test_direct_rerank()

        # Test rerank with different enable_rerank settings
        await test_rerank_with_different_settings()

        print('\n✅ Example completed successfully!')
        print('\n💡 Key Points:')
        print("   ✓ Rerank is now controlled per query via 'enable_rerank' parameter")
        print('   ✓ Default value for enable_rerank is True')
        print('   ✓ Rerank function is configured at LightRAG initialization')
        print('   ✓ Per-query enable_rerank setting overrides default behavior')
        print('   ✓ If enable_rerank=True but no rerank model is configured, a warning is issued')
        print('   ✓ Monitor API usage and costs when using rerank services')

    except Exception as e:
        print(f'\n❌ Example failed: {e}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
