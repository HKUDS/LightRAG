# Rerank Integration Guide

LightRAG supports reranking functionality to improve retrieval quality by re-ordering documents based on their relevance to the query. Reranking is now controlled per query via the `enable_rerank` parameter (default: True).

## Quick Start

### Environment Variables

Set these variables in your `.env` file or environment for rerank model configuration:

```bash
# Rerank model configuration (required when enable_rerank=True in queries)
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=https://api.your-provider.com/v1/rerank
RERANK_BINDING_API_KEY=your_api_key_here
```

### Programmatic Configuration

```python
from lightrag import LightRAG, QueryParam
from lightrag.rerank import custom_rerank, RerankModel

# Method 1: Using a custom rerank function with all settings included
async def my_rerank_func(query: str, documents: list, top_n: int = None, **kwargs):
    return await custom_rerank(
        query=query,
        documents=documents,
        model="BAAI/bge-reranker-v2-m3",
        base_url="https://api.your-provider.com/v1/rerank",
        api_key="your_api_key_here",
        top_n=top_n or 10,  # Handle top_n within the function
        **kwargs
    )

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
    rerank_model_func=my_rerank_func,  # Configure rerank function
)

# Query with rerank enabled (default)
result = await rag.aquery(
    "your query",
    param=QueryParam(enable_rerank=True)  # Control rerank per query
)

# Query with rerank disabled
result = await rag.aquery(
    "your query",
    param=QueryParam(enable_rerank=False)
)

# Method 2: Using RerankModel wrapper
rerank_model = RerankModel(
    rerank_func=custom_rerank,
    kwargs={
        "model": "BAAI/bge-reranker-v2-m3",
        "base_url": "https://api.your-provider.com/v1/rerank",
        "api_key": "your_api_key_here",
    }
)

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
    rerank_model_func=rerank_model.rerank,
)

# Control rerank per query
result = await rag.aquery(
    "your query",
    param=QueryParam(
        enable_rerank=True,  # Enable rerank for this query
        chunk_top_k=5       # Number of chunks to keep after reranking
    )
)
```

## Supported Providers

### 1. Custom/Generic API (Recommended)

For Jina/Cohere compatible APIs:

```python
from lightrag.rerank import custom_rerank

# Your custom API endpoint
result = await custom_rerank(
    query="your query",
    documents=documents,
    model="BAAI/bge-reranker-v2-m3",
    base_url="https://api.your-provider.com/v1/rerank",
    api_key="your_api_key_here",
    top_n=10
)
```

### 2. Jina AI

```python
from lightrag.rerank import jina_rerank

result = await jina_rerank(
    query="your query",
    documents=documents,
    model="BAAI/bge-reranker-v2-m3",
    api_key="your_jina_api_key",
    top_n=10
)
```

### 3. Cohere

```python
from lightrag.rerank import cohere_rerank

result = await cohere_rerank(
    query="your query",
    documents=documents,
    model="rerank-english-v2.0",
    api_key="your_cohere_api_key",
    top_n=10
)
```

## Integration Points

Reranking is automatically applied at these key retrieval stages:

1. **Naive Mode**: After vector similarity search in `_get_vector_context`
2. **Local Mode**: After entity retrieval in `_get_node_data`
3. **Global Mode**: After relationship retrieval in `_get_edge_data`
4. **Hybrid/Mix Modes**: Applied to all relevant components

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_rerank` | bool | False | Enable/disable reranking |
| `rerank_model_func` | callable | None | Custom rerank function containing all configurations (model, API keys, top_n, etc.) |

## Example Usage

### Basic Usage

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embedding
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.rerank import jina_rerank

async def my_rerank_func(query: str, documents: list, top_n: int = None, **kwargs):
    """Custom rerank function with all settings included"""
    return await jina_rerank(
        query=query,
        documents=documents,
        model="BAAI/bge-reranker-v2-m3",
        api_key="your_jina_api_key_here",
        top_n=top_n or 10,  # Default top_n if not provided
        **kwargs
    )

async def main():
    # Initialize with rerank enabled
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding,
        rerank_model_func=my_rerank_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Insert documents
    await rag.ainsert([
        "Document 1 content...",
        "Document 2 content...",
    ])

    # Query with rerank (automatically applied)
    result = await rag.aquery(
        "Your question here",
        param=QueryParam(enable_rerank=True)  # This top_n is passed to rerank function
    )

    print(result)

asyncio.run(main())
```

### Direct Rerank Usage

```python
from lightrag.rerank import custom_rerank

async def test_rerank():
    documents = [
        {"content": "Text about topic A"},
        {"content": "Text about topic B"},
        {"content": "Text about topic C"},
    ]

    reranked = await custom_rerank(
        query="Tell me about topic A",
        documents=documents,
        model="BAAI/bge-reranker-v2-m3",
        base_url="https://api.your-provider.com/v1/rerank",
        api_key="your_api_key_here",
        top_n=2
    )

    for doc in reranked:
        print(f"Score: {doc.get('rerank_score')}, Content: {doc.get('content')}")
```

## Best Practices

1. **Self-Contained Functions**: Include all necessary configurations (API keys, models, top_n handling) within your rerank function
2. **Performance**: Use reranking selectively for better performance vs. quality tradeoff
3. **API Limits**: Monitor API usage and implement rate limiting within your rerank function
4. **Fallback**: Always handle rerank failures gracefully (returns original results)
5. **Top-n Handling**: Handle top_n parameter appropriately within your rerank function
6. **Cost Management**: Consider rerank API costs in your budget planning

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure API keys are properly configured within your rerank function
2. **Network Issues**: Check API endpoints and network connectivity
3. **Model Errors**: Verify the rerank model name is supported by your API
4. **Document Format**: Ensure documents have `content` or `text` fields

### Debug Mode

Enable debug logging to see rerank operations:

```python
import logging
logging.getLogger("lightrag.rerank").setLevel(logging.DEBUG)
```

### Error Handling

The rerank integration includes automatic fallback:

```python
# If rerank fails, original documents are returned
# No exceptions are raised to the user
# Errors are logged for debugging
```

## API Compatibility

The generic rerank API expects this response format:

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95
    },
    {
      "index": 2,
      "relevance_score": 0.87
    }
  ]
}
```

This is compatible with:
- Jina AI Rerank API
- Cohere Rerank API
- Custom APIs following the same format
