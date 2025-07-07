# Rerank Integration in LightRAG

This document explains how to configure and use the rerank functionality in LightRAG to improve retrieval quality.

## ⚠️ Important: Parameter Priority

**QueryParam.top_k has higher priority than rerank_top_k configuration:**

- When you set `QueryParam(top_k=5)`, it will override the `rerank_top_k=10` setting in LightRAG configuration
- This means the actual number of documents sent to rerank will be determined by QueryParam.top_k
- For optimal rerank performance, always consider the top_k value in your QueryParam calls
- Example: `rag.aquery(query, param=QueryParam(mode="naive", top_k=20))` will use 20, not rerank_top_k

## Overview

Reranking is an optional feature that improves the quality of retrieved documents by re-ordering them based on their relevance to the query. This is particularly useful when you want higher precision in document retrieval across all query modes (naive, local, global, hybrid, mix).

## Architecture

The rerank integration follows the same design pattern as the LLM integration:

- **Configurable Models**: Support for multiple rerank providers through a generic API
- **Async Processing**: Non-blocking rerank operations
- **Error Handling**: Graceful fallback to original results
- **Optional Feature**: Can be enabled/disabled via configuration
- **Code Reuse**: Single generic implementation for Jina/Cohere compatible APIs

## Configuration

### Environment Variables

Set these variables in your `.env` file or environment:

```bash
# Enable/disable reranking
ENABLE_RERANK=True

# Rerank model configuration
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_MAX_ASYNC=4
RERANK_TOP_K=10

# API configuration
RERANK_API_KEY=your_rerank_api_key_here
RERANK_BASE_URL=https://api.your-provider.com/v1/rerank

# Provider-specific keys (optional alternatives)
JINA_API_KEY=your_jina_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

### Programmatic Configuration

```python
from lightrag import LightRAG
from lightrag.rerank import custom_rerank, RerankModel

# Method 1: Using environment variables (recommended)
rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
    # Rerank automatically configured from environment variables
)

# Method 2: Explicit configuration
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
    enable_rerank=True,
    rerank_model_func=rerank_model.rerank,
    rerank_top_k=10,
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
    top_k=10
)
```

### 2. Jina AI

```python
from lightrag.rerank import jina_rerank

result = await jina_rerank(
    query="your query",
    documents=documents,
    model="BAAI/bge-reranker-v2-m3",
    api_key="your_jina_api_key"
)
```

### 3. Cohere

```python
from lightrag.rerank import cohere_rerank

result = await cohere_rerank(
    query="your query",
    documents=documents,
    model="rerank-english-v2.0",
    api_key="your_cohere_api_key"
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
| `rerank_model_name` | str | "BAAI/bge-reranker-v2-m3" | Model identifier |
| `rerank_model_max_async` | int | 4 | Max concurrent rerank calls |
| `rerank_top_k` | int | 10 | Number of top results to return ⚠️ **Overridden by QueryParam.top_k** |
| `rerank_model_func` | callable | None | Custom rerank function |
| `rerank_model_kwargs` | dict | {} | Additional rerank parameters |

## Example Usage

### Basic Usage

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embedding

async def main():
    # Initialize with rerank enabled
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding,
        enable_rerank=True,
    )
    
    # Insert documents
    await rag.ainsert([
        "Document 1 content...",
        "Document 2 content...",
    ])
    
    # Query with rerank (automatically applied)
    result = await rag.aquery(
        "Your question here",
        param=QueryParam(mode="hybrid", top_k=5)  # ⚠️ This top_k=5 overrides rerank_top_k
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
        top_k=2
    )
    
    for doc in reranked:
        print(f"Score: {doc.get('rerank_score')}, Content: {doc.get('content')}")
```

## Best Practices

1. **Parameter Priority Awareness**: Remember that QueryParam.top_k always overrides rerank_top_k configuration
2. **Performance**: Use reranking selectively for better performance vs. quality tradeoff
3. **API Limits**: Monitor API usage and implement rate limiting if needed
4. **Fallback**: Always handle rerank failures gracefully (returns original results)
5. **Top-k Selection**: Choose appropriate `top_k` values in QueryParam based on your use case
6. **Cost Management**: Consider rerank API costs in your budget planning

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure `RERANK_API_KEY` or provider-specific keys are set
2. **Network Issues**: Check `RERANK_BASE_URL` and network connectivity
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