# LightRAG LLM Integration

> Complete guide to configuring LLM providers and embedding models

**Version**: 1.4.9.2 | **Last Updated**: December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Providers](#supported-providers)
3. [OpenAI Integration](#openai-integration)
4. [Ollama Integration](#ollama-integration)
5. [Azure OpenAI](#azure-openai)
6. [AWS Bedrock](#aws-bedrock)
7. [Anthropic Claude](#anthropic-claude)
8. [Other Providers](#other-providers)
9. [Embedding Models](#embedding-models)
10. [Reranking](#reranking)
11. [Configuration Reference](#configuration-reference)

---

## Overview

LightRAG requires two core AI components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM Integration Architecture                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     LightRAG Core                                │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│              ┌───────────────┴───────────────┐                         │
│              │                               │                         │
│              ▼                               ▼                         │
│  ┌───────────────────────┐       ┌───────────────────────┐            │
│  │    LLM Function       │       │   Embedding Function  │            │
│  │    (Text Generation)  │       │   (Vector Creation)   │            │
│  └───────────┬───────────┘       └───────────┬───────────┘            │
│              │                               │                         │
│              ▼                               ▼                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Provider Bindings                           │   │
│  │                                                                  │   │
│  │  OpenAI │ Ollama │ Azure │ Bedrock │ Anthropic │ HuggingFace   │   │
│  │  Jina   │ lollms │ NVIDIA │ SiliconCloud │ ZhipuAI │ ...        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### LLM Function Requirements

The LLM function is used for:
- Entity/Relation extraction from text chunks
- Description summarization during merges
- Query keyword extraction
- Response generation from context

### Embedding Function Requirements

The embedding function is used for:
- Converting text chunks to vectors
- Converting entities/relations to vectors
- Query embedding for similarity search

---

## Supported Providers

| Provider | LLM | Embedding | Rerank | Module |
|----------|-----|-----------|--------|--------|
| **OpenAI** | ✅ | ✅ | ❌ | `lightrag.llm.openai` |
| **Ollama** | ✅ | ✅ | ❌ | `lightrag.llm.ollama` |
| **Azure OpenAI** | ✅ | ✅ | ❌ | `lightrag.llm.azure_openai` |
| **AWS Bedrock** | ✅ | ✅ | ❌ | `lightrag.llm.bedrock` |
| **Anthropic** | ✅ | ❌ | ❌ | `lightrag.llm.anthropic` |
| **Jina AI** | ❌ | ✅ | ✅ | `lightrag.llm.jina` |
| **HuggingFace** | ✅ | ✅ | ❌ | `lightrag.llm.hf` |
| **NVIDIA** | ✅ | ✅ | ❌ | `lightrag.llm.nvidia_openai` |
| **SiliconCloud** | ✅ | ✅ | ❌ | `lightrag.llm.siliconcloud` |
| **ZhipuAI** | ✅ | ✅ | ❌ | `lightrag.llm.zhipu` |
| **lollms** | ✅ | ✅ | ❌ | `lightrag.llm.lollms` |
| **LMDeploy** | ✅ | ❌ | ❌ | `lightrag.llm.lmdeploy` |

---

## OpenAI Integration

### Basic Setup

```python
import os
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Set API key
os.environ["OPENAI_API_KEY"] = "sk-..."

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=gpt_4o_mini_complete,
    llm_model_name="gpt-4o-mini",
    embedding_func=openai_embed,
)

await rag.initialize_storages()
```

### Available Models

```python
from lightrag.llm.openai import (
    gpt_4o_complete,          # GPT-4o (flagship)
    gpt_4o_mini_complete,     # GPT-4o-mini (cost-effective)
    openai_complete,          # Generic OpenAI completion
    openai_embed,             # text-embedding-3-small
)
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIM=1536
```

### Advanced Configuration

```python
from lightrag.llm.openai import create_openai_async_client, openai_complete_if_cache

# Custom client
client = create_openai_async_client(
    api_key="sk-...",
    base_url="https://your-proxy.com/v1",
    client_configs={"timeout": 60.0}
)

# Custom model configuration
rag = LightRAG(
    llm_model_func=lambda prompt, **kwargs: openai_complete_if_cache(
        model="gpt-4-turbo",
        prompt=prompt,
        system_prompt="You are a knowledge extraction expert.",
        **kwargs
    ),
    llm_model_name="gpt-4-turbo",
    llm_model_kwargs={
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    embedding_func=openai_embed,
)
```

---

## Ollama Integration

### Prerequisites

Install and run Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Basic Setup

```python
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.2:3b",
    embedding_func=lambda texts: ollama_embed(
        texts,
        embed_model="nomic-embed-text"
    ),
)
```

### Environment Variables

```bash
# Ollama server
LLM_BINDING_HOST=http://localhost:11434
OLLAMA_HOST=http://localhost:11434

# Model configuration
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

### Server Mode Configuration

For LightRAG API server with Ollama:

```bash
# Start API server with Ollama
python -m lightrag.api.lightrag_server \
    --llm-binding ollama \
    --llm-model llama3.2:3b \
    --embedding-binding ollama \
    --embedding-model nomic-embed-text \
    --llm-binding-host http://localhost:11434 \
    --embedding-binding-host http://localhost:11434
```

### Ollama Models Recommended

| Purpose | Model | Size | Notes |
|---------|-------|------|-------|
| **LLM (Fast)** | `llama3.2:3b` | 2GB | Good balance |
| **LLM (Quality)** | `llama3.1:8b` | 4.7GB | Better extraction |
| **LLM (Best)** | `llama3.1:70b` | 40GB | Production quality |
| **Embedding** | `nomic-embed-text` | 274MB | General purpose |
| **Embedding** | `bge-m3` | 2.2GB | Multilingual |
| **Embedding** | `mxbai-embed-large` | 669MB | High quality |

---

## Azure OpenAI

### Setup

```python
from lightrag import LightRAG
from lightrag.llm.azure_openai import azure_openai_complete, azure_embed

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=azure_openai_complete,
    llm_model_name="gpt-4o-deployment",
    embedding_func=azure_embed,
)
```

### Environment Variables

```bash
# Required
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-01

# Deployment names
AZURE_OPENAI_DEPLOYMENT=gpt-4o-deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-deployment

# Optional
AZURE_OPENAI_EMBEDDING_DIM=1536
```

### Server Mode

```bash
python -m lightrag.api.lightrag_server \
    --llm-binding azure_openai \
    --llm-model gpt-4o \
    --embedding-binding azure_openai \
    --embedding-model text-embedding-3-small
```

---

## AWS Bedrock

### Setup

```python
from lightrag import LightRAG
from lightrag.llm.bedrock import bedrock_complete, bedrock_embed

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=bedrock_complete,
    llm_model_name="anthropic.claude-3-sonnet-20240229-v1:0",
    embedding_func=bedrock_embed,
)
```

### Environment Variables

```bash
# AWS credentials
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1

# Bedrock configuration
BEDROCK_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
```

### Supported Bedrock Models

| Provider | Model ID | Type |
|----------|----------|------|
| Anthropic | `anthropic.claude-3-sonnet-20240229-v1:0` | LLM |
| Anthropic | `anthropic.claude-3-haiku-20240307-v1:0` | LLM |
| Amazon | `amazon.titan-text-express-v1` | LLM |
| Amazon | `amazon.titan-embed-text-v2:0` | Embedding |
| Cohere | `cohere.embed-multilingual-v3` | Embedding |

---

## Anthropic Claude

### Setup

```python
from lightrag import LightRAG
from lightrag.llm.anthropic import anthropic_complete
from lightrag.llm.openai import openai_embed  # Use OpenAI for embedding

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=anthropic_complete,
    llm_model_name="claude-3-5-sonnet-20241022",
    embedding_func=openai_embed,  # Anthropic doesn't have embeddings
)
```

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...

# Model selection
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `claude-3-5-sonnet-20241022` | 200K | Best quality |
| `claude-3-haiku-20240307` | 200K | Fast & cheap |
| `claude-3-opus-20240229` | 200K | Complex tasks |

---

## Other Providers

### Jina AI (Embedding + Rerank)

```python
from lightrag.llm.jina import jina_embed, jina_rerank
from lightrag.llm.openai import gpt_4o_mini_complete

rag = LightRAG(
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=jina_embed,
    rerank_model_func=jina_rerank,
)
```

```bash
JINA_API_KEY=jina_...
JINA_EMBEDDING_MODEL=jina-embeddings-v3
JINA_RERANK_MODEL=jina-reranker-v2-base-multilingual
```

### HuggingFace

```python
from lightrag.llm.hf import hf_model_complete, hf_embed

rag = LightRAG(
    llm_model_func=hf_model_complete,
    llm_model_name="meta-llama/Llama-3.2-3B-Instruct",
    embedding_func=hf_embed,
)
```

```bash
HF_TOKEN=hf_...
HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
HF_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

### NVIDIA NIM

```python
from lightrag.llm.nvidia_openai import nvidia_complete, nvidia_embed

rag = LightRAG(
    llm_model_func=nvidia_complete,
    llm_model_name="meta/llama-3.1-70b-instruct",
    embedding_func=nvidia_embed,
)
```

```bash
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

### SiliconCloud

```python
from lightrag.llm.siliconcloud import siliconcloud_complete, siliconcloud_embed

rag = LightRAG(
    llm_model_func=siliconcloud_complete,
    llm_model_name="Qwen/Qwen2.5-72B-Instruct",
    embedding_func=siliconcloud_embed,
)
```

```bash
SILICONCLOUD_API_KEY=sk-...
```

### ZhipuAI (GLM)

```python
from lightrag.llm.zhipu import zhipu_complete, zhipu_embed

rag = LightRAG(
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-flash",
    embedding_func=zhipu_embed,
)
```

```bash
ZHIPUAI_API_KEY=...
```

---

## Embedding Models

### Embedding Function Signature

```python
from lightrag.utils import EmbeddingFunc
import numpy as np

async def custom_embedding(texts: list[str]) -> np.ndarray:
    """
    Args:
        texts: List of strings to embed

    Returns:
        np.ndarray: Shape (len(texts), embedding_dim)
    """
    # Your embedding logic
    return embeddings
```

### Wrapping Custom Embeddings

```python
from lightrag.utils import wrap_embedding_func_with_attrs

@wrap_embedding_func_with_attrs(
    embedding_dim=1024,
    max_token_size=8192,
)
async def my_embed(texts: list[str]) -> np.ndarray:
    # Implementation
    pass
```

### Embedding Dimension Reference

| Provider | Model | Dimension |
|----------|-------|-----------|
| OpenAI | text-embedding-3-small | 1536 |
| OpenAI | text-embedding-3-large | 3072 |
| OpenAI | text-embedding-ada-002 | 1536 |
| Ollama | nomic-embed-text | 768 |
| Ollama | bge-m3 | 1024 |
| Ollama | mxbai-embed-large | 1024 |
| Jina | jina-embeddings-v3 | 1024 |
| Cohere | embed-multilingual-v3 | 1024 |
| HuggingFace | BAAI/bge-large-en-v1.5 | 1024 |

### Batched Embedding

```python
rag = LightRAG(
    embedding_func=openai_embed,
    embedding_batch_num=20,           # Process 20 texts per batch
    embedding_func_max_async=8,       # Max concurrent batches
    default_embedding_timeout=30,     # Timeout per batch (seconds)
)
```

---

## Reranking

### Enabling Reranking

```python
from lightrag import LightRAG
from lightrag.llm.jina import jina_rerank

rag = LightRAG(
    # ... other config
    rerank_model_func=jina_rerank,
    min_rerank_score=0.3,  # Filter chunks below this score
)
```

### Reranking Providers

| Provider | Function | Model |
|----------|----------|-------|
| Jina AI | `jina_rerank` | jina-reranker-v2-base-multilingual |
| Cohere | Custom | rerank-english-v3.0 |

### Query-Time Reranking

```python
from lightrag.base import QueryParam

result = await rag.aquery(
    "What is the capital of France?",
    param=QueryParam(
        enable_rerank=True,     # Enable for this query
        chunk_top_k=50,         # Retrieve more, rerank to top_k
    )
)
```

---

## Configuration Reference

### LightRAG LLM Parameters

```python
rag = LightRAG(
    # LLM Configuration
    llm_model_func=gpt_4o_mini_complete,    # LLM function
    llm_model_name="gpt-4o-mini",           # Model name for logging
    llm_model_kwargs={                       # Passed to LLM function
        "temperature": 1.0,
        "max_tokens": 4096,
    },
    llm_model_max_async=4,                  # Concurrent LLM calls
    default_llm_timeout=180,                # Timeout (seconds)

    # Caching
    enable_llm_cache=True,                  # Cache LLM responses
    enable_llm_cache_for_entity_extract=True,  # Cache extraction
)
```

### Environment Variables Summary

```bash
# === OpenAI ===
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1

# === Ollama ===
LLM_BINDING_HOST=http://localhost:11434
EMBEDDING_BINDING_HOST=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text

# === Azure OpenAI ===
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# === AWS Bedrock ===
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
BEDROCK_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# === Anthropic ===
ANTHROPIC_API_KEY=sk-ant-...

# === Jina ===
JINA_API_KEY=jina_...

# === Processing ===
MAX_ASYNC=4
EMBEDDING_BATCH_NUM=10
LLM_TIMEOUT=180
EMBEDDING_TIMEOUT=30
```

### API Server Bindings

```bash
# LLM binding options
--llm-binding openai|ollama|azure_openai|aws_bedrock|lollms

# Embedding binding options
--embedding-binding openai|ollama|azure_openai|aws_bedrock|jina|lollms

# Rerank binding options
--rerank-binding null|jina
```

---

## Best Practices

### Production Recommendations

1. **Use OpenAI** for best extraction quality
2. **Use Ollama** for cost-effective local deployment
3. **Enable LLM caching** to reduce costs
4. **Set appropriate timeouts** for reliability
5. **Monitor token usage** for cost control

### Cost Optimization

```python
rag = LightRAG(
    # Use cost-effective model
    llm_model_func=gpt_4o_mini_complete,

    # Enable aggressive caching
    enable_llm_cache=True,
    enable_llm_cache_for_entity_extract=True,

    # Limit parallel processing
    llm_model_max_async=2,

    # Use smaller chunks
    chunk_token_size=800,
)
```

### Quality Optimization

```python
rag = LightRAG(
    # Use best model
    llm_model_func=gpt_4o_complete,
    llm_model_name="gpt-4o",

    # More extraction attempts
    entity_extract_max_gleaning=2,

    # Larger chunks for context
    chunk_token_size=1500,
    chunk_overlap_token_size=200,

    # Enable reranking
    rerank_model_func=jina_rerank,
)
```

---

**Version**: 1.4.9.2 | **License**: MIT
