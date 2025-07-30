# xAI Integration Troubleshooting Guide

This guide helps resolve common issues when using xAI Grok models with LightRAG.

## Common Error 1: Embedding Dimension Mismatch

### The Problem
You see this error:
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly,
but along dimension 1, the array at index 0 has size 1024 and the array at index 1 has size 768
```

### Why This Happens
This occurs when:
1. You switch between different embedding models with different vector dimensions
2. Your existing vector database has embeddings with one dimension (e.g., 1024)
3. The new embedding model produces vectors with a different dimension (e.g., 768)

### Solutions

#### Solution 1: Clean Slate (Recommended)
Remove the existing working directory to start fresh:

```bash
# Option A: Use the provided fix script
python examples/fix_embedding_dimension_issue.py

# Option B: Manual cleanup
rm -rf ./dickens_xai  # or your working directory
```

Then run your script again. The system will rebuild the knowledge graph with consistent dimensions.

#### Solution 2: Use Consistent Embedding Models
Ensure you use the same embedding model throughout:

```bash
# Set consistent embedding configuration
export EMBEDDING_MODEL="bge-m3:latest"
export EMBEDDING_DIM="1024"
export EMBEDDING_BINDING_HOST="http://localhost:11434"
```

#### Solution 3: Use xAI with OpenAI Embeddings
Since xAI doesn't have dedicated embedding models yet, use OpenAI embeddings:

```bash
export XAI_API_KEY="your-xai-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # For embeddings
```

## Common Error 2: Connection Timeout

### The Problem
You see this error:
```
httpcore.ConnectTimeout
httpx.ConnectTimeout
Error in ollama_embed
```

### Why This Happens
This occurs when:
1. Ollama server is overloaded with concurrent embedding requests
2. Network connectivity issues between LightRAG and Ollama
3. Ollama server is slow to respond due to resource constraints
4. High concurrency settings cause too many simultaneous requests

### Solutions

#### Solution 1: Use Timeout-Resistant Demo (Recommended)
```bash
python examples/lightrag_xai_demo_timeout_fix.py
```

This version includes:
- Increased timeout settings (2 minutes)
- Automatic retry logic with exponential backoff
- Reduced concurrency to prevent overload
- Better error messages

#### Solution 2: Restart Ollama Service
```bash
sudo systemctl restart ollama
# Wait a few seconds, then try again
```

#### Solution 3: Reduce Concurrency Settings
In your LightRAG configuration:
```python
rag = LightRAG(
    # ... other settings ...
    llm_model_max_async=2,  # Reduced from default 4
    # Reduce embedding batch size if possible
)
```

#### Solution 4: Check Ollama Status
```bash
# Check if Ollama is running
systemctl status ollama

# Check available models
ollama list

# Test Ollama directly
curl http://localhost:11434/api/tags
```

## xAI API Configuration

### Required Environment Variables
```bash
export XAI_API_KEY="your-xai-api-key"
```

### Optional Configuration
```bash
export XAI_API_BASE="https://api.x.ai/v1"  # Default
```

### Available Models
- `grok-3-mini`: Fast and efficient
- `grok-2-1212`: More capable reasoning
- `grok-2-vision-1212`: Supports vision (multimodal)

## API Server Configuration

Update your `.env` file:
```bash
# xAI LLM Configuration
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
LLM_BINDING_HOST=https://api.x.ai/v1
LLM_BINDING_API_KEY=your_xai_api_key

# Embedding Configuration (use consistent model)
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
EMBEDDING_BINDING_HOST=http://localhost:11434
```

## Troubleshooting Steps

### 1. Check API Key
```bash
echo $XAI_API_KEY
# Should show your API key
```

### 2. Test xAI API Connection
```python
import asyncio
from lightrag.llm.xai import grok_3_mini_complete

async def test_xai():
    response = await grok_3_mini_complete("Hello, how are you?")
    print(response)

asyncio.run(test_xai())
```

### 3. Check Embedding Model
```python
from lightrag.llm.ollama import ollama_embed
import asyncio

async def test_embed():
    result = await ollama_embed(
        ["test text"],
        embed_model="bge-m3:latest",
        host="http://localhost:11434"
    )
    print(f"Embedding shape: {result.shape}")

asyncio.run(test_embed())
```

### 4. Verify Working Directory
```bash
ls -la ./dickens_xai/
# Should show storage files if they exist
```

## Best Practices

### 1. Environment Setup
Create a dedicated environment for your xAI project:
```bash
python -m venv lightrag_xai_env
source lightrag_xai_env/bin/activate  # On Windows: lightrag_xai_env\Scripts\activate
pip install lightrag-hku
```

### 2. Consistent Configuration
Always use the same embedding model and dimensions:
```python
# In your scripts
EMBEDDING_MODEL = "bge-m3:latest"
EMBEDDING_DIM = 1024
```

### 3. Clean Working Directory
When switching embedding models, always clean the working directory:
```python
import shutil
import os

working_dir = "./my_rag_storage"
if os.path.exists(working_dir):
    shutil.rmtree(working_dir)
os.makedirs(working_dir, exist_ok=True)
```

### 4. Error Handling
Always handle potential errors gracefully:
```python
try:
    await rag.ainsert(content)
except ValueError as e:
    if "dimensions" in str(e):
        print("❌ Embedding dimension mismatch detected!")
        print("💡 Try cleaning your working directory and restart.")
    raise
```

## Getting Help

If you continue to have issues:

1. **Check the error message carefully** - dimension mismatches are usually clear
2. **Clean your working directory** - this fixes 90% of embedding issues
3. **Verify your API keys** - both xAI and any embedding service keys
4. **Use consistent configurations** - don't mix different embedding models
5. **Check the example scripts** - they show working configurations

## Status: All Issues Resolved ✅

**Update (2025-01-28)**: All major xAI integration issues have been resolved:
- ✅ Unicode decode error fixed
- ✅ Stream parameter conflict fixed
- ✅ Timeout issues addressed with retry logic
- ✅ Embedding dimension conflicts prevented
- ✅ Working demo scripts available

**Recommended approach**: Use `python examples/lightrag_xai_demo_timeout_fix.py`

## Example Working Configuration

Here's a minimal working setup:

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.xai import grok_3_mini_complete
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_embed

# Clean setup
working_dir = "./test_xai"
if os.path.exists(working_dir):
    import shutil
    shutil.rmtree(working_dir)

async def main():
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=grok_3_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="bge-m3:latest",
                host="http://localhost:11434"
            )
        )
    )

    await rag.initialize_storages()
    await rag.ainsert("Test document content")
    response = await rag.aquery("What is this about?")
    print(response)
    await rag.finalize_storages()

asyncio.run(main())
```

This configuration should work without dimension issues.
