# LightRAG Quick Start Guide

Get up and running with LightRAG in 5 minutes.

## What is LightRAG?

LightRAG is a **Graph-Enhanced Retrieval-Augmented Generation (RAG)** system that combines knowledge graphs with vector search for superior context retrieval.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Document   │───▶│   LightRAG   │───▶│   Knowledge  │
│    Input     │    │   Indexing   │    │    Graph     │
└──────────────┘    └──────────────┘    └──────────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐    ┌──────────────┐
                    │   Vector     │    │   Entity     │
                    │   Chunks     │    │   Relations  │
                    └──────────────┘    └──────────────┘
                           │                    │
                           └────────┬───────────┘
                                    ▼
                           ┌──────────────┐
                           │   Hybrid     │
                           │   Query      │
                           └──────────────┘
```

---

## Installation

### Option 1: pip (Recommended)

```bash
# Basic installation
pip install lightrag-hku

# With API server
pip install "lightrag-hku[api]"

# With storage backends
pip install "lightrag-hku[postgres]"   # PostgreSQL
pip install "lightrag-hku[neo4j]"      # Neo4j
pip install "lightrag-hku[milvus]"     # Milvus
```

### Option 2: From Source

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e ".[api]"
```

### Option 3: Docker

```bash
docker pull ghcr.io/hkuds/lightrag:latest
docker run -p 9621:9621 -e OPENAI_API_KEY=sk-xxx ghcr.io/hkuds/lightrag:latest
```

---

## Quick Start (Python)

### 1. Basic Usage

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    # Initialize LightRAG with OpenAI
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )
    
    # Initialize storage backends
    await rag.initialize_storages()

    # Insert documents
    await rag.ainsert("Marie Curie was a physicist who discovered radium. "
                      "She was born in Poland and later moved to France. "
                      "She won the Nobel Prize in Physics in 1903.")

    # Query with different modes
    result = await rag.aquery(
        "What did Marie Curie discover?",
        param=QueryParam(mode="hybrid")
    )
    print(result)

asyncio.run(main())
```

### 2. Insert Documents from Files

```python
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def insert_files():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )
    await rag.initialize_storages()

    # Insert from file
    with open("document.txt", "r") as f:
        text = f.read()
    await rag.ainsert(text)

    # Insert from multiple files
    documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
    for doc in documents:
        with open(doc, "r") as f:
            await rag.ainsert(f.read())

asyncio.run(insert_files())
```

### 3. Query Modes Explained

```python
from lightrag import QueryParam

# NAIVE: Traditional vector search (fastest)
result = await rag.aquery("question", param=QueryParam(mode="naive"))

# LOCAL: Focuses on specific entities mentioned in query
result = await rag.aquery("question", param=QueryParam(mode="local"))

# GLOBAL: Uses high-level summaries for broad questions
result = await rag.aquery("question", param=QueryParam(mode="global"))

# HYBRID: Combines local + global (recommended)
result = await rag.aquery("question", param=QueryParam(mode="hybrid"))

# MIX: Uses all modes and combines results
result = await rag.aquery("question", param=QueryParam(mode="mix"))
```

---

## Quick Start (REST API)

### 1. Start the Server

```bash
# Set your API key
export OPENAI_API_KEY=sk-xxx

# Start server
python -m lightrag.api.lightrag_server

# Server runs at http://localhost:9621
```

### 2. Insert Documents

```bash
# Insert text
curl -X POST http://localhost:9621/documents/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Albert Einstein developed the theory of relativity."}'

# Upload file
curl -X POST http://localhost:9621/documents/file \
  -F "file=@document.pdf"
```

### 3. Query

```bash
# Query with hybrid mode
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the theory of relativity?",
    "mode": "hybrid"
  }'
```

### 4. View Knowledge Graph

```bash
# Get all entities
curl http://localhost:9621/graphs/entities

# Get entity relationships
curl http://localhost:9621/graphs/relations
```

---

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-xxx              # OpenAI API key

# LLM Configuration
LLM_MODEL=gpt-4o-mini              # Model name
LLM_BINDING=openai                 # Provider: openai, ollama, anthropic

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIM=1536

# Server
PORT=9621                          # API server port
HOST=0.0.0.0                       # Bind address
```

### Using Different LLM Providers

```python
from lightrag import LightRAG

# OpenAI (default)
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed
)

# Ollama (local)
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3",
    llm_model_kwargs={"host": "http://localhost:11434"},
    embedding_func=ollama_embed
)

# Anthropic Claude
from lightrag.llm.anthropic import anthropic_complete

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=anthropic_complete,
    llm_model_name="claude-sonnet-4-20250514",
    embedding_func=openai_embed  # Use OpenAI for embeddings
)
```

### Storage Backends

```python
# Default (file-based, great for development)
rag = LightRAG(working_dir="./rag_storage")

# PostgreSQL (production)
rag = LightRAG(
    working_dir="./rag_storage",
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="PGGraphStorage"
)

# Neo4j (advanced graph queries)
rag = LightRAG(
    working_dir="./rag_storage",
    graph_storage="Neo4JStorage"
)
```

---

## Common Patterns

### Pattern 1: Document Processing Pipeline

```python
import asyncio
from pathlib import Path
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def process_documents(folder_path: str):
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )
    await rag.initialize_storages()

    # Process all text files
    for file_path in Path(folder_path).glob("*.txt"):
        print(f"Processing: {file_path}")
        with open(file_path, "r") as f:
            await rag.ainsert(f.read())

    print("All documents indexed!")

asyncio.run(process_documents("./documents"))
```

### Pattern 2: Conversational RAG

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def chat():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )
    await rag.initialize_storages()

    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break

        response = await rag.aquery(
            question,
            param=QueryParam(mode="hybrid")
        )
        print(f"Assistant: {response}")

asyncio.run(chat())
```

### Pattern 3: Batch Queries

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def batch_query(questions: list):
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )
    await rag.initialize_storages()

    results = []
    for question in questions:
        result = await rag.aquery(
            question,
            param=QueryParam(mode="hybrid")
        )
        results.append({"question": question, "answer": result})

    return results

questions = [
    "What is quantum computing?",
    "Who invented the telephone?",
    "What is machine learning?"
]

answers = asyncio.run(batch_query(questions))
```

---

## Verify Installation

```python
# Test script - requires OPENAI_API_KEY environment variable
import asyncio
import shutil
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def test():
    # Initialize
    rag = LightRAG(
        working_dir="./test_rag",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )
    await rag.initialize_storages()

    # Insert test document
    await rag.ainsert("Python is a programming language created by Guido van Rossum.")

    # Query
    result = await rag.aquery("Who created Python?", param=QueryParam(mode="hybrid"))
    print(f"Answer: {result}")

    # Cleanup
    shutil.rmtree("./test_rag")

asyncio.run(test())
```

Expected output:
```
Answer: Python was created by Guido van Rossum.
```

---

## Next Steps

1. **[Architecture Overview](0002-architecture-overview.md)** - Understand how LightRAG works
2. **[API Reference](0003-api-reference.md)** - Complete REST API documentation
3. **[Storage Backends](0004-storage-backends.md)** - Configure production storage
4. **[LLM Integration](0005-llm-integration.md)** - Use different LLM providers
5. **[Deployment Guide](0006-deployment-guide.md)** - Deploy to production

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: lightrag` | Run `pip install lightrag-hku` |
| `OPENAI_API_KEY not set` | Export your API key: `export OPENAI_API_KEY=sk-xxx` |
| `Connection refused` on port 9621 | Check if server is running: `python -m lightrag.api.lightrag_server` |
| Slow indexing | Use batch inserts or consider PostgreSQL for large datasets |
| Memory errors | Reduce chunk size or use streaming mode |

---

## Example Projects

```bash
# Clone and explore examples
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG/examples

# Run OpenAI demo
python lightrag_openai_demo.py

# Run Ollama demo (local LLM)
python lightrag_ollama_demo.py

# Visualize knowledge graph
python graph_visual_with_html.py
```

---

**Need Help?**
- GitHub Issues: https://github.com/HKUDS/LightRAG/issues
- Documentation: https://github.com/HKUDS/LightRAG/wiki
