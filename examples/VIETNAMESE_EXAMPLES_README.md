# Vietnamese Embedding Examples

This directory contains example scripts demonstrating how to use the AITeamVN/Vietnamese_Embedding model with LightRAG.

## Available Examples

### 1. Simple Example: `lightrag_vietnamese_embedding_simple.py`

**Purpose:** Minimal code to get started quickly

**What it does:**
- Initializes LightRAG with Vietnamese embedding
- Inserts a simple Vietnamese text
- Performs a basic query
- Clean and easy to understand

**Run:**
```bash
export HUGGINGFACE_API_KEY="your_hf_token"
export OPENAI_API_KEY="your_openai_key"
python examples/lightrag_vietnamese_embedding_simple.py
```

**Expected output:**
```
Inserting Vietnamese text...
Query: Thủ đô của Việt Nam là gì?
Answer: [Response about Hanoi being the capital]
```

**Code size:** ~50 lines  
**Execution time:** ~30 seconds (first run with model download: ~2 minutes)

---

### 2. Comprehensive Demo: `vietnamese_embedding_demo.py`

**Purpose:** Full-featured demonstration with multiple scenarios

**What it does:**
- **Demo 1:** Vietnamese text processing
  - Inserts Vietnamese content about Vietnam
  - Performs multiple queries in Vietnamese
  - Demonstrates hybrid mode retrieval

- **Demo 2:** English text processing (multilingual support)
  - Inserts English content about AI
  - Queries in English
  - Shows model's multilingual capabilities

- **Demo 3:** Mixed language processing
  - Inserts mixed Vietnamese-English content
  - Queries in both languages
  - Demonstrates language-agnostic retrieval

**Run:**
```bash
export HUGGINGFACE_API_KEY="your_hf_token"
export OPENAI_API_KEY="your_openai_key"
python examples/vietnamese_embedding_demo.py
```

**Expected output:**
```
=============================================================
DEMO 1: Vietnamese Text Processing
=============================================================
✓ Initializing LightRAG with Vietnamese Embedding Model
✓ Text inserted successfully!

Querying in Vietnamese:
------------------------------------------------------------
❓ Query: Thủ đô của Việt Nam là gì?
💡 Answer: [Detailed response about Hanoi]
...

[Similar output for Demo 2 and Demo 3]

=============================================================
✓ All demos completed successfully!
=============================================================
```

**Code size:** ~300 lines  
**Execution time:** ~2-5 minutes (depending on LLM speed)

---

## Prerequisites

### Required Environment Variables

```bash
# HuggingFace token (required for model access)
export HUGGINGFACE_API_KEY="hf_your_token_here"
# or
export HF_TOKEN="hf_your_token_here"

# LLM API key (using OpenAI as example)
export OPENAI_API_KEY="sk-your_key_here"
```

### Get Your Tokens

1. **HuggingFace Token:**
   - Visit: https://huggingface.co/settings/tokens
   - Create a new token with "Read" permission
   - Copy and export it

2. **OpenAI API Key:**
   - Visit: https://platform.openai.com/api-keys
   - Create a new key
   - Copy and export it

### Alternative: Use `.env` File

Create a `.env` file in the project root:

```env
HUGGINGFACE_API_KEY=hf_your_token_here
OPENAI_API_KEY=sk-your_key_here
```

---

## What to Expect on First Run

### Model Download (First Time Only)
- Size: ~2 GB
- Time: 2-5 minutes (depending on internet speed)
- Location: Cached in `~/.cache/huggingface/`

After the first run, the model is cached and loads instantly.

### Resource Usage
- **GPU Memory:** 2-4 GB (if using GPU)
- **RAM:** 4-8 GB
- **Disk Space:** 2 GB for model + storage for RAG data

---

## Common Issues & Solutions

### Issue: "No HuggingFace token found"
**Solution:**
```bash
export HUGGINGFACE_API_KEY="your_token"
```

### Issue: "Model download fails"
**Possible causes:**
1. No internet connection
2. Invalid HuggingFace token
3. Insufficient disk space

**Solution:**
- Check internet connection
- Verify token is correct
- Ensure 2GB+ free space

### Issue: "Out of memory error"
**Solution:**
- Close other applications
- Use CPU instead of GPU (slower but less memory)
- Reduce batch size (if processing many texts)

### Issue: "Slow performance"
**Solution:**
- Install CUDA-enabled PyTorch for GPU support
- Check GPU is being used (enable DEBUG logging)
- Use GPU instead of CPU (10-50x faster)

---

## Tips for Best Results

### 1. Enable Debug Logging
See what's happening under the hood:
```python
from lightrag.utils import setup_logger
setup_logger("lightrag", level="DEBUG")
```

### 2. Use GPU for Production
Much faster than CPU:
```bash
# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. Optimize for Your Use Case
Adjust parameters based on your text length:
```python
embedding_func=EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=1024,  # Reduce if your texts are shorter
    func=vietnamese_embed
)
```

### 4. Batch Processing
Process multiple texts together for efficiency:
```python
texts = ["Text 1", "Text 2", ..., "Text N"]
await rag.ainsert(texts)  # More efficient than one by one
```

---

## Understanding the Examples

### Key Components

1. **Embedding Function:**
```python
from lightrag.llm.vietnamese_embed import vietnamese_embed
```
This loads the Vietnamese_Embedding model.

2. **LightRAG Configuration:**
```python
embedding_func=EmbeddingFunc(
    embedding_dim=1024,      # Vietnamese_Embedding outputs 1024 dimensions
    max_token_size=2048,     # Model supports up to 2048 tokens
    func=vietnamese_embed    # The embedding function
)
```

3. **Text Insertion:**
```python
await rag.ainsert(text)  # Asynchronous insertion
# or
rag.insert(text)         # Synchronous insertion
```

4. **Querying:**
```python
result = await rag.aquery(
    query,
    param=QueryParam(mode="hybrid")  # hybrid, local, global, naive, mix
)
```

### Query Modes

- **naive:** Simple vector similarity search
- **local:** Context-dependent retrieval
- **global:** Global knowledge retrieval
- **hybrid:** Combines local and global
- **mix:** Integrates knowledge graph and vector retrieval

For Vietnamese text, **hybrid** mode typically works best.

---

## Modifying the Examples

### Use Different LLM

Replace `gpt_4o_mini_complete` with your preferred LLM:

```python
# Using Ollama
from lightrag.llm.ollama import ollama_model_complete
llm_model_func=ollama_model_complete

# Using Azure OpenAI
from lightrag.llm.azure_openai import azure_openai_complete
llm_model_func=azure_openai_complete
```

### Use Different Embedding Model

While keeping the Vietnamese embedding:

```python
from lightrag.llm.vietnamese_embed import vietnamese_embed

# Use different model from HuggingFace
func=lambda texts: vietnamese_embed(
    texts,
    model_name="BAAI/bge-m3",  # Use base model
    token=hf_token
)
```

### Add Your Own Data

Replace the sample text with your own:

```python
# Read from file
with open("your_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

await rag.ainsert(text)
```

---

## Next Steps

1. **Try the simple example first** to verify setup
2. **Run the comprehensive demo** to see all features
3. **Modify examples** for your specific use case
4. **Read the documentation** for advanced usage:
   - English: `docs/VietnameseEmbedding.md`
   - Vietnamese: `docs/VietnameseEmbedding_VI.md`
5. **Run the test suite** to validate your environment:
   ```bash
   python tests/test_vietnamese_embedding_integration.py
   ```

---

## Support

- **Documentation:** See `docs/VietnameseEmbedding.md`
- **Issues:** https://github.com/HKUDS/LightRAG/issues
- **Model:** https://huggingface.co/AITeamVN/Vietnamese_Embedding

---

## Related Examples in This Directory

While you're here, check out these other LightRAG examples:

- `lightrag_openai_demo.py` - Basic OpenAI integration
- `lightrag_ollama_demo.py` - Using Ollama models
- `lightrag_hf_demo.py` - HuggingFace models
- `rerank_example.py` - Adding reranking
- `graph_visual_with_neo4j.py` - Neo4J visualization

---

**Happy coding!** 🚀

For questions or feedback, please open an issue on GitHub with the `vietnamese-embedding` tag.
