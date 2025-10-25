# Vietnamese Embedding - Quick Reference

## 🚀 Quick Setup (5 minutes)

### 1. Install & Configure
```bash
# Navigate to LightRAG directory
cd LightRAG

# Install in editable mode
pip install -e .

# Set your HuggingFace token
export HUGGINGFACE_API_KEY="your key here"

# Set your OpenAI key (or other LLM provider)
export OPENAI_API_KEY="your_openai_key"
```

### 2. Run Example
```bash
python examples/lightrag_vietnamese_embedding_simple.py
```

## 📝 Minimal Code Example

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.llm.vietnamese_embed import vietnamese_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

async def main():
    rag = LightRAG(
        working_dir="./vietnamese_rag",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=2048,
            func=vietnamese_embed
        )
    )
    
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Insert Vietnamese text
    await rag.ainsert("Việt Nam là một quốc gia ở Đông Nam Á.")
    
    # Query
    result = await rag.aquery("Việt Nam ở đâu?", param=QueryParam(mode="hybrid"))
    print(result)
    
    await rag.finalize_storages()

asyncio.run(main())
```

## 🔧 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | 1024 | Output dimensions |
| `max_token_size` | 2048 | Maximum tokens per input |
| `model_name` | AITeamVN/Vietnamese_Embedding | HuggingFace model ID |
| `token` | Your HF token | HuggingFace API token |

## 🌐 Supported Languages

✅ **Vietnamese** (optimized)  
✅ **English** (inherited from BGE-M3)  
✅ **Chinese** (inherited from BGE-M3)  
✅ **100+ other languages** (multilingual support)

## 📊 Performance

| Metric | Value |
|--------|-------|
| GPU Memory | 2-4 GB |
| RAM | 4-8 GB |
| Disk Space | ~2 GB (first download) |
| Speed (GPU) | 200-1000 texts/sec |
| Speed (CPU) | 20-100 texts/sec |

## 📚 Resources

### Documentation
- **English:** `docs/VietnameseEmbedding.md`
- **Tiếng Việt:** `docs/VietnameseEmbedding_VI.md`

### Examples
- **Simple:** `examples/lightrag_vietnamese_embedding_simple.py`
- **Comprehensive:** `examples/vietnamese_embedding_demo.py`

### Testing
```bash
python tests/test_vietnamese_embedding_integration.py
```

## 🐛 Common Issues

### "No HuggingFace token found"
```bash
export HUGGINGFACE_API_KEY="your_token"
```

### "Model download fails"
- Check internet connection
- Verify HuggingFace token is valid
- Ensure 2GB+ free disk space

### "Out of memory"
- Reduce batch size
- Use CPU instead of GPU
- Close other GPU applications

### "Slow embedding"
- Install CUDA-enabled PyTorch
- Check GPU is being used (see logs)
- Reduce `max_token_size` for shorter texts

## 💡 Tips

1. **First run is slow:** Model downloads (~2GB), cached afterward
2. **Use GPU:** 10-50x faster than CPU
3. **Batch requests:** Process multiple texts together
4. **Enable debug logs:** See device being used
   ```python
   from lightrag.utils import setup_logger
   setup_logger("lightrag", level="DEBUG")
   ```

## 🔗 Links

- **Model:** [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)
- **Base Model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **LightRAG:** [GitHub](https://github.com/HKUDS/LightRAG)

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/HKUDS/LightRAG/issues)
- **Tag:** Use `vietnamese-embedding` label
- **Model Issues:** [Vietnamese_Embedding page](https://huggingface.co/AITeamVN/Vietnamese_Embedding)

## ✅ Quick Validation

Run this to test your setup:
```bash
python -c "
import asyncio
from lightrag.llm.vietnamese_embed import vietnamese_embed
async def test():
    result = await vietnamese_embed(['Test text'])
    print(f'✓ Success! Shape: {result.shape}')
asyncio.run(test())
"
```

Expected output: `✓ Success! Shape: (1, 1024)`
