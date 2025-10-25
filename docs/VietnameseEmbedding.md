# Vietnamese Embedding Integration for LightRAG

This integration adds support for the **AITeamVN/Vietnamese_Embedding** model to LightRAG, enabling enhanced retrieval capabilities for Vietnamese text.

## Model Information

- **Model**: [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)
- **Base Model**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Type**: Sentence Transformer
- **Maximum Sequence Length**: 2048 tokens
- **Output Dimensionality**: 1024 dimensions
- **Similarity Function**: Dot product similarity
- **Language**: Vietnamese (also supports other languages as it's based on BGE-M3)
- **Training Data**: ~300,000 triplets of queries, positive documents, and negative documents for Vietnamese

## Features

✅ **High-quality Vietnamese embeddings** - Fine-tuned specifically for Vietnamese text retrieval  
✅ **Multilingual support** - Inherits multilingual capabilities from BGE-M3  
✅ **Long context support** - Handles up to 2048 tokens per input  
✅ **Efficient processing** - Automatic device detection (CUDA/MPS/CPU)  
✅ **Normalized embeddings** - Ready for dot product similarity  
✅ **Easy integration** - Drop-in replacement for other embedding functions  

## Installation

### 1. Install LightRAG

```bash
cd LightRAG
pip install -e .
```

### 2. Install Required Dependencies

The Vietnamese embedding integration requires:
- `transformers` (automatically installed)
- `torch` (automatically installed)
- `numpy` (automatically installed)

These will be automatically installed via `pipmaster` when you first use the Vietnamese embedding function.

### 3. Set Up HuggingFace Token

You need a HuggingFace token to access the model:

```bash
export HUGGINGFACE_API_KEY="your_hf_token_here"
# or
export HF_TOKEN="your_hf_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

## Quick Start

### Simple Example

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.llm.vietnamese_embed import vietnamese_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./vietnamese_rag_storage"

async def main():
    # Get HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_API_KEY")
    
    # Initialize LightRAG with Vietnamese embedding
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=2048,
            func=lambda texts: vietnamese_embed(
                texts,
                model_name="AITeamVN/Vietnamese_Embedding",
                token=hf_token
            )
        ),
    )
    
    # Initialize storage and pipeline
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Insert Vietnamese text
    await rag.ainsert("Việt Nam là một quốc gia nằm ở Đông Nam Á.")
    
    # Query
    result = await rag.aquery(
        "Việt Nam ở đâu?",
        param=QueryParam(mode="hybrid")
    )
    print(result)
    
    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using with `.env` File

Create a `.env` file in your project directory:

```env
# HuggingFace Token for Vietnamese Embedding
HUGGINGFACE_API_KEY=your_key_here

# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini

# Embedding Configuration
EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding
EMBEDDING_DIM=1024
```

## Example Scripts

We provide several example scripts demonstrating different use cases:

### 1. Simple Example
```bash
python examples/lightrag_vietnamese_embedding_simple.py
```

A minimal example showing basic Vietnamese text processing.

### 2. Comprehensive Demo
```bash
python examples/vietnamese_embedding_demo.py
```

A comprehensive demo including:
- Vietnamese text processing
- English text processing (multilingual support)
- Mixed language processing
- Multiple query examples

## API Reference

### `vietnamese_embed()`

Generate embeddings for texts using the Vietnamese Embedding model.

```python
async def vietnamese_embed(
    texts: list[str],
    model_name: str = "AITeamVN/Vietnamese_Embedding",
    token: str | None = None,
) -> np.ndarray
```

**Parameters:**
- `texts` (list[str]): List of texts to embed
- `model_name` (str): HuggingFace model identifier
- `token` (str, optional): HuggingFace API token (reads from env if not provided)

**Returns:**
- `np.ndarray`: Array of embeddings with shape (len(texts), 1024)

**Example:**
```python
from lightrag.llm.vietnamese_embed import vietnamese_embed

texts = ["Xin chào", "Hello", "你好"]
embeddings = await vietnamese_embed(texts)
print(embeddings.shape)  # (3, 1024)
```

### `vietnamese_embedding_func()`

Convenience wrapper that automatically reads token from environment.

```python
async def vietnamese_embedding_func(texts: list[str]) -> np.ndarray
```

**Example:**
```python
from lightrag.llm.vietnamese_embed import vietnamese_embedding_func

# Token automatically read from HUGGINGFACE_API_KEY or HF_TOKEN
embeddings = await vietnamese_embedding_func(["Xin chào"])
```

## Advanced Usage

### Custom Model Configuration

```python
from lightrag.llm.vietnamese_embed import vietnamese_embed

# Use a different model based on BGE-M3
embeddings = await vietnamese_embed(
    texts=["Sample text"],
    model_name="BAAI/bge-m3",  # Use base model
    token=your_token
)
```

### Device Selection

The model automatically detects and uses the best available device:
1. CUDA (if available)
2. MPS (for Apple Silicon)
3. CPU (fallback)

You can check which device is being used by enabling debug logging:

```python
from lightrag.utils import setup_logger

setup_logger("lightrag", level="DEBUG")
```

### Batch Processing

The embedding function supports efficient batch processing:

```python
# Process multiple texts efficiently
large_batch = ["Text 1", "Text 2", ..., "Text 1000"]
embeddings = await vietnamese_embed(large_batch)
```

## Integration with LightRAG Server

To use Vietnamese embedding with LightRAG Server, update your `.env` file:

```env
# Vietnamese Embedding Configuration
EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding
EMBEDDING_DIM=1024
HUGGINGFACE_API_KEY=your_hf_token

# Or use custom binding
EMBEDDING_BINDING=huggingface
```

Then start the server:

```bash
lightrag-server
```

## Performance Considerations

### Memory Requirements

- **GPU Memory**: ~2-4 GB for the model
- **RAM**: ~4-8 GB recommended
- **Disk Space**: ~2 GB for model weights (cached after first download)

### Speed

On a typical GPU:
- ~1000 texts/second for short texts (< 512 tokens)
- ~200-400 texts/second for longer texts (1024-2048 tokens)

### Optimization Tips

1. **Use GPU**: Significantly faster than CPU (10-50x)
2. **Batch Requests**: Process multiple texts together
3. **Cache Model**: First run downloads model; subsequent runs are faster
4. **Adjust max_length**: Use shorter max_length if your texts are shorter

```python
# Example: Optimize for shorter texts
embedding_func=EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=512,  # Reduce if texts are shorter
    func=lambda texts: vietnamese_embed(texts)
)
```

## Troubleshooting

### Issue: "No HuggingFace token found"

**Solution:** Set the environment variable:
```bash
export HUGGINGFACE_API_KEY="your_token"
# or
export HF_TOKEN="your_token"
```

### Issue: "Model download fails"

**Solution:** 
1. Check your internet connection
2. Verify your HuggingFace token is valid
3. Ensure you have enough disk space (~2 GB)

### Issue: "Out of memory error"

**Solution:**
1. Reduce batch size
2. Use CPU instead of GPU (slower but uses less memory)
3. Close other applications using GPU/RAM

### Issue: "Slow embedding generation"

**Solution:**
1. Ensure you're using GPU (check logs for device info)
2. Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. Reduce max_token_size if your texts are shorter

## Comparison with Other Embedding Models

| Model | Dimensions | Max Tokens | Languages | Fine-tuned for Vietnamese |
|-------|------------|------------|-----------|--------------------------|
| Vietnamese_Embedding | 1024 | 2048 | Multilingual | ✅ Yes |
| BGE-M3 | 1024 | 8192 | Multilingual | ❌ No |
| text-embedding-3-large | 3072 | 8191 | Multilingual | ❌ No |
| text-embedding-3-small | 1536 | 8191 | Multilingual | ❌ No |

## Citation

If you use the Vietnamese Embedding model in your research, please cite:

```bibtex
@misc{vietnamese_embedding_2024,
  title={Vietnamese Embedding: Fine-tuned BGE-M3 for Vietnamese Retrieval},
  author={AITeamVN},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/AITeamVN/Vietnamese_Embedding}
}
```

## Support

For issues specific to the Vietnamese embedding integration:
- Open an issue on [LightRAG GitHub](https://github.com/HKUDS/LightRAG/issues)
- Tag with `vietnamese-embedding` label

For issues with the model itself:
- Visit [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)

## License

This integration follows LightRAG's license. The Vietnamese_Embedding model may have its own license terms - please check the [model page](https://huggingface.co/AITeamVN/Vietnamese_Embedding) for details.

## Acknowledgments

- **AITeamVN** for training and releasing the Vietnamese_Embedding model
- **BAAI** for the base BGE-M3 model
- **LightRAG team** for the excellent RAG framework
