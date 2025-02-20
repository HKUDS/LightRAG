
1. **LlamaIndex** (`llm/llama_index.py`):
   - Provides integration with OpenAI and other providers through LlamaIndex
   - Supports both direct API access and proxy services like LiteLLM
   - Handles embeddings and completions with consistent interfaces
   - See example implementations:
     - [Direct OpenAI Usage](../../examples/lightrag_llamaindex_direct_demo.py)
     - [LiteLLM Proxy Usage](../../examples/lightrag_llamaindex_litellm_demo.py)

<details>
<summary> <b>Using LlamaIndex</b> </summary>

LightRAG supports LlamaIndex for embeddings and completions in two ways: direct OpenAI usage or through LiteLLM proxy.

### Setup

First, install the required dependencies:
```bash
pip install llama-index-llms-litellm llama-index-embeddings-litellm
```

### Standard OpenAI Usage

```python
from lightrag import LightRAG
from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from lightrag.utils import EmbeddingFunc

# Initialize with direct OpenAI access
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    try:
        # Initialize OpenAI if not in kwargs
        if 'llm_instance' not in kwargs:
            llm_instance = OpenAI(
                model="gpt-4",
                api_key="your-openai-key",
                temperature=0.7,
            )
            kwargs['llm_instance'] = llm_instance

        response = await llama_index_complete_if_cache(
            kwargs['llm_instance'],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
        return response
    except Exception as e:
        logger.error(f"LLM request failed: {str(e)}")
        raise

# Initialize LightRAG with OpenAI
rag = LightRAG(
    working_dir="your/path",
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=lambda texts: llama_index_embed(
            texts,
            embed_model=OpenAIEmbedding(
                model="text-embedding-3-large",
                api_key="your-openai-key"
            )
        ),
    ),
)
```

### Using LiteLLM Proxy

1. Use any LLM provider through LiteLLM
2. Leverage LlamaIndex's embedding and completion capabilities
3. Maintain consistent configuration across services

```python
from lightrag import LightRAG
from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding
from lightrag.utils import EmbeddingFunc

# Initialize with LiteLLM proxy
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    try:
        # Initialize LiteLLM if not in kwargs
        if 'llm_instance' not in kwargs:
            llm_instance = LiteLLM(
                model=f"openai/{settings.LLM_MODEL}",  # Format: "provider/model_name"
                api_base=settings.LITELLM_URL,
                api_key=settings.LITELLM_KEY,
                temperature=0.7,
            )
            kwargs['llm_instance'] = llm_instance

        response = await llama_index_complete_if_cache(
            kwargs['llm_instance'],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
        return response
    except Exception as e:
        logger.error(f"LLM request failed: {str(e)}")
        raise

# Initialize LightRAG with LiteLLM
rag = LightRAG(
    working_dir="your/path",
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=lambda texts: llama_index_embed(
            texts,
            embed_model=LiteLLMEmbedding(
                model_name=f"openai/{settings.EMBEDDING_MODEL}",
                api_base=settings.LITELLM_URL,
                api_key=settings.LITELLM_KEY,
            )
        ),
    ),
)
```

### Environment Variables

For OpenAI direct usage:
```bash
OPENAI_API_KEY=your-openai-key
```

For LiteLLM proxy:
```bash
# LiteLLM Configuration
LITELLM_URL=http://litellm:4000
LITELLM_KEY=your-litellm-key

# Model Configuration
LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_MAX_TOKEN_SIZE=8192
```

### Key Differences
1. **Direct OpenAI**:
   - Simpler setup
   - Direct API access
   - Requires OpenAI API key

2. **LiteLLM Proxy**:
   - Model provider agnostic
   - Centralized API key management
   - Support for multiple providers
   - Better cost control and monitoring

</details>
