# Programming With LightRAG Core

> If you want to integrate LightRAG into your project, we recommend using the REST API provided by the LightRAG Server. LightRAG Core is intended for embedded applications or researchers conducting studies and evaluations.

## A Simple Program

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    return rag

async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()
        await rag.ainsert("Your text")

        # Perform hybrid search
        mode = "hybrid"
        print(
          await rag.aquery(
              "What are the top themes in this story?",
              param=QueryParam(mode=mode)
          )
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
```

Notes:
- Export your `OPENAI_API_KEY` environment variable before running.
- All data is persisted to `WORKING_DIR`.

**Important:**

**LightRAG requires explicit initialization before use.** You must call `await rag.initialize_storages()` after creating a LightRAG instance, otherwise you will encounter errors.


## LightRAG Init Parameters

**Parameters**

| **Parameter** | **Type** | **Explanation** | **Default** |
| -------------- | ---------- | ----------------- | ------------- |
| **working_dir** | `str` | Directory where the cache will be stored | `./rag_storage` |
| **workspace** | str | Workspace name for data isolation between different LightRAG Instances | |
| **kv_storage** | `str` | Storage type for documents and text chunks. Supported types: `JsonKVStorage`,`PGKVStorage`,`RedisKVStorage`,`MongoKVStorage`,`OpenSearchKVStorage` | `JsonKVStorage` |
| **vector_storage** | `str` | Storage type for embedding vectors. Supported types: `NanoVectorDBStorage`,`PGVectorStorage`,`MilvusVectorDBStorage`,`ChromaVectorDBStorage`,`FaissVectorDBStorage`,`MongoVectorDBStorage`,`QdrantVectorDBStorage`,`OpenSearchVectorDBStorage` | `NanoVectorDBStorage` |
| **graph_storage** | `str` | Storage type for graph edges and nodes. Supported types: `NetworkXStorage`,`Neo4JStorage`,`PGGraphStorage`,`PGTableGraphStorage`,`AGEStorage`,`OpenSearchGraphStorage` | `NetworkXStorage` |
| **doc_status_storage** | `str` | Storage type for documents process status. Supported types: `JsonDocStatusStorage`,`PGDocStatusStorage`,`MongoDocStatusStorage`,`OpenSearchDocStatusStorage` | `JsonDocStatusStorage` |
| **chunk_token_size** | `int` | Maximum token size per chunk when splitting documents | `1200` |
| **chunk_overlap_token_size** | `int` | Overlap token size between two chunks when splitting documents | `100` |
| **embedding_chunk_overlap_token_size** | `int` | Overlap token size the embedding hard fallback borrows from the previous window when a chunk is still over the embedding model's context limit after chunking. Independent from `chunk_overlap_token_size` (some chunking strategies, e.g. V, deliberately zero that one out for unrelated reasons); `0` disables the fallback's overlap; negative values raise `ValueError` at construction. Configured by env var `EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE`. | `100` |
| **tokenizer** | `Tokenizer` | The function used to convert text into tokens (numbers) and back using .encode() and .decode() functions following `TokenizerInterface` protocol. If you don't specify one, it will use the default Tiktoken tokenizer. An injected tokenizer must be safe to call concurrently from multiple threads and must survive `copy.deepcopy` — see [Injecting a custom tokenizer](#injecting-a-custom-tokenizer). | `TiktokenTokenizer` |
| **tiktoken_model_name** | `str` | If you're using the default Tiktoken tokenizer, this is the name of the specific Tiktoken model to use. This setting is ignored if you provide your own tokenizer. | `gpt-4o-mini` |
| **entity_extract_max_gleaning** | `int` | Number of loops in the entity extraction process, appending history messages | `1` |
| **kg_extraction_validator** | `Callable \| None` | Optional per-chunk hook run after extraction and **before the merge**, so rejected entities/relations never enter the graph, the vector stores, or a `source_id` chain. Signature `(chunk_key, chunk_text, maybe_nodes, maybe_edges)` returning a filtered pair; sync or async. See [Extraction quality hook](#extraction-quality-hook-kg_extraction_validator). | `None` |
| **node_embedding_algorithm** | `str` | Algorithm for node embedding (currently not used) | `node2vec` |
| **node2vec_params** | `dict` | Parameters for node embedding | `{"dimensions": 1536,"num_walks": 10,"walk_length": 40,"window_size": 2,"iterations": 3,"random_seed": 3,}` |
| **embedding_func** | `EmbeddingFunc` | Function to generate embedding vectors from text | `openai_embed` |
| **embedding_batch_num** | `int` | Maximum batch size for embedding processes (multiple texts sent per batch) | `32` |
| **embedding_func_max_async** | `int` | Maximum number of concurrent asynchronous embedding processes | `16` |
| **llm_model_func** | `callable` | Function for LLM generation | `gpt_4o_mini_complete` |
| **llm_model_name** | `str` | LLM model name for generation | `gpt-4o-mini` |
| **summary_context_size** | `int` | Maximum tokens send to LLM to generate summaries for entity relation merging | `10000`（configured by env var SUMMARY_CONTEXT_SIZE) |
| **summary_max_tokens** | `int` | Maximum token size for entity/relation description | `500`（configured by env var SUMMARY_MAX_TOKENS) |
| **llm_model_max_async** | `int` | Base maximum LLM concurrency; also caps per-document chunk extraction tasks, while each entity/relation merge phase uses twice this task limit | `4`（default value changed by env var MAX_ASYNC_LLM; MAX_ASYNC is still accepted as a deprecated alias; `EXTRACT_MAX_ASYNC_LLM` can independently limit actual Extract-role requests) |
| **llm_model_kwargs** | `dict` | Additional parameters for LLM generation | |
| **vector_db_storage_cls_kwargs** | `dict` | Additional parameters for vector database, like setting the threshold for nodes and relations retrieval | cosine_better_than_threshold: 0.2（default value changed by env var COSINE_THRESHOLD) |
| **enable_llm_cache** | `bool` | If `TRUE`, stores LLM results in cache; repeated prompts return cached responses | `TRUE` |
| **enable_llm_cache_for_entity_extract** | `bool` | If `TRUE`, stores LLM results in cache for entity extraction; Good for beginners to debug your application | `TRUE` |
| **addon_params** | `dict` | Runtime knobs for extraction prompts and chunking. See [addon_params](#addon_params). | Env-backed defaults from `SUMMARY_LANGUAGE`, `ENTITY_TYPE_PROMPT_FILE`, and `CHUNK_*` |
| **embedding_cache_config** | `dict` | Configuration for question-answer caching. Contains three parameters: `enabled`: Boolean value to enable/disable cache lookup functionality. When enabled, the system will check cached responses before generating new answers. `similarity_threshold`: Float value (0-1), similarity threshold. When a new question's similarity with a cached question exceeds this threshold, the cached answer will be returned directly without calling the LLM. `use_llm_check`: Boolean value to enable/disable LLM similarity verification. When enabled, LLM will be used as a secondary check to verify the similarity between questions before returning cached answers. | Default: `{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}` |


## addon_params

`addon_params` is a live configuration mapping on each `LightRAG` instance. LightRAG currently reads the fields below; unknown custom keys may remain in the dict, but core LightRAG behavior does not use them.

### Supported Fields

| Field | Value | Purpose |
|---|---|---|
| `language` | Non-empty string. Defaults to `SUMMARY_LANGUAGE`, then `English`. | Output language used in entity and relationship extraction, entity/relation summaries, keyword extraction, and multimodal analysis prompts. |
| `entity_type_prompt_file` | `.yml` or `.yaml` file name only. Loaded from `${PROMPT_DIR:-./prompts}/entity_type`. | Loads an entity extraction prompt profile. The profile can define `entity_types_guidance`, `entity_extraction_examples`, and `entity_extraction_json_examples`. The active extraction mode must have matching examples: text mode needs `entity_extraction_examples`; JSON mode needs `entity_extraction_json_examples`. |
| `entity_types_guidance` | Non-empty string. | Inline entity type guidance injected into extraction prompts. This overrides both the prompt profile file and the built-in default guidance. |
| `chunker` | Dict with F/R/V/P chunking settings (the `C` selector reuses the `fixed_token` sub-dictionary). | Runtime baseline for chunker parameters. Each document gets a slim `chunk_options` snapshot at enqueue time; later edits affect only future enqueues. |

Compact `chunker` shape:

```jsonc
{
  "chunk_token_size": 1200,
  "fixed_token": {
    "chunk_token_size": 1200,
    "chunk_overlap_token_size": 100,
    "split_by_character": null,
    "split_by_character_only": false
  },
  "recursive_character": {
    "chunk_token_size": 1200,
    "chunk_overlap_token_size": 100,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
  },
  "semantic_vector": {
    "chunk_token_size": 1200,
    "breakpoint_threshold_type": "percentile",
    "breakpoint_threshold_amount": null,
    "buffer_size": 1,
    // env/SDK only (CHUNK_V_SENTENCE_SPLIT_REGEX); the REST chunking.params
    // object rejects this key with 422 — see GHSA-32jh-39m7-8x84 (ReDoS)
    "sentence_split_regex": "(?<=[.?!])\\s+|(?<=[。？！])"
  },
  "paragraph_semantic": {
    "chunk_token_size": 2000,
    "chunk_overlap_token_size": 100
  }
}
```

### Initialization

When you create a `LightRAG` object, `addon_params` is normalized before storage initialization:

- If `addon_params` is omitted, LightRAG builds defaults from `SUMMARY_LANGUAGE`, `ENTITY_TYPE_PROMPT_FILE`, and the chunker-related `CHUNK_*` environment variables.
- If you pass a partial dict, missing `language`, `entity_type_prompt_file`, and `chunker` values are still backfilled from the same env-backed defaults.
- `entity_type_prompt_file` and `entity_types_guidance` are resolved into a cached entity extraction prompt profile during construction.
- `chunk_token_size` and `chunk_overlap_token_size` constructor arguments are overlaid into `addon_params["chunker"]` only for slots that were not already set by explicit `addon_params` or strategy-specific env vars.

Example:

```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    addon_params={
        "language": "Chinese",
        "entity_type_prompt_file": "entity_type_prompt.sample.yml",
        "entity_types_guidance": "- Paper: academic papers, reports, and preprints",
        "chunker": {
            "chunk_token_size": 1000,
            "recursive_character": {
                "separators": ["\n\n", "\n", "。", "！", "？", " "]
            }
        },
    },
)
await rag.initialize_storages()
```

### Updating After Creation

`rag.addon_params` is an observable mapping. Top-level updates mark the derived prompt cache dirty; the cache is refreshed the next time LightRAG builds runtime config for extraction or query work.

Update one field:

```python
rag.addon_params["language"] = "Chinese"
rag.addon_params["entity_types_guidance"] = "- Dataset: structured research data"
```

Replace the whole mapping:

```python
rag.addon_params = {
    "language": "German",
    "entity_type_prompt_file": "domain_profile.yml",
}
```

Replacing `rag.addon_params` creates a new observable mapping. If you kept an old reference, discard it and re-read `rag.addon_params` before making more changes.

Change F-strategy fixed-token splitting defaults for future documents:

```python
rag.addon_params["chunker"]["fixed_token"]["split_by_character"] = "\n\n"
rag.addon_params["chunker"]["fixed_token"]["split_by_character_only"] = True
```

`split_by_character` pre-splits text by the given separator before token-window chunking. When `split_by_character_only` is `True`, an oversized segment raises an error instead of being split again by token size.

Change R-strategy recursive splitting defaults for future documents:

```python
rag.addon_params["chunker"]["recursive_character"]["separators"] = [
    "\n\n",
    "\n",
    "###",
    "。",
    "！",
    "？",
    " ",
]
```

Nested `chunker` edits are read when future documents are enqueued. Documents already enqueued keep their persisted `chunk_options` snapshot.

`semantic_vector.sentence_split_regex` is the one exception: it is re-read from `addon_params` (seeded by `CHUNK_V_SENTENCE_SPLIT_REGEX`) on **every** processing run, and any value inside a persisted `chunk_options` snapshot is discarded and logged at WARNING. This also applies to an explicit `chunk_options=` passed to `apipeline_enqueue_documents` — a per-document splitter pattern is not supported. The pattern is applied by `re.split` to the document body while CPython holds the GIL, so an untrusted one can freeze the whole worker process; see [GHSA-32jh-39m7-8x84](https://github.com/HKUDS/LightRAG/security/advisories/GHSA-32jh-39m7-8x84).

### Notes and Precedence

- Entity type guidance precedence is: `addon_params["entity_types_guidance"]` > `entity_type_prompt_file` profile > built-in default guidance.
- Chunker precedence is: explicit `addon_params["chunker"]` values > strategy-specific `CHUNK_*` env vars > legacy constructor fields (`chunk_token_size`, `chunk_overlap_token_size`) > legacy env vars (`CHUNK_SIZE`, `CHUNK_OVERLAP_SIZE`).
- Per-strategy `chunk_token_size`: every strategy reads `chunk_token_size` from its own sub-dict first and falls back to the top-level `chunk_token_size` when its sub-dict doesn't set one. F, R, and V can each seed their sub-dict value from a dedicated env var (`CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE`) or set it explicitly in `addon_params`; when neither is set they inherit the top-level value.
- `paragraph_semantic.chunk_token_size` is the exception: unlike F/R/V it never inherits the top-level `chunk_token_size`; if not explicit it uses `CHUNK_P_SIZE`, then the built-in default `2000`.
- `enable_multimodal_pipeline` is deprecated and ignored if passed in `addon_params`. Use per-document `process_options` such as `i`, `t`, and `e` to control multimodal processing.


## QueryParam

Use `QueryParam` to control the behavior of your query:

```python
class QueryParam:
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    - "naive": Performs a basic search without advanced techniques.
    - "mix": Integrates knowledge graph and vector retrieval.
    """

    only_need_context: bool = False
    """If True, only returns the retrieved context without generating a response."""

    only_need_prompt: bool = False
    """If True, only returns the generated prompt without producing a response."""

    response_type: str = "Multiple Paragraphs"
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""

    stream: bool = False
    """If True, enables streaming output for real-time responses."""

    top_k: int = int(os.getenv("TOP_K", "60"))
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K", "20"))
    """Number of text chunks to retrieve initially from vector search and keep after reranking.
    If None, defaults to top_k value.
    """

    max_entity_tokens: int = int(os.getenv("MAX_ENTITY_TOKENS", "6000"))
    """Maximum number of tokens allocated for entity context in unified token control system."""

    max_relation_tokens: int = int(os.getenv("MAX_RELATION_TOKENS", "8000"))
    """Maximum number of tokens allocated for relationship context in unified token control system."""

    max_total_tokens: int = int(os.getenv("MAX_TOTAL_TOKENS", "30000"))
    """Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt)."""

    # History messages are only sent to LLM for context, not used for retrieval
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    user_prompt: str | None = None
    """User-provided prompt for the query.
    Additional instructions for LLM. If provided, this will be injected into the prompt template.
    Its purpose is to let the user customize the way LLM generates the response.
    """

    disable_user_prompt_prefix: bool = False
    """If True, the server-side global prompt prefix is NOT prepended to `user_prompt`."""

    enable_rerank: bool = True
    """Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued.
    Default is True to enable reranking when rerank model is available.
    """
```

> The default value of `top_k` can be changed by the environment variable `TOP_K`.


## LLM and Embedding Injection

LightRAG requires LLM and Embedding models for document indexing and querying. During initialization, inject the relevant model functions into LightRAG.

### Model Selection Requirements

- **LLM**: at least 32B parameters, 32KB context (64KB recommended). Avoid reasoning models during indexing; use stronger models at query time.
- **Embedding**: must be consistent across indexing and querying. Recommended: `BAAI/bge-m3`, `text-embedding-3-large`. Changing models requires clearing vector storage.
- **Reranker**: significantly improves retrieval. When enabled, set query mode to `mix`. Recommended: `BAAI/bge-reranker-v2-m3`, Jina rerankers.

#### Using OpenAI-like APIs

LightRAG supports OpenAI-like chat/embeddings APIs:

```python
import os
import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "solar-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
        **kwargs
    )

@wrap_embedding_func_with_attrs(embedding_dim=4096, max_token_size=8192, model_name="solar-embedding-1-large-query")
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed.func(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar"
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func  # Pass the decorated function directly
    )
    await rag.initialize_storages()
    return rag
```

> **Important Note on Embedding Function Wrapping:**
>
> `EmbeddingFunc` cannot be nested. Functions decorated with `@wrap_embedding_func_with_attrs` (such as `openai_embed`, `ollama_embed`, etc.) cannot be wrapped again using `EmbeddingFunc()`. This is why we call `xxx_embed.func` (the underlying unwrapped function) instead of `xxx_embed` directly when creating custom embedding functions.

#### Using Hugging Face Models

See `lightrag_hf_demo.py`

```python
from functools import partial
from transformers import AutoTokenizer, AutoModel

# Pre-load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Initialize LightRAG with Hugging Face model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
    llm_model_name='meta-llama/Llama-3.1-8B-Instruct',  # Model name from Hugging Face
    # Use Hugging Face embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=2048,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        func=partial(
            hf_embed.func,  # Use .func to access the unwrapped function
            tokenizer=tokenizer,
            embed_model=embed_model
        )
    ),
)
```

#### Using Ollama Models

Pull the model you plan to use and an embedding model, for example `nomic-embed-text`:

```python
import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192, model_name="nomic-embed-text")
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await ollama_embed.func(texts, embed_model="nomic-embed-text")

# Initialize LightRAG with Ollama model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name='your_model_name',
    embedding_func=embedding_func,
)
```

#### Increasing context size

LightRAG requires at least 32k context tokens. Ollama defaults to 8k. Two approaches:

*Approach 1: Edit Modelfile*

```bash
ollama pull qwen2
ollama show --modelfile qwen2 > Modelfile
# Add this line to Modelfile:
# PARAMETER num_ctx 32768
ollama create -f Modelfile qwen2m
```

*Approach 2: Set `num_ctx` via `llm_model_kwargs`*

```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name='your_model_name',
    llm_model_kwargs={"options": {"num_ctx": 32768}},
    embedding_func=embedding_func,
)
```

> **Important Note on Embedding Function Wrapping:**
>
> `EmbeddingFunc` cannot be nested. Use `xxx_embed.func` to access the underlying unwrapped function.

**Low RAM GPUs**

For low-RAM GPUs (e.g. 6GB), select a small model and tune the context window. For example, `gemma2:2b` with `num_ctx=26000` can find ~197 entities and 19 relations on `book.txt`.

#### LlamaIndex

LightRAG supports integration with LlamaIndex (`llm/llama_index_impl.py`):

```python
import asyncio
from lightrag import LightRAG
from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

async def initialize_rag():
    rag = LightRAG(
        working_dir="your/path",
        llm_model_func=llama_index_complete_if_cache,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=2048,
            model_name=embed_model,
            func=partial(llama_index_embed.func, embed_model=embed_model)
        ),
    )
    await rag.initialize_storages()
    return rag
```

**Further reading:**
- [LlamaIndex Documentation](https://developers.llamaindex.ai/python/framework/)
- [Direct OpenAI Example](../examples/unofficial-sample/lightrag_llamaindex_direct_demo.py)
- [LiteLLM Proxy Example](../examples/unofficial-sample/lightrag_llamaindex_litellm_demo.py)
- [LiteLLM Proxy with Opik Example](../examples/unofficial-sample/lightrag_llamaindex_litellm_opik_demo.py)

#### Using Azure OpenAI Models

```python
import os
import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        **kwargs
    )

@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=8192,
    model_name=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await azure_openai_embed.func(
        texts,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    )

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=embedding_func
)
```

#### Using Google Gemini Models

```python
import os
import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.gemini import gemini_model_complete, gemini_embed

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash",
        **kwargs
    )

@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=2048,
    model_name="models/text-embedding-004"
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await gemini_embed.func(
        texts,
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/text-embedding-004"
    )

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    llm_model_name="gemini-2.0-flash",
    embedding_func=embedding_func
)
```

### Custom Embedding Functions

Use the `@wrap_embedding_func_with_attrs` decorator, and call `.func` when building on an already-decorated function — a decorated function cannot be wrapped again, so the underlying callable must be reached through `.func`:

```python
from lightrag.utils import wrap_embedding_func_with_attrs

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def custom_embed(texts: list[str]) -> np.ndarray:
    # Call the underlying function, not the wrapped version
    return await openai_embed.func(texts, model="text-embedding-3-large")

# Wrong: EmbeddingFunc(func=openai_embed)
# Right: EmbeddingFunc(func=openai_embed.func)
```

`max_token_size` declares the model's real input limit. It is what keeps an over-long text from reaching a service that would split it internally and return one vector per segment — which the return contract below rejects as a vector count mismatch.

> **Pitfall — switching embedding models**: when changing the embedding model you MUST clear the data directory (optionally keeping `kv_store_llm_response_cache.json` for the LLM cache). Existing vectors will not match the new model's space.

### Embedding Function Return Contract

Every embedding function — built-in or custom — MUST return a 2D numpy array of shape `(len(texts), embedding_dim)`: exactly one row per input text, in input order. Every vector storage backend consumes the result positionally (`embeddings[i]` is stored for `texts[i]`), so `EmbeddingFunc` validates the result on every call and raises `ValueError` on any mismatch. It never reshapes, slices or pads the result — once the row-to-input mapping is wrong it cannot be recovered, and a silent repair would store vectors under the wrong records.

The array rank and the dimension are always checked. The row count is checked against the input batch, which is read from the first positional argument, or — for a keyword call — from the kwarg matching the wrapped function's first parameter name. If the batch cannot be resolved that way (a callable whose first parameter is positional-only or `*args`, or one exposing no signature), the row count alone is left unverified rather than guessed at.

| Returned shape | Result |
| --- | --- |
| `(len(texts), embedding_dim)` | Accepted |
| Empty array for an empty input list | Accepted (including a bare `np.array([])`) |
| `(embedding_dim,)` for a single input | `ValueError` — a single input still returns `(1, embedding_dim)` |
| More rows than inputs | `ValueError: Vector count mismatch` |
| Fewer rows than inputs | `ValueError: Vector count mismatch` |
| Wrong number of columns | `ValueError: Embedding dimension mismatch` |
| Flattened (1D) or nested (3D) array | `ValueError: unexpected shape` |

The two mismatches that look alike are distinguished deliberately, because their fixes differ:

- **Vector count mismatch** (more rows than inputs) usually means the embedding service split an over-long input internally and returned one vector per segment. Declare the model's real token limit so texts are truncated before the call — `EMBEDDING_TOKEN_LIMIT` on the API server, or `max_token_size` on `@wrap_embedding_func_with_attrs` for a custom function. A provider that legitimately emits several vectors per input needs a dedicated adapter that normalizes its output to one vector per input, with an explicit mapping, before it reaches `EmbeddingFunc`.
- **Embedding dimension mismatch** (wrong number of columns) means the declared `embedding_dim` does not match the model actually being called, or the endpoint ignored the requested output dimension. Reconcile `EMBEDDING_DIM` / `embedding_dim` with the model. Vectors already stored under the previously declared dimension do not match the corrected one, so clear the data directory as well unless nothing has been indexed yet.

Each `ValueError` is accompanied by a `logger.error` carrying the likely cause and the remedy, so the diagnosis stays in the server log even when only the short exception message surfaces.

### Rerank Function Injection

To enhance retrieval quality, documents can be re-ranked based on a more effective relevance scoring model. The `rerank.py` file provides three Reranker provider driver functions:

- **Cohere / vLLM**: `cohere_rerank`
- **Jina AI**: `jina_rerank`
- **Aliyun**: `ali_rerank`

Inject one of these functions into the `rerank_model_func` attribute of the LightRAG object. For detailed usage, refer to `examples/rerank_example.py`.

### Injecting a Custom Tokenizer

Any object with `encode(str) -> list[int]` and `decode(list[int]) -> str` can be
wrapped in `Tokenizer` and passed as `tokenizer=`. Two requirements apply:

1. **It must be safe to call concurrently from multiple threads.** Token counting
   is CPU-bound, so LightRAG runs it in worker threads to keep the asyncio event
   loop responsive; several of them may enter your `encode`/`decode` at once.
   LightRAG deliberately does not serialize calls on your behalf — a lock owned
   by LightRAG would end up being waited on by the event loop behind a worker
   thread, which is exactly the stall the threading is there to avoid.
2. **It must survive `copy.deepcopy`.** `LightRAG` is a dataclass and builds its
   internal config with `dataclasses.asdict`, which deep-copies non-dataclass
   fields.

The two interact: if you achieve thread safety with an internal `threading.Lock`,
deep-copying it raises `TypeError: cannot pickle '_thread.lock' object`. Declare
`__deepcopy__` returning `self`, which is sound precisely because a thread-safe
tokenizer is safe to share:

```python
class MyTokenizer:
    def __init__(self):
        self._lock = threading.Lock()

    def __deepcopy__(self, memo):
        return self  # thread-safe, therefore shareable

    def encode(self, content: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...

rag = LightRAG(..., tokenizer=Tokenizer("my-model", MyTokenizer()))
```

The built-in `TiktokenTokenizer` satisfies both. Note that copying it is not a
way to get isolation: `tiktoken` caches encodings in a process-wide registry, so
every `TiktokenTokenizer` for a given model — copies included — resolves to the
same underlying BPE engine.

### Extraction Quality Hook (`kg_extraction_validator`)

`kg_extraction_validator` is an optional per-chunk hook that runs after entity /
relation extraction and **before the merge**, so anything it rejects never
enters the knowledge graph, the vector stores, or an entity's `source_id`
chain. Filtering here is cheaper than cleaning up afterwards: deleting an entity
post-merge leaves its `source_id` contributions behind, and unwinding those
takes a purge and a re-ingest.

```python
from lightrag.utils import logger, normalize_entity_name


def validate(chunk_key, chunk_text, maybe_nodes, maybe_edges):
    # maybe_nodes: entity_name -> list[entity_dict]
    # maybe_edges: (src, tgt)  -> list[edge_dict]
    # Both carry source_id / file_path already — these are the merge inputs.

    # Canonicalize BOTH sides of the grounding comparison. Extraction runs
    # every name through normalize_entity_name, which does more than case:
    # full-width ＡＩ becomes AI, and spaces between Chinese and Latin are
    # removed, so "北京 AI" in the source arrives as the entity 北京AI. Testing
    # such a name against raw chunk text finds nothing and silently deletes a
    # legitimate entity, so run the same normalization over the haystack.
    # Case-folding on top: the prompt asks the model to title-case
    # case-insensitive names, so "machine learning" arrives as
    # "Machine Learning".
    haystack = normalize_entity_name(chunk_text).casefold()

    # The keys are already normalized by extraction, so only case-fold here.
    def is_junk(name: str) -> bool:
        return len(name) < 2 or name.casefold() not in haystack

    # Apply it to entities...
    for name in [n for n in maybe_nodes if is_junk(n)]:
        logger.info("rejected entity %s from %s", name, chunk_key)
        del maybe_nodes[name]
    # ...and to relation endpoints. A name reaches the graph through either
    # shape: an endpoint with no entity record is materialized as an UNKNOWN
    # node, so skipping this loop lets the names you just deleted back in.
    for key in [k for k in maybe_edges if is_junk(k[0]) or is_junk(k[1])]:
        logger.info("rejected relation %s from %s", key, chunk_key)
        del maybe_edges[key]
    return maybe_nodes, maybe_edges

rag = LightRAG(..., kg_extraction_validator=validate)
```

Contract:

- **Signature** `(chunk_key, chunk_text, maybe_nodes, maybe_edges)`, returning a
  `(maybe_nodes, maybe_edges)` pair of dicts with the same shapes. Filtering in
  place and returning the same objects is fine.
- **`chunk_text` is the text the model received** as the prompt's
  `---Input Text---`, not the stored chunk content: parser-internal markup
  (`<drawing id/path/src>`, `<table id>`, `<equation id>`, `<cite refid>`) is
  stripped, and the result goes through `sanitize_text_for_encoding` just as
  the LLM wrapper does before calling the provider — keeping the chunk's own
  boundary whitespace, which the model sees because the chunk sits inside the
  prompt's fenced `---Input Text---` section. Grounding against the
  stored form would accept an entity named after a hidden identifier or file
  path the model never saw, and reject one it did see wherever removing a
  `<cite>` wrapper joins two words.
- **It is the input text, not the whole prompt.** For a chunk with a heading,
  the prompt also carries a `---Section Context---` block with the heading
  path. That is deliberately excluded: the extraction prompt tells the model to
  use the heading path as background only and **not** to extract entities or
  relationships from the heading text. Folding it into `chunk_text` would make
  a grounding rule accept precisely the names the prompt forbids, legitimising
  heading-derived entities instead of catching them. Heading context can
  legitimately shape a *description*, but descriptions are model-authored prose
  that no substring check grounds anyway.
- **Synchronous or async.** An awaitable return value is awaited. A sync hook
  runs on the event loop, so a CPU-heavy validator should do its own
  `asyncio.to_thread` — the same guidance as a custom `chunking_func`.
- **`None` (the default) leaves the pipeline unchanged.**
- **Coverage.** Every path that extracts goes through it: the document pipeline
  and `ainsert_custom_chunks` alike. `ainsert_custom_kg` does not extract — the
  caller supplies the graph directly — so it is not filtered.
- **Multimodal entities are not shown to the hook.** For a drawing / table /
  equation chunk, LightRAG synthesizes one entity from the sidecar and links it
  to the chunk's surviving entities. That injection happens *after* the hook, so
  a grounding rule like the one above cannot delete it, and an entity the hook
  rejected cannot reappear as the endpoint of an injected edge.
- **Filter relation endpoints too — core does not do it for you.** A name
  reaches the graph through either shape: `_merge_edges_then_upsert`
  materializes an endpoint that has no entity record as an `UNKNOWN`-typed
  node, into the graph, the vector store and the `source_id` chain. So
  deleting an entity while leaving a relation pointing at it undoes the
  deletion, and a chunk where the name appears *only* as an endpoint stores it
  even though another chunk rejected it. The example above closes both by
  running the same `is_junk` over `maybe_edges`.
- **Make the rule a function of the name**, so every chunk decides the same
  way. Chunks are extracted concurrently and independently; a rule that is not
  name-deterministic can reject a name in one chunk and keep it in the next,
  and the entity survives through whichever chunk kept it. Core deliberately
  does not aggregate rejections across chunks — a verdict on one chunk is not
  a verdict on the document, and only your rule knows whether it was meant to
  be.
- **Failures are not swallowed.** A hook that raises, or that returns anything
  other than a two-element sequence of dicts (`TypeError`), fails the chunk and
  therefore the ingest. A validator that is silently skipped is a validator
  that is not validating.
- **What a failed ingest leaves behind depends on the path.** The pipeline
  marks the document FAILED. `ainsert_custom_chunks` also marks it FAILED but
  **retains** its journal and any staged data instead of rolling back:
  repeating the same call resumes the operation (roll-forward belongs to the
  SDK caller), while `/documents/scan` — through
  `arollback_failed_custom_chunk_patches` — is what rolls it back. Do not
  assume a failed custom-chunk call left no recoverable state.
- **A stateful validator keeps its identity.** A bound method or callable object
  that accumulates an audit log sees its own instance, not a per-document copy.
  As with a custom tokenizer, however, `LightRAG` builds its internal config
  with `dataclasses.asdict`, so a validator holding something `copy.deepcopy`
  rejects (a bare `threading.Lock`) must declare `__deepcopy__` returning
  `self` — see *Injecting a Custom Tokenizer* above.

### User Prompt vs. Query

When using LightRAG for content queries, avoid combining the search process with unrelated output processing, as this significantly impacts query effectiveness. The `user_prompt` parameter in `QueryParam` does not participate in the RAG retrieval phase — it guides the LLM on how to process the retrieved results after the query is completed.

"Does not participate in retrieval" means it does not influence *what* is found or *how* it is ranked: it is not used for keyword extraction, vector search, or reranking. It does still consume part of the token budget, because it genuinely occupies space in the final prompt alongside the retrieved context.

```python
query_param = QueryParam(
    mode="hybrid",
    user_prompt="For diagrams, use mermaid format with English/Pinyin node names and Chinese display labels",
)

response_default = rag.query(
    "Please draw a character relationship diagram for Scrooge",
    param=query_param
)
print(response_default)
```

### A Global User Prompt Prefix

`user_prompt` is supplied per request, so it cannot express an output policy
that should hold for every caller. `LightRAG.user_prompt_prefix` is that policy:
a server-side string prepended to each request's `user_prompt`.

```python
rag = LightRAG(..., user_prompt_prefix="Answer in the language of the question.\n\n")
```

For the API server it comes from the environment instead — `USER_PROMPT_PREFIX`
for a short value, or `USER_PROMPT_PREFIX_FILE` (a `.md`/`.txt` file name under
`PROMPT_DIR/user_prompt`) when the text is long, multi-paragraph, or contains
`${...}`, which python-dotenv would otherwise interpolate away.

The two strings are concatenated **verbatim, with no separator inserted** — end
the prefix with your own `\n\n` so it does not run into the caller's text. The
prefix comes first because a model weights later instructions more heavily on
conflict, so the per-request prompt wins.

**An empty `user_prompt` does not disable the prefix.** When a request sends no
`user_prompt` — `None`, `""`, or the field omitted entirely — the prefix alone
becomes the instructions sent to the LLM. This is the common deployment: the
operator sets one policy and callers send nothing.

```python
# All three send exactly "Answer in the language of the question." to the model.
rag.query("...", param=QueryParam(mode="hybrid"))
rag.query("...", param=QueryParam(mode="hybrid", user_prompt=None))
rag.query("...", param=QueryParam(mode="hybrid", user_prompt=""))
```

This matters for the WebUI in particular, which ships `user_prompt: ""` as its
default: leaving the box blank applies the operator's policy rather than
clearing it. The `Additional Instructions` section falls back to `n/a` only when
**both** the prefix and the request's `user_prompt` are empty.

A request opts out with `disable_user_prompt_prefix` — the only way to suppress
the prefix — which is what lets a front-end take full control of the final
instruction text:

```python
QueryParam(user_prompt="...", disable_user_prompt_prefix=True)
```

The prefix is configuration, not request data: a request can decline it but can
never read or replace it. Three limits are worth knowing:

- **`bypass` mode ignores it**, as it ignores `user_prompt` entirely — empty or
  not. That path has no `{user_prompt}` slot and its `system_prompt` argument
  belongs to the caller, so this is not an exception to the rule above: bypass
  simply sends no user instructions at all.
- **`only_need_prompt=True` returns the composed prompt**, so any client that
  can set that debug flag can read the prefix verbatim.
- **`only_need_context` and `only_need_prompt` are charged for the prefix**, even
  though `only_need_context` returns before any prompt is sent. These switches
  preview the real request: if retrieval-only calls skipped the charge they
  would report more chunks than a live query retrieves, and context sized
  against that number would be truncated at answer time. `/query/data`
  (`aquery_data`) is retrieval-only and follows the same rule.
- **A custom `system_prompt` without a `{user_prompt}` placeholder drops it**,
  the same way it already drops `user_prompt`. The token budget accounts for
  this: the prefix is charged against the context allowance only when the
  template that will actually be rendered has somewhere to put it.

The prefix participates in the answer cache key, so editing it invalidates
answers generated under the old one. With no prefix configured the key is
unchanged, so existing cache entries keep hitting.


## Storage Backends

### Sotrage Types

LightRAG uses 4 types of storage for different purposes:

| Storage Type | Purpose |
|---|---|
| **KV_STORAGE** | LLM response cache, text chunks, document information |
| **VECTOR_STORAGE** | Entity/relation/chunk embedding vectors |
| **GRAPH_STORAGE** | Entity-relation graph structure |
| **DOC_STATUS_STORAGE** | Document indexing status |

The default implementation of each storage type (marked `(default)` below) is an
in-memory database persisted to local files under `working_dir`: the whole
dataset resides in the process's memory, so capacity is bounded by available
RAM. The defaults are suitable **only for small-scale testing, evaluation, and
debugging, and are not suitable for production** — for production, PostgreSQL is
the recommended backend and can serve all four storage types on its own.

### Supported Implementations

**KV_STORAGE**
```
JsonKVStorage        JsonFile (default)
PGKVStorage          Postgres
RedisKVStorage       Redis
MongoKVStorage       MongoDB
OpenSearchKVStorage  OpenSearch
```

**GRAPH_STORAGE**
```
NetworkXStorage          NetworkX (default)
Neo4JStorage             Neo4J
PGGraphStorage           PostgreSQL with AGE plugin
PGTableGraphStorage      PostgreSQL, plain tables (no AGE, no extensions)
MemgraphStorage          Memgraph
OpenSearchGraphStorage   OpenSearch
```

> Testing has shown that Neo4J delivers superior performance in production environments compared to PostgreSQL with AGE plugin.
>
> `PGTableGraphStorage` implements the graph layer on ordinary indexed tables plus
> JSONB, so it runs on any stock PostgreSQL 14+ — including managed instances
> (RDS, Cloud SQL, Supabase, Neon) where the AGE extension cannot be installed.
> It shares the same `POSTGRES_*` configuration and connection pool as the other
> PG storages. Choose `PGGraphStorage` only if you specifically need AGE/Cypher.

**VECTOR_STORAGE**
```
NoopVectorDBStorage         Disabled (graph-only ingestion)
NanoVectorDBStorage         NanoVector (default)
PGVectorStorage             Postgres
MilvusVectorDBStorage       Milvus
FaissVectorDBStorage        Faiss
QdrantVectorDBStorage       Qdrant
MongoVectorDBStorage        MongoDB
OpenSearchVectorDBStorage   OpenSearch
```

#### Graph-only ingestion

`NoopVectorDBStorage` is intended for an initial or offline corpus backfill
where the graph and KV stores are authoritative and vector indexes can be
materialized once from the final state. It avoids embedding and persisting
intermediate entity, relationship, and chunk revisions during ingestion.

Do not use this workflow when newly inserted documents must become queryable
immediately. Normal incremental ingestion should use the intended persistent
vector backend from the beginning.

Configure the backfill process with the no-op backend. `embedding_func=None` is
supported when no other configured component requires embeddings:

```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=None,
    vector_storage="NoopVectorDBStorage",
)
```

The backend accepts vector mutations without calling the embedding function or
persisting vectors. Graph, full-document, text-chunk, LLM-cache, document-status,
and graph-recovery writes continue normally.

While `NoopVectorDBStorage` is active, only `bypass` queries are available.
`local`, `global`, `hybrid`, `mix`, and `naive` modes require vector indexes and
raise an error that points to `lightrag-rebuild-vdb`.

If the semantic-vector (`V`) chunker is selected while `embedding_func=None`,
it logs a warning and falls back to recursive-character chunking. Configure an
embedding function during ingestion if semantic-vector chunk boundaries are
required; this is separate from whether vectors are persisted.

##### Switching to persistent vectors

After the backfill, stop the server and all ingestion writers. Keep
`WORKING_DIR`, `WORKSPACE`, graph storage, KV storage, and their connection
settings unchanged. For example, switch from Noop to NanoVector with an OpenAI
embedding model while pointing to the same graph and KV sources:

```bash
export WORKING_DIR=/data/lightrag/rag_storage
export WORKSPACE=project_a
export LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
export LIGHTRAG_KV_STORAGE=JsonKVStorage
export LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
export EMBEDDING_BINDING=openai
export EMBEDDING_MODEL=text-embedding-3-small
export EMBEDDING_DIM=1536

lightrag-rebuild-vdb  # Select "Rebuild ALL vector storages"
```

Replace all example values with the backfill's actual storage settings and the
production embedding model, dimension, host, and credentials. Backend-specific
connection variables must remain available. If the same `.env` already contains
these values, only the vector and embedding entries need to change. Run this in
a new process or after a restart, then keep the persistent configuration for
later queries and incremental ingestion.

Rebuild cost and memory grow with the final graph and chunk data. If rebuilding
fails or is interrupted, keep writers stopped and rerun it with the same
configuration; the graph and KV sources remain unchanged. Start the server only
after the tool reports a successful rebuild, for example with
`lightrag-server`, because LightRAG has no persisted vector-index readiness
marker.

See `lightrag/tools/README_REBUILD_VDB.md` for rebuild options and operational
details.

**DOC_STATUS_STORAGE**
```
JsonDocStatusStorage        JsonFile (default)
PGDocStatusStorage          Postgres
MongoDocStatusStorage       MongoDB
OpenSearchDocStatusStorage  OpenSearch
```

Example connection configurations for each storage type can be found in the repository's `env.example` file. The database instance in the connection string must be created beforehand — LightRAG only creates tables within the instance, not the instance itself.

###  Backend-Specific Setup

#### Using Neo4J Storage

For production level scenarios you will most likely want to leverage an enterprise solution for KG storage. Running Neo4J in Docker is recommended for seamless local testing. See: https://hub.docker.com/_/neo4j

```bash
export NEO4J_URI="bolt://localhost:7687"  # Single instance / Docker: direct connection
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
export NEO4J_DATABASE="neo4j"  # Required for community edition
```

**Choosing the URI scheme.** The scheme prefix of `NEO4J_URI` selects the driver's connection mode:

- `bolt://` / `bolt+s://` — direct connection. The driver talks only to the host and port written in the URI. Use this for a single Neo4j instance, a single Docker container, or any deployment reached through a port mapping or proxy.
- `neo4j://` / `neo4j+s://` — routing connection. The driver first asks the server for a routing table and then connects to the addresses the server advertises. Use this for Neo4j Aura and clustered deployments, where it provides read/write routing and failover.

**Troubleshooting `Unable to retrieve routing information`.** This error is raised by the Neo4j driver, not by LightRAG, and it only occurs with the `neo4j://` scheme. It means none of the routers returned a routing table. Check, in order:

1. The Neo4j server is actually running and reachable on the configured host and port (a stopped or restarting instance produces exactly this message).
2. If the instance is a single node or a Docker container, switch `NEO4J_URI` to `bolt://`. A single instance advertises its own address in the routing table; when the server is reached through Docker port mapping or a proxy, that advertised address is often not reachable from the client, so the routing refresh fails even though the initial connection succeeded. The direct scheme skips the routing table entirely.
3. If you must keep `neo4j://` against a single Docker container, set `server.default_advertised_address` on the Neo4j side to an address the client can reach.

LightRAG retries transient `ServiceUnavailable` errors a few times with backoff, but it cannot recover from a database that stays unreachable; the storage call fails after the retries are exhausted.

```python
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage",
    )
    await rag.initialize_storages()
    return rag
```

See `test_neo4j.py` for a working example.

#### Using PostgreSQL Storage

PostgreSQL can provide a one-stop solution as KV store, VectorDB (pgvector), and GraphDB (`PGTableGraphStorage` on plain indexed tables, or `PGGraphStorage` on Apache AGE). PostgreSQL version 16.6 or higher is supported.

- PostgreSQL is lightweight; the whole binary distribution including all necessary plugins can be zipped to 40MB: Ref to [Windows Release](https://github.com/ShanGor/apache-age-windows/releases/tag/PG17%2Fv1.5.0-rc0) as it is easy to install for Linux/Mac.
- If you prefer Docker and graph storage is `PGTableGraphStorage` (the recommended choice, which needs no Apache AGE), the official pgvector image `pgvector/pgvector:pg18` is all you need.
- Only `PGGraphStorage` requires an AGE-bundled image; to avoid hiccups there, start with https://hub.docker.com/r/gzdaniel/postgres-for-rag (published for `linux/amd64` and `linux/arm64`). The latest image no longer ships hardcoded credentials; on first start it creates the user, password, and database from the `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` environment variables (these are set automatically when you deploy via the `scripts/setup/setup.sh` wizard, so you can pick any values).
- How to start: see [examples/lightrag_gemini_postgres_demo.py](https://github.com/HKUDS/LightRAG/blob/main/examples/lightrag_gemini_postgres_demo.py)
- For high-performance graph database requirements, Neo4j is recommended as Apache AGE's performance is not as competitive.

#### Using Faiss Storage

Before using Faiss, manually install `faiss-cpu` or `faiss-gpu`:

```bash
pip install faiss-cpu
```

```python
async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=2048,
        model_name="all-MiniLM-L6-v2",
        func=embedding_func,
    ),
    vector_storage="FaissVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.3
    }
)
```

#### Using Memgraph for Storage

Memgraph is a high-performance, in-memory graph database compatible with the Neo4j Bolt protocol. See: https://memgraph.com/download

```bash
export MEMGRAPH_URI="bolt://localhost:7687"
```

```python
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="MemgraphStorage",
    )
    await rag.initialize_storages()
    return rag
```

#### Using Milvus for Vector Storage

Milvus is a high-performance, scalable vector database for production-level vector storage. For full configuration options including index types (HNSW, HNSW_SQ, IVF, DISKANN, etc.) and metric types, see [docs/MilvusConfigurationGuide.md](./MilvusConfigurationGuide.md).

**Quick setup via environment variables:**

```bash
MILVUS_URI=http://localhost:19530
MILVUS_DB_NAME=lightrag
LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage
```

**Quick setup via Python SDK:**

```python
rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=...,
    embedding_func=...,
    vector_storage="MilvusVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "milvus_uri": "http://localhost:19530",
        "milvus_db_name": "lightrag",
        "cosine_better_than_threshold": 0.2,
    },
)
```

#### Using MongoDB Storage

MongoDB provides a one-stop storage solution for LightRAG with native KV storage and vector storage. LightRAG uses MongoDB collections to implement a simple graph storage.

`MongoVectorDBStorage` requires a MongoDB deployment with Atlas Search / Vector Search support (e.g., MongoDB Atlas or Atlas local). The setup wizard's bundled local Docker MongoDB service is MongoDB Community Edition — it can be used for KV/graph/doc-status storage but **not** for `MongoVectorDBStorage`.

#### Using Redis Storage

LightRAG supports Redis as KV storage. Configure persistence and memory usage carefully. Recommended Redis configuration:

```
save 900 1
save 300 10
save 60 1000
stop-writes-on-bgsave-error yes
maxmemory 4gb
maxmemory-policy noeviction
maxclients 500
```

When the interactive setup manages a local Redis container, it stages a user-editable config at `./data/config/redis.conf` and mounts it into the container. Setup preserves that file on reruns so local Redis tuning can be adjusted without losing manual edits.

#### Using OpenSearch Storage

OpenSearch provides a unified storage solution for all four LightRAG storage types (KV, Vector, Graph, DocStatus). It offers native k-NN vector search, full-text search, and horizontal scalability without cloud-only restrictions.

**Requirements**: OpenSearch 3.x or higher with k-NN plugin enabled.

Install with Docker (without plugins):
```bash
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=<custom-admin-password>" \
  opensearchproject/opensearch:latest
```

Install with Docker Compose (Recommended, with plugins):
```bash
curl -O https://raw.githubusercontent.com/opensearch-project/opensearch-build/main/docker/release/dockercomposefiles/docker-compose-3.x.yml
OPENSEARCH_INITIAL_ADMIN_PASSWORD=<custom-admin-password> docker-compose -f docker-compose-3.x.yml up -d
```

**Configuration** (see `env.example` for full list):
```bash
export OPENSEARCH_HOSTS=localhost:9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASSWORD=<custom-admin-password>
export OPENSEARCH_USE_SSL=true
export OPENSEARCH_VERIFY_CERTS=false
```

**Usage**:
```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=your_llm_func,
    embedding_func=your_embed_func,
    kv_storage="OpenSearchKVStorage",
    doc_status_storage="OpenSearchDocStatusStorage",
    graph_storage="OpenSearchGraphStorage",
    vector_storage="OpenSearchVectorDBStorage",
)
```

**Graph Traversal**: When the OpenSearch SQL plugin with PPL support is available, graph queries use server-side BFS via the `graphlookup` command for optimal performance. Otherwise, it falls back to client-side batched BFS. Auto-detected at startup, or force via `OPENSEARCH_USE_PPL_GRAPHLOOKUP=true|false`.

**Integration Testing**:

1. Start OpenSearch using Docker Compose:
```bash
OPENSEARCH_INITIAL_ADMIN_PASSWORD=<custom-admin-password> docker-compose -f docker-compose-3.x.yml up -d
```

2. Verify the cluster is running:
```bash
curl -sk -u admin:<custom-admin-password> https://localhost:9200
curl -sk -u admin:<custom-admin-password> https://localhost:9200/_cat/plugins?v
```

3. Run unit tests (no OpenSearch required — uses mocks):
```bash
python -m pytest tests/kg/opensearch_impl/test_opensearch_storage.py -v
```

4. Run the OpenSearch storage demo:
```bash
export OPENSEARCH_HOSTS=localhost:9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASSWORD=<custom-admin-password>
export OPENSEARCH_USE_SSL=true
export OPENSEARCH_VERIFY_CERTS=false
python examples/opensearch_storage_demo.py
```

5. Run the full OpenAI + OpenSearch demo (requires `OPENAI_API_KEY`):
```bash
export OPENAI_API_KEY=your-api-key
python examples/lightrag_openai_opensearch_graph_demo.py
```

6. Visualize the knowledge graph via LightRAG WebUI:
```bash
LIGHTRAG_KV_STORAGE=OpenSearchKVStorage \
LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage \
LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage \
LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage \
LLM_BINDING=openai \
EMBEDDING_BINDING=openai \
EMBEDDING_MODEL=text-embedding-3-large \
EMBEDDING_DIM=3072 \
OPENAI_API_KEY=your-api-key \
lightrag-server
```


## Data Isolation Between LightRAG Instances

The `workspace` parameter ensures data isolation between different LightRAG instances. Once initialized, the `workspace` is immutable.

| Storage Type | Isolation Method |
|---|---|
| `JsonKVStorage`, `JsonDocStatusStorage`, `NetworkXStorage`, `NanoVectorDBStorage`, `FaissVectorDBStorage` | Workspace subdirectories |
| `RedisKVStorage`, `MilvusVectorDBStorage`, `MongoKVStorage`, `MongoVectorDBStorage`, `MongoGraphStorage`, `PGGraphStorage` | Workspace prefix on collection name |
| `QdrantVectorDBStorage` | Payload-based partitioning (Qdrant multitenancy) |
| `PGKVStorage`, `PGVectorStorage`, `PGDocStatusStorage`, `PGTableGraphStorage` | `workspace` field in tables |
| `Neo4JStorage` | Labels |
| `OpenSearch*` | Index name prefixes |

**Legacy compatibility**: Default workspace for PostgreSQL non-graph storage is `default`; for PostgreSQL AGE graph storage is null; for Neo4j graph storage is `base`.

Storage-specific workspace environment variables override the common `WORKSPACE` variable: `REDIS_WORKSPACE`, `MILVUS_WORKSPACE`, `QDRANT_WORKSPACE`, `MONGODB_WORKSPACE`, `POSTGRES_WORKSPACE`, `NEO4J_WORKSPACE`, `OPENSEARCH_WORKSPACE`.

For a practical demonstration of managing multiple isolated knowledge bases, see [Workspace Demo](../examples/lightrag_gemini_workspace_demo.py).


## Insert

* Basic Insert

```python
rag.insert("Text")
```

* Batch Insert

```python
# Basic Batch Insert
rag.insert(["TEXT1", "TEXT2", ...])

# Batch Insert with custom batch size
rag = LightRAG(
    ...
    working_dir=WORKING_DIR,
    max_parallel_insert=4
)
rag.insert(["TEXT1", "TEXT2", "TEXT3", ...])  # Processed in batches of 4
```

The `max_parallel_insert` parameter determines the number of documents processed concurrently. Default is **3**. Recommended to keep **below 10**, as the bottleneck typically lies with the LLM.

* Insert with ID

The number of documents and IDs must be the same.

```python
# Single text with ID
rag.insert("TEXT1", ids=["ID_FOR_TEXT1"])

# Multiple texts with IDs
rag.insert(["TEXT1", "TEXT2", ...], ids=["ID_FOR_TEXT1", "ID_FOR_TEXT2"])
```

* Insert using Pipeline

`apipeline_enqueue_documents` and `apipeline_process_enqueue_documents` allow incremental insertion of documents in the background while the main thread continues executing.

```python
rag = LightRAG(..)
await rag.apipeline_enqueue_documents(input)
# Your routine in loop
await rag.apipeline_process_enqueue_documents(input)
```

* Insert Multi-file Type Support

The `textract` library supports reading TXT, DOCX, PPTX, CSV, and PDF:

```python
import textract

file_path = 'TEXT.pdf'
text_content = textract.process(file_path)
rag.insert(text_content.decode('utf-8'))
```

* Citation Functionality

By providing file paths, the system ensures sources can be traced back to their original documents:

```python
documents = ["Document content 1", "Document content 2"]
file_paths = ["path/to/doc1.txt", "path/to/doc2.txt"]

rag.insert(documents, file_paths=file_paths)
```


## Edit Entities and Relations

LightRAG supports comprehensive knowledge graph management: create, edit, and delete entities and relationships.

* Create Entities and Relations

```python
# Create entity
entity = rag.create_entity("Google", {
    "description": "Google is a multinational technology company specializing in internet-related services and products.",
    "entity_type": "company"
})

product = rag.create_entity("Gmail", {
    "description": "Gmail is an email service developed by Google.",
    "entity_type": "product"
})

# Create relation
relation = rag.create_relation("Google", "Gmail", {
    "description": "Google develops and operates Gmail.",
    "keywords": "develops operates service",
    "source_id": "chunk-google-gmail",
    "weight": 1.5
})
```

* Edit Entities and Relations

```python
# Edit entity attributes
updated_entity = rag.edit_entity("Google", {
    "description": "Google is a subsidiary of Alphabet Inc., founded in 1998.",
    "entity_type": "tech_company"
})

# Rename entity (with all its relationships properly migrated)
renamed_entity = rag.edit_entity("Gmail", {
    "entity_name": "Google Mail",
    "description": "Google Mail (formerly Gmail) is an email service."
})

# Edit relation
updated_relation = rag.edit_relation("Google", "Google Mail", {
    "description": "Google created and maintains Google Mail service.",
    "keywords": "creates maintains email service",
    "weight": 3.0
})
```

Entity names supplied to `create_entity` and new names supplied during
`edit_entity` renames use the same normalization rules as extracted entity
names. When editing an existing entity, LightRAG first preserves an exact
legacy name match and otherwise falls back to the normalized name.
`insert_custom_kg` applies the same rules to declared entity names and both
endpoints of every relationship before writing any custom KG data.
`merge_entities` resolves existing exact legacy source/target names first and
otherwise uses normalized names. The target may be an existing entity or a
new normalized name created by the merge.

### Relation Weight Contract

Relation `weight` has an evidence-count floor. Each distinct real ID in the
`source_id` field contributes one unit of evidence, and a larger explicit
weight is an optional importance boost:

```text
weight >= len(distinct real source IDs)
```

Multiple source IDs use the `<SEP>` separator. Empty values and the historical
no-source placeholders `manual_creation` and `UNKNOWN` are not evidence. When
a relation has no real source IDs, its evidence count is zero, so a
non-negative fractional weight is valid. When creating a source-less relation,
omit `source_id`; when editing an existing relation, set `source_id` to an
empty string in the same edit that lowers the weight.

`create_relation`, `edit_relation`, and `insert_custom_kg` validate this
contract before writing graph or vector data; invalid Python API inputs raise
`ValueError` and the REST graph API returns HTTP 400. Relation edits validate
the complete post-edit shape, so `source_id` and `weight` can be changed
together. Existing legacy relations are repaired upward when extraction adds
evidence, an entity rename rewrites their endpoints, an unrelated relation edit
rewrites the row, or a relation is rebuilt from surviving chunks (document
purge, resume, and custom-chunk rollback). `lightrag-rebuild-vdb` is not such a
repair point: it mirrors each graph edge into the vector storage field for
field, copying the stored weight verbatim without touching the graph.

A rebuild re-derives the relation from the extraction results cached for the
surviving chunks, so — like the rebuilt description and keywords — the weight is
recomputed rather than preserved: it becomes the summed fragment weights lifted
to the surviving evidence count. Weight therefore follows evidence downward as a
purge removes chunks, and an importance boost applied through `edit_relation`
does not survive a rebuild that finds cached fragments. When no cached fragment
survives, the rebuild keeps the stored weight.

When entity merging redirects multiple relations onto the same endpoint, the
result is:

```text
merged weight = max(all input weights, distinct merged real source IDs)
```

This preserves a larger manual boost while preventing the merged weight from
falling below its evidence count.

### Chunk tracking across a rename or merge

A rename and a merge do not drop chunk tracking, they migrate it: the row moves
to the surviving key. Two orderings have to hold at once for that migration to
be crash-safe, and satisfying either one alone re-breaks the other:

1. **The new row is written before the old one is deleted.** Otherwise a failure
   in between leaves the row under neither key, which turns a curated row absent
   and re-arms the reseed from a possibly stale graph `source_id`.
2. **The old row is deleted only after a confirmed graph commit** has removed the
   object it described. Otherwise the old object — which is what is still on disk
   until that commit — sits there with no authoritative provenance, and a later
   document purge can read its truncated `source_id` as "no remaining sources".

The commit between them is checked, not assumed: a graph backend may *decline* to
commit (`NetworkXStorage.index_done_callback` returns `False` when another process
published a newer file, reloading from disk and discarding the in-memory change).
A declined commit is treated as a failed operation, because the rename or merge it
was supposed to persist no longer exists in memory either.

Residue on failure is therefore always the recoverable direction: the old objects
are live and still carry their rows, plus an orphaned row under the new key that a
retry overwrites. Retrying the operation is the recovery step.

All operations are available in both synchronous and asynchronous versions. Async versions have the prefix "a" (e.g., `acreate_entity`, `aedit_relation`).

* Insert Custom KG

```python
custom_kg = {
    "chunks": [
        {
            "content": "Alice and Bob are collaborating on quantum computing research.",
            "source_id": "doc-1",
            "file_path": "test_file",
        }
    ],
    "entities": [
        {
            "entity_name": "Alice",
            "entity_type": "person",
            "description": "Alice is a researcher specializing in quantum physics.",
            "source_id": "doc-1",
            "file_path": "test_file"
        },
        {
            "entity_name": "Bob",
            "entity_type": "person",
            "description": "Bob is a mathematician.",
            "source_id": "doc-1",
            "file_path": "test_file"
        },
        {
            "entity_name": "Quantum Computing",
            "entity_type": "technology",
            "description": "Quantum computing utilizes quantum mechanical phenomena for computation.",
            "source_id": "doc-1",
            "file_path": "test_file"
        }
    ],
    "relationships": [
        {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "description": "Alice and Bob are research partners.",
            "keywords": "collaboration research",
            "weight": 1.0,
            "source_id": "doc-1",
            "file_path": "test_file"
        },
        {
            "src_id": "Alice",
            "tgt_id": "Quantum Computing",
            "description": "Alice conducts research on quantum computing.",
            "keywords": "research expertise",
            "weight": 1.0,
            "source_id": "doc-1",
            "file_path": "test_file"
        },
        {
            "src_id": "Bob",
            "tgt_id": "Quantum Computing",
            "description": "Bob researches quantum computing.",
            "keywords": "research application",
            "weight": 1.0,
            "source_id": "doc-1",
            "file_path": "test_file"
        }
    ]
}

rag.insert_custom_kg(custom_kg)
```

* Other Entity and Relation Operations
  - **create_entity**: Creates a new entity with specified attributes
  - **edit_entity**: Updates an existing entity's attributes or renames it
  - **create_relation**: Creates a new relation between existing entities
  - **edit_relation**: Updates an existing relation's attributes

These operations maintain data consistency across both the graph database and vector database components.


## Delete Functions

LightRAG provides comprehensive deletion capabilities.

### Delete Entities

```python
# Synchronous
rag.delete_by_entity("Google")

# Asynchronous
await rag.adelete_by_entity("Google")
```

When deleting an entity:
- Removes the entity node from the knowledge graph
- Deletes all associated relationships
- Removes related embedding vectors from the vector database
- Deletes and persists the entity and incident-relation chunk-tracking rows, so recreating the entity does not inherit pre-deletion provenance
- Maintains knowledge graph integrity

A deletion is staged so that no failure can leave a live entity without its
authoritative provenance — the state from which a later document purge concludes
"no remaining sources" and removes an entity other documents still reference.
The graph object's removal is committed first, on its own; only then are the
tracking rows deleted and committed; the vector storages are flushed last. This
holds for any mix of backends, including a deferred graph with an immediate-write
tracking store, which is why the staging is by *durability* rather than by call
order.

That one-directional rule — **a graph object must never be durable while the
tracking row carrying its attribution is not** — governs the other admin paths
too, and it makes their commit order the mirror of the deletion order. On a
create or an edit the row is the half that starts out absent, so
`_persist_graph_updates` commits the tracking rows first and the graph and
vector stores second. A failure in the first phase skips the second entirely: a
tracking commit that did not land is never followed by publishing the object it
describes. Both directions therefore converge on the same tolerated residue, a
row whose object is not (or no longer) in the graph.

Callers writing directly against `lightrag.utils_graph` inherit that contract.
A helper that *removes* a tracking row must commit the graph itself first via
`_commit_graph_or_raise` and only then flush the tracking stores; passing a
graph store and a tracking store to `_persist_graph_updates` together is
correct only in the add/update direction.

Removing the object and cleaning up its rows is additionally one region a
cancellation cannot cut in half. It has to begin at the graph mutation: a cancel
before the commit leaves the removal in the in-memory graph with the backend
marked dirty, so the pipeline's next commit publishes it while the cleanup never
runs, and a cancel *during* the commit is deferred by the storage-IO layer until
the write and its notification hook have landed. Cancelling a deletion therefore
waits for the object's removal and its tracking cleanup to finish; only the
vector flush is skipped.

That protection covers the *caller*'s cancellation. Cancelling the deletion task
itself — which the event loop does to every remaining task at shutdown — is not
deferred, because the exception says nothing about whether the write had been
submitted, and assuming it had would delete the tracking rows of a node whose
removal never left memory. The cleanup therefore runs only once the commit has
demonstrably returned.

Every remaining failure state is therefore recoverable, and repeating the
deletion is always the recovery step:

| Failure point | On-disk result | Recovery |
| --- | --- | --- |
| Graph commit | Entity live, rows live | Consistent; retry the deletion |
| Tracking delete or commit | Entity gone, its row stale | Retry: a deletion reporting `not_found` sweeps a stale row for that name and flushes pending tracking state whether or not a row is still visible in memory |
| A failing tracking delete leaves incident relation rows of an entity deletion | Entity gone, relation rows stale | Not reachable automatically — the node is gone, so its edges are unknowable. Logged with the exact storage keys; delete the relation directly to sweep its row, or run the [chunk-tracking repair](#repairing-chunk-tracking) |
| The cleanup never runs although the graph commit landed — process exit before a **deferred** tracking backend (JSON) flushes, or a direct cancellation of the deletion task mid-write | Entity gone, its rows and its relations' rows stale | The entity's own row is swept by a repeated deletion; its incident relation rows are not, and their keys are not logged because nothing failed — so there is nothing to delete directly *by*. The recovery is the [chunk-tracking repair](#repairing-chunk-tracking). Not closed by this staging: neither a hard process exit nor a cancellation carries evidence about what landed. A commit notification that raises *after* the write does, and no longer reaches this row — it arrives as `CommitBookkeepingError` and `_commit_graph_or_raise` continues with the cleanup |
| Vector flush | Entity and rows gone, vector record stale | The rebuildable window this codebase accepts elsewhere; `lightrag-rebuild-vdb` restores it |

#### Concurrent admin writes

On a graph storage that declares `requires_single_writer` — `NetworkXStorage`,
the only one — every public admin graph writer (`acreate_entity`,
`acreate_relation`, `aedit_entity`, `aedit_relation`, `adelete_by_entity`,
`adelete_by_relation`, `amerge_entities`, `ainsert_custom_kg`, and therefore
every `/graph/*` mutation endpoint) runs inside `LightRAG._admin_write_gate`
(issue #3899), which serializes it in two directions for the whole
mutate-and-commit body, embedding round-trip included:

- **Against other admin writes**, through a workspace-wide admin lock
  (`{workspace}:GraphAdmin`, key `admin`; cross-process). A second admin write
  *queues* behind the first for up to `ADMIN_WRITE_LOCK_ACQUIRE_TIMEOUT`
  (default 30 s, `LIGHTRAG_ADMIN_WRITE_LOCK_ACQUIRE_TIMEOUT`) and is refused
  only on expiry. That timeout is the *only* bound on the wait — the
  multiprocess keyed lock polls with backoff and has no timeout of its own.
- **Against the document pipeline**, through the pipeline `busy` reservation
  (`kind="admin"`, never `destructive_busy`, so uploads stay allowed). While an
  admin write holds it, a pipeline start is *deferred*: the start is reduced to a
  sticky auto-rescan request in the workspace ingress mailbox, and the gate
  drives the queue once when it releases. A pipeline that is already running or
  scanning refuses the admin write, as before.

  That drive runs in the background on a long-lived loop (the API server), but is
  **awaited inline under the synchronous wrappers** (`create_entity`,
  `edit_relation`, …), because `run_until_complete` stops the loop as soon as the
  admin call returns and a background task there would park after taking the
  `busy` reservation, wedging the workspace under a live pid. A synchronous admin
  write therefore blocks until the queue is drained — only when a pipeline start
  was actually deferred during its hold.

  **The background drive needs an event loop that outlives the admin call**, so
  what an SDK caller of the `a*` methods gets depends on how the loop is driven.
  Nothing here risks data: the auto-rescan request is sticky, so a drive that
  does not run leaves it armed for the next scan or upload. What varies is
  whether the queue is drained now.

  | How the `a*` call is driven | What happens to the deferred drive |
  |---|---|
  | A loop that keeps running (an API server, any long-lived app) | Runs to completion. This is the case the background path is for. |
  | `asyncio.run(main())` with further `await`s after the edit | Runs to completion. |
  | `asyncio.run(main())` where the edit is the last step, or followed straight by `finalize_storages()` | Cancelled in flight. `finalize_storages` cancels pending drives and the pipeline's own cleanup releases `busy`, so the end state is clean and the request stays armed — the document simply waits for the next trigger. |
  | A hand-managed loop stopped and restarted with repeated `loop.run_until_complete(...)`, never finalized | The task advances only while the loop happens to run, so it can take the `busy` reservation and then park. Dead-owner reclaim cannot clear a live pid, so the workspace stays busy until the process exits. **Not a supported pattern** — use `asyncio.run`, or `await finalize_storages()` before going idle. |

  Note that `asyncio.run` **cancels** pending tasks on exit; it does not run them
  to completion. That is why the third row is a cancellation rather than a drain,
  and why the fourth row — which never reaches such a cancellation — is the only
  one that can strand the reservation.

The hold is bounded by `ADMIN_WRITE_MAX_HOLD_SECONDS`
(`LIGHTRAG_ADMIN_WRITE_MAX_HOLD_SECONDS`), which defaults to
`max(180, 6 × EMBEDDING_TIMEOUT)` — 180 s at the default embedding timeout. On
expiry the write fails with HTTP 500 and both gates release, so a hung embedding
endpoint cannot fence ingestion indefinitely.

The default is *derived* rather than fixed because the embedding round-trip runs
inside the hold: raising `EMBEDDING_TIMEOUT` alone would otherwise leave a
ceiling sized for the old value, killing every retry it was meant to allow. One
hold has to cover an embedding retry storm (`3 × EMBEDDING_TIMEOUT` plus 8 s of
backoff — 98 s at the default) plus one whole-graph GraphML commit (~17 s at
200k nodes, since `write_nx_graph` rewrites the entire graph however small the
edit was), so ~115 s; the remainder is headroom for the edges an edit touches.

The ceiling is resolved per `LightRAG` instance from that instance's
`default_embedding_timeout`, not from the environment at import, so a direct
`LightRAG(default_embedding_timeout=300)` is followed without any environment
variable. Setting it *below* the embedding timeout is refused at startup.
Prefer erring high: a ceiling that is
too high only defers ingestion, and a deferred start is sticky in the ingress
mailbox so it self-heals, whereas one that is too low kills edits whose commit
may already have landed.

These two knobs are **not** derived from each other. The ceiling asks how long
the worst legitimate write may run; the acquire timeout asks how long a caller
should wait before being told to retry. Deriving one from the other would make
them equal — which would park an interactive edit for the full worst case
instead of returning the actionable 409. When the edit ahead runs long, the
queue is *meant* to degrade to fast failure.

An acquire timeout *above* the ceiling is not wasted, either. The admin lock is
taken before the ceiling starts (the `pipeline_status` fetch and the reservation
acquire run inside the lock and outside the ceiling) and released after it ends,
and a cancellation-resistant commit runs to completion past the expiry — a 1 s
ceiling was measured holding the lock for 5.01 s. So the lock always outlives
the ceiling by an amount no static comparison can bound, and a queued write can
still be rewarded for waiting.

**A 500 from that ceiling does not mean the edit was undone.** The ceiling stops
the operation by cancelling it, and an admin write withholds a cancellation while
a storage commit is in flight, so a ceiling firing mid-commit lets that commit
land. A multi-step flow can also have committed at an earlier step: a merge
commits the merged node before it removes the source entities. The error message
says which of the two happened, and either way the caller must **re-read the
entity or relation before retrying** rather than assume the operation is undone.
Retrying blind can hit `Entity 'X' already exists` or re-apply an edit that is
already durable. The graph endpoints return that wording as the 500's `detail`
rather than the sanitized generic body, because a REST client is the caller that
has to act on it and does not read server logs. Every other failure on those
endpoints keeps the sanitized body.

Server-backed graph stores (Neo4j, PostgreSQL, Memgraph, MongoDB, OpenSearch)
never take the gate, whatever the KV or vector storage beside them: only the
graph storage can lose an uncommitted mutation to a peer commit
(`NanoVectorDBStorage` / `FaissVectorDBStorage` replay their pending buffers over
a reloaded snapshot, and `JsonKVStorage` has no reload path at all). There, two
admin writes for different keys still run concurrently under per-entity keyed
locks, as they always did.

**Two 409s, told apart by the `detail` text.** The graph endpoints return
HTTP 409 for two different reasons, and the retry semantics differ, so the
`detail` starts with a stable phrase for each (the WebUI surfaces the response
body verbatim, so no client change is needed):

| Leading phrase | Cause | Clears when |
|---|---|---|
| `Another knowledge graph edit is in progress` | the admin lock was held by a peer admin write past the acquire timeout | that peer admin write finishes — retry the same request |
| `Pipeline is busy with another operation` | the pipeline holds `busy` (a processing run or a destructive job) or `scanning` — from the router's early check or from the gate's reservation | ingestion or the scan finishes |

The router's early check refuses on `busy` **except when an admin write owns
it**. Without that exemption the second concurrent REST edit would be refused
before it ever reached the admin lock, so the queueing above would exist only
for direct SDK callers — and the client would be told to wait for document
ingestion when what is ahead of it is another UI edit. A `busy` holder that
cannot be identified is never exempt.

**Accepted residue: queueing can still end in a refusal.** Every admin write
drives the queue once when it releases, so with two of them in flight the first
one's drive races the second one's reservation for `busy`. In one process the
second write wins (it is woken by the lock release, while the drive is a task
created after it). Across workers it may not: the second write waits on the
lease lock with backoff, so the first one's drive can take `busy` first and the
second write gets the pipeline-busy 409 despite having queued. No data is at
risk — the write simply did not happen and the client retries — so this is
accepted rather than fixed; treat it as the queued request having missed a turn.

Why a lock rather than teaching the file backend to replay its pending work over
a reloaded snapshot: graph payloads are accumulate-over-read (`source_id` is an
evidence set merged from what the writer read; `weight` is floored by the
evidence count it read), so a replay over a peer's newer state would drop the
peer's evidence and republish a stale `weight`, silently violating the
[relation weight contract](#relation-weight-contract). It would exchange a loss
the reload fence can still see for one nothing can. The one part of that idea
that was kept is loud: `NetworkXStorage` records when a reload discards
uncommitted mutations and refuses the next commit in that process with
`GraphMutationsDiscardedError`, so a writer that bypasses the gate fails instead
of succeeding without its changes.

What remains, and why it is tolerated:

- Two admin requests a second apart, load-balanced to different workers, where a
  lost reload notification lets the second mutate a stale snapshot: the admin
  lock was released long before it was re-acquired, so it cannot help. That is
  the file-fingerprint fence's case (issue #3854): the stale writer declines its
  commit, the caller gets a 500, and retrying re-applies the edit against the
  peer's snapshot. Loud, and recoverable by the operator.
- A hard process exit can leave a tracking row whose graph object never became
  durable. That row is harmless to queries, cannot be inherited as evidence by a
  later object (the explicit creation paths reset attribution), and is removed by
  the [chunk-tracking repair](#repairing-chunk-tracking).
- The forbidden mirror — an object durable without the row that carries its
  attribution — is not produced by a single-writer crash **on the creation
  paths**, because there the tracking row is written *and committed before the
  graph mutation is issued at all*. Ordering only the flushes would not have
  been enough: on Neo4j or PostgreSQL the `upsert_node` is durable the moment it
  returns, and on NetworkX it is already in the process-wide in-memory graph,
  where the next flush by any co-tenant publishes it.
- An edit that *removes* evidence IDs from a row cannot use the creation order —
  a narrowed row landing ahead of the graph write is itself the over-deleting
  state. `aedit_relation` therefore stages such an edit as grow-then-shrink: the
  superset row, then the graph write, then the final row. If the last step
  fails, the edit is already durable and the row keeps naming a chunk the
  relation no longer cites — under-deletion, the accepted direction, repaired by
  the [chunk-tracking repair](#repairing-chunk-tracking). The accepted *state*
  does not make it a silent one: a failure of that last step raises
  `VectorStorageConsistencyError` (a 500 naming the row and the repair tool),
  because a retry cannot heal it — the second edit sees an unchanged `source_id`
  and skips the tracking update — so the operator is the only recovery path, and
  a 200 would guarantee they never learn to take it.
  Both ways that last step could be *skipped* rather than fail are closed
  (issue #3895): it runs *inside* a cancellation-deferring region alongside the
  graph commit, so a cancellation deferred through that commit cannot walk past
  it, and *before* the relation's vector work, so no vector failure can strand
  it. The reverse exposure is the acceptable one — a shrink failure skips the
  vector write, leaving records a re-issued edit rewrites and
  `lightrag-rebuild-vdb` restores, and the message names both. A declined graph
  commit still outranks a vector failure, because the commit is confirmed on its
  own before any vector call is made.
  **One residue stays open, deliberately, the same one the entity path carries
  below:** a cancellation — or an ordinary error, an acknowledgement lost after
  the fact carries the same ambiguity — delivered inside `upsert_edge`'s own
  await tears the edit down before the shrink, so the edge may be narrowed while
  the row keeps the superset. It cannot be deferred (the exception originates in
  that coroutine) and must not be settled blind (narrowing a row whose write the
  backend never accepted is the over-deleting mirror), so the row stays wide and
  the failure is *logged* with the row key and the repair tool.
- `aedit_entity`'s non-rename path stages its row the same way, for the same
  reason: a **growing** edit there used to commit the node before flushing the
  row that attributes it, leaving `rows ⊂ graph` — reachable from `POST
  /graph/entity/edit`, whose `updated_data` accepts `source_id`. The superset
  row is now durable before `upsert_node` is called at all, and the removals are
  applied after the graph commit. Its shrink is staged exactly like the relation
  one's: it runs *inside* the cancellation-deferring region and *before* the
  vector write, so neither a cancellation nor a vector failure can skip it. It
  either completes or raises with the row key and the repair tool named.
  **One residue there stays open, deliberately:** a cancellation delivered
  *inside* `upsert_node`'s own await — after an immediate-write backend accepted
  the row, before control returns — tears the edit down before the shrink, so
  the node is narrowed while the row keeps the superset. It cannot be deferred
  (the `CancelledError` originates in that coroutine, so there is nothing for
  `_finish_deferring_cancellation` to defer, and issuing the call from inside
  that region gives the identical residue), and it must not be settled blind:
  whether the backend accepted the write is unknowable there, and narrowing the
  row when it did not would leave `rows ⊂ graph` — over-deletion, which this
  ranking treats as losing data, traded against a residue that merely retains a
  chunk ID. So the row stays wide and the failure is *logged* with the row key
  and the repair tool, which is the part that was actually owed — for an
  ordinary backend error as much as for a cancellation, since an
  acknowledgement lost after the write was applied carries the same ambiguity
  and the caller's error says only that the write failed. Its
  **rename** path needs no such staging and deliberately keeps its own ordering:
  it writes a fresh node whose `source_id` already equals the row it migrates,
  and it retires the old key only after the commit that removes the old node
  (see the [merge and rename failure model](design/PurgeRecoveryContract.md#merge-and-rename-failure-model)).
- A graph backend that *declines* its commit (the NetworkX reload fence) raises
  out of the create, edit, merge and delete paths alike, so the caller sees a
  500 instead of a success for a write that was discarded.

A workspace-wide admin lock was specified and dropped: it would not have changed
what a crash can leave behind, since the same residue is reachable with no
concurrency at all. If you drive the public Python admin API yourself
(`acreate_entity`, `aedit_entity`, `amerge_entities`, `adelete_by_entity` and
their relation counterparts) **do not call them concurrently on a file-backed
workspace** — one at a time, or use a server-backed graph and KV store.
`ainsert_custom_kg` is subject to the same rule.

#### Repairing chunk tracking

A stale or orphaned `entity_chunks` / `relation_chunks` row cannot be found, let
alone pruned, one row at a time: `BaseKVStorage` has no enumeration API, so
nothing can sweep for it. The repair is therefore whole-namespace — it replaces
one or both namespaces from current graph keys, retaining authoritative rows for
live objects and supplementing them from cached extraction results. Because that
replacement cannot be coordinated with writers in other processes, it is
available only as an offline tool.

Before every run, stop **all** LightRAG API servers, pipeline workers, and SDK
writers that use the same backing stores and workspace. The default invocation
only scans and prints the complete replacement plan:

```bash
lightrag-repair-chunk-tracking
lightrag-repair-chunk-tracking --apply
lightrag-repair-chunk-tracking --apply --namespace entity  # or relation
lightrag-repair-chunk-tracking --apply --resume-plan /path/from/failed/run.sqlite3
# equivalent: python -m lightrag.tools.chunk_tracking_repair [--apply]
```

The repair scans document status and graph objects in bounded batches. Its
deduplication and replacement plan live in a disk-backed SQLite database, so
client memory is bounded by a batch plus the largest individual tracking row;
local disk usage grows with the complete plan. Process-buffered KV backends are
flushed after each repair batch; pending operations fail the apply instead of
being counted as completed. Dry-run plans are temporary, while apply plans remain
available for recovery until success.

An apply durably seals that SQLite plan before the first namespace drop. If the
apply fails or the process is interrupted, keep the workspace offline and use
the printed `--resume-plan` path. Resume validates the configured storage
identity and rewrites from the pre-drop snapshot without reading the partial
tracking namespace. The plan is deleted only after all selected namespaces have
been rebuilt successfully.

The tool asks for an offline confirmation before initializing storage and asks
again before the destructive apply. `--yes` is intended for an already-isolated
maintenance environment. It prints the configured working directory, workspace,
and concrete storage classes before planning. See
[`README_CHUNK_TRACKING_REPAIR.md`](../lightrag/tools/README_CHUNK_TRACKING_REPAIR.md)
for configuration and recovery instructions.

It is deliberately **not** the startup migration:

|                | startup chunk-tracking migration | offline repair tool |
| --- | --- | --- |
| When           | startup / first explicit creation | operator, on demand |
| Gate           | only when the namespace `is_empty()` | never gated |
| Seed           | graph `source_id` | live-object tracking rows + cached extraction results |
| Existing rows  | left untouched | current graph keys retained; orphan keys removed |

The seed is the point. Graph `source_id` is KEEP-truncated and chunk tracking
outranks it, so re-seeding from it downgrades provenance across the whole
install. The repair never reads it; the graph is consulted only for current
object keys, so rows for deleted objects are not copied into the replacement.
Rows for live objects remain authoritative: rename, merge, and manual creation
can produce keys or attribution the extraction cache cannot reproduce. Cached
extraction (`text_chunks.llm_cache_list` → `llm_response_cache`) supplements
those rows at chunk granularity. The `full_entities` / `full_relations` anchors
remain too coarse to write a tracking row from.

Two consequences an operator has to plan for, both reported in the plan:

- An object with neither an existing authoritative row nor matching cached
  extraction remains without a row and is reported. An existing row—including
  an authoritative empty row—is never discarded merely because cache evidence
  is absent.
- If retained rows plus cached evidence would leave a namespace **empty while
  the graph contains corresponding objects**, apply fails before the first drop.
- Any plan that leaves a current graph object without a row is blocked by
  default. `--allow-missing-rows` accepts that explicitly after review of the
  existing/planned row denominators printed by the dry run.
- A completely empty graph also blocks apply by default: it may mean the wrong
  backend/workspace or an unavailable graph index. `--allow-empty-graph` is an
  explicit override after the operator independently verifies the empty graph.

The tool computes the whole mapping before the first `drop()`, so a read failure
leaves every existing row untouched. Each selected namespace is dropped and
fully rewritten before the next namespace is touched, reducing the partial
failure window. If an apply still fails after a drop, keep every writer stopped,
fix the cause, and re-run the tool until it completes.

### Delete Relations

```python
# Synchronous
rag.delete_by_relation("Google", "Gmail")

# Asynchronous
await rag.adelete_by_relation("Google", "Gmail")
```

When deleting a relationship:
- Removes the specified relationship edge
- Deletes the relationship's embedding vector
- Deletes and persists its chunk-tracking row regardless of endpoint order, so recreating the relation starts with new provenance
- Preserves both entity nodes and their other relationships

Relation deletion is staged exactly as entity deletion is (see the table above),
and repeating a deletion that reports `not_found` sweeps a stale row and commits
tracking state an earlier attempt left pending.

### Delete by Document ID

```python
# Asynchronous only (complex reconstruction process)
await rag.adelete_by_doc_id("doc-12345")
```

The deletion process:
1. Delete all text chunks related to the document
2. Identify and delete entities/relationships that belong only to this document
3. Rebuild entities/relationships that still exist in other documents
4. Update all related vector indexes
5. Clean up document status records

**Important Reminders:**
1. All deletion operations are **irreversible** — use with caution
2. Deleting large amounts of data may take time, especially deletion by document ID
3. Deletion operations automatically maintain consistency between the graph and vector databases
4. Consider backing up data before performing important deletions


## Entity Merging

**Merge Entities and Their Relationships**

```python
# Basic merge
rag.merge_entities(
    source_entities=["Artificial Intelligence", "AI", "Machine Intelligence"],
    target_entity="AI Technology"
)

# With custom merge strategy
rag.merge_entities(
    source_entities=["John Smith", "Dr. Smith", "J. Smith"],
    target_entity="John Smith",
    merge_strategy={
        "description": "concatenate",  # Combine all descriptions
        "entity_type": "keep_first",   # Keep the type from the first entity
        "source_id": "join_unique"     # Combine all unique source IDs
    }
)

# With custom target entity data
rag.merge_entities(
    source_entities=["New York", "NYC", "Big Apple"],
    target_entity="New York City",
    target_entity_data={
        "entity_type": "LOCATION",
        "description": "New York City is the most populous city in the United States.",
    }
)

# Advanced: combining both strategy and custom data
rag.merge_entities(
    source_entities=["Microsoft Corp", "Microsoft Corporation", "MSFT"],
    target_entity="Microsoft",
    merge_strategy={
        "description": "concatenate",
        "source_id": "join_unique"
    },
    target_entity_data={
        "entity_type": "ORGANIZATION",
    }
)
```

When merging entities:
- All relationships from source entities are redirected to the target entity
- Duplicate relationships are intelligently merged
- Self-relationships (loops) are prevented
- Source entities are removed after merging
- Relationship attributes are merged, and each resulting weight is the larger
  of every input weight and the distinct merged real-source count


## Troubleshooting

### Common Initialization Errors

1. **`AttributeError: __aenter__`**
   - **Cause**: Storage backends not initialized
   - **Solution**: Call `await rag.initialize_storages()` after creating the LightRAG instance

2. **`KeyError: 'history_messages'`**
   - **Cause**: Pipeline status not initialized
   - **Solution**: Call `await rag.initialize_storages()` after creating the LightRAG instance

3. **Both errors in sequence**
   - **Solution**: Always follow this pattern:
   ```python
   rag = LightRAG(...)
   await rag.initialize_storages()
   ```

### Model Switching Issues

When switching between different embedding models, you must clear the data directory to avoid errors. The only file you may want to preserve is `kv_store_llm_response_cache.json` if you wish to retain the LLM cache.
