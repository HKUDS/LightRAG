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
| **working_dir** | `str` | Directory where the cache will be stored | `lightrag_cache+timestamp` |
| **workspace** | str | Workspace name for data isolation between different LightRAG Instances | |
| **kv_storage** | `str` | Storage type for documents and text chunks. Supported types: `JsonKVStorage`,`PGKVStorage`,`RedisKVStorage`,`MongoKVStorage`,`OpenSearchKVStorage` | `JsonKVStorage` |
| **vector_storage** | `str` | Storage type for embedding vectors. Supported types: `NanoVectorDBStorage`,`PGVectorStorage`,`MilvusVectorDBStorage`,`ChromaVectorDBStorage`,`FaissVectorDBStorage`,`MongoVectorDBStorage`,`QdrantVectorDBStorage`,`OpenSearchVectorDBStorage` | `NanoVectorDBStorage` |
| **graph_storage** | `str` | Storage type for graph edges and nodes. Supported types: `NetworkXStorage`,`Neo4JStorage`,`PGGraphStorage`,`AGEStorage`,`OpenSearchGraphStorage` | `NetworkXStorage` |
| **doc_status_storage** | `str` | Storage type for documents process status. Supported types: `JsonDocStatusStorage`,`PGDocStatusStorage`,`MongoDocStatusStorage`,`OpenSearchDocStatusStorage` | `JsonDocStatusStorage` |
| **chunk_token_size** | `int` | Maximum token size per chunk when splitting documents | `1200` |
| **chunk_overlap_token_size** | `int` | Overlap token size between two chunks when splitting documents | `100` |
| **tokenizer** | `Tokenizer` | The function used to convert text into tokens (numbers) and back using .encode() and .decode() functions following `TokenizerInterface` protocol. If you don't specify one, it will use the default Tiktoken tokenizer. | `TiktokenTokenizer` |
| **tiktoken_model_name** | `str` | If you're using the default Tiktoken tokenizer, this is the name of the specific Tiktoken model to use. This setting is ignored if you provide your own tokenizer. | `gpt-4o-mini` |
| **entity_extract_max_gleaning** | `int` | Number of loops in the entity extraction process, appending history messages | `1` |
| **node_embedding_algorithm** | `str` | Algorithm for node embedding (currently not used) | `node2vec` |
| **node2vec_params** | `dict` | Parameters for node embedding | `{"dimensions": 1536,"num_walks": 10,"walk_length": 40,"window_size": 2,"iterations": 3,"random_seed": 3,}` |
| **embedding_func** | `EmbeddingFunc` | Function to generate embedding vectors from text | `openai_embed` |
| **embedding_batch_num** | `int` | Maximum batch size for embedding processes (multiple texts sent per batch) | `32` |
| **embedding_func_max_async** | `int` | Maximum number of concurrent asynchronous embedding processes | `16` |
| **llm_model_func** | `callable` | Function for LLM generation | `gpt_4o_mini_complete` |
| **llm_model_name** | `str` | LLM model name for generation | `meta-llama/Llama-3.2-1B-Instruct` |
| **summary_context_size** | `int` | Maximum tokens send to LLM to generate summaries for entity relation merging | `10000`（configured by env var SUMMARY_CONTEXT_SIZE) |
| **summary_max_tokens** | `int` | Maximum token size for entity/relation description | `500`（configured by env var SUMMARY_MAX_TOKENS) |
| **llm_model_max_async** | `int` | Maximum number of concurrent asynchronous LLM processes | `4`（default value changed by env var MAX_ASYNC) |
| **llm_model_kwargs** | `dict` | Additional parameters for LLM generation | |
| **vector_db_storage_cls_kwargs** | `dict` | Additional parameters for vector database, like setting the threshold for nodes and relations retrieval | cosine_better_than_threshold: 0.2（default value changed by env var COSINE_THRESHOLD) |
| **enable_llm_cache** | `bool` | If `TRUE`, stores LLM results in cache; repeated prompts return cached responses | `TRUE` |
| **enable_llm_cache_for_entity_extract** | `bool` | If `TRUE`, stores LLM results in cache for entity extraction; Good for beginners to debug your application | `TRUE` |
| **addon_params** | `dict` | Extraction output language and external entity-type prompt file | `{"language": "English", "entity_type_prompt_file": ""}` |
| **embedding_cache_config** | `dict` | Configuration for question-answer caching. Contains three parameters: `enabled`: Boolean value to enable/disable cache lookup functionality. When enabled, the system will check cached responses before generating new answers. `similarity_threshold`: Float value (0-1), similarity threshold. When a new question's similarity with a cached question exceeds this threshold, the cached answer will be returned directly without calling the LLM. `use_llm_check`: Boolean value to enable/disable LLM similarity verification. When enabled, LLM will be used as a secondary check to verify the similarity between questions before returning cached answers. | Default: `{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}` |


## QueryParam

Use `QueryParam` to control the behavior of your query:

```python
class QueryParam:
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "global"
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

    model_func: Callable[..., object] | None = None
    """Optional override for the LLM model function to use for this specific query.
    If provided, this will be used instead of the global model function.
    This allows using different models for different query modes.
    """

    user_prompt: str | None = None
    """User-provided prompt for the query.
    Addition instructions for LLM. If provided, this will be inject into the prompt template.
    It's purpose is the let user customize the way LLM generate the response.
    """

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
- [Direct OpenAI Example](examples/unofficial-sample/lightrag_llamaindex_direct_demo.py)
- [LiteLLM Proxy Example](examples/unofficial-sample/lightrag_llamaindex_litellm_demo.py)
- [LiteLLM Proxy with Opik Example](examples/unofficial-sample/lightrag_llamaindex_litellm_opik_demo.py)

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

### Rerank Function Injection

To enhance retrieval quality, documents can be re-ranked based on a more effective relevance scoring model. The `rerank.py` file provides three Reranker provider driver functions:

- **Cohere / vLLM**: `cohere_rerank`
- **Jina AI**: `jina_rerank`
- **Aliyun**: `ali_rerank`

Inject one of these functions into the `rerank_model_func` attribute of the LightRAG object. For detailed usage, refer to `examples/rerank_example.py`.

### User Prompt vs. Query

When using LightRAG for content queries, avoid combining the search process with unrelated output processing, as this significantly impacts query effectiveness. The `user_prompt` parameter in `QueryParam` does not participate in the RAG retrieval phase — it guides the LLM on how to process the retrieved results after the query is completed.

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


## Storage Backends

### Sotrage Types

LightRAG uses 4 types of storage for different purposes:

| Storage Type | Purpose |
|---|---|
| **KV_STORAGE** | LLM response cache, text chunks, document information |
| **VECTOR_STORAGE** | Entity/relation/chunk embedding vectors |
| **GRAPH_STORAGE** | Entity-relation graph structure |
| **DOC_STATUS_STORAGE** | Document indexing status |

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
MemgraphStorage          Memgraph
OpenSearchGraphStorage   OpenSearch
```

> Testing has shown that Neo4J delivers superior performance in production environments compared to PostgreSQL with AGE plugin.

**VECTOR_STORAGE**
```
NanoVectorDBStorage         NanoVector (default)
PGVectorStorage             Postgres
MilvusVectorDBStorage       Milvus
FaissVectorDBStorage        Faiss
QdrantVectorDBStorage       Qdrant
MongoVectorDBStorage        MongoDB
OpenSearchVectorDBStorage   OpenSearch
```

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
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
export NEO4J_DATABASE="neo4j"  # Required for community edition
```

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

PostgreSQL can provide a one-stop solution as KV store, VectorDB (pgvector), and GraphDB (apache AGE). PostgreSQL version 16.6 or higher is supported.

- PostgreSQL is lightweight; the whole binary distribution including all necessary plugins can be zipped to 40MB: Ref to [Windows Release](https://github.com/ShanGor/apache-age-windows/releases/tag/PG17%2Fv1.5.0-rc0) as it is easy to install for Linux/Mac.
- If you prefer Docker, start with this image to avoid hiccups (Default user password: rag/rag): https://hub.docker.com/r/gzdaniel/postgres-for-rag
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
python -m pytest tests/test_opensearch_storage.py -v
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
| `PGKVStorage`, `PGVectorStorage`, `PGDocStatusStorage` | `workspace` field in tables |
| `Neo4JStorage` | Labels |
| `OpenSearch*` | Index name prefixes |

**Legacy compatibility**: Default workspace for PostgreSQL non-graph storage is `default`; for PostgreSQL AGE graph storage is null; for Neo4j graph storage is `base`.

Storage-specific workspace environment variables override the common `WORKSPACE` variable: `REDIS_WORKSPACE`, `MILVUS_WORKSPACE`, `QDRANT_WORKSPACE`, `MONGODB_WORKSPACE`, `POSTGRES_WORKSPACE`, `NEO4J_WORKSPACE`, `OPENSEARCH_WORKSPACE`.

For a practical demonstration of managing multiple isolated knowledge bases, see [Workspace Demo](examples/lightrag_gemini_workspace_demo.py).


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

The `max_parallel_insert` parameter determines the number of documents processed concurrently. Default is **2**. Recommended to keep **below 10**, as the bottleneck typically lies with the LLM.

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
    "weight": 2.0
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
- Maintains knowledge graph integrity

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
- Preserves both entity nodes and their other relationships

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
- Relationship weights and attributes are preserved


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
