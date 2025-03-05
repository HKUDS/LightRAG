<center><h2>🚀 LightRAG: Simple and Fast Retrieval-Augmented Generation</h2></center>
<div align="center">
<table border="0" width="100%">
<tr>
<td width="100" align="center">
<img src="./assets/logo.png" width="80" height="80" alt="lightrag">
</td>
<td>
<div>
    <p>
        <a href='https://lightrag.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
        <a href='https://youtu.be/oageL-1I0GE'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
        <a href='https://arxiv.org/abs/2410.05779'><img src='https://img.shields.io/badge/arXiv-2410.05779-b31b1b'></a>
        <a href='https://learnopencv.com/lightrag'><img src='https://img.shields.io/badge/LearnOpenCV-blue'></a>
    </p>
    <p>
        <img src='https://img.shields.io/github/stars/hkuds/lightrag?color=green&style=social' />
        <img src="https://img.shields.io/badge/python-3.10-blue">
        <a href="https://pypi.org/project/lightrag-hku/"><img src="https://img.shields.io/pypi/v/lightrag-hku.svg"></a>
        <a href="https://pepy.tech/project/lightrag-hku"><img src="https://static.pepy.tech/badge/lightrag-hku/month"></a>
    </p>
    <p>
        <a href='https://discord.gg/yF2MmDJyGJ'><img src='https://discordapp.com/api/guilds/1296348098003734629/widget.png?style=shield'></a>
        <a href='https://github.com/HKUDS/LightRAG/issues/285'><img src='https://img.shields.io/badge/群聊-wechat-green'></a>
    </p>
</div>
</td>
</tr>
</table>

<div align="center">
This repository hosts the code of LightRAG. The structure of this code is based on <a href="https://github.com/gusye1234/nano-graphrag">nano-graphrag</a>.

<img src="https://i-blog.csdnimg.cn/direct/b2aaf634151b4706892693ffb43d9093.png" width="800" alt="LightRAG Diagram">
</div>
</div>
</br>


<details>
  <summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; display: list-item;">
    🎉 News
  </summary>


- [x] [2025.02.05]🎯📢Our team has released [VideoRAG](https://github.com/HKUDS/VideoRAG) understanding extremely long-context videos.
- [x] [2025.01.13]🎯📢Our team has released [MiniRAG](https://github.com/HKUDS/MiniRAG) making RAG simpler with small models.
- [x] [2025.01.06]🎯📢You can now [use PostgreSQL for Storage](#using-postgresql-for-storage).
- [x] [2024.12.31]🎯📢LightRAG now supports [deletion by document ID](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#delete).
- [x] [2024.11.25]🎯📢LightRAG now supports seamless integration of [custom knowledge graphs](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#insert-custom-kg), empowering users to enhance the system with their own domain expertise.
- [x] [2024.11.19]🎯📢A comprehensive guide to LightRAG is now available on [LearnOpenCV](https://learnopencv.com/lightrag). Many thanks to the blog author.
- [x] [2024.11.12]🎯📢LightRAG now supports [Oracle Database 23ai for all storage types (KV, vector, and graph)](https://github.com/HKUDS/LightRAG/blob/main/examples/lightrag_oracle_demo.py).
- [x] [2024.11.11]🎯📢LightRAG now supports [deleting entities by their names](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#delete).
- [x] [2024.11.09]🎯📢Introducing the [LightRAG Gui](https://lightrag-gui.streamlit.app), which allows you to insert, query, visualize, and download LightRAG knowledge.
- [x] [2024.11.04]🎯📢You can now [use Neo4J for Storage](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#using-neo4j-for-storage).
- [x] [2024.10.29]🎯📢LightRAG now supports multiple file types, including PDF, DOC, PPT, and CSV via `textract`.
- [x] [2024.10.20]🎯📢We've added a new feature to LightRAG: Graph Visualization.
- [x] [2024.10.18]🎯📢We've added a link to a [LightRAG Introduction Video](https://youtu.be/oageL-1I0GE). Thanks to the author!
- [x] [2024.10.17]🎯📢We have created a [Discord channel](https://discord.gg/yF2MmDJyGJ)! Welcome to join for sharing and discussions! 🎉🎉
- [x] [2024.10.16]🎯📢LightRAG now supports [Ollama models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!
- [x] [2024.10.15]🎯📢LightRAG now supports [Hugging Face models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!

</details>

<details>
  <summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; display: list-item;">
    Algorithm Flowchart
  </summary>

![LightRAG Indexing Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)
*Figure 1: LightRAG Indexing Flowchart - Img Caption : [Source](https://learnopencv.com/lightrag/)*
![LightRAG Retrieval and Querying Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)
*Figure 2: LightRAG Retrieval and Querying Flowchart - Img Caption : [Source](https://learnopencv.com/lightrag/)*

</details>

## Install

* Install from source (Recommend)

```bash
cd LightRAG
pip install -e .
```
* Install from PyPI
```bash
pip install lightrag-hku
```

## Quick Start
* [Video demo](https://www.youtube.com/watch?v=g21royNJ4fw) of running LightRAG locally.
* All the code can be found in the `examples`.
* Set OpenAI API key in environment if using OpenAI models: `export OPENAI_API_KEY="sk-...".`
* Download the demo text "A Christmas Carol by Charles Dickens":
```bash
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```

## Query

Use the below Python snippet (in a script) to initialize LightRAG and perform queries:

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

async def initialize_rag():
    rag = LightRAG(
        working_dir="your/path",
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    # Insert text
    rag.insert("Your text")

    # Perform naive search
    mode="naive"
    # Perform local search
    mode="local"
    # Perform global search
    mode="global"
    # Perform hybrid search
    mode="hybrid"
    # Mix mode Integrates knowledge graph and vector retrieval.
    mode="mix"

    rag.query(
        "What are the top themes in this story?",
        param=QueryParam(mode=mode)
    )

if __name__ == "__main__":
    main()
```

### Query Param

```python
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global"
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    - "naive": Performs a basic search without advanced techniques.
    - "mix": Integrates knowledge graph and vector retrieval. Mix mode combines knowledge graph and vector search:
        - Uses both structured (KG) and unstructured (vector) information
        - Provides comprehensive answers by analyzing relationships and context
        - Supports image content through HTML img tags
        - Allows control over retrieval depth via top_k parameter
    """
    only_need_context: bool = False
    """If True, only returns the retrieved context without generating a response."""
    response_type: str = "Multiple Paragraphs"
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""
    top_k: int = 60
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""
    max_token_for_text_unit: int = 4000
    """Maximum number of tokens allowed for each retrieved text chunk."""
    max_token_for_global_context: int = 4000
    """Maximum number of tokens allocated for relationship descriptions in global retrieval."""
    max_token_for_local_context: int = 4000
    """Maximum number of tokens allocated for entity descriptions in local retrieval."""
    ...
```

> default value of Top_k can be change by environment  variables  TOP_K.

<details>
<summary> <b>Using Open AI-like APIs</b> </summary>

* LightRAG also supports Open AI-like chat/embeddings APIs:
```python
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

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar"
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4096,
            max_token_size=8192,
            func=embedding_func
        )
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag
```
</details>

<details>
<summary> <b>Using Hugging Face Models</b> </summary>

* If you want to use Hugging Face models, you only need to set LightRAG as follows:

See `lightrag_hf_demo.py`

```python
# Initialize LightRAG with Hugging Face model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
    llm_model_name='meta-llama/Llama-3.1-8B-Instruct',  # Model name from Hugging Face
    # Use Hugging Face embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        )
    ),
)
```
</details>

<details>
<summary> <b>Using Ollama Models</b> </summary>

### Overview
If you want to use Ollama models, you need to pull model you plan to use and embedding model, for example `nomic-embed-text`.

Then you only need to set LightRAG as follows:

```python
# Initialize LightRAG with Ollama model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    llm_model_name='your_model_name', # Your model name
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```

### Increasing context size
In order for LightRAG to work context should be at least 32k tokens. By default Ollama models have context size of 8k. You can achieve this using one of two ways:

#### Increasing the `num_ctx` parameter in Modelfile.

1. Pull the model:
```bash
ollama pull qwen2
```

2. Display the model file:
```bash
ollama show --modelfile qwen2 > Modelfile
```

3. Edit the Modelfile by adding the following line:
```bash
PARAMETER num_ctx 32768
```

4. Create the modified model:
```bash
ollama create -f Modelfile qwen2m
```

#### Setup `num_ctx` via Ollama API.
Tiy can use `llm_model_kwargs` param to configure ollama:

```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    llm_model_name='your_model_name', # Your model name
    llm_model_kwargs={"options": {"num_ctx": 32768}},
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```
#### Low RAM GPUs

In order to run this experiment on low RAM GPU you should select small model and tune context window (increasing context increase memory consumption). For example, running this ollama example on repurposed mining GPU with 6Gb of RAM required to set context size to 26k while using `gemma2:2b`. It was able to find 197 entities and 19 relations on `book.txt`.

</details>
<details>
<summary> <b>LlamaIndex</b> </summary>

LightRAG supports integration with LlamaIndex.

1. **LlamaIndex** (`llm/llama_index_impl.py`):
   - Integrates with OpenAI and other providers through LlamaIndex
   - See [LlamaIndex Documentation](lightrag/llm/Readme.md) for detailed setup and examples

### Example Usage

```python
# Using LlamaIndex with direct OpenAI access
import asyncio
from lightrag import LightRAG
from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup log handler for LightRAG
setup_logger("lightrag", level="INFO")

async def initialize_rag():
    rag = LightRAG(
        working_dir="your/path",
        llm_model_func=llama_index_complete_if_cache,  # LlamaIndex-compatible completion function
        embedding_func=EmbeddingFunc(    # LlamaIndex-compatible embedding function
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: llama_index_embed(texts, embed_model=embed_model)
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Perform naive search
    print(
        rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
    )

    # Perform local search
    print(
        rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
    )

    # Perform global search
    print(
        rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
    )

    # Perform hybrid search
    print(
        rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
    )

if __name__ == "__main__":
    main()
```

#### For detailed documentation and examples, see:
- [LlamaIndex Documentation](lightrag/llm/Readme.md)
- [Direct OpenAI Example](examples/lightrag_llamaindex_direct_demo.py)
- [LiteLLM Proxy Example](examples/lightrag_llamaindex_litellm_demo.py)

</details>
<details>
<summary> <b>Conversation History Support</b> </summary>

LightRAG now supports multi-turn dialogue through the conversation history feature. Here's how to use it:

```python
# Create conversation history
conversation_history = [
    {"role": "user", "content": "What is the main character's attitude towards Christmas?"},
    {"role": "assistant", "content": "At the beginning of the story, Ebenezer Scrooge has a very negative attitude towards Christmas..."},
    {"role": "user", "content": "How does his attitude change?"}
]

# Create query parameters with conversation history
query_param = QueryParam(
    mode="mix",  # or any other mode: "local", "global", "hybrid"
    conversation_history=conversation_history,  # Add the conversation history
    history_turns=3  # Number of recent conversation turns to consider
)

# Make a query that takes into account the conversation history
response = rag.query(
    "What causes this change in his character?",
    param=query_param
)
```

</details>

<details>
<summary> <b>Custom Prompt Support</b> </summary>

LightRAG now supports custom prompts for fine-tuned control over the system's behavior. Here's how to use it:

```python
# Create query parameters
query_param = QueryParam(
    mode="hybrid",  # or other mode: "local", "global", "hybrid", "mix" and "naive"
)

# Example 1: Using the default system prompt
response_default = rag.query(
    "What are the primary benefits of renewable energy?",
    param=query_param
)
print(response_default)

# Example 2: Using a custom prompt
custom_prompt = """
You are an expert assistant in environmental science. Provide detailed and structured answers with examples.
---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
"""
response_custom = rag.query(
    "What are the primary benefits of renewable energy?",
    param=query_param,
    system_prompt=custom_prompt  # Pass the custom prompt
)
print(response_custom)
```

</details>

<details>
<summary> <b>Separate Keyword Extraction</b> </summary>

We've introduced a new function `query_with_separate_keyword_extraction` to enhance the keyword extraction capabilities. This function separates the keyword extraction process from the user's prompt, focusing solely on the query to improve the relevance of extracted keywords.

##### How It Works?
The function operates by dividing the input into two parts:
- `User Query`
- `Prompt`

It then performs keyword extraction exclusively on the `user query`. This separation ensures that the extraction process is focused and relevant, unaffected by any additional language in the `prompt`. It also allows the `prompt` to serve purely for response formatting, maintaining the intent and clarity of the user's original question.

##### Usage Example
This `example` shows how to tailor the function for educational content, focusing on detailed explanations for older students.

```python
rag.query_with_separate_keyword_extraction(
    query="Explain the law of gravity",
    prompt="Provide a detailed explanation suitable for high school students studying physics.",
    param=QueryParam(mode="hybrid")
)
```

</details>

<details>
<summary> <b>Insert Custom KG</b> </summary>

```python
custom_kg = {
    "chunks": [
        {
            "content": "Alice and Bob are collaborating on quantum computing research.",
            "source_id": "doc-1"
        }
    ],
    "entities": [
        {
            "entity_name": "Alice",
            "entity_type": "person",
            "description": "Alice is a researcher specializing in quantum physics.",
            "source_id": "doc-1"
        },
        {
            "entity_name": "Bob",
            "entity_type": "person",
            "description": "Bob is a mathematician.",
            "source_id": "doc-1"
        },
        {
            "entity_name": "Quantum Computing",
            "entity_type": "technology",
            "description": "Quantum computing utilizes quantum mechanical phenomena for computation.",
            "source_id": "doc-1"
        }
    ],
    "relationships": [
        {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "description": "Alice and Bob are research partners.",
            "keywords": "collaboration research",
            "weight": 1.0,
            "source_id": "doc-1"
        },
        {
            "src_id": "Alice",
            "tgt_id": "Quantum Computing",
            "description": "Alice conducts research on quantum computing.",
            "keywords": "research expertise",
            "weight": 1.0,
            "source_id": "doc-1"
        },
        {
            "src_id": "Bob",
            "tgt_id": "Quantum Computing",
            "description": "Bob researches quantum computing.",
            "keywords": "research application",
            "weight": 1.0,
            "source_id": "doc-1"
        }
    ]
}

rag.insert_custom_kg(custom_kg)
```
</details>

## Insert

#### Basic Insert

```python
# Basic Insert
rag.insert("Text")
```

<details>
  <summary> <b> Batch Insert </b></summary>

```python
# Basic Batch Insert: Insert multiple texts at once
rag.insert(["TEXT1", "TEXT2",...])

# Batch Insert with custom batch size configuration
rag = LightRAG(
    working_dir=WORKING_DIR,
    addon_params={
        "insert_batch_size": 20  # Process 20 documents per batch
    }
)

rag.insert(["TEXT1", "TEXT2", "TEXT3", ...])  # Documents will be processed in batches of 20
```

The `insert_batch_size` parameter in `addon_params` controls how many documents are processed in each batch during insertion. This is useful for:
- Managing memory usage with large document collections
- Optimizing processing speed
- Providing better progress tracking
- Default value is 10 if not specified

</details>

<details>
  <summary> <b> Insert with ID </b></summary>

If you want to provide your own IDs for your documents, number of documents and number of IDs must be the same.

```python
# Insert single text, and provide ID for it
rag.insert("TEXT1", ids=["ID_FOR_TEXT1"])

# Insert multiple texts, and provide IDs for them
rag.insert(["TEXT1", "TEXT2",...], ids=["ID_FOR_TEXT1", "ID_FOR_TEXT2"])
```

</details>

<details>
  <summary><b>Insert using Pipeline</b></summary>

The `apipeline_enqueue_documents` and `apipeline_process_enqueue_documents` functions allow you to perform incremental insertion of documents into the graph.

This is useful for scenarios where you want to process documents in the background while still allowing the main thread to continue executing.

And using a routine to process news documents.

```python
rag = LightRAG(..)

await rag.apipeline_enqueue_documents(input)
# Your routine in loop
await rag.apipeline_process_enqueue_documents(input)
```

</details>

<details>
  <summary><b>Insert Multi-file Type Support</b></summary>

The `textract` supports reading file types such as TXT, DOCX, PPTX, CSV, and PDF.

```python
import textract

file_path = 'TEXT.pdf'
text_content = textract.process(file_path)

rag.insert(text_content.decode('utf-8'))
```
</details>

## Storage

<details>
<summary> <b>Using Neo4J for Storage</b> </summary>

* For production level scenarios you will most likely want to leverage an enterprise solution
* for KG storage. Running Neo4J in Docker is recommended for seamless local testing.
* See: https://hub.docker.com/_/neo4j

```python
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

# Setup logger for LightRAG
setup_logger("lightrag", level="INFO")

# When you launch the project be sure to override the default KG: NetworkX
# by specifying kg="Neo4JStorage".

# Note: Default settings use NetworkX
# Initialize LightRAG with Neo4J implementation.
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
        graph_storage="Neo4JStorage", #<-----------override KG default
    )

    # Initialize database connections
    await rag.initialize_storages()
    # Initialize pipeline status for document processing
    await initialize_pipeline_status()

    return rag
```
see test_neo4j.py for a working example.

</details>

<details>
<summary> <b>Using PostgreSQL for Storage</b> </summary>

For production level scenarios you will most likely want to leverage an enterprise solution. PostgreSQL can provide a one-stop solution for you as KV store, VectorDB (pgvector) and GraphDB (apache AGE).
* PostgreSQL is lightweight,the whole binary distribution including all necessary plugins can be zipped to 40MB: Ref to [Windows Release](https://github.com/ShanGor/apache-age-windows/releases/tag/PG17%2Fv1.5.0-rc0) as it is easy to install for Linux/Mac.
* If you prefer docker, please start with this image if you are a beginner to avoid hiccups (DO read the overview): https://hub.docker.com/r/shangor/postgres-for-rag
* How to start? Ref to: [examples/lightrag_zhipu_postgres_demo.py](https://github.com/HKUDS/LightRAG/blob/main/examples/lightrag_zhipu_postgres_demo.py)
* Create index for AGE example: (Change below `dickens` to your graph name if necessary)
  ```sql
  load 'age';
  SET search_path = ag_catalog, "$user", public;
  CREATE INDEX CONCURRENTLY entity_p_idx ON dickens."Entity" (id);
  CREATE INDEX CONCURRENTLY vertex_p_idx ON dickens."_ag_label_vertex" (id);
  CREATE INDEX CONCURRENTLY directed_p_idx ON dickens."DIRECTED" (id);
  CREATE INDEX CONCURRENTLY directed_eid_idx ON dickens."DIRECTED" (end_id);
  CREATE INDEX CONCURRENTLY directed_sid_idx ON dickens."DIRECTED" (start_id);
  CREATE INDEX CONCURRENTLY directed_seid_idx ON dickens."DIRECTED" (start_id,end_id);
  CREATE INDEX CONCURRENTLY edge_p_idx ON dickens."_ag_label_edge" (id);
  CREATE INDEX CONCURRENTLY edge_sid_idx ON dickens."_ag_label_edge" (start_id);
  CREATE INDEX CONCURRENTLY edge_eid_idx ON dickens."_ag_label_edge" (end_id);
  CREATE INDEX CONCURRENTLY edge_seid_idx ON dickens."_ag_label_edge" (start_id,end_id);
  create INDEX CONCURRENTLY vertex_idx_node_id ON dickens."_ag_label_vertex" (ag_catalog.agtype_access_operator(properties, '"node_id"'::agtype));
  create INDEX CONCURRENTLY entity_idx_node_id ON dickens."Entity" (ag_catalog.agtype_access_operator(properties, '"node_id"'::agtype));
  CREATE INDEX CONCURRENTLY entity_node_id_gin_idx ON dickens."Entity" using gin(properties);
  ALTER TABLE dickens."DIRECTED" CLUSTER ON directed_sid_idx;

  -- drop if necessary
  drop INDEX entity_p_idx;
  drop INDEX vertex_p_idx;
  drop INDEX directed_p_idx;
  drop INDEX directed_eid_idx;
  drop INDEX directed_sid_idx;
  drop INDEX directed_seid_idx;
  drop INDEX edge_p_idx;
  drop INDEX edge_sid_idx;
  drop INDEX edge_eid_idx;
  drop INDEX edge_seid_idx;
  drop INDEX vertex_idx_node_id;
  drop INDEX entity_idx_node_id;
  drop INDEX entity_node_id_gin_idx;
  ```
* Known issue of the Apache AGE: The released versions got below issue:
  > You might find that the properties of the nodes/edges are empty.
  > It is a known issue of the release version: https://github.com/apache/age/pull/1721
  >
  > You can Compile the AGE from source code and fix it.

</details>

<details>
<summary> <b>Using Faiss for Storage</b> </summary>

- Install the required dependencies:
```
pip install faiss-cpu
```
You can also install `faiss-gpu` if you have GPU support.

- Here we are using `sentence-transformers` but you can also use `OpenAIEmbedding` model with `3072` dimensions.

```python
async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

# Initialize LightRAG with the LLM model function and embedding function
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=embedding_func,
    ),
    vector_storage="FaissVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.3  # Your desired threshold
    }
)
```

</details>

## Delete

```python
#  Delete Entity: Deleting entities by their names
rag.delete_by_entity("Project Gutenberg")

#  Delete Document: Deleting entities and relationships associated with the document by doc id
rag.delete_by_doc_id("doc_id")
```

## Edit Entities and Relations

LightRAG now supports comprehensive knowledge graph management capabilities, allowing you to create, edit, and delete entities and relationships within your knowledge graph.

<details>
<summary> <b>Create Entities and Relations</b> </summary>

```python
# Create new entity
entity = rag.create_entity("Google", {
    "description": "Google is a multinational technology company specializing in internet-related services and products.",
    "entity_type": "company"
})

# Create another entity
product = rag.create_entity("Gmail", {
    "description": "Gmail is an email service developed by Google.",
    "entity_type": "product"
})

# Create relation between entities
relation = rag.create_relation("Google", "Gmail", {
    "description": "Google develops and operates Gmail.",
    "keywords": "develops operates service",
    "weight": 2.0
})
```
</details>

<details>
<summary> <b>Edit Entities and Relations</b> </summary>

```python
# Edit an existing entity
updated_entity = rag.edit_entity("Google", {
    "description": "Google is a subsidiary of Alphabet Inc., founded in 1998.",
    "entity_type": "tech_company"
})

# Rename an entity (with all its relationships properly migrated)
renamed_entity = rag.edit_entity("Gmail", {
    "entity_name": "Google Mail",
    "description": "Google Mail (formerly Gmail) is an email service."
})

# Edit a relation between entities
updated_relation = rag.edit_relation("Google", "Google Mail", {
    "description": "Google created and maintains Google Mail service.",
    "keywords": "creates maintains email service",
    "weight": 3.0
})
```
</details>

All operations are available in both synchronous and asynchronous versions. The asynchronous versions have the prefix "a" (e.g., `acreate_entity`, `aedit_relation`).

#### Entity Operations

- **create_entity**: Creates a new entity with specified attributes
- **edit_entity**: Updates an existing entity's attributes or renames it

#### Relation Operations

- **create_relation**: Creates a new relation between existing entities
- **edit_relation**: Updates an existing relation's attributes

These operations maintain data consistency across both the graph database and vector database components, ensuring your knowledge graph remains coherent.

## Entity Merging

<details>
<summary> <b>Merge Entities and Their Relationships</b> </summary>

LightRAG now supports merging multiple entities into a single entity, automatically handling all relationships:

```python
# Basic entity merging
rag.merge_entities(
    source_entities=["Artificial Intelligence", "AI", "Machine Intelligence"],
    target_entity="AI Technology"
)
```

With custom merge strategy:

```python
# Define custom merge strategy for different fields
rag.merge_entities(
    source_entities=["John Smith", "Dr. Smith", "J. Smith"],
    target_entity="John Smith",
    merge_strategy={
        "description": "concatenate",  # Combine all descriptions
        "entity_type": "keep_first",   # Keep the entity type from the first entity
        "source_id": "join_unique"     # Combine all unique source IDs
    }
)
```

With custom target entity data:

```python
# Specify exact values for the merged entity
rag.merge_entities(
    source_entities=["New York", "NYC", "Big Apple"],
    target_entity="New York City",
    target_entity_data={
        "entity_type": "LOCATION",
        "description": "New York City is the most populous city in the United States.",
    }
)
```

Advanced usage combining both approaches:

```python
# Merge company entities with both strategy and custom data
rag.merge_entities(
    source_entities=["Microsoft Corp", "Microsoft Corporation", "MSFT"],
    target_entity="Microsoft",
    merge_strategy={
        "description": "concatenate",  # Combine all descriptions
        "source_id": "join_unique"     # Combine source IDs
    },
    target_entity_data={
        "entity_type": "ORGANIZATION",
    }
)
```

When merging entities:
* All relationships from source entities are redirected to the target entity
* Duplicate relationships are intelligently merged
* Self-relationships (loops) are prevented
* Source entities are removed after merging
* Relationship weights and attributes are preserved

</details>

## Cache

<details>
  <summary> <b>Clear Cache</b> </summary>

You can clear the LLM response cache with different modes:

```python
# Clear all cache
await rag.aclear_cache()

# Clear local mode cache
await rag.aclear_cache(modes=["local"])

# Clear extraction cache
await rag.aclear_cache(modes=["default"])

# Clear multiple modes
await rag.aclear_cache(modes=["local", "global", "hybrid"])

# Synchronous version
rag.clear_cache(modes=["local"])
```

Valid modes are:
- `"default"`: Extraction cache
- `"naive"`: Naive search cache
- `"local"`: Local search cache
- `"global"`: Global search cache
- `"hybrid"`: Hybrid search cache
- `"mix"`: Mix search cache

</details>

## LightRAG init parameters

<details>
<summary> Parameters </summary>

| **Parameter**                                | **Type** | **Explanation**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | **Default**                                                                                                 |
|----------------------------------------------| --- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **working\_dir**                             | `str` | Directory where the cache will be stored                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `lightrag_cache+timestamp`                                                                                  |
| **kv\_storage**                              | `str` | Storage type for documents and text chunks. Supported types: `JsonKVStorage`, `OracleKVStorage`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `JsonKVStorage`                                                                                             |
| **vector\_storage**                          | `str` | Storage type for embedding vectors. Supported types: `NanoVectorDBStorage`, `OracleVectorDBStorage`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `NanoVectorDBStorage`                                                                                       |
| **graph\_storage**                           | `str` | Storage type for graph edges and nodes. Supported types: `NetworkXStorage`, `Neo4JStorage`, `OracleGraphStorage`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `NetworkXStorage`                                                                                           |
| **chunk\_token\_size**                       | `int` | Maximum token size per chunk when splitting documents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `1200`                                                                                                      |
| **chunk\_overlap\_token\_size**              | `int` | Overlap token size between two chunks when splitting documents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `100`                                                                                                       |
| **tiktoken\_model\_name**                    | `str` | Model name for the Tiktoken encoder used to calculate token numbers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `gpt-4o-mini`                                                                                               |
| **entity\_extract\_max\_gleaning**           | `int` | Number of loops in the entity extraction process, appending history messages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `1`                                                                                                         |
| **entity\_summary\_to\_max\_tokens**         | `int` | Maximum token size for each entity summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `500`                                                                                                       |
| **node\_embedding\_algorithm**               | `str` | Algorithm for node embedding (currently not used)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `node2vec`                                                                                                  |
| **node2vec\_params**                         | `dict` | Parameters for node embedding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `{"dimensions": 1536,"num_walks": 10,"walk_length": 40,"window_size": 2,"iterations": 3,"random_seed": 3,}` |
| **embedding\_func**                          | `EmbeddingFunc` | Function to generate embedding vectors from text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `openai_embed`                                                                                          |
| **embedding\_batch\_num**                    | `int` | Maximum batch size for embedding processes (multiple texts sent per batch)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `32`                                                                                                        |
| **embedding\_func\_max\_async**              | `int` | Maximum number of concurrent asynchronous embedding processes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `16`                                                                                                        |
| **llm\_model\_func**                         | `callable` | Function for LLM generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `gpt_4o_mini_complete`                                                                                      |
| **llm\_model\_name**                         | `str` | LLM model name for generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `meta-llama/Llama-3.2-1B-Instruct`                                                                          |
| **llm\_model\_max\_token\_size**             | `int` | Maximum token size for LLM generation (affects entity relation summaries)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `32768`（default value changed by  env var MAX_TOKENS)                                    |
| **llm\_model\_max\_async**                   | `int` | Maximum number of concurrent asynchronous LLM processes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `16`（default value changed by  env var MAX_ASYNC)                                           |
| **llm\_model\_kwargs**                       | `dict` | Additional parameters for LLM generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |                                                                                                             |
| **vector\_db\_storage\_cls\_kwargs**         | `dict` | Additional parameters for vector database, like setting the threshold for nodes and relations retrieval.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | cosine_better_than_threshold: 0.2（default value changed by  env var COSINE_THRESHOLD) |
| **enable\_llm\_cache**                       | `bool` | If `TRUE`, stores LLM results in cache; repeated prompts return cached responses                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `TRUE`                                                                                                      |
| **enable\_llm\_cache\_for\_entity\_extract** | `bool` | If `TRUE`, stores LLM results in cache for entity extraction; Good for beginners to debug your application                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `TRUE`                                                                                                     |
| **addon\_params**                            | `dict` | Additional parameters, e.g., `{"example_number": 1, "language": "Simplified Chinese", "entity_types": ["organization", "person", "geo", "event"], "insert_batch_size": 10}`: sets example limit, output language, and batch size for document processing                                                                                                                                                                                                                                                                                                                                                                                                                            | `example_number: all examples, language: English, insert_batch_size: 10`                                    |
| **convert\_response\_to\_json\_func**        | `callable` | Not used                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `convert_response_to_json`                                                                                  |
| **embedding\_cache\_config**                 | `dict` | Configuration for question-answer caching. Contains three parameters:<br>- `enabled`: Boolean value to enable/disable cache lookup functionality. When enabled, the system will check cached responses before generating new answers.<br>- `similarity_threshold`: Float value (0-1), similarity threshold. When a new question's similarity with a cached question exceeds this threshold, the cached answer will be returned directly without calling the LLM.<br>- `use_llm_check`: Boolean value to enable/disable LLM similarity verification. When enabled, LLM will be used as a secondary check to verify the similarity between questions before returning cached answers. | Default: `{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}`                         |

</details>

## Error Handling

<details>
<summary>Click to view error handling details</summary>

The API includes comprehensive error handling:
- File not found errors (404)
- Processing errors (500)
- Supports multiple file encodings (UTF-8 and GBK)
</details>

## API
LightRag can be installed with API support to serve a Fast api interface to perform data upload and indexing/Rag operations/Rescan of the input folder etc..

[LightRag API](lightrag/api/README.md)

## Graph Visualization

<details>
<summary> <b>Graph visualization with html</b> </summary>

* The following code can be found in `examples/graph_visual_with_html.py`

```python
import networkx as nx
from pyvis.network import Network

# Load the GraphML file
G = nx.read_graphml('./dickens/graph_chunk_entity_relation.graphml')

# Create a Pyvis network
net = Network(notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

# Save and display the network
net.show('knowledge_graph.html')
```

</details>

<details>
<summary> <b>Graph visualization with Neo4</b> </summary>


* The following code can be found in `examples/graph_visual_with_neo4j.py`

```python
import os
import json
from lightrag.utils import xml_to_json
from neo4j import GraphDatabase

# Constants
WORKING_DIR = "./dickens"
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100

# Neo4j connection credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password"

def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON file created: {output_path}")
        return json_data
    else:
        print("Failed to create JSON data")
        return None

def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})

def main():
    # Paths
    xml_file = os.path.join(WORKING_DIR, 'graph_chunk_entity_relation.graphml')
    json_file = os.path.join(WORKING_DIR, 'graph_data.json')

    # Convert XML to JSON
    json_data = convert_xml_to_json(xml_file, json_file)
    if json_data is None:
        return

    # Load nodes and edges
    nodes = json_data.get('nodes', [])
    edges = json_data.get('edges', [])

    # Neo4j queries
    create_nodes_query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id
    REMOVE e:Entity
    WITH e, node
    CALL apoc.create.addLabels(e, [node.entity_type]) YIELD node AS labeledNode
    RETURN count(*)
    """

    create_edges_query = """
    UNWIND $edges AS edge
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})
    WITH source, target, edge,
         CASE
            WHEN edge.keywords CONTAINS 'lead' THEN 'lead'
            WHEN edge.keywords CONTAINS 'participate' THEN 'participate'
            WHEN edge.keywords CONTAINS 'uses' THEN 'uses'
            WHEN edge.keywords CONTAINS 'located' THEN 'located'
            WHEN edge.keywords CONTAINS 'occurs' THEN 'occurs'
           ELSE REPLACE(SPLIT(edge.keywords, ',')[0], '\"', '')
         END AS relType
    CALL apoc.create.relationship(source, relType, {
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }, target) YIELD rel
    RETURN count(*)
    """

    set_displayname_and_labels_query = """
    MATCH (n)
    SET n.displayName = n.id
    WITH n
    CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
    RETURN count(*)
    """

    # Create a Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        # Execute queries in batches
        with driver.session() as session:
            # Insert nodes in batches
            session.execute_write(process_in_batches, create_nodes_query, nodes, BATCH_SIZE_NODES)

            # Insert edges in batches
            session.execute_write(process_in_batches, create_edges_query, edges, BATCH_SIZE_EDGES)

            # Set displayName and labels
            session.run(set_displayname_and_labels_query)

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        driver.close()

if __name__ == "__main__":
    main()
```

</details>

<details>
<summary> <b>Graphml 3d visualizer</b> </summary>

LightRag can be installed with Tools support to add extra tools like the graphml 3d visualizer.

[LightRag Visualizer](lightrag/tools/lightrag_visualizer/README.md)

</details>

## Evaluation
### Dataset
The dataset used in LightRAG can be downloaded from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain).

### Generate Query
LightRAG uses the following prompt to generate high-level queries, with the corresponding code in `example/generate_query.py`.

<details>
<summary> Prompt </summary>

```python
Given the following description of a dataset:

{description}

Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

Output the results in the following structure:
- User 1: [user description]
    - Task 1: [task description]
        - Question 1:
        - Question 2:
        - Question 3:
        - Question 4:
        - Question 5:
    - Task 2: [task description]
        ...
    - Task 5: [task description]
- User 2: [user description]
    ...
- User 5: [user description]
    ...
```
</details>

### Batch Eval
To evaluate the performance of two RAG systems on high-level queries, LightRAG uses the following prompt, with the specific code available in `example/batch_eval.py`.

<details>
<summary> Prompt </summary>

```python
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
---Goal---
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
```
</details>

### Overall Performance Table

|                      | **Agriculture**         |                       | **CS**                |                       | **Legal**             |                       | **Mix**               |                       |
|----------------------|-------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
|                      | NaiveRAG                | **LightRAG**          | NaiveRAG              | **LightRAG**          | NaiveRAG              | **LightRAG**          | NaiveRAG              | **LightRAG**          |
| **Comprehensiveness** | 32.4%                  | **67.6%**             | 38.4%                | **61.6%**             | 16.4%                | **83.6%**             | 38.8%                | **61.2%**             |
| **Diversity**         | 23.6%                  | **76.4%**             | 38.0%                | **62.0%**             | 13.6%                | **86.4%**             | 32.4%                | **67.6%**             |
| **Empowerment**       | 32.4%                  | **67.6%**             | 38.8%                | **61.2%**             | 16.4%                | **83.6%**             | 42.8%                | **57.2%**             |
| **Overall**           | 32.4%                  | **67.6%**             | 38.8%                | **61.2%**             | 15.2%                | **84.8%**             | 40.0%                | **60.0%**             |
|                      | RQ-RAG                  | **LightRAG**          | RQ-RAG               | **LightRAG**          | RQ-RAG               | **LightRAG**          | RQ-RAG               | **LightRAG**          |
| **Comprehensiveness** | 31.6%                  | **68.4%**             | 38.8%                | **61.2%**             | 15.2%                | **84.8%**             | 39.2%                | **60.8%**             |
| **Diversity**         | 29.2%                  | **70.8%**             | 39.2%                | **60.8%**             | 11.6%                | **88.4%**             | 30.8%                | **69.2%**             |
| **Empowerment**       | 31.6%                  | **68.4%**             | 36.4%                | **63.6%**             | 15.2%                | **84.8%**             | 42.4%                | **57.6%**             |
| **Overall**           | 32.4%                  | **67.6%**             | 38.0%                | **62.0%**             | 14.4%                | **85.6%**             | 40.0%                | **60.0%**             |
|                      | HyDE                    | **LightRAG**          | HyDE                 | **LightRAG**          | HyDE                 | **LightRAG**          | HyDE                 | **LightRAG**          |
| **Comprehensiveness** | 26.0%                  | **74.0%**             | 41.6%                | **58.4%**             | 26.8%                | **73.2%**             | 40.4%                | **59.6%**             |
| **Diversity**         | 24.0%                  | **76.0%**             | 38.8%                | **61.2%**             | 20.0%                | **80.0%**             | 32.4%                | **67.6%**             |
| **Empowerment**       | 25.2%                  | **74.8%**             | 40.8%                | **59.2%**             | 26.0%                | **74.0%**             | 46.0%                | **54.0%**             |
| **Overall**           | 24.8%                  | **75.2%**             | 41.6%                | **58.4%**             | 26.4%                | **73.6%**             | 42.4%                | **57.6%**             |
|                      | GraphRAG                | **LightRAG**          | GraphRAG             | **LightRAG**          | GraphRAG             | **LightRAG**          | GraphRAG             | **LightRAG**          |
| **Comprehensiveness** | 45.6%                  | **54.4%**             | 48.4%                | **51.6%**             | 48.4%                | **51.6%**             | **50.4%**            | 49.6%                |
| **Diversity**         | 22.8%                  | **77.2%**             | 40.8%                | **59.2%**             | 26.4%                | **73.6%**             | 36.0%                | **64.0%**             |
| **Empowerment**       | 41.2%                  | **58.8%**             | 45.2%                | **54.8%**             | 43.6%                | **56.4%**             | **50.8%**            | 49.2%                |
| **Overall**           | 45.2%                  | **54.8%**             | 48.0%                | **52.0%**             | 47.2%                | **52.8%**             | **50.4%**            | 49.6%                |

## Reproduce
All the code can be found in the `./reproduce` directory.

### Step-0 Extract Unique Contexts
First, we need to extract unique contexts in the datasets.

<details>
<summary> Code </summary>

```python
def extract_unique_contexts(input_directory, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(input_directory, '*.jsonl'))
    print(f"Found {len(jsonl_files)} JSONL files.")

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_unique_contexts.json"
        output_path = os.path.join(output_directory, output_filename)

        unique_contexts_dict = {}

        print(f"Processing file: {filename}")

        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        context = json_obj.get('context')
                        if context and context not in unique_contexts_dict:
                            unique_contexts_dict[context] = None
                    except json.JSONDecodeError as e:
                        print(f"JSON decoding error in file {filename} at line {line_number}: {e}")
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue
        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")
            continue

        unique_contexts_list = list(unique_contexts_dict.keys())
        print(f"There are {len(unique_contexts_list)} unique `context` entries in the file {filename}.")

        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(unique_contexts_list, outfile, ensure_ascii=False, indent=4)
            print(f"Unique `context` entries have been saved to: {output_filename}")
        except Exception as e:
            print(f"An error occurred while saving to the file {output_filename}: {e}")

    print("All files have been processed.")

```
</details>

### Step-1 Insert Contexts
For the extracted contexts, we insert them into the LightRAG system.

<details>
<summary> Code </summary>

```python
def insert_text(rag, file_path):
    with open(file_path, mode='r') as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")
```
</details>

### Step-2 Generate Queries

We extract tokens from the first and the second half of each context in the dataset, then combine them as dataset descriptions to generate queries.

<details>
<summary> Code </summary>

```python
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_summary(context, tot_tokens=2000):
    tokens = tokenizer.tokenize(context)
    half_tokens = tot_tokens // 2

    start_tokens = tokens[1000:1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens):1000]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return summary
```
</details>

### Step-3 Query
For the queries generated in Step-2, we will extract them and query LightRAG.

<details>
<summary> Code </summary>

```python
def extract_queries(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    data = data.replace('**', '')

    queries = re.findall(r'- Question \d+: (.+)', data)

    return queries
```
</details>

## Star History

<a href="https://star-history.com/#HKUDS/LightRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date" />
 </picture>
</a>

## Contribution

Thank you to all our contributors!

<a href="https://github.com/HKUDS/LightRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/LightRAG" />
</a>

## 🌟Citation

```python
@article{guo2024lightrag,
title={LightRAG: Simple and Fast Retrieval-Augmented Generation},
author={Zirui Guo and Lianghao Xia and Yanhua Yu and Tu Ao and Chao Huang},
year={2024},
eprint={2410.05779},
archivePrefix={arXiv},
primaryClass={cs.IR}
}
```
**Thank you for your interest in our work!**
