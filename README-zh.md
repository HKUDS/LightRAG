# LightRAG: Simple and Fast Retrieval-Augmented Generation

<img src="./README.assets/b2aaf634151b4706892693ffb43d9093.png" width="800" alt="LightRAG Diagram">

## 🎉 新闻

- [X] [2025.03.18]🎯📢LightRAG现已支持引文功能。
- [X] [2025.02.05]🎯📢我们团队发布了[VideoRAG](https://github.com/HKUDS/VideoRAG)，用于理解超长上下文视频。
- [X] [2025.01.13]🎯📢我们团队发布了[MiniRAG](https://github.com/HKUDS/MiniRAG)，使用小型模型简化RAG。
- [X] [2025.01.06]🎯📢现在您可以[使用PostgreSQL进行存储](#using-postgresql-for-storage)。
- [X] [2024.12.31]🎯📢LightRAG现在支持[通过文档ID删除](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#delete)。
- [X] [2024.11.25]🎯📢LightRAG现在支持无缝集成[自定义知识图谱](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#insert-custom-kg)，使用户能够用自己的领域专业知识增强系统。
- [X] [2024.11.19]🎯📢LightRAG的综合指南现已在[LearnOpenCV](https://learnopencv.com/lightrag)上发布。非常感谢博客作者。
- [X] [2024.11.11]🎯📢LightRAG现在支持[通过实体名称删除实体](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#delete)。
- [X] [2024.11.09]🎯📢推出[LightRAG Gui](https://lightrag-gui.streamlit.app)，允许您插入、查询、可视化和下载LightRAG知识。
- [X] [2024.11.04]🎯📢现在您可以[使用Neo4J进行存储](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#using-neo4j-for-storage)。
- [X] [2024.10.29]🎯📢LightRAG现在通过`textract`支持多种文件类型，包括PDF、DOC、PPT和CSV。
- [X] [2024.10.20]🎯📢我们为LightRAG添加了一个新功能：图形可视化。
- [X] [2024.10.18]🎯📢我们添加了[LightRAG介绍视频](https://youtu.be/oageL-1I0GE)的链接。感谢作者！
- [X] [2024.10.17]🎯📢我们创建了一个[Discord频道](https://discord.gg/yF2MmDJyGJ)！欢迎加入分享和讨论！🎉🎉
- [X] [2024.10.16]🎯📢LightRAG现在支持[Ollama模型](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)！
- [X] [2024.10.15]🎯📢LightRAG现在支持[Hugging Face模型](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)！

<details>
  <summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; display: list-item;">
    算法流程图
  </summary>

![LightRAG索引流程图](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)
*图1：LightRAG索引流程图 - 图片来源：[Source](https://learnopencv.com/lightrag/)*
![LightRAG检索和查询流程图](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)
*图2：LightRAG检索和查询流程图 - 图片来源：[Source](https://learnopencv.com/lightrag/)*

</details>

## 安装

### 安装LightRAG服务器

LightRAG服务器旨在提供Web UI和API支持。Web UI便于文档索引、知识图谱探索和简单的RAG查询界面。LightRAG服务器还提供兼容Ollama的接口，旨在将LightRAG模拟为Ollama聊天模型。这使得AI聊天机器人（如Open WebUI）可以轻松访问LightRAG。

* 从PyPI安装

```bash
pip install "lightrag-hku[api]"
```

* 从源代码安装

```bash
# 如有必要，创建Python虚拟环境
# 以可编辑模式安装并支持API
pip install -e ".[api]"
```

### 安装LightRAG Core

* 从源代码安装（推荐）

```bash
cd LightRAG
pip install -e .
```

* 从PyPI安装

```bash
pip install lightrag-hku
```

## 快速开始

### 使用LightRAG服务器

**有关LightRAG服务器的更多信息，请参阅[LightRAG服务器](./lightrag/api/README.md)。**

### 使用LightRAG Core

LightRAG核心功能的示例代码请参见`examples`目录。您还可参照[视频](https://www.youtube.com/watch?v=g21royNJ4fw)视频完成环境配置。若已持有OpenAI API密钥，可以通过以下命令运行演示代码：

```bash
### you should run the demo code with project folder
cd LightRAG
### provide your API-KEY for OpenAI
export OPENAI_API_KEY="sk-...your_opeai_key..."
### download the demo document of "A Christmas Carol" by Charles Dickens
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
### run the demo code
python examples/lightrag_openai_demo.py
```

如需流式响应示例的实现代码，请参阅 `examples/lightrag_openai_compatible_demo.py`。运行前，请确保根据需求修改示例代码中的LLM及嵌入模型配置。

**注意事项**：在运行demo程序的时候需要注意，不同的测试程序可能使用的是不同的embedding模型，更换不同的embeding模型的时候需要把清空数据目录（`./dickens`），否则层序执行会出错。如果你想保留LLM缓存，可以在清除数据目录是保留`kv_store_llm_response_cache.json`文件。

## 使用LightRAG Core进行编程

### 一个简单程序

以下Python代码片段演示了如何初始化LightRAG、插入文本并进行查询：

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
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
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def main():
    try:
        # 初始化RAG实例
        rag = await initialize_rag()
        # 插入文本
        await rag.insert("Your text")

        # 执行混合检索
        mode = "hybrid"
        print(
            await rag.query(
                "这个故事的主要主题是什么？",
                param=QueryParam(mode=mode)
            )
        )

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
```

重要说明：
- 运行脚本前请先导出你的OPENAI_API_KEY环境变量。
- 该程序使用LightRAG的默认存储设置，所有数据将持久化在WORKING_DIR/rag_storage目录下。
- 该示例仅展示了初始化LightRAG对象的最简单方式：注入embedding和LLM函数，并在创建LightRAG对象后初始化存储和管道状态。

### LightRAG初始化参数

以下是完整的LightRAG对象初始化参数清单：

<details>
<summary> 参数 </summary>

| **参数** | **类型** | **说明** | **默认值** |
|--------------|----------|-----------------|-------------|
| **working_dir** | `str` | 存储缓存的目录 | `lightrag_cache+timestamp` |
| **kv_storage** | `str` | Storage type for documents and text chunks. Supported types: `JsonKVStorage`,`PGKVStorage`,`RedisKVStorage`,`MongoKVStorage` | `JsonKVStorage` |
| **vector_storage** | `str` | Storage type for embedding vectors. Supported types: `NanoVectorDBStorage`,`PGVectorStorage`,`MilvusVectorDBStorage`,`ChromaVectorDBStorage`,`FaissVectorDBStorage`,`MongoVectorDBStorage`,`QdrantVectorDBStorage` | `NanoVectorDBStorage` |
| **graph_storage** | `str` | Storage type for graph edges and nodes. Supported types: `NetworkXStorage`,`Neo4JStorage`,`PGGraphStorage`,`AGEStorage` | `NetworkXStorage` |
| **doc_status_storage** | `str` | Storage type for documents process status. Supported types: `JsonDocStatusStorage`,`PGDocStatusStorage`,`MongoDocStatusStorage` | `JsonDocStatusStorage` |
| **chunk_token_size** | `int` | 拆分文档时每个块的最大令牌大小 | `1200` |
| **chunk_overlap_token_size** | `int` | 拆分文档时两个块之间的重叠令牌大小 | `100` |
| **tokenizer** | `Tokenizer` | 用于将文本转换为 tokens（数字）以及使用遵循 TokenizerInterface 协议的 .encode() 和 .decode() 函数将 tokens 转换回文本的函数。 如果您不指定，它将使用默认的 Tiktoken tokenizer。 | `TiktokenTokenizer` |
| **tiktoken_model_name** | `str` | 如果您使用的是默认的 Tiktoken tokenizer，那么这是要使用的特定 Tiktoken 模型的名称。如果您提供自己的 tokenizer，则忽略此设置。 | `gpt-4o-mini` |
| **entity_extract_max_gleaning** | `int` | 实体提取过程中的循环次数，附加历史消息 | `1` |
| **entity_summary_to_max_tokens** | `int` | 每个实体摘要的最大令牌大小 | `500` |
| **node_embedding_algorithm** | `str` | 节点嵌入算法（当前未使用） | `node2vec` |
| **node2vec_params** | `dict` | 节点嵌入的参数 | `{"dimensions": 1536,"num_walks": 10,"walk_length": 40,"window_size": 2,"iterations": 3,"random_seed": 3,}` |
| **embedding_func** | `EmbeddingFunc` | 从文本生成嵌入向量的函数 | `openai_embed` |
| **embedding_batch_num** | `int` | 嵌入过程的最大批量大小（每批发送多个文本） | `32` |
| **embedding_func_max_async** | `int` | 最大并发异步嵌入进程数 | `16` |
| **llm_model_func** | `callable` | LLM生成的函数 | `gpt_4o_mini_complete` |
| **llm_model_name** | `str` | 用于生成的LLM模型名称 | `meta-llama/Llama-3.2-1B-Instruct` |
| **llm_model_max_token_size** | `int` | LLM生成的最大令牌大小（影响实体关系摘要） | `32768`（默认值由环境变量MAX_TOKENS更改） |
| **llm_model_max_async** | `int` | 最大并发异步LLM进程数 | `4`（默认值由环境变量MAX_ASYNC更改） |
| **llm_model_kwargs** | `dict` | LLM生成的附加参数 | |
| **vector_db_storage_cls_kwargs** | `dict` | 向量数据库的附加参数，如设置节点和关系检索的阈值 | cosine_better_than_threshold: 0.2（默认值由环境变量COSINE_THRESHOLD更改） |
| **enable_llm_cache** | `bool` | 如果为`TRUE`，将LLM结果存储在缓存中；重复的提示返回缓存的响应 | `TRUE` |
| **enable_llm_cache_for_entity_extract** | `bool` | 如果为`TRUE`，将实体提取的LLM结果存储在缓存中；适合初学者调试应用程序 | `TRUE` |
| **addon_params** | `dict` | 附加参数，例如`{"example_number": 1, "language": "Simplified Chinese", "entity_types": ["organization", "person", "geo", "event"]}`：设置示例限制、输出语言和文档处理的批量大小 | `example_number: 所有示例, language: English` |
| **convert_response_to_json_func** | `callable` | 未使用 | `convert_response_to_json` |
| **embedding_cache_config** | `dict` | 问答缓存的配置。包含三个参数：`enabled`：布尔值，启用/禁用缓存查找功能。启用时，系统将在生成新答案之前检查缓存的响应。`similarity_threshold`：浮点值（0-1），相似度阈值。当新问题与缓存问题的相似度超过此阈值时，将直接返回缓存的答案而不调用LLM。`use_llm_check`：布尔值，启用/禁用LLM相似度验证。启用时，在返回缓存答案之前，将使用LLM作为二次检查来验证问题之间的相似度。 | 默认：`{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}` |

</details>

### 查询参数

使用QueryParam控制你的查询行为：

```python
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global"
    """指定检索模式：
    - "local"：专注于上下文相关信息。
    - "global"：利用全局知识。
    - "hybrid"：结合本地和全局检索方法。
    - "naive"：执行基本搜索，不使用高级技术。
    - "mix"：集成知识图谱和向量检索。混合模式结合知识图谱和向量搜索：
        - 同时使用结构化（KG）和非结构化（向量）信息
        - 通过分析关系和上下文提供全面的答案
        - 通过HTML img标签支持图像内容
        - 允许通过top_k参数控制检索深度
    """
    only_need_context: bool = False
    """如果为True，仅返回检索到的上下文而不生成响应。"""
    response_type: str = "Multiple Paragraphs"
    """定义响应格式。示例：'Multiple Paragraphs'（多段落）, 'Single Paragraph'（单段落）, 'Bullet Points'（要点列表）。"""
    top_k: int = 60
    """要检索的顶部项目数量。在'local'模式下代表实体，在'global'模式下代表关系。"""
    max_token_for_text_unit: int = 4000
    """每个检索文本块允许的最大令牌数。"""
    max_token_for_global_context: int = 4000
    """全局检索中关系描述的最大令牌分配。"""
    max_token_for_local_context: int = 4000
    """本地检索中实体描述的最大令牌分配。"""
    ids: list[str] | None = None # 仅支持PG向量数据库
    """用于过滤RAG的ID列表。"""
    model_func: Callable[..., object] | None = None
    """查询使用的LLM模型函数。如果提供了此选项，它将代替LightRAG全局模型函数。
    这允许为不同的查询模式使用不同的模型。
    """
    ...
```

> top_k的默认值可以通过环境变量TOP_K更改。

### LLM and Embedding注入

LightRAG 需要利用LLM和Embeding模型来完成文档索引和知识库查询工作。在初始化LightRAG的时候需要把阶段，需要把LLM和Embedding的操作函数注入到对象中：

<details>
<summary> <b>使用类OpenAI的API</b> </summary>

* LightRAG还支持类OpenAI的聊天/嵌入API：

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
<summary> <b>使用Hugging Face模型</b> </summary>

* 如果您想使用Hugging Face模型，只需要按如下方式设置LightRAG：

参见`lightrag_hf_demo.py`

```python
# 使用Hugging Face模型初始化LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # 使用Hugging Face模型进行文本生成
    llm_model_name='meta-llama/Llama-3.1-8B-Instruct',  # Hugging Face的模型名称
    # 使用Hugging Face嵌入函数
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
<summary> <b>使用Ollama模型</b> </summary>
如果您想使用Ollama模型，您需要拉取计划使用的模型和嵌入模型，例如`nomic-embed-text`。

然后您只需要按如下方式设置LightRAG：

```python
# 使用Ollama模型初始化LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # 使用Ollama模型进行文本生成
    llm_model_name='your_model_name', # 您的模型名称
    # 使用Ollama嵌入函数
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

* **增加上下文大小**

为了使LightRAG正常工作，上下文应至少为32k令牌。默认情况下，Ollama模型的上下文大小为8k。您可以通过以下两种方式之一实现这一点：

* **在Modelfile中增加`num_ctx`参数**

1. 拉取模型：

```bash
ollama pull qwen2
```

2. 显示模型文件：

```bash
ollama show --modelfile qwen2 > Modelfile
```

3. 编辑Modelfile，添加以下行：

```bash
PARAMETER num_ctx 32768
```

4. 创建修改后的模型：

```bash
ollama create -f Modelfile qwen2m
```

* **通过Ollama API设置`num_ctx`**

您可以使用`llm_model_kwargs`参数配置ollama：

```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # 使用Ollama模型进行文本生成
    llm_model_name='your_model_name', # 您的模型名称
    llm_model_kwargs={"options": {"num_ctx": 32768}},
    # 使用Ollama嵌入函数
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

* **低RAM GPU**

为了在低RAM GPU上运行此实验，您应该选择小型模型并调整上下文窗口（增加上下文会增加内存消耗）。例如，在6Gb RAM的改装挖矿GPU上运行这个ollama示例需要将上下文大小设置为26k，同时使用`gemma2:2b`。它能够在`book.txt`中找到197个实体和19个关系。

</details>
<details>
<summary> <b>LlamaIndex</b> </summary>

LightRAG支持与LlamaIndex集成 (`llm/llama_index_impl.py`):

- 通过LlamaIndex与OpenAI和其他提供商集成
- 详细设置和示例请参见[LlamaIndex文档](lightrag/llm/Readme.md)

**使用示例：**

```python
# 使用LlamaIndex直接访问OpenAI
import asyncio
from lightrag import LightRAG
from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# 为LightRAG设置日志处理程序
setup_logger("lightrag", level="INFO")

async def initialize_rag():
    rag = LightRAG(
        working_dir="your/path",
        llm_model_func=llama_index_complete_if_cache,  # LlamaIndex兼容的完成函数
        embedding_func=EmbeddingFunc(    # LlamaIndex兼容的嵌入函数
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: llama_index_embed(texts, embed_model=embed_model)
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def main():
    # 初始化RAG实例
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # 执行朴素搜索
    print(
        rag.query("这个故事的主要主题是什么？", param=QueryParam(mode="naive"))
    )

    # 执行本地搜索
    print(
        rag.query("这个故事的主要主题是什么？", param=QueryParam(mode="local"))
    )

    # 执行全局搜索
    print(
        rag.query("这个故事的主要主题是什么？", param=QueryParam(mode="global"))
    )

    # 执行混合搜索
    print(
        rag.query("这个故事的主要主题是什么？", param=QueryParam(mode="hybrid"))
    )

if __name__ == "__main__":
    main()
```

**详细文档和示例，请参见：**

- [LlamaIndex文档](lightrag/llm/Readme.md)
- [直接OpenAI示例](examples/lightrag_llamaindex_direct_demo.py)
- [LiteLLM代理示例](examples/lightrag_llamaindex_litellm_demo.py)

</details>

### 对话历史

LightRAG现在通过对话历史功能支持多轮对话。以下是使用方法：

```python
# 创建对话历史
conversation_history = [
    {"role": "user", "content": "主角对圣诞节的态度是什么？"},
    {"role": "assistant", "content": "在故事开始时，埃比尼泽·斯克鲁奇对圣诞节持非常消极的态度..."},
    {"role": "user", "content": "他的态度是如何改变的？"}
]

# 创建带有对话历史的查询参数
query_param = QueryParam(
    mode="mix",  # 或其他模式："local"、"global"、"hybrid"
    conversation_history=conversation_history,  # 添加对话历史
    history_turns=3  # 考虑最近的对话轮数
)

# 进行考虑对话历史的查询
response = rag.query(
    "是什么导致了他性格的这种变化？",
    param=query_param
)
```

### 自定义提示词

LightRAG现在支持自定义提示，以便对系统行为进行精细控制。以下是使用方法：

```python
# 创建查询参数
query_param = QueryParam(
    mode="hybrid",  # 或其他模式："local"、"global"、"hybrid"、"mix"和"naive"
)

# 示例1：使用默认系统提示
response_default = rag.query(
    "可再生能源的主要好处是什么？",
    param=query_param
)
print(response_default)

# 示例2：使用自定义提示
custom_prompt = """
您是环境科学领域的专家助手。请提供详细且结构化的答案，并附带示例。
---对话历史---
{history}

---知识库---
{context_data}

---响应规则---

- 目标格式和长度：{response_type}
"""
response_custom = rag.query(
    "可再生能源的主要好处是什么？",
    param=query_param,
    system_prompt=custom_prompt  # 传递自定义提示
)
print(response_custom)
```

### 关键词提取

我们引入了新函数`query_with_separate_keyword_extraction`来增强关键词提取功能。该函数将关键词提取过程与用户提示分开，专注于查询以提高提取关键词的相关性。

* 工作原理

该函数将输入分为两部分：

- `用户查询`
- `提示`

然后仅对`用户查询`执行关键词提取。这种分离确保提取过程是集中和相关的，不受`提示`中任何额外语言的影响。它还允许`提示`纯粹用于响应格式化，保持用户原始问题的意图和清晰度。

* 使用示例

这个`示例`展示了如何为教育内容定制函数，专注于为高年级学生提供详细解释。

```python
rag.query_with_separate_keyword_extraction(
    query="解释重力定律",
    prompt="提供适合学习物理的高中生的详细解释。",
    param=QueryParam(mode="hybrid")
)
```

### 插入自定义知识

```python
custom_kg = {
    "chunks": [
        {
            "content": "Alice和Bob正在合作进行量子计算研究。",
            "source_id": "doc-1"
        }
    ],
    "entities": [
        {
            "entity_name": "Alice",
            "entity_type": "person",
            "description": "Alice是一位专门研究量子物理的研究员。",
            "source_id": "doc-1"
        },
        {
            "entity_name": "Bob",
            "entity_type": "person",
            "description": "Bob是一位数学家。",
            "source_id": "doc-1"
        },
        {
            "entity_name": "量子计算",
            "entity_type": "technology",
            "description": "量子计算利用量子力学现象进行计算。",
            "source_id": "doc-1"
        }
    ],
    "relationships": [
        {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "description": "Alice和Bob是研究伙伴。",
            "keywords": "合作 研究",
            "weight": 1.0,
            "source_id": "doc-1"
        },
        {
            "src_id": "Alice",
            "tgt_id": "量子计算",
            "description": "Alice进行量子计算研究。",
            "keywords": "研究 专业",
            "weight": 1.0,
            "source_id": "doc-1"
        },
        {
            "src_id": "Bob",
            "tgt_id": "量子计算",
            "description": "Bob研究量子计算。",
            "keywords": "研究 应用",
            "weight": 1.0,
            "source_id": "doc-1"
        }
    ]
}

rag.insert_custom_kg(custom_kg)
```

### 插入

<details>
  <summary> <b> 基本插入 </b></summary>

```python
# 基本插入
rag.insert("文本")
```

</details>

<details>
  <summary> <b> 批量插入 </b></summary>

```python
# 基本批量插入：一次插入多个文本
rag.insert(["文本1", "文本2",...])

# 带有自定义批量大小配置的批量插入
rag = LightRAG(
    ...
    working_dir=WORKING_DIR,
    max_parallel_insert = 4
)

rag.insert(["文本1", "文本2", "文本3", ...])  # 文档将以4个为一批进行处理
```

参数 `max_parallel_insert` 用于控制文档索引流水线中并行处理的文档数量。若未指定，默认值为 **2**。建议将该参数设置为 **10 以下**，因为性能瓶颈通常出现在大语言模型（LLM）的处理环节。

</details>

<details>
  <summary> <b> 带ID插入 </b></summary>

如果您想为文档提供自己的ID，文档数量和ID数量必须相同。

```python
# 插入单个文本，并为其提供ID
rag.insert("文本1", ids=["文本1的ID"])

# 插入多个文本，并为它们提供ID
rag.insert(["文本1", "文本2",...], ids=["文本1的ID", "文本2的ID"])
```

</details>

<details>
  <summary><b>使用管道插入</b></summary>

`apipeline_enqueue_documents`和`apipeline_process_enqueue_documents`函数允许您对文档进行增量插入到图中。

这对于需要在后台处理文档的场景很有用，同时仍允许主线程继续执行。

并使用例程处理新文档。

```python
rag = LightRAG(..)

await rag.apipeline_enqueue_documents(input)
# 您的循环例程
await rag.apipeline_process_enqueue_documents(input)
```

</details>

<details>
  <summary><b>插入多文件类型支持</b></summary>

`textract`支持读取TXT、DOCX、PPTX、CSV和PDF等文件类型。

```python
import textract

file_path = 'TEXT.pdf'
text_content = textract.process(file_path)

rag.insert(text_content.decode('utf-8'))
```

</details>

<details>
  <summary><b>引文功能</b></summary>

通过提供文件路径，系统确保可以将来源追溯到其原始文档。

```python
# 定义文档及其文件路径
documents = ["文档内容1", "文档内容2"]
file_paths = ["path/to/doc1.txt", "path/to/doc2.txt"]

# 插入带有文件路径的文档
rag.insert(documents, file_paths=file_paths)
```

</details>

### 存储

LightRAG使用到4种类型的存储，每一种存储都有多种实现方案。在初始化LightRAG的时候可以通过参数设定这四类存储的实现方案。详情请参看前面的LightRAG初始化参数。

<details>
<summary> <b>使用Neo4J进行存储</b> </summary>

* 对于生产级场景，您很可能想要利用企业级解决方案
* 进行KG存储。推荐在Docker中运行Neo4J以进行无缝本地测试。
* 参见：https://hub.docker.com/_/neo4j

```python
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

# 为LightRAG设置日志记录器
setup_logger("lightrag", level="INFO")

# 当您启动项目时，请确保通过指定kg="Neo4JStorage"来覆盖默认的KG：NetworkX。

# 注意：默认设置使用NetworkX
# 使用Neo4J实现初始化LightRAG。
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,  # 使用gpt_4o_mini_complete LLM模型
        graph_storage="Neo4JStorage", #<-----------覆盖KG默认值
    )

    # 初始化数据库连接
    await rag.initialize_storages()
    # 初始化文档处理的管道状态
    await initialize_pipeline_status()

    return rag
```

参见test_neo4j.py获取工作示例。

</details>

<details>
<summary> <b>使用Faiss进行存储</b> </summary>

- 安装所需依赖：

```
pip install faiss-cpu
```

如果您有GPU支持，也可以安装`faiss-gpu`。

- 这里我们使用`sentence-transformers`，但您也可以使用维度为`3072`的`OpenAIEmbedding`模型。

```python
async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

# 使用LLM模型函数和嵌入函数初始化LightRAG
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
        "cosine_better_than_threshold": 0.3  # 您期望的阈值
    }
)
```

</details>

<details>
<summary> <b>使用PostgreSQL进行存储</b> </summary>

对于生产级场景，您很可能想要利用企业级解决方案。PostgreSQL可以为您提供一站式解决方案，作为KV存储、向量数据库（pgvector）和图数据库（apache AGE）。

* PostgreSQL很轻量，整个二进制发行版包括所有必要的插件可以压缩到40MB：参考[Windows发布版](https://github.com/ShanGor/apache-age-windows/releases/tag/PG17%2Fv1.5.0-rc0)，它在Linux/Mac上也很容易安装。
* 如果您是初学者并想避免麻烦，推荐使用docker，请从这个镜像开始（请务必阅读概述）：https://hub.docker.com/r/shangor/postgres-for-rag
* 如何开始？参考：[examples/lightrag_zhipu_postgres_demo.py](https://github.com/HKUDS/LightRAG/blob/main/examples/lightrag_zhipu_postgres_demo.py)
* 为AGE创建索引示例：（如有必要，将下面的`dickens`改为您的图名）
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

  -- 如有必要可以删除
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
* Apache AGE的已知问题：发布版本存在以下问题：
  > 您可能会发现节点/边的属性是空的。
  > 这是发布版本的已知问题：https://github.com/apache/age/pull/1721
  >
  > 您可以从源代码编译AGE来修复它。
  >

</details>

## 编辑实体和关系

LightRAG现在支持全面的知识图谱管理功能，允许您在知识图谱中创建、编辑和删除实体和关系。

<details>
<summary> <b>创建实体和关系</b> </summary>

```python
# 创建新实体
entity = rag.create_entity("Google", {
    "description": "Google是一家专注于互联网相关服务和产品的跨国科技公司。",
    "entity_type": "company"
})

# 创建另一个实体
product = rag.create_entity("Gmail", {
    "description": "Gmail是由Google开发的电子邮件服务。",
    "entity_type": "product"
})

# 创建实体之间的关系
relation = rag.create_relation("Google", "Gmail", {
    "description": "Google开发和运营Gmail。",
    "keywords": "开发 运营 服务",
    "weight": 2.0
})
```

</details>

<details>
<summary> <b>编辑实体和关系</b> </summary>

```python
# 编辑现有实体
updated_entity = rag.edit_entity("Google", {
    "description": "Google是Alphabet Inc.的子公司，成立于1998年。",
    "entity_type": "tech_company"
})

# 重命名实体（所有关系都会正确迁移）
renamed_entity = rag.edit_entity("Gmail", {
    "entity_name": "Google Mail",
    "description": "Google Mail（前身为Gmail）是一项电子邮件服务。"
})

# 编辑实体之间的关系
updated_relation = rag.edit_relation("Google", "Google Mail", {
    "description": "Google创建并维护Google Mail服务。",
    "keywords": "创建 维护 电子邮件服务",
    "weight": 3.0
})
```

</details>

所有操作都有同步和异步版本。异步版本带有前缀"a"（例如，`acreate_entity`，`aedit_relation`）。

#### 实体操作

- **create_entity**：创建具有指定属性的新实体
- **edit_entity**：更新现有实体的属性或重命名它

#### 关系操作

- **create_relation**：在现有实体之间创建新关系
- **edit_relation**：更新现有关系的属性

这些操作在图数据库和向量数据库组件之间保持数据一致性，确保您的知识图谱保持连贯。

## Token统计功能
<details>
<summary> <b>概述和使用</b> </summary>

LightRAG提供了TokenTracker工具来跟踪和管理大模型的token消耗。这个功能对于控制API成本和优化性能特别有用。

### 使用方法

```python
from lightrag.utils import TokenTracker

# 创建TokenTracker实例
token_tracker = TokenTracker()

# 方法1：使用上下文管理器（推荐）
# 适用于需要自动跟踪token使用的场景
with token_tracker:
    result1 = await llm_model_func("你的问题1")
    result2 = await llm_model_func("你的问题2")

# 方法2：手动添加token使用记录
# 适用于需要更精细控制token统计的场景
token_tracker.reset()

rag.insert()

rag.query("你的问题1", param=QueryParam(mode="naive"))
rag.query("你的问题2", param=QueryParam(mode="mix"))

# 显示总token使用量（包含插入和查询操作）
print("Token usage:", token_tracker.get_usage())
```

### 使用建议
- 在长会话或批量操作中使用上下文管理器，可以自动跟踪所有token消耗
- 对于需要分段统计的场景，使用手动模式并适时调用reset()
- 定期检查token使用情况，有助于及时发现异常消耗
- 在开发测试阶段积极使用此功能，以便优化生产环境的成本

### 实际应用示例
您可以参考以下示例来实现token统计：
- `examples/lightrag_gemini_track_token_demo.py`：使用Google Gemini模型的token统计示例
- `examples/lightrag_siliconcloud_track_token_demo.py`：使用SiliconCloud模型的token统计示例

这些示例展示了如何在不同模型和场景下有效地使用TokenTracker功能。

</details>

## 数据导出功能

### 概述

LightRAG允许您以各种格式导出知识图谱数据，用于分析、共享和备份目的。系统支持导出实体、关系和关系数据。

### 导出功能

#### 基本用法

```python
# 基本CSV导出（默认格式）
rag.export_data("knowledge_graph.csv")

# 指定任意格式
rag.export_data("output.xlsx", file_format="excel")
```

#### 支持的不同文件格式

```python
# 以CSV格式导出数据
rag.export_data("graph_data.csv", file_format="csv")

# 导出数据到Excel表格
rag.export_data("graph_data.xlsx", file_format="excel")

# 以markdown格式导出数据
rag.export_data("graph_data.md", file_format="md")

# 导出数据为文本
rag.export_data("graph_data.txt", file_format="txt")
```

#### 附加选项

在导出中包含向量嵌入（可选）：

```python
rag.export_data("complete_data.csv", include_vector_data=True)
```

### 导出数据包括

所有导出包括：

* 实体信息（名称、ID、元数据）
* 关系数据（实体之间的连接）
* 来自向量数据库的关系信息

## 实体合并

<details>
<summary> <b>合并实体及其关系</b> </summary>

LightRAG现在支持将多个实体合并为单个实体，自动处理所有关系：

```python
# 基本实体合并
rag.merge_entities(
    source_entities=["人工智能", "AI", "机器智能"],
    target_entity="AI技术"
)
```

使用自定义合并策略：

```python
# 为不同字段定义自定义合并策略
rag.merge_entities(
    source_entities=["约翰·史密斯", "史密斯博士", "J·史密斯"],
    target_entity="约翰·史密斯",
    merge_strategy={
        "description": "concatenate",  # 组合所有描述
        "entity_type": "keep_first",   # 保留第一个实体的类型
        "source_id": "join_unique"     # 组合所有唯一的源ID
    }
)
```

使用自定义目标实体数据：

```python
# 为合并后的实体指定确切值
rag.merge_entities(
    source_entities=["纽约", "NYC", "大苹果"],
    target_entity="纽约市",
    target_entity_data={
        "entity_type": "LOCATION",
        "description": "纽约市是美国人口最多的城市。",
    }
)
```

结合两种方法的高级用法：

```python
# 使用策略和自定义数据合并公司实体
rag.merge_entities(
    source_entities=["微软公司", "Microsoft Corporation", "MSFT"],
    target_entity="微软",
    merge_strategy={
        "description": "concatenate",  # 组合所有描述
        "source_id": "join_unique"     # 组合源ID
    },
    target_entity_data={
        "entity_type": "ORGANIZATION",
    }
)
```

合并实体时：

* 所有来自源实体的关系都会重定向到目标实体
* 重复的关系会被智能合并
* 防止自我关系（循环）
* 合并后删除源实体
* 保留关系权重和属性

</details>

## 缓存

<details>
  <summary> <b>清除缓存</b> </summary>

您可以使用不同模式清除LLM响应缓存：

```python
# 清除所有缓存
await rag.aclear_cache()

# 清除本地模式缓存
await rag.aclear_cache(modes=["local"])

# 清除提取缓存
await rag.aclear_cache(modes=["default"])

# 清除多个模式
await rag.aclear_cache(modes=["local", "global", "hybrid"])

# 同步版本
rag.clear_cache(modes=["local"])
```

有效的模式包括：

- `"default"`：提取缓存
- `"naive"`：朴素搜索缓存
- `"local"`：本地搜索缓存
- `"global"`：全局搜索缓存
- `"hybrid"`：混合搜索缓存
- `"mix"`：混合搜索缓存

</details>

## LightRAG API

LightRAG服务器旨在提供Web UI和API支持。**有关LightRAG服务器的更多信息，请参阅[LightRAG服务器](./lightrag/api/README.md)。**

## 知识图谱可视化

LightRAG服务器提供全面的知识图谱可视化功能。它支持各种重力布局、节点查询、子图过滤等。**有关LightRAG服务器的更多信息，请参阅[LightRAG服务器](./lightrag/api/README.md)。**

![iShot_2025-03-23_12.40.08](./README.assets/iShot_2025-03-23_12.40.08.png)

## 评估

### 数据集

LightRAG使用的数据集可以从[TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain)下载。

### 生成查询

LightRAG使用以下提示生成高级查询，相应的代码在`example/generate_query.py`中。

<details>
<summary> 提示 </summary>

```python
给定以下数据集描述：

{description}

请识别5个可能会使用此数据集的潜在用户。对于每个用户，列出他们会使用此数据集执行的5个任务。然后，对于每个（用户，任务）组合，生成5个需要对整个数据集有高级理解的问题。

按以下结构输出结果：
- 用户1：[用户描述]
    - 任务1：[任务描述]
        - 问题1：
        - 问题2：
        - 问题3：
        - 问题4：
        - 问题5：
    - 任务2：[任务描述]
        ...
    - 任务5：[任务描述]
- 用户2：[用户描述]
    ...
- 用户5：[用户描述]
    ...
```

</details>

### 批量评估

为了评估两个RAG系统在高级查询上的性能，LightRAG使用以下提示，具体代码可在`example/batch_eval.py`中找到。

<details>
<summary> 提示 </summary>

```python
---角色---
您是一位专家，负责根据三个标准评估同一问题的两个答案：**全面性**、**多样性**和**赋能性**。
---目标---
您将根据三个标准评估同一问题的两个答案：**全面性**、**多样性**和**赋能性**。

- **全面性**：答案提供了多少细节来涵盖问题的所有方面和细节？
- **多样性**：答案在提供关于问题的不同视角和见解方面有多丰富多样？
- **赋能性**：答案在多大程度上帮助读者理解并对主题做出明智判断？

对于每个标准，选择更好的答案（答案1或答案2）并解释原因。然后，根据这三个类别选择总体赢家。

这是问题：
{query}

这是两个答案：

**答案1：**
{answer1}

**答案2：**
{answer2}

使用上述三个标准评估两个答案，并为每个标准提供详细解释。

以下列JSON格式输出您的评估：

{{
    "全面性": {{
        "获胜者": "[答案1或答案2]",
        "解释": "[在此提供解释]"
    }},
    "赋能性": {{
        "获胜者": "[答案1或答案2]",
        "解释": "[在此提供解释]"
    }},
    "总体获胜者": {{
        "获胜者": "[答案1或答案2]",
        "解释": "[根据三个标准总结为什么这个答案是总体获胜者]"
    }}
}}
```

</details>

### 总体性能表

|                      |**农业**|            |**计算机科学**|            |**法律**|            |**混合**|            |
|----------------------|---------------|------------|------|------------|---------|------------|-------|------------|
|                      |NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|
|**全面性**|32.4%|**67.6%**|38.4%|**61.6%**|16.4%|**83.6%**|38.8%|**61.2%**|
|**多样性**|23.6%|**76.4%**|38.0%|**62.0%**|13.6%|**86.4%**|32.4%|**67.6%**|
|**赋能性**|32.4%|**67.6%**|38.8%|**61.2%**|16.4%|**83.6%**|42.8%|**57.2%**|
|**总体**|32.4%|**67.6%**|38.8%|**61.2%**|15.2%|**84.8%**|40.0%|**60.0%**|
|                      |RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|
|**全面性**|31.6%|**68.4%**|38.8%|**61.2%**|15.2%|**84.8%**|39.2%|**60.8%**|
|**多样性**|29.2%|**70.8%**|39.2%|**60.8%**|11.6%|**88.4%**|30.8%|**69.2%**|
|**赋能性**|31.6%|**68.4%**|36.4%|**63.6%**|15.2%|**84.8%**|42.4%|**57.6%**|
|**总体**|32.4%|**67.6%**|38.0%|**62.0%**|14.4%|**85.6%**|40.0%|**60.0%**|
|                      |HyDE|**LightRAG**|HyDE|**LightRAG**|HyDE|**LightRAG**|HyDE|**LightRAG**|
|**全面性**|26.0%|**74.0%**|41.6%|**58.4%**|26.8%|**73.2%**|40.4%|**59.6%**|
|**多样性**|24.0%|**76.0%**|38.8%|**61.2%**|20.0%|**80.0%**|32.4%|**67.6%**|
|**赋能性**|25.2%|**74.8%**|40.8%|**59.2%**|26.0%|**74.0%**|46.0%|**54.0%**|
|**总体**|24.8%|**75.2%**|41.6%|**58.4%**|26.4%|**73.6%**|42.4%|**57.6%**|
|                      |GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|
|**全面性**|45.6%|**54.4%**|48.4%|**51.6%**|48.4%|**51.6%**|**50.4%**|49.6%|
|**多样性**|22.8%|**77.2%**|40.8%|**59.2%**|26.4%|**73.6%**|36.0%|**64.0%**|
|**赋能性**|41.2%|**58.8%**|45.2%|**54.8%**|43.6%|**56.4%**|**50.8%**|49.2%|
|**总体**|45.2%|**54.8%**|48.0%|**52.0%**|47.2%|**52.8%**|**50.4%**|49.6%|

## 复现

所有代码都可以在`./reproduce`目录中找到。

### 步骤0 提取唯一上下文

首先，我们需要提取数据集中的唯一上下文。

<details>
<summary> 代码 </summary>

```python
def extract_unique_contexts(input_directory, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(input_directory, '*.jsonl'))
    print(f"找到{len(jsonl_files)}个JSONL文件。")

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_unique_contexts.json"
        output_path = os.path.join(output_directory, output_filename)

        unique_contexts_dict = {}

        print(f"处理文件：{filename}")

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
                        print(f"文件{filename}第{line_number}行JSON解码错误：{e}")
        except FileNotFoundError:
            print(f"未找到文件：{filename}")
            continue
        except Exception as e:
            print(f"处理文件{filename}时发生错误：{e}")
            continue

        unique_contexts_list = list(unique_contexts_dict.keys())
        print(f"文件{filename}中有{len(unique_contexts_list)}个唯一的`context`条目。")

        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(unique_contexts_list, outfile, ensure_ascii=False, indent=4)
            print(f"唯一的`context`条目已保存到：{output_filename}")
        except Exception as e:
            print(f"保存到文件{output_filename}时发生错误：{e}")

    print("所有文件已处理完成。")

```

</details>

### 步骤1 插入上下文

对于提取的上下文，我们将它们插入到LightRAG系统中。

<details>
<summary> 代码 </summary>

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
            print(f"插入失败，重试（{retries}/{max_retries}），错误：{e}")
            time.sleep(10)
    if retries == max_retries:
        print("超过最大重试次数后插入失败")
```

</details>

### 步骤2 生成查询

我们从数据集中每个上下文的前半部分和后半部分提取令牌，然后将它们组合为数据集描述以生成查询。

<details>
<summary> 代码 </summary>

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

### 步骤3 查询

对于步骤2中生成的查询，我们将提取它们并查询LightRAG。

<details>
<summary> 代码 </summary>

```python
def extract_queries(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    data = data.replace('**', '')

    queries = re.findall(r'- Question \d+: (.+)', data)

    return queries
```

</details>

## Star历史

<a href="https://star-history.com/#HKUDS/LightRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date" />
 </picture>
</a>

## 贡献

感谢所有贡献者！

<a href="https://github.com/HKUDS/LightRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/LightRAG" />
</a>

## 🌟引用

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

**感谢您对我们工作的关注！**
