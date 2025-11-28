<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# 🚀 LightRAG: Simple and Fast Retrieval-Augmented Generation

<div align="center">
    <a href="https://trendshift.io/repositories/13043" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13043" alt="HKUDS%2FLightRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
  <div style="width: 100%; height: 2px; margin: 20px 0; background: linear-gradient(90deg, transparent, #00d9ff, transparent);"></div>
</div>

<div align="center">
  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; text-align: center;">
    <p>
      <a href='https://github.com/HKUDS/LightRAG'><img src='https://img.shields.io/badge/🔥项目-主页-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e'></a>
      <a href='https://arxiv.org/abs/2410.05779'><img src='https://img.shields.io/badge/📄arXiv-2410.05779-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e'></a>
      <a href="https://github.com/HKUDS/LightRAG/stargazers"><img src='https://img.shields.io/github/stars/HKUDS/LightRAG?color=00d9ff&style=for-the-badge&logo=star&logoColor=white&labelColor=1a1a2e' /></a>
    </p>
    <p>
      <img src="https://img.shields.io/badge/🐍Python-3.10-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
      <a href="https://pypi.org/project/lightrag-hku/"><img src="https://img.shields.io/pypi/v/lightrag-hku.svg?style=for-the-badge&logo=pypi&logoColor=white&labelColor=1a1a2e&color=ff6b6b"></a>
    </p>
    <p>
      <a href="https://discord.gg/yF2MmDJyGJ"><img src="https://img.shields.io/badge/💬Discord-社区-7289da?style=for-the-badge&logo=discord&logoColor=white&labelColor=1a1a2e"></a>
      <a href="https://github.com/HKUDS/LightRAG/issues/285"><img src="https://img.shields.io/badge/💬微信群-交流-07c160?style=for-the-badge&logo=wechat&logoColor=white&labelColor=1a1a2e"></a>
    </p>
    <p>
      <a href="README-zh.md"><img src="https://img.shields.io/badge/🇨🇳中文版-1a1a2e?style=for-the-badge"></a>
      <a href="README.md"><img src="https://img.shields.io/badge/🇺🇸English-1a1a2e?style=for-the-badge"></a>
    </p>
    <p>
      <a href="https://pepy.tech/projects/lightrag-hku"><img src="https://static.pepy.tech/personalized-badge/lightrag-hku?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads"></a>
    </p>
  </div>
</div>

</div>

<div align="center" style="margin: 30px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="800">
</div>

<div align="center" style="margin: 30px 0;">
    <img src="./README.assets/b2aaf634151b4706892693ffb43d9093.png" width="800" alt="LightRAG Diagram">
</div>

---

## 🎉 新闻

- [2025.11.05]🎯添加**基于RAGAS的**评估框架和**Langfuse**可观测性支持（API可随查询结果返回召回上下文）。
- [2025.10.22]🎯消除处理**大规模数据集**的性能瓶颈。
- [2025.09.15]🎯显著提升**小型LLM**（如Qwen3-30B-A3B）的知识图谱提取准确性。
- [2025.08.29]🎯现已支持**Reranker**，显著提升混合查询性能(现已设为默认查询模式)。
- [2025.08.04]🎯支持**文档删除**并重新生成知识图谱以确保查询性能。
- [2025.06.16]🎯我们的团队发布了[RAG-Anything](https://github.com/HKUDS/RAG-Anything)，一个用于无缝处理文本、图像、表格和方程式的全功能多模态 RAG 系统。
- [2025.06.05]🎯LightRAG现已集成[RAG-Anything](https://github.com/HKUDS/RAG-Anything)，支持全面的多模态文档解析与RAG能力（PDF、图片、Office文档、表格、公式等）。详见下方[多模态处理模块](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#多模态文档处理rag-anything集成)。
- [2025.03.18]🎯LightRAG现已支持参考文献功能。
- [2025.02.12]🎯现在您可以使用MongoDB作为一体化存储解决方案。
- [2025.02.05]🎯我们团队发布了[VideoRAG](https://github.com/HKUDS/VideoRAG)，用于理解超长上下文视频。
- [2025.01.13]🎯我们团队发布了[MiniRAG](https://github.com/HKUDS/MiniRAG)，使用小型模型简化RAG。
- [2025.01.06]🎯现在您可以使用PostgreSQL作为一体化存储解决方案。
- [2024.11.19]🎯LightRAG的综合指南现已在[LearnOpenCV](https://learnopencv.com/lightrag)上发布。非常感谢博客作者。
- [2024.11.09]🎯推出LightRAG Webui，允许您插入、查询、可视化LightRAG知识。
- [2024.11.04]🎯现在您可以[使用Neo4J进行存储](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#using-neo4j-for-storage)。
- [2024.10.18]🎯我们添加了[LightRAG介绍视频](https://youtu.be/oageL-1I0GE)的链接。感谢作者！
- [2024.10.17]🎯我们创建了一个[Discord频道](https://discord.gg/yF2MmDJyGJ)！欢迎加入分享和讨论！🎉🎉
- [2024.10.16]🎯LightRAG现在支持[Ollama模型](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)！

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

> **💡 使用 uv 进行包管理**: 本项目使用 [uv](https://docs.astral.sh/uv/) 进行快速可靠的 Python 包管理。
> 首先安装 uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (Unix/macOS) 或 `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows)
>
> **注意**: 如果您更喜欢使用 pip 也可以，但我们推荐使用 uv 以获得更好的性能和更可靠的依赖管理。

### 安装LightRAG服务器

LightRAG服务器旨在提供Web UI和API支持。Web UI便于文档索引、知识图谱探索和简单的RAG查询界面。LightRAG服务器还提供兼容Ollama的接口，旨在将LightRAG模拟为Ollama聊天模型。这使得AI聊天机器人（如Open WebUI）可以轻松访问LightRAG。

* 从PyPI安装

```bash
# 使用 uv (推荐)
uv pip install "lightrag-hku[api]"
# 或使用 pip
# pip install "lightrag-hku[api]"

cp env.example .env  # 使用你的LLM和Embedding模型访问参数更新.env文件

lightrag-server
```

* 从源代码安装

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG

# 使用 uv (推荐)
# 注意: uv sync 会自动在 .venv/ 目录创建虚拟环境
uv sync --extra api
source .venv/bin/activate  # 激活虚拟环境 (Linux/macOS)
# Windows 系统: .venv\Scripts\activate

# 或使用 pip 和虚拟环境
# python -m venv .venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
# pip install -e ".[api]"

cp env.example .env  # 使用你的LLM和Embedding模型访问参数更新.env文件

# 构建前端代码
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

lightrag-server
```

* 使用 Docker Compose 启动 LightRAG 服务器

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
cp env.example .env  # 使用你的LLM和Embedding模型访问参数更新.env文件
# modify LLM and Embedding settings in .env
docker compose up
```

> 在此获取LightRAG docker镜像历史版本: [LightRAG Docker Images]( https://github.com/HKUDS/LightRAG/pkgs/container/lightrag)

### 安装LightRAG Core

* 从源代码安装（推荐）

```bash
cd LightRAG
# 注意: uv sync 会自动在 .venv/ 目录创建虚拟环境
uv sync
source .venv/bin/activate  # 激活虚拟环境 (Linux/macOS)
# Windows 系统: .venv\Scripts\activate

# 或: pip install -e .
```

* 从PyPI安装

```bash
uv pip install lightrag-hku
# 或: pip install lightrag-hku
```

## 快速开始

### LightRAG的LLM及配套技术栈要求

LightRAG对大型语言模型（LLM）的能力要求远高于传统RAG，因为它需要LLM执行文档中的实体关系抽取任务。配置合适的Embedding和Reranker模型对提高查询表现也至关重要。

- **LLM选型**：
  - 推荐选用参数量至少为32B的LLM。
  - 上下文长度至少为32KB，推荐达到64KB。
  - 在文档索引阶段不建议选择推理模型。
  - 在查询阶段建议选择比索引阶段能力更强的模型，以达到更高的查询效果。
- **Embedding模型**：
  - 高性能的Embedding模型对RAG至关重要。
  - 推荐使用主流的多语言Embedding模型，例如：BAAI/bge-m3 和 text-embedding-3-large。
  - **重要提示**：在文档索引前必须确定使用的Embedding模型，且在文档查询阶段必须沿用与索引阶段相同的模型。有些存储（例如PostgreSQL）在首次建立数表的时候需要确定向量维度，因此更换Embedding模型后需要删除向量相关库表，以便让LightRAG重建新的库表。
- **Reranker模型配置**：
  - 配置Reranker模型能够显著提升LightRAG的检索效果。
  - 启用Reranker模型后，推荐将“mix模式”设为默认查询模式。
  - 推荐选用主流的Reranker模型，例如：BAAI/bge-reranker-v2-m3 或 Jina 等服务商提供的模型。

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

**注意1**：在运行demo程序的时候需要注意，不同的测试程序可能使用的是不同的embedding模型，更换不同的embeding模型的时候需要把清空数据目录（`./dickens`），否则层序执行会出错。如果你想保留LLM缓存，可以在清除数据目录时保留`kv_store_llm_response_cache.json`文件。

**注意2**：官方支持的示例代码仅为 `lightrag_openai_demo.py` 和 `lightrag_openai_compatible_demo.py` 两个文件。其他示例文件均为社区贡献内容，尚未经过完整测试与优化。

## 使用LightRAG Core进行编程

> ⚠️ **如果您希望将LightRAG集成到您的项目中，建议您使用LightRAG Server提供的REST API**。LightRAG Core通常用于嵌入式应用，或供希望进行研究与评估的学者使用。

### ⚠️ 重要：初始化要求

LightRAG 在使用前需要显式初始化。 创建 LightRAG 实例后，您必须调用 await rag.initialize_storages()，否则将出现错误。

### 一个简单程序

以下Python代码片段演示了如何初始化LightRAG、插入文本并进行查询：

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
    await rag.initialize_storages()    return rag

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
| **node_embedding_algorithm** | `str` | 节点嵌入算法（当前未使用） | `node2vec` |
| **node2vec_params** | `dict` | 节点嵌入的参数 | `{"dimensions": 1536,"num_walks": 10,"walk_length": 40,"window_size": 2,"iterations": 3,"random_seed": 3,}` |
| **embedding_func** | `EmbeddingFunc` | 从文本生成嵌入向量的函数 | `openai_embed` |
| **embedding_batch_num** | `int` | 嵌入过程的最大批量大小（每批发送多个文本） | `32` |
| **embedding_func_max_async** | `int` | 最大并发异步嵌入进程数 | `16` |
| **llm_model_func** | `callable` | LLM生成的函数 | `gpt_4o_mini_complete` |
| **llm_model_name** | `str` | 用于生成的LLM模型名称 | `meta-llama/Llama-3.2-1B-Instruct` |
| **summary_context_size** | `int` | 合并实体关系摘要时送给LLM的最大令牌数 | `10000`（由环境变量 SUMMARY_MAX_CONTEXT 设置） |
| **summary_max_tokens** | `int` | 合并实体关系描述的最大令牌数长度 | `500`（由环境变量 SUMMARY_MAX_TOKENS 设置） |
| **llm_model_max_async** | `int` | 最大并发异步LLM进程数 | `4`（默认值由环境变量MAX_ASYNC更改） |
| **llm_model_kwargs** | `dict` | LLM生成的附加参数 | |
| **vector_db_storage_cls_kwargs** | `dict` | 向量数据库的附加参数，如设置节点和关系检索的阈值 | cosine_better_than_threshold: 0.2（默认值由环境变量COSINE_THRESHOLD更改） |
| **enable_llm_cache** | `bool` | 如果为`TRUE`，将LLM结果存储在缓存中；重复的提示返回缓存的响应 | `TRUE` |
| **enable_llm_cache_for_entity_extract** | `bool` | 如果为`TRUE`，将实体提取的LLM结果存储在缓存中；适合初学者调试应用程序 | `TRUE` |
| **addon_params** | `dict` | 附加参数，例如`{"language": "Simplified Chinese", "entity_types": ["organization", "person", "location", "event"]}`：设置示例限制、输出语言和文档处理的批量大小 | language: English` |
| **embedding_cache_config** | `dict` | 问答缓存的配置。包含三个参数：`enabled`：布尔值，启用/禁用缓存查找功能。启用时，系统将在生成新答案之前检查缓存的响应。`similarity_threshold`：浮点值（0-1），相似度阈值。当新问题与缓存问题的相似度超过此阈值时，将直接返回缓存的答案而不调用LLM。`use_llm_check`：布尔值，启用/禁用LLM相似度验证。启用时，在返回缓存答案之前，将使用LLM作为二次检查来验证问题之间的相似度。 | 默认：`{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}` |

</details>

### 查询参数

使用QueryParam控制你的查询行为：

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

    hl_keywords: list[str] = field(default_factory=list)
    """List of high-level keywords to prioritize in retrieval."""

    ll_keywords: list[str] = field(default_factory=list)
    """List of low-level keywords to refine retrieval focus."""

    # History mesages is only send to LLM for context, not used for retrieval
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    ids: list[str] | None = None
    """List of ids to filter the results."""

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

> top_k的默认值可以通过环境变量TOP_K更改。

### LLM and Embedding注入

LightRAG 需要利用LLM和Embeding模型来完成文档索引和知识库查询工作。在初始化LightRAG的时候需要把阶段，需要把LLM和Embedding的操作函数注入到对象中：

<details>
<summary> <b>使用类OpenAI的API</b> </summary>

* LightRAG还支持类OpenAI的聊天/嵌入API：

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

@wrap_embedding_func_with_attrs(embedding_dim=4096, max_token_size=8192)
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
        embedding_func=embedding_func  # 直接传入装饰后的函数
    )

    await rag.initialize_storages()
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
import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await ollama_embed.func(texts, embed_model="nomic-embed-text")

# 使用Ollama模型初始化LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # 使用Ollama模型进行文本生成
    llm_model_name='your_model_name', # 您的模型名称
    embedding_func=embedding_func,  # 直接传入装饰后的函数
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
import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await ollama_embed.func(texts, embed_model="nomic-embed-text")

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # 使用Ollama模型进行文本生成
    llm_model_name='your_model_name', # 您的模型名称
    llm_model_kwargs={"options": {"num_ctx": 32768}},
    embedding_func=embedding_func,  # 直接传入装饰后的函数
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
from lightrag.utils import setup_logger

# 为LightRAG设置日志处理程序
setup_logger("lightrag", level="INFO")

async def initialize_rag():
    rag = LightRAG(
        working_dir="your/path",
        llm_model_func=llama_index_complete_if_cache,  # LlamaIndex兼容的完成函数
        embedding_func=EmbeddingFunc(    # LlamaIndex兼容的嵌入函数
            embedding_dim=1536,
            func=lambda texts: llama_index_embed(texts, embed_model=embed_model)
        ),
    )

    await rag.initialize_storages()
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

### Rerank函数注入

为了提高检索质量，可以根据更有效的相关性评分模型对文档进行重排序。`rerank.py`文件提供了三个Reranker提供商的驱动函数：

* **Cohere / vLLM**: `cohere_rerank`
* **Jina AI**: `jina_rerank`
* **Aliyun阿里云**: `ali_rerank`

您可以将这些函数之一注入到LightRAG对象的`rerank_model_func`属性中。这将使LightRAG的查询功能能够使用注入的函数对检索到的文本块进行重新排序。有关详细用法，请参阅`examples/rerank_example.py`文件。

### 用户提示词 vs. 查询内容

当使用LightRAG查询内容的时候，不要把内容查询和与查询结果无关的输出加工写在一起。因为把两者混在一起会严重影响查询的效果。Query Param中的`user_prompt`就是为解决这一问题而设计的。`user_prompt`中的内容不参与RAG中的查询过程，它仅会在获得查询结果之后，与查询结果一起送给LLM，指导LLM如何处理查询结果。以下是使用方法：

```python
# Create query parameters
query_param = QueryParam(
    mode = "hybrid",  # Other modes：local, global, hybrid, mix, naive
    user_prompt = "如需画图使用mermaid格式，节点名称用英文或拼音，显示名称用中文",
)

# Query and process
response_default = rag.query(
    "请画出 Scrooge 的人物关系图谱",
    param=query_param
)
print(response_default)
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
  <summary><b>使用流水线插入</b></summary>

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

LightRAG 使用 4 种类型的存储用于不同目的：

* KV_STORAGE：llm 响应缓存、文本块、文档信息
* VECTOR_STORAGE：实体向量、关系向量、块向量
* GRAPH_STORAGE：实体关系图
* DOC_STATUS_STORAGE：文档索引状态

每种存储类型都有几种实现：

* KV_STORAGE 支持的实现名称

```
JsonKVStorage    JsonFile(默认)
PGKVStorage      Postgres
RedisKVStorage   Redis
MongoKVStorage   MogonDB
```

* GRAPH_STORAGE 支持的实现名称

```
NetworkXStorage      NetworkX(默认)
Neo4JStorage         Neo4J
PGGraphStorage       PostgreSQL with AGE plugin
```

> 在测试中Neo4j图形数据库相比PostgreSQL AGE有更好的性能表现。

* VECTOR_STORAGE 支持的实现名称

```
NanoVectorDBStorage         NanoVector(默认)
PGVectorStorage             Postgres
MilvusVectorDBStorge        Milvus
FaissVectorDBStorage        Faiss
QdrantVectorDBStorage       Qdrant
MongoVectorDBStorage        MongoDB
```

* DOC_STATUS_STORAGE 支持的实现名称

```
JsonDocStatusStorage        JsonFile(默认)
PGDocStatusStorage          Postgres
MongoDocStatusStorage       MongoDB
```

每一种存储类型的链接配置范例可以在 `env.example` 文件中找到。链接字符串中的数据库实例是需要你预先在数据库服务器上创建好的，LightRAG 仅负责在数据库实例中创建数据表，不负责创建数据库实例。如果使用 Redis 作为存储，记得给 Redis 配置自动持久化数据规则，否则 Redis 服务重启后数据会丢失。如果使用PostgreSQL数据库，推荐使用16.6版本或以上。

<details>
<summary> <b>使用Neo4J存储</b> </summary>

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
    return rag
```

参见test_neo4j.py获取工作示例。

</details>

<details>
<summary> <b>使用Faiss存储</b> </summary>
在使用Faiss向量数据库之前必须手工安装`faiss-cpu`或`faiss-gpu`。

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
<summary> <b>使用PostgreSQL存储</b> </summary>

对于生产级场景，您很可能想要利用企业级解决方案。PostgreSQL可以为您提供一站式储解解决方案，作为KV存储、向量数据库（pgvector）和图数据库（apache AGE）。支持 PostgreSQL 版本为16.6或以上。

* 如果您是初学者并想避免麻烦，推荐使用docker，请从这个镜像开始（默认帐号密码:rag/rag）：https://hub.docker.com/r/gzdaniel/postgres-for-rag
* Apache AGE的性能不如Neo4j。追求高性能的图数据库请使用Noe4j。

</details>

<details>
<summary> <b>使用MogonDB存储</b> </summary>

MongoDB为LightRAG提供了一站式的存储解决方案。MongoDB提供原生的KV存储和向量存储。LightRAG使用MogoDB的集合实现了一个简易的图存储。MongoDB 官方的向量检索功能（`$vectorSearch`）目前必须依赖其官方的云服务 MongoDB Atlas。无法在自托管的 MongoDB Community/Enterprise 版本上使用此功能。

</details>

<details>
<summary> <b>使用Redis存储</b> </summary>

LightRAG支持使用Reidis作为KV存储。使用Redis存储的时候需要注意进行持久化配置和内存使用量配置。以下是推荐的redis配置

```
save 900 1
save 300 10
save 60 1000
stop-writes-on-bgsave-error yes
maxmemory 4gb
maxmemory-policy noeviction
maxclients 500
```

</details>

### LightRAG实例间的数据隔离

通过 workspace 参数可以不同实现不同LightRAG实例之间的存储数据隔离。LightRAG在初始化后workspace就已经确定，之后修改workspace是无效的。下面是不同类型的存储实现工作空间的方式：

- **对于本地基于文件的数据库，数据隔离通过工作空间子目录实现：** JsonKVStorage, JsonDocStatusStorage, NetworkXStorage, NanoVectorDBStorage, FaissVectorDBStorage。
- **对于将数据存储在集合（collection）中的数据库，通过在集合名称前添加工作空间前缀来实现：** RedisKVStorage, RedisDocStatusStorage, MilvusVectorDBStorage, QdrantVectorDBStorage, MongoKVStorage, MongoDocStatusStorage, MongoVectorDBStorage, MongoGraphStorage, PGGraphStorage。
- **对于关系型数据库，数据隔离通过向表中添加 `workspace` 字段进行数据的逻辑隔离：** PGKVStorage, PGVectorStorage, PGDocStatusStorage。

* **对于Neo4j图数据库，通过label来实现数据的逻辑隔离**：Neo4JStorage

为了保持对遗留数据的兼容，在未配置工作空间时PostgreSQL非图存储的工作空间为`default`，PostgreSQL AGE图存储的工作空间为空，Neo4j图存储的默认工作空间为`base`。对于所有的外部存储，系统都提供了专用的工作空间环境变量，用于覆盖公共的 `WORKSPACE`环境变量配置。这些适用于指定存储类型的工作空间环境变量为：`REDIS_WORKSPACE`, `MILVUS_WORKSPACE`, `QDRANT_WORKSPACE`, `MONGODB_WORKSPACE`, `POSTGRES_WORKSPACE`, `NEO4J_WORKSPACE`。

### AGENTS.md – 自动编程引导文件

AGENTS.md 是一种简洁、开放的格式，用于指导自动编程代理完成工作（https://agents.md/）。它为 LightRAG 项目提供了一个专属且可预测的上下文与指令位置，帮助 AI 代码代理更好地开展工作。不同的 AI 代码代理不应各自维护独立的引导文件。如果某个 AI 代理无法自动识别 AGENTS.md，可使用符号链接来解决。建立符号链接后，可通过配置本地的 `.gitignore_global` 文件防止其被提交至 Git 仓库。

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

所有操作都有同步和异步版本。异步版本带有前缀"a"（例如，`acreate_entity`，`aedit_relation`）。

</details>

<details>
<summary> <b>插入自定义知识</b> </summary>

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

</details>

<details>
<summary> <b>其它实体与关系操作</b> </summary>

- **create_entity**：创建具有指定属性的新实体
- **edit_entity**：更新现有实体的属性或重命名它

- **create_relation**：在现有实体之间创建新关系
- **edit_relation**：更新现有关系的属性

这些操作在图数据库和向量数据库组件之间保持数据一致性，确保您的知识图谱保持连贯。

</details>

## 删除功能

LightRAG提供了全面的删除功能，允许您删除文档、实体和关系。

<details>
<summary> <b>删除实体</b> </summary>

您可以通过实体名称删除实体及其所有关联关系：

```python
# 删除实体及其所有关系（同步版本）
rag.delete_by_entity("Google")

# 异步版本
await rag.adelete_by_entity("Google")
```

删除实体时会：
- 从知识图谱中移除该实体节点
- 删除该实体的所有关联关系
- 从向量数据库中移除相关的嵌入向量
- 保持知识图谱的完整性

</details>

<details>
<summary> <b>删除关系</b> </summary>

您可以删除两个特定实体之间的关系：

```python
# 删除两个实体之间的关系（同步版本）
rag.delete_by_relation("Google", "Gmail")

# 异步版本
await rag.adelete_by_relation("Google", "Gmail")
```

删除关系时会：
- 移除指定的关系边
- 从向量数据库中删除关系的嵌入向量
- 保留两个实体节点及其他关系

</details>

<details>
<summary> <b>通过文档ID删除</b> </summary>

您可以通过文档ID删除整个文档及其相关的所有知识：

```python
# 通过文档ID删除（异步版本）
await rag.adelete_by_doc_id("doc-12345")
```

通过文档ID删除时的优化处理：
- **智能清理**：自动识别并删除仅属于该文档的实体和关系
- **保留共享知识**：如果实体或关系在其他文档中也存在，则会保留并重新构建描述
- **缓存优化**：清理相关的LLM缓存以减少存储开销
- **增量重建**：从剩余文档重新构建受影响的实体和关系描述

删除过程包括：
1. 删除文档相关的所有文本块
2. 识别仅属于该文档的实体和关系并删除
3. 重新构建在其他文档中仍存在的实体和关系
4. 更新所有相关的向量索引
5. 清理文档状态记录

注意：通过文档ID删除是一个异步操作，因为它涉及复杂的知识图谱重构过程。

</details>

<details>
<summary> <b>删除注意事项</b> </summary>

**重要提醒：**

1. **不可逆操作**：所有删除操作都是不可逆的，请谨慎使用
2. **性能考虑**：删除大量数据时可能需要一些时间，特别是通过文档ID删除
3. **数据一致性**：删除操作会自动维护知识图谱和向量数据库之间的一致性
4. **备份建议**：在执行重要删除操作前建议备份数据

**批量删除建议：**
- 对于批量删除操作，建议使用异步方法以获得更好的性能
- 大规模删除时，考虑分批进行以避免系统负载过高

</details>

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

## 多模态文档处理（RAG-Anything集成）

LightRAG 现已与 [RAG-Anything](https://github.com/HKUDS/RAG-Anything) 实现无缝集成，这是一个专为 LightRAG 构建的**全能多模态文档处理RAG系统**。RAG-Anything 提供先进的解析和检索增强生成（RAG）能力，让您能够无缝处理多模态文档，并从各种文档格式中提取结构化内容——包括文本、图片、表格和公式——以集成到您的RAG流程中。

**主要特性：**
- **端到端多模态流程**：从文档摄取解析到智能多模态问答的完整工作流程
- **通用文档支持**：无缝处理PDF、Office文档（DOC/DOCX/PPT/PPTX/XLS/XLSX）、图片和各种文件格式
- **专业内容分析**：针对图片、表格、数学公式和异构内容类型的专用处理器
- **多模态知识图谱**：自动实体提取和跨模态关系发现以增强理解
- **混合智能检索**：覆盖文本和多模态内容的高级搜索能力，具备上下文理解

**快速开始：**
1. 安装RAG-Anything：
   ```bash
   pip install raganything
   ```
2. 处理多模态文档：
    <details>
    <summary> <b> RAGAnything 使用示例 </b></summary>

    ```python
        import asyncio
        from raganything import RAGAnything
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        import os

        async def load_existing_lightrag():
            # 首先，创建或加载现有的 LightRAG 实例
            lightrag_working_dir = "./existing_lightrag_storage"

            # 检查是否存在之前的 LightRAG 实例
            if os.path.exists(lightrag_working_dir) and os.listdir(lightrag_working_dir):
                print("✅ Found existing LightRAG instance, loading...")
            else:
                print("❌ No existing LightRAG instance found, will create new one")

            # 使用您的配置创建/加载 LightRAG 实例
            lightrag_instance = LightRAG(
                working_dir=lightrag_working_dir,
                llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
                    "gpt-4o-mini",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key="your-api-key",
                    **kwargs,
                ),
                embedding_func=EmbeddingFunc(
                    embedding_dim=3072,
                    func=lambda texts: openai_embed(
                        texts,
                        model="text-embedding-3-large",
                        api_key=api_key,
                        base_url=base_url,
                    ),
                )
            )

            # 初始化存储（如果有现有数据，这将加载现有数据）
            await lightrag_instance.initialize_storages()

            # 现在使用现有的 LightRAG 实例初始化 RAGAnything
            rag = RAGAnything(
                lightrag=lightrag_instance,  # 传递现有的 LightRAG 实例
                # 仅需要视觉模型用于多模态处理
                vision_model_func=lambda prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs: openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]} if image_data else {"role": "user", "content": prompt}
                    ],
                    api_key="your-api-key",
                    **kwargs,
                ) if image_data else openai_complete_if_cache(
                    "gpt-4o-mini",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key="your-api-key",
                    **kwargs,
                )
                # 注意：working_dir、llm_model_func、embedding_func 等都从 lightrag_instance 继承
            )

            # 查询现有的知识库
            result = await rag.query_with_multimodal(
                "What data has been processed in this LightRAG instance?",
                mode="hybrid"
            )
            print("Query result:", result)

            # 向现有的 LightRAG 实例添加新的多模态文档
            await rag.process_document_complete(
                file_path="path/to/new/multimodal_document.pdf",
                output_dir="./output"
            )

        if __name__ == "__main__":
            asyncio.run(load_existing_lightrag())
    ```

    </details>

如需详细文档和高级用法，请参阅 [RAG-Anything 仓库](https://github.com/HKUDS/RAG-Anything)。

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

## Langfuse 可观测性集成

Langfuse 为 OpenAI 客户端提供了直接替代方案，可自动跟踪所有 LLM 交互，使开发者能够在无需修改代码的情况下监控、调试和优化其 RAG 系统。

### 安装 Langfuse 可选依赖

```
pip install lightrag-hku
pip install lightrag-hku[observability]

# 或从源代码安装并启用调试模式
pip install -e .
pip install -e ".[observability]"
```

### 配置 Langfuse 环境变量

修改 .env 文件：

```
## Langfuse 可观测性（可选）
# LLM 可观测性和追踪平台
# 安装命令: pip install lightrag-hku[observability]
# 注册地址: https://cloud.langfuse.com 或自托管部署
LANGFUSE_SECRET_KEY=""
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_HOST="https://cloud.langfuse.com"  # 或您的自托管实例地址
LANGFUSE_ENABLE_TRACE=true
```

### Langfuse 使用说明

安装并配置完成后，Langfuse 会自动追踪所有 OpenAI LLM 调用。Langfuse 仪表板功能包括：

- **追踪**：查看完整的 LLM 调用链
- **分析**：Token 使用量、延迟、成本指标
- **调试**：检查提示词和响应内容
- **评估**：比较模型输出结果
- **监控**：实时告警功能

### 重要提示

**注意**：LightRAG 目前仅把 OpenAI 兼容的 API 调用接入了 Langfuse。Ollama、Azure 和 AWS Bedrock 等 API 还无法使用 Langfuse 的可观测性功能。

## RAGAS评估

**RAGAS**（Retrieval Augmented Generation Assessment，检索增强生成评估）是一个使用LLM对RAG系统进行无参考评估的框架。我们提供了基于RAGAS的评估脚本。详细信息请参阅[基于RAGAS的评估框架](lightrag/evaluation/README.md)。

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
