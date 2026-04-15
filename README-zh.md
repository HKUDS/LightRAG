<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# 🚀 LightRAG: 简单且快速的检索增强生成（RAG）框架

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

<div align="center">
  <table>
    <tr>
      <td style="vertical-align: middle;">
        <img src="./assets/LiteWrite.png"
             width="56"
             height="56"
             alt="LiteWrite"
             style="border-radius: 12px;" />
      </td>
      <td style="vertical-align: middle; padding-left: 12px;">
        <a href="https://litewrite.ai">
          <img src="https://img.shields.io/badge/🚀%20LiteWrite-AI%20原生%20LaTeX%20编辑器-ff6b6b?style=for-the-badge&logoColor=white&labelColor=1a1a2e">
        </a>
      </td>
    </tr>
  </table>
</div>

---

## 🎉 新闻
- [2026.03]🎯[新功能]: 集成了 **OpenSearch** 作为统一存储后端，为 LightRAG 的全部四种存储类型提供全面支持。
- [2026.03]🎯[新功能]: 推出交互式安装向导，支持通过 Docker 在本地部署 Embedding、Reranking 及存储后端服务。
- [2025.11]🎯[新功能]: 集成了 **RAGAS 评估**和 **Langfuse 追踪**。更新了 API 以在查询结果中返回召回上下文，支持上下文精度指标。
- [2025.10]🎯[可扩展性增强]: 消除了处理瓶颈，以高效支持**大规模数据集**。
- [2025.09]🎯[新功能]: 显著提升了 Qwen3-30B-A3B 等**开源 LLM** 的知识图谱提取准确性。
- [2025.08]🎯[新功能]: 现已支持 **Reranker**，显著提升混合查询性能（已设为默认查询模式）。
- [2025.08]🎯[新功能]: 添加了**文档删除**功能，并支持自动重新生成知识图谱，以确保最佳查询性能。
- [2025.06]🎯[新发布]: 我们的团队发布了 [RAG-Anything](https://github.com/HKUDS/RAG-Anything) —— 一个用于无缝处理文本、图像、表格和方程式的**全功能多模态 RAG** 系统。
- [2025.06]🎯[新功能]: LightRAG 现已集成 [RAG-Anything](https://github.com/HKUDS/RAG-Anything)，支持全面的多模态数据处理，实现对 PDF、图像、Office 文档、表格和公式等多种格式的无缝文档解析和 RAG 能力。详见[多模态文档处理部分](https://github.com/HKUDS/LightRAG/?tab=readme-ov-file#multimodal-document-processing-rag-anything-integration)。
- [2025.03]🎯[新功能]: LightRAG 现已支持引用功能，实现了准确的源归因和增强的文档可追溯性。
- [2025.02]🎯[新功能]: 现在您可以使用 MongoDB 作为一体化存储解决方案，实现统一的数据管理。
- [2025.02]🎯[新发布]: 我们的团队发布了 [VideoRAG](https://github.com/HKUDS/VideoRAG) —— 一个用于理解超长上下文视频的 RAG 系统。
- [2025.01]🎯[新发布]: 我们的团队发布了 [MiniRAG](https://github.com/HKUDS/MiniRAG)，使用小型模型简化 RAG。
- [2025.01]🎯现在您可以使用 PostgreSQL 作为一体化存储解决方案进行数据管理。
- [2024.11]🎯[新资源]: LightRAG 的综合指南现已在 [LearnOpenCV](https://learnopencv.com/lightrag) 上发布 —— 探索深入的教程和最佳实践。非常感谢博客作者的杰出贡献！
- [2024.11]🎯[新功能]: 推出 LightRAG WebUI —— 一个允许您通过直观的 Web 界面插入、查询和可视化 LightRAG 知识的仪表板。
- [2024.11]🎯[新功能]: 现在您可以[使用 Neo4J 进行存储](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#using-neo4j-for-storage) —— 开启图数据库支持。
- [2024.10]🎯[新功能]: 我们添加了 [LightRAG 介绍视频](https://youtu.be/oageL-1I0GE) 的链接 —— 演示 LightRAG 的各项功能。感谢作者的杰出贡献！
- [2024.10]🎯[新频道]: 我们创建了一个 [Discord 频道](https://discord.gg/yF2MmDJyGJ)！💬 欢迎加入我们的社区进行分享、讨论和协作！ 🎉🎉

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

**💡 使用 uv 进行包管理**: 本项目使用 [uv](https://docs.astral.sh/uv/) 进行快速可靠的 Python 包管理。首先安装 uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (Unix/macOS) 或 `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows)

> **注意**：如果您愿意，也可以使用 pip，但为了获得更好的性能 and 更可靠的依赖管理，建议使用 uv。
>
> **📦 离线部署**: 对于离线或隔离环境，请参阅[离线部署指南](./docs/OfflineDeployment.md)，了解预安装所有依赖项和缓存文件的说明。

### 安装LightRAG服务器

LightRAG服务器旨在提供Web UI和API支持。Web UI便于文档索引、知识图谱探索和简单的RAG查询界面。LightRAG服务器还提供兼容Ollama的接口，旨在将LightRAG模拟为Ollama聊天模型。这使得AI聊天机器人（如Open WebUI）可以轻松访问LightRAG。

* 从PyPI安装

```bash
### 使用 uv 安装 LightRAG 服务器（作为工具，推荐)
uv tool install "lightrag-hku[api]"

### 或使用 pip
# python -m venv .venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
# pip install "lightrag-hku[api]"

### 构建前端代码
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# 配置 env 文件
# 从 GitHub 仓库的根目录上下载 env.example 文件
# 或从本地检出的源代码中获取 env.example 文件
cp env.example .env  # 使用你的LLM和Embedding模型访问参数更新.env文件
# 启动API-WebUI服务
lightrag-server
```

* 从源代码安装

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG

# 一键初始化开发环境（推荐）
make dev
source .venv/bin/activate  # 激活虚拟环境 (Linux/macOS)
# Windows 系统: .venv\Scripts\activate

# make dev 会安装测试工具链以及完整的离线依赖栈
# （API、存储后端与各类 Provider 集成），并构建前端；不会生成 .env。
# 启动服务前请先运行 make env-base，或手动从 env.example 复制并配置 .env。

# 使用 uv 的等价手动步骤
# 注意: uv sync 会自动在 .venv/ 目录创建虚拟环境
uv sync --extra test --extra offline
source .venv/bin/activate  # 激活虚拟环境 (Linux/macOS)
# Windows 系统: .venv\Scripts\activate

### 或使用 pip 和虚拟环境
# python -m venv .venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
# pip install -e ".[test,offline]"

# 构建前端代码
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# 配置 env 文件
make env-base  # 或: cp env.example .env 后手动修改
# 启动API-WebUI服务
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
>
> 由 GitHub Actions 发布到 GHCR 的官方镜像已使用 GitHub OIDC 和 Sigstore Cosign 进行签名。校验方式请参阅 [docs/DockerDeployment.md](./docs/DockerDeployment.md#verify-official-ghcr-images-with-cosign)。

### 使用 Setup 工具创建 .env 文件

除了手动编辑 `env.example` 之外，您还可以使用交互式向导生成配置好的 `.env`，并在需要时生成 `docker-compose.final.yml`：

```bash
make env-base           # 必跑第一步：配置 LLM、Embedding、Reranker
make env-storage        # 可选：配置存储后端和数据库服务
make env-server         # 可选：配置服务端口、鉴权和 SSL
make env-base-rewrite   # 可选：强制重建向导托管的 compose 服务块
make env-storage-rewrite # 可选：强制重建向导托管的 compose 服务块
make env-security-check # 可选：审计当前 .env 中的安全风险
```

每个目标的详细说明请参阅 [docs/InteractiveSetup.md](./docs/InteractiveSetup.md)。
这些 setup 向导只负责更新配置；如需在部署前审计当前 `.env` 的安全风险，请额外运行
`make env-security-check`。
默认情况下，重新运行 setup 会保留未变化的向导托管 compose 服务块；只有在需要按模板强制重建这些托管块时，才使用
`*-rewrite` 目标。

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

LightRAG 服务器旨在提供 Web UI 和 API 支持，同时提供了全面的知识图谱可视化功能，支持各种重力布局、节点查询、子图过滤等。有关LightRAG服务器的更多信息，请参阅[LightRAG服务器](./docs/LightRAG-API-Server-zh.md)。

![iShot_2025-03-23_12.40.08](./README.assets/iShot_2025-03-23_12.40.08.png)


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

完整的 Core API 参考 —— 包括初始化参数、`QueryParam`、各 LLM/Embedding 接入示例（OpenAI、Ollama、Azure、Gemini、HuggingFace、LlamaIndex）、Rerank 注入、插入操作、实体/关系管理、删除与合并 —— 详见 **[docs/ProgramingWithCore.md](./docs/ProgramingWithCore.md)**（英文）。

> ⚠️ **如果您希望将LightRAG集成到您的项目中，建议您使用LightRAG Server提供的REST API**。LightRAG Core通常用于嵌入式应用，或供希望进行研究与评估的学者使用。

### 高级功能

LightRAG 提供 Token 用量追踪、知识图谱数据导出、LLM 缓存管理、Langfuse 可观测性集成和基于 RAGAS 的评估框架。详见 **[docs/AdvancedFeatures.md](./docs/AdvancedFeatures.md)**（英文）。

### 多模态文档处理（RAG-Anything 集成）

LightRAG 与 [RAG-Anything](https://github.com/HKUDS/RAG-Anything) 集成，支持对 PDF、Office 文档、图像、表格和公式的端到端多模态 RAG。详见 **[docs/AdvancedFeatures.md](./docs/AdvancedFeatures.md)**（英文）。

> LightRAG Server 将会在不久的将来把 RAG-Anything 的多模态处理能力整合到其文件件处理流水线中。敬请期待。

## 重现论文结果

LightRAG 在农业、计算机科学、法律和混合等领域均显著优于 NaiveRAG、RQ-RAG、HyDE 和 GraphRAG。完整评估方法论、提示词和复现步骤详见 **[docs/Reproduce.md](./docs/Reproduce.md)**（英文）。

### 总体性能表

||**农业**||**计算机科学**||**法律**||**混合**||
|----------------------|---------------|------------|------|------------|---------|------------|-------|------------|
||NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|
|**全面性**|32.4%|**67.6%**|38.4%|**61.6%**|16.4%|**83.6%**|38.8%|**61.2%**|
|**多样性**|23.6%|**76.4%**|38.0%|**62.0%**|13.6%|**86.4%**|32.4%|**67.6%**|
|**赋能性**|32.4%|**67.6%**|38.8%|**61.2%**|16.4%|**83.6%**|42.8%|**57.2%**|
|**总体**|32.4%|**67.6%**|38.8%|**61.2%**|15.2%|**84.8%**|40.0%|**60.0%**|
||RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|
|**全面性**|31.6%|**68.4%**|38.8%|**61.2%**|15.2%|**84.8%**|39.2%|**60.8%**|
|**多样性**|29.2%|**70.8%**|39.2%|**60.8%**|11.6%|**88.4%**|30.8%|**69.2%**|
|**赋能性**|31.6%|**68.4%**|36.4%|**63.6%**|15.2%|**84.8%**|42.4%|**57.6%**|
|**总体**|32.4%|**67.6%**|38.0%|**62.0%**|14.4%|**85.6%**|40.0%|**60.0%**|
||HyDE|**LightRAG**|HyDE|**LightRAG**|HyDE|**LightRAG**|HyDE|**LightRAG**|
|**全面性**|26.0%|**74.0%**|41.6%|**58.4%**|26.8%|**73.2%**|40.4%|**59.6%**|
|**多样性**|24.0%|**76.0%**|38.8%|**61.2%**|20.0%|**80.0%**|32.4%|**67.6%**|
|**赋能性**|25.2%|**74.8%**|40.8%|**59.2%**|26.0%|**74.0%**|46.0%|**54.0%**|
|**总体**|24.8%|**75.2%**|41.6%|**58.4%**|26.4%|**73.6%**|42.4%|**57.6%**|
||GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|
|**全面性**|45.6%|**54.4%**|48.4%|**51.6%**|48.4%|**51.6%**|**50.4%**|49.6%|
|**多样性**|22.8%|**77.2%**|40.8%|**59.2%**|26.4%|**73.6%**|36.0%|**64.0%**|
|**赋能性**|41.2%|**58.8%**|45.2%|**54.8%**|43.6%|**56.4%**|**50.8%**|49.2%|
|**总体**|45.2%|**54.8%**|48.0%|**52.0%**|47.2%|**52.8%**|**50.4%**|49.6%|


## 🔗 相关项目

*生态与扩展*

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/HKUDS/RAG-Anything">
          <div style="width: 100px; height: 100px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%); border-radius: 15px; border: 1px solid rgba(0, 217, 255, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
            <span style="font-size: 32px;">📸</span>
          </div>
          <b>RAG-Anything</b><br>
          <sub>多模态 RAG</sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/HKUDS/VideoRAG">
          <div style="width: 100px; height: 100px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%); border-radius: 15px; border: 1px solid rgba(0, 217, 255, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
            <span style="font-size: 32px;">🎥</span>
          </div>
          <b>VideoRAG</b><br>
          <sub>极端长上下文视频 RAG</sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/HKUDS/MiniRAG">
          <div style="width: 100px; height: 100px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%); border-radius: 15px; border: 1px solid rgba(0, 217, 255, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
            <span style="font-size: 32px;">✨</span>
          </div>
          <b>MiniRAG</b><br>
          <sub>极简 RAG</sub>
        </a>
      </td>
    </tr>
  </table>
</div>

---

## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date)](https://star-history.com/#HKUDS/LightRAG&Date)

## 🤝 贡献

<div align="center">
  我们欢迎各种形式的贡献——Bug 修复、新功能、文档改进等。<br>
  提交 Pull Request 前，请阅读 <a href=".github/CONTRIBUTING.md"><strong>贡献指南</strong></a>。
</div>

<br>

<div align="center">
  我们感谢所有贡献者做出的宝贵贡献。
</div>

<div align="center">
  <a href="https://github.com/HKUDS/LightRAG/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=HKUDS/LightRAG" style="border-radius: 15px; box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);" />
  </a>
</div>


## 📖 引用

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

---

<div align="center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 30px; margin: 30px 0;">
  <div>
    <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500">
  </div>
  <div style="margin-top: 20px;">
    <a href="https://github.com/HKUDS/LightRAG" style="text-decoration: none;">
      <img src="https://img.shields.io/badge/⭐%20在%20GitHub%20上点亮星星-1a1a2e?style=for-the-badge&logo=github&logoColor=white">
    </a>
    <a href="https://github.com/HKUDS/LightRAG/issues" style="text-decoration: none;">
      <img src="https://img.shields.io/badge/🐛%20报告问题-ff6b6b?style=for-the-badge&logo=github&logoColor=white">
    </a>
    <a href="https://github.com/HKUDS/LightRAG/discussions" style="text-decoration: none;">
      <img src="https://img.shields.io/badge/💬%20讨论-4ecdc4?style=for-the-badge&logo=github&logoColor=white">
    </a>
  </div>
</div>

<div align="center">
  <div style="width: 100%; max-width: 600px; margin: 20px auto; padding: 20px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%); border-radius: 15px; border: 1px solid rgba(0, 217, 255, 0.2);">
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px;">
      <span style="font-size: 24px;">⭐</span>
      <span style="color: #00d9ff; font-size: 18px;">感谢您访问 LightRAG!</span>
      <span style="font-size: 24px;">⭐</span>
    </div>
  </div>
</div>
