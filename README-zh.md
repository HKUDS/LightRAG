<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# 🚀 LightRAG: 简单且快速的检索增强生成（RAG）框架

<div align="center">
    <a href="https://trendshift.io/repositories/13043" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13043" alt="HKUDS%2FLightRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>
<p>
</p>
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
      <a href="README-ja.md"><img src="https://img.shields.io/badge/🇯🇵日本語版-1a1a2e?style=for-the-badge"></a>
    </p>
    <p>
      <a href="https://pepy.tech/projects/lightrag-hku"><img src="https://static.pepy.tech/personalized-badge/lightrag-hku?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads"></a>
      <a href="https://hvtracker.net/agents/lightrag/"><img src="https://hvtracker.net/badge/lightrag.svg"></a>
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
- [2026.05]🎯[新功能]：**将 RagAnything 合并至 LightRAG**🎉。支持通过 **MinerU / Docling** 服务进行多模态内容解析与提取。
- [2026.05]🎯[新功能]：引入四种可选的文本分块策略：`Fix`（固定）、`Recursive`（递归）、`Vector`（向量）和 `Paragraph`（段落语义）。
- [2026.05]🎯[新功能]：**支持按角色配置 LLM**，提供四个独立角色：EXTRACT、QUERY、KEYWORDS 和 VLM，每个角色拥有独立的 LLM 设置。
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
# 启动 API-WebUI 服务。默认绑定所有网络接口(0.0.0.0)。
# 安全提示:对外网暴露前,请在 .env 中配置认证(LIGHTRAG_API_KEY,或
# AUTH_ACCOUNTS 搭配 TOKEN_SECRET);若仅需本机访问,可绑定 127.0.0.1;
# 否则所有接口都将公开可访问。
# 注意:为兼容 Ollama 客户端,/api/* 路由默认不鉴权;如需对其启用认证,
# 请将 WHITELIST_PATHS 收窄为 /health。
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

### 使用设置向导创建 .env 文件

除了手动编辑 `env.example` 之外，您还可以使用交互式向导生成配置好的 `.env`，并在需要时生成 `docker-compose.final.yml`：

```bash
make env-base           # 必跑第一步：配置 LLM、Embedding、Reranker
make env-storage        # 可选：配置存储后端和数据库服务
make env-server         # 可选：配置服务端口、鉴权和 SSL
make env-base-rewrite   # 可选：强制重建向导托管的 compose 服务块
make env-storage-rewrite # 可选：强制重建向导托管的 compose 服务块
make env-security-check # 可选：审计当前 .env 中的安全风险
```

设置向导工具的详细说明请参阅 [docs/InteractiveSetup.md](./docs/InteractiveSetup.md)。

## 关于LightRAG

### 基于图的轻量级RAG框架

**LightRAG** 是一个轻量级的知识图谱 RAG 框架，被视为 Microsoft GraphRAG 的高效替代方案。它采用双层架构来同时管理知识图谱（KG）和向量嵌入，完美填补了传统基于向量的 RAG 与基于图谱的 RAG 之间的技术鸿沟。LightRAG专为高扩展性而设计，有效地解决了大规模图谱索引和查询时计算开销大、响应缓慢以及增量更新成本高等问题；LightRAG在支持大规模数据集的同时，即使搭载 30B开源大语言模型（LLM），也能保持极高的RAG质量。

### 特点与优势

1. **深度上下文理解**：通过图结构索引，LightRAG 能够捕捉实体间复杂的语义依赖关系，克服了传统分块检索方法上下文割裂的缺陷。在需要全局理解或逻辑推理的垂直领域（如法律、金融），其生成质量与上下文感知能力尤为突出。
2. **卓越的全面性与多样性**：LightRAG的双层检索机制使其能够同时整合详细事实与抽象概念，让其在查询结果全面性（Comprehensiveness）和多样性（Diversity）取得卓越的成绩，有效应对复杂的跨文档查询。
3. **极高的检索效率与低成本**：LightRAG不需要依赖低效的社区报告和复杂查询时的多跳推理，大幅度减少了索引和查询阶段对LLM的调用，显著减少了响应延迟与LLM计算成本。
4. **快速适应动态数据**：LightRAG 支持无缝的增量知识库更新。新数据只需经过标准的图索引流程生成局部图谱，即可通过集合合并的方式直接融入现有图谱，无需破坏原有结构或重建全局索引，保证了系统在动态数据环境下的时效性。删除文档时可以利用构建阶段的LLM缓存快速重建受影响的实体关系，大幅度提高了知识库更新效率。

### 多模态能力的升级

从 LightRAG v1.5 版本开始，该框架正式引入了对多模态文档的分析和检索能力：

* **多引擎文档解析：** 其文件处理流水线（Pipeline）支持使用 MinerU、Docling 和 Native 文档解析引擎，可高效提取文档中的文字、表格、公式和图片。
* **跨模态实体与关系映射：** 在统一的框架内实现跨模态的实体提取和关系映射，从而达成无缝的索引与查询。
* **应用场景提升：** 全新的多模态处理流水线能够大幅提高操作说明书、学术论文等含有丰富多模态内容文档的 RAG 质量。

### LightRAG API 服务器

LightRAG 服务器不仅提供给了一个供出选择体验LightRAG功能的Web UI，还提供了一个完整的 `REST API`。有关LightRAG服务器的更多信息，请参阅[LightRAG服务器](./docs/LightRAG-API-Server-zh.md)。

![iShot_2025-03-23_12.40.08](./README.assets/iShot_2025-03-23_12.40.08.png)

## 关键配置说明

### LLM 模型的选择

LightRAG 的工作过程中需要使用到 4 种角色的 LLM/VLM。应该为不同角色的 LLM 配置不同能力和速度的模型，以获得速度和能力之间的平衡。LightRAG 对大型语言模型（LLM）的能力要求会高于传统 RAG，因为它需要 LLM 执行文档中的实体关系抽取任务。在查询阶段，LLM 模型需要处理 LightRAG 召回的实体、关系和文本块等大量信息，需要模型具备在含有噪声的长上下文中作出高质量回答的能力。详细的模型配置请参见 [RoleSpecificLLMConfiguration-zh.md](./docs/RoleSpecificLLMConfiguration-zh.md)

### 查询模式的选择

LightRAG 支持 4 种查询模式：

- **local**：聚焦于局部上下文与具体实体的精准匹配。在知识图谱中检索对应的候选实体及其直接关联属性，适用于针对特定对象、具体概念或细节事实的问答，能够提供高度相关且细致的局部上下文支持。
- **global**：侧重于宏观主题、跨文档推理与实体间的深层关系。检索覆盖广泛主题与概念的关系链，适用于需要跨多个上下文进行总结、趋势分析或理解复杂语义依赖关系的查询。
- **hybrid**：融合 local 和 global 两种模式的检索结果。通过同时召回具体实体与全局关系上下文，进行综合推理与生成。
- **naive**：基于文本块的传统 RAG 检索，不使用知识图谱，直接依赖向量相似性在原始文本块中进行检索。
- **mix**：全功能模式，融合 local、global 和 naive 三种模式的检索结果，提供最为丰富和全面的检索结果。

LightRAG 的默认查询模式为 mix。使用 mix 模式通常可以获得最为理想的查询结果。mix 模式比 naive 耗时略长；其他查询模式在耗时上基本相当。

### Embedding 模型

在选择 Embedding 模型的时候需要注意其对多语言的支持能力。LightRAG 的检索质量对 Embedding 模型的依赖有限，因此建议尽量选择低维度和速度快的模型。通常 `BAAI/bge-m3` 已经足够使用。建议尽量本地部署 Embedding 模型，以获得最好的性能。

**重要提示**：在文档索引前必须确定使用的 Embedding 模型，且在文档查询阶段必须沿用与索引阶段相同的模型。嵌入模型一旦选定通常就不能修改。如果修改的话，需要对所有文本块、实体和关系进行重新嵌入。LightRAG 目前没有提供重新嵌入的工具。有些存储（例如 PostgreSQL）在首次建立数据表的时候需要确定向量维度，因此更换 Embedding 模型后需要删除向量相关库表，以便让 LightRAG 重建新的库表。

### 开启 Rerank 选项

查询阶段开启 Rerank 选项可以显著提高查询的质量。开启 Rerank 通常会引入 1～2 秒的延时。为了降低延时，建议尽量在本地部署 Rerank 模型。Rerank 的相关配置方式请参考 `.env.example` 文件。Rerank 模型与 Embedding 模型不同，可以在查询阶段随时更换。

### 文档处理流水线的配置

LightRAG 的默认流水线配置并不能让系统发挥最好的性能。文件内容解析的好坏会极大地影响文档的索引和查询效果。因此建议配置流水线开启 MinerU 文件解析引擎，并开启流水线的图片分析功能。建议添加的配置为：

```
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R

VLM_PROCESS_ENABLE=true
VLM_LLM_MODEL=<your_vlm_model_name>
```

由于云端的 MinerU 服务有使用量、文件大小和页数等限制，建议使用本地部署的 MinerU。文件处理流水线的具体配置方法请参考 [FileProcessingPipeline-zh.md](./docs/FileProcessingPipeline-zh.md)

### 文件处理并发优化

对于大规模的文档处理，需要提高文档处理的并发能力。几个涉及文件并发处理性能的关键环境变量包括：

- **MAX_ASYNC_LLM/EXTRACT_ASYNC_LLM**：控制 LLM 模型的最大并发数。
- **MAX_PARALLEL_INSERT**：控制并行处理文件的最大数量。单个文件内的文本、表格、公式、图片之间的处理也会并发进行。`MAX_PARALLEL_INSERT` 应该为 `MAX_ASYNC_LLM` 的 1/3 左右为宜。
- **MAX_PARALLEL_PARSE_MINERU**：控制 MinerU 文件解析的并发处理文件数。
- **MAX_PARALLEL_PARSE_DOCLING**：控制 Docling 文件解析的并发处理文件数。
- **EMBEDDING_FUNC_MAX_ASYNC**：控制嵌入模型的最大并发数。
- **EMBEDDING_BATCH_NUM**：控制每个嵌入模型请求包含的待嵌入文本的数量（每批做多少个嵌入）；提高这个数量可以大幅度减少调用嵌入模型的次数，提高嵌入存储的落盘速度。

```
# 设置示例
MAX_ASYNC_LLM=8
MAX_PARALLEL_INSERT=3
EMBEDDING_FUNC_MAX_ASYNC=16
EMBEDDING_BATCH_NUM=32
```

### 后台存储的选择

LightRAG 需要使用到 4 种后台存储类型，分别是：

- **KV_STORAGE**：用于保存 LLM 响应缓存、文本分块结果、实体关系提取结果等信息。
- **VECTOR_STORAGE**：用于保存文本块、实体和关系的向量信息。
- **GRAPH_STORAGE**：用于保存知识图谱。
- **DOC_STATUS_STORAGE**：用于保存文件列表。

LightRAG 的默认存储全部都是基于文件进行持久化的内存数据库。默认存储仅用于开发调试，不适合用于生产环境部署。生产环境如果希望使用同一个后台数据解决 4 种类型的后台存储，可以选择 PostgreSQL、MongoDB 或 OpenSearch。也可以单独为向量存储或图存储选择专业化的数据库，例如使用 Milvus 或 Qdrant 作为向量存储，使用 Neo4j 或 Memgraph 作为图存储。

### 文档处理阶段其他重要配置

在文档插入阶段还有以下环境变量建议根据实际需要进行调整：

- **SUMMARY_LANGUAGE**：控制 LLM 输出实体关系名称和摘要时使用的语言，例如：`Chinese`, `English`。
- **ENTITY_EXTRACTION_USE_JSON**：控制 LLM 输出实体关系的时候是否使用 JSON 格式。使用 JSON 格式通常可以获得更加稳定的效果，但是输出需要消耗更多的 Token，速度也会略微慢一些。
- **ENABLE_CONTENT_HEADINGS**：控制查询阶段是否把文本块所属章节标题信息送给LLM（默认允许，为LLM提供更多的上下文信息）
- **FORCE_LLM_SUMMARY_ON_MERGE / MAX_SOURCE_IDS_PER_RELATION**：控制每个`实体/关系`能够最多与多少个文本块保持关联
- **SOURCE_IDS_LIMIT_METHOD**：控制`实体/关系`关联文本块超过限制后是否继续更新实体关系的描述（默认不再更新，因为此时实体关系的描述已经足够丰富，继续更新的意义不大；放弃更新可以极大地提高知识库的构建速度）
- **DEFAULT_MAX_FILE_PATHS**：控制`实体/关系`关联的原始文件的最大数量，超过这个数量之后新的文件名不再写入到向量存储。
- **OPENAI_LLM_MAX_TOKENS / OPENAI_LLM_MAX_COMPLETION_TOKENS**: 为了解决循环输出或输出太多实体关系导致LLM调用超时问题，可以设置LLM模型最大的输出token数量。不同LLM供应商需要设置不同参数，详见`env.example`中的说明。

### 文档查询阶段其他重要配置

在文档查询阶段还有以下环境变量建议根据实际需要进行调整：
- **MAX_ENTITY_TOKENS / MAX_RELATION_TOKENS / MAX_TOTAL_TOKENS**：控制召回内容送给LLM上下文的Token长度。召回内容包含`实体`、`关系`和`文本块`三部分，实体和关系的长度可以单独控制长度，文本块的长度由总长度减去实体和关系的长度来控制。
- **ENABLE_CONTENT_HEADINGS**：控制是否把文本块所在的章节标题送给LLM；默认开启，可以为LLM提供更加丰富的上下文信息，提高回答质量。
- **ENABLE_LLM_CACHE**：是否允许缓存查询结果。默认开启，相同的查询问题、查询模式、LLM模型参数将返回相同的结果。

## 使用LightRAG SDK

> ⚠️ **如果您希望将LightRAG集成到您的项目中，建议您使用LightRAG Server提供的REST API**。LightRAG SDK通常用于嵌入式应用，或供希望进行研究与评估的学者使用。

### 安装LightRAG SDK

* 从源代码安装

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

### LightRAG SDK示例代码

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

### 使用SDK的注意事项

SDK的使用说明详见 **[docs/ProgramingWithCore.md](./docs/ProgramingWithCore.md)**（英文）。有部份LightRAG功能没有提供 REST API，仅能够通过SDK使用。这部份功能往往是不稳定，不能保证在将来的版本上可以兼容。

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
