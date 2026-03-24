> generated_by: nexus-mapper v2
> verified_at: 2026-03-24
> provenance: AST-backed for code locations; concept summaries are synthesized from README, API README, storage registry, setup targets, tests, and selected source files.

# 核心领域概念

## 1. LightRAG 运行时

`LightRAG` 是仓库的中心对象。它把工作目录、查询参数、存储实现、工作区隔离、实体关系抽取、文档状态与查询流程串到一起，并通过 `initialize_storages()` 建立运行态。

## 2. 四类存储契约

LightRAG 不把“数据库”看成单一后端，而是拆成四个并行契约：

- `KV_STORAGE`：缓存、文本块、文档元信息
- `VECTOR_STORAGE`：实体、关系、文本块向量
- `GRAPH_STORAGE`：实体关系图
- `DOC_STATUS_STORAGE`：文档处理状态

这套四分法仍是仓库最稳定的架构骨架，`lightrag/kg/__init__.py` 的注册表与环境变量要求都围绕它展开。

## 3. 文档摄取生命周期

从 README、API 路由和运行时实现看，典型流程是：

1. 上传文件或扫描输入目录
2. 做 chunking
3. 抽取实体与关系
4. 合并到图谱与向量存储
5. 更新文档状态
6. 在查询时回收图谱、向量和上下文片段

`DocumentManager`、`document_routes.py`、`lightrag/operate.py` 与 `LightRAG` 生命周期共同承担这条链路。

## 4. 查询模式

API 暴露的查询模式包括：

- `local`
- `global`
- `hybrid`
- `naive`
- `mix`
- `bypass`

它们属于运行时查询策略，而不是前端专用概念。`QueryParam` 与 API 请求模型共同控制 top-k、token budget、上下文、rerank 与引用返回策略。

## 5. 模型绑定三件套

这个仓库把模型侧能力拆成三类绑定：

- `LLM_BINDING`
- `EMBEDDING_BINDING`
- `RERANK_BINDING`

配置向导、API 启动参数和示例代码都围绕这三类绑定展开，因此后续任何“接入新模型”工作都最好沿用这套术语，而不是把 provider 名直接散落进核心运行时。

## 6. 工作区与运行目标

仓库里有两个容易混淆但必须分开的概念：

- `workspace`：运行时数据隔离边界
- `LIGHTRAG_RUNTIME_TARGET`：部署边界，区分 host 与 compose 输出

前者影响数据和锁，后者影响 `.env` 与 `docker-compose.final.yml` 的生成逻辑。两者不能混为一谈。

## 7. 存储后端扩展面

存储层不再只是传统的 `NetworkX`、`Postgres`、`Neo4j`、`Milvus`、`Qdrant`、`MongoDB`、`Redis` 和 `OpenSearch`。当前 `NebulaGraphStorage` 已经进入实现和测试热点，这意味着图存储扩展仍在继续演化。

## 8. 交付面

LightRAG 至少有三种主要交付面：

- Python 库 / 示例脚本
- FastAPI + Ollama 兼容服务
- React WebUI

再加上 `scripts/setup/` 提供的环境生成与 compose 拼装，仓库实际上已经是“框架 + 服务 + UI + 部署向导”的复合项目。
