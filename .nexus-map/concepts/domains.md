> generated_by: nexus-mapper v2
> verified_at: 2026-03-25
> provenance: AST-backed for Python/JavaScript/TypeScript/TSX/Bash; Bash files have module-only coverage, and WebUI internal import relations under `@/...` are supplemented by manual reading because current raw import edges treat those aliases as external.

# 核心领域概念

## 1. LightRAG 运行时

`LightRAG` 仍是仓库的中心对象。它把工作目录、查询参数、存储实现、实体关系抽取、文档状态与查询流程串到一起；现在它还负责创建 `PromptVersionStore`，并把激活 prompt 版本解析为 `active_prompt_groups` 放进运行时配置。

## 2. 四类存储契约

LightRAG 不把“数据库”看成单一后端，而是拆成四个并行契约：

- `KV_STORAGE`：缓存、文本块、文档元信息
- `VECTOR_STORAGE`：实体、关系、文本块向量
- `GRAPH_STORAGE`：实体关系图
- `DOC_STATUS_STORAGE`：文档处理状态

这套四分法仍是仓库最稳定的架构骨架，`lightrag/kg/__init__.py` 的注册表与环境变量要求都围绕它展开。

## 3. 文档摄取生命周期

典型流程仍然是：

1. 上传文件或扫描输入目录
2. 做 chunking
3. 抽取实体与关系
4. 合并到图谱与向量存储
5. 更新文档状态
6. 在查询时回收图谱、向量和上下文片段

`DocumentManager`、`document_routes.py`、`lightrag/operate.py` 与 `LightRAG` 生命周期共同承担这条链路。

## 4. 查询模式

API 暴露的查询模式仍包括：

- `local`
- `global`
- `hybrid`
- `naive`
- `mix`
- `bypass`

它们属于运行时查询策略，而不是前端专用概念。`QueryParam` 与 API 请求模型共同控制 top-k、token budget、上下文、rerank 与引用返回策略。

## 5. Prompt 配置分层

现在需要区分四种不同层级的 prompt 控制方式：

- `user_prompt`：答案阶段的附加指令
- `prompt_config`：实例级默认模板配置
- `prompt_overrides`：单次查询的结构化覆盖
- `active_prompt_groups`：按 workspace 持久化、可激活的 prompt 版本组

`lightrag/operate.py` 的真实生效顺序是：默认模板 → 实例级 `prompt_config` → 激活 retrieval 版本 → 单次请求 `prompt_overrides`。

## 6. Workspace 提示词版本组

prompt version management 现在是一个明确的领域概念，而不只是若干 API：

- `indexing` 组：管理 `ENTITY_TYPES`、`SUMMARY_LANGUAGE` 和建库期 prompt family
- `retrieval` 组：管理查询回答与关键词提取 family
- 保存版本不会自动生效，必须显式 activate
- 如果没有激活版本，系统会继续回退到内置 / 默认行为

版本数据通过 `PromptVersionStore` 持久化到 workspace 下的 `prompt_versions/registry.json`；这意味着它是运行时数据边界的一部分，而不是纯前端状态。

## 7. 模型绑定三件套

这个仓库仍把模型侧能力拆成三类绑定：

- `LLM_BINDING`
- `EMBEDDING_BINDING`
- `RERANK_BINDING`

配置向导、API 启动参数和示例代码都围绕这三类绑定展开，因此后续任何“接入新模型”工作都最好沿用这套术语，而不是把 provider 名直接散落进核心运行时。

## 8. 工作区与运行目标

仓库里仍有两个容易混淆但必须分开的概念：

- `workspace`：运行时数据隔离边界，现在也决定 prompt 版本 registry 的落盘位置
- `LIGHTRAG_RUNTIME_TARGET`：部署边界，区分 host 与 compose 输出

前者影响数据、锁和 prompt version registry，后者影响 `.env` 与 `docker-compose.final.yml` 的生成逻辑。两者不能混为一谈。

## 9. 交付面与当前激活状态可见性

LightRAG 至少有三种主要交付面：

- Python 库 / 示例脚本
- FastAPI + Ollama 兼容服务
- React WebUI

再加上 `scripts/setup/` 提供的环境生成与 compose 拼装，仓库实际上仍是“框架 + 服务 + UI + 部署向导”的复合项目；新增的是 `/health` 和 WebUI 都能直接显示当前激活 prompt 版本摘要。
