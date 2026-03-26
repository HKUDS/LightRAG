> generated_by: nexus-mapper v2
> verified_at: 2026-03-25
> provenance: AST-backed for Python/JavaScript/TypeScript/TSX/Bash; Bash files have module-only coverage, and WebUI internal import relations under `@/...` are supplemented by manual reading because current raw import edges treat those aliases as external.

# 系统边界

## 1. 核心编排层

- 代码位置：`lightrag/lightrag.py`、`lightrag/operate.py`、`lightrag/base.py`、`lightrag/utils.py`、`lightrag/constants.py`、`lightrag/prompt.py`、`lightrag/prompt_versions.py`、`lightrag/prompt_version_store.py`
- 责任：定义 `LightRAG` 生命周期、文档摄取、查询流程、实体关系抽取、工作区隔离，以及激活 prompt 版本到运行时配置的解析逻辑。
- 证据：
  - `lightrag.lightrag` 仍是高 fan-out 核心，静态依赖 16 个内部模块。
  - `lightrag.utils` 是最大 fan-in 枢纽，被 70 个模块导入。
  - `LightRAG` 初始化阶段会创建 `PromptVersionStore`，`_build_runtime_global_config()` 会写入 `active_prompt_groups`。
  - `lightrag/operate.py` 会先应用激活 retrieval 版本，再应用单次请求 `prompt_overrides`；激活 indexing 版本还能覆盖 `summary_language` 与 `entity_types`。
  - `lightrag/lightrag.py` 近 90 天变更 22 次，`lightrag/operate.py` 变更 17 次，说明核心运行时仍处在高演化区。
- 边界说明：这是运行时内核，不直接承载 HTTP 路由或 WebUI 页面，但现在已经承担 prompt 版本的生效语义。

## 2. 存储适配层

- 代码位置：`lightrag/kg/`
- 责任：实现 KV、Vector、Graph、DocStatus 四类存储契约，并维护实现注册、环境变量约束与共享存储状态。
- 证据：
  - `lightrag/kg/__init__.py` 中 `STORAGE_IMPLEMENTATIONS` 和 `STORAGES` 继续集中注册多种后端。
  - `NebulaGraphStorage` 仍是最显著的图存储扩展面，`lightrag/kg/nebula_impl.py` 与 `tests/test_nebula_graph_storage.py` 继续处于高耦合热点。
  - `milvus_impl.py`、`postgres_impl.py`、`opensearch_impl.py` 仍位于热点榜前列，说明主力后端都还在频繁演进。
- 边界说明：这一层依赖核心抽象和命名空间，但不承载 API 路由或前端展示。

## 3. 模型绑定与重排层

- 代码位置：`lightrag/llm/`、`lightrag/rerank.py`
- 责任：封装多供应商 LLM、Embedding 与 Rerank 的认证、参数桥接和兼容调用。
- 证据：
  - `lightrag/api/lightrag_server.py` 启动时会根据 binding 类型导入 `lightrag.llm.*` 与 `lightrag.rerank`。
  - `lightrag.llm.openai` 仍是高 fan-in provider 模块，被 16 个内部模块导入。
- 分层例外：
  - `lightrag/llm/openai.py`、`anthropic.py`、`ollama.py` 仍会导入 `lightrag.api.__api_version__`，说明绑定层对 API 包存在轻微反向耦合。

## 4. API 服务层

- 代码位置：`lightrag/api/`
- 责任：组装 FastAPI 应用、配置解析、鉴权、文档管理、查询接口、图谱接口、Ollama 兼容接口、prompt 版本管理接口，以及 WebUI 静态托管。
- 证据：
  - `lightrag.api.lightrag_server` 是当前最大 fan-out 入口，内部依赖 23 个模块。
  - `create_app()` 现在会显式挂载 `create_prompt_config_routes(rag, api_key)`。
  - `/health` 会汇总 `configuration.active_prompt_versions`，把当前激活的 indexing / retrieval 版本摘要暴露给客户端。
  - `lightrag/api/run_with_gunicorn.py` 仍是生产模式启动包装层，下游依赖 `lightrag_server.py`。
- 边界说明：这是面向交付的运行层，直接消费核心运行时、模型绑定和存储能力。

## 5. WebUI 前端层

- 代码位置：`lightrag_webui/src/`
- 责任：提供文档管理、知识图谱浏览、检索调试、API 页面，以及正式的 prompt 版本管理界面。
- 证据：
  - `query_graph --summary` 显示前端当前静态可见 108 个模块，高于旧知识库中的 91 个模块。
  - `App.tsx` 新增 `prompt-management` 顶层页签，并挂载 `PromptManagement` 功能页。
  - `PromptManagement.tsx` 负责初始化、列出、保存、激活、删除和 diff prompt 版本；`RetrievalTesting.tsx` 负责临时选择 retrieval 版本或 `Custom / Draft` 覆盖。
  - `lightrag_webui/src/` 现在已经出现 5 个 `test` 文件，说明前端不再是“无测试证据”区域。
- 边界说明：前端源码独立于 Python 包，但部署时通常由 API 服务端托管；前端内部依赖关系部分来自人工阅读，因为 `@/` 别名没有被原始 AST 边解析出来。

## 6. 配置向导与部署胶水层

- 代码位置：`scripts/setup/`、`Makefile`、`docs/InteractiveSetup.md`
- 责任：通过 `make env-*` 引导生成 `.env`、`docker-compose.final.yml`，并管理 host / compose 运行目标、鉴权和 SSL 配置。
- 证据：
  - `scripts/setup/setup.sh` 近 90 天变更 133 次，仍是仓库第一热点。
  - `Makefile` 继续把 `make env-base/env-storage/env-server/env-security-check/env-backup` 固化为推荐入口。
  - `tests/test_interactive_setup_outputs.py` 与 `scripts/setup/setup.sh` 的耦合分数仍高达 0.738，是最关键的回归锚点。
- 边界说明：这层不是业务运行时，但它决定 API、存储和本地交付体验的环境契约。

## 7. 质量与回归保护层

- 代码位置：`tests/`、`lightrag_webui/src/*.test.*`
- 责任：覆盖核心运行时、prompt versioning、API、存储后端、配置向导与前端关键组件/工具函数的回归保护。
- 证据：
  - `tests/` 当前静态可见 47 个模块，比旧知识库记录的 40 个模块更多。
  - backend 端新增 `test_prompt_version_store.py`、`test_prompt_version_runtime.py`、`test_prompt_config_routes.py`、`test_query_prompt_overrides_api.py` 等 prompt 版本测试。
  - frontend 端当前可见 `PromptManagement.test.tsx`、`promptVersioning.test.ts`、`promptOverrides.test.ts`、`graphLabel.test.ts`、`graphProperties.test.ts`。
- 边界说明：这层不是生产系统，但它是当前最直接的变更风险缓冲区，尤其覆盖配置向导、Nebula 存储和 prompt 版本化新能力。

## 跨系统能力：Workspace Prompt Version Management

- 核心锚点：`lightrag/prompt.py`、`lightrag/prompt_versions.py`、`lightrag/prompt_version_store.py`
- API 锚点：`lightrag/api/routers/prompt_config_routes.py`、`lightrag/api/lightrag_server.py#/health`
- UI 锚点：`lightrag_webui/src/features/PromptManagement.tsx`、`lightrag_webui/src/components/prompt-management/`、`lightrag_webui/src/components/retrieval/RetrievalPromptVersionSelector.tsx`
- 这项能力没有单独上升为一级系统：它的持久化、生效和展示都复用了既有的核心运行时、API 服务和 WebUI 边界，而不是拥有独立启动面。

## 支撑面但非一级系统

- `examples/`：示例脚本与后端接入演示。
- `lightrag/tools/`：迁移、清理和运维辅助工具。
- `lightrag/evaluation/`：评测相关能力，当前不构成主入口。
