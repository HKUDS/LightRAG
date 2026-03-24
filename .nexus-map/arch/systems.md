> generated_by: nexus-mapper v2
> verified_at: 2026-03-24
> provenance: AST-backed for Python/JavaScript/TypeScript/TSX/Bash; Bash files have module-only coverage; setup-system conclusions additionally use manual reading of `Makefile` and `lightrag/api/README.md`.

# 系统边界

## 1. 核心编排层

- 代码位置：`lightrag/lightrag.py`、`lightrag/operate.py`、`lightrag/base.py`、`lightrag/utils.py`、`lightrag/constants.py`
- 责任：定义 `LightRAG` 生命周期、文档摄取、查询参数、工作区隔离、实体关系抽取与共享工具。
- 证据：
  - `lightrag.lightrag` 仍是高 fan-out 核心，静态影响面有 15 个上游内部依赖。
  - `lightrag.utils` 是最大 fan-in 枢纽，被 68 个模块导入。
  - `lightrag/lightrag.py` 近 90 天变更 20 次，并与 `tests/test_doc_status_chunk_preservation.py` 强耦合。
- 边界说明：这是运行时内核，不直接承载 HTTP 路由或 WebUI 页面，但被 API、存储、工具、示例和测试广泛复用。

## 2. 存储适配层

- 代码位置：`lightrag/kg/`
- 责任：实现 KV、Vector、Graph、DocStatus 四类存储契约，并维护实现注册与环境变量要求。
- 证据：
  - `lightrag/kg/__init__.py` 中 `STORAGE_IMPLEMENTATIONS` 和 `STORAGES` 明确注册了多种后端。
  - Graph 存储已包含 `NebulaGraphStorage`，对应实现 `lightrag/kg/nebula_impl.py` 与测试 `tests/test_nebula_graph_storage.py` 都进入热点榜。
  - `milvus_impl.py`、`postgres_impl.py`、`opensearch_impl.py` 仍处于高变化区。
- 边界说明：这一层依赖核心抽象和共享工具，但不承担 API 路由或前端展示职责。

## 3. 模型绑定与重排层

- 代码位置：`lightrag/llm/`、`lightrag/rerank.py`
- 责任：封装多供应商 LLM、Embedding 与 Rerank 的认证、参数桥接和兼容调用。
- 证据：
  - `lightrag/api/lightrag_server.py` 启动时会根据 binding 类型导入 `lightrag.llm.*`。
  - `lightrag/api/README.md` 明确暴露 `ollama`、`lollms`、`openai`、`azure_openai`、`aws_bedrock`、`gemini` 等绑定入口。
- 分层例外：
  - `lightrag/llm/openai.py`、`anthropic.py`、`ollama.py` 会导入 `lightrag.api.__api_version__`，说明绑定层对 API 包存在轻微反向耦合。

## 4. API 服务层

- 代码位置：`lightrag/api/`
- 责任：组装 FastAPI 应用、配置解析、鉴权、文档管理、查询接口、图谱接口、Ollama 兼容接口，以及 WebUI 静态托管。
- 证据：
  - `lightrag/api/lightrag_server.py` 是最大 fan-out 入口，内部依赖 22 个模块。
  - `create_app()` 会构造 `LightRAG`、调用 `initialize_storages()` 与 `check_and_migrate_data()`，并挂载 document/query/graph/Ollama 路由。
  - `lightrag/api/run_with_gunicorn.py` 是生产模式启动包装层，下游依赖 `lightrag_server.py`。
- 边界说明：这是面向交付的运行层，直接消费核心编排、模型绑定、共享工具和存储能力。

## 5. WebUI 前端层

- 代码位置：`lightrag_webui/src/`
- 责任：提供文档管理、知识图谱浏览、检索调试和 API 页面等交互界面。
- 证据：
  - `query_graph --summary` 显示前端当前静态可见 91 个模块。
  - `lightrag/api/README.md` 明确把 WebUI 作为服务交付面的一部分。
  - Git 热点中 `lightrag_webui/package.json` 与 `bun.lock` 仍是高频联动文件。
- 边界说明：前端源码独立于 Python 包，但部署时通常由 API 服务端托管。

## 6. 配置向导与部署胶水层

- 代码位置：`scripts/setup/`、`Makefile`、`docs/InteractiveSetup.md`
- 责任：通过 `make env-*` 引导生成 `.env`、`docker-compose.final.yml`，并管理 host/compose 运行目标、鉴权和 SSL 配置。
- 证据：
  - `scripts/setup/setup.sh` 近 90 天变更 132 次，仍是仓库第一热点。
  - `Makefile` 把 `make env-base/env-storage/env-server/env-security-check/env-backup` 固化为推荐入口。
  - `tests/test_interactive_setup_outputs.py` 与 `scripts/setup/setup.sh` 高耦合，说明该层有专门回归保护。
- 边界说明：这层不是业务运行时，但它决定部署形态和环境契约，影响 API、存储与本地交付体验。

## 7. 质量与回归保护层

- 代码位置：`tests/`
- 责任：覆盖离线单元测试、数据库/API 集成测试、迁移测试、配置向导测试和后端兼容性测试。
- 证据：
  - `tests/` 当前静态可见 40 个模块。
  - `tests/test_interactive_setup_outputs.py` 和 `tests/test_nebula_graph_storage.py` 都进入热点榜前列。
  - `tests/conftest.py` 定义 `offline`、`integration`、`requires_db`、`requires_api` 等标记及集成开关。
- 边界说明：这层不是生产系统，但在热点区域改动时是最直接的回归安全网。

## 支撑面但非一级系统

- `examples/`：示例脚本与后端接入演示。
- `lightrag/tools/`：迁移、清理和运维辅助工具。
- `lightrag/evaluation/`：评测相关能力，当前不构成主入口。
