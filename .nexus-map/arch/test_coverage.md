> generated_by: nexus-mapper v2
> verified_at: 2026-03-25
> provenance: AST-backed for Python/JavaScript/TypeScript/TSX/Bash; Bash files have module-only coverage, and WebUI internal import relations under `@/...` are supplemented by manual reading because current raw import edges treat those aliases as external.

# 测试面

## 总览

- `tests/` 目录当前静态可见 47 个模块。
- `lightrag_webui/src/` 当前静态可见 5 个前端 `test` 文件，旧知识库中“未发现前端测试”的说法已不再成立。
- 默认测试策略仍偏离线执行，`tests/pytest.ini` 通过 `-m "not integration"` 排除集成测试；集成测试可经 `tests/conftest.py` 的 `--run-integration` 或环境变量开启。
- 自动化运行时，仓库规范仍优先推荐 `./scripts/test.sh`，而不是直接裸跑 `pytest`。
- 可见标记：
  - `offline`
  - `integration`
  - `requires_db`
  - `requires_api`

## 主要覆盖面

### 1. 核心编排与 Prompt 生效逻辑

- `tests/test_chunking.py`
- `tests/test_extract_entities.py`
- `tests/test_doc_status_chunk_preservation.py`
- `tests/test_unified_lock_safety.py`
- `tests/test_workspace_isolation.py`
- `tests/test_prompt_config.py`
- `tests/test_prompt_versioning.py`
- `tests/test_prompt_version_store.py`
- `tests/test_prompt_version_runtime.py`
- `tests/test_query_prompt_customization.py`

这一组现在不仅覆盖 chunking、实体抽取、工作区隔离和状态保留，也覆盖 prompt 默认模板、seed 版本、registry 原子写入、active group 解析，以及 query-time override 的生效顺序。

### 2. API、鉴权与 Prompt Config 接口

- `tests/test_aquery_data_endpoint.py`
- `tests/test_auth.py`
- `tests/test_description_api_validation.py`
- `tests/test_document_file_path_normalization.py`
- `tests/test_lightrag_ollama_chat.py`
- `tests/test_prompt_config_routes.py`
- `tests/test_query_prompt_overrides_api.py`

这组测试说明 API 层现在既覆盖原有 REST / 鉴权 / Ollama 兼容行为，也覆盖 `/prompt-config/*` 路由和 `/health` 中的 active prompt version 摘要。

### 3. 存储后端与迁移

- `tests/test_graph_storage.py`
- `tests/test_opensearch_storage.py`
- `tests/test_postgres_halfvec.py`
- `tests/test_postgres_index_name.py`
- `tests/test_postgres_migration.py`
- `tests/test_postgres_retry_integration.py`
- `tests/test_postgres_upsert.py`
- `tests/test_qdrant_migration.py`
- `tests/test_qdrant_upsert_batching.py`
- `tests/test_neo4j_fulltext_index.py`
- `tests/test_milvus_index_config.py`
- `tests/test_milvus_index_creation.py`
- `tests/test_milvus_kwargs_bridge.py`
- `tests/test_nebula_graph_storage.py`
- `tests/test_faiss_meta_inconsistency.py`
- `tests/test_dimension_mismatch.py`

这块覆盖面仍然很广，而且 `tests/test_nebula_graph_storage.py` 继续处于 Git 热点榜前列，说明 Nebula 支持依旧是高变化区。

### 4. 配置向导与部署契约

- `tests/test_interactive_setup_outputs.py`
- `tests/test_runtime_target_validation.py`

结合 Git 热点可判断：`scripts/setup/` 不是轻量辅助脚本，而是持续受回归保护的一级系统。

### 5. WebUI 组件与工具函数

- `lightrag_webui/src/features/PromptManagement.test.tsx`
- `lightrag_webui/src/utils/promptVersioning.test.ts`
- `lightrag_webui/src/utils/promptOverrides.test.ts`
- `lightrag_webui/src/utils/graphLabel.test.ts`
- `lightrag_webui/src/utils/graphProperties.test.ts`

当前前端测试面仍偏组件 / 工具函数级，但已经能为 prompt 版本编辑器、query-time override 投影和图谱工具函数提供最基本的回归锚点。

## 证据缺口

- 本次未执行 `./scripts/test.sh`、`pytest` 或 `bun test`，因此这里只能证明“存在测试面”，不能证明当前工作树全部通过。
- 前端测试目前仍以 Vitest 组件 / util 级为主，尚未看到完整的端到端 UI 流程测试证据。
- Bash 脚本没有 AST 级结构覆盖，所以 `scripts/setup/` 的测试映射仍主要依赖文件名、Git 耦合和人工阅读。
- 集成测试依赖外部数据库或 API 环境；静态分析无法替代这些端到端检查。

## 改动时优先关注

- 改 `lightrag/lightrag.py` 或 `lightrag/operate.py`：
  - 先看 `tests/test_prompt_version_runtime.py`
  - 再看 `tests/test_query_prompt_customization.py`、`tests/test_doc_status_chunk_preservation.py`
- 改 `lightrag/api/routers/prompt_config_routes.py` 或 `/health`：
  - 先看 `tests/test_prompt_config_routes.py`
  - 再看 `tests/test_query_prompt_overrides_api.py`
- 改 `lightrag/kg/nebula_impl.py`：
  - 必看 `tests/test_nebula_graph_storage.py`
- 改 `scripts/setup/setup.sh`：
  - 必看 `tests/test_interactive_setup_outputs.py`
  - 同时检查 `tests/test_runtime_target_validation.py`
- 改 WebUI prompt 管理或检索 prompt 选择：
  - 至少看 `lightrag_webui/src/features/PromptManagement.test.tsx`
  - `lightrag_webui/src/utils/promptVersioning.test.ts`
  - `lightrag_webui/src/utils/promptOverrides.test.ts`
