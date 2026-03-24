> generated_by: nexus-mapper v2
> verified_at: 2026-03-24
> provenance: Static test-surface analysis only; no tests were executed in this run.

# 测试面

## 总览

- `tests/` 目录当前静态可见 40 个模块。
- 默认测试策略偏离线执行，`tests/pytest.ini` 通过 `-m "not integration"` 排除集成测试；集成测试可经 `tests/conftest.py` 的 `--run-integration` 或环境变量开启。
- 自动化运行时，仓库规范优先推荐 `./scripts/test.sh`，而不是直接裸跑 `pytest`。
- 可见标记：
  - `offline`
  - `integration`
  - `requires_db`
  - `requires_api`

## 主要覆盖面

### 1. 核心编排与运行时

- `tests/test_chunking.py`
- `tests/test_extract_entities.py`
- `tests/test_doc_status_chunk_preservation.py`
- `tests/test_unified_lock_safety.py`
- `tests/test_write_json_optimization.py`
- `tests/test_workspace_isolation.py`
- `tests/test_workspace_migration_isolation.py`
- `tests/test_workspace_sanitization.py`

这些测试覆盖 chunking、实体抽取、文档状态保留、锁安全、JSON 写入优化和工作区隔离，是核心运行时的主要静态保护网。

### 2. API、鉴权与接口行为

- `tests/test_aquery_data_endpoint.py`
- `tests/test_auth.py`
- `tests/test_description_api_validation.py`
- `tests/test_document_file_path_normalization.py`
- `tests/test_lightrag_ollama_chat.py`

这些文件说明 API 层既测 REST 数据接口，也测鉴权、路径规范化与 Ollama 兼容行为。

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

这块覆盖面很广，而且 `tests/test_nebula_graph_storage.py` 已进入 Git 热点榜前列，说明 Nebula 支持正在快速演化。

### 4. 配置向导与部署契约

- `tests/test_interactive_setup_outputs.py`
- `tests/test_runtime_target_validation.py`

结合 Git 热点可判断：`scripts/setup/` 不是轻量辅助脚本，而是有持续回归保护的一级系统。

### 5. 模型与调用兼容性

- `tests/test_batch_embeddings.py`
- `tests/test_rerank_chunking.py`
- `tests/test_token_auto_renewal.py`
- `tests/test_no_model_suffix_safety.py`
- `tests/test_zhipu_llm.py`
- `tests/test_llm_cache_tools_opensearch.py`

这些测试指向模型调用、缓存工具和 provider 兼容层。

## 证据缺口

- 本次未执行 `./scripts/test.sh`、`pytest` 或 `bun test`，因此这里只能证明“存在测试面”，不能证明当前工作树全部通过。
- `lightrag_webui/src/` 下仍未发现前端 `test/spec` 文件；若改动 WebUI，需要单独确认 Bun 测试是否位于别处或尚未建立。
- Bash 脚本没有 AST 级结构覆盖，所以 `scripts/setup/` 的测试映射主要依赖文件名、Git 耦合和人工阅读。
- 集成测试依赖外部数据库或 API 环境；静态分析无法替代这些端到端检查。

## 改动时优先关注

- 改 `lightrag/lightrag.py`：
  - 先看 `tests/test_doc_status_chunk_preservation.py`
  - 再看 chunking / workspace / lock safety 相关测试
- 改 `lightrag/api/routers/document_routes.py` 或 `query_routes.py`：
  - 先看 `tests/test_aquery_data_endpoint.py`
  - 再看 `tests/test_document_file_path_normalization.py`、`tests/test_lightrag_ollama_chat.py`
- 改 `lightrag/kg/nebula_impl.py`：
  - 必看 `tests/test_nebula_graph_storage.py`
- 改 `scripts/setup/setup.sh`：
  - 必看 `tests/test_interactive_setup_outputs.py`
  - 同时检查 `tests/test_runtime_target_validation.py`
