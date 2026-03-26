> generated_by: nexus-mapper v2
> verified_at: 2026-03-25
> provenance: Derived from `.nexus-map/raw/git_stats.json` over the most recent 90 days; hotspot interpretation is supplemented by current architecture inspection for prompt version management and WebUI testing changes.

# Git 热点与耦合

## 热点结论

- 近 90 天的主热点仍然不是单纯算法文件，而是“配置向导 + 配套测试 + 环境模板 + 交付文档”。
- 当前仓库的高变更面主要仍有三条：
  - 部署与配置体验：`scripts/setup/`、`env.example`、`docs/InteractiveSetup.md`
  - 图存储扩展：`lightrag/kg/nebula_impl.py`、`tests/test_nebula_graph_storage.py`
  - 运行时核心：`lightrag/lightrag.py`、`lightrag/operate.py` 以及与其相邻的主力存储实现
- 新增的 prompt 版本管理能力在结构上已经很重要，但在 90 天 Git 热点榜里还不算“老牌热点”；这意味着它的风险更多来自跨系统边界，而不是长期 co-change 历史。

## Top Hotspots

1. `scripts/setup/setup.sh` — 133 次变更，`high`
2. `tests/test_interactive_setup_outputs.py` — 107 次变更，`high`
3. `env.example` — 64 次变更，`high`
4. `lightrag_webui/package.json` — 45 次变更，`high`
5. `scripts/setup/lib/file_ops.sh` — 45 次变更，`high`
6. `lightrag_webui/bun.lock` — 44 次变更，`high`
7. `README.md` — 37 次变更，`high`
8. `README-zh.md` — 31 次变更，`high`
9. `tests/test_nebula_graph_storage.py` — 28 次变更，`high`
10. `scripts/setup/lib/validation.sh` — 27 次变更，`high`
11. `lightrag/kg/nebula_impl.py` — 23 次变更，`high`
12. `lightrag/lightrag.py` — 22 次变更，`high`
13. `docs/InteractiveSetup.md` — 21 次变更，`high`
14. `lightrag/operate.py` — 17 次变更，`high`
15. `lightrag/kg/milvus_impl.py` — 17 次变更，`high`
16. `lightrag/kg/postgres_impl.py` — 16 次变更，`high`
17. `lightrag/kg/opensearch_impl.py` — 16 次变更，`high`
18. `Makefile` — 16 次变更，`high`
19. `lightrag/api/routers/document_routes.py` — 15 次变更，`high`
20. `docs/DockerDeployment.md` — 15 次变更，`high`

## 强耦合对

- `scripts/setup/setup.sh` ↔ `tests/test_interactive_setup_outputs.py`
  - `co_changes=79`
  - `coupling_score=0.738`
  - 含义：改配置向导时，测试同步变化非常频繁。
- `lightrag_webui/bun.lock` ↔ `lightrag_webui/package.json`
  - `co_changes=44`
  - `coupling_score=1.00`
  - 含义：前端依赖升级几乎总是成对出现。
- `scripts/setup/lib/file_ops.sh` ↔ `scripts/setup/setup.sh`
  - `co_changes=33`
  - `coupling_score=0.733`
  - 含义：向导主脚本和文件输出逻辑持续联动演化。
- `README-zh.md` ↔ `README.md`
  - `co_changes=29`
  - `coupling_score=0.935`
  - 含义：中英文文档仍然需要同步维护，产品功能说明改动通常是双语联动。
- `scripts/setup/lib/validation.sh` ↔ `scripts/setup/setup.sh`
  - `co_changes=25`
  - `coupling_score=0.926`
  - 含义：配置校验与主向导逻辑高度绑定。
- `lightrag/kg/nebula_impl.py` ↔ `tests/test_nebula_graph_storage.py`
  - `co_changes=23`
  - `coupling_score=1.00`
  - 含义：Nebula 图存储支持仍在快速演化，改实现几乎一定要改测试。

## 新兴风险区

- `prompt version management` 还没有进入 Git 热点前 20，但它已经横跨 `lightrag/`、`lightrag/api/`、`lightrag_webui/src/` 和测试层。
- 这类“结构重要、历史热度还低”的区域，最容易让人误判成“小改动”；实际上它的风险来自语义一致性：seed 版本、active version、query-time override、`/health` 摘要和 UI 状态必须保持同一套规则。

## 风险解释

- 高风险不等于“代码差”，而是意味着改动频繁、联动多、回归面大。
- 当前最危险的错误假设是：
  - 认为 `scripts/setup/` 只是部署脚本，可以随手改。
  - 认为 `lightrag/kg/nebula_impl.py` 只是新增后端，不会牵动测试和存储契约。
  - 认为 prompt 版本化文件不在热点榜里，就不会牵动核心运行时、API、UI 和文档。
  - 认为 `lightrag/lightrag.py` 只影响 Python SDK，不会牵动 API 与存储行为。

## 后续改动建议

- 改配置向导：
  - 连看 `scripts/setup/setup.sh`
  - `scripts/setup/lib/file_ops.sh`
  - `scripts/setup/lib/validation.sh`
  - `tests/test_interactive_setup_outputs.py`
  - `docs/InteractiveSetup.md`
- 改 Nebula 图存储：
  - 必看 `lightrag/kg/nebula_impl.py`
  - `tests/test_nebula_graph_storage.py`
- 改 prompt 版本化：
  - 必看 `lightrag/prompt.py`
  - `lightrag/prompt_versions.py`
  - `lightrag/prompt_version_store.py`
  - `lightrag/lightrag.py`
  - `lightrag/operate.py`
  - `lightrag/api/routers/prompt_config_routes.py`
  - `lightrag_webui/src/features/PromptManagement.tsx`
  - `tests/test_prompt_version_runtime.py`
  - `tests/test_prompt_config_routes.py`
- 改 `lightrag/lightrag.py`：
  - 先看 `tests/test_doc_status_chunk_preservation.py`
  - 再看 prompt / workspace / chunking 相关测试
