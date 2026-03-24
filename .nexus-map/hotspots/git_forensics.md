> generated_by: nexus-mapper v2
> verified_at: 2026-03-24
> provenance: Derived from `.nexus-map/raw/git_stats.json` over the most recent 90 days.

# Git 热点与耦合

## 热点结论

- 近 90 天的主热点仍然不是单纯算法文件，而是“配置向导 + 配套测试 + 环境模板 + 交付文档”。
- 当前仓库的高变更面主要有三条：
  - 部署与配置体验：`scripts/setup/`、`env.example`、`docs/InteractiveSetup.md`
  - 图存储扩展：`lightrag/kg/nebula_impl.py`、`tests/test_nebula_graph_storage.py`
  - 运行时核心与老牌存储实现：`lightrag/lightrag.py`、`lightrag/kg/milvus_impl.py`、`postgres_impl.py`、`opensearch_impl.py`

## Top Hotspots

1. `scripts/setup/setup.sh` — 132 次变更，`high`
2. `tests/test_interactive_setup_outputs.py` — 107 次变更，`high`
3. `env.example` — 63 次变更，`high`
4. `lightrag_webui/package.json` — 45 次变更，`high`
5. `scripts/setup/lib/file_ops.sh` — 45 次变更，`high`
6. `lightrag_webui/bun.lock` — 44 次变更，`high`
7. `README.md` — 36 次变更，`high`
8. `README-zh.md` — 29 次变更，`high`
9. `scripts/setup/lib/validation.sh` — 26 次变更，`high`
10. `tests/test_nebula_graph_storage.py` — 25 次变更，`high`
11. `docs/InteractiveSetup.md` — 21 次变更，`high`
12. `lightrag/kg/nebula_impl.py` — 20 次变更，`high`
13. `lightrag/lightrag.py` — 20 次变更，`high`
14. `lightrag/kg/milvus_impl.py` — 17 次变更，`high`
15. `lightrag/kg/postgres_impl.py` — 16 次变更，`high`
16. `lightrag/kg/opensearch_impl.py` — 16 次变更，`high`
17. `Makefile` — 16 次变更，`high`
18. `lightrag/api/routers/document_routes.py` — 15 次变更，`high`
19. `lightrag/operate.py` — 15 次变更，`high`
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
- `scripts/setup/lib/file_ops.sh` ↔ `tests/test_interactive_setup_outputs.py`
  - `co_changes=31`
  - `coupling_score=0.689`
  - 含义：配置输出格式与测试快照边界很紧。
- `lightrag/kg/nebula_impl.py` ↔ `tests/test_nebula_graph_storage.py`
  - `co_changes=20`
  - `coupling_score=1.00`
  - 含义：Nebula 图存储支持仍在快速演化，改实现就几乎一定要改测试。

## 风险解释

- 高风险不等于“代码差”，而是意味着改动频繁、联动多、回归面大。
- 当前最危险的错误假设是：
  - 认为 `scripts/setup/` 只是部署脚本，可以随手改
  - 认为 `lightrag/kg/nebula_impl.py` 只是新增后端，不会牵动测试和存储契约
  - 认为 `lightrag/lightrag.py` 只影响 Python SDK，不会牵动 API 与存储行为

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
- 改 `lightrag/lightrag.py`：
  - 先看 `tests/test_doc_status_chunk_preservation.py`
  - 再看工作区和 chunking 相关测试
