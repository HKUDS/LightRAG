> generated_by: nexus-mapper v2
> verified_at: 2026-03-25
> provenance: AST-backed for Python/JavaScript/TypeScript/TSX/Bash; Bash files have module-only coverage, and WebUI internal import relations under `@/...` are supplemented by manual reading because current raw import edges treat those aliases as external.

# LightRAG 知识库索引

LightRAG 现在是一个“核心运行时 + 多后端存储 + FastAPI 服务 + React WebUI + 交互式配置向导 + workspace 级 prompt 版本管理”的复合仓库。

## 一眼看懂

- 主运行时仍由 `lightrag/lightrag.py` 中的 `LightRAG` 驱动，但现在额外挂载 `PromptVersionStore`，把 `indexing` / `retrieval` 两组激活版本注入 `global_config.active_prompt_groups`。
- 服务主入口仍是 `lightrag/api/lightrag_server.py`，并新增 `lightrag/api/routers/prompt_config_routes.py`；`/health` 现在会暴露 `configuration.active_prompt_versions`。
- `lightrag/kg/` 继续是四类存储契约的实现中心，Nebula 图存储仍是高风险热点，但本轮最大的跨系统能力新增并不在存储，而在 prompt 版本化。
- `lightrag_webui/src/` 已扩展到 108 个静态模块，新增 `Prompt Management` 顶层页签，并在 `Retrieval` 页支持临时选择 retrieval 版本或 `Custom / Draft` 覆盖。
- `scripts/setup/` 仍是仓库第一热点；配置向导、环境模板和配套测试依旧是最高风险联动区。

## 关键事实

- `lightrag.utils` 仍是最大共享底座，静态 fan-in 升到 70 个内部模块。
- `lightrag.api.lightrag_server` 现在是 23 个内部依赖的最大 fan-out 入口，新增依赖 `lightrag.api.routers.prompt_config_routes`。
- prompt 版本管理是 workspace 级持久化能力：核心定义在 `lightrag/prompt.py`、`lightrag/prompt_versions.py`、`lightrag/prompt_version_store.py`，运行时在 `lightrag/operate.py` 先应用激活 retrieval 版本，再允许单次请求 `prompt_overrides` 覆盖它。
- 前端测试证据已经出现：`lightrag_webui/src/` 当前至少有 5 个 `test` 文件，旧知识库里“未发现前端测试”的结论已失效。
- API 启动面仍需分清两层：`lightrag-server` 直接走 `lightrag_server.py`，`lightrag-gunicorn` 通过 `run_with_gunicorn.py` 复用同一应用工厂。

## 测试面速览

- `tests/` 当前静态可见 47 个模块，新增 prompt 版本相关测试簇：`test_prompt_versioning.py`、`test_prompt_version_store.py`、`test_prompt_version_runtime.py`、`test_prompt_config_routes.py`、`test_query_prompt_overrides_api.py` 等。
- `lightrag_webui/src/` 目前静态可见 5 个前端测试文件，已覆盖 `PromptManagement` 和多个 prompt / graph 工具函数。
- 当前知识库只做静态测试面分析，未实际执行 `./scripts/test.sh`、`pytest` 或 `bun test`。

## 证据缺口

- AST 结果仍被截断，当前 `truncated_nodes=14816`，因此函数级细节并不完整；本知识库主要依赖 Module/Class 结构、hub 分析、Git 热点和定向文件阅读。
- Bash 只有 Module 级覆盖，所以 `scripts/setup/` 的依赖关系仍主要来自 `Makefile`、文档与 Git 耦合，而非细粒度 AST 边。
- WebUI 使用 `@/` 别名导入，当前 `query_graph.py` 会把这些边视为 external；因此前端页面之间的组合关系依赖人工阅读 `App.tsx`、`SiteHeader.tsx`、`RetrievalTesting.tsx` 与 `PromptManagement.tsx` 补证。

## [操作指南] 强制执行步骤

> 本节是对所有读取本文件的 AI 发出的硬性操作指令，不是建议。

### 步骤1 — 必须先读完以下所有文件（顺序不限）

读完本文件后，在执行任何任务之前，必须依次 read 以下文件完整内容：

- `.nexus-map/arch/systems.md` — 系统边界与代码位置
- `.nexus-map/arch/dependencies.md` — 系统间依赖关系与 Mermaid 图
- `.nexus-map/arch/test_coverage.md` — 测试面与证据缺口
- `.nexus-map/hotspots/git_forensics.md` — Git 热点与耦合风险
- `.nexus-map/concepts/domains.md` — 核心领域概念

> 这些文件均为高密度摘要，总量通常 < 5000 tokens，是必要的上下文成本。
> 不得以"任务简单"或"只改一个文件"为由跳过。

### 步骤2 — 按任务类型追加操作（步骤1 完成后执行）

- 若任务涉及**接口修改、新增跨模块调用、删除/重命名公共函数**：
  → 必须运行 `query_graph.py --impact <目标文件>` 确认影响半径后再写代码。
- 若任务需要**判断某文件被谁引用**：
  → 运行 `query_graph.py --who-imports <模块名>`。
- 若仓库结构已发生重大变化（新增系统、重构模块边界）或 prompt 版本管理边界继续外扩：
  → 任务完成后评估是否需要重新运行 nexus-mapper 更新知识库。
