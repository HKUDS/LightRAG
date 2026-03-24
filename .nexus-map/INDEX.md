> generated_by: nexus-mapper v2
> verified_at: 2026-03-24
> provenance: AST-backed for Python/JavaScript/TypeScript/TSX/Bash; Bash files have module-only coverage; extraction used a one-off local compatibility workaround to remove an unused built-in `csharp` parser mapping because the current `tree-sitter-language-pack` does not provide it, and this repository contains no `.cs` files.

# LightRAG 知识库索引

LightRAG 是一个“核心编排 + 多后端存储 + FastAPI 服务 + React WebUI + 交互式配置向导”的复合仓库。

## 一眼看懂

- 主运行时入口仍是 `lightrag/lightrag.py` 中的 `LightRAG`，负责工作目录、查询参数、文档摄取、实体关系抽取和存储初始化。
- 服务交付面以 `lightrag/api/lightrag_server.py` 为主入口；生产模式通过 `lightrag/api/run_with_gunicorn.py` 包装同一套应用工厂。
- `lightrag/kg/` 继续是四类存储契约的实现中心，而且 `NebulaGraphStorage` 已经进入真实实现与测试热点，不再只是规划项。
- `scripts/setup/` 仍是一级系统，`make env-base/env-storage/env-server/...` 是推荐入口，它决定 `.env`、`docker-compose.final.yml` 和 host/compose 运行目标。
- `lightrag_webui/src/` 是独立前端应用，当前静态能看到 91 个模块，但仍未发现前端测试文件证据。

## 关键事实

- `lightrag.utils` 仍是最大共享底座，静态扇入为 68 个内部模块。
- `lightrag.api.lightrag_server` 是最大 fan-out 入口，内部依赖 22 个模块。
- Git 近 90 天热点依旧由配置向导主导，但新增了 `tests/test_nebula_graph_storage.py` 与 `lightrag/kg/nebula_impl.py` 这组高耦合热点，说明图存储能力正在继续外扩。
- API 启动面要分清两层：`lightrag-server` 直接走 `lightrag_server.py`，`lightrag-gunicorn` 则通过 `run_with_gunicorn.py` 复用同一应用工厂。
- 一个分层例外仍然存在：部分 `lightrag/llm/*.py` 会导入 `lightrag.api.__api_version__`，因此模型绑定层并非完全独立于 API 包。

## 测试面速览

- `tests/` 当前静态可见 40 个模块，覆盖核心运行时、API、存储后端、配置向导与迁移流程。
- `tests/test_interactive_setup_outputs.py` 仍是配置向导最重要的回归锚点；`tests/test_nebula_graph_storage.py` 已成为新的高频测试文件。
- 当前知识库只做静态测试面分析，未实际执行 `./scripts/test.sh`、`pytest` 或 `bun test`。

## 证据缺口

- AST 结果仍被截断，当前 `truncated_nodes=13660`，因此函数级细节并不完整；本知识库主要依赖 Module/Class 结构、hub 分析、Git 热点和定向文件阅读。
- Bash 只有 Module 级覆盖，所以 `scripts/setup/` 的依赖关系主要来自 `Makefile`、文档与 Git 耦合，而非细粒度 AST 边。
- `lightrag_webui/src/` 未发现 `test/spec` 文件，若后续改 UI，仍需单独检查 Bun 测试侧是否在别处定义。

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
- 若仓库结构已发生重大变化（新增系统、重构模块边界）：
  → 任务完成后评估是否需要重新运行 nexus-mapper 更新知识库。
