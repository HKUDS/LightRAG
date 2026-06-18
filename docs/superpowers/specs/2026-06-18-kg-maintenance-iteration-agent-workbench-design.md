# KG Maintenance 知识库迭代 Agent 工作台设计

Date: 2026-06-18
Status: Approved for spec review

## 背景

KG Maintenance 页面需要从当前“按工具/面板分组”的控制台，重设计为面向维护者的“知识库迭代 Agent 工作台”。用户要在一个页面里看见本轮 Agent 产出的审阅包文件，理解当前阶段、质量问题、图谱快照、proposal 队列、backlog，以及 accepted/rejected 决策记忆。

本设计已确认采用 A 方案：知识库迭代 Agent 工作台。用户明确要求：不需要图谱画布。`snapshots/kg_snapshot.json` 仍然可见，但以结构摘要和原始 JSON 方式展示，不渲染节点关系图。

## 目标

1. 页面文字以中文为主，保留 KG、LLM、workspace、profile、source_id、file_path、chunk、Diff、Markdown、JSON、proposal 等技术词。
2. 让以下 9 个产物成为一等可见内容：
   - `kb_context.md`：当前 KB 摘要
   - `quality_report.md`：质量报告
   - `snapshots/kg_snapshot.json`：图谱快照
   - `snapshots/quality_score.json`：质量分数
   - `approval_queue.md`：待审批 proposal
   - `improvement_backlog.md`：改进 backlog
   - `accepted_changes.md`：已接受变更记忆
   - `rejected_changes.md`：已拒绝变更记忆
   - `iteration_log.md`：当前阶段
3. 将页面组织为审阅流程，而不是散落的文件浏览器。
4. 保持 LightRAG KG 维护安全边界：不自动修改 KG，不自动 rebuild，不自动接受 LLM 建议。

## 非目标

- 不新增图谱可视化画布。
- 不实现自动 patch apply。
- 不改 KG fact、prompt、rule 或 workspace rebuild 行为。
- 不把 LLM 输出表现为已验证医学事实。
- 不引入新的前端视觉体系或重做全站主题。

## 信息架构

页面保留三栏工作台结构，但重新定义内容。

### 顶栏

顶栏显示当前工作上下文：

- 标题：`知识库迭代 Agent`
- workspace 下拉选择
- profile / latest run 信息
- `刷新` 按钮
- `运行审阅包` 按钮
- 错误或运行状态提示

运行提示文案需要说明：本轮审阅包会更新可审阅产物，但不会自动修改 KG。

### 左侧流程栏

左侧导航按维护流程组织，不按历史组件名组织：

1. `当前阶段`：对应 `iteration_log.md`
2. `当前 KB 摘要`：对应 `kb_context.md`
3. `质量检查`：对应 `quality_report.md` 和 `quality_score.json`
4. `快照审阅`：对应 `kg_snapshot.json`
5. `Proposal 审批`：对应 `approval_queue.md`
6. `改进 backlog`：对应 `improvement_backlog.md`
7. `决策记忆`：对应 `accepted_changes.md` 和 `rejected_changes.md`
8. `LLM 审阅材料`：作为辅助审阅材料，不进入主决策自动化

默认进入 `当前阶段` 或综合 `审阅包概览`。概览应清楚标出 9 个产物是否存在、更新时间、内容类型和审阅状态。

### 主内容区

主内容区是 Agent 审阅包的主要阅读区域。

`当前阶段`：
- 展示 `iteration_log.md` 的 Markdown 内容。
- 顶部显示本轮状态、latest run、当前下一步。

`当前 KB 摘要`：
- 展示 `kb_context.md` 的 Markdown 内容。
- 摘要区显示实体数、关系数、来源数、质量分数等已有可得指标。

`质量检查`：
- 展示 `quality_report.md`。
- 展示 `quality_score.json` 的结构摘要，例如 score、severity、blocking findings、review required。
- 提供展开原始 JSON 的入口。

`快照审阅`：
- 展示 `snapshots/kg_snapshot.json` 的结构摘要和原始 JSON。
- 摘要优先显示节点数、关系数、实体类型分布、关系类型分布、snapshot id。
- 不渲染图谱画布。

`Proposal 审批`：
- 展示 `approval_queue.md`。
- 对可解析的 proposal 使用现有审批能力展示接受、拒绝、需要更多证据等动作。
- 每个动作需要填写或保留 review reason / impact scope / verification。

`改进 backlog`：
- 展示 `improvement_backlog.md`。
- 用于维护暂不处理但需要跟踪的问题。

`决策记忆`：
- 同屏或分段展示 `accepted_changes.md` 和 `rejected_changes.md`。
- 已接受变更突出 reason、impact scope、verification。
- 已拒绝变更突出拒绝原因，帮助 Agent 避免重复建议。

`LLM 审阅材料`：
- 保留现有 LLM review report、generated proposals、patch candidates、judge report 展示能力。
- 明确标注为“辅助材料”，不自动修改 KG。

### 右侧审阅栏

右侧栏展示上下文敏感信息：

- 当前阶段摘要：来自 `iteration_log.md`
- 选中 proposal 的审批细节
- 与当前内容相关的 evidence / Diff / patch candidate
- 安全边界说明：所有事实和规则变更都需要人工审批

当未选中 proposal 或 evidence 时，右侧显示简洁空状态，而不是空白。

## 数据流

前端加载 workspace 后并行请求现有 KB iteration API 和 artifact API。

需要新增或补齐前端读取的 artifact：

- `getKBIterationArtifact(workspace, 'kb_context')`
- `getKBIterationArtifact(workspace, 'kg_snapshot')`
- `getKBIterationArtifact(workspace, 'quality_score')`

已存在但需要在新工作台里重新组织展示的内容：

- `approval_queue`
- `improvement_backlog`
- `iteration_log`
- `rules` 返回的 accepted / rejected 记忆
- quality、summary、diff、LLM review artifacts

如果 API artifact key 与文件路径命名不一致，实现时以现有 API 白名单和后端类型为准，不在前端硬编码未验证路径。

## 交互和状态

Loading：
- 使用内容骨架或紧凑状态条，不用页面中央大 spinner。

Empty：
- 没有 workspace：提示选择或运行 KB iteration。
- 产物不存在：显示文件名、中文用途、以及“运行审阅包后生成”。
- JSON 为空或不可解析：保留原始内容入口，并显示解析失败提示。

Error：
- 顶部显示错误摘要和重试按钮。
- 单个 artifact 失败时，尽量保留其他 artifact 可读。

Approval：
- 审批动作必须显式。
- 接受/拒绝后刷新审阅包数据。
- 不将 patch candidate 应用到源码或 KG。

Responsive：
- 桌面：左流程栏 + 主内容 + 右审阅栏。
- 窄屏：流程栏变为顶部分段导航，右侧审阅栏下移。

## 视觉方向

采用 restrained product UI：

- 保留现有 Tailwind/theme token。
- 使用紧凑、清晰、低装饰的工作台布局。
- 主强调色仅用于当前阶段、主操作和状态标记。
- 不使用营销式 hero、装饰渐变、玻璃拟态或大型图形背景。
- 卡片只用于具体文件产物、proposal 或状态块，不做嵌套装饰卡片。

## 组件边界

推荐实现时拆分为小组件：

- `KGMaintenanceShell`：顶栏、三栏布局、流程导航。
- `IterationWorkbenchOverview`：审阅包概览和 9 个产物状态。
- `ArtifactMarkdownPanel`：Markdown 产物展示。
- `ArtifactJsonPanel`：JSON 摘要、原始 JSON、解析失败状态。
- `IterationStagePanel`：`iteration_log.md` 当前阶段。
- `DecisionMemoryPanel`：accepted / rejected 记忆。
- `ApprovalPanel`：保留现有审批行为并中文化。
- `LLMReviewPanels`：保留为辅助材料区域并中文化。

实际文件拆分应遵循现有 `lightrag_webui/src/components/kg-maintenance/` 结构，避免一次性重写无关组件。

## 测试计划

先写前端测试，再实现：

1. `KGMaintenanceShell.test.tsx`
   - 断言中文流程导航可见。
   - 断言没有主流程图谱入口文案。
2. 新增或更新工作台概览测试
   - 断言 9 个产物中文名称和文件名可见。
   - 断言 `kg_snapshot.json` 和 `quality_score.json` 以 JSON/摘要方式展示。
3. 数据加载测试
   - 断言 `kb_context`、`kg_snapshot`、`quality_score` artifact 被请求并传入 UI。
4. 审批测试
   - 保留 proposal decision 行为。
   - 断言中文审批文案和刷新行为。

实现后运行：

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/KGMaintenanceShell.test.tsx
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/kgMaintenanceData.test.ts src/stores/kgMaintenance.test.ts
npx --yes bun run lint
npx --yes bun run build
```

如果启动 WebUI 进行视觉验证，使用本地浏览器检查桌面和窄屏布局，确认文本不溢出、右侧栏不会遮挡主内容。

## 验收标准

- KG Maintenance 首屏为中文 `知识库迭代 Agent` 工作台。
- 用户能看到 9 个指定产物的中文用途和文件名。
- `kb_context.md`、`kg_snapshot.json`、`quality_score.json` 不再缺席。
- 主页面没有图谱画布。
- `kg_snapshot.json` 以结构摘要和原始 JSON 方式可读。
- proposal 审批仍然需要人工显式操作。
- LLM 审阅输出被标为辅助材料。
- 前端 focused tests、lint、build 通过。
