# KG 维护控制台 Web 界面设计

## 目标

在 LightRAG WebUI 中新增一个面向知识库维护者的 `KG Maintenance` 工作台。它不是单纯的图谱展示页，而是医学知识库从“看清楚当前图谱”到“发现质量问题、审阅证据、批准改进、比较迭代结果”的完整控制台。

第一验收场景是当前流感医学知识库：

- Workspace: `influenza_medical_v1`
- Profile: `clinical_guideline_zh`
- 输入目录: `data/inputs/influenza_medical_v1`
- 存储目录: `data/rag_storage/influenza_medical_v1`
- 迭代产物目录: `work/kb-iteration/influenza_medical_v1`

设计必须保留未来扩展到其他医学文档和其他 workspace 的能力，不能把界面写死为只服务流感。

## 使用者与核心任务

主要使用者是知识库维护者。使用者不是每天只看一张漂亮图，而是在持续维护一个医学 KG：

1. 看当前知识库是否健康。
2. 看图谱是否符合人类理解的医学层级。
3. 检查实体、关系、证据和来源是否可靠。
4. 判断 agent 提出的改进建议是否应该批准。
5. 在重建或规则变更后比较质量是否真的提升。

界面成功标准是：使用者能够快速回答三个问题。

```text
现在知识库里有什么？
哪里不合理，为什么？
我应该批准什么改动，改完是否真的变好了？
```

## 产品定位

采用 A 方案：`KG 维护控制台`。

理由：

- 图谱可读性和质量迭代同等重要，不能把其中一项做成附属功能。
- 现有 `knowledge-graph` tab 应保留通用图谱能力；医学 KG 维护需要更明确的工作流。
- KB iteration agent 已经能产生 snapshot、catalog、quality、proposal、diff 等文件型产物，适合被组织成一个审阅控制台。
- 医学安全要求决定所有事实、提示词、本体、层级、关系、重建类操作必须清楚进入审批流。

## 非目标

第一版 Web 控制台不直接成为全自动医学知识编辑器。

不做：

- 自动写入原文未支持的医学事实。
- 自动修改提示词、本体、关系规则、层级规则或 WebUI 展示规则。
- 自动删除、清空、重建 workspace。
- 把 LLM 分析当作医学证据。
- 以炫技视觉效果替代证据、来源和审阅状态。

## 信息架构

新增顶部 tab：

```text
Documents | Config | Knowledge Graph | KG Maintenance | Retrieval | API
```

`KG Maintenance` 内部采用左侧工作流导航、中心主工作区、右侧详情/证据/审批面板。

左侧导航：

```text
工作台
- 总览
- 医学图谱
- 实体目录
- 关系目录

审阅
- 证据审阅
- 质量报告
- 审批队列

迭代
- 运行记录
- Diff 对比
- 规则记忆
```

这个结构比把所有内容塞进旧 GraphViewer 更合适，因为每个页面都有明确任务，不会让图谱、质量报告、审批和 diff 互相抢空间。

## 页面设计

### 1. 总览

总览是进入控制台后的默认页，回答“当前知识库状态如何”。

必须包含：

- Workspace 名称、profile、最近运行时间、当前阶段。
- 文档数量、节点数、关系数、chunk 数。
- 总体质量分与子分数。
- 证据覆盖率。
- 待审批数量。
- 高风险问题数量。
- 最近一次 `iteration_log.md` 摘要。
- 快捷操作：运行审阅、打开质量报告、打开审批队列、比较上一轮。

总览不应使用夸张的大数字 dashboard。数字需要服务审阅，而不是制造营销感。每个指标都应该能点击进入对应详情。

### 2. 医学图谱

医学图谱是控制台的核心视图，解决当前图谱“节点缺乏层次、关系含糊、节点大小一致”的问题。

视图模式：

- `医学层级`：默认模式。按疾病中心、一级医学分类、二级分组、事实叶子组织。
- `原始抽取`：保留 LightRAG 原始 KG，帮助排查抽取结果。
- `证据视图`：突出 source、chunk、证据覆盖状态。
- `质量视图`：突出可疑节点、可疑关系、孤立节点、泛化关系。

医学层级规则：

- 疾病中心节点最大。
- 一级医学分类节点作为视觉分区，例如病原体、临床表现、诊断检查、治疗、预防、人群、指南证据。
- 二级分组节点作为结构中间层，例如 `临床表现 -> 全身症状 -> 高热不退`。
- 事实叶子节点必须来自原文支持，或者明确显示证据不足。
- 组织性节点和事实性节点视觉上要能区分，但不在界面上使用“系统生成”这类标签打扰使用者。

节点大小：

- 不能全部一致。
- 大小由医学角色、层级深度、连接度、证据数量、质量风险共同决定。
- 疾病中心 > 一级分类 > 二级分组 > 事实叶子。
- 高风险或证据不足不能靠变大表达，应使用状态标记或质量高亮，避免把问题误读为重要性。

关系显示：

- 禁止使用 `邻接` 作为主要关系文案。
- 有 relation keyword 时显示语义关系，例如 `临床表现：高热不退`。
- 有方向时显示 `outgoing 临床表现 -> 高热不退` 或 `incoming 病原导致 <- 流感病毒`。
- 缺失关键词时显示 `未标注关系：节点名`，并进入质量提示。
- 图谱画布可以保持视觉无向，但详情、tooltip、关系列表必须保留语义方向。

图例：

- 放在右下角或右侧可折叠面板。
- 说明颜色、节点大小、节点形状、关系类型、证据状态、质量状态。
- 图例必须能解释“为什么这个节点更大”“为什么这条关系是风险项”。

### 3. 实体目录

实体目录服务不想在图里寻找的维护任务。

必须包含：

- 按实体类型分组。
- 搜索、排序、过滤。
- 代表性实体。
- 可疑实体。
- 同义词或规范名提示。
- 所属医学层级位置。
- 来源覆盖摘要。

可疑实体类型：

- 纯数值或剂量节点。
- 时间片段节点。
- 页码、表格碎片、低信息节点。
- 同义词拆分，例如 `流感` 与 `流行性感冒`。
- 没有证据或来源缺失的实体。

### 4. 关系目录

关系目录回答“KG 中有哪些语义关系，是否清楚可靠”。

必须包含：

- 关系类型统计。
- 关系三元组列表。
- source、relation、target、方向。
- 来源文档与证据。
- 泛化关系统计，例如 `相关`、空关系、未标注关系。
- source/target 类型一致性检查。
- 高风险关系筛选。

主格式：

```text
source - relation -> target
```

即使图谱画布视觉上不画箭头，关系目录也必须保留方向。

### 5. 证据审阅

证据审阅是医学 KG 的信任基础。

点击任意节点或关系时，右侧详情面板必须显示：

- 名称。
- 类型。
- 描述。
- 关系方向。
- 来源文档。
- source_id。
- file_path。
- chunk 或段落预览。
- 证据覆盖状态。
- 是否为事实边或组织边。
- 质量提示。

证据展示原则：

- LLM 总结不能替代原文证据。
- 缺证据时显示为质量问题，而不是自动补全。
- 证据片段应足够短，便于快速核对。
- 允许从证据面板跳转到关联实体、关联关系和质量报告。

### 6. 质量报告

质量报告读取 `quality_report.md` 和 `snapshots/quality_score.json`。

必须包含：

- 总分。
- 子分数。
- 指标趋势。
- critical blockers。
- findings 列表。
- 每个 finding 的 severity、category、message、evidence、suggested_fix_type、requires_approval。

推荐分组：

- 实体卫生。
- 关系语义。
- 证据绑定。
- 医学层级。
- Web 可读性。
- 迭代风险。

质量报告不是静态 Markdown 阅读器。它应该把问题变成可筛选、可追踪、可进入审批队列的审阅对象。

### 7. 审批队列

审批队列是 agent 自我维护升级机制的安全阀。

读取：

- `approval_queue.md`
- `improvement_backlog.md`
- `accepted_changes.md`
- `rejected_changes.md`

每条 proposal 显示：

- proposal id。
- type。
- target。
- proposed_change。
- reason。
- evidence。
- confidence。
- risk。
- expected_metric_change。
- requires_approval。

操作：

- 接受。
- 拒绝。
- 暂缓。
- 要求重新分析。
- 打开证据。
- 打开影响文件。

审批规则：

- `prompt_edit` 必须审批。
- `ontology_rule_change` 必须审批。
- `hierarchy_rule_change` 必须审批。
- `relation_rule_change` 必须审批。
- `workspace_rebuild` 必须审批。
- `kg_fact_correction` 必须审批。
- `web_display_change` 必须审批。
- 未知 proposal type 默认需要审批。

接受后写入 `accepted_changes.md`，必须记录审阅人、原因、影响范围、后续验证方式。

拒绝后写入 `rejected_changes.md`，成为负向记忆，避免后续 agent 重复提出同一建议。

### 8. 运行记录

运行记录展示 `iteration_log.md`。

必须包含：

- 每次 run 的时间。
- workspace。
- phase。
- artifact manifest。
- quality score。
- 是否已人工审阅。
- 是否触发后续 proposal。
- 是否进入 diff 对比。

第一版 runner 的 `pending_user_review` 必须解释为“审阅包已生成，等待人工审阅”，不能解释为“已经提出或应用改动”。

### 9. Diff 对比

Diff 对比用于判断重建、规则变更或提示词变更是否真的改善 KG。

读取：

- `diff_report.md`
- `snapshots/diff_summary.json`

必须显示：

- before/after workspace 或 run。
- 新增节点。
- 删除节点。
- 类型变化。
- 新增关系。
- 删除关系。
- 关系关键词变化。
- 质量分变化。
- evidence coverage 变化。
- generic relation count 变化。
- dangerous regression flags。

危险回归必须醒目：

- 核心疾病节点被删除。
- 证据覆盖下降。
- 泛化关系增加。
- 关键医学分类消失。
- 原本明确的 relation keyword 被替换为泛化关系。

### 10. 规则记忆

规则记忆是长期维护能力。

读取和维护：

- `quality_rules.md`
- `known_issues.md`
- `accepted_changes.md`
- `rejected_changes.md`

必须支持：

- 查看长期规则。
- 查看已知问题。
- 查看历史接受/拒绝决策。
- 把新规则加入长期记忆。
- 防止下一轮 agent 重复提出已拒绝建议。

## 前端视觉设计

整体气质：现代 AI 工作台，但严肃可信。

设计策略：

- Product UI，不做营销页。
- 深色/浅色都支持，第一版可优先适配现有主题系统。
- 使用克制的中性色作为大面积背景。
- 使用青绿色或蓝绿色作为主强调色，延续现有 LightRAG WebUI 的 emerald 倾向。
- 风险状态使用清晰语义色：warning、danger、success、info。
- 不使用装饰性玻璃拟态、夸张渐变文字或大型营销 hero。

布局原则：

- 顶部使用现有 tab 系统。
- 内部使用三栏工作台：左导航、中心主工作区、右详情面板。
- 图谱画布占中心主区域最大空间。
- 证据和审批信息常驻右侧，不默认弹 modal。
- 页面组件保持紧凑，支持长时间维护工作。

组件原则：

- 图谱工具使用图标按钮和 tooltip。
- 视图模式使用 segmented control。
- 筛选使用 select、checkbox、toggle。
- 数值阈值使用 slider 或 input。
- 审批操作使用明确按钮，并在高风险动作前二次确认。
- loading 使用 skeleton，不用空白等待。
- 空状态要告诉用户下一步，例如“先运行一次 KB iteration 审阅”。

动效原则：

- 动效只表达状态变化。
- 150-250ms。
- 支持 reduced motion。
- 图谱布局变化要平滑，但不能让节点位置跳动影响阅读。

## 前端接入方式

现有 WebUI 是 React + TypeScript + Vite + Tailwind，顶部 tab 已由 `SiteHeader` 和 `settings-storage` 管理。

新增前端模块建议：

```text
lightrag_webui/src/features/KGMaintenanceConsole.tsx
lightrag_webui/src/components/kg-maintenance/
  KGMaintenanceShell.tsx
  KGMaintenanceOverview.tsx
  MedicalHierarchyGraph.tsx
  EntityCatalogPanel.tsx
  RelationCatalogPanel.tsx
  EvidenceInspector.tsx
  QualityReportPanel.tsx
  ApprovalQueuePanel.tsx
  IterationRunPanel.tsx
  DiffReviewPanel.tsx
  RuleMemoryPanel.tsx
lightrag_webui/src/stores/kgMaintenance.ts
```

需要更新：

- `settings.ts` 的 tab union。
- `SiteHeader.tsx` 增加 `KG Maintenance` tab。
- `App.tsx` 增加 tab content。
- `api/lightrag.ts` 增加 KB iteration API wrapper。
- `locales/zh.json` 和 `locales/en.json` 增加文案。

因为现有 `TabsContent` 使用 force mount，新控制台的网络请求必须受 currentTab 和内部 active view 控制，避免隐藏 tab 仍持续轮询。

## 后端/API 需求

第一版完整 WebUI 需要新增只读和审批相关 API。不要让前端直接拼本地文件路径。

推荐路由前缀：

```text
/kb-iteration
```

需要 API：

```text
GET  /kb-iteration/workspaces
GET  /kb-iteration/{workspace}/runs
POST /kb-iteration/{workspace}/runs
GET  /kb-iteration/{workspace}/runs/{run_id}
GET  /kb-iteration/{workspace}/runs/{run_id}/artifacts
GET  /kb-iteration/{workspace}/runs/{run_id}/artifacts/{artifact_key}
GET  /kb-iteration/{workspace}/runs/{run_id}/snapshot
GET  /kb-iteration/{workspace}/runs/{run_id}/quality
GET  /kb-iteration/{workspace}/runs/{run_id}/catalog/entities
GET  /kb-iteration/{workspace}/runs/{run_id}/catalog/relations
GET  /kb-iteration/{workspace}/runs/{run_id}/diff
POST /kb-iteration/{workspace}/proposals
POST /kb-iteration/{workspace}/proposals/{proposal_id}/accept
POST /kb-iteration/{workspace}/proposals/{proposal_id}/reject
POST /kb-iteration/{workspace}/proposals/{proposal_id}/defer
GET  /kb-iteration/{workspace}/evidence/{source_id}
```

API 规则：

- workspace 必须使用现有 `validate_workspace`。
- artifact_key 必须来自白名单。
- 不接受任意文件路径。
- run 应有互斥锁，避免并发生成覆盖。
- 所有 mutation proposal 只写审批记录，不直接修改 KG。
- 高风险操作必须单独 API，且默认不开启自动执行。

## 数据映射

现有 agent 产物映射到 UI：

| UI 区域 | 数据来源 |
| --- | --- |
| 总览 | `kb_context.md`, `quality_score.json`, `iteration_log.md` |
| 医学图谱 | `kg_snapshot.json`, `hierarchy_paths.json`, `/graphs?medical_view=true` |
| 实体目录 | `entity_catalog.md`, `entity_stats.json`, `kg_snapshot.json` |
| 关系目录 | `relation_catalog.md`, `relation_stats.json`, `kg_snapshot.json` |
| 证据审阅 | `source_coverage.json`, `kg_snapshot.json`, source/chunk lookup API |
| 质量报告 | `quality_report.md`, `quality_score.json` |
| 审批队列 | `approval_queue.md`, `improvement_backlog.md` |
| Diff 对比 | `diff_report.md`, `diff_summary.json` |
| 规则记忆 | `quality_rules.md`, `known_issues.md`, `accepted_changes.md`, `rejected_changes.md` |

## 医学安全边界

界面必须把以下边界显式呈现：

- 当前 deterministic runner 对 workspace 是只读的。
- LLM 输出是分析和建议，不是事实来源。
- 医学事实必须有 source、file path 或 chunk 证据链。
- 组织性节点和分类节点用于导航，不应被误读为临床事实。
- 缺失预期医学分类时应诚实显示缺失，而不是自动补医学常识。
- 所有 mutation proposal 都必须通过审批队列。

高风险操作必须显示确认文案：

```text
该操作会改变知识库行为或重建结果。请确认已检查来源证据、影响范围和回滚方式。
```

## 状态设计

必须支持：

- Loading：骨架屏。
- Empty：提示先运行 KB iteration。
- Error：显示失败原因、重试、查看日志。
- Stale：提示当前报告不是最新。
- Running：显示运行进度和当前阶段。
- Pending review：审阅包已生成，等待人工处理。
- Approved：已接受，等待执行或验证。
- Rejected：已拒绝，写入负向记忆。
- Diff ready：可比较前后结果。

## 验收标准

功能验收：

- 用户能从新 tab 进入 KG 维护控制台。
- 用户能看到 workspace 总览和质量摘要。
- 用户能在医学层级视图中理解流感 KG 的疾病、分类、分组、事实层次。
- 用户能区分事实节点、组织节点、证据不足节点和质量风险节点。
- 用户能看到关系语义和方向，不再看到主要文案为 `邻接`。
- 用户能打开实体目录、关系目录和证据详情。
- 用户能查看质量报告和审批队列。
- 用户能接受、拒绝或暂缓 proposal，并记录原因。
- 用户能比较两轮 snapshot diff。
- 用户能查看长期规则记忆。

视觉验收：

- 界面第一眼像专业维护工具，而不是营销页。
- 图谱节点大小不再全部一致。
- 图例解释颜色、大小、类型、证据和风险。
- 右侧详情面板在图谱浏览时始终可用。
- 中英文文案不溢出。
- 深浅主题下文本对比度达到 WCAG AA。
- 小屏幕下三栏布局能降级为主区域加折叠详情。

安全验收：

- 没有未审批的 KG 事实修改入口。
- 没有把 LLM 分析展示成医学证据。
- 没有任意文件路径读取 API。
- 没有隐藏执行 workspace rebuild 的按钮。
- 所有高风险 proposal 都有审批状态和审阅记录。

## 推荐实施阶段

### Phase 1：只读控制台

- 新增 tab 和控制台 shell。
- 读取已有 artifacts。
- 展示总览、质量报告、实体/关系目录、规则记忆。
- 不做审批写入，不做 run trigger。

### Phase 2：医学层级图谱

- 实现医学层级视图。
- 解决节点大小、关系文案、方向展示、图例。
- 接入证据详情面板。

### Phase 3：审批与运行

- 新增 run API。
- 新增 proposal accept/reject/defer API。
- 写入 accepted/rejected changes。
- 增加运行状态和错误状态。

### Phase 4：Diff 与长期维护

- 实现 before/after run 对比。
- 增加危险回归提示。
- 把 rejected changes 注入后续质量审阅上下文。
- 支持未来 workspace 和非流感文档。

## 设计决策

采用 A 方案并落为 `KG Maintenance` 新 tab。

不建议把所有功能叠进现有 `knowledge-graph` tab。旧图谱页已经承担通用探索任务，医学 KG 维护控制台需要更强的 workflow、证据审阅和审批状态。如果把它塞进旧图谱页，会继续放大当前问题：图谱看起来像主角，但维护流程分散在角落里。

新的控制台应当让图谱成为工作流的一部分，而不是唯一界面。
