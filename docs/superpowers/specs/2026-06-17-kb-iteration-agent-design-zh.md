# 知识库迭代 Agent 设计

## 目标

设计一个用于 LightRAG 知识库长期维护的半自动迭代 Agent。它服务于反复导入文档、重建知识图谱、检查质量、沉淀规则和比较版本的完整循环。

这个 Agent 的核心目标不是替代医学专家，也不是自动创造医学事实，而是帮助 LLM 快速理解当前知识库状态，评估实体和关系质量，提出结构化改进建议，并留下可审计的迭代记录。

它可以自动生成报告和建议，但以下动作必须经过用户确认：

- 修改抽取提示词。
- 修改医学本体、同义词、关系词或层级规则。
- 修改事实型 KG 数据。
- 删除、清空或重建 workspace。
- 将 LLM 推断内容写入知识库。

第一版验收场景是当前流感医学知识库：

- Workspace: `influenza_medical_v1`
- 存储目录: `data/rag_storage/influenza_medical_v1`
- 输入目录: `data/inputs/influenza_medical_v1`
- 医学 profile: `clinical_guideline_zh`
- 关键数据文件:
  - `graph_chunk_entity_relation.graphml`
  - `kv_store_full_entities.json`
  - `kv_store_full_relations.json`
  - `vdb_entities.json`
  - `vdb_relationships.json`

设计也必须支持未来新增的 workspace 和非流感医学文档，不能把 Agent 固化成只适用于流感的维护脚本。

## 问题背景

LightRAG 已经可以从文档中抽取实体和关系，并构建知识图谱。但一个持续迭代的知识库还需要回答这些问题：

- 当前知识库里有哪些实体、关系、分类和来源文档？
- 哪些图谱结构存在噪声、重复、证据不足或不适合人类理解？
- 最新一轮提示词、规则或 workspace 迭代，是让 KG 变好了还是变差了？
- 哪些修复建议可以自动生成报告，哪些必须等待人工确认？
- 哪些失败案例已经在之前被接受、拒绝或归入长期规则？

如果没有持久化的知识库迭代层，每次 LLM 会话都要重新理解原始图谱，质量检查也容易停留在主观判断，已拒绝的建议还会反复出现。

## 已确认方向

采用方案 B：半自动知识库迭代 Agent。

Agent 可以自动执行：

- 读取当前 KG 并生成机器快照。
- 生成 LLM 易读的 Markdown 记忆。
- 执行质量检查并为 KG 打分。
- 生成结构化改进建议。
- 比较两个 workspace 或两轮快照。
- 更新报告、日志和待办。

Agent 不可以自动执行：

- 新增原文未支持的医学事实。
- 修改抽取提示词。
- 修改本体、同义词、关系词或层级规则。
- 删除、清空或重建 LightRAG workspace。
- 把事实级改动直接写入 KG 数据。

这些动作必须进入审批队列，等待用户明确确认。

## 非目标

第一版不构建完全自治的医学本体维护系统，也不替代 LightRAG 的抽取流程，不重写医学 KG profile，不新增 WebUI 审核后台。

第一版以文件化工作流为核心：生成快照、Markdown 记忆、质量报告、审批队列和迭代日志。未来可以再让 WebUI 读取这些文件，但初始版本以本地文件为准。

## 当前项目挂点

Agent 应接入现有 LightRAG 数据和源码边界。

| 需求 | 现有挂点 |
| --- | --- |
| 当前 workspace 配置 | `.env`, `WORKING_DIR`, `INPUT_DIR`, `WORKSPACE` |
| 图谱拓扑 | `graph_chunk_entity_relation.graphml` |
| 文档级实体索引 | `kv_store_full_entities.json` |
| 文档级关系索引 | `kv_store_full_relations.json` |
| 实体和关系元数据 | GraphML 节点/边属性、向量元数据文件 |
| 医学抽取提示词 | `prompts/entity_type/医学实体类型提示词.yml` |
| 医学本体/同义词 | `lightrag/medical_kg/ontology.py` |
| 医学层级补全 | `lightrag/medical_kg/hierarchy.py` |
| 抽取归一化挂点 | `lightrag/operate.py::extract_entities` |
| 图谱医学分组 | `lightrag/medical_kg/graph_projection.py` 和 WebUI 医学关系分组代码 |

Agent 应尽量使用结构化解析器和稳定文件 API 读取数据。不要在可以保留字段结构的情况下，把大型 JSON 或 GraphML 当成普通文本硬读。

## Agent 角色

### 1. KG Snapshotter

负责读取当前 workspace，生成机器可读快照，并计算基础图谱统计。

输入：

- Workspace 名称。
- 存储目录和输入目录。
- GraphML 图谱。
- `kv_store_full_entities.json`。
- `kv_store_full_relations.json`。
- 可选：指定根节点的 `/graphs` API 返回结果。

输出：

- `snapshots/kg_snapshot.json`
- `snapshots/entity_stats.json`
- `snapshots/relation_stats.json`
- `snapshots/hierarchy_paths.json`
- `snapshots/source_coverage.json`

### 2. KB Memory Writer

负责把机器快照压缩成分层 Markdown 记忆，让 LLM 不需要读取所有原始图谱行，也能快速理解知识库。

输出：

- `kb_context.md`
- `entity_catalog.md`
- `relation_catalog.md`
- `kg_structure.md`
- 大型 workspace 可选拆分目录，例如 `entities/` 和 `relations/`。

### 3. KG Quality Reviewer

负责评估实体卫生、关系语义、证据绑定、层级完整性、Web 可读性和迭代就绪度。它应结合确定性规则和 LLM 质检。

输出：

- `quality_report.md`
- `quality_score.json`
- 进入 `improvement_backlog.md` 和 `approval_queue.md` 的问题与建议。

### 4. Improvement Planner

负责把质量问题转成可执行建议。它只提出建议，不直接修改提示词、规则或 KG 数据。

输出：

- `improvement_backlog.md`
- 更新后的 `approval_queue.md`

### 5. Diff Engine

负责比较两个 workspace 快照或两轮迭代结果，判断改进和回归。

输出：

- `diff_report.md`
- `snapshots/diff_summary.json`

### 6. Rule Memory Manager

负责维护长期质量规则、已知问题、已接受决策和已拒绝决策，使未来质检不会从零开始。

输出：

- `quality_rules.md`
- `known_issues.md`
- `accepted_changes.md`
- `rejected_changes.md`
- 更新 `iteration_log.md`

## 产物目录

所有知识库迭代产物放在：

```text
work/kb-iteration/<workspace>/
```

以流感 workspace 为例：

```text
work/kb-iteration/influenza_medical_v1/
  kb_context.md
  entity_catalog.md
  relation_catalog.md
  kg_structure.md
  quality_report.md
  improvement_backlog.md
  approval_queue.md
  quality_rules.md
  known_issues.md
  accepted_changes.md
  rejected_changes.md
  diff_report.md
  iteration_log.md
  snapshots/
    kg_snapshot.json
    entity_stats.json
    relation_stats.json
    hierarchy_paths.json
    source_coverage.json
    quality_score.json
    diff_summary.json
```

## Markdown 产物

### `kb_context.md`

LLM 的紧凑入口文件。它应该足够短，可以直接放进提示词上下文。

必须包含：

- Workspace 和本轮运行元数据。
- 当前 profile 和文档集合。
- KG 规模：节点数、边数、文档数、chunk 数。
- 核心主题和根疾病节点。
- 实体类型分布。
- 关系类型分布。
- 当前层级摘要。
- 最新质量评分摘要。
- 已知关键问题。
- 指向更详细文件的链接。

这是知识库维护 Agent 每次优先读取的文件。

### `entity_catalog.md`

实体类型索引。

必须包含：

- 按实体类型统计数量。
- 每类代表性实体。
- 每类可疑实体。
- 同义词和规范名说明。
- 已知父级或分类位置。
- 来源覆盖摘要。

该文件不能直接倾倒 embedding、向量 payload 或每一条低价值原始记录。大型 workspace 应拆成 `entities/<type>.md`。

### `relation_catalog.md`

关系和三元组索引。

必须包含：

- 按关系词统计数量。
- 每类关系的代表性三元组。
- 非法或泛化关系标签。
- 需要保留方向的三元组。
- source/target 类型一致性。
- 缺少描述或证据的关系。

三元组主格式：

```text
source - relation -> target
```

即使 Web 图谱画布视觉上是无向图，关系目录也必须保留语义方向。

### `kg_structure.md`

人类可读的层级和图谱结构。

必须包含：

- 疾病中心导航路径。
- 一级类别和二级子类。
- 分组后的关系类别。
- 缺失分支。
- 过载分支。
- 事实边和导航/分类边的区分。

示例路径：

```text
流行性感冒 -> 临床表现 -> 全身症状 -> 发热
```

### `quality_report.md`

最新质量审查结果。

必须包含：

- 总分和子评分。
- 关键阻塞问题。
- 实体问题。
- 关系问题。
- 层级问题。
- 证据/来源问题。
- Web 可读性问题。
- 回归风险。
- 推荐下一步行动。
- 已进入审批队列的事项。

每个可执行问题都必须包含严重程度、证据、预期影响、建议修复类型，以及是否需要人工确认。

### `improvement_backlog.md`

非立即执行的改进待办。

每条待办必须包含：

- Backlog id。
- 问题摘要。
- 修复类别。
- 目标文件或目标数据区域。
- 证据。
- 优先级。
- 是否需要审批。
- 预期指标变化。
- 状态。

### `approval_queue.md`

所有会改变行为或数据的建议都进入这里。

需要审批的动作包括：

- 修改提示词。
- 合并同义词。
- 增加或删除层级边。
- 重映射关系词。
- 增加受控分类。
- 删除或重建 workspace。
- 任何会改变 KG 的来源型修正。

每项必须包含：

- Proposal id。
- 建议改动。
- 原因。
- 证据。
- 风险。
- 预期结果。
- 回滚或重建说明。
- 所需确认。

### `quality_rules.md`

长期质量规则库。

规则必须分层：

- 通用医学 KG 规则。
- LightRAG 项目规则。
- 当前 workspace 或文档域规则。
- 流感专用规则。

已接受规则应注入未来的质检提示词。

### `known_issues.md`

失败案例库。

第一版应包含这些问题类型：

- 值型节点：`75 mg`、`每日2次`、`发病48小时内`、阈值、页码、表格碎片。
- 同义词拆分：`流感` 和 `流行性感冒`。
- 泛化关系标签：`相关`、`邻接`。
- 疾病中心过载：大量叶子节点直接连接疾病中心。
- 缺少症状、病原体、诊断、治疗、预防或人群类别路径。
- 关系缺少来源证据或语义方向。

已拒绝建议应被链接，避免 Agent 反复提出同样建议。

### `accepted_changes.md` 和 `rejected_changes.md`

决策历史。

`accepted_changes.md` 记录：批准了什么、为什么批准、何时批准、由谁批准、后续需要什么验证。

`rejected_changes.md` 记录：为什么拒绝、证据哪里不足、将来是否可以重新考虑。

### `diff_report.md`

workspace 或运行结果对比报告。

必须包含：

- 被比较的两个快照。
- 新增、删除、合并、类型变化的实体。
- 新增、删除、关系词变化、描述变化的关系。
- 层级路径变化。
- 改动前后的质量评分。
- 回归。
- 改进。
- 是否建议接受本轮重建结果。

## 机器快照

### `kg_snapshot.json`

标准图谱快照。建议结构：

```json
{
  "workspace": "influenza_medical_v1",
  "generated_at": "2026-06-17T00:00:00+08:00",
  "source_files": [],
  "nodes": [],
  "edges": [],
  "metadata": {
    "profile": "clinical_guideline_zh",
    "graph_node_count": 0,
    "graph_edge_count": 0
  }
}
```

实现可以增加字段，但必须尽量保留稳定 id、标签、实体类型、描述、source id、file path、关系词和边方向。

### `quality_score.json`

机器可读评分输出。建议结构：

```json
{
  "overall": 0,
  "subscores": {
    "entity_hygiene": 0,
    "relation_semantics": 0,
    "hierarchy_completeness": 0,
    "evidence_grounding": 0,
    "web_readability": 0,
    "iteration_readiness": 0
  },
  "metrics": {},
  "critical_blockers": []
}
```

## 质量评分模型

质量审查应输出 0 到 100 分。这个分数不是医学真理评分，而是本项目 KG 的维护就绪度评分。

建议权重：

| 子评分 | 权重 | 目的 |
| --- | ---: | --- |
| 实体卫生 | 20 | 稳定概念、同义词合并、值型噪声移除 |
| 关系语义 | 20 | 合法关系词、方向、描述是否支撑关系 |
| 层级完整性 | 20 | 疾病/类别/子类/叶子组织结构 |
| 证据绑定 | 20 | source id、file path、chunk 链接、事实边和生成边区分 |
| Web 可读性 | 10 | 原始 KG 可读性、医学分组质量、关系化属性面板 |
| 迭代就绪度 | 10 | 可对比、可审批、待办是否可执行 |

### 实体卫生指标

跟踪：

- 值型节点数量。
- 已知同义词重复数量。
- 近似重复实体候选数量。
- 未知或不允许的实体类型数量。
- 可疑短片段数量。
- 实体类型分布异常峰值。

值型节点示例：

- `75 mg`
- `每日2次`
- `5日疗程`
- `发病48小时内`
- `PaO2/FiO2≤300`
- 页码、表格、脚注片段。

### 关系语义指标

跟踪：

- 非法关系词数量。
- 泛化关系数量。
- 缺少描述的关系数量。
- 缺少方向的关系数量。
- source/target 类型不匹配数量。
- 描述不能支撑三元组的关系。
- 同一关系对存在冲突关系词。

泛化标签包括 `相关`、`邻接`，以及其他不能说明医学语义的标签。

### 层级完整性指标

跟踪：

- 必需一级类别覆盖率。
- 缺失子类数量。
- 已有类别路径时，叶子事实仍直接挂到疾病中心的数量。
- 孤立节点数量。
- 疾病中心过载比例。
- 不在受控表中的类别变体数量。

流感 workspace 的必需一级分支：

- `病原体`
- `传播/流行病学`
- `临床表现`
- `并发症/重症`
- `诊断/检查`
- `治疗`
- `预防`
- `高危人群`
- `指南/证据来源`

### 证据绑定指标

跟踪：

- 缺少 `source_id` 的数量。
- 缺少 `file_path` 的数量。
- 缺少 chunk/source 链接的数量。
- 生成型导航边缺少内部来源标记的数量。
- 只出现在生成描述中、缺乏原文支撑的事实声明。
- 需要人工复核来源的关系。

审查者不得建议新增原文未写明的医学事实。它可以建议把已有事实组织进导航分类。

### Web 可读性指标

跟踪：

- 原始 KG 是否清晰呈现事实节点与关系。
- 医学分组是否减少属性面板中的噪声。
- 属性面板是否显示关系标签，而不是 `邻接`。
- 关系详情是否保留完整三元组方向。

### 严重程度

| 严重程度 | 含义 | 处理要求 |
| --- | --- | --- |
| Critical | 阻塞重建知识库验收 | 发布前必须修复或明确豁免 |
| High | 可能影响检索、事实性或 Web 理解 | 发布前需要审阅 |
| Medium | 适合进入 backlog 的质量问题 | 加入改进待办 |
| Low | 信息性或清理项 | 记录即可 |

## 质检提示词

第一版质检提示词应作为版本化产物保存，例如 `quality_reviewer_prompt.md`。

草案：

```text
你是医学知识图谱质量审查专家。你的任务不是补充医学常识，而是审查当前知识库是否忠实、清晰、可检索、适合人类理解。

必须遵守：
1. 不得新增原文未支持的医学事实。
2. 可以建议新增导航/分类节点，但必须说明它们是为了组织已有事实。
3. 优先发现结构问题、实体类型问题、关系语义问题、证据缺失问题和 Web 可读性问题。
4. 对每个问题给出严重程度、证据、影响、建议修复方式和是否需要人工确认。
5. 建议分为：提示词优化、同义词归一、层级规则、关系规则、Web 展示、需要人工确认。

检查维度：
- 实体是否是稳定医学概念。
- 是否存在剂量、阈值、时间窗口、页码、表格碎片等值型噪声节点。
- 同义词、简称、全称是否被合并。
- 关系是否有明确医学语义，而不是“相关/邻接”。
- 关系方向和描述是否支持三元组。
- 每条事实关系是否有来源元数据。
- 疾病、症状、诊断、治疗、预防、人群、指南等层级是否适合人类理解。
- 医学分组是否让原始 KG 更容易理解。
- 本轮建议是否适合自动报告、进入待办、进入审批队列或需要人工复核。

输出格式：
- 总体评分
- 子评分
- 关键问题列表
- 实体问题
- 关系问题
- 层级问题
- 证据/来源问题
- Web 可视化问题
- 结构化改进建议
- 下一轮重建建议
```

## 结构化改进建议格式

任何可能改变行为或数据的建议，都必须表示为 `ImprovementProposal` 记录，并由 `proposals.py` 渲染到 `approval_queue.md` 或 `improvement_backlog.md`。

已实现字段：

- `id`：非空字符串。
- `type`：规范的小写 snake_case 字符串。
- `target`：非空字符串。
- `proposed_change`：非空字符串。
- `reason`：非空字符串。
- `evidence`：字符串列表。
- `confidence`：0 到 1 之间的数字。
- `risk`：`low`、`medium` 或 `high`。
- `requires_approval`：布尔值。
- `expected_metric_change`：指标标识符到数字增量的映射。

渲染后的记录示例：

```yaml
id: proposal-20260617-001
type: hierarchy_rule_change
target: lightrag/medical_kg/hierarchy.py
proposed_change: Add a controlled symptom branch for fever under the existing clinical presentation hierarchy.
reason: Direct disease-to-leaf overload was detected in the current workspace.
evidence:
  - "流行性感冒 -> 发热"
  - "quality finding: disease hub overload"
confidence: 0.8
risk: medium
requires_approval: true
expected_metric_change:
  hierarchy_completeness: 5
```

已实现且必须审批的 mutation proposal 类型：

- `prompt_edit`
- `ontology_rule_change`
- `hierarchy_rule_change`
- `relation_rule_change`
- `workspace_rebuild`
- `kg_fact_correction`
- `web_display_change`

已实现的免审批报告备注类型：

- `quality_report_note`

未知 proposal 类型默认需要审批，除非它们在代码中被明确加入免审批报告备注集合。

## Agent 闭环

Agent 的核心行为是一条闭环：

```text
观察 -> 思考 -> 提案 -> 审批 -> 执行 -> 评估 -> 记忆 -> 下一轮
```

这条闭环是每次知识库迭代的主要契约。下面的“工作流”章节只是这条闭环在不同入口场景中的具体展开；真正的运行模型是闭环本身。

### 闭环落点

这条闭环不能只停留在流程图里。每个阶段都必须落到明确文件或模块上，让未来的 LLM、开发者或审查者不用猜测就能检查本轮迭代发生了什么。

- 观察落在 `snapshot.py`、`markdown.py`、`snapshots/kg_snapshot.json`、`kb_context.md`、`entity_catalog.md`、`relation_catalog.md` 和 `kg_structure.md`。
- 思考/诊断落在确定性的 `quality.py`、未来的 `quality_reviewer_prompt.md`、`quality_score.json` 和 `quality_report.md`。Agent 不应持久化私密推理链，而应持久化可审计的诊断、证据、假设、不确定性和下一步建议。
- 提案落在 `proposals.py`、`approval_queue.md` 和 `improvement_backlog.md`。
- 审批落在人工审阅后的 `accepted_changes.md`、`rejected_changes.md`，以及 `iteration_log.md` 中的审批记录。
- 执行在第一版中刻意不自动开放，除非已经存在审批记录。未来执行适配器可以修改提示词、规则、本体文件、Web 展示代码或重建 workspace，但每个动作都必须在 `iteration_log.md` 记录修改文件、执行命令、workspace 名称和回滚说明。
- 评估落在 `diff.py`、刷新后的快照、刷新后的 `quality_score.json`、`snapshots/diff_summary.json` 和 `diff_report.md`。
- 记忆/下一轮落在 `quality_rules.md`、`known_issues.md`、已接受/已拒绝历史，以及下一轮刷新的 `kb_context.md`。

因此，第一版确定性 runner 默认应实现 `观察 -> 确定性质量报告 -> pending_user_review`。它写入快照、Markdown 记忆、质量产物和迭代日志记录。结构化 proposal 生成是单独步骤：只有在已有或后续生成 `ImprovementProposal` 对象时，`proposals.py` 才会填充 `approval_queue.md` 和 `improvement_backlog.md`。当用户批准重建或规则修改后，Agent 可以带着 `previous_snapshot` 再运行一次，进入评估和记忆阶段。

### 1. 观察

Agent 读取当前 workspace，建立可靠的知识库状态图。

输入：

- `.env` workspace 配置。
- 图谱、实体、关系和来源元数据文件。
- 已存在的 `kb_context.md`、质量规则、已知问题、已接受决策和已拒绝决策。
- 可选的历史 workspace 快照。

输出：

- 新机器快照。
- 刷新的 Markdown 记忆。
- 写入 `iteration_log.md` 的本轮运行记录。

### 2. 思考

Agent 在提出修改建议前，先分析当前状态。

思考必须结合确定性检查和 LLM 审查：

- 确定性检查负责计算值型节点、缺失来源、非法关系词、必需分支覆盖率等可度量问题。
- LLM 审查负责判断更高层的结构、可读性、歧义和可能原因。
- Agent 必须说明诊断、证据、假设和不确定性。

这一步的输出不是改动，而是进入质量评分和建议生成的分析包。

### 3. 提案

Agent 把诊断出的问题转化为结构化改进建议。

每个建议必须包含：

- 问题。
- 建议改动。
- 目标文件、规则、提示词、workspace 或 UI 行为。
- 支撑证据。
- 预期指标改善。
- 风险和置信度。
- 是否需要审批。

所有会改变行为的建议进入 `approval_queue.md`。低风险观察进入 `improvement_backlog.md`。

### 4. 审批

审批是分析和变更之间的人工闸门。

Agent 可以继续生成报告，但在审批记录存在之前，不能应用提示词编辑、本体编辑、层级改动、关系规则改动、事实型 KG 修正或 workspace 重建。

审批结果写入：

- `accepted_changes.md`
- `rejected_changes.md`
- `iteration_log.md`

被拒绝的建议会成为未来的负向记忆，避免 Agent 反复提出同一项改动。

### 5. 执行

执行只应用已批准的改动。

执行可以包括：

- 修改抽取提示词。
- 更新同义词、关系、分类或层级规则。
- 创建新的 workspace。
- 重建 workspace。
- 更新 Web 展示行为。

执行必须记录实际修改的文件、运行的命令、workspace 名称和重建参数。如果执行失败，必须记录失败尝试，下一次尝试不能重复同样的失败动作。

### 6. 评估

执行后，Agent 必须立刻评估结果。

评估包括：

- 重新生成快照。
- 重新计算质量分数。
- 比较前后指标。
- 写入 `diff_report.md`。
- 检测回归。
- 判断本轮迭代应该接受、继续修改还是回滚。

一次迭代不能仅因为“改动已经应用”就算成功。只有质量指标和回归检查支持该结论，或用户明确接受取舍，才算成功。

### 7. 记忆

Agent 更新长期记忆。

记忆更新包括：

- 写入 `quality_rules.md` 的新接受规则。
- 写入 `known_issues.md` 的新失败案例。
- 已接受和已拒绝建议。
- `iteration_log.md` 中的运行摘要。
- 应影响未来质检提示词的经验。

这样每一轮循环都会比上一轮更了解项目。

### 闭环状态记录

每轮闭环都应产生紧凑状态记录：

```yaml
loop_id: kb-loop-20260617-001
workspace: influenza_medical_v1
phase: evaluate
input_snapshot: snapshots/kg_snapshot.json
previous_snapshot: null
analysis: quality_report.md
proposals:
  - proposal-20260617-001
approved_changes: []
executed_changes: []
quality_before: null
quality_after: snapshots/quality_score.json
decision: pending_user_review
```

该状态记录可以写在 `iteration_log.md` 中，也可以单独保存为 `snapshots/loop_state.json`。

### 闭环停止条件

满足以下任一条件时，闭环停止：

- 质量分达到验收阈值，且没有 critical blocker。
- Agent 已生成需要审批的建议，正在等待用户审阅。
- 重建或变更需要用户明确确认。
- 关键错误导致无法可靠生成快照或评分。
- 用户主动暂停本轮迭代。

### 闭环不变量

- 观察和评估必须基于结构化数据，而不仅是自然语言摘要。
- 思考必须输出有证据的诊断，而不是静默假设。
- 提案必须结构化、可审阅。
- 执行必须经过确认。
- 每次执行后必须评估。
- 每次接受或拒绝后必须更新记忆。

## 工作流

### 文档入库后运行

1. 读取 `.env`，解析 workspace 路径。
2. 读取图谱、实体、关系存储文件。
3. 生成机器快照。
4. 生成或刷新 Markdown 记忆。
5. 运行确定性质量检查。
6. 使用紧凑上下文和必要证据运行 LLM 质检。
7. 写入 `quality_report.md`。
8. 写入 `improvement_backlog.md`。
9. 将需要审批的建议写入 `approval_queue.md`。
10. 将运行信息追加到 `iteration_log.md`。

### 重建前运行

1. 审阅 `approval_queue.md`。
2. 只应用已批准的提示词或规则改动。
3. 将批准项记录到 `accepted_changes.md`。
4. 将拒绝项记录到 `rejected_changes.md`。
5. 如果要重建，创建新的 workspace 名称。
6. 只有在用户明确确认后才重建。

### 重建后运行

1. 生成新快照。
2. 比较旧快照和新快照。
3. 写入 `diff_report.md`。
4. 更新 `quality_report.md`。
5. 标记回归。
6. 建议接受、继续修改或回滚。

## 安全边界和医学约束

医学事实必须保持来源支撑。Agent 应将 LLM 输出视为分析和建议，而不是事实权威。

必要护栏：

- 区分事实边和导航/分类边。
- 为事实发现保留来源元数据。
- 不使用常识补充导入文档中没有写明的医学事实。
- 不静默修改提示词、规则或 workspace。
- 持久保存已接受和已拒绝的决策。
- 所有高影响建议都必须可审阅。

如果缺少证据，Agent 应建议修复来源链接、重新抽取或人工复核，而不是直接接受该三元组为可靠事实。

## 领域扩展策略

设计必须在新增医学文档后仍然可用。

规则：

- 通用医学规则写入 `quality_rules.md` 的通用医学 KG 规则区。
- LightRAG 特定行为写入项目规则区。
- 当前文档集规则写入 workspace/domain 规则区。
- 流感专用同义词和层级种子必须明确标记为流感专用。
- 新类别必须映射到受控扩展类别，否则进入审批队列。

受控扩展类别：

| Key | 标签 | 示例 |
| --- | --- | --- |
| `differential_diagnosis` | `鉴别诊断` | 相似疾病、鉴别标准 |
| `nursing_care` | `护理` | 照护、居家护理、观察 |
| `follow_up` | `随访` | 复诊、随访观察 |
| `rehabilitation` | `康复` | 恢复期管理 |
| `contraindication` | `禁忌证` | 用药禁忌、不适用人群 |
| `adverse_reaction` | `不良反应` | 副作用、不良事件 |
| `public_health` | `公共卫生处置` | 报告、隔离管理、学校防控 |

## 错误处理

Agent 必须记录错误并优雅降级：

- 缺少图谱文件：写入 `quality_report.md`，停止依赖快照的检查。
- 空 workspace：生成空快照，将迭代就绪度标为低。
- JSON 或 GraphML 解析失败：记录解析错误，不覆盖上一次可用快照。
- 缺少来源元数据：继续分析，但标记证据覆盖问题。
- LLM 质检失败：保留确定性检查结果，并标记 LLM review 未完成。
- Markdown 目录过大：按实体类型或关系组拆分。

## 测试期望

实施计划应包含以下测试：

- 从小型 fixture 图生成快照。
- 按类型生成实体目录。
- 按关系词和方向生成关系目录。
- 识别值型节点。
- 识别泛化关系。
- 识别缺少来源元数据。
- 计算质量评分。
- 校验审批队列格式。
- 校验结构化 proposal schema。
- 比较两个 fixture 快照并生成 diff report。
- 控制 Markdown 大小并测试拆分目录。

医学流感 fixture 应包含：

- `流感` 和 `流行性感冒` 的同义词重复。
- `75 mg` 和 `发病48小时内` 作为值型节点示例。
- 泛化 `邻接` 关系示例。
- 缺失 `临床表现 -> 全身症状 -> 发热` 路径的示例。
- 疾病中心过载示例。

## 验收标准

设计完成标准：

- 文件布局和各产物职责明确。
- 半自动安全边界明确。
- 质量评分模型包含指标和严重程度。
- 证据绑定和来源约束明确。
- 审批流程是一等能力，所有 mutation 都需要确认。
- diff 工作流能比较重建前后 workspace。
- 包含长期规则库和失败案例库。
- 支持流感之外的未来医学文档。
- 结构化 proposal 格式明确。
- 测试期望足够具体，可以直接进入实施计划。

## 推荐实施阶段

建议实施顺序：

1. 生成快照和确定性统计。
2. 生成 Markdown 记忆。
3. 生成确定性质量检查和 `quality_score.json`。
4. 接入 LLM 质检提示词和报告生成。
5. 生成改进 backlog 和审批队列。
6. 生成结构化 proposal 记录并做 schema 校验。
7. 生成快照 diff 和回归报告。
8. 补充文档和测试。

这样即使 LLM 质检还没有接入，第一版也已经能产生有用的知识库状态报告和确定性质量指标。

## 已实现的确定性工作流

Task 7 已在 `docs/KBIterationAgent-zh.md` 中补充面向使用者的中文操作文档，并记录当前已经位于 `lightrag/kb_iteration/` 下的确定性模块。

当前已实现模块：

- `snapshot.py` 从 GraphML 构建 `KGSnapshot`，并通过 `write_snapshot_artifacts` 写入快照产物。
- `markdown.py` 通过 `write_markdown_memory` 写入适合 LLM 阅读的记忆文件。
- `quality.py` 通过 `evaluate_snapshot_quality` 计算确定性质量指标，并通过 `write_quality_artifacts` 写入质量产物。
- `proposals.py` 校验需要审批的 `ImprovementProposal`，并写入审批队列和改进 backlog。
- `diff.py` 通过 `compare_snapshots` 比较快照，并写入 `diff_report.md` 和 `snapshots/diff_summary.json`。
- `runner.py` 提供第一版确定性端到端入口 `run_iteration`。

第一版 runner 刻意保持非变更型。它会在路径拼接前校验 workspace 名称，读取现有 workspace 图谱，写入快照、Markdown、质量和迭代日志产物，然后记录 `pending_user_review`。它不会自动应用提示词变更、规则变更、本体编辑、事实修正、WebUI 变更或 workspace 重建。
