# 知识库迭代 Agent

知识库迭代 Agent 是 LightRAG 知识库的确定性维护工作流。它帮助维护者查看当前图谱、生成便于后续 LLM 审阅的摘要、计算质量信号、准备可审阅的改进建议，并比较不同迭代运行的变化。

它不是医学事实编辑器。它不会自动修改已抽取的医学事实、提示词、本体规则、层级规则、关系规则、WebUI 行为或 workspace 数据。LLM 输出只能作为分析和建议，不能作为事实来源。

## 安全边界

第一版确定性 runner 对 LightRAG workspace 保持只读。它读取图谱和输入文件，然后把报告写入迭代输出目录。

工作流可以自动执行：

- 生成图谱快照和统计信息。
- 生成紧凑的 Markdown 记忆，供 LLM 和人工审阅。
- 运行确定性质量检查。
- 写入质量报告和评分 JSON。
- 在已有 proposal 对象时写入审批队列和改进 backlog。
- 比较两个快照并写入 diff 报告。

工作流不能自动执行：

- 新增导入文档中没有来源支撑的医学事实。
- 未经明确审批就修正事实级 KG 数据。
- 修改抽取提示词、本体、同义词、层级规则、关系规则或 WebUI 展示行为。
- 删除、清空、重建或重命名 workspace。
- 把 LLM 结论当成证据。

生成文件可以帮助 LLM 理解知识库状态，但任何医学事实或 KG 变更在接受前仍然需要来源支撑证据。

## 确定性模块

当前确定性实现位于 `lightrag/kb_iteration/`：

- `snapshot.py`：从 GraphML 构建 `KGSnapshot`，并通过 `write_snapshot_artifacts` 写入快照 JSON。
- `markdown.py`：通过 `write_markdown_memory` 写入 `kb_context.md`、`entity_catalog.md`、`relation_catalog.md`、`kg_structure.md`，并初始化长期记忆文件。
- `quality.py`：通过 `evaluate_snapshot_quality` 计算确定性质量指标，并通过 `write_quality_artifacts` 写入 `quality_report.md` 和 `snapshots/quality_score.json`。
- `proposals.py`：校验结构化 `ImprovementProposal`，并写入 `approval_queue.md` 和 `improvement_backlog.md`。
- `diff.py`：通过 `compare_snapshots` 比较快照，并通过 `write_diff_report` 写入 `diff_report.md` 和 `snapshots/diff_summary.json`。
- `runner.py`：通过 `run_iteration` 编排第一版确定性运行。

`run_iteration` 会在拼接路径前校验 workspace 名称，因此路径穿越等不安全名称会在写入报告前被拒绝。

## 产物目录

产物写入：

```text
work/kb-iteration/<workspace>/
```

预期文件包括：

```text
work/kb-iteration/<workspace>/
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

当前 runner 会写入核心快照、Markdown、质量报告和迭代日志。它不会创建 proposal 对象、proposal YAML 文件或已填充的队列条目。审批队列和 backlog 由 `proposals.py` 在已有或后续生成 `ImprovementProposal` 对象时写入。diff 产物在已有前后快照时由 `diff.py` 写入。

## 第一次确定性运行

在文档入库后或 workspace 重建后，用第一次运行生成审阅包。

1. 确认 LightRAG workspace 中已经存在 `graph_chunk_entity_relation.graphml`。
2. 使用 workspace、storage root、input root、output root 和可选 profile 调用 `run_iteration`。
3. 先阅读 `kb_context.md`，快速了解当前状态。
4. 审阅 `quality_report.md` 和 `snapshots/quality_score.json`。
5. 审阅 `approval_queue.md`、`improvement_backlog.md`、`known_issues.md` 和 `quality_rules.md` 等长期记忆文件。
6. 除非人工明确批准变更或重建，否则停在日志记录的 `pending_user_review` 阶段。

示例：

```python
from lightrag.kb_iteration.runner import run_iteration

result = run_iteration(
    workspace="influenza_medical_v1",
    storage_root="data/rag_storage",
    input_root="data/inputs",
    output_root="work/kb-iteration",
    profile="clinical_guideline_zh",
)

print(result.output_dir)
print(result.quality_score.overall)
```

第一版 runner 会向 `iteration_log.md` 追加 `phase: pending_user_review`。这个阶段表示审阅包已经生成，不表示任何建议已经被生成或应用。

## LLM 审阅循环

确定性产物生成后，并且已经配置或注入 `LLMReviewClient` 时，维护者可以运行可选的 LLM 审阅循环。该循环读取 `snapshots/kg_snapshot.json`、`snapshots/quality_score.json`、`accepted_changes.md` 和 `rejected_changes.md`；随后在 `review_context/` 下生成聚焦上下文，写入 `llm_review_trace.json`、`llm_review_report.md`、`proposals.generated.yaml`，并通过现有 proposal 校验器更新审批队列。

LLM 审阅循环不会应用 patch，不会修改 KG 事实，不会编辑提示词或规则，也不会重建 workspace。LLM 输出只作为分析和建议材料；所有 mutation proposal 仍然必须经过审批。

## 来源约束

把 `source_id`、`file_path` 和 chunk 链接视为事实型 KG 内容的证据链。如果证据缺失，正确处理方式是标记来源修复、重新抽取或人工复核。

Agent 可以建议新增导航或分类结构来组织已经抽取的事实，但不能发明临床结论。一个有用的建议应说明目标、证据、风险、预期指标变化和所需审批。

## 审批队列行为

会造成变更的 proposal 必须经过明确审阅。例如：

- `prompt_edit`
- `ontology_rule_change`
- `hierarchy_rule_change`
- `relation_rule_change`
- `workspace_rebuild`
- `kg_fact_correction`
- `web_display_change`

`proposals.py` 会强制已知 mutation proposal 类型需要审批；未知 proposal 类型也默认需要审批，除非它们是明确安全的报告备注。`write_approval_queue` 只写入需要审批的项目，`write_improvement_backlog` 会记录所有有效 proposal。

已接受的变更应写入 `accepted_changes.md`，记录审阅人、原因、影响文件和后续验证。已拒绝的变更应写入 `rejected_changes.md`，说明拒绝原因以及以后是否可以重新考虑。被拒绝的项目会成为负向记忆，帮助后续运行避免重复提出同一请求。

## Diff 和后续运行

在已批准的重建或规则变更之后，生成新快照，并使用 `compare_snapshots` 和 `write_diff_report` 与旧快照比较。Diff 审阅应关注新增或删除的节点、关系词变化、实体类型变化、质量指标增量和危险回归标记。

一次迭代只有在证据、质量指标和回归审阅支持变更，或用户明确接受取舍时，才算可以接受。
