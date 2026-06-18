# KG Iteration 多阶段 LLM Agent Pipeline 设计

Date: 2026-06-19
Status: Draft for user review

## 背景

当前知识库迭代 Agent 已经能生成确定性审阅包，并且已经接入 Agent 专用 DeepSeek 配置。现有 LLM 能通过 `POST /kb-iteration/{workspace}/llm-review/runs` 被调用，运行后会写入 `llm_review_trace.json`、`llm_review_report.md`、`proposals.generated.yaml`、`approval_queue.md` 和 `improvement_backlog.md`。

最近一次在 `influenza_medical_v1` 上运行 LLM 审阅的结果是 `stop_reason=all_proposals_invalid`，4 轮都聚焦 `hierarchy_missing_branch`，但 proposal 为空。问题不在于模型没有接入，而在于 LLM 收到的上下文过于抽象：质量发现只说明“医学层级缺少 4 个一级分支”，没有明确列出缺失分支、已有分支、候选证据、可生成 proposal 类型和修复优先级。因此 LLM 无法稳定生成可通过 `ImprovementProposal` 校验的候选项。

本设计把当前单段 LLM Review Loop 升级为多阶段 Agent pipeline，让 LLM 明确参与以下环节：

1. 发现问题解释
2. proposal 生成
3. 缺失分支推断
4. 证据定位
5. 修复方案排序
6. proposal judge

## 目标

1. 将 LLM 从“单次审阅输出 proposal”升级为“多阶段 Agent pipeline”，每个阶段有独立输入、输出和落盘产物。
2. 让 LLM 能解释问题、推断缺失分支、定位证据、生成 proposal、排序修复方案，并由 judge 阶段预审 proposal。
3. 保持人工审批边界：LLM 只能生成分析、候选 proposal、候选修复计划和 judge 建议，不能自动修改 KG、自动应用 patch 或自动重建 workspace。
4. 强化 deterministic context：质量分数和 review context 必须提供具体、可验证的数据，尤其是医学层级 required / present / missing 分支。
5. 让前端 `LLM 审阅材料` 页面可见多阶段 Agent 过程，文字为中文。
6. 保持兼容现有 artifact 和审批流程：`approval_queue.md` 仍然是人工审批入口，`proposals.generated.yaml` 仍然使用现有 proposal schema。

## 非目标

- 不让 LLM 直接执行 KG mutation。
- 不让 LLM 自动应用候选 patch。
- 不新增自动 rebuild workspace 的执行器。
- 不把 LLM 输出当作医学事实证据。
- 不暴露模型隐藏推理链，只保存可审阅的结论、证据、风险和建议。
- 不把前端改回图谱画布流程，KG Maintenance 主流程继续保持“知识库迭代 Agent 工作台”。

## 总体架构

多阶段 pipeline 在后端运行，沿用当前 `/kb-iteration/{workspace}/llm-review/runs` 入口，内部从单次 LLM 调用变为阶段编排。

推荐新增模块：

```text
lightrag/kb_iteration/agent_pipeline.py
lightrag/kb_iteration/agent_context.py
lightrag/kb_iteration/agent_outputs.py
lightrag/kb_iteration/agent_prompts.py
lightrag/kb_iteration/prompts/explain_zh.md
lightrag/kb_iteration/prompts/infer_branches_zh.md
lightrag/kb_iteration/prompts/locate_evidence_zh.md
lightrag/kb_iteration/prompts/propose_zh.md
lightrag/kb_iteration/prompts/rank_repairs_zh.md
```

保留并复用：

```text
lightrag/kb_iteration/review_loop.py
lightrag/kb_iteration/review_context.py
lightrag/kb_iteration/llm_review.py
lightrag/kb_iteration/proposals.py
lightrag/kb_iteration/patches.py
lightrag/kb_iteration/prompts/judge_zh.md
```

职责划分：

- `agent_pipeline.py`：阶段编排、trace、停止条件、错误恢复。
- `agent_context.py`：按阶段构建上下文，读取 snapshot、quality、kb_context、历史决策和证据窗口。
- `agent_outputs.py`：解析阶段 JSON、写 markdown/json/yaml artifact。
- `review_loop.py`：保留公共配置和现有 API 兼容层，可调用新的 pipeline。
- `proposals.py`：继续负责 proposal 强校验和人工审批约束。
- `llm_review.py`：继续负责通用 LLM 输出解析和 report 渲染，必要时扩展为多阶段输出。

## Pipeline 阶段

### 1. Observe

确定性读取当前审阅包：

```text
kb_context.md
quality_report.md
snapshots/kg_snapshot.json
snapshots/quality_score.json
accepted_changes.md
rejected_changes.md
approval_queue.md
improvement_backlog.md
```

Observe 不调用 LLM，只构建阶段输入摘要，记录 artifact 是否存在、质量分数、finding 列表、节点/关系数量、历史 accepted/rejected 记忆。

输出：

```text
agent_context/observe.json
```

### 2. Explain

LLM 针对 `quality_score.findings` 生成中文问题解释。输出必须是可审阅结论，不保存隐藏推理链。

输入：

- quality findings
- quality metrics
- kb_context 摘要
- accepted/rejected 历史

输出：

```json
{
  "issue_explanations": [
    {
      "id": "issue-hierarchy-completeness-001",
      "category": "hierarchy_completeness",
      "severity": "medium",
      "explanation": "医学一级分支不完整会导致实体被挂到疾病中心节点或错误父类下，降低后续查询和 Web 展示的可读性。",
      "impact": "影响层级导航、聚合统计和候选修复优先级。",
      "evidence_refs": ["quality:hierarchy_missing_branch_count=4"],
      "needs_more_evidence": false
    }
  ]
}
```

落盘：

```text
llm_issue_analysis.json
llm_issue_analysis.md
```

### 3. Infer Missing Branches

LLM 参与缺失分支推断，但 deterministic 层必须先提供 required / present / missing 的结构化事实。

需要先增强 `quality.py` 或 `review_context.py`：

```json
{
  "hierarchy_branches": {
    "required": [
      {"key": "disease", "label": "疾病", "aliases": ["诊断", "疾病名称"]},
      {"key": "symptom", "label": "症状", "aliases": ["临床表现"]}
    ],
    "present": [
      {"key": "pathogen", "matched_node_ids": ["病原体"]},
      {"key": "drug", "matched_node_ids": ["神经氨酸酶抑制剂"]}
    ],
    "missing": [
      {"key": "symptom", "label": "症状", "aliases": ["临床表现"]},
      {"key": "test", "label": "检查", "aliases": ["实验室检测"]}
    ]
  }
}
```

LLM 在此基础上生成推断说明、候选父类、风险和是否需要更多证据。

落盘：

```text
llm_missing_branch_inference.json
llm_missing_branch_inference.md
```

### 4. Locate Evidence

LLM 根据 snapshot 中的 `source_id`、`file_path`、entity/relation 描述和 quality finding 证据，整理证据映射。它只能定位和总结已有证据，不能发明新证据。

输入：

- stage 2 的 issue ids
- stage 3 的 missing branches
- selected nodes/edges
- evidence_windows
- source_id/file_path

输出：

```json
{
  "evidence_map": [
    {
      "issue_id": "issue-hierarchy-completeness-001",
      "target": "symptom",
      "supporting_items": [
        {
          "item_type": "entity",
          "item_id": "发热",
          "source_id": "doc-example-flu-guideline-chunk-001",
          "file_path": "流行性感冒诊疗方案.pdf",
          "evidence_status": "grounded"
        }
      ],
      "missing_evidence": [],
      "confidence": 0.73
    }
  ]
}
```

落盘：

```text
llm_evidence_map.json
llm_evidence_map.md
```

### 5. Propose

LLM 基于已解释问题、缺失分支推断和证据映射生成 proposal。所有 proposal 必须通过 `validate_proposal()`。

允许优先生成的 proposal 类型：

```text
hierarchy_rule_change
ontology_rule_change
relation_rule_change
source_evidence_repair
quality_report_note
review_context_request
```

规则：

- mutation proposal 必须 `requires_approval=true`。
- `evidence` 不能为空。
- 无证据时生成 `review_context_request` 或 `missing_evidence` 记录，不生成 mutation proposal。
- `expected_metric_change` 只能写已有 metric key，例如 `hierarchy_completeness`、`evidence_grounding`、`web_readability`。
- LLM 输出必须是 JSON，再由后端转换为 `ImprovementProposal`。

落盘：

```text
proposals.generated.yaml
approval_queue.md
improvement_backlog.md
```

### 6. Rank Repairs

LLM 对候选 proposal 和非 mutation 修复建议排序，给维护者一个处理顺序。

排序维度：

- 质量收益
- 医学风险
- 证据充分度
- 人工成本
- 是否影响 rebuild
- 是否重复历史 rejected 方案

输出：

```json
{
  "repair_plan": [
    {
      "rank": 1,
      "proposal_id": "proposal-hierarchy-branch-symptom-001",
      "priority": "high",
      "reason": "证据充分，能提升 hierarchy_completeness，且不直接改医学事实。",
      "risk": "medium",
      "human_checks": ["确认分支命名", "确认不会重复 rejected_changes"]
    }
  ]
}
```

落盘：

```text
llm_repair_plan.json
llm_repair_plan.md
```

### 7. Judge

Judge 是最后一道 LLM 预审，不重新生成方案，只评估 proposal 是否可信。

输入：

- proposal
- evidence map
- repair ranking
- rejected history
- optional patch candidate

输出：

```json
{
  "proposal_id": "proposal-hierarchy-branch-symptom-001",
  "decision": "needs_human",
  "reason": "该变更会影响医学层级规则，必须人工确认。",
  "risk_override": "medium",
  "required_human_checks": ["确认源文档支持该分支", "确认分支命名符合 ontology"],
  "patch_consistency": {
    "matches_proposal": true,
    "touches_allowed_files": true,
    "introduces_unsupported_medical_claim": false
  }
}
```

落盘：

```text
llm_judge_report.json
llm_judge_report.md
```

## Trace 和停止条件

`llm_review_trace.json` 扩展为阶段级 trace：

```json
{
  "workspace": "influenza_medical_v1",
  "profile": "clinical_guideline_zh",
  "mode": "agent_pipeline",
  "started_at": "2026-06-19T00:00:00Z",
  "completed_at": "2026-06-19T00:01:00Z",
  "stop_reason": "pending_human_review",
  "stages": [
    {
      "stage": "explain",
      "state": "completed",
      "model": "deepseek-v4-pro",
      "input_token_estimate": 2400,
      "output_token_estimate": 900,
      "artifact_keys": ["llm_issue_analysis"]
    },
    {
      "stage": "propose",
      "state": "completed",
      "proposal_ids": ["proposal-hierarchy-branch-symptom-001"]
    }
  ],
  "proposal_ids": ["proposal-hierarchy-branch-symptom-001"]
}
```

停止条件：

- 有 proposal 进入人工审批：`pending_human_review`
- 所有 finding 都缺证据：`needs_more_evidence`
- LLM 输出解析失败：`invalid_llm_output`
- proposal 全部无法通过校验：`all_proposals_invalid`
- context 超预算：`context_too_large`
- LLM 客户端失败：`llm_client_error`
- 没有可处理 finding：`no_actionable_findings`

## API 设计

第一版沿用现有入口：

```text
POST /kb-iteration/{workspace}/llm-review/runs
GET  /kb-iteration/{workspace}/llm-review/trace
GET  /kb-iteration/{workspace}/llm-review/report
GET  /kb-iteration/{workspace}/llm-review/proposals
GET  /kb-iteration/{workspace}/llm-review/judge-report
GET  /kb-iteration/{workspace}/llm-review/context/{round_id}
```

新增 artifact 读取 key：

```text
llm_issue_analysis
llm_missing_branch_inference
llm_evidence_map
llm_repair_plan
llm_judge_report
```

`RunLLMReviewRequest` 建议增加：

```json
{
  "mode": "agent_pipeline",
  "max_stage_retries": 1,
  "allow_llm_judge": true,
  "require_human_for_mutation": true,
  "generate_patch_candidates": false
}
```

兼容规则：

- 默认 `mode=agent_pipeline`，必要时可以保留 `mode=single_review` 用于回退。
- 旧前端仍可读取 `llm_review_report.md` 和 `proposals.generated.yaml`。
- 所有路径读取继续使用 API 白名单，不能让客户端传任意路径。

## 前端设计

`LLM 审阅材料` 页面改为多阶段 Agent 视图，中文展示：

1. 阶段进度：Explain、Infer、Evidence、Propose、Rank、Judge。
2. 问题解释：显示 `llm_issue_analysis.md`。
3. 缺失分支推断：显示 `llm_missing_branch_inference.md`，并列出 required / present / missing。
4. 证据定位：显示 `llm_evidence_map.md`，突出 source_id、file_path 和 evidence status。
5. Proposal 生成：继续显示 `proposals.generated.yaml` 和 `approval_queue.md`。
6. 修复方案排序：显示 `llm_repair_plan.md`。
7. Judge 结果：显示 `llm_judge_report.md`。

页面固定提示：

```text
LLM Agent 只生成分析、proposal、证据定位和修复排序。所有会改变 KG、规则、prompt、workspace 或 WebUI 行为的 proposal 都必须人工审批。
```

## 错误处理

- 单个阶段失败时写入 trace，并生成可读失败报告。
- JSON 解析失败时允许一次修复重试；重试仍失败则停止，不写入 approval queue。
- Propose 阶段没有有效 proposal 时，保留 Explain / Infer / Evidence / Rank 的产物，方便用户理解为什么没有 proposal。
- Evidence 阶段找不到 source_id/file_path 时，只允许输出 `needs_more_evidence`，不允许生成 mutation proposal。
- Judge 失败时 proposal 可进入人工队列，但必须标记 `judge_unavailable`。

## 安全边界

1. LLM 不自动修改 KG。
2. LLM 不自动应用 patch。
3. LLM 不自动 rebuild workspace。
4. mutation proposal 必须 `requires_approval=true`。
5. proposal 必须通过 deterministic validator。
6. evidence 为空时不能进入 mutation proposal。
7. rejected history 必须进入上下文，避免重复建议。
8. LLM 输出不是医学证据，只能引用已有 source_id/file_path/chunk。
9. trace 和阶段产物必须落盘，方便审计。
10. `.env` 中的模型密钥只在后端读取，不进入 trace、report 或前端。

## 测试策略

后端测试：

- `quality.py` 输出明确的 hierarchy branch details。
- `agent_context.py` 能为层级缺失、证据缺失、泛化关系构建阶段上下文。
- 每个阶段的 mock LLM 输出能解析并落盘。
- Propose 阶段生成的 proposal 必须通过 `validate_proposal()`。
- 无证据时不会写 mutation proposal。
- Judge 阶段不能让高风险 mutation 自动通过。
- API 能运行 pipeline，并返回 trace、report、proposal 和新 artifact。

前端测试：

- `LLMReviewPanels` 渲染多阶段 Agent 进度。
- 页面显示问题解释、缺失分支推断、证据定位、修复排序和 Judge。
- 空 artifact 时显示中文空状态。
- proposal 审批仍需要人工理由、影响范围和验证说明。

验证命令建议：

```powershell
cd D:\LightRAG
uv run pytest tests/api/routes/test_kb_iteration_routes.py
uv run pytest tests/kb_iteration

cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx
npx --yes bun run lint
npx --yes bun run build
```

## 分阶段实施

### Phase 1: Deterministic context 增强

- 输出 hierarchy required / present / missing details。
- 更新 quality report 和 review context。
- 加后端单元测试。

### Phase 2: Agent pipeline 后端

- 新增 Explain / Infer / Evidence / Propose / Rank / Judge 阶段编排。
- 新增阶段 artifact。
- 用 mock LLM 测试完整 pipeline。

### Phase 3: API 兼容接入

- 让现有 `/llm-review/runs` 调用 pipeline。
- 新增 artifact keys。
- 保留旧 report/proposals 兼容。

### Phase 4: WebUI 展示

- 改造 `LLM 审阅材料` 页面为多阶段 Agent 视图。
- 显示新增 artifact 和中文状态。
- 保持 approval queue 人工审批。

### Phase 5: 真实模型验证

- 使用 Agent 专用 DeepSeek 配置运行 `influenza_medical_v1`。
- 验证不再只输出 `proposals: []`。
- 检查 proposal 是否有 evidence、risk、expected metric 和 judge。

## 验收标准

- 用户能在前端看到 LLM 参与的问题解释、缺失分支推断、证据定位、proposal、修复排序和 Judge。
- 层级缺失 finding 不再只有 “缺 4 个” 的计数，而是有具体 missing branch details。
- LLM 能在证据充分时生成至少一个通过校验的 proposal；证据不足时能清楚输出 `needs_more_evidence`。
- `approval_queue.md` 仍然是人工审批入口。
- LLM 失败不会破坏确定性审阅包。
- 测试、lint、build 通过。
