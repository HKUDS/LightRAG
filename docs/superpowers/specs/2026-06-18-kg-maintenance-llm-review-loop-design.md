# KG Maintenance LLM Review Loop Design

## 目标

在现有 KG 维护 agent 中加入一个循环式 LLM 审阅器，用于提升知识库质量分析、建议生成和候选修改准备的效率。该审阅器应读取现有确定性审阅产物，按需选择局部上下文，多轮分析问题，生成结构化 proposal 和可选候选 patch，并把结果交给审批/评判层处理。

该设计不让 LLM 直接修改医学事实、KG 数据、提示词、规则、WebUI 行为或 workspace。LLM 的输出只能作为分析、建议、候选 patch 和评判意见；医学事实仍然必须由 `source_id`、`file_path` 和 chunk 证据支撑。

第一验收场景仍以 `influenza_medical_v1` 为主，但设计必须支持未来其他医学 workspace。

## 当前状态

现有确定性 KG 迭代能力位于 `lightrag/kb_iteration/`：

- `snapshot.py`：从 `graph_chunk_entity_relation.graphml` 生成 `KGSnapshot` 和统计快照。
- `markdown.py`：生成 `kb_context.md`、`entity_catalog.md`、`relation_catalog.md`、`kg_structure.md` 等 LLM/人工可读记忆。
- `quality.py`：计算确定性质量分，识别泛化关系、值型实体、证据缺失、层级缺失、疾病中心过载等问题。
- `proposals.py`：校验 `ImprovementProposal`，写入 `approval_queue.md` 和 `improvement_backlog.md`。
- `diff.py`：比较前后快照，生成 diff 报告。
- `runner.py`：编排确定性审阅运行，停在 `pending_user_review`。

现有 WebUI 的 KG 维护页可以展示图谱、目录、证据、质量报告、审批队列、Diff 和规则记忆。缺口是：还没有 LLM 调用来分析这些产物、生成 proposal 或候选 patch。

## 核心决策

采用 Agent 循环式审阅器，而不是单次大上下文审阅器。

原因：

- 大型 KG 不适合一次性塞入 LLM。
- 循环式审阅可以先确定焦点，再按需取证，减少 token 成本。
- 每轮 trace 落盘后更容易审计和复盘。
- 可以把确定性质量检查、LLM 分析、LLM Judge、人工审批拆成清晰边界。
- 适合后续扩展到更多医学文档和 workspace。

## 总体流程

```text
确定性审阅 runner
→ 生成 snapshot / Markdown / quality_score
→ LLM Review Loop 开始
→ 选择当前最值得审阅的问题
→ 按需读取局部上下文
→ 生成诊断和待确认假设
→ 必要时再读取更多证据
→ 输出结构化 proposals
→ 生成可选候选 patch
→ 进入 Proposal Validator
→ 进入 LLM Judge / Human Approval Gate
→ 写入审批队列和审阅产物
```

LLM 可以：

- 分析当前 KG 的结构和质量问题。
- 生成证据绑定的诊断。
- 生成结构化 proposal。
- 生成候选 patch。
- 作为 Judge 预评判 proposal 和 patch。

LLM 不可以：

- 自动应用 patch。
- 自动修改 KG。
- 自动修改提示词、规则、本体或 WebUI。
- 自动重建 workspace。
- 把 LLM 判断当作医学证据。
- 发明原文没有支持的医学事实。

## 循环状态机

LLM Review Loop 使用显式状态机：

```text
observe
→ select_focus
→ retrieve_context
→ analyze
→ propose
→ validate
→ judge
→ queue
→ stop / continue
```

### observe

读取当前确定性审阅包：

```text
snapshots/kg_snapshot.json
snapshots/quality_score.json
quality_report.md
kb_context.md
entity_catalog.md
relation_catalog.md
kg_structure.md
quality_rules.md
known_issues.md
accepted_changes.md
rejected_changes.md
```

目标是形成当前 KG 状态概览，不读取全部原始图谱细节。

### select_focus

选择本轮最值得审阅的 1-3 个问题。默认优先级：

```text
critical blockers
→ 高风险 evidence 缺失
→ 泛化关系
→ 医学层级缺失
→ 疾病中心节点过载
→ 重复/值型实体
→ Web 可读性问题
```

### retrieve_context

根据本轮 focus 构造局部证据包。例如 focus 是 `generic_relation` 时，只取相关 edge、source/target 实体、证据元数据、chunk 窗口和历史规则。

### analyze

LLM 分析必须输出四类结果：

```text
confirmed_issue: 有足够证据的问题
hypothesis: 需要更多证据的假设
missing_evidence: 当前证据不足，不能下结论
out_of_scope: 不应由本轮处理的问题
```

只有 `confirmed_issue` 可以进入 proposal 生成。`hypothesis` 只能触发下一轮上下文检索或进入待查记录。

### propose

把 `confirmed_issue` 转换成结构化 `ImprovementProposal` 草案。proposal 必须包含目标、理由、证据、风险、置信度、预期指标变化和审批要求。

### validate

使用确定性校验：

- 字段完整。
- `type` 合法。
- `risk` 合法。
- `requires_approval` 正确。
- `evidence` 非空。
- patch target 合法。
- mutation 类型不能绕过审批。
- 不允许包含未经来源支持的新医学事实。

### judge

LLM Judge 只评判 proposal 和候选 patch，不重新生成方案。输出：

```text
recommend_accept
recommend_reject
needs_human
needs_more_evidence
```

高风险 mutation 即使被 Judge 推荐接受，也仍然必须进入人工确认。

### queue

写入审阅产物和审批队列：

```text
approval_queue.md
improvement_backlog.md
proposals.generated.yaml
llm_review_report.md
llm_judge_report.md
patch_candidates/<proposal_id>.patch
```

### stop / continue

默认停止条件：

- 达到 `max_review_rounds`。
- 本轮没有生成有效 proposal。
- 所有高优先级问题都已进入 proposal 或 backlog。
- 需要人工审批。
- 需要更多原文证据但当前系统无法自动取得。
- LLM Judge 判断 proposal 质量不足。
- token/cost 超过预算。
- 解析或校验失败超过阈值。

默认配置建议：

```yaml
max_review_rounds: 4
max_focus_items_per_round: 3
max_context_tokens_per_round: 12000
allow_llm_auto_accept: false
allow_low_risk_auto_reject: true
require_human_for_mutation: true
```

## 审批与评判层

审批层设计为可插拔 Approval Gate，而不是固定只能人工审批。

```text
LLM Review Loop
→ Proposal Validator
→ Approval Gate
   → Human Review
   → LLM Judge Review
   → Hybrid Review
→ accepted / rejected / deferred
```

### Human Required

默认安全模式。所有会改变行为、规则、数据或 workspace 的 proposal 都必须人工确认。

必须人工确认的类型包括：

```text
kg_fact_correction
prompt_edit
ontology_rule_change
hierarchy_rule_change
relation_rule_change
workspace_rebuild
web_display_change
```

### LLM Judge Pre-review

推荐默认开启。LLM Judge 先评判 proposal 质量，帮助维护者预筛：

- 是否有证据。
- 是否重复历史 rejected changes。
- 风险等级是否合理。
- 目标文件是否匹配。
- 预期指标变化是否可信。
- patch 是否和 proposal 一致。
- 是否存在医学事实幻觉。

Judge 输出：

```yaml
judge_decision: recommend_accept | recommend_reject | needs_human | needs_more_evidence
judge_reason: ""
risk_override: low | medium | high
required_human_checks: []
```

### LLM Delegated Low-risk

可选半自动模式，仅允许 LLM 自动处理极低风险项目，例如：

- `quality_report_note`
- 重复 proposal 去重。
- 明显无效 proposal 驳回。
- 格式修正。
- 非事实性报告整理。

自动决策必须写入：

```text
auto_accepted_by_llm_judge.md
auto_rejected_by_llm_judge.md
```

禁止 LLM 自动批准任何 mutation proposal。

## 新增文件与模块

建议新增后端模块：

```text
lightrag/kb_iteration/llm_review.py
lightrag/kb_iteration/review_loop.py
lightrag/kb_iteration/review_context.py
lightrag/kb_iteration/patches.py
lightrag/kb_iteration/prompts/planner_zh.md
lightrag/kb_iteration/prompts/reviewer_zh.md
lightrag/kb_iteration/prompts/judge_zh.md
lightrag/kb_iteration/prompts/patch_generator_zh.md
```

模块职责：

- `review_loop.py`：状态机、循环配置、停止条件、trace 记录。
- `review_context.py`：按 focus 从 snapshot/catalog/quality/rules 中构造局部上下文。
- `llm_review.py`：LLM 调用适配、结构化输出解析、prompt 渲染。
- `patches.py`：候选 patch 生成、白名单校验、patch 与 proposal 绑定。
- `prompts/*.md`：版本化 planner/reviewer/judge/patch generator prompt。

## 新增产物

输出目录仍为：

```text
work/kb-iteration/<workspace>/
```

新增产物：

```text
llm_review_report.md
llm_review_trace.json
llm_judge_report.md
proposals.generated.yaml
patch_candidates/
  <proposal_id>.patch
review_context/
  round-001-context.json
  round-002-context.json
```

### llm_review_trace.json

记录每一轮循环状态：

```json
{
  "workspace": "influenza_medical_v1",
  "profile": "clinical_guideline_zh",
  "started_at": "2026-06-18T00:00:00Z",
  "rounds": [
    {
      "round_id": "round-001",
      "focus": ["generic_relation", "hierarchy_missing_branch"],
      "state": "queued",
      "context_files": ["review_context/round-001-context.json"],
      "model": "configured-review-model",
      "input_token_estimate": 8200,
      "output_token_estimate": 1400,
      "proposal_ids": ["proposal-20260618-001"],
      "judge_decision": "needs_human"
    }
  ],
  "stop_reason": "pending_human_review"
}
```

### review_context

保存本轮 LLM 实际看到的证据包，不保存私密推理链：

```json
{
  "focus": ["generic_relation"],
  "quality_findings": [],
  "entities": [],
  "relations": [],
  "evidence_windows": [],
  "rules_memory": {
    "accepted_changes": [],
    "rejected_changes": []
  }
}
```

### llm_review_report.md

给维护者阅读的报告结构：

```markdown
# LLM Review Report

## Summary
## Confirmed Issues
## Hypotheses
## Missing Evidence
## Out Of Scope
## Generated Proposals
## Patch Candidates
## Human Review Required
```

### proposals.generated.yaml

兼容现有 `ImprovementProposal`，并扩展候选 patch 和 Judge 信息：

```yaml
proposals:
  - id: proposal-20260618-001
    type: hierarchy_rule_change
    target: lightrag/medical_kg/hierarchy.py
    proposed_change: Add a symptom grouping layer under clinical presentation.
    reason: Direct disease-to-leaf symptom edges reduce readability.
    evidence:
      - "quality finding: disease_hub_overload_ratio=0.62"
      - "edge:流感->高热不退"
    confidence: 0.78
    risk: medium
    requires_approval: true
    expected_metric_change:
      hierarchy_completeness: 5
      web_readability: 8
    patch_candidate: patch_candidates/proposal-20260618-001.patch
    judge:
      decision: needs_human
      reason: "Patch changes hierarchy behavior and requires maintainer approval."
```

### patch_candidates

候选 patch 只能与 proposal id 一一对应，不能自动应用。

允许的目标范围建议：

```text
prompts/
lightrag/medical_kg/
docs/
lightrag_webui/src/components/kg-maintenance/
```

禁止默认修改：

```text
data/rag_storage/
work/kb-iteration/
.env
uv.lock
```

## Proposal 类型

保留已有类型：

```text
prompt_edit
ontology_rule_change
hierarchy_rule_change
relation_rule_change
workspace_rebuild
kg_fact_correction
web_display_change
quality_report_note
```

建议新增：

```text
source_evidence_repair
synonym_merge_rule
relation_keyword_mapping
review_context_request
llm_judge_rejection
```

`review_context_request` 不表示修改建议，而是记录下一轮需要补充的证据。`llm_judge_rejection` 用于记录 LLM Judge 对低质量 proposal 的拒绝建议。

## Prompt 设计

### Planner Prompt

输入：

```text
kb_context.md
quality_score.json
quality_report.md
known_issues.md
accepted_changes.md
rejected_changes.md
上一轮 trace 摘要
```

输出：

```json
{
  "focus_items": [
    {
      "category": "generic_relation",
      "reason": "Generic relation count is high.",
      "priority": "high",
      "needed_context": ["relations", "source_target_entities", "evidence_windows"]
    }
  ],
  "stop_if": []
}
```

### Reviewer Prompt

输入：

```text
本轮 focus
review_context/round-xxx-context.json
quality_rules.md
known_issues.md
accepted_changes.md
rejected_changes.md
```

输出：

```json
{
  "confirmed_issues": [],
  "hypotheses": [],
  "missing_evidence": [],
  "out_of_scope": [],
  "proposals": []
}
```

约束：

- 不得新增原文未支持的医学事实。
- 不得把 LLM 判断当作证据。
- 只能基于提供的 entity/relation/source/chunk 证据生成 proposal。
- 缺少证据时输出 `missing_evidence`。
- 涉及 prompt/rule/KG/workspace/WebUI 行为的建议必须 `requires_approval=true`。

### Judge Prompt

输入：

```text
proposal
patch_candidate
review_context
quality_rules
accepted/rejected history
```

输出：

```json
{
  "decision": "recommend_accept | recommend_reject | needs_human | needs_more_evidence",
  "reason": "",
  "risk_override": "low | medium | high",
  "required_human_checks": [],
  "patch_consistency": {
    "matches_proposal": true,
    "touches_allowed_files": true,
    "introduces_unsupported_medical_claim": false
  }
}
```

## 效率优化

- 先确定性过滤，再调用 LLM。
- 每轮只审阅 1-3 个 focus。
- 按 focus 取证，不读全图。
- 保存 `review_context`，相同上下文可复用。
- 注入 `rejected_changes.md` 摘要，避免重复建议。
- Planner 和 Judge 可使用低成本模型，Reviewer/Patch Generator 使用能力更强模型。
- 对 `focus + evidence ids + rules hash` 做 cache key，避免重复 LLM 调用。
- patch 延迟生成：只有 evidence 足够、proposal 通过校验、target 在白名单内时才生成候选 patch。
- LLM 输出必须结构化，解析失败最多修复一次；仍失败则记录错误，不进入审批队列。

## API 设计

在现有 `/kb-iteration` 下新增：

```text
POST /kb-iteration/{workspace}/llm-review/runs
GET  /kb-iteration/{workspace}/llm-review/trace
GET  /kb-iteration/{workspace}/llm-review/report
GET  /kb-iteration/{workspace}/llm-review/proposals
GET  /kb-iteration/{workspace}/llm-review/judge-report
GET  /kb-iteration/{workspace}/llm-review/context/{round_id}
GET  /kb-iteration/{workspace}/llm-review/patches/{proposal_id}
POST /kb-iteration/{workspace}/llm-review/proposals/{proposal_id}/judge
```

`POST /llm-review/runs` 请求体：

```json
{
  "profile": "clinical_guideline_zh",
  "mode": "loop",
  "max_review_rounds": 4,
  "max_focus_items_per_round": 3,
  "allow_llm_judge": true,
  "allow_llm_auto_accept": false,
  "allow_low_risk_auto_reject": true,
  "generate_patch_candidates": true,
  "require_human_for_mutation": true
}
```

API 安全规则：

- workspace 必须使用现有 `validate_workspace`。
- 所有 artifact 和 patch 读取必须白名单化。
- `POST /llm-review/runs` 必须有 workspace 级锁。
- 不提供 apply-patch API。
- 不提供 rebuild API，除非未来单独设计审批执行器。

## WebUI 设计

在 KG 维护左侧导航的审阅分组新增：

```text
LLM 审阅
候选 Patch
Judge 评判
```

### LLM 审阅页

展示：

- 运行状态。
- 本轮 focus。
- 已读取证据包。
- Confirmed issues。
- Hypotheses。
- Missing evidence。
- Out of scope。
- Generated proposals。
- Stop reason。

### 候选 Patch 页

展示：

- proposal id。
- 目标文件。
- 风险等级。
- patch diff。
- 是否通过 deterministic validator。
- 是否通过 LLM Judge。
- 人工审批状态。

### Judge 评判页

展示：

- `recommend_accept`
- `recommend_reject`
- `needs_human`
- `needs_more_evidence`
- `risk_override`
- `required_human_checks`
- `patch_consistency`

### 审批队列增强

在现有审批队列增加状态徽标：

```text
确定性校验：通过 / 失败
LLM Judge：建议接受 / 建议拒绝 / 需要人工 / 需要更多证据
Patch：有候选 / 无候选 / patch 不一致
审批：待处理 / 已接受 / 已拒绝 / 已暂缓
```

页面固定提示：

```text
LLM 审阅只产生分析、proposal 和候选 patch。医学事实仍以 source_id、file_path 和 chunk 为准。高影响变更必须由维护者批准。
```

## 使用路径

```text
1. 运行确定性审阅。
2. 点击运行 LLM 审阅。
3. 查看 LLM 审阅报告。
4. 查看 Judge 建议。
5. 打开候选 Patch。
6. 在审批队列接受、拒绝或暂缓。
7. 后续由人工或未来审批执行器应用已批准 patch。
8. 重建 workspace。
9. 查看 Diff 验证结果。
```

## 错误处理

- LLM 超时：保留确定性审阅结果，记录 `llm_review_error`。
- JSON 解析失败：最多要求模型修复一次。
- 修复仍失败：本轮失败，不写入 `approval_queue.md`。
- 上下文过大：缩小 focus 后重试。
- proposal 全部无效：`stop_reason=all_proposals_invalid`。
- Judge 失败：proposal 可进入人工队列，但标记 `judge_unavailable`。
- patch 与 proposal 不一致：阻断 patch，proposal 标记 `patch_invalid`。

## 测试策略

### 后端测试

- 循环状态机推进和停止条件。
- 上下文选择：泛化关系、层级问题、证据缺失、rejected history 注入。
- LLM 输出解析：合法 JSON、格式错误 JSON、缺字段 proposal、unsupported medical claim、patch 越权、重复 proposal。
- Judge 规则：高风险 mutation 不能自动通过；低风险 report note 可按配置自动处理；patch 不一致必须阻断。
- API 路由：run、trace、report、proposals、judge-report、context、patch 读取、workspace 校验、路径白名单、并发锁。

### 前端测试

- LLM 审阅页空状态。
- 运行中状态。
- trace 展示。
- proposal 状态徽标。
- patch diff 展示。
- Judge 建议展示。
- 审批队列增强。
- 错误状态。

## 风险控制

必须保持以下护栏：

```text
1. 不自动应用 patch。
2. 不自动改 KG。
3. 不自动重建 workspace。
4. mutation proposal 必须 requires_approval=true。
5. patch 目标必须在白名单。
6. LLM 输出必须结构化解析。
7. LLM 输出必须经过 proposals.py 校验。
8. evidence 为空不能进入 mutation proposal。
9. rejected_changes.md 注入上下文，避免重复建议。
10. 每轮 trace 落盘，不能只存在内存。
```

## 实施阶段

### Phase 1：文件模型与状态机

新增 `review_loop.py`、`review_context.py`、`llm_review.py`，先用 mock LLM 跑通循环。

验收：

- 能生成 `llm_review_trace.json`。
- 能生成 `llm_review_report.md`。
- 能在 mock proposal 下写 `proposals.generated.yaml`。

### Phase 2：Proposal 与候选 Patch

新增 `patches.py`、`patch_candidates/`、proposal 扩展字段和 patch 白名单校验。

验收：

- 候选 patch 能生成。
- 越权 patch 被拒绝。
- patch 不会自动应用。

### Phase 3：LLM Judge

新增 Judge prompt、Judge report、Judge result schema 和低风险自动处理规则。

验收：

- Judge 可以推荐。
- 高风险 mutation 仍必须人工审批。

### Phase 4：API 接入

新增 `/kb-iteration/{workspace}/llm-review/*` 路由。

验收：

- WebUI 可触发 LLM 审阅。
- 可读取 trace、report、proposals、patch、judge。
- 并发锁有效。

### Phase 5：WebUI 接入

新增 LLM 审阅、候选 Patch、Judge 评判页面，并增强审批队列。

验收：

- 维护者能从 KG 维护页运行 LLM 审阅。
- 能看到每轮 focus 和停止原因。
- 能看到 proposal、Judge 建议和 patch。
- 能最终人工接受、拒绝或暂缓。

## 最小可用版本

第一版建议只实现：

```text
mock/real LLM review runner
Planner + Reviewer
proposals.generated.yaml
llm_review_report.md
llm_review_trace.json
approval_queue.md 写入
```

第一版暂不实现：

```text
patch generator
auto decision
真实执行器
workspace rebuild
```

这样可以快速验证 LLM 审阅是否真正提升知识库维护效率，同时保持医学 KG 的安全边界。

## 验收标准

- LLM Review Loop 可以在确定性 runner 后独立运行。
- LLM 每轮只读取按 focus 构造的审计上下文。
- 所有 LLM 输出都落盘到可审计文件。
- 有效 proposal 进入 `approval_queue.md` 或 `improvement_backlog.md`。
- mutation proposal 必须人工审批。
- Judge 可以预评判，但不能绕过高风险审批。
- 候选 patch 可审阅但不会自动应用。
- WebUI 能展示 LLM 审阅过程、proposal、Judge 结果和候选 patch。
- 失败时保留确定性审阅结果，不破坏现有 KG 维护流程。
