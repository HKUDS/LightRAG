# KG Iteration 多阶段 LLM Agent Pipeline 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 KB iteration 的 LLM 审阅从单段 review loop 升级为多阶段 Agent pipeline，让 LLM 参与问题解释、缺失分支推断、证据定位、proposal 生成、修复方案排序和 Judge 预审，同时保持人工审批边界。

**Architecture:** 先增强确定性质量产物，让 `quality_score.json` 和 review context 明确写出医学层级 required / present / missing 分支。再新增后端 Agent pipeline，将 Explain、Infer、Evidence、Propose、Rank、Judge 六个阶段的输入、输出和 trace 全部落盘。最后扩展 API 和 WebUI，让用户在“LLM 审阅材料”里看到中文多阶段 Agent 过程。

**Tech Stack:** Python dataclasses / JSON / YAML、FastAPI route tests、LightRAG `lightrag.kb_iteration` 模块、React 19 + TypeScript + Bun tests、Tailwind UI components。

---

## 设计来源

执行前先读：

- `D:\LightRAG\docs\superpowers\specs\2026-06-19-kg-iteration-multistage-agent-pipeline-design.md`
- `D:\LightRAG\docs\superpowers\plans\2026-06-19-kg-iteration-multistage-agent-pipeline-implementation.md`
- `D:\LightRAG\AGENTS.md`

说明：

- 本文件是中文执行版。
- 英文计划文件包含更完整的测试代码和实现代码片段。
- 如果中文说明和英文详细代码片段冲突，以中文目标和安全边界为准，以英文代码片段为具体实现参考。

## 关键约束

- 不泄露 `.env` 中的 API key。
- 不让 LLM 自动修改 KG。
- 不让 LLM 自动应用 patch。
- 不让 LLM 自动 rebuild workspace。
- 所有 mutation proposal 必须 `requires_approval=true`。
- LLM 输出不是医学证据，只能引用已有 `source_id`、`file_path`、chunk、entity、relation 或 quality metric。
- 前端新增文本使用中文，保留 LLM、KG、proposal、source_id、file_path、JSON、Markdown、workspace、profile 等技术词。
- 目前工作区已有脏改动：`env.example`、`lightrag/api/routers/kb_iteration_routes.py`、`tests/api/routes/test_kb_iteration_routes.py`。执行时只能在需要修改这些文件时叠加改动，不能回退用户或之前任务的内容。

## 文件边界

新增后端文件：

- `D:\LightRAG\lightrag\kb_iteration\agent_context.py`
  - 构建 observe context 和各阶段上下文。
- `D:\LightRAG\lightrag\kb_iteration\agent_outputs.py`
  - 解析阶段 JSON，写 JSON/Markdown artifact，校验 propose 阶段 proposal。
- `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
  - 编排 Explain / Infer / Evidence / Propose / Rank / Judge。
- `D:\LightRAG\lightrag\kb_iteration\prompts\explain_zh.md`
- `D:\LightRAG\lightrag\kb_iteration\prompts\infer_branches_zh.md`
- `D:\LightRAG\lightrag\kb_iteration\prompts\locate_evidence_zh.md`
- `D:\LightRAG\lightrag\kb_iteration\prompts\propose_zh.md`
- `D:\LightRAG\lightrag\kb_iteration\prompts\rank_repairs_zh.md`

新增后端测试：

- `D:\LightRAG\tests\kg\test_kb_iteration_agent_context.py`
- `D:\LightRAG\tests\kg\test_kb_iteration_agent_outputs.py`
- `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`

修改后端文件：

- `D:\LightRAG\lightrag\kb_iteration\models.py`
- `D:\LightRAG\lightrag\kb_iteration\quality.py`
- `D:\LightRAG\lightrag\kb_iteration\review_context.py`
- `D:\LightRAG\lightrag\kb_iteration\review_loop.py`
- `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`

修改前端文件：

- `D:\LightRAG\lightrag_webui\src\api\lightrag.ts`
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\kgIterationLoadUtils.ts`
- `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`

## Task 1：补齐确定性层级分支详情

**目的：** 解决当前 LLM 只看到“缺 4 个分支”而不知道“缺哪几个”的问题。

**文件：**

- Modify: `D:\LightRAG\lightrag\kb_iteration\models.py`
- Modify: `D:\LightRAG\lightrag\kb_iteration\quality.py`
- Modify: `D:\LightRAG\tests\kg\test_kb_iteration_quality.py`

- [ ] **Step 1：先写失败测试**

在 `tests/kg/test_kb_iteration_quality.py` 增加测试，覆盖：

- medical profile 下 `QualityScore.details["hierarchy_branches"]` 存在。
- `required` 等于 `TOP_LEVEL_MEDICAL_CATEGORIES`。
- `present` 包含通过 key、label、alias、`properties.medical_group` 命中的分支。
- `missing` 明确列出未命中的分支。
- `quality_score.json` 写出 `details.hierarchy_branches`。
- `quality_report.md` 写出 `## Hierarchy Branches`。

- [ ] **Step 2：运行失败测试**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_quality.py -q
```

Expected:

- FAIL，因为 `QualityScore` 还没有 `details` 字段。

- [ ] **Step 3：扩展 `QualityScore`**

在 `models.py` 的 `QualityScore` 增加：

```python
details: dict[str, Any] = field(default_factory=dict)
```

并在 `to_dict()` 输出：

```python
"details": self.details
```

- [ ] **Step 4：实现 hierarchy branch detail**

在 `quality.py` 中新增或重构：

- `_hierarchy_details(snapshot)`
- `_hierarchy_metrics_from_details(details)`
- `_matched_hierarchy_nodes_by_key(snapshot)`

输出结构：

```json
{
  "hierarchy_branches": {
    "required": [{"key": "symptom", "label": "症状", "aliases": ["临床表现"]}],
    "present": [{"key": "pathogen", "label": "病原体", "aliases": ["病原"], "matched_node_ids": ["病原体"]}],
    "missing": [{"key": "symptom", "label": "症状", "aliases": ["临床表现"]}]
  }
}
```

- [ ] **Step 5：更新 quality report**

在 `_quality_report()` 中新增 `## Hierarchy Branches` 段落，至少显示：

- required 数量和 key/label
- present 数量、key/label、matched node ids
- missing 数量和 key/label

- [ ] **Step 6：运行测试并提交**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_quality.py -q
git add lightrag/kb_iteration/models.py lightrag/kb_iteration/quality.py tests/kg/test_kb_iteration_quality.py
git commit -m "feat: expose kg hierarchy branch details"
```

Expected:

- PASS。

## Task 2：构建 Agent 上下文

**目的：** 为每个 Agent 阶段提供结构化上下文，避免 LLM 读到无关 edge。

**文件：**

- Create: `D:\LightRAG\lightrag\kb_iteration\agent_context.py`
- Create: `D:\LightRAG\tests\kg\test_kb_iteration_agent_context.py`
- Modify: `D:\LightRAG\lightrag\kb_iteration\review_context.py`
- Modify: `D:\LightRAG\tests\kg\test_kb_iteration_review_context.py`

- [ ] **Step 1：写失败测试**

新增测试覆盖：

- `build_agent_observation()` 读取 workspace、quality、hierarchy branches、artifact 状态、accepted/rejected memory。
- `build_stage_context(package, workspace="influenza_medical_v1", stage="locate_evidence")` 能根据 missing branch 选择候选 entity/relation/evidence windows。
- `write_agent_context()` 写入 `agent_context/<stage>-context.json`。
- legacy `build_review_context()` 在 hierarchy-only focus 下包含 `hierarchy_branches`，且不回退到无关 `edges[:10]`。

- [ ] **Step 2：运行失败测试**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_context.py tests/kg/test_kb_iteration_review_context.py -q
```

Expected:

- FAIL，因为 `agent_context.py` 还不存在，legacy context 也没有 `hierarchy_branches`。

- [ ] **Step 3：实现 `agent_context.py`**

实现函数：

```python
build_agent_observation(package_dir, *, workspace)
build_stage_context(package_dir, *, workspace, stage, previous_outputs=None)
write_agent_context(package_dir, stage, context)
```

核心输出：

- `workspace`
- `snapshot` 摘要
- `quality` 摘要
- `hierarchy_branches`
- `artifact_status`
- `kb_context`
- `rules_memory`
- `candidate_entities`
- `candidate_relations`
- `evidence_windows`

- [ ] **Step 4：修改 legacy `review_context.py`**

要求：

- 在返回 context 中加入 `hierarchy_branches`。
- 增加 `_hierarchy_branches(quality)` helper。
- 在 `_select_edges()` 中，如果 focus 是 `hierarchy_missing_branch` 且没有证据 edge，不再返回 `edges[:10]`，而是返回 `[]`。

- [ ] **Step 5：运行测试并提交**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_context.py tests/kg/test_kb_iteration_review_context.py -q
git add lightrag/kb_iteration/agent_context.py lightrag/kb_iteration/review_context.py tests/kg/test_kb_iteration_agent_context.py tests/kg/test_kb_iteration_review_context.py
git commit -m "feat: build staged kb agent contexts"
```

Expected:

- PASS。

## Task 3：解析阶段输出并写 Agent artifacts

**目的：** 每个 LLM 阶段都输出可审计 JSON/Markdown，proposal 阶段仍走 deterministic validator。

**文件：**

- Create: `D:\LightRAG\lightrag\kb_iteration\agent_outputs.py`
- Create: `D:\LightRAG\tests\kg\test_kb_iteration_agent_outputs.py`

- [ ] **Step 1：写失败测试**

新增测试覆盖：

- 非 JSON 输出报错。
- JSON array 输出报错。
- `propose` 阶段能解析并校验 `ImprovementProposal`。
- 非 `review_context_request` 的 mutation proposal 如果 `evidence=[]` 必须失败。
- `stage_output_to_markdown()` 能渲染中文标题。
- `write_agent_stage_artifacts()` 写入 JSON 和 Markdown。

- [ ] **Step 2：运行失败测试**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_outputs.py -q
```

Expected:

- FAIL，因为 `agent_outputs.py` 不存在。

- [ ] **Step 3：实现 `agent_outputs.py`**

实现：

```python
@dataclass(frozen=True)
class AgentStageOutput:
    stage: str
    payload: dict[str, Any]
    proposals: list[ImprovementProposal] = field(default_factory=list)
```

实现函数：

```python
parse_agent_stage_output(stage, raw_text)
write_agent_stage_artifacts(output_dir, output)
stage_output_to_markdown(output)
```

阶段 artifact 映射：

```text
explain -> llm_issue_analysis.json / llm_issue_analysis.md
infer_branches -> llm_missing_branch_inference.json / llm_missing_branch_inference.md
locate_evidence -> llm_evidence_map.json / llm_evidence_map.md
rank_repairs -> llm_repair_plan.json / llm_repair_plan.md
judge -> llm_judge_report.json / llm_judge_report.md
```

- [ ] **Step 4：运行测试并提交**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_outputs.py -q
git add lightrag/kb_iteration/agent_outputs.py tests/kg/test_kb_iteration_agent_outputs.py
git commit -m "feat: parse kg agent stage outputs"
```

Expected:

- PASS。

## Task 4：实现多阶段 Agent pipeline

**目的：** 串联 Explain、Infer、Evidence、Propose、Rank、Judge，并生成阶段 trace。

**文件：**

- Create: `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
- Create: `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`
- Create: `D:\LightRAG\lightrag\kb_iteration\prompts\explain_zh.md`
- Create: `D:\LightRAG\lightrag\kb_iteration\prompts\infer_branches_zh.md`
- Create: `D:\LightRAG\lightrag\kb_iteration\prompts\locate_evidence_zh.md`
- Create: `D:\LightRAG\lightrag\kb_iteration\prompts\propose_zh.md`
- Create: `D:\LightRAG\lightrag\kb_iteration\prompts\rank_repairs_zh.md`
- Modify: `D:\LightRAG\tests\kg\test_kb_iteration_review_loop.py`

- [ ] **Step 1：写失败测试**

新增 `test_kb_iteration_agent_pipeline.py`，用 mock client 顺序返回 6 个阶段输出，覆盖：

- pipeline 写出 `llm_review_trace.json`，其中 `mode="agent_pipeline"`。
- trace 的 stage 顺序是 `explain`、`infer_branches`、`locate_evidence`、`propose`、`rank_repairs`、`judge`。
- 写出：
  - `llm_issue_analysis.md`
  - `llm_missing_branch_inference.md`
  - `llm_evidence_map.md`
  - `llm_repair_plan.md`
  - `llm_judge_report.md`
  - `proposals.generated.yaml`
  - `approval_queue.md`
- 有 proposal 时 `stop_reason="pending_human_review"`。
- 无证据或无 proposal 时仍保留分析产物，并返回 `needs_more_evidence`。

- [ ] **Step 2：运行失败测试**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_pipeline.py -q
```

Expected:

- FAIL，因为 `agent_pipeline.py` 不存在。

- [ ] **Step 3：创建 prompt 文件**

每个 prompt 都必须要求：

- 只输出 JSON object。
- 不输出隐藏推理链。
- 不把 LLM 判断当医学证据。
- mutation proposal 必须 `requires_approval=true`。

文件：

```text
lightrag/kb_iteration/prompts/explain_zh.md
lightrag/kb_iteration/prompts/infer_branches_zh.md
lightrag/kb_iteration/prompts/locate_evidence_zh.md
lightrag/kb_iteration/prompts/propose_zh.md
lightrag/kb_iteration/prompts/rank_repairs_zh.md
```

- [ ] **Step 4：实现 `agent_pipeline.py`**

核心类型：

```python
@dataclass(frozen=True)
class LLMAgentPipelineConfig:
    max_context_tokens_per_stage: int = 12000
    max_stage_retries: int = 1
    allow_llm_judge: bool = True
    generate_patch_candidates: bool = False
    require_human_for_mutation: bool = True

@dataclass(frozen=True)
class LLMAgentPipelineResult:
    output_dir: Path
    stop_reason: str
    proposal_ids: list[str] = field(default_factory=list)
    artifact_paths: dict[str, Path] = field(default_factory=dict)
```

核心函数：

```python
run_llm_agent_pipeline(
    *,
    workspace: str,
    package_dir: str | Path,
    client: LLMReviewClient,
    config: LLMAgentPipelineConfig | None = None,
    profile: str | None = None,
) -> LLMAgentPipelineResult
```

实现要求：

- 先校验 `snapshots/kg_snapshot.json` 和 `snapshots/quality_score.json` 存在。
- 每阶段写 `agent_context/<stage>-context.json`。
- 每阶段调用 `client.complete(system_prompt=stage_prompt, user_prompt=context_json)`。
- 每阶段用 `parse_agent_stage_output()` 解析。
- `propose` 阶段用 `write_llm_review_artifacts()`、`write_approval_queue()`、`write_improvement_backlog()`。
- 最终写 `llm_review_trace.json`。

- [ ] **Step 5：运行 pipeline 和旧 loop 测试**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_pipeline.py tests/kg/test_kb_iteration_review_loop.py -q
```

Expected:

- PASS。
- 旧 `review_loop` 行为不应被破坏。

- [ ] **Step 6：提交**

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/agent_pipeline.py lightrag/kb_iteration/prompts/explain_zh.md lightrag/kb_iteration/prompts/infer_branches_zh.md lightrag/kb_iteration/prompts/locate_evidence_zh.md lightrag/kb_iteration/prompts/propose_zh.md lightrag/kb_iteration/prompts/rank_repairs_zh.md tests/kg/test_kb_iteration_agent_pipeline.py
git commit -m "feat: run multistage kg llm agent pipeline"
```

## Task 5：API 接入和 artifact keys

**目的：** 让现有 `POST /llm-review/runs` 默认运行多阶段 pipeline，同时保留旧 `loop` 模式回退。

**文件：**

- Modify: `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`
- Modify: `D:\LightRAG\tests\api\routes\test_kb_iteration_routes.py`
- Modify: `D:\LightRAG\lightrag_webui\src\api\lightrag.ts`

- [ ] **Step 1：写失败 API 测试**

新增测试覆盖：

- artifact route 能读取：
  - `llm_issue_analysis`
  - `llm_missing_branch_inference`
  - `llm_evidence_map`
  - `llm_repair_plan`
- `POST /kb-iteration/{workspace}/llm-review/runs` 默认调用 `run_llm_agent_pipeline()`。
- 请求 `{"mode": "loop"}` 时仍调用旧 `run_llm_review_loop()`。

- [ ] **Step 2：运行失败测试**

```powershell
cd D:\LightRAG
uv run pytest tests/api/routes/test_kb_iteration_routes.py -q
```

Expected:

- FAIL，因为 API 还没有新 artifact keys 和 pipeline dispatch。

- [ ] **Step 3：修改后端路由**

在 `kb_iteration_routes.py`：

- import `LLMAgentPipelineConfig` 和 `run_llm_agent_pipeline`。
- `ARTIFACTS` 增加：

```python
"llm_issue_analysis": ("llm_issue_analysis.md", "text/markdown")
"llm_missing_branch_inference": ("llm_missing_branch_inference.md", "text/markdown")
"llm_evidence_map": ("llm_evidence_map.md", "text/markdown")
"llm_repair_plan": ("llm_repair_plan.md", "text/markdown")
```

- `RunLLMReviewRequest.mode` 改为：

```python
Literal["agent_pipeline", "loop"] = "agent_pipeline"
```

- 增加：

```python
max_stage_retries: int = Field(default=1, ge=0, le=3)
```

- 在 `create_llm_review_run()` 中按 `request.mode` 分发到新旧 runner。

- [ ] **Step 4：修改前端 API 类型**

在 `lightrag_webui/src/api/lightrag.ts`：

```ts
export type KBIterationLLMReviewRunRequest = {
  profile?: string | null
  mode?: 'agent_pipeline' | 'loop'
  max_review_rounds?: number
  max_focus_items_per_round?: number
  max_context_tokens_per_round?: number
  max_stage_retries?: number
  allow_llm_judge?: boolean
  allow_llm_auto_accept?: boolean
  allow_low_risk_auto_reject?: boolean
  generate_patch_candidates?: boolean
  require_human_for_mutation?: boolean
}
```

- [ ] **Step 5：运行测试并提交**

```powershell
cd D:\LightRAG
uv run pytest tests/api/routes/test_kb_iteration_routes.py -q
git add lightrag/api/routers/kb_iteration_routes.py tests/api/routes/test_kb_iteration_routes.py lightrag_webui/src/api/lightrag.ts
git commit -m "feat: expose multistage kg llm agent api"
```

Expected:

- PASS。

## Task 6：前端加载 Agent artifacts

**目的：** WebUI 加载新增的 LLM Agent 阶段产物。

**文件：**

- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\kgIterationLoadUtils.ts`
- Create or Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\kgIterationLoadUtils.test.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`

- [ ] **Step 1：写失败前端加载测试**

测试 `loadKGMaintenanceWorkspaceBundle()` 会请求并返回：

- `llm_issue_analysis`
- `llm_missing_branch_inference`
- `llm_evidence_map`
- `llm_repair_plan`

- [ ] **Step 2：运行失败测试**

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/kgIterationLoadUtils.test.ts
```

Expected:

- FAIL，因为 bundle 类型还没有这些字段。

- [ ] **Step 3：扩展 bundle**

在 `kgIterationLoadUtils.ts` 中给 `KGMaintenanceWorkspaceBundle` 增加：

```ts
llmIssueAnalysisArtifact: string
llmMissingBranchInferenceArtifact: string
llmEvidenceMapArtifact: string
llmRepairPlanArtifact: string
```

在 `Promise.all` 中新增 `optionalArtifactContent()` 请求四个 artifact。

- [ ] **Step 4：扩展 Console state**

在 `KGMaintenanceConsole.tsx` 增加 state：

```ts
const [llmIssueAnalysis, setLlmIssueAnalysis] = useState('')
const [llmMissingBranchInference, setLlmMissingBranchInference] = useState('')
const [llmEvidenceMap, setLlmEvidenceMap] = useState('')
const [llmRepairPlan, setLlmRepairPlan] = useState('')
```

加载 bundle 后写入 state，并传给 `MainPanel`。

- [ ] **Step 5：运行测试并提交**

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/kgIterationLoadUtils.test.ts

cd D:\LightRAG
git add lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.ts lightrag_webui/src/components/kg-maintenance/kgIterationLoadUtils.test.ts lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "feat: load kg llm agent artifacts"
```

Expected:

- PASS。

## Task 7：前端展示多阶段 Agent

**目的：** 在“LLM 审阅材料”中展示 Explain、Infer、Evidence、Propose、Rank、Judge。

**文件：**

- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`

- [ ] **Step 1：写失败 UI 测试**

测试 `LLMReviewPanel` 能显示：

- `多阶段 LLM Agent`
- `Explain`
- `Infer`
- `Evidence`
- `Propose`
- `Rank`
- `Judge`
- `问题解释`
- `缺失分支推断`
- `证据定位`
- `修复方案排序`
- 安全提示：LLM Agent 只生成分析，不自动修改 KG。

- [ ] **Step 2：运行失败测试**

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx
```

Expected:

- FAIL，因为 `LLMReviewPanel` 尚未接收新 artifact props。

- [ ] **Step 3：扩展 `LLMReviewPanel` props**

新增：

```ts
issueAnalysis: string
missingBranchInference: string
evidenceMap: string
repairPlan: string
```

新增 trace stage 类型：

```ts
export type TraceStage = {
  stage?: string
  state?: string
  artifact_keys?: string[]
  proposal_ids?: string[]
}
```

- [ ] **Step 4：渲染多阶段视图**

在 `LLMReviewPanel` 里展示：

- stage cards
- issue analysis artifact
- missing branch inference artifact
- evidence map artifact
- repair plan artifact
- 原有 review report 和 proposals

新增 helper：

```ts
function stageLabel(stage: string) {
  const labels: Record<string, string> = {
    explain: 'Explain / 问题解释',
    infer_branches: 'Infer / 缺失分支推断',
    locate_evidence: 'Evidence / 证据定位',
    propose: 'Propose / proposal 生成',
    rank_repairs: 'Rank / 修复排序',
    judge: 'Judge / 评判'
  }
  return labels[stage] || stage || '未知阶段'
}
```

- [ ] **Step 5：Console 调用 LLM 审阅时启用 agent mode**

在 `handleRunLLMReview()` 的 request 中增加：

```ts
mode: 'agent_pipeline',
max_stage_retries: 1,
```

并把四个新 artifact prop 传给 `LLMReviewPanel`。

- [ ] **Step 6：运行测试并提交**

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/kgIterationLoadUtils.test.ts

cd D:\LightRAG
git add lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "feat: show multistage kg llm agent review"
```

Expected:

- PASS。

## Task 8：整体验证和真实模型试跑

**目的：** 验证后端、前端、构建和真实 DeepSeek Agent run。

**文件：**

- 仅在发现实际问题时修改相关源文件和测试。

- [ ] **Step 1：跑后端聚焦测试**

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_quality.py tests/kg/test_kb_iteration_review_context.py tests/kg/test_kb_iteration_agent_context.py tests/kg/test_kb_iteration_agent_outputs.py tests/kg/test_kb_iteration_agent_pipeline.py tests/kg/test_kb_iteration_review_loop.py tests/api/routes/test_kb_iteration_routes.py -q
```

Expected:

- PASS。

- [ ] **Step 2：跑前端聚焦测试**

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/kgIterationLoadUtils.test.ts src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected:

- PASS。

- [ ] **Step 3：跑 lint 和 build**

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun run lint
npx --yes bun run build
```

Expected:

- PASS。

- [ ] **Step 4：真实模型试跑**

API server 运行后执行：

```powershell
$body = @{
  profile = "clinical_guideline_zh"
  mode = "agent_pipeline"
  allow_llm_judge = $true
  require_human_for_mutation = $true
  generate_patch_candidates = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9621/kb-iteration/influenza_medical_v1/llm-review/runs" `
  -ContentType "application/json" `
  -Body $body
```

Expected:

- 返回包含 `stopReason`。
- `D:\LightRAG\work\kb-iteration\influenza_medical_v1\llm_review_trace.json` 中有 `"mode": "agent_pipeline"`。
- 生成：
  - `llm_issue_analysis.md`
  - `llm_missing_branch_inference.md`
  - `llm_evidence_map.md`
  - `llm_repair_plan.md`
  - `llm_judge_report.md`
  - `proposals.generated.yaml`

- [ ] **Step 5：浏览器检查**

打开当前 WebUI：

```text
http://127.0.0.1:5176/#/
```

检查：

- `LLM 审阅材料` 显示 `多阶段 LLM Agent`。
- 能看到 `问题解释`、`缺失分支推断`、`证据定位`、`修复方案排序`、`Judge`。
- 页面明确说明 LLM 不自动修改 KG。
- 如果生成 proposal，仍进入 `Proposal 审批`。
- 如果没有 proposal，也能看到原因和证据不足说明。

- [ ] **Step 6：如果有修复，提交**

如果验证过程中修改了文件：

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration lightrag/api/routers/kb_iteration_routes.py tests/kg tests/api/routes/test_kb_iteration_routes.py lightrag_webui/src/api/lightrag.ts lightrag_webui/src/components/kg-maintenance lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "fix: polish multistage kg llm agent pipeline"
```

如果没有修改，不要创建空提交。

## 完成标准

- [ ] `quality_score.json` 有具体 hierarchy branch details。
- [ ] hierarchy-only review context 不再塞入无关 edge fallback。
- [ ] 新增 `agent_context.py`、`agent_outputs.py`、`agent_pipeline.py`。
- [ ] 多阶段 LLM Agent 生成 JSON/Markdown 阶段产物。
- [ ] `llm-review` API 默认运行 `agent_pipeline`，并保留 `loop` 回退。
- [ ] 新 artifact keys 可通过 API 读取。
- [ ] WebUI 显示问题解释、缺失分支推断、证据定位、proposal、修复排序和 Judge。
- [ ] mutation proposal 仍然必须人工审批。
- [ ] 后端聚焦测试通过。
- [ ] 前端聚焦测试、lint、build 通过。
- [ ] 真实 DeepSeek-backed run 能产出 stage trace 或清晰失败 artifact，且不暴露 API key。
