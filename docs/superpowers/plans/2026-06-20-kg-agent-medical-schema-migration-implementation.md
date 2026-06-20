# KG Agent 医学 Schema 迁移改造实施计划

> 执行方式：Subagent-Driven Development。每个实现任务需要实现者、规格审阅、代码质量审阅三段式推进；如果审阅发现问题，先修复再复审。

## 目标

把知识库迭代 Agent 从“发现通用 KG 质量问题”升级为“能发现医院/医学场景中的关系 schema 缺陷、生成可审批 proposal、由后端白名单安全执行、并持续压缩长期记忆”的医学知识图谱迭代 Agent。

典型问题包括：

- `干咳 -> 属于 -> 流行性感冒` 这类症状指向疾病的错误关系。
- `属于` 被滥用于疾病-症状、药物-适应症、诊断依据等非分类关系。
- 剂量、频次、阈值、时间窗等值节点被当成主实体。
- 同义词、别名、英文缩写和中文名缺少约束合并。
- LLM 多轮迭代后 accepted/rejected 记忆膨胀，影响上下文和速度。

## 架构原则

- LLM 负责解释问题、推断缺失分支、定位证据、生成 proposal、排序修复方案和 judge 风险。
- LLM 不能直接改 KG，不能把自己的推理当医学证据。
- KG 真正修改只由 `lightrag/kb_iteration/apply.py` 的白名单确定性动作执行。
- 所有会改变 KG、规则、prompt、workspace 或 WebUI 行为的 proposal 必须先人工接受。
- 前端继续使用无图谱画布的工作台，用中文展示医学 schema 问题、审批、执行结果和压缩记忆。

## 医学关系 Schema v1

本阶段采用面向医院/患者/医生诊断场景的关系族：

- 术语与本体：`is_a`, `part_of`, `names`, `maps_to_code`
- 疾病与诊断：`causative_agent`, `has_manifestation`, `has_complication`, `has_risk_factor`, `has_diagnostic_criterion`, `criterion_requires`, `differential_with`, `has_severity_grade`, `has_evidence`, `ruled_out`, `due_to`
- 检查与证据：`orders_test`, `has_result`, `observes`, `has_value`, `has_interpretation`, `uses_specimen`, `performed_by_method`, `supports_or_refutes`
- 药物与治疗：`has_active_ingredient`, `belongs_to_drug_class`, `has_indication`, `recommends`, `recommended_for`, `not_recommended_for`, `contraindicated_for`, `precaution_for`, `has_dosing_regimen`, `may_cause_adverse_reaction`, `interaction_with`, `monitor_with`
- 疫苗与预防：`targets_disease`, `reduces_risk_of`, `has_dose_schedule`, `risk_group_for`, `defined_by_characteristic`
- 指南与医院服务：`evidenced_by`, `asserted_by`, `issued_by`, `valid_during`, `provided_by`, `available_at`

## 任务拆分与状态

- [x] Task 1: 建立正式医学关系 schema registry，并为 `ImprovementProposal` 增加 `action_payload`。
- [x] Task 2: 在质量评分中增加确定性的医学 KG 缺陷检测，输出 `medical_schema_issues` 和 `entity_cleanup_issues`。
- [x] Task 3: 更新多阶段 Agent 上下文和中文 prompt，让 LLM 能看到 schema 问题、证据、拒绝原因和压缩记忆。
- [x] Task 4: 扩展确定性 apply，支持安全子集的 `medical_relation_schema_migration` 和 `value_node_to_qualifier`。
- [x] Task 5: 增加记忆整理层，生成小而稳定的 `agent_memory_summary.md`，避免 accepted/rejected 历史无限膨胀。
- [x] Task 6: 更新 API 与 WebUI，中文展示医学 schema 问题表、审批队列、执行结果和 Agent 压缩记忆。
- [x] Task 7: 完成 focused 后端/API、前端、lint、build 和真实 workspace 验证。

## 文件范围

核心后端：

- `lightrag/kb_iteration/medical_schema.py`
- `lightrag/kb_iteration/models.py`
- `lightrag/kb_iteration/proposals.py`
- `lightrag/kb_iteration/quality.py`
- `lightrag/kb_iteration/agent_context.py`
- `lightrag/kb_iteration/agent_pipeline.py`
- `lightrag/kb_iteration/review_context.py`
- `lightrag/kb_iteration/review_loop.py`
- `lightrag/kb_iteration/apply.py`
- `lightrag/kb_iteration/memory.py`
- `lightrag/kb_iteration/prompts/*.md`
- `lightrag/api/routers/kb_iteration_routes.py`

WebUI：

- `lightrag_webui/src/api/lightrag.ts`
- `lightrag_webui/src/features/KGMaintenanceConsole.tsx`
- `lightrag_webui/src/components/kg-maintenance/kgMaintenanceArtifacts.ts`
- `lightrag_webui/src/components/kg-maintenance/IterationWorkbenchPanels.tsx`
- `lightrag_webui/src/components/kg-maintenance/SnapshotTables.tsx`

测试：

- `tests/kg/test_kb_iteration_medical_schema.py`
- `tests/kg/test_kb_iteration_quality.py`
- `tests/kg/test_kb_iteration_agent_context.py`
- `tests/kg/test_kb_iteration_agent_pipeline.py`
- `tests/kg/test_kb_iteration_apply.py`
- `tests/kg/test_kb_iteration_memory.py`
- `tests/api/routes/test_kb_iteration_routes.py`
- `lightrag_webui/src/components/kg-maintenance/*.test.ts*`

文档：

- `docs/KBIterationAgent.md`
- `docs/superpowers/plans/2026-06-20-kg-agent-medical-schema-migration-implementation.md`

## 关键行为

- `quality_score.json` 顶层指标包含 `medical_schema_issue_count`，详情包含 `details.medical_schema_issues`。
- 每条医学 schema issue 尽量带上 `edge_id`, `source_id`, `file_path`, `current_keywords`, `candidate_predicates` 等确定性定位字段。
- 医学变更 proposal 必须携带 `action_payload`，例如替换关系方向、替换 canonical predicate、或把孤立值节点转成边 qualifier。
- 后端 apply 会重新读取当前图谱并校验 edge/node 是否仍匹配，避免把过期 proposal 应用到已经变化的 KG。
- 被拒绝、要求返工、已接受、已执行的记录会进入压缩记忆，而不是每次把完整历史都塞给 LLM。
- 前端展示中文标签；即使源 artifact 是英文，WebUI 也优先展示中文标签和中文解释。

## 验证记录

已完成的 focused 验证：

- 后端/API：`231 passed`
- 后端 ruff：通过
- 前端 focused tests：`76 pass`
- 前端 lint：通过，保留一个既有 Fast Refresh warning
- 前端 build：通过
- 真实 workspace `influenza_medical_v1`：
  - 确定性检查发现 `medical_schema_issue_count = 815`
  - LLM Agent review 生成 proposal 并停在 `pending_human_review`
  - 接受一条安全迁移后执行成功，schema issue 数量 `815 -> 814`
  - 第二次执行为幂等结果 `no_applicable_changes`

## 提交注意

- `env.example` 当前有无关脏改动，本计划不提交也不回滚。
- `task_plan.md`, `findings.md`, `progress.md` 是本地规划记忆，除非用户明确要求，否则不强制加入版本提交。
- `docs/tutorials/` 中旧教程草稿不是本次医学 schema 改造的核心交付，提交前需单独确认编码和内容质量。
