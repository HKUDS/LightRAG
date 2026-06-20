export type KGMaintenanceWorkflowStepId =
  | 'check'
  | 'llm-review'
  | 'approval'
  | 'execute'
  | 'validate'

export type KGMaintenanceArtifactDefinition = {
  key: string
  title: string
  sourceFile: string
  zhFile: string
  step: KGMaintenanceWorkflowStepId
  contentType: 'application/json' | 'text/markdown' | string
}

export const WORKFLOW_STEPS: KGMaintenanceWorkflowStepId[] = [
  'check',
  'llm-review',
  'approval',
  'execute',
  'validate'
]

export const KG_MAINTENANCE_ARTIFACTS: KGMaintenanceArtifactDefinition[] = [
  {
    key: 'kb_context',
    title: '当前 KB 摘要',
    sourceFile: 'kb_context.md',
    zhFile: 'kb_context.zh.md',
    step: 'check',
    contentType: 'text/markdown'
  },
  {
    key: 'quality_report',
    title: '质量报告',
    sourceFile: 'quality_report.md',
    zhFile: 'quality_report.zh.md',
    step: 'check',
    contentType: 'text/markdown'
  },
  {
    key: 'kg_snapshot',
    title: '图谱快照',
    sourceFile: 'snapshots/kg_snapshot.json',
    zhFile: 'snapshots/kg_snapshot.zh.json',
    step: 'check',
    contentType: 'application/json'
  },
  {
    key: 'quality_score',
    title: '质量分数',
    sourceFile: 'snapshots/quality_score.json',
    zhFile: 'snapshots/quality_score.zh.json',
    step: 'check',
    contentType: 'application/json'
  },
  {
    key: 'entity_catalog',
    title: '实体目录',
    sourceFile: 'entity_catalog.md',
    zhFile: 'entity_catalog.zh.md',
    step: 'check',
    contentType: 'text/markdown'
  },
  {
    key: 'relation_catalog',
    title: '关系目录',
    sourceFile: 'relation_catalog.md',
    zhFile: 'relation_catalog.zh.md',
    step: 'check',
    contentType: 'text/markdown'
  },
  {
    key: 'kg_structure',
    title: '图谱结构',
    sourceFile: 'kg_structure.md',
    zhFile: 'kg_structure.zh.md',
    step: 'check',
    contentType: 'text/markdown'
  },
  {
    key: 'source_coverage',
    title: '来源覆盖',
    sourceFile: 'snapshots/source_coverage.json',
    zhFile: 'snapshots/source_coverage.zh.json',
    step: 'check',
    contentType: 'application/json'
  },
  {
    key: 'llm_issue_analysis',
    title: 'LLM 问题分析',
    sourceFile: 'llm_issue_analysis.md',
    zhFile: 'llm_issue_analysis.zh.md',
    step: 'llm-review',
    contentType: 'text/markdown'
  },
  {
    key: 'proposals_generated',
    title: '生成的 Proposal',
    sourceFile: 'proposals.generated.yaml',
    zhFile: 'proposals.generated.zh.yaml',
    step: 'llm-review',
    contentType: 'text/markdown'
  },
  {
    key: 'llm_review_trace',
    title: 'LLM 审阅轨迹',
    sourceFile: 'llm_review_trace.json',
    zhFile: 'llm_review_trace.zh.json',
    step: 'llm-review',
    contentType: 'application/json'
  },
  {
    key: 'llm_missing_branch_inference',
    title: '缺失分支推断',
    sourceFile: 'llm_missing_branch_inference.md',
    zhFile: 'llm_missing_branch_inference.zh.md',
    step: 'llm-review',
    contentType: 'text/markdown'
  },
  {
    key: 'llm_evidence_map',
    title: 'LLM 证据映射',
    sourceFile: 'llm_evidence_map.md',
    zhFile: 'llm_evidence_map.zh.md',
    step: 'llm-review',
    contentType: 'text/markdown'
  },
  {
    key: 'llm_repair_plan',
    title: 'LLM 修复计划',
    sourceFile: 'llm_repair_plan.md',
    zhFile: 'llm_repair_plan.zh.md',
    step: 'llm-review',
    contentType: 'text/markdown'
  },
  {
    key: 'llm_judge_report',
    title: 'LLM 裁判报告',
    sourceFile: 'llm_judge_report.md',
    zhFile: 'llm_judge_report.zh.md',
    step: 'llm-review',
    contentType: 'text/markdown'
  },
  {
    key: 'approval_queue',
    title: '待审批 Proposal',
    sourceFile: 'approval_queue.md',
    zhFile: 'approval_queue.zh.md',
    step: 'approval',
    contentType: 'text/markdown'
  },
  {
    key: 'proposal_revision_requests',
    title: 'Proposal 修订请求',
    sourceFile: 'proposal_revision_requests.md',
    zhFile: 'proposal_revision_requests.zh.md',
    step: 'approval',
    contentType: 'text/markdown'
  },
  {
    key: 'agent_memory_summary',
    title: 'Agent 压缩记忆',
    sourceFile: 'agent_memory_summary.md',
    zhFile: 'agent_memory_summary.zh.md',
    step: 'approval',
    contentType: 'text/markdown'
  },
  {
    key: 'accepted_changes',
    title: '已接受变更',
    sourceFile: 'accepted_changes.md',
    zhFile: 'accepted_changes.zh.md',
    step: 'approval',
    contentType: 'text/markdown'
  },
  {
    key: 'rejected_changes',
    title: '已拒绝变更',
    sourceFile: 'rejected_changes.md',
    zhFile: 'rejected_changes.zh.md',
    step: 'approval',
    contentType: 'text/markdown'
  },
  {
    key: 'accepted_changes_apply_result',
    title: '真实应用结果',
    sourceFile: 'accepted_changes_apply_result.md',
    zhFile: 'accepted_changes_apply_result.zh.md',
    step: 'execute',
    contentType: 'text/markdown'
  },
  {
    key: 'accepted_changes_execution',
    title: 'Agent 执行记录',
    sourceFile: 'accepted_changes_execution.md',
    zhFile: 'accepted_changes_execution.zh.md',
    step: 'execute',
    contentType: 'text/markdown'
  },
  {
    key: 'iteration_log',
    title: '迭代日志',
    sourceFile: 'iteration_log.md',
    zhFile: 'iteration_log.zh.md',
    step: 'execute',
    contentType: 'text/markdown'
  },
  {
    key: 'improvement_backlog',
    title: '改进 Backlog',
    sourceFile: 'improvement_backlog.md',
    zhFile: 'improvement_backlog.zh.md',
    step: 'validate',
    contentType: 'text/markdown'
  },
  {
    key: 'quality_rules',
    title: '质量规则',
    sourceFile: 'quality_rules.md',
    zhFile: 'quality_rules.zh.md',
    step: 'validate',
    contentType: 'text/markdown'
  },
  {
    key: 'known_issues',
    title: '已知问题',
    sourceFile: 'known_issues.md',
    zhFile: 'known_issues.zh.md',
    step: 'validate',
    contentType: 'text/markdown'
  }
]

export function findArtifactDefinition(key: string): KGMaintenanceArtifactDefinition | undefined {
  return KG_MAINTENANCE_ARTIFACTS.find((artifact) => artifact.key === key)
}

export function artifactsForStep(
  step: KGMaintenanceWorkflowStepId
): KGMaintenanceArtifactDefinition[] {
  return KG_MAINTENANCE_ARTIFACTS.filter((artifact) => artifact.step === step)
}
