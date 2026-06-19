import type { KGMaintenanceSection } from '@/stores/kgMaintenance'

export type KGMaintenanceNextActionId =
  | 'run-check'
  | 'run-llm-review'
  | 'open-approval'
  | 'execute-accepted'
  | 'validate-result'
  | 'start-next-iteration'

export type KGMaintenanceNextAction = {
  id: KGMaintenanceNextActionId
  label: string
  section: KGMaintenanceSection
  reason: string
}

type KGMaintenanceQualitySummary = {
  findings?: unknown[] | null
}

type KGMaintenanceSummaryLike = {
  pendingApprovalCount?: number | null
  quality?: KGMaintenanceQualitySummary | null
}

export type ResolveKGMaintenanceNextActionArgs = {
  summary?: KGMaintenanceSummaryLike | null
  qualityFindings?: unknown[] | null
  llmTrace?: unknown | null
  pendingApprovalCount?: number | null
  acceptedChanges?: string | null
  acceptedApplyResult?: string | null
  justExecuted?: boolean
}

const acceptedDecisionSectionPattern = /^##\s+/m

export function resolveKGMaintenanceNextAction({
  summary,
  qualityFindings,
  llmTrace,
  pendingApprovalCount,
  acceptedChanges,
  acceptedApplyResult,
  justExecuted = false
}: ResolveKGMaintenanceNextActionArgs): KGMaintenanceNextAction {
  if (!summary) {
    return {
      id: 'run-check',
      label: '运行检查',
      section: 'check',
      reason: '还没有维护摘要，先生成检查包。'
    }
  }

  const findings = qualityFindings ?? summary.quality?.findings ?? []
  if (findings.length > 0 && !llmTrace) {
    return {
      id: 'run-llm-review',
      label: '运行 LLM 复核',
      section: 'llm-review',
      reason: '质量检查仍有发现项，需要让 LLM 生成修复建议。'
    }
  }

  const approvalCount = pendingApprovalCount ?? summary.pendingApprovalCount ?? 0
  if (approvalCount > 0) {
    return {
      id: 'open-approval',
      label: '查看待审批',
      section: 'approval',
      reason: `还有 ${approvalCount} 条 proposal 等待人工审批。`
    }
  }

  const hasAcceptedDecisionSections = acceptedDecisionSectionPattern.test(acceptedChanges ?? '')
  const hasApplyResult = Boolean(acceptedApplyResult?.trim())
  if (hasAcceptedDecisionSections && !hasApplyResult) {
    return {
      id: 'execute-accepted',
      label: '执行已接受变更',
      section: 'execute',
      reason: 'accepted_changes.md 中已有已接受决策，尚未生成执行结果。'
    }
  }

  if (justExecuted) {
    return {
      id: 'validate-result',
      label: '验证执行结果',
      section: 'validate',
      reason: '刚执行过变更，需要检查结果并确认是否进入下一轮。'
    }
  }

  return {
    id: 'start-next-iteration',
    label: '开始下一轮复核',
    section: 'llm-review',
    reason: '当前没有待处理事项，可以启动下一轮维护。'
  }
}
