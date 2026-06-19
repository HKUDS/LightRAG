import {
  getKBIterationArtifact,
  getKBIterationLLMJudgeReport,
  getKBIterationLLMReviewProposals,
  getKBIterationLLMReviewReport,
  getKBIterationLLMReviewTrace,
  getKBIterationQuality,
  getKBIterationRules,
  getKBIterationSummary,
  recordKBIterationProposalDecision,
  type KBIterationProposalDecision,
  type KBIterationQualityResponse,
  type KBIterationRulesResponse,
  type KBIterationSummaryResponse
} from '@/api/lightrag'
import type {
  ProposalDecisionReview,
  ProposalSummary
} from '@/components/kg-maintenance/kgMaintenanceData'

export const normalizeOptionalMarkdown = (value: unknown): string =>
  typeof value === 'string' ? value : ''

export const normalizeWorkspaceList = (value: unknown): string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string') ? value : []

export const shouldApplyWorkspaceResponse = (
  requestWorkspace: string,
  getCurrentWorkspace: () => string | null
): boolean => getCurrentWorkspace() === requestWorkspace

export const applyWorkspaceResponse = (
  requestWorkspace: string,
  getCurrentWorkspace: () => string | null,
  apply: () => void
): boolean => {
  if (!shouldApplyWorkspaceResponse(requestWorkspace, getCurrentWorkspace)) return false
  apply()
  return true
}

export const runWorkspaceAction = async <T,>({
  requestWorkspace,
  getCurrentWorkspace,
  action,
  onSuccess,
  onError,
  onComplete
}: {
  requestWorkspace: string
  getCurrentWorkspace: () => string | null
  action: () => Promise<T>
  onSuccess?: (value: T, shouldApply: () => boolean) => void | Promise<void>
  onError?: (error: unknown, shouldApply: () => boolean) => void | Promise<void>
  onComplete?: () => void
}): Promise<void> => {
  const shouldApply = () => shouldApplyWorkspaceResponse(requestWorkspace, getCurrentWorkspace)
  try {
    const value = await action()
    if (shouldApply()) {
      await onSuccess?.(value, shouldApply)
    }
  } catch (error) {
    if (shouldApply()) {
      await onError?.(error, shouldApply)
    }
  } finally {
    onComplete?.()
  }
}

export const optionalMissingResponse = async <T,>(
  loader: () => Promise<T>,
  fallback: T
): Promise<T> => {
  try {
    return await loader()
  } catch (error) {
    if (isMissingResourceError(error)) return fallback
    throw error
  }
}

const artifactPayload = (artifact: Awaited<ReturnType<typeof getKBIterationArtifact>>) =>
  'payload' in artifact ? artifact.payload : null

const optionalArtifactContent = async (
  loader: () => Promise<Awaited<ReturnType<typeof getKBIterationArtifact>>>
) => {
  const artifact = await optionalMissingResponse<
    Awaited<ReturnType<typeof getKBIterationArtifact>> | null
  >(loader, null)
  return artifact && 'content' in artifact ? artifact.content : ''
}

const optionalArtifactPayload = async (
  loader: () => Promise<Awaited<ReturnType<typeof getKBIterationArtifact>>>
) => {
  const artifact = await optionalMissingResponse<
    Awaited<ReturnType<typeof getKBIterationArtifact>> | null
  >(loader, null)
  return artifact ? artifactPayload(artifact) : null
}

type KGMaintenanceWorkspaceBundle = {
  summaryPayload: KBIterationSummaryResponse
  qualityPayload: KBIterationQualityResponse
  rulesPayload: KBIterationRulesResponse
  kbContextArtifact: string
  kgSnapshotArtifact: unknown
  qualityScoreArtifact: unknown
  approvalArtifact: string
  backlogArtifact: string
  logArtifact: string
  llmTraceArtifact: unknown
  llmReportArtifact: string
  llmProposalsArtifact: string
  llmJudgeReportArtifact: string
  llmIssueAnalysisArtifact: string
  llmMissingBranchInferenceArtifact: string
  llmEvidenceMapArtifact: string
  llmRepairPlanArtifact: string
}

type KGMaintenanceWorkspaceLoaders = {
  getSummary: typeof getKBIterationSummary
  getQuality: typeof getKBIterationQuality
  getRules: typeof getKBIterationRules
  getArtifact: typeof getKBIterationArtifact
  getTrace: typeof getKBIterationLLMReviewTrace
  getReport: typeof getKBIterationLLMReviewReport
  getProposals: typeof getKBIterationLLMReviewProposals
  getJudgeReport: typeof getKBIterationLLMJudgeReport
}

const defaultWorkspaceLoaders: KGMaintenanceWorkspaceLoaders = {
  getSummary: getKBIterationSummary,
  getQuality: getKBIterationQuality,
  getRules: getKBIterationRules,
  getArtifact: getKBIterationArtifact,
  getTrace: getKBIterationLLMReviewTrace,
  getReport: getKBIterationLLMReviewReport,
  getProposals: getKBIterationLLMReviewProposals,
  getJudgeReport: getKBIterationLLMJudgeReport
}

export async function loadKGMaintenanceWorkspaceBundle(
  requestWorkspace: string,
  loaders: KGMaintenanceWorkspaceLoaders = defaultWorkspaceLoaders
): Promise<KGMaintenanceWorkspaceBundle> {
  const [
    summaryPayload,
    qualityPayload,
    rulesPayload,
    kbContextArtifact,
    kgSnapshotArtifact,
    qualityScoreArtifact,
    approvalArtifact,
    backlogArtifact,
    logArtifact,
    llmTraceArtifact,
    llmReportArtifact,
    llmProposalsArtifact,
    llmJudgeReportArtifact,
    llmIssueAnalysisArtifact,
    llmMissingBranchInferenceArtifact,
    llmEvidenceMapArtifact,
    llmRepairPlanArtifact
  ] = await Promise.all([
    loaders.getSummary(requestWorkspace),
    loaders.getQuality(requestWorkspace),
    loaders.getRules(requestWorkspace),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'kb_context')),
    optionalArtifactPayload(() => loaders.getArtifact(requestWorkspace, 'kg_snapshot')),
    optionalArtifactPayload(() => loaders.getArtifact(requestWorkspace, 'quality_score')),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'approval_queue')),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'improvement_backlog')),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'iteration_log')),
    optionalArtifactPayload(() => loaders.getTrace(requestWorkspace)),
    optionalArtifactContent(() => loaders.getReport(requestWorkspace)),
    optionalArtifactContent(() => loaders.getProposals(requestWorkspace)),
    optionalArtifactContent(() => loaders.getJudgeReport(requestWorkspace)),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'llm_issue_analysis')),
    optionalArtifactContent(() =>
      loaders.getArtifact(requestWorkspace, 'llm_missing_branch_inference')
    ),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'llm_evidence_map')),
    optionalArtifactContent(() => loaders.getArtifact(requestWorkspace, 'llm_repair_plan'))
  ])

  return {
    summaryPayload,
    qualityPayload,
    rulesPayload,
    kbContextArtifact,
    kgSnapshotArtifact,
    qualityScoreArtifact,
    approvalArtifact,
    backlogArtifact,
    logArtifact,
    llmTraceArtifact,
    llmReportArtifact,
    llmProposalsArtifact,
    llmJudgeReportArtifact,
    llmIssueAnalysisArtifact,
    llmMissingBranchInferenceArtifact,
    llmEvidenceMapArtifact,
    llmRepairPlanArtifact
  }
}

type ProposalDecisionActionArgs = {
  requestWorkspace: string
  getCurrentWorkspace: () => string | null
  proposal: ProposalSummary
  decision: KBIterationProposalDecision
  review: ProposalDecisionReview
  reloadWorkspaceData: () => Promise<void>
  recordDecision?: typeof recordKBIterationProposalDecision
  onError?: (error: unknown) => void
}

export async function submitProposalDecisionForWorkspace({
  requestWorkspace,
  getCurrentWorkspace,
  proposal,
  decision,
  review,
  reloadWorkspaceData,
  recordDecision = recordKBIterationProposalDecision,
  onError
}: ProposalDecisionActionArgs): Promise<void> {
  await runWorkspaceAction({
    requestWorkspace,
    getCurrentWorkspace,
    action: () =>
      recordDecision(requestWorkspace, proposal.id, decision, {
        reviewer: 'maintainer',
        reason: review.reason,
        impact_scope: review.impactScope,
        verification: review.verification
      }),
    onSuccess: async (_result, shouldApply) => {
      if (!shouldApply()) return
      await reloadWorkspaceData()
    },
    onError: (error) => {
      onError?.(error)
    }
  })
}

function isMissingResourceError(error: unknown): boolean {
  const status = getErrorStatus(error)
  if (status === 404) return true

  const message = error instanceof Error ? error.message : String(error)
  const firstLine = message.split(/\r?\n/, 1)[0]?.trim() || ''
  return /^404\b/.test(firstLine)
}

function getErrorStatus(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') return undefined
  const record = error as Record<string, any>
  const responseStatus = record.response?.status
  if (typeof responseStatus === 'number') return responseStatus
  const status = record.status ?? record.statusCode
  return typeof status === 'number' ? status : undefined
}
