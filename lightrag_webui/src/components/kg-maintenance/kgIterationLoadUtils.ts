import {
  getKBIterationArtifact,
  getKBIterationDisplayArtifact,
  getKBIterationLLMJudgeReport,
  getKBIterationLLMReviewProposals,
  getKBIterationLLMReviewReport,
  getKBIterationLLMReviewTrace,
  getKBIterationQuality,
  getKBIterationRules,
  getKBIterationSummary,
  recordKBIterationProposalDecision,
  requestKBIterationProposalRevision,
  type KBIterationArtifactResponse,
  type KBIterationDisplayArtifactResponse,
  type KBIterationProposalDecision,
  type KBIterationQualityResponse,
  type KBIterationRulesResponse,
  type KBIterationSummaryResponse
} from '@/api/lightrag'
import { findArtifactDefinition } from '@/components/kg-maintenance/kgMaintenanceArtifacts'
import {
  buildProposalDecisionReview,
  type ProposalDecisionReview,
  type ProposalSummary
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

export const runWorkspaceAction = async <T>({
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

export const optionalMissingResponse = async <T>(
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

const artifactContentOrPayload = (
  artifact: KBIterationArtifactResponse | KBIterationDisplayArtifactResponse | null
) => {
  if (!artifact) return null
  if ('payload' in artifact) return artifact.payload
  return artifact.content
}

const artifactTextOrEmpty = (
  artifact: KBIterationArtifactResponse | KBIterationDisplayArtifactResponse | null
) => {
  if (!artifact || !('content' in artifact)) return ''
  return artifact.content
}

const normalizeRecordArtifact = (value: unknown): Record<string, any> | null =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, any>)
    : null

export const normalizeTraceArtifactForLogic = (
  sourceTraceArtifact: unknown,
  _displayTraceArtifact: unknown
): Record<string, any> | null => normalizeRecordArtifact(sourceTraceArtifact)

type LoadedDisplayArtifact = {
  key: string
  artifact: KBIterationArtifactResponse | KBIterationDisplayArtifactResponse | null
  sourceArtifact: KBIterationArtifactResponse | null
  displayArtifact: KBIterationDisplayArtifactResponse
}

export type KGMaintenanceDisplayArtifacts = Record<string, KBIterationDisplayArtifactResponse>

export function isGeneratedDisplayArtifact(
  artifact: KBIterationDisplayArtifactResponse | undefined
): boolean {
  const display = artifact?.display
  if (!display) return false
  if (display.fallbackToSource) return false
  return display.generated === true || display.zhExists === true
}

type OptionalDisplayArtifactOptions = {
  fallbackContentType?: string
  fallbackLoader?: () => Promise<KBIterationArtifactResponse>
  loadSource?: boolean
}

const optionalDisplayArtifact = async (
  requestWorkspace: string,
  artifactKey: string,
  loaders: KGMaintenanceWorkspaceLoaders,
  {
    fallbackContentType = 'text/markdown',
    fallbackLoader = () => loaders.getArtifact(requestWorkspace, artifactKey),
    loadSource = false
  }: OptionalDisplayArtifactOptions = {}
): Promise<LoadedDisplayArtifact> => {
  let displayArtifact: KBIterationDisplayArtifactResponse | null = null
  try {
    displayArtifact = await loaders.getDisplayArtifact(requestWorkspace, artifactKey)
  } catch {
    // Display artifacts are optional; source loaders preserve the existing fallback behavior.
  }

  const sourceArtifact =
    loadSource || !displayArtifact
      ? await optionalMissingResponse<KBIterationArtifactResponse | null>(fallbackLoader, null)
      : null
  const artifact = displayArtifact ?? sourceArtifact

  return {
    key: artifactKey,
    artifact,
    sourceArtifact,
    displayArtifact:
      displayArtifact ?? fallbackDisplayArtifact(artifactKey, sourceArtifact, fallbackContentType)
  }
}

const fallbackDisplayArtifact = (
  artifactKey: string,
  artifact: KBIterationArtifactResponse | null,
  fallbackContentType: string
): KBIterationDisplayArtifactResponse => {
  const contentType = artifact?.contentType ?? fallbackContentType
  const artifactDefinition = findArtifactDefinition(artifactKey)
  const display = {
    language: 'zh' as const,
    sourceFile: artifactDefinition?.sourceFile,
    zhFile:
      artifactDefinition?.zhFile ??
      `${artifactKey}.zh${contentType === 'application/json' ? '.json' : '.md'}`,
    exists: false,
    zhExists: false,
    generated: false,
    fallbackToSource: true
  }

  if (artifact && 'payload' in artifact) {
    return {
      artifactKey: artifact.artifactKey,
      contentType,
      payload: artifact.payload,
      display
    }
  }

  return {
    artifactKey: artifact?.artifactKey ?? artifactKey,
    contentType,
    content: artifact && 'content' in artifact ? artifact.content : '',
    display
  }
}

type KGMaintenanceWorkspaceBundle = {
  summaryPayload: KBIterationSummaryResponse
  qualityPayload: KBIterationQualityResponse
  rulesPayload: KBIterationRulesResponse
  displayArtifacts: KGMaintenanceDisplayArtifacts
  kbContextArtifact: string
  kgSnapshotArtifact: unknown
  qualityScoreArtifact: unknown
  qualityScoreSourceArtifact: unknown
  approvalArtifact: string
  approvalArtifactSource: string
  backlogArtifact: string
  deferredChangesArtifact: string
  deferredChangesSourceArtifact: string
  acceptedApplyResultArtifact: string
  acceptedApplyResultSourceArtifact: string
  llmTraceArtifact: unknown
  llmTraceSourceArtifact: unknown
  llmReportArtifact: string
  llmProposalsArtifact: string
  llmProposalsSourceArtifact: string
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
  getDisplayArtifact: typeof getKBIterationDisplayArtifact
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
  getDisplayArtifact: getKBIterationDisplayArtifact,
  getTrace: getKBIterationLLMReviewTrace,
  getReport: getKBIterationLLMReviewReport,
  getProposals: getKBIterationLLMReviewProposals,
  getJudgeReport: getKBIterationLLMJudgeReport
}

export async function loadKGMaintenanceWorkspaceBundle(
  requestWorkspace: string,
  loaders: Partial<KGMaintenanceWorkspaceLoaders> = {}
): Promise<KGMaintenanceWorkspaceBundle> {
  const resolvedLoaders = { ...defaultWorkspaceLoaders, ...loaders }
  const [
    summaryPayload,
    qualityPayload,
    rulesPayload,
    kbContextResult,
    kgSnapshotResult,
    qualityScoreResult,
    approvalResult,
    backlogResult,
    deferredChangesResult,
    acceptedApplyResult,
    llmTraceResult,
    llmReportResult,
    llmProposalsResult,
    llmJudgeReportResult,
    llmIssueAnalysisResult,
    llmMissingBranchInferenceResult,
    llmEvidenceMapResult,
    llmRepairPlanResult
  ] = await Promise.all([
    resolvedLoaders.getSummary(requestWorkspace),
    resolvedLoaders.getQuality(requestWorkspace),
    resolvedLoaders.getRules(requestWorkspace),
    optionalDisplayArtifact(requestWorkspace, 'kb_context', resolvedLoaders),
    optionalDisplayArtifact(requestWorkspace, 'kg_snapshot', resolvedLoaders, {
      fallbackContentType: 'application/json'
    }),
    optionalDisplayArtifact(requestWorkspace, 'quality_score', resolvedLoaders, {
      fallbackContentType: 'application/json',
      loadSource: true
    }),
    optionalDisplayArtifact(requestWorkspace, 'approval_queue', resolvedLoaders, {
      loadSource: true
    }),
    optionalDisplayArtifact(requestWorkspace, 'improvement_backlog', resolvedLoaders),
    optionalDisplayArtifact(requestWorkspace, 'deferred_changes', resolvedLoaders, {
      loadSource: true
    }),
    optionalDisplayArtifact(requestWorkspace, 'accepted_changes_apply_result', resolvedLoaders, {
      loadSource: true
    }),
    optionalDisplayArtifact(requestWorkspace, 'llm_review_trace', resolvedLoaders, {
      fallbackContentType: 'application/json',
      fallbackLoader: () => resolvedLoaders.getTrace(requestWorkspace),
      loadSource: true
    }),
    optionalDisplayArtifact(requestWorkspace, 'llm_review_report', resolvedLoaders, {
      fallbackLoader: () => resolvedLoaders.getReport(requestWorkspace)
    }),
    optionalDisplayArtifact(requestWorkspace, 'proposals_generated', resolvedLoaders, {
      fallbackLoader: () => resolvedLoaders.getProposals(requestWorkspace),
      loadSource: true
    }),
    optionalDisplayArtifact(requestWorkspace, 'llm_judge_report', resolvedLoaders, {
      fallbackLoader: () => resolvedLoaders.getJudgeReport(requestWorkspace)
    }),
    optionalDisplayArtifact(requestWorkspace, 'llm_issue_analysis', resolvedLoaders),
    optionalDisplayArtifact(requestWorkspace, 'llm_missing_branch_inference', resolvedLoaders),
    optionalDisplayArtifact(requestWorkspace, 'llm_evidence_map', resolvedLoaders),
    optionalDisplayArtifact(requestWorkspace, 'llm_repair_plan', resolvedLoaders)
  ])
  const displayResults = [
    kbContextResult,
    kgSnapshotResult,
    qualityScoreResult,
    approvalResult,
    backlogResult,
    deferredChangesResult,
    acceptedApplyResult,
    llmTraceResult,
    llmReportResult,
    llmProposalsResult,
    llmJudgeReportResult,
    llmIssueAnalysisResult,
    llmMissingBranchInferenceResult,
    llmEvidenceMapResult,
    llmRepairPlanResult
  ]
  const displayArtifacts = Object.fromEntries(
    displayResults.map((result) => [result.key, result.displayArtifact])
  )

  return {
    summaryPayload,
    qualityPayload,
    rulesPayload,
    displayArtifacts,
    kbContextArtifact: artifactTextOrEmpty(kbContextResult.artifact),
    kgSnapshotArtifact: artifactContentOrPayload(kgSnapshotResult.artifact),
    qualityScoreArtifact: artifactContentOrPayload(qualityScoreResult.artifact),
    qualityScoreSourceArtifact: artifactContentOrPayload(qualityScoreResult.sourceArtifact),
    approvalArtifact: artifactTextOrEmpty(approvalResult.artifact),
    approvalArtifactSource: artifactTextOrEmpty(approvalResult.sourceArtifact),
    backlogArtifact: artifactTextOrEmpty(backlogResult.artifact),
    deferredChangesArtifact: artifactTextOrEmpty(deferredChangesResult.artifact),
    deferredChangesSourceArtifact: artifactTextOrEmpty(deferredChangesResult.sourceArtifact),
    acceptedApplyResultArtifact: artifactTextOrEmpty(acceptedApplyResult.artifact),
    acceptedApplyResultSourceArtifact: artifactTextOrEmpty(acceptedApplyResult.sourceArtifact),
    llmTraceArtifact: artifactContentOrPayload(llmTraceResult.artifact),
    llmTraceSourceArtifact: artifactContentOrPayload(llmTraceResult.sourceArtifact),
    llmReportArtifact: artifactTextOrEmpty(llmReportResult.artifact),
    llmProposalsArtifact: artifactTextOrEmpty(llmProposalsResult.artifact),
    llmProposalsSourceArtifact: artifactTextOrEmpty(llmProposalsResult.sourceArtifact),
    llmJudgeReportArtifact: artifactTextOrEmpty(llmJudgeReportResult.artifact),
    llmIssueAnalysisArtifact: artifactTextOrEmpty(llmIssueAnalysisResult.artifact),
    llmMissingBranchInferenceArtifact: artifactTextOrEmpty(
      llmMissingBranchInferenceResult.artifact
    ),
    llmEvidenceMapArtifact: artifactTextOrEmpty(llmEvidenceMapResult.artifact),
    llmRepairPlanArtifact: artifactTextOrEmpty(llmRepairPlanResult.artifact)
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

type ProposalRevisionActionArgs = {
  requestWorkspace: string
  getCurrentWorkspace: () => string | null
  proposal: ProposalSummary
  reloadWorkspaceData: () => Promise<void>
  requestRevision?: typeof requestKBIterationProposalRevision
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
  const auditReview = buildProposalDecisionReview(proposal, decision, review)
  await runWorkspaceAction({
    requestWorkspace,
    getCurrentWorkspace,
    action: () =>
      recordDecision(requestWorkspace, proposal.id, decision, {
        reviewer: 'maintainer',
        reason: auditReview.reason,
        impact_scope: auditReview.impactScope,
        verification: auditReview.verification
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

export async function requestProposalRevisionForWorkspace({
  requestWorkspace,
  getCurrentWorkspace,
  proposal,
  reloadWorkspaceData,
  requestRevision = requestKBIterationProposalRevision,
  onError
}: ProposalRevisionActionArgs): Promise<void> {
  await runWorkspaceAction({
    requestWorkspace,
    getCurrentWorkspace,
    action: () => requestRevision(requestWorkspace, proposal.id),
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
