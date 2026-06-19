import { useCallback, useEffect, useState } from 'react'
import Button from '@/components/ui/Button'
import {
  executeKBIterationAcceptedChanges,
  getKBIterationLLMReviewPatch,
  getKBIterationWorkspaces,
  runKBIteration,
  runKBIterationLLMReview,
  type KBIterationQualityResponse,
  type KBIterationRulesResponse,
  type KBIterationSummaryResponse,
  type KBIterationProposalDecision
} from '@/api/lightrag'
import type {
  ProposalDecisionReview,
  ProposalSummary
} from '@/components/kg-maintenance/kgMaintenanceData'
import KGMaintenanceShell from '@/components/kg-maintenance/KGMaintenanceShell'
import {
  DecisionExecutionPanel,
  IterationOverviewPanel,
  IterationStagePanel
} from '@/components/kg-maintenance/IterationWorkbenchPanels'
import {
  normalizeWorkspaceList,
  loadKGMaintenanceWorkspaceBundle,
  normalizeOptionalMarkdown,
  runWorkspaceAction,
  submitProposalDecisionForWorkspace,
  shouldApplyWorkspaceResponse
} from '@/components/kg-maintenance/kgIterationLoadUtils'
import {
  LLMJudgePanel,
  LLMReviewPanel,
  PatchCandidatesPanel
} from '@/components/kg-maintenance/LLMReviewPanels'
import { ApprovalPanel } from '@/components/kg-maintenance/QualityAndApprovalPanels'
import { errorMessage } from '@/lib/utils'
import { useKGMaintenanceStore, type KGMaintenanceSection } from '@/stores/kgMaintenance'
import { useSettingsStore } from '@/stores/settings'

const PREFERRED_WORKSPACE = 'influenza_medical_v1'

type TransitionalSectionTarget =
  | KGMaintenanceSection
  | 'overview'
  | 'stage'
  | 'kb-summary'
  | 'quality'
  | 'snapshot'
  | 'decisions'
  | 'backlog'
  | 'memory'

function mapTransitionalSection(section: TransitionalSectionTarget): KGMaintenanceSection {
  if (section === 'llm-review' || section === 'approval') return section
  if (section === 'decisions' || section === 'backlog' || section === 'memory') return 'execute'
  if (section === 'stage') return 'validate'
  return 'check'
}

export default function KGMaintenanceConsole() {
  const currentTab = useSettingsStore.use.currentTab()
  const activeSection = useKGMaintenanceStore.use.activeSection()
  const selectedWorkspace = useKGMaintenanceStore.use.selectedWorkspace()
  const setActiveSection = useKGMaintenanceStore.use.setActiveSection()
  const setSelectedWorkspace = useKGMaintenanceStore.use.setSelectedWorkspace()
  const setLatestRunId = useKGMaintenanceStore.use.setLatestRunId()

  const [workspaces, setWorkspaces] = useState<string[]>([])
  const [summary, setSummary] = useState<KBIterationSummaryResponse | null>(null)
  const [quality, setQuality] = useState<KBIterationQualityResponse | null>(null)
  const [rules, setRules] = useState<KBIterationRulesResponse | null>(null)
  const [kbContext, setKbContext] = useState('')
  const [kgSnapshot, setKgSnapshot] = useState<Record<string, any> | null>(null)
  const [qualityScore, setQualityScore] = useState<Record<string, any> | null>(null)
  const [approvalQueue, setApprovalQueue] = useState('')
  const [improvementBacklog, setImprovementBacklog] = useState('')
  const [acceptedApplyResult, setAcceptedApplyResult] = useState('')
  const [acceptedExecution, setAcceptedExecution] = useState('')
  const [iterationLog, setIterationLog] = useState('')
  const [llmTrace, setLlmTrace] = useState<Record<string, any> | null>(null)
  const [llmReport, setLlmReport] = useState('')
  const [llmProposals, setLlmProposals] = useState('')
  const [llmJudgeReport, setLlmJudgeReport] = useState('')
  const [llmIssueAnalysis, setLlmIssueAnalysis] = useState('')
  const [llmMissingBranchInference, setLlmMissingBranchInference] = useState('')
  const [llmEvidenceMap, setLlmEvidenceMap] = useState('')
  const [llmRepairPlan, setLlmRepairPlan] = useState('')
  const [patchText, setPatchText] = useState('')
  const [acceptedExecuting, setAcceptedExecuting] = useState(false)
  const [llmRunning, setLlmRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [artifactDrawerOpen, setArtifactDrawerOpen] = useState(false)
  const activeProfile = summary?.profile || 'clinical_guideline_zh'

  const loadWorkspaces = useCallback(async () => {
    if (currentTab !== 'kg-maintenance') return
    setError(null)
    try {
      const data = await getKBIterationWorkspaces()
      const workspaceList = normalizeWorkspaceList(data?.workspaces)
      setWorkspaces(workspaceList)
      if (
        !Array.isArray(data?.workspaces) ||
        data.workspaces.some((workspace) => typeof workspace !== 'string')
      ) {
        setError('KG workspace list response was malformed. Showing no workspaces.')
      }
      if (!selectedWorkspace) {
        const nextWorkspace = workspaceList.includes(PREFERRED_WORKSPACE)
          ? PREFERRED_WORKSPACE
          : workspaceList[0] || null
        setSelectedWorkspace(nextWorkspace)
      }
    } catch (err) {
      setError(errorMessage(err))
    }
  }, [currentTab, selectedWorkspace, setSelectedWorkspace])

  const loadWorkspaceData = useCallback(async () => {
    if (currentTab !== 'kg-maintenance' || !selectedWorkspace) return
    setLoading(true)
    setError(null)
    setPatchText('')
    const requestWorkspace = selectedWorkspace
    const isCurrentWorkspace = () =>
      shouldApplyWorkspaceResponse(
        requestWorkspace,
        () => useKGMaintenanceStore.getState().selectedWorkspace
      )
    try {
      const {
        summaryPayload,
        qualityPayload,
        rulesPayload,
        kbContextArtifact,
        kgSnapshotArtifact,
        qualityScoreArtifact,
        approvalArtifact,
        backlogArtifact,
        acceptedApplyResultArtifact,
        acceptedExecutionArtifact,
        logArtifact,
        llmTraceArtifact,
        llmReportArtifact,
        llmProposalsArtifact,
        llmJudgeReportArtifact,
        llmIssueAnalysisArtifact,
        llmMissingBranchInferenceArtifact,
        llmEvidenceMapArtifact,
        llmRepairPlanArtifact
      } = await loadKGMaintenanceWorkspaceBundle(requestWorkspace)
      if (!isCurrentWorkspace()) return
      setSummary(summaryPayload)
      setQuality(qualityPayload)
      setRules(rulesPayload)
      setKbContext(normalizeOptionalMarkdown(kbContextArtifact))
      setKgSnapshot(
        kgSnapshotArtifact &&
          typeof kgSnapshotArtifact === 'object' &&
          !Array.isArray(kgSnapshotArtifact)
          ? kgSnapshotArtifact
          : null
      )
      setQualityScore(
        qualityScoreArtifact &&
          typeof qualityScoreArtifact === 'object' &&
          !Array.isArray(qualityScoreArtifact)
          ? qualityScoreArtifact
          : null
      )
      setApprovalQueue(normalizeOptionalMarkdown(approvalArtifact))
      setImprovementBacklog(normalizeOptionalMarkdown(backlogArtifact))
      setAcceptedApplyResult(normalizeOptionalMarkdown(acceptedApplyResultArtifact))
      setAcceptedExecution(normalizeOptionalMarkdown(acceptedExecutionArtifact))
      setIterationLog(normalizeOptionalMarkdown(logArtifact))
      setLlmTrace(
        typeof llmTraceArtifact === 'object' &&
          llmTraceArtifact !== null &&
          !Array.isArray(llmTraceArtifact)
          ? llmTraceArtifact
          : null
      )
      setLlmReport(typeof llmReportArtifact === 'string' ? llmReportArtifact : '')
      setLlmProposals(typeof llmProposalsArtifact === 'string' ? llmProposalsArtifact : '')
      setLlmJudgeReport(typeof llmJudgeReportArtifact === 'string' ? llmJudgeReportArtifact : '')
      setLlmIssueAnalysis(normalizeOptionalMarkdown(llmIssueAnalysisArtifact))
      setLlmMissingBranchInference(normalizeOptionalMarkdown(llmMissingBranchInferenceArtifact))
      setLlmEvidenceMap(normalizeOptionalMarkdown(llmEvidenceMapArtifact))
      setLlmRepairPlan(normalizeOptionalMarkdown(llmRepairPlanArtifact))
      if (summaryPayload) setLatestRunId(summaryPayload.latestRunId)
    } catch (err) {
      if (isCurrentWorkspace()) setError(errorMessage(err))
    } finally {
      if (isCurrentWorkspace()) setLoading(false)
    }
  }, [currentTab, selectedWorkspace, setLatestRunId])

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void loadWorkspaces()
    }, 0)
    return () => window.clearTimeout(timeoutId)
  }, [loadWorkspaces])

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void loadWorkspaceData()
    }, 0)
    return () => window.clearTimeout(timeoutId)
  }, [loadWorkspaceData])

  const handleWorkspaceChange = useCallback(
    (workspace: string) => {
      setPatchText('')
      setSelectedWorkspace(workspace || null)
    },
    [setSelectedWorkspace]
  )

  const handleRunReview = useCallback(async () => {
    if (!selectedWorkspace || running || llmRunning) return
    const requestWorkspace = selectedWorkspace
    const getCurrentWorkspace = () => useKGMaintenanceStore.getState().selectedWorkspace
    setPatchText('')
    setRunning(true)
    setError(null)
    await runWorkspaceAction({
      requestWorkspace,
      getCurrentWorkspace,
      action: () =>
        runKBIteration(requestWorkspace, {
          profile: activeProfile
        }),
      onSuccess: async (refreshedSummary) => {
        setSummary(refreshedSummary)
        await loadWorkspaceData()
      },
      onError: (err) => {
        setError(errorMessage(err))
      },
      onComplete: () => {
        setRunning(false)
      }
    })
  }, [activeProfile, llmRunning, loadWorkspaceData, running, selectedWorkspace])

  const handleRunLLMReview = useCallback(async () => {
    if (!selectedWorkspace || running || llmRunning) return
    const requestWorkspace = selectedWorkspace
    const getCurrentWorkspace = () => useKGMaintenanceStore.getState().selectedWorkspace
    setLlmRunning(true)
    setError(null)
    setPatchText('')
    await runWorkspaceAction({
      requestWorkspace,
      getCurrentWorkspace,
      action: () =>
        runKBIterationLLMReview(requestWorkspace, {
          profile: activeProfile,
          mode: 'agent_pipeline',
          max_stage_retries: 5,
          max_review_rounds: 4,
          max_focus_items_per_round: 3,
          allow_llm_judge: true,
          allow_llm_auto_accept: false,
          allow_low_risk_auto_reject: true,
          generate_patch_candidates: false,
          require_human_for_mutation: true
        }),
      onSuccess: async (_result, shouldApply) => {
        await loadWorkspaceData()
        if (shouldApply()) setPatchText('')
      },
      onError: (err) => {
        setError(errorMessage(err))
      },
      onComplete: () => {
        setLlmRunning(false)
      }
    })
  }, [activeProfile, llmRunning, loadWorkspaceData, running, selectedWorkspace])

  const handleLoadPatch = useCallback(
    async (proposalId: string) => {
      if (!selectedWorkspace) return
      const requestWorkspace = selectedWorkspace
      const getCurrentWorkspace = () => useKGMaintenanceStore.getState().selectedWorkspace
      setError(null)
      setPatchText('')
      await runWorkspaceAction({
        requestWorkspace,
        getCurrentWorkspace,
        action: () => getKBIterationLLMReviewPatch(requestWorkspace, proposalId),
        onSuccess: (artifact) => {
          setPatchText('content' in artifact ? artifact.content : '')
        },
        onError: (err) => {
          setError(errorMessage(err))
        }
      })
    },
    [selectedWorkspace]
  )

  const handleProposalDecision = useCallback(
    async (
      proposal: ProposalSummary,
      decision: KBIterationProposalDecision,
      review: ProposalDecisionReview
    ) => {
      if (!selectedWorkspace) return
      const requestWorkspace = selectedWorkspace
      const getCurrentWorkspace = () => useKGMaintenanceStore.getState().selectedWorkspace
      setError(null)
      await submitProposalDecisionForWorkspace({
        requestWorkspace,
        getCurrentWorkspace,
        proposal,
        decision,
        review,
        reloadWorkspaceData: loadWorkspaceData,
        onError: (err) => {
          setError(errorMessage(err))
        }
      })
    },
    [loadWorkspaceData, selectedWorkspace]
  )

  const handleExecuteAcceptedChanges = useCallback(async () => {
    if (!selectedWorkspace || running || llmRunning || acceptedExecuting) return
    const requestWorkspace = selectedWorkspace
    const getCurrentWorkspace = () => useKGMaintenanceStore.getState().selectedWorkspace
    setAcceptedExecuting(true)
    setError(null)
    setPatchText('')
    await runWorkspaceAction({
      requestWorkspace,
      getCurrentWorkspace,
      action: () => executeKBIterationAcceptedChanges(requestWorkspace),
      onSuccess: async () => {
        await loadWorkspaceData()
      },
      onError: (err) => {
        setError(errorMessage(err))
      },
      onComplete: () => {
        setAcceptedExecuting(false)
      }
    })
  }, [acceptedExecuting, llmRunning, loadWorkspaceData, running, selectedWorkspace])

  const handleOpenArtifacts = useCallback(() => {
    setArtifactDrawerOpen(true)
  }, [])

  const handleOpenSection = useCallback(
    (section: TransitionalSectionTarget) => {
      setActiveSection(mapTransitionalSection(section))
    },
    [setActiveSection]
  )

  return (
    <>
      <KGMaintenanceShell
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        workspaces={workspaces}
        selectedWorkspace={selectedWorkspace}
        onWorkspaceChange={handleWorkspaceChange}
        onRefresh={loadWorkspaceData}
        onOpenArtifacts={handleOpenArtifacts}
        loading={loading}
        running={running || llmRunning || acceptedExecuting}
        error={error}
      >
        <MainPanel
          activeSection={activeSection}
          summary={summary}
          quality={quality}
          rules={rules}
          kbContext={kbContext}
          kgSnapshot={kgSnapshot}
          qualityScore={qualityScore}
          approvalQueue={approvalQueue}
          improvementBacklog={improvementBacklog}
          acceptedApplyResult={acceptedApplyResult}
          acceptedExecution={acceptedExecution}
          iterationLog={iterationLog}
          llmTrace={llmTrace}
          llmReport={llmReport}
          llmProposals={llmProposals}
          llmJudgeReport={llmJudgeReport}
          llmIssueAnalysis={llmIssueAnalysis}
          llmMissingBranchInference={llmMissingBranchInference}
          llmEvidenceMap={llmEvidenceMap}
          llmRepairPlan={llmRepairPlan}
          patchText={patchText}
          acceptedExecuting={acceptedExecuting}
          llmRunning={llmRunning}
          running={running}
          loading={loading}
          onOpenSection={handleOpenSection}
          onRunReview={handleRunReview}
          onProposalDecision={handleProposalDecision}
          onExecuteAcceptedChanges={handleExecuteAcceptedChanges}
          onRunLLMReview={handleRunLLMReview}
          onLoadPatch={handleLoadPatch}
        />
      </KGMaintenanceShell>
      {artifactDrawerOpen && (
        <div
          role="dialog"
          aria-label="全部产物"
          className="bg-background border-border fixed inset-y-0 right-0 z-40 w-full max-w-md border-l p-4 shadow-lg"
        >
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-semibold">全部产物</h2>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setArtifactDrawerOpen(false)}
            >
              关闭
            </Button>
          </div>
          <p className="text-muted-foreground mt-3 text-sm">
            产物抽屉将在后续任务中接入。当前占位用于保留工作流入口。
          </p>
        </div>
      )}
    </>
  )
}

interface MainPanelProps {
  activeSection: KGMaintenanceSection
  summary: KBIterationSummaryResponse | null
  quality: KBIterationQualityResponse | null
  rules: KBIterationRulesResponse | null
  kbContext: string
  kgSnapshot: Record<string, any> | null
  qualityScore: Record<string, any> | null
  approvalQueue: string
  improvementBacklog: string
  acceptedApplyResult: string
  acceptedExecution: string
  iterationLog: string
  llmTrace: Record<string, any> | null
  llmReport: string
  llmProposals: string
  llmJudgeReport: string
  llmIssueAnalysis: string
  llmMissingBranchInference: string
  llmEvidenceMap: string
  llmRepairPlan: string
  patchText: string
  acceptedExecuting: boolean
  llmRunning: boolean
  running: boolean
  loading: boolean
  onOpenSection: (section: TransitionalSectionTarget) => void
  onRunReview: () => void
  onProposalDecision: (
    proposal: ProposalSummary,
    decision: KBIterationProposalDecision,
    review: ProposalDecisionReview
  ) => void | Promise<void>
  onExecuteAcceptedChanges: () => void
  onRunLLMReview: () => void
  onLoadPatch: (proposalId: string) => void
}

export function MainPanel({
  activeSection,
  summary,
  quality,
  rules,
  kbContext,
  kgSnapshot,
  qualityScore,
  approvalQueue,
  improvementBacklog,
  acceptedApplyResult,
  acceptedExecution,
  iterationLog,
  llmTrace,
  llmReport,
  llmProposals,
  llmJudgeReport,
  llmIssueAnalysis,
  llmMissingBranchInference,
  llmEvidenceMap,
  llmRepairPlan,
  patchText,
  acceptedExecuting,
  llmRunning,
  running,
  loading,
  onOpenSection,
  onRunReview,
  onProposalDecision,
  onExecuteAcceptedChanges,
  onRunLLMReview,
  onLoadPatch
}: MainPanelProps) {
  void quality
  void kbContext
  void kgSnapshot
  void qualityScore
  void onRunReview

  if (activeSection === 'check') {
    return (
      <section>
        <h2 className="sr-only">审阅包概览</h2>
        <IterationOverviewPanel summary={summary} loading={loading} onOpenSection={onOpenSection} />
      </section>
    )
  }
  if (activeSection === 'approval') {
    return (
      <ApprovalPanel
        approvalQueue={approvalQueue}
        improvementBacklog={improvementBacklog}
        acceptedChanges={rules?.acceptedChanges || ''}
        rejectedChanges={rules?.rejectedChanges || ''}
        onDecision={onProposalDecision}
      />
    )
  }
  if (activeSection === 'execute') {
    return (
      <DecisionExecutionPanel
        improvementBacklog={improvementBacklog}
        acceptedChanges={rules?.acceptedChanges || ''}
        rejectedChanges={rules?.rejectedChanges || ''}
        acceptedApplyResult={acceptedApplyResult}
        acceptedExecution={acceptedExecution}
        executing={acceptedExecuting}
        onExecuteAcceptedChanges={onExecuteAcceptedChanges}
      />
    )
  }
  if (activeSection === 'validate') {
    return <IterationStagePanel iterationLog={iterationLog} />
  }
  if (activeSection === 'llm-review') {
    return (
      <section className="space-y-4">
        <h2 className="sr-only">LLM 审阅材料</h2>
        <LLMReviewPanel
          trace={llmTrace}
          report={llmReport}
          proposals={llmProposals}
          issueAnalysis={llmIssueAnalysis}
          missingBranchInference={llmMissingBranchInference}
          evidenceMap={llmEvidenceMap}
          repairPlan={llmRepairPlan}
          running={llmRunning || running}
          onRun={onRunLLMReview}
        />
        <PatchCandidatesPanel
          proposals={llmProposals}
          patchText={patchText}
          onLoadPatch={onLoadPatch}
        />
        <LLMJudgePanel report={llmJudgeReport} />
      </section>
    )
  }
  return (
    <IterationOverviewPanel summary={summary} loading={loading} onOpenSection={onOpenSection} />
  )
}
