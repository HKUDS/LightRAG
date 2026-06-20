import { useCallback, useEffect, useMemo, useState } from 'react'
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
import { ArtifactDrawer, type DrawerArtifact } from '@/components/kg-maintenance/ArtifactDrawer'
import KGMaintenanceShell from '@/components/kg-maintenance/KGMaintenanceShell'
import {
  KG_MAINTENANCE_ARTIFACTS,
  findArtifactDefinition
} from '@/components/kg-maintenance/kgMaintenanceArtifacts'
import { IterationOverviewPanel } from '@/components/kg-maintenance/IterationWorkbenchPanels'
import {
  ExecutionPanel,
  ValidationPanel
} from '@/components/kg-maintenance/ExecutionAndValidationPanels'
import { extractQualityBefore } from '@/components/kg-maintenance/kgMaintenanceDisplay'
import {
  normalizeWorkspaceList,
  loadKGMaintenanceWorkspaceBundle,
  normalizeOptionalMarkdown,
  runWorkspaceAction,
  requestProposalRevisionForWorkspace,
  submitProposalDecisionForWorkspace,
  shouldApplyWorkspaceResponse,
  type KGMaintenanceDisplayArtifacts
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

export function ArtifactsDrawerPlaceholder({
  open,
  onOpenChange
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <ArtifactDrawer
      open={open}
      artifacts={[]}
      onClose={() => onOpenChange(false)}
      onOpenArtifact={() => undefined}
    />
  )
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
  const [deferredChanges, setDeferredChanges] = useState('')
  const [acceptedApplyResult, setAcceptedApplyResult] = useState('')
  const [llmTrace, setLlmTrace] = useState<Record<string, any> | null>(null)
  const [llmReport, setLlmReport] = useState('')
  const [llmProposals, setLlmProposals] = useState('')
  const [llmJudgeReport, setLlmJudgeReport] = useState('')
  const [llmIssueAnalysis, setLlmIssueAnalysis] = useState('')
  const [llmMissingBranchInference, setLlmMissingBranchInference] = useState('')
  const [llmEvidenceMap, setLlmEvidenceMap] = useState('')
  const [llmRepairPlan, setLlmRepairPlan] = useState('')
  const [displayArtifacts, setDisplayArtifacts] = useState<KGMaintenanceDisplayArtifacts>({})
  const [patchText, setPatchText] = useState('')
  const [acceptedExecuting, setAcceptedExecuting] = useState(false)
  const [llmRunning, setLlmRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [artifactDrawerOpen, setArtifactDrawerOpen] = useState(false)
  const activeProfile = summary?.profile || 'clinical_guideline_zh'
  const drawerArtifacts = useMemo<DrawerArtifact[]>(() => {
    const artifactExists = new Map(
      (summary?.artifacts || []).map((artifact) => [artifact.key, artifact.exists])
    )

    return KG_MAINTENANCE_ARTIFACTS.map((artifact) => ({
      key: artifact.key,
      title: artifact.title,
      sourceFile: artifact.sourceFile,
      zhFile: artifact.zhFile,
      step: artifact.step,
      status:
        artifactExists.get(artifact.key) || Boolean(displayArtifacts[artifact.key]?.display.exists)
          ? 'generated'
          : 'missing'
    }))
  }, [displayArtifacts, summary?.artifacts])

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
        displayArtifacts: loadedDisplayArtifacts,
        kbContextArtifact,
        kgSnapshotArtifact,
        qualityScoreArtifact,
        approvalArtifact,
        backlogArtifact,
        deferredChangesArtifact,
        acceptedApplyResultArtifact,
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
      setDisplayArtifacts(loadedDisplayArtifacts)
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
      setDeferredChanges(normalizeOptionalMarkdown(deferredChangesArtifact))
      setAcceptedApplyResult(normalizeOptionalMarkdown(acceptedApplyResultArtifact))
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
      setDisplayArtifacts({})
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

  const handleRequestProposalRevision = useCallback(
    async (proposal: ProposalSummary) => {
      if (!selectedWorkspace) return
      const requestWorkspace = selectedWorkspace
      const getCurrentWorkspace = () => useKGMaintenanceStore.getState().selectedWorkspace
      setError(null)
      await requestProposalRevisionForWorkspace({
        requestWorkspace,
        getCurrentWorkspace,
        proposal,
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

  const handleCloseArtifacts = useCallback(() => {
    setArtifactDrawerOpen(false)
  }, [])

  const handleOpenArtifact = useCallback(
    (artifactKey: string) => {
      const artifactDefinition = findArtifactDefinition(artifactKey)
      if (!artifactDefinition) return

      setActiveSection(artifactDefinition.step)
      setArtifactDrawerOpen(false)
    },
    [setActiveSection]
  )

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
          displayArtifacts={displayArtifacts}
          approvalQueue={approvalQueue}
          improvementBacklog={improvementBacklog}
          deferredChanges={deferredChanges}
          acceptedApplyResult={acceptedApplyResult}
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
          onRequestProposalRevision={handleRequestProposalRevision}
          onExecuteAcceptedChanges={handleExecuteAcceptedChanges}
          onRunLLMReview={handleRunLLMReview}
          onLoadPatch={handleLoadPatch}
        />
      </KGMaintenanceShell>
      <ArtifactDrawer
        open={artifactDrawerOpen}
        artifacts={drawerArtifacts}
        onClose={handleCloseArtifacts}
        onOpenArtifact={handleOpenArtifact}
      />
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
  displayArtifacts?: KGMaintenanceDisplayArtifacts
  approvalQueue: string
  improvementBacklog: string
  deferredChanges: string
  acceptedApplyResult: string
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
  onRequestProposalRevision?: (proposal: ProposalSummary) => void | Promise<void>
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
  displayArtifacts = {},
  approvalQueue,
  improvementBacklog,
  deferredChanges,
  acceptedApplyResult,
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
  onRequestProposalRevision,
  onExecuteAcceptedChanges,
  onRunLLMReview,
  onLoadPatch
}: MainPanelProps) {
  void quality
  void kbContext
  void kgSnapshot

  if (activeSection === 'check') {
    return (
      <section className="space-y-3">
        <h2 className="sr-only">审阅包概览</h2>
        <div className="border-border/70 bg-muted/20 flex flex-wrap items-center justify-between gap-3 rounded-lg border p-3">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold">检查知识库</h3>
            <p className="text-muted-foreground mt-1 text-xs">
              生成或刷新审阅包产物，进入后续 LLM 审阅与人工审批流程。
            </p>
          </div>
          <Button type="button" size="sm" onClick={onRunReview} disabled={running || loading}>
            {running ? '检查中' : '运行检查'}
          </Button>
        </div>
        <IterationOverviewPanel
          summary={summary}
          loading={loading}
          displayArtifacts={displayArtifacts}
          onOpenSection={onOpenSection}
        />
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
        deferredChanges={deferredChanges}
        onDecision={onProposalDecision}
        onRequestRevision={onRequestProposalRevision}
      />
    )
  }
  if (activeSection === 'execute') {
    return (
      <ExecutionPanel
        acceptedChanges={rules?.acceptedChanges || ''}
        applyResult={acceptedApplyResult}
        executing={acceptedExecuting}
        onExecute={onExecuteAcceptedChanges}
      />
    )
  }
  if (activeSection === 'validate') {
    return (
      <ValidationPanel
        qualityBefore={extractQualityBefore(acceptedApplyResult)}
        qualityAfter={qualityScore}
        applyResult={acceptedApplyResult}
      />
    )
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
          displayArtifacts={displayArtifacts}
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
    <IterationOverviewPanel
      summary={summary}
      loading={loading}
      displayArtifacts={displayArtifacts}
      onOpenSection={onOpenSection}
    />
  )
}
