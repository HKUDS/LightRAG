import { useCallback, useEffect, useState } from 'react'
import {
  getKBIterationArtifact,
  getKBIterationLLMJudgeReport,
  getKBIterationLLMReviewPatch,
  getKBIterationLLMReviewProposals,
  getKBIterationLLMReviewReport,
  getKBIterationLLMReviewTrace,
  getKBIterationQuality,
  getKBIterationRules,
  getKBIterationSummary,
  getKBIterationWorkspaces,
  recordKBIterationProposalDecision,
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
  BacklogPanel,
  DecisionMemoryPanel,
  IterationOverviewPanel,
  IterationReviewAside,
  IterationStagePanel,
  KBSummaryPanel,
  QualityScoreJsonPanel,
  SnapshotReviewPanel
} from '@/components/kg-maintenance/IterationWorkbenchPanels'
import {
  LLMJudgePanel,
  LLMReviewPanel,
  PatchCandidatesPanel
} from '@/components/kg-maintenance/LLMReviewPanels'
import { ApprovalPanel, QualityPanel } from '@/components/kg-maintenance/QualityAndApprovalPanels'
import { errorMessage } from '@/lib/utils'
import { useKGMaintenanceStore, type KGMaintenanceSection } from '@/stores/kgMaintenance'
import { useSettingsStore } from '@/stores/settings'

const PREFERRED_WORKSPACE = 'influenza_medical_v1'

const markdownContent = (artifact: Awaited<ReturnType<typeof getKBIterationArtifact>>) =>
  'content' in artifact ? artifact.content : ''

const artifactPayload = (artifact: Awaited<ReturnType<typeof getKBIterationArtifact>>) =>
  'payload' in artifact ? artifact.payload : null

const optionalArtifactContent = async (
  loader: () => Promise<Awaited<ReturnType<typeof getKBIterationArtifact>>>
) => {
  try {
    const artifact = await loader()
    if ('content' in artifact) return artifact.content
    if ('payload' in artifact) return artifact.payload
    return ''
  } catch {
    return ''
  }
}

const optionalArtifactPayload = async (
  loader: () => Promise<Awaited<ReturnType<typeof getKBIterationArtifact>>>
) => {
  try {
    const artifact = await loader()
    return artifactPayload(artifact)
  } catch {
    return null
  }
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
  const [iterationLog, setIterationLog] = useState('')
  const [llmTrace, setLlmTrace] = useState<Record<string, any> | null>(null)
  const [llmReport, setLlmReport] = useState('')
  const [llmProposals, setLlmProposals] = useState('')
  const [llmJudgeReport, setLlmJudgeReport] = useState('')
  const [patchText, setPatchText] = useState('')
  const [llmRunning, setLlmRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const activeProfile = summary?.profile || 'clinical_guideline_zh'

  const loadWorkspaces = useCallback(async () => {
    if (currentTab !== 'kg-maintenance') return
    setError(null)
    try {
      const data = await getKBIterationWorkspaces()
      setWorkspaces(data.workspaces)
      if (!selectedWorkspace) {
        const nextWorkspace = data.workspaces.includes(PREFERRED_WORKSPACE)
          ? PREFERRED_WORKSPACE
          : data.workspaces[0] || null
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
    try {
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
        llmJudgeReportArtifact
      ] = await Promise.all([
        getKBIterationSummary(selectedWorkspace),
        getKBIterationQuality(selectedWorkspace),
        getKBIterationRules(selectedWorkspace),
        getKBIterationArtifact(selectedWorkspace, 'kb_context'),
        optionalArtifactPayload(() => getKBIterationArtifact(selectedWorkspace, 'kg_snapshot')),
        optionalArtifactPayload(() => getKBIterationArtifact(selectedWorkspace, 'quality_score')),
        getKBIterationArtifact(selectedWorkspace, 'approval_queue'),
        getKBIterationArtifact(selectedWorkspace, 'improvement_backlog'),
        getKBIterationArtifact(selectedWorkspace, 'iteration_log'),
        optionalArtifactContent(() => getKBIterationLLMReviewTrace(selectedWorkspace)),
        optionalArtifactContent(() => getKBIterationLLMReviewReport(selectedWorkspace)),
        optionalArtifactContent(() => getKBIterationLLMReviewProposals(selectedWorkspace)),
        optionalArtifactContent(() => getKBIterationLLMJudgeReport(selectedWorkspace))
      ])
      setSummary(summaryPayload)
      setQuality(qualityPayload)
      setRules(rulesPayload)
      setKbContext(markdownContent(kbContextArtifact))
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
      setApprovalQueue(markdownContent(approvalArtifact))
      setImprovementBacklog(markdownContent(backlogArtifact))
      setIterationLog(markdownContent(logArtifact))
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
      setLatestRunId(summaryPayload.latestRunId)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setLoading(false)
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
    setPatchText('')
    setRunning(true)
    setError(null)
    try {
      const refreshedSummary = await runKBIteration(selectedWorkspace, {
        profile: activeProfile
      })
      setSummary(refreshedSummary)
      await loadWorkspaceData()
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setRunning(false)
    }
  }, [activeProfile, llmRunning, loadWorkspaceData, running, selectedWorkspace])

  const handleRunLLMReview = useCallback(async () => {
    if (!selectedWorkspace || running || llmRunning) return
    setLlmRunning(true)
    setError(null)
    setPatchText('')
    try {
      await runKBIterationLLMReview(selectedWorkspace, {
        profile: activeProfile,
        max_review_rounds: 4,
        max_focus_items_per_round: 3,
        allow_llm_judge: true,
        allow_llm_auto_accept: false,
        allow_low_risk_auto_reject: true,
        generate_patch_candidates: false,
        require_human_for_mutation: true
      })
      await loadWorkspaceData()
      setPatchText('')
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setLlmRunning(false)
    }
  }, [activeProfile, llmRunning, loadWorkspaceData, running, selectedWorkspace])

  const handleLoadPatch = useCallback(
    async (proposalId: string) => {
      if (!selectedWorkspace) return
      const requestWorkspace = selectedWorkspace
      setError(null)
      setPatchText('')
      try {
        const artifact = await getKBIterationLLMReviewPatch(requestWorkspace, proposalId)
        if (useKGMaintenanceStore.getState().selectedWorkspace !== requestWorkspace) return
        setPatchText('content' in artifact ? artifact.content : '')
      } catch (err) {
        setError(errorMessage(err))
      }
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
      setError(null)
      try {
        await recordKBIterationProposalDecision(selectedWorkspace, proposal.id, decision, {
          reviewer: 'maintainer',
          reason: review.reason,
          impact_scope: review.impactScope,
          verification: review.verification
        })
        await loadWorkspaceData()
      } catch (err) {
        setError(errorMessage(err))
      }
    },
    [loadWorkspaceData, selectedWorkspace]
  )

  return (
    <KGMaintenanceShell
      activeSection={activeSection}
      onSectionChange={setActiveSection}
      workspaces={workspaces}
      selectedWorkspace={selectedWorkspace}
      onWorkspaceChange={handleWorkspaceChange}
      onRefresh={loadWorkspaceData}
      onRunReview={handleRunReview}
      loading={loading}
      running={running || llmRunning}
      error={error}
      inspector={
        <IterationReviewAside
          phase={summary?.phase}
          pendingApprovalCount={summary?.pendingApprovalCount}
          highRiskFindingCount={summary?.highRiskFindingCount}
        />
      }
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
        iterationLog={iterationLog}
        llmTrace={llmTrace}
        llmReport={llmReport}
        llmProposals={llmProposals}
        llmJudgeReport={llmJudgeReport}
        patchText={patchText}
        llmRunning={llmRunning}
        running={running}
        loading={loading}
        onOpenSection={setActiveSection}
        onProposalDecision={handleProposalDecision}
        onRunLLMReview={handleRunLLMReview}
        onLoadPatch={handleLoadPatch}
      />
    </KGMaintenanceShell>
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
  iterationLog: string
  llmTrace: Record<string, any> | null
  llmReport: string
  llmProposals: string
  llmJudgeReport: string
  patchText: string
  llmRunning: boolean
  running: boolean
  loading: boolean
  onOpenSection: (section: KGMaintenanceSection) => void
  onProposalDecision: (
    proposal: ProposalSummary,
    decision: KBIterationProposalDecision,
    review: ProposalDecisionReview
  ) => void | Promise<void>
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
  iterationLog,
  llmTrace,
  llmReport,
  llmProposals,
  llmJudgeReport,
  patchText,
  llmRunning,
  running,
  loading,
  onOpenSection,
  onProposalDecision,
  onRunLLMReview,
  onLoadPatch
}: MainPanelProps) {
  if (activeSection === 'overview') {
    return (
      <section>
        <h2 className="sr-only">审阅包概览</h2>
        <IterationOverviewPanel summary={summary} loading={loading} onOpenSection={onOpenSection} />
      </section>
    )
  }
  if (activeSection === 'stage') {
    return <IterationStagePanel iterationLog={iterationLog} />
  }
  if (activeSection === 'kb-summary') {
    return <KBSummaryPanel kbContext={kbContext} />
  }
  if (activeSection === 'quality') {
    return (
      <section className="space-y-4">
        <QualityPanel quality={quality} />
        <QualityScoreJsonPanel qualityScore={qualityScore} />
      </section>
    )
  }
  if (activeSection === 'snapshot') {
    return <SnapshotReviewPanel snapshot={kgSnapshot} />
  }
  if (activeSection === 'approval') {
    return (
      <ApprovalPanel
        approvalQueue={approvalQueue}
        improvementBacklog={improvementBacklog}
        onDecision={onProposalDecision}
      />
    )
  }
  if (activeSection === 'backlog') {
    return <BacklogPanel improvementBacklog={improvementBacklog} />
  }
  if (activeSection === 'memory') {
    return (
      <DecisionMemoryPanel
        acceptedChanges={rules?.acceptedChanges || ''}
        rejectedChanges={rules?.rejectedChanges || ''}
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
  return <IterationOverviewPanel summary={summary} loading={loading} onOpenSection={onOpenSection} />
}
