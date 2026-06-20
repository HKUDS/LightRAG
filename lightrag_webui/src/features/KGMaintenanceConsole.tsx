import { useCallback, useEffect, useMemo, useState } from 'react'
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
  artifactsForStep,
  findArtifactDefinition
} from '@/components/kg-maintenance/kgMaintenanceArtifacts'
import AgentStepHeader from '@/components/kg-maintenance/AgentStepHeader'
import {
  ArtifactFileSection,
  type DisplayArtifactItem
} from '@/components/kg-maintenance/ArtifactFileSection'
import type { KGMaintenanceNextAction } from '@/components/kg-maintenance/kgMaintenanceNextAction'
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
  isGeneratedDisplayArtifact,
  normalizeTraceArtifactForLogic,
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

type StepPresentation = {
  title: string
  description: string
  action: KGMaintenanceNextAction
}

const STEP_PRESENTATION: Record<KGMaintenanceSection, StepPresentation> = {
  check: {
    title: '检查知识库',
    description: '生成或刷新知识库检查包，确认当前图谱、质量与来源覆盖情况。',
    action: {
      id: 'run-check',
      label: '运行检查',
      section: 'check',
      reason: '先生成检查包，再进入 LLM 审阅与人工审批流程。'
    }
  },
  'llm-review': {
    title: 'LLM 审阅',
    description: '让 LLM Agent 基于检查包生成问题分析、证据定位与候选 proposal。',
    action: {
      id: 'run-llm-review',
      label: '运行 LLM 审阅',
      section: 'llm-review',
      reason: '只生成审阅材料和候选 proposal，不会自动修改 KG。'
    }
  },
  approval: {
    title: 'Proposal 审批',
    description: '逐条检查 proposal 的证据、风险与预期影响，保留人工审批门禁。',
    action: {
      id: 'open-approval',
      label: '查看待审批',
      section: 'approval',
      reason: '需要人工确认后，accepted changes 才能进入执行步骤。'
    }
  },
  execute: {
    title: '执行变更',
    description: '执行已接受的变更，并查看写入结果与执行记录。',
    action: {
      id: 'execute-accepted',
      label: '执行已接受变更',
      section: 'execute',
      reason: '只会执行已经人工接受的 proposal。'
    }
  },
  validate: {
    title: '验证结果',
    description: '对比执行前后的质量指标和写入结果，决定是否进入下一轮复核。',
    action: {
      id: 'start-next-iteration',
      label: '开始下一轮复核',
      section: 'llm-review',
      reason: '验证完成后可重新运行 LLM 审阅，继续迭代维护。'
    }
  }
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
  const [qualityScoreSource, setQualityScoreSource] = useState<Record<string, any> | null>(null)
  const [approvalQueue, setApprovalQueue] = useState('')
  const [approvalQueueSource, setApprovalQueueSource] = useState('')
  const [improvementBacklog, setImprovementBacklog] = useState('')
  const [deferredChanges, setDeferredChanges] = useState('')
  const [deferredChangesSource, setDeferredChangesSource] = useState('')
  const [acceptedApplyResult, setAcceptedApplyResult] = useState('')
  const [acceptedApplyResultSource, setAcceptedApplyResultSource] = useState('')
  const [llmTrace, setLlmTrace] = useState<Record<string, any> | null>(null)
  const [llmReport, setLlmReport] = useState('')
  const [llmProposals, setLlmProposals] = useState('')
  const [llmProposalsSource, setLlmProposalsSource] = useState('')
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
        artifactExists.get(artifact.key) ||
        isGeneratedDisplayArtifact(displayArtifacts[artifact.key])
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
        qualityScoreSourceArtifact,
        approvalArtifact,
        approvalArtifactSource,
        backlogArtifact,
        deferredChangesArtifact,
        deferredChangesSourceArtifact,
        acceptedApplyResultArtifact,
        acceptedApplyResultSourceArtifact,
        llmTraceArtifact,
        llmTraceSourceArtifact,
        llmReportArtifact,
        llmProposalsArtifact,
        llmProposalsSourceArtifact,
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
      setQualityScoreSource(
        qualityScoreSourceArtifact &&
          typeof qualityScoreSourceArtifact === 'object' &&
          !Array.isArray(qualityScoreSourceArtifact)
          ? qualityScoreSourceArtifact
          : null
      )
      setApprovalQueue(normalizeOptionalMarkdown(approvalArtifact))
      setApprovalQueueSource(normalizeOptionalMarkdown(approvalArtifactSource))
      setImprovementBacklog(normalizeOptionalMarkdown(backlogArtifact))
      setDeferredChanges(normalizeOptionalMarkdown(deferredChangesArtifact))
      setDeferredChangesSource(normalizeOptionalMarkdown(deferredChangesSourceArtifact))
      setAcceptedApplyResult(normalizeOptionalMarkdown(acceptedApplyResultArtifact))
      setAcceptedApplyResultSource(normalizeOptionalMarkdown(acceptedApplyResultSourceArtifact))
      setLlmTrace(normalizeTraceArtifactForLogic(llmTraceSourceArtifact, llmTraceArtifact))
      setLlmReport(typeof llmReportArtifact === 'string' ? llmReportArtifact : '')
      setLlmProposals(typeof llmProposalsArtifact === 'string' ? llmProposalsArtifact : '')
      setLlmProposalsSource(normalizeOptionalMarkdown(llmProposalsSourceArtifact))
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
      setQualityScoreSource(null)
      setApprovalQueueSource('')
      setDeferredChangesSource('')
      setAcceptedApplyResultSource('')
      setLlmProposalsSource('')
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
    (section: KGMaintenanceSection) => {
      setActiveSection(section)
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
          qualityScoreSource={qualityScoreSource}
          approvalQueue={approvalQueue}
          approvalQueueSource={approvalQueueSource}
          improvementBacklog={improvementBacklog}
          deferredChanges={deferredChanges}
          deferredChangesSource={deferredChangesSource}
          acceptedApplyResult={acceptedApplyResult}
          acceptedApplyResultSource={acceptedApplyResultSource}
          llmTrace={llmTrace}
          llmReport={llmReport}
          llmProposals={llmProposals}
          llmProposalsSource={llmProposalsSource}
          llmJudgeReport={llmJudgeReport}
          llmIssueAnalysis={llmIssueAnalysis}
          llmMissingBranchInference={llmMissingBranchInference}
          llmEvidenceMap={llmEvidenceMap}
          llmRepairPlan={llmRepairPlan}
          patchText={patchText}
          displayArtifacts={displayArtifacts}
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
  qualityScoreSource?: Record<string, any> | null
  approvalQueue: string
  approvalQueueSource: string
  improvementBacklog: string
  deferredChanges: string
  deferredChangesSource?: string
  acceptedApplyResult: string
  acceptedApplyResultSource?: string
  llmTrace: Record<string, any> | null
  llmReport: string
  llmProposals: string
  llmProposalsSource?: string
  llmJudgeReport: string
  llmIssueAnalysis: string
  llmMissingBranchInference: string
  llmEvidenceMap: string
  llmRepairPlan: string
  patchText: string
  displayArtifacts: KGMaintenanceDisplayArtifacts
  acceptedExecuting: boolean
  llmRunning: boolean
  running: boolean
  loading: boolean
  onOpenSection: (section: KGMaintenanceSection) => void
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
  qualityScoreSource,
  approvalQueue,
  approvalQueueSource,
  improvementBacklog,
  deferredChanges,
  deferredChangesSource,
  acceptedApplyResult,
  acceptedApplyResultSource,
  llmTrace,
  llmReport,
  llmProposals,
  llmProposalsSource,
  llmJudgeReport,
  llmIssueAnalysis,
  llmMissingBranchInference,
  llmEvidenceMap,
  llmRepairPlan,
  patchText,
  displayArtifacts,
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

  const sourceArtifacts = useMemo(
    () => ({
      kb_context: kbContext,
      kg_snapshot: stringifyArtifactContent(kgSnapshot),
      quality_score: stringifyArtifactContent(qualityScoreSource),
      approval_queue: approvalQueueSource,
      improvement_backlog: improvementBacklog,
      deferred_changes: deferredChangesSource,
      accepted_changes: rules?.acceptedChanges || '',
      rejected_changes: rules?.rejectedChanges || '',
      accepted_changes_apply_result: acceptedApplyResultSource,
      llm_review_trace: stringifyArtifactContent(llmTrace),
      llm_review_report: llmReport,
      proposals_generated: llmProposalsSource,
      llm_judge_report: llmJudgeReport,
      llm_issue_analysis: llmIssueAnalysis,
      llm_missing_branch_inference: llmMissingBranchInference,
      llm_evidence_map: llmEvidenceMap,
      llm_repair_plan: llmRepairPlan,
      quality_rules: rules?.qualityRules || '',
      known_issues: rules?.knownIssues || ''
    }),
    [
      acceptedApplyResultSource,
      approvalQueueSource,
      deferredChangesSource,
      improvementBacklog,
      kbContext,
      kgSnapshot,
      llmEvidenceMap,
      llmIssueAnalysis,
      llmJudgeReport,
      llmMissingBranchInference,
      llmProposalsSource,
      llmRepairPlan,
      llmReport,
      llmTrace,
      qualityScoreSource,
      rules?.acceptedChanges,
      rules?.knownIssues,
      rules?.qualityRules,
      rules?.rejectedChanges
    ]
  )
  const artifactExists = useMemo(
    () => new Map((summary?.artifacts || []).map((artifact) => [artifact.key, artifact.exists])),
    [summary?.artifacts]
  )
  const relatedArtifacts = useMemo(
    () =>
      buildDisplayArtifactItems({
        step: activeSection,
        displayArtifacts,
        sourceArtifacts,
        artifactExists
      }),
    [activeSection, artifactExists, displayArtifacts, sourceArtifacts]
  )
  const step = STEP_PRESENTATION[activeSection]

  const handleAction = (action: KGMaintenanceNextAction) => {
    if (action.id === 'run-check') {
      onRunReview()
      return
    }
    if (action.id === 'run-llm-review' || action.id === 'start-next-iteration') {
      onRunLLMReview()
      return
    }
    if (action.id === 'execute-accepted') {
      onExecuteAcceptedChanges()
      return
    }
    onOpenSection(action.section)
  }

  let content
  switch (activeSection) {
    case 'check':
      content = <CheckSummaryPanel summary={summary} loading={loading} />
      break
    case 'llm-review':
      content = (
        <div className="space-y-4">
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
            proposalIdSource={llmProposalsSource}
            patchText={patchText}
            onLoadPatch={onLoadPatch}
          />
          <LLMJudgePanel report={llmJudgeReport} />
        </div>
      )
      break
    case 'approval':
      content = (
        <ApprovalPanel
          approvalQueue={approvalQueue}
          approvalQueueSource={approvalQueueSource}
          improvementBacklog={improvementBacklog}
          acceptedChanges={rules?.acceptedChanges || ''}
          rejectedChanges={rules?.rejectedChanges || ''}
          deferredChanges={deferredChanges}
          deferredChangesSource={deferredChangesSource}
          onDecision={onProposalDecision}
          onRequestRevision={onRequestProposalRevision}
        />
      )
      break
    case 'execute':
      content = (
        <ExecutionPanel
          acceptedChanges={rules?.acceptedChanges || ''}
          applyResult={acceptedApplyResult}
          applyResultSource={acceptedApplyResultSource}
          executing={acceptedExecuting}
          onExecute={onExecuteAcceptedChanges}
        />
      )
      break
    case 'validate':
      content = (
        <ValidationPanel
          qualityBefore={extractQualityBefore(acceptedApplyResultSource || acceptedApplyResult)}
          qualityAfter={qualityScoreSource ?? null}
          applyResult={acceptedApplyResult}
          applyResultSource={acceptedApplyResultSource}
        />
      )
      break
  }

  return (
    <section className="space-y-4">
      <AgentStepHeader
        title={step.title}
        description={step.description}
        action={step.action}
        onAction={handleAction}
      />
      {content}
      <ArtifactFileSection title="相关产物" artifacts={relatedArtifacts} />
    </section>
  )
}

function CheckSummaryPanel({
  summary,
  loading
}: {
  summary: KBIterationSummaryResponse | null
  loading: boolean
}) {
  if (loading && !summary) {
    return (
      <section className="space-y-3" aria-label="检查摘要">
        <div className="bg-muted h-4 w-40 rounded" />
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="border-border/70 rounded-md border p-3">
              <div className="bg-muted h-3 w-20 rounded" />
              <div className="bg-muted mt-3 h-5 w-28 rounded" />
            </div>
          ))}
        </div>
      </section>
    )
  }

  if (!summary) {
    return (
      <section className="border-border/70 bg-muted/20 rounded-md border p-4" aria-label="检查摘要">
        <h2 className="text-sm font-semibold">暂无检查摘要</h2>
        <p className="text-muted-foreground mt-2 text-sm">
          请选择 workspace 并运行检查，生成当前知识库的质量与规模指标。
        </p>
      </section>
    )
  }

  const metrics = [
    ['当前阶段', summary.phase || '未开始'],
    ['质量分数', formatMaybeNumber(summary.quality?.overall)],
    ['待审批 proposal', String(summary.pendingApprovalCount ?? 0)],
    [
      '节点 / 关系 / 来源',
      `${summary.counts?.nodes ?? 0} / ${summary.counts?.edges ?? 0} / ${summary.counts?.sources ?? 0}`
    ]
  ]

  return (
    <section className="space-y-3" aria-label="检查摘要">
      <div>
        <h2 className="text-sm font-semibold">检查摘要</h2>
        <p className="text-muted-foreground mt-1 text-sm">
          {summary.workspace} / {summary.latestRunId}
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {metrics.map(([label, value]) => (
          <div key={label} className="border-border/70 rounded-md border p-3">
            <div className="text-muted-foreground text-xs">{label}</div>
            <div className="mt-1 text-lg font-semibold break-words">{value}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function buildDisplayArtifactItems({
  step,
  displayArtifacts,
  sourceArtifacts,
  artifactExists
}: {
  step: KGMaintenanceSection
  displayArtifacts: KGMaintenanceDisplayArtifacts
  sourceArtifacts: Record<string, string | undefined>
  artifactExists: Map<string, boolean>
}): DisplayArtifactItem[] {
  return artifactsForStep(step).map((artifact) => {
    const displayArtifact = displayArtifacts[artifact.key]
    const sourceContent = sourceArtifacts[artifact.key]
    return {
      key: artifact.key,
      title: artifact.title,
      sourceFile: artifact.sourceFile,
      zhFile: displayArtifact?.display?.zhFile ?? artifact.zhFile,
      contentType: displayArtifact?.contentType ?? artifact.contentType,
      displayStatus: displayArtifactStatus({
        displayArtifact,
        hasSource: Boolean(sourceContent || artifactExists.get(artifact.key))
      }),
      generatedAt: displayArtifact?.display?.generatedAt,
      model: displayArtifact?.display?.model,
      content: stringifyArtifactContent(artifactContent(displayArtifact)),
      originalContent: sourceContent
    }
  })
}

function displayArtifactStatus({
  displayArtifact,
  hasSource
}: {
  displayArtifact: KGMaintenanceDisplayArtifacts[string] | undefined
  hasSource: boolean
}): string {
  if (isGeneratedDisplayArtifact(displayArtifact)) return '中文已生成'
  if (displayArtifact?.display?.fallbackToSource || hasSource) return '原始文件'
  return '缺失'
}

function artifactContent(artifact: KGMaintenanceDisplayArtifacts[string] | undefined): unknown {
  if (!artifact) return ''
  if ('payload' in artifact) return artifact.payload
  return artifact.content
}

function stringifyArtifactContent(value: unknown): string | undefined {
  if (value === null || value === undefined || value === '') return undefined
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

function formatMaybeNumber(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : '—'
}
