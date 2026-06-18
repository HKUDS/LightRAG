import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  getKBIterationArtifact,
  getKBIterationDiff,
  getKBIterationEntityCatalog,
  getKBIterationGraph,
  getKBIterationLLMJudgeReport,
  getKBIterationLLMReviewPatch,
  getKBIterationLLMReviewProposals,
  getKBIterationLLMReviewReport,
  getKBIterationLLMReviewTrace,
  getKBIterationQuality,
  getKBIterationRelationCatalog,
  getKBIterationRules,
  getKBIterationSummary,
  getKBIterationWorkspaces,
  recordKBIterationProposalDecision,
  runKBIteration,
  runKBIterationLLMReview,
  type KBIterationDiffResponse,
  type KBIterationEntityCatalogResponse,
  type KBIterationGraphResponse,
  type KBIterationQualityResponse,
  type KBIterationRelationCatalogResponse,
  type KBIterationRulesResponse,
  type KBIterationSummaryResponse,
  type KBIterationProposalDecision
} from '@/api/lightrag'
import { EntityCatalogPanel, RelationCatalogPanel } from '@/components/kg-maintenance/CatalogPanels'
import EvidenceInspector from '@/components/kg-maintenance/EvidenceInspector'
import {
  findEdgeByIdAcrossSources,
  findNodeByIdAcrossSources
} from '@/components/kg-maintenance/kgMaintenanceData'
import type {
  ProposalDecisionReview,
  ProposalSummary
} from '@/components/kg-maintenance/kgMaintenanceData'
import KGMaintenanceOverview from '@/components/kg-maintenance/KGMaintenanceOverview'
import KGMaintenanceShell from '@/components/kg-maintenance/KGMaintenanceShell'
import {
  LLMJudgePanel,
  LLMReviewPanel,
  PatchCandidatesPanel
} from '@/components/kg-maintenance/LLMReviewPanels'
import MedicalHierarchyGraph from '@/components/kg-maintenance/MedicalHierarchyGraph'
import {
  ApprovalPanel,
  DiffPanel,
  QualityPanel,
  RuleMemoryPanel,
  RunLogPanel
} from '@/components/kg-maintenance/QualityAndApprovalPanels'
import { errorMessage } from '@/lib/utils'
import { useKGMaintenanceStore, type KGMaintenanceSection } from '@/stores/kgMaintenance'
import { useSettingsStore } from '@/stores/settings'

const PREFERRED_WORKSPACE = 'influenza_medical_v1'

const markdownContent = (artifact: Awaited<ReturnType<typeof getKBIterationArtifact>>) =>
  'content' in artifact ? artifact.content : ''

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

export default function KGMaintenanceConsole() {
  const currentTab = useSettingsStore.use.currentTab()
  const activeSection = useKGMaintenanceStore.use.activeSection()
  const selectedItem = useKGMaintenanceStore.use.selectedItem()
  const selectedWorkspace = useKGMaintenanceStore.use.selectedWorkspace()
  const setActiveSection = useKGMaintenanceStore.use.setActiveSection()
  const setSelectedItem = useKGMaintenanceStore.use.setSelectedItem()
  const setSelectedWorkspace = useKGMaintenanceStore.use.setSelectedWorkspace()
  const setLatestRunId = useKGMaintenanceStore.use.setLatestRunId()

  const [workspaces, setWorkspaces] = useState<string[]>([])
  const [summary, setSummary] = useState<KBIterationSummaryResponse | null>(null)
  const [graph, setGraph] = useState<KBIterationGraphResponse | null>(null)
  const [quality, setQuality] = useState<KBIterationQualityResponse | null>(null)
  const [entities, setEntities] = useState<KBIterationEntityCatalogResponse | null>(null)
  const [relations, setRelations] = useState<KBIterationRelationCatalogResponse | null>(null)
  const [diff, setDiff] = useState<KBIterationDiffResponse | null>(null)
  const [rules, setRules] = useState<KBIterationRulesResponse | null>(null)
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
        graphPayload,
        qualityPayload,
        entityPayload,
        relationPayload,
        diffPayload,
        rulesPayload,
        approvalArtifact,
        backlogArtifact,
        logArtifact,
        llmTraceArtifact,
        llmReportArtifact,
        llmProposalsArtifact,
        llmJudgeReportArtifact
      ] = await Promise.all([
        getKBIterationSummary(selectedWorkspace),
        getKBIterationGraph(selectedWorkspace),
        getKBIterationQuality(selectedWorkspace),
        getKBIterationEntityCatalog(selectedWorkspace),
        getKBIterationRelationCatalog(selectedWorkspace),
        getKBIterationDiff(selectedWorkspace),
        getKBIterationRules(selectedWorkspace),
        getKBIterationArtifact(selectedWorkspace, 'approval_queue'),
        getKBIterationArtifact(selectedWorkspace, 'improvement_backlog'),
        getKBIterationArtifact(selectedWorkspace, 'iteration_log'),
        optionalArtifactContent(() => getKBIterationLLMReviewTrace(selectedWorkspace)),
        optionalArtifactContent(() => getKBIterationLLMReviewReport(selectedWorkspace)),
        optionalArtifactContent(() => getKBIterationLLMReviewProposals(selectedWorkspace)),
        optionalArtifactContent(() => getKBIterationLLMJudgeReport(selectedWorkspace))
      ])
      setSummary(summaryPayload)
      setGraph(graphPayload)
      setQuality(qualityPayload)
      setEntities(entityPayload)
      setRelations(relationPayload)
      setDiff(diffPayload)
      setRules(rulesPayload)
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
      setSelectedItem(null)
    },
    [setSelectedItem, setSelectedWorkspace]
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

  const selectedNode = useMemo(
    () =>
      selectedItem?.kind === 'node'
        ? findNodeByIdAcrossSources(selectedItem.id, graph?.nodes, entities?.entities)
        : null,
    [entities, graph, selectedItem]
  )
  const selectedEdge = useMemo(
    () =>
      selectedItem?.kind === 'edge'
        ? findEdgeByIdAcrossSources(selectedItem.id, graph?.edges, relations?.relations)
        : null,
    [graph, relations, selectedItem]
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
      inspector={<EvidenceInspector node={selectedNode} edge={selectedEdge} />}
    >
      <MainPanel
        activeSection={activeSection}
        summary={summary}
        graph={graph}
        quality={quality}
        entities={entities}
        relations={relations}
        diff={diff}
        rules={rules}
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
        onSelectItem={setSelectedItem}
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
  graph: KBIterationGraphResponse | null
  quality: KBIterationQualityResponse | null
  entities: KBIterationEntityCatalogResponse | null
  relations: KBIterationRelationCatalogResponse | null
  diff: KBIterationDiffResponse | null
  rules: KBIterationRulesResponse | null
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
  onSelectItem: (item: { kind: 'node' | 'edge'; id: string } | null) => void
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
  graph,
  quality,
  entities,
  relations,
  diff,
  rules,
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
  onSelectItem,
  onProposalDecision,
  onRunLLMReview,
  onLoadPatch
}: MainPanelProps) {
  if (activeSection === 'overview') {
    return (
      <KGMaintenanceOverview summary={summary} loading={loading} onOpenSection={onOpenSection} />
    )
  }
  if (activeSection === 'graph' || activeSection === 'evidence') {
    return <MedicalHierarchyGraph graph={graph} onSelectItem={onSelectItem} />
  }
  if (activeSection === 'entities') {
    return <EntityCatalogPanel catalog={entities} onSelect={onSelectItem} />
  }
  if (activeSection === 'relations') {
    return <RelationCatalogPanel catalog={relations} onSelect={onSelectItem} />
  }
  if (activeSection === 'quality') {
    return <QualityPanel quality={quality} />
  }
  if (activeSection === 'approval') {
    return (
      <ApprovalPanel
        approvalQueue={approvalQueue}
        improvementBacklog={improvementBacklog}
        onOpenEvidence={(evidenceId) => {
          if (evidenceId.startsWith('edge:')) {
            onSelectItem({ kind: 'edge', id: evidenceId.slice(5) })
          } else if (evidenceId.startsWith('node:')) {
            onSelectItem({ kind: 'node', id: evidenceId.slice(5) })
          }
          onOpenSection('evidence')
        }}
        onDecision={onProposalDecision}
      />
    )
  }
  if (activeSection === 'runs') {
    return <RunLogPanel runsText={iterationLog} summary={summary} />
  }
  if (activeSection === 'diff') {
    return <DiffPanel diff={diff} />
  }
  if (activeSection === 'llm-review') {
    return (
      <LLMReviewPanel
        trace={llmTrace}
        report={llmReport}
        proposals={llmProposals}
        running={llmRunning || running}
        onRun={onRunLLMReview}
      />
    )
  }
  if (activeSection === 'patches') {
    return (
      <PatchCandidatesPanel
        proposals={llmProposals}
        patchText={patchText}
        onLoadPatch={onLoadPatch}
      />
    )
  }
  if (activeSection === 'judge') {
    return <LLMJudgePanel report={llmJudgeReport} />
  }
  if (activeSection === 'rules') {
    return <RuleMemoryPanel rules={rules} />
  }
  return <RuleMemoryPanel rules={rules} />
}
