import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  getKBIterationArtifact,
  getKBIterationDiff,
  getKBIterationEntityCatalog,
  getKBIterationGraph,
  getKBIterationQuality,
  getKBIterationRelationCatalog,
  getKBIterationRules,
  getKBIterationSummary,
  getKBIterationWorkspaces,
  recordKBIterationProposalDecision,
  runKBIteration,
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
import { findEdgeById, findNodeById } from '@/components/kg-maintenance/kgMaintenanceData'
import type {
  ProposalDecisionReview,
  ProposalSummary
} from '@/components/kg-maintenance/kgMaintenanceData'
import KGMaintenanceOverview from '@/components/kg-maintenance/KGMaintenanceOverview'
import KGMaintenanceShell from '@/components/kg-maintenance/KGMaintenanceShell'
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
        logArtifact
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
        getKBIterationArtifact(selectedWorkspace, 'iteration_log')
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
      setSelectedWorkspace(workspace || null)
      setSelectedItem(null)
    },
    [setSelectedItem, setSelectedWorkspace]
  )

  const handleRunReview = useCallback(async () => {
    if (!selectedWorkspace) return
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
  }, [activeProfile, loadWorkspaceData, selectedWorkspace])

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
      selectedItem?.kind === 'node' && graph
        ? findNodeById(graph.nodes, selectedItem.id)
        : null,
    [graph, selectedItem]
  )
  const selectedEdge = useMemo(
    () =>
      selectedItem?.kind === 'edge' && graph
        ? findEdgeById(graph.edges, selectedItem.id)
        : null,
    [graph, selectedItem]
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
      running={running}
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
        loading={loading}
        onOpenSection={setActiveSection}
        onSelectItem={setSelectedItem}
        onProposalDecision={handleProposalDecision}
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
  loading: boolean
  onOpenSection: (section: KGMaintenanceSection) => void
  onSelectItem: (item: { kind: 'node' | 'edge'; id: string } | null) => void
  onProposalDecision: (
    proposal: ProposalSummary,
    decision: KBIterationProposalDecision,
    review: ProposalDecisionReview
  ) => void | Promise<void>
}

function MainPanel({
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
  loading,
  onOpenSection,
  onSelectItem,
  onProposalDecision
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
  return <RuleMemoryPanel rules={rules} />
}
