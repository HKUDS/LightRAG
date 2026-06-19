import type { KBIterationGraphEdge, KBIterationGraphNode } from '@/api/lightrag'

export type ProposalSummary = {
  id: string
  type: string
  target: string
  proposedChange: string
  reason: string
  confidence: string
  risk: string
  requiresApproval: boolean
  evidence: string[]
  expectedMetricChange: string
}

export type ProposalDecisionReview = {
  reason: string
  impactScope: string
  verification: string
  confirmation: string
}

export const MEDICAL_REVIEW_CONFIRMATION =
  '该操作会改变知识库行为或重建结果。请确认已检查来源证据、影响范围和回滚方式。'

const APPROVAL_REQUIRED_TYPES = new Set([
  'prompt_edit',
  'ontology_rule_change',
  'hierarchy_rule_change',
  'relation_rule_change',
  'workspace_rebuild',
  'kg_fact_correction',
  'web_display_change'
])

const NO_APPROVAL_TYPES = new Set(['quality_report_note'])

export const countApprovalRequired = (proposals: { requiresApproval: boolean }[]) =>
  proposals.filter((proposal) => proposal.requiresApproval).length

export const highRiskFindingCount = (findings: { severity: string }[]) =>
  findings.filter((finding) => finding.severity === 'critical' || finding.severity === 'high')
    .length

export function getEvidenceCoveragePercent(quality?: {
  metrics?: Record<string, number | undefined>
  subscores?: Record<string, number | undefined>
}) {
  return Number(
    quality?.metrics?.evidence_coverage ?? quality?.subscores?.evidence_grounding ?? 0
  )
}

export function formatRunSubtitle(profile: string | null | undefined, phase: string) {
  const phaseLabel = phase === 'pending_user_review' ? 'Pending user review' : phase
  return `${profile || 'No profile'} / ${phaseLabel}`
}

export function parseProposalSummaries(markdown: string): ProposalSummary[] {
  const lines = markdown.split(/\r?\n/)
  const proposals: ProposalSummary[] = []
  let current: ProposalSummary | null = null
  let readingEvidence = false

  for (const line of lines) {
    const trimmed = line.trim()
    const itemMatch = trimmed.match(/^-\s+id:\s*(.+)$/)
    if (itemMatch) {
      if (current) proposals.push(current)
      current = {
        id: itemMatch[1].trim(),
        type: '',
        target: '',
        proposedChange: '',
        reason: '',
        confidence: '',
        risk: '',
        requiresApproval: true,
        evidence: [],
        expectedMetricChange: ''
      }
      readingEvidence = false
      continue
    }

    if (!current) continue
    if (trimmed.startsWith('evidence:')) {
      readingEvidence = true
      continue
    }
    if (readingEvidence && trimmed.startsWith('- ')) {
      current.evidence.push(trimmed.slice(2).trim())
      continue
    }
    if (trimmed.startsWith('expected_metric_change:')) {
      current.expectedMetricChange = trimmed
      readingEvidence = false
      continue
    }
    if (current.expectedMetricChange && /^\w/.test(trimmed)) {
      current.expectedMetricChange = `${current.expectedMetricChange}\n${trimmed}`
    }
    if (/^\w/.test(trimmed)) {
      readingEvidence = false
    }

    const fieldMatch = trimmed.match(/^([a-zA-Z_]+):\s*(.*)$/)
    if (!fieldMatch) continue
    const [, key, value] = fieldMatch
    if (key === 'type') current.type = value
    if (key === 'target') current.target = value
    if (key === 'proposed_change') current.proposedChange = value
    if (key === 'reason') current.reason = value
    if (key === 'confidence') current.confidence = value
    if (key === 'risk') current.risk = value
    if (key === 'requires_approval') current.requiresApproval = value === 'true'
  }

  if (current) proposals.push(current)
  return proposals
}

export function findNodeById(nodes: KBIterationGraphNode[], id: string) {
  return nodes.find((node) => node.id === id) || null
}

export function findEdgeById(edges: KBIterationGraphEdge[], id: string) {
  return edges.find((edge) => edge.id === id) || null
}

export function findNodeByIdAcrossSources(
  id: string,
  ...sources: Array<KBIterationGraphNode[] | null | undefined>
) {
  for (const source of sources) {
    const node = findNodeById(source ?? [], id)
    if (node) return node
  }
  return null
}

export function findEdgeByIdAcrossSources(
  id: string,
  ...sources: Array<KBIterationGraphEdge[] | null | undefined>
) {
  for (const source of sources) {
    const edge = findEdgeById(source ?? [], id)
    if (edge) return edge
  }
  return null
}

type ProposalSafetyFields = Pick<ProposalSummary, 'risk' | 'requiresApproval'> &
  Partial<Pick<ProposalSummary, 'type'>>

export function proposalNeedsConfirmation(proposal: ProposalSafetyFields) {
  const type = String(proposal.type || '')
  const knownReportOnly = NO_APPROVAL_TYPES.has(type)
  const approvalRequired =
    proposal.requiresApproval || APPROVAL_REQUIRED_TYPES.has(type) || !knownReportOnly
  return approvalRequired && ['medium', 'high', 'critical'].includes(proposal.risk.toLowerCase())
}

export function canSubmitProposalDecision(
  proposal: ProposalSafetyFields,
  review: ProposalDecisionReview
) {
  void review
  return Boolean(proposal)
}

export function buildProposalDecisionReview(
  proposal: ProposalSummary,
  decision: string,
  review?: Partial<ProposalDecisionReview>
): ProposalDecisionReview {
  const evidence = proposal.evidence.length ? proposal.evidence.join('; ') : 'no explicit evidence'
  return {
    reason:
      review?.reason?.trim() ||
      `User selected ${decision} for proposal ${proposal.id}; proposal reason: ${proposal.reason || 'not provided'}.`,
    impactScope:
      review?.impactScope?.trim() ||
      `Scope is constrained to proposal type ${proposal.type || 'unknown'} and target ${proposal.target || 'unknown'}; risk ${proposal.risk || 'unknown'}.`,
    verification:
      review?.verification?.trim() ||
      `Use proposal evidence and judge constraints for review; evidence: ${evidence}.`,
    confirmation: review?.confirmation?.trim() || ''
  }
}
