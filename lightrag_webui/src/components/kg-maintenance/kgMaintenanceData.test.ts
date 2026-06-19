import { describe, expect, test } from 'bun:test'
import {
  MEDICAL_REVIEW_CONFIRMATION,
  canSubmitProposalDecision,
  findEdgeByIdAcrossSources,
  findNodeByIdAcrossSources,
  formatRunSubtitle,
  getEvidenceCoveragePercent,
  parseProposalDecisionStates,
  parseProposalSummaries,
  proposalNeedsConfirmation
} from './kgMaintenanceData'

describe('KG maintenance proposal decision safety', () => {
  test('does not require manual confirmation for high-risk proposal decisions', () => {
    const proposal = {
      id: 'p1',
      type: 'hierarchy_rule_change',
      target: 'kg_structure.md',
      risk: 'high',
      requiresApproval: true,
      evidence: ['edge:e1']
    }

    expect(proposalNeedsConfirmation(proposal)).toBe(true)
    expect(
      canSubmitProposalDecision(proposal, {
        reason: '',
        impactScope: '',
        verification: '',
        confirmation: ''
      })
    ).toBe(true)
    expect(
      canSubmitProposalDecision(proposal, {
        reason: 'Evidence checked',
        impactScope: 'Hierarchy rules',
        verification: 'Re-run quality report',
        confirmation: MEDICAL_REVIEW_CONFIRMATION
      })
    ).toBe(true)
  })

  test('allows proposal decisions without manual review fields', () => {
    const proposal = {
      id: 'p2',
      type: 'quality_report_note',
      target: 'quality_report.md',
      risk: 'low',
      requiresApproval: false,
      evidence: []
    }

    expect(proposalNeedsConfirmation(proposal)).toBe(false)
    expect(
      canSubmitProposalDecision(proposal, {
        reason: '',
        impactScope: 'Report only',
        verification: 'No mutation',
        confirmation: ''
      })
    ).toBe(true)
    expect(
      canSubmitProposalDecision(proposal, {
        reason: 'No mutation requested',
        impactScope: '',
        verification: 'No mutation',
        confirmation: ''
      })
    ).toBe(true)
    expect(
      canSubmitProposalDecision(proposal, {
        reason: 'No mutation requested',
        impactScope: 'Report only',
        verification: 'No mutation',
        confirmation: ''
      })
    ).toBe(true)
  })

  test('treats unknown proposal types as confirmation-gated by default', () => {
    const proposal = {
      id: 'p3',
      type: 'new_mutation_type',
      target: 'unknown',
      risk: 'medium',
      requiresApproval: false,
      evidence: []
    }

    expect(proposalNeedsConfirmation(proposal)).toBe(true)
  })

  test('parses human approval fields from proposal markdown', () => {
    const [proposal] = parseProposalSummaries(`
proposals:
- id: p1
  type: web_display_change
  target: MedicalHierarchyGraph.tsx
  proposed_change: Add relation legend
  reason: Reviewers need relation semantics
  evidence:
  - edge:e1
  confidence: 0.75
  risk: high
  requires_approval: true
  expected_metric_change:
    web_readability: 5
`)

    expect(proposal.proposedChange).toBe('Add relation legend')
    expect(proposal.reason).toBe('Reviewers need relation semantics')
    expect(proposal.confidence).toBe('0.75')
    expect(proposal.expectedMetricChange).toContain('web_readability')
  })

  test('parses recorded proposal decisions from decision memory artifacts', () => {
    expect(
      parseProposalDecisionStates({
        acceptedChanges: `# Accepted Changes

## proposal-1

\`\`\`json
{"proposal_id":"proposal-1","decision":"accept"}
\`\`\`
`,
        rejectedChanges: `# Rejected Changes

## proposal-2

\`\`\`json
{"proposal_id":"proposal-2","decision":"reject"}
\`\`\`
`
      })
    ).toEqual({
      'proposal-1': 'accept',
      'proposal-2': 'reject'
    })
  })
})

describe('KG maintenance overview display helpers', () => {
  test('falls back to evidence grounding when evidence coverage metric is absent', () => {
    expect(
      getEvidenceCoveragePercent({
        metrics: {},
        subscores: {
          evidence_grounding: 100
        }
      })
    ).toBe(100)
  })

  test('formats run subtitle with an ASCII separator', () => {
    expect(formatRunSubtitle('clinical_guideline_zh', 'pending_user_review')).toBe(
      'clinical_guideline_zh / Pending user review'
    )
  })
})

describe('KG maintenance evidence selection helpers', () => {
  test('falls back to catalog nodes when a selected entity is absent from the graph projection', () => {
    expect(
      findNodeByIdAcrossSources(
        'catalog-only',
        [{ id: 'graph-node', label: 'Graph node', entity_type: 'Disease', properties: {} }],
        [{ id: 'catalog-only', label: 'Catalog node', entity_type: 'Symptom', properties: {} }]
      )?.label
    ).toBe('Catalog node')
  })

  test('falls back to catalog relations when a selected edge is absent from the graph projection', () => {
    expect(
      findEdgeByIdAcrossSources(
        'catalog-edge',
        [{ id: 'graph-edge', source: 'a', target: 'b', label: 'graph relation' }],
        [{ id: 'catalog-edge', source: 'c', target: 'd', label: 'catalog relation' }]
      )?.label
    ).toBe('catalog relation')
  })
})
