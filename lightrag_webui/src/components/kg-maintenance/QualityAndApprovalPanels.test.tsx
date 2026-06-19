import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { ApprovalPanel, requestProposalRevisionFromPanel } from './QualityAndApprovalPanels'
import type { ProposalSummary } from './kgMaintenanceData'

const approvalQueue = `# Approval Queue

proposals:
- id: prop-a
  type: rule_change
  target: relation_keyword_extraction
  proposed_change: Normalize relation keywords
  reason: Improve readability
  evidence:
  - source_id: chunk-1
  confidence: 0.8
  risk: medium
  requires_approval: true
  expected_metric_change: {}`

describe('ApprovalPanel', () => {
  test('renders compact rows with accept reject expand and no required textareas', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={approvalQueue}
        improvementBacklog=""
        acceptedChanges=""
        rejectedChanges=""
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).toContain('prop-a')
    expect(markup).toContain('待审批')
    expect(markup).toContain('Normalize relation keywords')
    expect(markup).toContain('接受')
    expect(markup).toContain('拒绝')
    expect(markup).toContain('展开')
    expect(markup).not.toContain('延后')
    expect(markup).not.toContain('审批理由')
    expect(markup).not.toContain('影响范围')
    expect(markup).not.toContain('验证 / 回滚说明')
  })

  test('shows agent revision action for rejected proposal', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={approvalQueue}
        improvementBacklog=""
        acceptedChanges=""
        rejectedChanges="## prop-a"
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).toContain('已拒绝')
    const revisionButton = markup.match(/<button[^>]*>让 Agent 修改<\/button>/)?.[0] || ''
    expect(revisionButton).toContain('让 Agent 修改')
    expect(revisionButton).not.toMatch(/\sdisabled(?:=|>|$)/)
  })

  test('revision helper calls the provided handler with the proposal', async () => {
    const proposal: ProposalSummary = {
      id: 'prop-a',
      type: 'rule_change',
      target: 'relation_keyword_extraction',
      proposedChange: 'Normalize relation keywords',
      reason: 'Improve readability',
      confidence: '0.8',
      risk: 'medium',
      requiresApproval: true,
      evidence: ['source_id: chunk-1'],
      expectedMetricChange: '{}'
    }
    const requested: string[] = []

    await Promise.resolve(
      requestProposalRevisionFromPanel(proposal, (nextProposal) => {
        requested.push(nextProposal.id)
      })
    )

    expect(requested).toEqual(['prop-a'])
  })
})
