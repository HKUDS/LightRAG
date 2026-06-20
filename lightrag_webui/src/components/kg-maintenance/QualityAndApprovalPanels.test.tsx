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

const approvalQueueWithRecordedRows = `# Approval Queue

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
  expected_metric_change: {}
- id: prop-b
  type: rule_change
  target: entity_name_normalization
  proposed_change: Normalize entity aliases
  reason: Improve matching
  evidence:
  - source_id: chunk-2
  confidence: 0.7
  risk: medium
  requires_approval: true
  expected_metric_change: {}
- id: prop-c
  type: rule_change
  target: hierarchy_mapping
  proposed_change: Delay hierarchy rule change
  reason: Needs more review
  evidence:
  - source_id: chunk-3
  confidence: 0.6
  risk: high
  requires_approval: true
  expected_metric_change: {}
- id: prop-d
  type: rule_change
  target: relation_cleanup
  proposed_change: Clean relation labels
  reason: Improve display
  evidence:
  - source_id: chunk-4
  confidence: 0.9
  risk: low
  requires_approval: true
  expected_metric_change: {}`

describe('ApprovalPanel', () => {
  test('parses proposals from source while rendering translated approval markdown', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue="# translated approval queue without proposal ids"
        approvalQueueSource={approvalQueue}
        improvementBacklog=""
        acceptedChanges=""
        rejectedChanges=""
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).toContain('prop-a')
    expect(markup).toContain('Normalize relation keywords')
    expect(markup).toContain('# translated approval queue without proposal ids')
    expect(markup).toContain('接受')
    expect(markup).toContain('拒绝')
  })

  test('does not parse proposals from display approval markdown when source is absent', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={approvalQueue}
        approvalQueueSource=""
        improvementBacklog=""
        acceptedChanges=""
        rejectedChanges=""
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).not.toContain('proposal-details-prop-a')
    expect(markup).not.toContain('Normalize relation keywords</span>')
    expect(markup).toContain('暂无待审批 proposal')
    expect(markup).toContain('id: prop-a')
  })

  test('parses deferred decisions from source while rendering display memories inertly', () => {
    const displayDeferredChanges = `# Deferred Changes

## prop-a

display-only deferred decision`

    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue="# translated approval queue without proposal ids"
        approvalQueueSource={approvalQueue}
        improvementBacklog=""
        acceptedChanges=""
        rejectedChanges=""
        deferredChanges={displayDeferredChanges}
        deferredChangesSource=""
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).toContain('prop-a')
    expect(markup).toContain('待审批')
    expect(markup).not.toContain('已延期')
    expect(markup).not.toContain('disabled=""')
  })

  test('renders compact rows with accept reject expand and no required textareas', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={approvalQueue}
        approvalQueueSource={approvalQueue}
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

  test('counts only unrecorded proposals as pending approval in the header', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={approvalQueueWithRecordedRows}
        approvalQueueSource={approvalQueueWithRecordedRows}
        improvementBacklog=""
        acceptedChanges="## prop-a"
        rejectedChanges="## prop-b"
        deferredChanges="## prop-c"
        deferredChangesSource="## prop-c"
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).toContain('1 个需要人工审批')
    expect(markup).not.toContain('4 个需要人工审批')
    expect(markup).toContain('已接受')
    expect(markup).toContain('已拒绝')
    expect(markup).toContain('已延后')
    expect(markup).toContain('待审批')
  })

  test('shows agent revision action for rejected proposal', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={approvalQueue}
        approvalQueueSource={approvalQueue}
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

  test('renders deferred records as already deferred without a default defer action', () => {
    const markup = renderToStaticMarkup(
      <ApprovalPanel
        approvalQueue={approvalQueue}
        approvalQueueSource={approvalQueue}
        improvementBacklog=""
        acceptedChanges=""
        rejectedChanges=""
        deferredChanges="## prop-a"
        deferredChangesSource="## prop-a"
        onDecision={() => undefined}
        onRequestRevision={() => undefined}
      />
    )

    expect(markup).toContain('已延后')
    expect(markup).not.toContain('bg-amber-100 text-amber-800')
    expect(markup).not.toContain('让 Agent 修改')
    expect(markup).not.toContain('>延后</button>')
    expect(markup).toContain('disabled=""')
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
