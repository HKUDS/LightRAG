import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { ApprovalPanel } from './QualityAndApprovalPanels'

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
    expect(markup).toContain('让 Agent 修改')
  })
})
