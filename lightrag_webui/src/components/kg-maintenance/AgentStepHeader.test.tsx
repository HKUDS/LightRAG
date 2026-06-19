import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import AgentStepHeader from './AgentStepHeader'
import type { KGMaintenanceNextAction } from './kgMaintenanceNextAction'

const action: KGMaintenanceNextAction = {
  id: 'open-approval',
  label: '查看待审批',
  section: 'approval',
  reason: '还有 2 条 proposal 等待人工审批。'
}

describe('AgentStepHeader', () => {
  test('renders title, description, action reason, action label, and badges', () => {
    const markup = renderToStaticMarkup(
      <AgentStepHeader
        title="审批队列"
        description="复核高风险 proposal，再决定是否进入执行阶段。"
        action={action}
        badges={['高风险 1', '待审批 2']}
        onAction={() => undefined}
      />
    )

    expect(markup).toContain('审批队列')
    expect(markup).toContain('复核高风险 proposal，再决定是否进入执行阶段。')
    expect(markup).toContain('还有 2 条 proposal 等待人工审批。')
    expect(markup).toContain('查看待审批')
    expect(markup).toContain('高风险 1')
    expect(markup).toContain('待审批 2')
    expect(markup).toContain('type="button"')
  })
})
