import { describe, expect, test } from 'bun:test'
import { resolveKGMaintenanceNextAction } from './kgMaintenanceNextAction'

const summary = {
  workspace: 'influenza_medical_v1',
  pendingApprovalCount: 0,
  quality: { overall: 92 }
}

describe('resolveKGMaintenanceNextAction', () => {
  test('returns run-check when no summary exists', () => {
    const action = resolveKGMaintenanceNextAction({ summary: null })

    expect(action.id).toBe('run-check')
    expect(action.section).toBe('check')
    expect(action.label).toBe('运行检查')
    expect(action.reason).toBe('还没有维护摘要，先生成检查包。')
  })

  test('returns run-llm-review when quality findings exist without an llm trace', () => {
    const action = resolveKGMaintenanceNextAction({
      summary,
      qualityFindings: [{ severity: 'high' }],
      llmTrace: null
    })

    expect(action.id).toBe('run-llm-review')
    expect(action.section).toBe('llm-review')
    expect(action.label).toBe('运行 LLM 复核')
    expect(action.reason).toBe('质量检查仍有发现项，需要让 LLM 生成修复建议。')
  })

  test('returns open-approval when approvals are pending', () => {
    const action = resolveKGMaintenanceNextAction({
      summary: { ...summary, pendingApprovalCount: 2 },
      qualityFindings: [],
      llmTrace: { rounds: [] }
    })

    expect(action.id).toBe('open-approval')
    expect(action.section).toBe('approval')
    expect(action.label).toBe('查看待审批')
    expect(action.reason).toBe('还有 2 条 proposal 等待人工审批。')
  })

  test('returns execute-accepted when accepted decision sections exist without apply result', () => {
    const action = resolveKGMaintenanceNextAction({
      summary,
      qualityFindings: [],
      llmTrace: { rounds: [] },
      acceptedChanges: `# Accepted Changes

## proposal-1

已接受修复建议。
`,
      acceptedApplyResult: ''
    })

    expect(action.id).toBe('execute-accepted')
    expect(action.section).toBe('execute')
    expect(action.label).toBe('执行已接受变更')
    expect(action.reason).toBe('accepted_changes.md 中已有已接受决策，尚未生成执行结果。')
  })

  test('returns validate-result immediately after execution', () => {
    const action = resolveKGMaintenanceNextAction({
      summary,
      qualityFindings: [],
      llmTrace: { rounds: [] },
      acceptedChanges: '',
      acceptedApplyResult: '',
      justExecuted: true
    })

    expect(action.id).toBe('validate-result')
    expect(action.section).toBe('validate')
    expect(action.label).toBe('验证执行结果')
    expect(action.reason).toBe('刚执行过变更，需要检查结果并确认是否进入下一轮。')
  })

  test('returns start-next-iteration when quality is clean and nothing is pending', () => {
    const action = resolveKGMaintenanceNextAction({
      summary,
      qualityFindings: [],
      llmTrace: { rounds: [] },
      acceptedChanges: '# Accepted Changes\n\n没有新的已接受变更。',
      acceptedApplyResult: ''
    })

    expect(action.id).toBe('start-next-iteration')
    expect(action.section).toBe('llm-review')
    expect(action.label).toBe('开始下一轮复核')
    expect(action.reason).toBe('当前没有待处理事项，可以启动下一轮维护。')
  })
})
