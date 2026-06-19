import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { ArtifactDrawer, type DrawerArtifact } from './ArtifactDrawer'

const artifacts: DrawerArtifact[] = [
  {
    key: 'kb_context',
    title: '当前 KB 摘要',
    sourceFile: 'kb_context.md',
    zhFile: 'kb_context.zh.md',
    step: 'check',
    status: '已生成'
  },
  {
    key: 'approval_queue',
    title: '待审批 Proposal',
    sourceFile: 'approval_queue.md',
    zhFile: 'approval_queue.zh.md',
    step: 'approval',
    status: '缺失'
  },
  {
    key: 'llm_issue_analysis',
    title: 'LLM 问题分析',
    sourceFile: 'llm_issue_analysis.md',
    zhFile: 'llm_issue_analysis.zh.md',
    step: 'llm-review',
    status: '已生成'
  }
]

describe('ArtifactDrawer', () => {
  test('groups artifacts by workflow step with Chinese labels and zh filenames', () => {
    const markup = renderToStaticMarkup(
      <ArtifactDrawer
        open={true}
        artifacts={artifacts}
        onClose={() => undefined}
        onOpenArtifact={() => undefined}
      />
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-label="全部产物"')
    expect(markup).toContain('全部产物')
    expect(markup).toContain('检查知识库')
    expect(markup).toContain('LLM 审阅')
    expect(markup).toContain('Proposal 审批')
    expect(markup).toContain('kb_context.zh.md')
    expect(markup).toContain('approval_queue.zh.md')
    expect(markup).toContain('llm_issue_analysis.zh.md')
  })

  test('renders nothing when closed', () => {
    const markup = renderToStaticMarkup(
      <ArtifactDrawer
        open={false}
        artifacts={artifacts}
        onClose={() => undefined}
        onOpenArtifact={() => undefined}
      />
    )

    expect(markup).toBe('')
  })
})
