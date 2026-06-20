import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { ArtifactFileSection, type DisplayArtifactItem } from './ArtifactFileSection'
import { resolveArtifactViewState } from './artifactViewState'

const artifacts: DisplayArtifactItem[] = [
  {
    key: 'kb_context',
    title: '当前 KB 摘要',
    sourceFile: 'kb_context.md',
    zhFile: 'kb_context.zh.md',
    contentType: 'text/markdown',
    displayStatus: '已生成',
    generatedAt: '2026-06-19 12:30',
    model: 'gpt-4.1',
    content: '# 当前 KB 摘要',
    originalContent: '# KB Context'
  }
]

describe('ArtifactFileSection', () => {
  test('renders Chinese and original filenames with regeneration controls', () => {
    const markup = renderToStaticMarkup(
      <ArtifactFileSection
        title="检查知识库产物"
        artifacts={artifacts}
        onRegenerate={() => undefined}
      />
    )

    expect(markup).toContain('<details')
    expect(markup).toContain('<summary')
    expect(markup).toContain('检查知识库产物')
    expect(markup).toContain('当前 KB 摘要')
    expect(markup).toContain('kb_context.zh.md')
    expect(markup).toContain('kb_context.md')
    expect(markup).toContain('中文显示')
    expect(markup).toContain('原始文件')
    expect(markup).toContain('重新生成')
    expect(markup).toContain('2026-06-19 12:30')
    expect(markup).toContain('gpt-4.1')
  })

  test('keeps source view empty when original content was not loaded', () => {
    const artifact: DisplayArtifactItem = {
      ...artifacts[0],
      content: '# 生成的中文展示内容',
      originalContent: undefined
    }

    const view = resolveArtifactViewState(artifact, 'source')

    expect(view.selectedFile).toBe('kb_context.md')
    expect(view.selectedContent).toBeUndefined()
    expect(view.emptyMessage).toBe('暂无原始文件内容。')
  })
})
