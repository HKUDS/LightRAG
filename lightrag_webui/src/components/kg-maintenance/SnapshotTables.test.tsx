import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { SnapshotTables } from './SnapshotTables'

describe('SnapshotTables', () => {
  test('renders searchable table tabs without graph markup', () => {
    const markup = renderToStaticMarkup(
      <SnapshotTables
        snapshot={{
          nodes: [
            {
              id: 'node-1',
              label: '高血压',
              entity_type: '疾病',
              source_id: '',
              file_path: 'source.md'
            }
          ],
          edges: [
            {
              source: 'node-1',
              target: 'node-2',
              keywords: '并发症',
              source_id: 'src-1',
              file_path: ''
            }
          ]
        }}
      />
    )

    expect(markup).toContain('节点')
    expect(markup).toContain('关系')
    expect(markup).toContain('证据问题')
    expect(markup).toContain('placeholder="搜索"')
    expect(markup).toContain('h-[520px]')
    expect(markup).toContain('高血压')
    expect(markup).toContain('详情')
    expect(markup).not.toContain('<svg')
    expect(markup).not.toContain('Medical knowledge graph hierarchy')
  })

  test('renders an empty row message when the active table has no rows', () => {
    const markup = renderToStaticMarkup(<SnapshotTables snapshot={null} />)

    expect(markup).toContain('暂无数据')
    expect(markup).toContain('placeholder="搜索"')
    expect(markup).not.toContain('<svg')
  })

  test('renders only the initial visible row window for large snapshots', () => {
    const nodes = Array.from({ length: 120 }, (_, index) => ({
      id: `node-${index + 1}`,
      label: `节点 ${index + 1}`,
      entity_type: '疾病',
      source_id: `source-${index + 1}`,
      file_path: `source-${index + 1}.md`
    }))

    const markup = renderToStaticMarkup(<SnapshotTables snapshot={{ nodes }} />)

    expect(markup).toContain('显示 100 / 120 行')
    expect(markup).toContain('显示更多')
    expect(markup).toContain('node-100')
    expect(markup).not.toContain('node-101')
    expect(markup).not.toContain('node-120')
  })

  test('renders fixed column extras in a per-row detail disclosure', () => {
    const markup = renderToStaticMarkup(
      <SnapshotTables
        snapshot={{
          nodes: [
            {
              id: 'node-detail',
              label: '糖尿病',
              entity_type: '疾病',
              source_id: 'src-detail',
              file_path: 'diabetes.md',
              description: '固定列之外的描述',
              metadata: { reviewed_by: '医生A' }
            }
          ]
        }}
      />
    )

    expect(markup).toContain('<summary')
    expect(markup).toContain('查看')
    expect(markup).toContain('固定列之外的描述')
    expect(markup).toContain('reviewed_by')
    expect(markup).toContain('医生A')
  })

  test('formats circular detail values without throwing', () => {
    const metadata: Record<string, unknown> = { label: '循环数据' }
    metadata.self = metadata

    const markup = renderToStaticMarkup(
      <SnapshotTables
        snapshot={{
          nodes: [
            {
              id: 'node-circular',
              label: '复杂节点',
              entity_type: '疾病',
              source_id: 'src-circular',
              file_path: 'circular.md',
              metadata
            }
          ]
        }}
      />
    )

    expect(markup).toContain('无法序列化')
    expect(markup).toContain('node-circular')
  })
})
