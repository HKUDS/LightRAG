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
    expect(markup).not.toContain('<svg')
    expect(markup).not.toContain('Medical knowledge graph hierarchy')
  })

  test('renders an empty row message when the active table has no rows', () => {
    const markup = renderToStaticMarkup(<SnapshotTables snapshot={null} />)

    expect(markup).toContain('暂无数据')
    expect(markup).toContain('placeholder="搜索"')
    expect(markup).not.toContain('<svg')
  })
})
