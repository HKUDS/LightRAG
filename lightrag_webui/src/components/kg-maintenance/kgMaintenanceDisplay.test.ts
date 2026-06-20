import { describe, expect, test } from 'bun:test'
import { buildEvidenceIssueRows, extractQualityBefore } from './kgMaintenanceDisplay'

describe('kgMaintenanceDisplay', () => {
  test('builds evidence issue rows for missing node and relation evidence fields', () => {
    const rows = buildEvidenceIssueRows({
      nodes: [
        { id: 'node-1', label: '疾病A', source_id: 'src-1', file_path: 'a.md' },
        { id: 'node-2', label: '疾病B', source_id: '', file_path: null }
      ],
      edges: [
        { id: 'edge-1', source: 'node-1', target: 'node-2', source_id: 'src-2' },
        { source: 'node-2', target: 'node-3', file_path: 'b.md' }
      ]
    })

    expect(rows).toEqual([
      {
        id: 'node:node-2:source_id',
        itemType: '节点',
        itemId: 'node-2',
        issue: '缺少证据来源 ID'
      },
      {
        id: 'node:node-2:file_path',
        itemType: '节点',
        itemId: 'node-2',
        issue: '缺少来源文件'
      },
      {
        id: 'edge:edge-1:file_path',
        itemType: '关系',
        itemId: 'edge-1',
        issue: '缺少来源文件'
      },
      {
        id: 'edge:node-2->node-3:source_id',
        itemType: '关系',
        itemId: 'node-2->node-3',
        issue: '缺少证据来源 ID'
      }
    ])
  })

  test('supports common snapshot aliases for entities relations and links', () => {
    const rows = buildEvidenceIssueRows({
      entities: {
        disease_a: { label: '疾病A', source_id: null, file_path: 'disease.md' }
      },
      relations: [{ source: '疾病A', target: '症状B', source_id: 'src-3', file_path: '' }]
    })

    expect(rows).toEqual([
      {
        id: 'node:disease_a:source_id',
        itemType: '节点',
        itemId: 'disease_a',
        issue: '缺少证据来源 ID'
      },
      {
        id: 'edge:疾病A->症状B:file_path',
        itemType: '关系',
        itemId: '疾病A->症状B',
        issue: '缺少来源文件'
      }
    ])
  })

  test('uses a safe fallback for unusual item ids', () => {
    const circularId: Record<string, unknown> = { id: 'circular-edge' }
    circularId.self = circularId

    const rows = buildEvidenceIssueRows({
      edges: [{ id: circularId, source: '疾病A', target: '症状B', source_id: 'src-1', file_path: '' }]
    })

    expect(rows).toEqual([
      {
        id: 'edge:无法序列化:file_path',
        itemType: '关系',
        itemId: '无法序列化',
        issue: '缺少来源文件'
      }
    ])
  })

  test('extracts before quality metrics from apply result delta lines', () => {
    expect(
      extractQualityBefore(`Applied: 2
overall: 88 -> 97
hierarchy_missing_branch_count: 4 -> 0`)
    ).toEqual({
      overall: 88,
      metrics: {
        hierarchy_missing_branch_count: 4
      }
    })
  })

  test('extracts before quality metrics from real bullet-form apply result lines', () => {
    expect(
      extractQualityBefore(`- Applied: 2
- Blocked: 0
- hierarchy_missing_branch_count: 4 -> 0`)
    ).toEqual({
      metrics: {
        hierarchy_missing_branch_count: 4
      }
    })
  })
})
