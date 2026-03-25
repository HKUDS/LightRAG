import { describe, expect, test } from 'bun:test'

import {
  getVisibleGraphPropertyKeys,
  isEmptyGraphPropertyValue
} from './graphProperties'

describe('graphProperties', () => {
  test('识别空属性值', () => {
    expect(isEmptyGraphPropertyValue('')).toBe(true)
    expect(isEmptyGraphPropertyValue('   ')).toBe(true)
    expect(isEmptyGraphPropertyValue(null)).toBe(true)
    expect(isEmptyGraphPropertyValue(undefined)).toBe(true)
    expect(isEmptyGraphPropertyValue('value')).toBe(false)
  })

  test('节点属性过滤空值、内部字段和重复 name', () => {
    expect(
      getVisibleGraphPropertyKeys(
        {
          entity_id: 'node-1',
          name: 'Display Name',
          entity_type: 'concept',
          description: '',
          file_path: 'doc/a.md',
          truncate: 'FIFO 1/2',
          created_at: '123',
          keywords: 'k1'
        },
        'node'
      )
    ).toEqual(['entity_id', 'entity_type', 'file_path', 'keywords'])
  })

  test('边属性过滤空值和内部字段', () => {
    expect(
      getVisibleGraphPropertyKeys(
        {
          keywords: '',
          description: 'edge-desc',
          created_at: 123,
          truncate: 'KEEP 1/2',
          weight: 1
        },
        'edge'
      )
    ).toEqual(['description', 'weight'])
  })

  test('边属性在显示关系名称时隐藏重复 keywords', () => {
    expect(
      getVisibleGraphPropertyKeys(
        {
          keywords: 'rel-keywords',
          description: 'edge-desc',
          weight: 1
        },
        'edge',
        { hideKeywords: true }
      )
    ).toEqual(['description', 'weight'])
  })
})
