import { describe, expect, test } from 'bun:test'
import { mergeGraphMetadata } from './graph'

describe('mergeGraphMetadata', () => {
  test('merges medical groups from expanded graph metadata without duplicating nodes', () => {
    const merged = mergeGraphMetadata(
      {
        source: 'initial',
        medical_groups: [
          { key: 'pathogen', label: '病原体', node_ids: ['甲型流感病毒'], count: 1 },
          { key: 'other', label: '其他', node_ids: ['流行性感冒'], count: 1 }
        ]
      },
      {
        expanded: true,
        medical_groups: [
          { key: 'pathogen', label: '病原体', node_ids: ['甲型流感病毒', 'A(H1N1)pdm09'], count: 2 },
          { key: 'treatment', label: '治疗', node_ids: ['奥司他韦'], count: 1 }
        ]
      }
    )

    expect(merged).toEqual({
      source: 'initial',
      expanded: true,
      medical_groups: [
        { key: 'pathogen', label: '病原体', node_ids: ['甲型流感病毒', 'A(H1N1)pdm09'], count: 2 },
        { key: 'other', label: '其他', node_ids: ['流行性感冒'], count: 1 },
        { key: 'treatment', label: '治疗', node_ids: ['奥司他韦'], count: 1 }
      ]
    })
  })

  test('merges medical browse metadata from expanded graph metadata', () => {
    const merged = mergeGraphMetadata(
      {
        medical_browse: {
          root_id: 'flu',
          default_depth: 'medium',
          category_order: ['clinical_manifestation'],
          node_roles: {
            flu: 'root',
            symptoms: 'category'
          },
          collapsed_groups: [
            {
              id: 'collapse:resp',
              parent_id: 'respiratory',
              label: 'Respiratory symptoms (1): cough',
              child_ids: ['cough'],
              count: 1,
              examples: ['cough']
            }
          ],
          relation_details: {
            e0: {
              source: 'flu',
              target: 'symptoms',
              relation: 'has',
              display: 'has: symptoms',
              triple: 'flu - has -> symptoms'
            }
          }
        }
      },
      {
        medical_browse: {
          root_id: 'flu',
          default_depth: 'medium',
          category_order: ['clinical_manifestation', 'treatment'],
          node_roles: {
            respiratory: 'subgroup',
            oseltamivir: 'leaf'
          },
          collapsed_groups: [
            {
              id: 'collapse:resp',
              parent_id: 'respiratory',
              label: 'Respiratory symptoms (2): cough, sore throat',
              child_ids: ['cough', 'sore throat'],
              count: 2,
              examples: ['cough', 'sore throat']
            },
            {
              id: 'collapse:treatment',
              parent_id: 'treatment',
              label: 'Treatment (1): oseltamivir',
              child_ids: ['oseltamivir'],
              count: 1,
              examples: ['oseltamivir']
            }
          ],
          relation_details: {
            e1: {
              source: 'flu',
              target: 'cough',
              relation: 'clinical_manifestation',
              display: 'clinical_manifestation: cough',
              triple: 'flu - clinical_manifestation -> cough'
            }
          }
        }
      }
    )

    expect(merged?.medical_browse?.category_order).toEqual([
      'clinical_manifestation',
      'treatment'
    ])
    expect(merged?.medical_browse?.node_roles).toEqual({
      flu: 'root',
      symptoms: 'category',
      respiratory: 'subgroup',
      oseltamivir: 'leaf'
    })
    expect(merged?.medical_browse?.collapsed_groups).toEqual([
      {
        id: 'collapse:resp',
        parent_id: 'respiratory',
        label: 'Respiratory symptoms (2): cough, sore throat',
        child_ids: ['cough', 'sore throat'],
        count: 2,
        examples: ['cough', 'sore throat']
      },
      {
        id: 'collapse:treatment',
        parent_id: 'treatment',
        label: 'Treatment (1): oseltamivir',
        child_ids: ['oseltamivir'],
        count: 1,
        examples: ['oseltamivir']
      }
    ])
    expect(merged?.medical_browse?.relation_details).toEqual({
      e0: {
        source: 'flu',
        target: 'symptoms',
        relation: 'has',
        display: 'has: symptoms',
        triple: 'flu - has -> symptoms'
      },
      e1: {
        source: 'flu',
        target: 'cough',
        relation: 'clinical_manifestation',
        display: 'clinical_manifestation: cough',
        triple: 'flu - clinical_manifestation -> cough'
      }
    })
  })
})
