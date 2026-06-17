import { describe, expect, test } from 'bun:test'
import {
  applyMedicalBrowseProjection,
  getMedicalBrowseNodeSize,
  getMedicalBrowseNodeRole,
  getMedicalBrowsePosition
} from './medicalBrowseGraph'
import type { LightragGraphType, LightragMedicalBrowseMetadata } from '@/api/lightrag'

describe('applyMedicalBrowseProjection', () => {
  test('adds collapsed group display nodes and hides collapsed child nodes', () => {
    const graph: LightragGraphType = {
      nodes: [
        { id: 'flu', labels: ['Disease'], properties: { entity_type: 'Disease' } },
        {
          id: 'respiratory',
          labels: ['MedicalGroup'],
          properties: { entity_type: 'MedicalGroup' }
        },
        { id: 'cough', labels: ['Symptom'], properties: { entity_type: 'Symptom' } },
        { id: 'sore_throat', labels: ['Symptom'], properties: { entity_type: 'Symptom' } },
        { id: 'fever', labels: ['Symptom'], properties: { entity_type: 'Symptom' } }
      ],
      edges: [
        {
          id: 'e-cough',
          source: 'cough',
          target: 'respiratory',
          type: 'related',
          properties: { keywords: 'belongs_to' }
        },
        {
          id: 'e-fever',
          source: 'fever',
          target: 'flu',
          type: 'related',
          properties: { keywords: 'symptom' }
        }
      ],
      metadata: {
        medical_browse: {
          root_id: 'flu',
          default_depth: 'medium',
          node_roles: {
            flu: 'root',
            respiratory: 'subgroup',
            fever: 'leaf'
          },
          collapsed_groups: [
            {
              id: 'collapse:respiratory',
              parent_id: 'respiratory',
              label: 'Respiratory symptoms (2): cough, sore throat',
              child_ids: ['cough', 'sore_throat'],
              count: 2,
              examples: ['cough', 'sore_throat']
            }
          ],
          relation_details: {}
        }
      }
    }

    const projected = applyMedicalBrowseProjection(graph)

    expect(projected.metadata).toBe(graph.metadata)
    expect(projected.nodes.map((node) => node.id)).toEqual([
      'flu',
      'respiratory',
      'fever',
      'collapse:respiratory'
    ])
    expect(projected.edges.map((edge) => edge.id)).toEqual([
      'e-fever',
      'edge:respiratory:collapse:respiratory'
    ])
    expect(projected.nodes.find((node) => node.id === 'collapse:respiratory')).toEqual({
      id: 'collapse:respiratory',
      labels: ['Respiratory symptoms (2): cough, sore throat'],
      properties: {
        entity_id: 'Respiratory symptoms (2): cough, sore throat',
        entity_type: 'MedicalCollapsedGroup',
        medical_group_parent: 'respiratory',
        child_ids: ['cough', 'sore_throat'],
        count: 2,
        examples: ['cough', 'sore_throat']
      }
    })
    expect(projected.edges.find((edge) => edge.id === 'edge:respiratory:collapse:respiratory')).toEqual(
      {
        id: 'edge:respiratory:collapse:respiratory',
        source: 'respiratory',
        target: 'collapse:respiratory',
        type: 'medicalCollapsedGroup',
        properties: { keywords: '包含', weight: 0.1 }
      }
    )
  })

  test('returns the original graph when browse metadata is absent', () => {
    const graph: LightragGraphType = {
      nodes: [{ id: 'flu', labels: ['Disease'], properties: { entity_type: 'Disease' } }],
      edges: []
    }

    expect(applyMedicalBrowseProjection(graph)).toBe(graph)
  })
})

describe('medical browse role layout', () => {
  const browse: LightragMedicalBrowseMetadata = {
    root_id: 'flu',
    category_order: ['clinical_manifestation'],
    node_roles: {
      flu: 'root',
      clinical_manifestation: 'category',
      respiratory: 'subgroup'
    },
    collapsed_groups: [
      {
        id: 'collapse:respiratory',
        parent_id: 'respiratory',
        label: 'Respiratory symptoms (2): cough, sore throat',
        child_ids: ['cough', 'sore_throat'],
        count: 2,
        examples: ['cough', 'sore_throat']
      }
    ],
    relation_details: {}
  }

  test('resolves collapsed group role from collapsed group ids', () => {
    expect(getMedicalBrowseNodeRole('flu', browse)).toBe('root')
    expect(getMedicalBrowseNodeRole('clinical_manifestation', browse)).toBe('category')
    expect(getMedicalBrowseNodeRole('respiratory', browse)).toBe('subgroup')
    expect(getMedicalBrowseNodeRole('collapse:respiratory', browse)).toBe('collapsed_group')
    expect(getMedicalBrowseNodeRole('unknown', browse)).toBe('leaf')
  })

  test('returns deterministic role-based positions', () => {
    expect(getMedicalBrowsePosition('flu', browse, 3)).toEqual({ x: 0, y: 0 })

    const firstCategory = getMedicalBrowsePosition('clinical_manifestation', browse, 0)
    const secondCategory = getMedicalBrowsePosition('clinical_manifestation', browse, 99)
    const subgroup = getMedicalBrowsePosition('respiratory', browse, 0)
    const collapsed = getMedicalBrowsePosition('collapse:respiratory', browse, 0)

    expect(secondCategory).toEqual(firstCategory)
    expect(Math.hypot(firstCategory.x, firstCategory.y)).toBeCloseTo(2, 6)
    expect(Math.hypot(subgroup.x, subgroup.y)).toBeCloseTo(4, 6)
    expect(Math.hypot(collapsed.x, collapsed.y)).toBeCloseTo(5.5, 6)
  })

  test('scales medical browse node size by role, degree, and collapsed count', () => {
    const ordinaryLeaf = getMedicalBrowseNodeSize('plain_leaf', 'Symptom', browse, {
      degree: 1,
      maxDegree: 20
    })
    const importantLeaf = getMedicalBrowseNodeSize('important_leaf', 'Drug', browse, {
      degree: 18,
      maxDegree: 20
    })
    const collapsedGroup = getMedicalBrowseNodeSize(
      'collapse:respiratory',
      'MedicalCollapsedGroup',
      browse,
      {
        degree: 3,
        maxDegree: 20
      }
    )
    const root = getMedicalBrowseNodeSize('flu', 'Disease', browse, {
      degree: 20,
      maxDegree: 20
    })

    expect(ordinaryLeaf).toBeLessThan(importantLeaf)
    expect(collapsedGroup).toBeGreaterThan(ordinaryLeaf)
    expect(root).toBeGreaterThan(importantLeaf)
    expect(new Set([ordinaryLeaf, importantLeaf, collapsedGroup, root]).size).toBe(4)
  })

  test('keeps low-degree medical leaves small even in sparse local views', () => {
    const sparseComplication = getMedicalBrowseNodeSize(
      'acute_respiratory_distress_syndrome',
      'disease',
      browse,
      {
        degree: 2,
        maxDegree: 2
      }
    )
    const connectedTreatment = getMedicalBrowseNodeSize('important_treatment', 'Drug', browse, {
      degree: 18,
      maxDegree: 20
    })

    expect(sparseComplication).toBeLessThanOrEqual(9)
    expect(connectedTreatment).toBeGreaterThan(sparseComplication)
  })
})
