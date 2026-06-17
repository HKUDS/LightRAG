import { describe, expect, test } from 'bun:test'
import { buildKGMaintenanceGraphView, formatRelationLabel } from './kgMaintenanceGraph'

describe('KG maintenance graph helpers', () => {
  test('sizes nodes by medical role instead of making all nodes equal', () => {
    const graph = buildKGMaintenanceGraphView({
      nodes: [
        { id: 'flu', label: '流行性感冒', entity_type: 'Disease', properties: {} },
        {
          id: 'symptom',
          label: '临床表现',
          entity_type: 'MedicalGroup',
          properties: { medical_group: 'clinical_manifestation' }
        },
        {
          id: 'fever',
          label: '高热不退',
          entity_type: 'Symptom',
          properties: { medical_group: 'clinical_manifestation' }
        }
      ],
      edges: []
    } as any)

    const sizes = new Map(graph.nodes.map((node) => [node.id, node.size]))
    expect(sizes.get('flu')).toBeGreaterThan(sizes.get('symptom')!)
    expect(sizes.get('symptom')).toBeGreaterThan(sizes.get('fever')!)
  })

  test('formats missing or generic relation keywords as unmarked relations', () => {
    expect(formatRelationLabel('')).toBe('未标注关系')
    expect(formatRelationLabel('邻接')).toBe('未标注关系')
    expect(formatRelationLabel('临床表现')).toBe('临床表现')
  })

  test('keeps semantic direction labels for relation details', () => {
    const graph = buildKGMaintenanceGraphView({
      nodes: [
        { id: 'flu', label: '流行性感冒', entity_type: 'Disease', properties: {} },
        { id: 'virus', label: '流感病毒', entity_type: 'Pathogen', properties: {} }
      ],
      edges: [
        {
          id: 'e1',
          source: 'virus',
          target: 'flu',
          keywords: '病原导致',
          sourceLabel: '流感病毒',
          targetLabel: '流行性感冒'
        }
      ]
    } as any)

    expect(graph.edges[0].detailLabel).toBe('outgoing 病原导致 -> 流行性感冒')
  })
})
