import { describe, expect, test } from 'bun:test'
import {
  groupMedicalRelations,
  type MedicalRelation,
  type MedicalRelationGroupMetadata
} from './medicalRelationGroups'

const relation = (
  id: string,
  entityType: string | undefined,
  keywords: string
): MedicalRelation => ({
  id,
  label: id,
  edgeKeywords: keywords,
  neighborEntityType: entityType,
  neighborLabels: entityType ? [entityType] : []
})

describe('groupMedicalRelations', () => {
  test('uses backend metadata node_ids and order before entity type fallback', () => {
    const metadata: { medical_groups: MedicalRelationGroupMetadata[] } = {
      medical_groups: [
        { key: 'treatment', label: 'Treatment', node_ids: ['drug-a', 'drug-b'], count: 2 },
        { key: 'pathogen', label: 'Pathogen', node_ids: ['virus-b'], count: 1 }
      ]
    }

    const groups = groupMedicalRelations(
      [
        relation('virus-b', 'Disease', 'treated by'),
        relation('drug-b', 'Disease', 'second treatment'),
        relation('drug-a', 'Disease', 'first treatment'),
        relation('symptom-c', 'Symptom', 'associated with')
      ],
      metadata
    )

    expect(groups.map((group) => group.key)).toEqual(['treatment', 'pathogen', 'symptom'])
    expect(groups[0].relations.map((item) => item.id)).toEqual(['drug-a', 'drug-b'])
    expect(groups[1].relations.map((item) => item.id)).toEqual(['virus-b'])
    expect(groups[2].relations.map((item) => item.id)).toEqual(['symptom-c'])
  })

  test('falls back by neighbor entity type and keeps disease or unknown as other', () => {
    const groups = groupMedicalRelations([
      relation('virus-a', 'Pathogen', 'treated by antiviral'),
      relation('drug-b', 'TreatmentRegimen', 'pathogen reduction'),
      relation('disease-c', 'Disease', 'pathogen treatment vaccine')
    ])

    expect(groups.map((group) => group.key)).toEqual(['pathogen', 'treatment', 'other'])
    expect(groups[0].relations.map((item) => item.id)).toEqual(['virus-a'])
    expect(groups[1].relations.map((item) => item.id)).toEqual(['drug-b'])
    expect(groups[2].relations.map((item) => item.id)).toEqual(['disease-c'])
  })

  test('uses relation keywords before neighbor node category', () => {
    const metadata: { medical_groups: MedicalRelationGroupMetadata[] } = {
      medical_groups: [{ key: 'other', label: 'Other', node_ids: ['流行性感冒'], count: 1 }]
    }

    const groups = groupMedicalRelations(
      [relation('流行性感冒', 'Disease', '推荐治疗')],
      metadata
    )

    expect(groups.map((group) => group.key)).toEqual(['treatment'])
    expect(groups[0].relations.map((item) => item.id)).toEqual(['流行性感冒'])
  })

  test('builds concise relation row text and full directed triple', () => {
    const groups = groupMedicalRelations([
      {
        id: '高热不退',
        label: '高热不退',
        edgeId: 'e1',
        selectedNodeId: '流行性感冒',
        sourceId: '流行性感冒',
        targetId: '高热不退',
        edgeKeywords: '临床表现',
        neighborEntityType: 'Symptom',
        neighborLabels: ['Symptom']
      }
    ])

    expect(groups[0].relations[0].displayName).toBe('临床表现')
    expect(groups[0].relations[0].displayValue).toBe('高热不退')
    expect(groups[0].relations[0].triple).toBe('流行性感冒 - 临床表现 -> 高热不退')
  })
})
