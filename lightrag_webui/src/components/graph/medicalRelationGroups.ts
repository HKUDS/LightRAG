import type { LightragGraphMetadata, LightragMedicalGroupMetadata } from '@/api/lightrag'

export const MEDICAL_RELATION_GROUP_ORDER = [
  'pathogen',
  'symptom',
  'complication',
  'diagnosis',
  'treatment',
  'prevention',
  'population',
  'guideline',
  'other'
] as const

export type MedicalRelationGroupKey = (typeof MEDICAL_RELATION_GROUP_ORDER)[number]

export type MedicalRelationGroupMetadata = LightragMedicalGroupMetadata

export type MedicalRelation = {
  id: string
  label: string
  edgeId?: string
  selectedNodeId?: string
  sourceId?: string
  targetId?: string
  edgeKeywords?: string
  neighborEntityType?: string
  neighborLabels?: string[]
  displayName?: string
  displayValue?: string
  triple?: string
}

export type GroupedMedicalRelations = {
  key: MedicalRelationGroupKey
  metadataLabel?: string
  relations: MedicalRelation[]
}

const ENTITY_TYPE_GROUPS: Record<string, MedicalRelationGroupKey> = {
  pathogen: 'pathogen',
  symptom: 'symptom',
  complication: 'complication',
  diagnostictest: 'diagnosis',
  diagnosticcriterion: 'diagnosis',
  drug: 'treatment',
  treatmentregimen: 'treatment',
  vaccine: 'prevention',
  publichealthmeasure: 'prevention',
  population: 'population',
  riskfactor: 'population',
  guideline: 'guideline',
  recommendation: 'guideline',
  disease: 'other'
}

const MEDICAL_RELATION_GROUP_KEYS = new Set<string>(MEDICAL_RELATION_GROUP_ORDER)

const normalizeGroupKey = (key: string | undefined): MedicalRelationGroupKey => {
  const normalizedKey = key?.trim().toLowerCase()
  return normalizedKey && MEDICAL_RELATION_GROUP_KEYS.has(normalizedKey)
    ? (normalizedKey as MedicalRelationGroupKey)
    : 'other'
}

const normalizeEntityType = (entityType: string | undefined): string =>
  entityType?.trim().toLowerCase().replace(/[\s_-]/g, '') ?? ''

const normalizeRelationKeyword = (keywords: string | undefined): string =>
  keywords?.trim() || '相关'

const buildRelationDisplay = (relation: MedicalRelation): MedicalRelation => {
  const displayName = normalizeRelationKeyword(relation.edgeKeywords)
  const source = relation.sourceId || relation.selectedNodeId || relation.id
  const target = relation.targetId || relation.id

  return {
    ...relation,
    displayName,
    displayValue: relation.label,
    triple: `${source} - ${displayName} -> ${target}`
  }
}

const RELATION_KEYWORD_GROUPS: Array<{
  key: MedicalRelationGroupKey
  patterns: string[]
}> = [
  {
    key: 'treatment',
    patterns: ['推荐治疗', '治疗', '抗病毒', '用药', '给药', '剂量', '疗程', '禁忌', '慎用']
  },
  {
    key: 'diagnosis',
    patterns: ['诊断', '检测', '检查', '检验', '判定', '依据', '排除', '鉴别']
  },
  {
    key: 'pathogen',
    patterns: ['病原', '病毒', '毒株', '亚型', '感染']
  },
  {
    key: 'symptom',
    patterns: ['症状', '体征', '临床表现', '表现为']
  },
  {
    key: 'complication',
    patterns: ['并发', '重症风险', '危重症', '导致死亡']
  },
  {
    key: 'prevention',
    patterns: ['预防', '疫苗', '接种', '隔离', '报告', '监测', '院感']
  },
  {
    key: 'population',
    patterns: ['适用于', '适用对象', '适用人群', '高危', '风险人群', '儿童', '孕妇', '老年']
  },
  {
    key: 'guideline',
    patterns: ['指南建议', '推荐意见', '强制要求']
  }
]

const getKeywordGroupKey = (relation: MedicalRelation): MedicalRelationGroupKey | undefined => {
  const keywords = relation.edgeKeywords?.trim()
  if (!keywords) {
    return undefined
  }

  for (const group of RELATION_KEYWORD_GROUPS) {
    if (group.patterns.some((pattern) => keywords.includes(pattern))) {
      return group.key
    }
  }

  return undefined
}

const getFallbackGroupKey = (relation: MedicalRelation): MedicalRelationGroupKey => {
  const entityType = normalizeEntityType(relation.neighborEntityType)
  if (ENTITY_TYPE_GROUPS[entityType]) {
    return ENTITY_TYPE_GROUPS[entityType]
  }

  for (const label of relation.neighborLabels ?? []) {
    const labelGroup = ENTITY_TYPE_GROUPS[normalizeEntityType(label)]
    if (labelGroup) {
      return labelGroup
    }
  }

  return 'other'
}

export const groupMedicalRelations = (
  relations: MedicalRelation[],
  metadata?: LightragGraphMetadata
): GroupedMedicalRelations[] => {
  const metadataGroups = metadata?.medical_groups ?? []
  const metadataGroupByNodeId = new Map<string, MedicalRelationGroupKey>()
  const metadataOrderByNodeId = new Map<string, number>()
  const metadataLabelByKey = new Map<MedicalRelationGroupKey, string>()
  const orderedKeys: MedicalRelationGroupKey[] = []

  for (const group of metadataGroups) {
    const key = normalizeGroupKey(group.key)
    if (!orderedKeys.includes(key)) {
      orderedKeys.push(key)
    }
    if (group.label && !metadataLabelByKey.has(key)) {
      metadataLabelByKey.set(key, group.label)
    }
    for (const [index, nodeId] of (group.node_ids ?? []).entries()) {
      metadataGroupByNodeId.set(nodeId, key)
      metadataOrderByNodeId.set(nodeId, index)
    }
  }

  for (const key of MEDICAL_RELATION_GROUP_ORDER) {
    if (!orderedKeys.includes(key)) {
      orderedKeys.push(key)
    }
  }

  const grouped = new Map<
    MedicalRelationGroupKey,
    Array<{ relation: MedicalRelation; metadataOrder?: number; inputOrder: number }>
  >()
  relations.forEach((relation, inputOrder) => {
    const displayRelation = buildRelationDisplay(relation)
    const key =
      getKeywordGroupKey(displayRelation) ??
      metadataGroupByNodeId.get(displayRelation.id) ??
      getFallbackGroupKey(displayRelation)
    const items = grouped.get(key) ?? []
    items.push({
      relation: displayRelation,
      metadataOrder: metadataOrderByNodeId.get(displayRelation.id),
      inputOrder
    })
    grouped.set(key, items)
  })

  return orderedKeys
    .map((key) => ({
      key,
      metadataLabel: metadataLabelByKey.get(key),
      relations: [...(grouped.get(key) ?? [])]
        .sort((left, right) => {
          if (left.metadataOrder !== undefined && right.metadataOrder !== undefined) {
            return left.metadataOrder - right.metadataOrder
          }
          if (left.metadataOrder !== undefined) {
            return -1
          }
          if (right.metadataOrder !== undefined) {
            return 1
          }
          return left.inputOrder - right.inputOrder
        })
        .map((item) => item.relation)
    }))
    .filter((group) => group.relations.length > 0)
}
