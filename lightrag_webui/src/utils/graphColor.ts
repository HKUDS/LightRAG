const DEFAULT_NODE_COLOR = '#5D6D7E'

const TYPE_SYNONYMS: Record<string, string> = {
  unknown: 'unknown',
  未知: 'unknown',

  other: 'other',
  其它: 'other',

  concept: 'concept',
  object: 'concept',
  type: 'concept',
  category: 'concept',
  model: 'concept',
  project: 'concept',
  condition: 'concept',
  rule: 'concept',
  regulation: 'concept',
  article: 'concept',
  law: 'concept',
  legalclause: 'concept',
  policy: 'concept',
  disease: 'concept',
  概念: 'concept',
  对象: 'concept',
  类别: 'concept',
  分类: 'concept',
  模型: 'concept',
  项目: 'concept',
  条件: 'concept',
  规则: 'concept',
  法律: 'concept',
  法律条款: 'concept',
  条文: 'concept',
  政策: 'policy',
  疾病: 'concept',

  method: 'method',
  process: 'method',
  方法: 'method',
  过程: 'method',

  artifact: 'artifact',
  technology: 'artifact',
  tech: 'artifact',
  product: 'artifact',
  equipment: 'artifact',
  device: 'artifact',
  stuff: 'artifact',
  component: 'artifact',
  material: 'artifact',
  chemical: 'artifact',
  drug: 'artifact',
  medicine: 'artifact',
  food: 'artifact',
  weapon: 'artifact',
  arms: 'artifact',
  人工制品: 'artifact',
  人造物品: 'artifact',
  技术: 'technology',
  科技: 'technology',
  产品: 'artifact',
  设备: 'artifact',
  装备: 'artifact',
  物品: 'artifact',
  材料: 'artifact',
  化学: 'artifact',
  药物: 'artifact',
  食品: 'artifact',
  武器: 'artifact',
  军火: 'artifact',

  naturalobject: 'naturalobject',
  natural: 'naturalobject',
  phenomena: 'naturalobject',
  substance: 'naturalobject',
  plant: 'naturalobject',
  自然对象: 'naturalobject',
  自然物体: 'naturalobject',
  自然现象: 'naturalobject',
  物质: 'naturalobject',
  物体: 'naturalobject',

  data: 'data',
  figure: 'data',
  value: 'data',
  数据: 'data',
  数字: 'data',
  数值: 'data',

  content: 'content',
  book: 'content',
  video: 'content',
  内容: 'content',
  作品: 'content',
  书籍: 'content',
  视频: 'content',

  organization: 'organization',
  org: 'organization',
  company: 'organization',
  组织: 'organization',
  公司: 'organization',
  机构: 'organization',
  组织机构: 'organization',

  event: 'event',
  事件: 'event',
  activity: 'event',
  活动: 'event',

  person: 'person',
  people: 'person',
  human: 'person',
  role: 'person',
  人物: 'person',
  人类: 'person',
  人: 'person',
  角色: 'person',

  creature: 'creature',
  animal: 'creature',
  beings: 'creature',
  being: 'creature',
  alien: 'creature',
  ghost: 'creature',
  动物: 'creature',
  生物: 'creature',
  神仙: 'creature',
  鬼怪: 'creature',
  妖怪: 'creature',

  location: 'location',
  geography: 'location',
  geo: 'location',
  place: 'location',
  address: 'location',
  地点: 'location',
  位置: 'location',
  地址: 'location',
  地理: 'location',
  地域: 'location',
}

const NODE_TYPE_COLORS: Record<string, string> = {
  person: '#4169E1',
  creature: '#bd7ebe',
  organization: '#00cc00',
  location: '#cf6d17',
  event: '#00bfa0',
  concept: '#e3493b',
  method: '#b71c1c',
  content: '#0f558a',
  data: '#0000ff',
  artifact: '#4421af',
  naturalobject: '#b2e061',
  other: '#f4d371',
  unknown: '#b0b0b0',
}

const EXTENDED_COLORS = [
  '#84a3e1',
  '#5a2c6d',
  '#2F4F4F',
  '#003366',
  '#9b3a31',
  '#00CED1',
  '#b300b3',
  '#0f705d',
  '#ff99cc',
  '#6ef7b3',
  '#cd071e',
]

const PREDEFINED_COLOR_SET = new Set(Object.values(NODE_TYPE_COLORS))

interface ResolveNodeColorResult {
  color: string
  map: Map<string, string>
  updated: boolean
}

export const resolveNodeColor = (
  nodeType: string | undefined,
  currentMap: Map<string, string> | undefined
): ResolveNodeColorResult => {
  const typeColorMap = currentMap ?? new Map<string, string>()
  const normalizedType = nodeType ? nodeType.toLowerCase() : 'unknown'
  const standardType = TYPE_SYNONYMS[normalizedType]
  const cacheKey = standardType || normalizedType

  if (typeColorMap.has(cacheKey)) {
    return {
      color: typeColorMap.get(cacheKey) || DEFAULT_NODE_COLOR,
      map: typeColorMap,
      updated: false,
    }
  }

  if (standardType) {
    const color = NODE_TYPE_COLORS[standardType] || DEFAULT_NODE_COLOR
    const newMap = new Map(typeColorMap)
    newMap.set(standardType, color)
    return {
      color,
      map: newMap,
      updated: true,
    }
  }

  const usedExtendedColors = new Set(
    Array.from(typeColorMap.values()).filter((color) => !PREDEFINED_COLOR_SET.has(color))
  )

  const unusedColor = EXTENDED_COLORS.find((color) => !usedExtendedColors.has(color))
  const color = unusedColor || DEFAULT_NODE_COLOR

  const newMap = new Map(typeColorMap)
  newMap.set(normalizedType, color)

  return {
    color,
    map: newMap,
    updated: true,
  }
}

export { DEFAULT_NODE_COLOR }
