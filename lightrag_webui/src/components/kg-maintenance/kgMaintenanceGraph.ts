import type { KBIterationGraphEdge, KBIterationGraphNode } from '@/api/lightrag'

export type KGMaintenanceNodeRole = 'disease' | 'category' | 'subgroup' | 'leaf' | 'entity'

export type KGMaintenanceGraphViewNode = KBIterationGraphNode & {
  role: KGMaintenanceNodeRole
  depth: number
  size: number
  x: number
  y: number
  colorClass: string
}

export type KGMaintenanceGraphViewEdge = KBIterationGraphEdge & {
  label: string
  detailLabel: string
  sourceX: number
  sourceY: number
  targetX: number
  targetY: number
}

export type KGMaintenanceGraphView = {
  nodes: KGMaintenanceGraphViewNode[]
  edges: KGMaintenanceGraphViewEdge[]
}

const ROLE_DEPTH: Record<KGMaintenanceNodeRole, number> = {
  disease: 0,
  category: 1,
  subgroup: 2,
  leaf: 3,
  entity: 3
}

const ROLE_SIZE: Record<KGMaintenanceNodeRole, number> = {
  disease: 36,
  category: 28,
  subgroup: 22,
  leaf: 14,
  entity: 16
}

const ROLE_COLOR: Record<KGMaintenanceNodeRole, string> = {
  disease: 'bg-emerald-500 text-white border-emerald-300',
  category: 'bg-cyan-500 text-white border-cyan-300',
  subgroup: 'bg-indigo-500 text-white border-indigo-300',
  leaf: 'bg-background text-foreground border-border',
  entity: 'bg-background text-foreground border-border'
}

const GENERIC_RELATIONS = new Set(['', '邻接', '相关', '关联', 'neighbor', 'adjacent'])

export function formatRelationLabel(keywords: string | null | undefined): string {
  const label = String(keywords || '').trim()
  return GENERIC_RELATIONS.has(label.toLowerCase()) ? '未标注关系' : label
}

export function detectNodeRole(node: KBIterationGraphNode): KGMaintenanceNodeRole {
  if (node.role && ['disease', 'category', 'subgroup', 'leaf', 'entity'].includes(node.role)) {
    return node.role as KGMaintenanceNodeRole
  }

  const entityType = String(node.entity_type || '').toLowerCase()
  const properties = node.properties || {}
  if (entityType === 'disease') return 'disease'
  if (entityType === 'medicalgroup') {
    return properties.parent_group || properties.parent ? 'subgroup' : 'category'
  }
  if (properties.medical_group) return 'leaf'
  return 'entity'
}

export function buildKGMaintenanceGraphView(input: {
  nodes: KBIterationGraphNode[]
  edges: KBIterationGraphEdge[]
}): KGMaintenanceGraphView {
  const degree = new Map<string, number>()
  input.edges.forEach((edge) => {
    degree.set(edge.source, (degree.get(edge.source) || 0) + 1)
    degree.set(edge.target, (degree.get(edge.target) || 0) + 1)
  })

  const buckets = new Map<KGMaintenanceNodeRole, KBIterationGraphNode[]>()
  input.nodes.forEach((node) => {
    const role = detectNodeRole(node)
    buckets.set(role, [...(buckets.get(role) || []), node])
  })

  const nodes = input.nodes.map((node) => {
    const role = detectNodeRole(node)
    const peers = buckets.get(role) || []
    const peerIndex = Math.max(0, peers.findIndex((peer) => peer.id === node.id))
    const depth = ROLE_DEPTH[role]
    const radius = 80 + depth * 120
    const angle = peers.length <= 1 ? -Math.PI / 2 : (Math.PI * 2 * peerIndex) / peers.length
    const evidenceIncrement = node.evidenceStatus === 'grounded' ? 1 : 0
    const degreeIncrement = Math.min(degree.get(node.id) || 0, 4)

    return {
      ...node,
      role,
      depth,
      size: ROLE_SIZE[role] + evidenceIncrement + degreeIncrement,
      x: Math.round(360 + Math.cos(angle) * radius),
      y: Math.round(260 + Math.sin(angle) * radius),
      colorClass: ROLE_COLOR[role]
    }
  })

  const nodeById = new Map(nodes.map((node) => [node.id, node]))
  const edges = input.edges.map((edge) => {
    const source = nodeById.get(edge.source)
    const target = nodeById.get(edge.target)
    const label = formatRelationLabel(edge.label || edge.keywords)
    const targetLabel = edge.targetLabel || target?.label || edge.target
    return {
      ...edge,
      label,
      detailLabel: `outgoing ${label} -> ${targetLabel}`,
      sourceX: source?.x || 0,
      sourceY: source?.y || 0,
      targetX: target?.x || 0,
      targetY: target?.y || 0
    }
  })

  return { nodes, edges }
}
