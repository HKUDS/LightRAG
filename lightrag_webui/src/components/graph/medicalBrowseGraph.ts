import type { LightragGraphType, LightragMedicalBrowseMetadata } from '@/api/lightrag'

export type MedicalBrowseNodeRole =
  | 'root'
  | 'category'
  | 'subgroup'
  | 'collapsed_group'
  | 'leaf'
  | string

export type MedicalBrowsePosition = {
  x: number
  y: number
}

export type MedicalBrowseNodeSizeOptions = {
  degree?: number
  maxDegree?: number
}

const ROLE_RADII: Record<string, number> = {
  root: 0,
  category: 2,
  subgroup: 4,
  collapsed_group: 5.5,
  leaf: 6.5
}

const roundPosition = (value: number): number => Number(value.toFixed(6))

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))

const degreeImportance = (degree: number = 0, maxDegree: number = 0): number => {
  if (degree <= 0 || maxDegree <= 0) {
    return 0
  }

  const relativeImportance = Math.sqrt(clamp(degree / maxDegree, 0, 1))
  const absoluteConfidence = 1 - Math.exp(-degree / 8)

  return relativeImportance * absoluteConfidence
}

const collapsedGroupCount = (
  nodeId: string,
  browse?: LightragMedicalBrowseMetadata
): number => browse?.collapsed_groups?.find((group) => group.id === nodeId)?.count ?? 0

const sortedRoleNodeIds = (
  role: MedicalBrowseNodeRole,
  browse: LightragMedicalBrowseMetadata
): string[] => {
  if (role === 'collapsed_group') {
    return [...(browse.collapsed_groups ?? []).map((group) => group.id)].sort((a, b) =>
      a.localeCompare(b)
    )
  }

  const nodeRoles = browse.node_roles ?? {}
  const categoryOrder = browse.category_order ?? []

  return Object.entries(nodeRoles)
    .filter(([, nodeRole]) => nodeRole === role)
    .map(([nodeId]) => nodeId)
    .sort((left, right) => {
      if (role === 'category') {
        const leftIndex = categoryOrder.indexOf(left)
        const rightIndex = categoryOrder.indexOf(right)

        if (leftIndex !== -1 || rightIndex !== -1) {
          if (leftIndex === -1) return 1
          if (rightIndex === -1) return -1
          return leftIndex - rightIndex
        }
      }

      return left.localeCompare(right)
    })
}

export const getMedicalBrowseNodeRole = (
  nodeId: string,
  browse?: LightragMedicalBrowseMetadata
): MedicalBrowseNodeRole => {
  if (!browse) {
    return 'leaf'
  }

  if (browse.root_id === nodeId) {
    return 'root'
  }

  if (browse.collapsed_groups?.some((group) => group.id === nodeId)) {
    return 'collapsed_group'
  }

  return browse.node_roles?.[nodeId] ?? 'leaf'
}

export const getMedicalBrowseNodeSize = (
  nodeId: string,
  nodeEntityType: string | undefined,
  browse?: LightragMedicalBrowseMetadata,
  options: MedicalBrowseNodeSizeOptions = {}
): number | undefined => {
  if (!browse) {
    return undefined
  }

  const role = getMedicalBrowseNodeRole(nodeId, browse)
  const importance = degreeImportance(options.degree, options.maxDegree)

  if (role === 'root') {
    return 24
  }

  if (role === 'category') {
    return Math.round(14 + 4 * importance)
  }

  if (role === 'subgroup') {
    return Math.round(11 + 4 * importance)
  }

  if (role === 'collapsed_group' || nodeEntityType === 'MedicalCollapsedGroup') {
    const count = collapsedGroupCount(nodeId, browse)
    const countBoost = count > 0 ? Math.sqrt(count) : 0
    return Math.round(clamp(9 + 3 * importance + countBoost, 10, 16))
  }

  return Math.round(6 + 10 * importance)
}

export const getMedicalBrowsePosition = (
  nodeId: string,
  browse: LightragMedicalBrowseMetadata,
  fallbackIndex: number = 0
): MedicalBrowsePosition => {
  const role = getMedicalBrowseNodeRole(nodeId, browse)

  if (role === 'root') {
    return { x: 0, y: 0 }
  }

  const radius = ROLE_RADII[role] ?? ROLE_RADII.leaf
  const roleNodeIds = sortedRoleNodeIds(role, browse)
  const roleIndex = roleNodeIds.indexOf(nodeId)
  const index = roleIndex === -1 ? fallbackIndex : roleIndex
  const count = Math.max(roleNodeIds.length, index + 1, 1)
  const angle = -Math.PI / 2 + (2 * Math.PI * index) / count

  return {
    x: roundPosition(Math.cos(angle) * radius),
    y: roundPosition(Math.sin(angle) * radius)
  }
}

export const applyMedicalBrowseProjection = (graph: LightragGraphType): LightragGraphType => {
  const browse = graph.metadata?.medical_browse

  if (!browse?.collapsed_groups?.length) {
    return graph
  }

  const collapsedChildren = new Set(
    browse.collapsed_groups.flatMap((group) => group.child_ids ?? [])
  )
  const nodes = graph.nodes.filter((node) => !collapsedChildren.has(node.id))
  const edges = graph.edges.filter(
    (edge) => !collapsedChildren.has(edge.source) && !collapsedChildren.has(edge.target)
  )

  for (const group of browse.collapsed_groups) {
    nodes.push({
      id: group.id,
      labels: [group.label],
      properties: {
        entity_id: group.label,
        entity_type: 'MedicalCollapsedGroup',
        medical_group_parent: group.parent_id,
        child_ids: [...group.child_ids],
        count: group.count,
        examples: [...group.examples]
      }
    })
    edges.push({
      id: `edge:${group.parent_id}:${group.id}`,
      source: group.parent_id,
      target: group.id,
      type: 'medicalCollapsedGroup',
      properties: { keywords: '包含', weight: 0.1 }
    })
  }

  return { ...graph, nodes, edges }
}
