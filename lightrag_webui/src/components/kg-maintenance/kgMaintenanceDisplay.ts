export type EvidenceIssueRow = {
  id: string
  itemType: '节点' | '关系'
  itemId: string
  issue: '缺少证据来源 ID' | '缺少来源文件'
}

export type QualityBeforeSnapshot = {
  overall?: number
  metrics?: Record<string, number>
}

export function extractQualityBefore(applyResult: string): QualityBeforeSnapshot {
  const metrics: Record<string, number> = {}
  const snapshot: QualityBeforeSnapshot = {}

  applyResult.split(/\r?\n/).forEach((line) => {
    const match = line.match(/^\s*([A-Za-z0-9_.-]+)\s*:\s*(-?\d+(?:\.\d+)?)\s*->\s*-?\d+(?:\.\d+)?\s*$/)
    if (!match) return

    const [, key, beforeValue] = match
    const numericBefore = Number(beforeValue)
    if (!Number.isFinite(numericBefore)) return

    if (key === 'overall') {
      snapshot.overall = numericBefore
      return
    }

    metrics[key] = numericBefore
  })

  if (Object.keys(metrics).length > 0) {
    snapshot.metrics = metrics
  }

  return snapshot
}

export function buildEvidenceIssueRows(
  snapshot: Record<string, any> | null | undefined
): EvidenceIssueRow[] {
  const record = asRecord(snapshot)
  if (!record) return []

  const rows: EvidenceIssueRow[] = []
  const nodes = normalizeCollection(record.nodes ?? record.entities)
  const edges = normalizeCollection(record.edges ?? record.relations ?? record.links)

  nodes.forEach((node, index) => {
    const itemId = itemIdForNode(node, index)
    appendIssueRows(rows, 'node', '节点', itemId, node)
  })

  edges.forEach((edge, index) => {
    const itemId = itemIdForEdge(edge, index)
    appendIssueRows(rows, 'edge', '关系', itemId, edge)
  })

  return rows
}

function appendIssueRows(
  rows: EvidenceIssueRow[],
  idPrefix: 'node' | 'edge',
  itemType: EvidenceIssueRow['itemType'],
  itemId: string,
  item: Record<string, any>
) {
  if (isMissing(item.source_id)) {
    rows.push({
      id: `${idPrefix}:${itemId}:source_id`,
      itemType,
      itemId,
      issue: '缺少证据来源 ID'
    })
  }

  if (isMissing(item.file_path)) {
    rows.push({
      id: `${idPrefix}:${itemId}:file_path`,
      itemType,
      itemId,
      issue: '缺少来源文件'
    })
  }
}

function normalizeCollection(value: unknown): Array<Record<string, any>> {
  if (Array.isArray(value)) {
    return value.filter(isRecord)
  }

  if (!isRecord(value)) return []

  return Object.entries(value).flatMap(([id, item]) => {
    if (!isRecord(item)) return []
    return item.id === undefined || item.id === null || item.id === '' ? [{ ...item, id }] : [item]
  })
}

function itemIdForNode(node: Record<string, any>, index: number): string {
  return stringifyId(node.id ?? node.label ?? node.entity_name ?? node.name ?? `node-${index + 1}`)
}

function itemIdForEdge(edge: Record<string, any>, index: number): string {
  const explicitId = edge.id
  if (!isMissing(explicitId)) return stringifyId(explicitId)

  const source = edge.source ?? edge.from ?? edge.src
  const target = edge.target ?? edge.to ?? edge.dst
  if (!isMissing(source) && !isMissing(target)) {
    return `${stringifyId(source)}->${stringifyId(target)}`
  }

  return `edge-${index + 1}`
}

function stringifyId(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return safeStringify(value)
}

function safeStringify(value: unknown): string {
  try {
    const result = JSON.stringify(value)
    if (result) return result
  } catch {
    return '无法序列化'
  }

  try {
    return String(value)
  } catch {
    return '无法序列化'
  }
}

function isMissing(value: unknown): boolean {
  if (value === null || value === undefined) return true
  if (typeof value === 'string') return value.trim().length === 0
  if (Array.isArray(value)) return value.length === 0
  return false
}

function asRecord(value: unknown): Record<string, any> | null {
  return isRecord(value) ? value : null
}

function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
