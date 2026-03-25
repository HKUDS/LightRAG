type NodeDisplayCandidate = {
  id?: string
  labels?: string[]
  properties?: Record<string, unknown>
}

const asNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== 'string') {
    return undefined
  }

  const trimmed = value.trim()
  return trimmed ? trimmed : undefined
}

export const resolveNodeDisplayName = (
  node: NodeDisplayCandidate | null | undefined
): string => {
  const name = asNonEmptyString(node?.properties?.name)
  if (name) {
    return name
  }

  const entityId = asNonEmptyString(node?.properties?.entity_id)
  if (entityId) {
    return entityId
  }

  const firstLabel = node?.labels?.find(
    (label) => typeof label === 'string' && label.trim()
  )
  if (firstLabel) {
    return firstLabel
  }

  const id = asNonEmptyString(node?.id)
  return id ?? ''
}
