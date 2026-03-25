const HIDDEN_PROPERTY_KEYS = new Set(['created_at', 'truncate'])
const HIDDEN_NODE_PROPERTY_KEYS = new Set(['name'])

export const isEmptyGraphPropertyValue = (value: unknown): boolean => {
  if (value === null || value === undefined) {
    return true
  }

  if (typeof value === 'string') {
    return value.trim() === ''
  }

  return false
}

export const getVisibleGraphPropertyKeys = (
  properties: Record<string, unknown>,
  type: 'node' | 'edge',
  options?: {
    hideKeywords?: boolean
  }
): string[] => {
  return Object.keys(properties)
    .sort()
    .filter((name) => {
      if (HIDDEN_PROPERTY_KEYS.has(name)) {
        return false
      }

      if (type === 'node' && HIDDEN_NODE_PROPERTY_KEYS.has(name)) {
        return false
      }

      if (type === 'edge' && options?.hideKeywords && name === 'keywords') {
        return false
      }

      return !isEmptyGraphPropertyValue(properties[name])
    })
}
