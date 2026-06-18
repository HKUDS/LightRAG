export const normalizeOptionalMarkdown = (value: unknown): string =>
  typeof value === 'string' ? value : ''

export const shouldApplyWorkspaceResponse = (
  requestWorkspace: string,
  getCurrentWorkspace: () => string | null
): boolean => getCurrentWorkspace() === requestWorkspace

export const optionalMissingResponse = async <T,>(
  loader: () => Promise<T>,
  fallback: T
): Promise<T> => {
  try {
    return await loader()
  } catch (error) {
    if (isMissingResourceError(error)) return fallback
    throw error
  }
}

function isMissingResourceError(error: unknown): boolean {
  const status = getErrorStatus(error)
  if (status === 404) return true

  const message = error instanceof Error ? error.message : String(error)
  return /\b404\b/.test(message) || /not found/i.test(message) || /missing artifact/i.test(message)
}

function getErrorStatus(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') return undefined
  const record = error as Record<string, any>
  const responseStatus = record.response?.status
  if (typeof responseStatus === 'number') return responseStatus
  const status = record.status ?? record.statusCode
  return typeof status === 'number' ? status : undefined
}
