export const normalizeOptionalMarkdown = (value: unknown): string =>
  typeof value === 'string' ? value : ''

export const shouldApplyWorkspaceResponse = (
  requestWorkspace: string,
  getCurrentWorkspace: () => string | null
): boolean => getCurrentWorkspace() === requestWorkspace

export const applyWorkspaceResponse = (
  requestWorkspace: string,
  getCurrentWorkspace: () => string | null,
  apply: () => void
): boolean => {
  if (!shouldApplyWorkspaceResponse(requestWorkspace, getCurrentWorkspace)) return false
  apply()
  return true
}

export const runWorkspaceAction = async <T,>({
  requestWorkspace,
  getCurrentWorkspace,
  action,
  onSuccess,
  onError,
  onComplete
}: {
  requestWorkspace: string
  getCurrentWorkspace: () => string | null
  action: () => Promise<T>
  onSuccess?: (value: T, shouldApply: () => boolean) => void | Promise<void>
  onError?: (error: unknown, shouldApply: () => boolean) => void | Promise<void>
  onComplete?: () => void
}): Promise<void> => {
  const shouldApply = () => shouldApplyWorkspaceResponse(requestWorkspace, getCurrentWorkspace)
  try {
    const value = await action()
    if (shouldApply()) {
      await onSuccess?.(value, shouldApply)
    }
  } catch (error) {
    if (shouldApply()) {
      await onError?.(error, shouldApply)
    }
  } finally {
    onComplete?.()
  }
}

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
  const firstLine = message.split(/\r?\n/, 1)[0]?.trim() || ''
  return /^404\b/.test(firstLine)
}

function getErrorStatus(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') return undefined
  const record = error as Record<string, any>
  const responseStatus = record.response?.status
  if (typeof responseStatus === 'number') return responseStatus
  const status = record.status ?? record.statusCode
  return typeof status === 'number' ? status : undefined
}
