import type { DocStatusResponse } from '@/api/lightrag'

export const KG_RECOVERY_WARNINGS_METADATA_KEY = 'kg_recovery_warnings'

export const hasKgRecoveryWarnings = (
  metadata: Record<string, any> | undefined
): boolean => {
  const warnings = metadata?.[KG_RECOVERY_WARNINGS_METADATA_KEY]
  return (
    Array.isArray(warnings) &&
    warnings.some((warning) => warning && typeof warning === 'object')
  )
}

export const hasDocumentWarning = (
  doc: Pick<DocStatusResponse, 'error_msg' | 'metadata'>
): boolean => Boolean(doc.error_msg) || hasKgRecoveryWarnings(doc.metadata)
