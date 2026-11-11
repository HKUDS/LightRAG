/**
 * QuotaExceededError detection, tolerant of the browser variants (modern
 * DOMException name, legacy Firefox name, legacy numeric codes) and of test
 * doubles that throw a plain Error carrying the standard name. Deliberately
 * NOT message-based: a generic write failure whose message merely mentions
 * "quota" must not trigger the quota-specific degradation paths.
 */
export function isQuotaExceededError(error: unknown): boolean {
  if (typeof DOMException !== 'undefined' && error instanceof DOMException) {
    return (
      error.name === 'QuotaExceededError' ||
      error.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
      error.code === 22 ||
      error.code === 1014
    )
  }
  return error instanceof Error && error.name === 'QuotaExceededError'
}
