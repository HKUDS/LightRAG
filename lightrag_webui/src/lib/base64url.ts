/**
 * Decode the base64url alphabet a JWT actually uses, returning null instead
 * of throwing on anything malformed.
 *
 * Bare `atob` rejects the `-` and `_` that base64url substitutes for `+` and
 * `/`, and rejects the unpadded form. A caller that swallows that failure
 * does not merely lose the payload — it silently proceeds on its fallback,
 * so this lives in one place rather than being re-derived per call site.
 */
export const decodeBase64Url = (value: string): string | null => {
  try {
    let base64 = value.replace(/-/g, '+').replace(/_/g, '/')
    const pad = base64.length % 4
    if (pad === 1) return null
    if (pad > 0) base64 += '='.repeat(4 - pad)
    return atob(base64)
  } catch {
    return null
  }
}
