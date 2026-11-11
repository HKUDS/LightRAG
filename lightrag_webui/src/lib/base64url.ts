/**
 * Decode a JWT's base64url payload to the TEXT it encodes, returning null
 * instead of throwing on anything malformed.
 *
 * Two things bare `atob` gets wrong here, both silently:
 *
 *  1. It rejects the `-`/`_` that base64url substitutes for `+`/`/`, and the
 *     unpadded form. A caller that swallows that does not merely lose the
 *     payload — it proceeds on its fallback as if the token said something
 *     else.
 *  2. It yields one CHARACTER PER BYTE (a latin-1 string), while a JWT
 *     payload is UTF-8. So `"sub":"كظدج"` came back as `ÙØ¸Ø¯Ø¬` — a
 *     mangled username on screen, and a mangled identity in the
 *     retrieval-history cleanup.
 *
 * Hence one decoder, in one place, rather than re-derived per call site.
 * Invalid UTF-8 is rejected outright: a JWT payload is JSON, which must be
 * UTF-8, and letting U+FFFD through would produce a token that parses into
 * plausible-looking but wrong claims.
 */
export const decodeBase64Url = (value: string): string | null => {
  try {
    let base64 = value.replace(/-/g, '+').replace(/_/g, '/')
    const pad = base64.length % 4
    if (pad === 1) return null
    if (pad > 0) base64 += '='.repeat(4 - pad)
    const binary = atob(base64)
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0))
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    return null
  }
}

/**
 * The text `decodeBase64Url` used to return for `value`: the byte string
 * `atob` produces, before it is read as UTF-8.
 *
 * Only for recognizing values PERSISTED by an older build. Correcting the
 * decoding changes what a non-ASCII identity string looks like, so a marker
 * written before the fix would no longer match the same person — and the
 * identity-change cleanup would read that as a new user and wipe both
 * entries' retrieval histories once, on the upgrade load. Comparing against
 * this form instead recognizes them. For ASCII input it is the identity
 * function, so it costs nothing and changes nothing for everyone else.
 */
export const legacyMojibakeForm = (value: string): string => {
  let out = ''
  for (const byte of new TextEncoder().encode(value)) out += String.fromCharCode(byte)
  return out
}
