import { describe, expect, test } from 'bun:test'
import { readdirSync, readFileSync, statSync } from 'fs'
import { join } from 'path'
import { decodeBase64Url, legacyMojibakeForm } from './base64url'

/**
 * The single decoder every JWT payload in this app goes through.
 *
 * Three separate readers of the same token drifted onto bare `atob` at
 * different times — `isTokenLocallyValid`, `parseTokenPayload`,
 * `loginIdentityFromToken` and the `x-new-token` renewal interceptor — and
 * each disagreed with the others about tokens using the `-`/`_` alphabet.
 * Every one of them failed SILENTLY, so the source sweep below is the part
 * that keeps the next reader from repeating it.
 */

describe('decodeBase64Url', () => {
  const encode = (value: string) =>
    Buffer.from(value, 'utf8')
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '')

  test('decodes the url alphabet that bare atob rejects', () => {
    const payload = JSON.stringify({ sub: '~~h', role: 'user', exp: 9999999999 })
    const encoded = encode(payload)
    // Only meaningful if this vector genuinely needs the url alphabet.
    expect(encoded).toMatch(/[-_]/)
    expect(() => atob(encoded)).toThrow()

    expect(decodeBase64Url(encoded)).toBe(payload)
  })

  test('accepts the unpadded form and every valid remainder', () => {
    for (const value of ['a', 'ab', 'abc', 'abcd', 'abcde', 'abcdef']) {
      expect(decodeBase64Url(encode(value))).toBe(value)
    }
  })

  test('reads the payload as UTF-8, not one character per byte', () => {
    // `atob` yields a latin-1 byte string, so a non-ASCII name came back
    // mangled: a wrong username on screen, and a wrong identity string in
    // the retrieval-history cleanup.
    for (const sub of ['كظدج', '张三', 'José', 'Ünal', '日本語ユーザー']) {
      const payload = JSON.stringify({ sub, role: 'user', exp: 9999999999 })
      expect(decodeBase64Url(encode(payload))).toBe(payload)
      expect(JSON.parse(decodeBase64Url(encode(payload))!).sub).toBe(sub)
    }
  })

  test('rejects invalid UTF-8 instead of substituting replacement characters', () => {
    // 0xFF is never valid UTF-8. Letting U+FFFD through would yield claims
    // that parse but say something the token does not.
    const invalid = Buffer.from([0x7b, 0x22, 0xff, 0x22, 0x7d])
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '')
    expect(decodeBase64Url(invalid)).toBeNull()
  })

  test('returns null rather than throwing on malformed input', () => {
    // A length of 4n+1 can never be valid base64, padded or not.
    expect(decodeBase64Url('A')).toBeNull()
    expect(decodeBase64Url('abcde')).toBeNull()
    // Characters outside the alphabet.
    expect(decodeBase64Url('@@@@')).toBeNull()
  })
})

describe('no reader decodes a JWT payload on its own', () => {
  const walk = (dir: string): string[] =>
    readdirSync(dir).flatMap((entry) => {
      const full = join(dir, entry)
      if (statSync(full).isDirectory()) return walk(full)
      return /\.(ts|tsx)$/.test(entry) && !/\.test\.tsx?$/.test(entry) ? [full] : []
    })

  test('atob appears only inside this module', () => {
    // A silent decoder mismatch is invisible to behavioral tests unless one
    // happens to use a url-alphabet vector, so pin it at the source level:
    // any new `atob` call is a reader that will disagree with the others.
    const src = join(import.meta.dir, '..')
    const offenders = walk(src).filter(
      (file) =>
        file !== join(src, 'lib', 'base64url.ts') && /\batob\s*\(/.test(readFileSync(file, 'utf8'))
    )

    expect(offenders.map((f) => f.slice(src.length + 1))).toEqual([])
  })
})


describe('legacyMojibakeForm', () => {
  test('reproduces what the pre-fix decoder returned', () => {
    // Pinned so the upgrade path in `isIdentityChange` keeps recognizing
    // markers written by that build.
    const sub = 'كظدج'
    const payload = JSON.stringify({ sub })
    const oldReading = atob(
      Buffer.from(payload, 'utf8').toString('base64')
    )
    expect(legacyMojibakeForm(payload)).toBe(oldReading)
    expect(JSON.parse(legacyMojibakeForm(payload)).sub).toBe(legacyMojibakeForm(sub))
  })

  test('is the identity function for ASCII, so it costs nothing elsewhere', () => {
    for (const value of ['alice', 'guest', 'user.name-1', '']) {
      expect(legacyMojibakeForm(value)).toBe(value)
    }
  })
})
