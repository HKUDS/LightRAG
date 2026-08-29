import { describe, expect, test } from 'bun:test'
import { readdirSync, readFileSync, statSync } from 'fs'
import { join } from 'path'
import { decodeBase64Url } from './base64url'

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
