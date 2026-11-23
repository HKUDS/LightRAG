/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { readdirSync, readFileSync } from 'fs'
import { createRequire } from 'module'
import { dirname, join } from 'path'

import { PRISM_LANGUAGE_ALIASES, resolvePrismLanguage } from './prismLanguages'

const require = createRequire(import.meta.url)

/*
 * Why the table is checked as data rather than by rendering a ```ts block:
 * under `bun test`, `react-syntax-highlighter` resolves to its CJS barrel,
 * which eagerly loads the full `Prism` export and registers every refractor
 * language, aliases included, on the shared refractor instance. Every fence
 * then highlights with or without this table; only the Vite build is lazy.
 */

/** The keys `PrismAsyncLight` can load, read from its own loader table. */
const loaderKeys = (): Set<string> => {
  const pkg = dirname(require.resolve('react-syntax-highlighter/package.json'))
  const source = readFileSync(join(pkg, 'dist/esm/async-languages/prism.js'), 'utf8')
  return new Set([...source.matchAll(/createLanguageAsyncLoader\("([^"]+)"/g)].map((m) => m[1]))
}

/** Recompute the alias table from refractor, under the rules the module states. */
const expectedAliases = (): Record<string, string> => {
  const keys = loaderKeys()
  const langDir = join(dirname(require.resolve('refractor/package.json')), 'lang')
  const expected: Record<string, string> = {}
  for (const file of readdirSync(langDir).filter((name) => name.endsWith('.js'))) {
    const source = readFileSync(join(langDir, file), 'utf8')
    const match = source.match(/^([\w$]+)\.displayName = '([^']+)'\n\1\.aliases = \[([^\]]*)\]/m)
    if (!match) throw new Error(`refractor/lang/${file} no longer declares displayName + aliases`)
    const [, , name, list] = match
    if (!keys.has(name)) continue
    for (const alias of [...list.matchAll(/'([^']+)'/g)].map((m) => m[1])) {
      if (/^\w+$/.test(alias) && !keys.has(alias)) expected[alias] = name
    }
  }
  return expected
}

describe('PRISM_LANGUAGE_ALIASES', () => {
  test('matches the aliases of the installed refractor', () => {
    const expected = expectedAliases()
    // Sanity: the parse found the table at all (refractor has ~80 aliases).
    expect(Object.keys(expected).length).toBeGreaterThan(50)
    // Compare sorted entry lists so a failure names the differing alias.
    const entries = (table: Readonly<Record<string, string>>) =>
      Object.entries(table).sort(([a], [b]) => a.localeCompare(b))
    expect(entries(PRISM_LANGUAGE_ALIASES)).toEqual(entries(expected))
  })

  test.each([
    ['ts', 'typescript'],
    ['python', 'python'],
    ['constructor', 'constructor'],
    [undefined, undefined]
  ])('resolvePrismLanguage(%p) is %p', (input, output) => {
    expect(resolvePrismLanguage(input)).toBe(output)
  })
})
