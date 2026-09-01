import { describe, expect, test } from 'bun:test'
import { readFileSync, readdirSync } from 'fs'
import { join } from 'path'

/**
 * The test harness must stay out of the production bundle.
 *
 * Vite builds from the import graph rooted at `index.html` / `workspace.html`,
 * so `devDependencies` alone guarantee nothing: happy-dom, Testing Library and
 * everything under `src/test/` ship the moment ONE production module imports
 * them. The `bunfig.toml` preload is what keeps the harness reachable only
 * from `bun test` — it is loaded by the runner, never imported by `src`.
 *
 * `@faker-js/faker` is the standing proof that the dependency section is not
 * the deciding factor: it is a runtime `dependencies` entry precisely because
 * `hooks/useRandomGraph.tsx` imports it, and so it is bundled.
 */
const SRC = join(import.meta.dir, '..')

// Every module specifier, whatever the import form: `from '…'`, a bare
// side-effect `import '…'`, a dynamic `import('…')` and `require('…')`. An
// earlier version of this test matched only `from '…'` and let a bare
// side-effect import through — which is exactly the form that drags a whole
// library in for its globals.
const SPECIFIER = /(?:\bfrom\s*|\bimport\s*\(?\s*|\brequire\s*\(\s*)['"]([^'"]+)['"]/g

const HARNESS_SPECIFIER = [
  /^@\/test\//,
  /(?:^|\/)\.{1,2}\/(?:\.\.\/)*test\//,
  /^@testing-library\//,
  /^@happy-dom\//
]

/**
 * Does this source delete the `window` or `document` GLOBAL?
 *
 * Only the last property segment counts: `delete window.__LIGHTRAG_CONFIG__`
 * removes a property OF window and is fine, while
 * `delete (globalThis as Record<string, unknown>).window` removes the DOM
 * itself. Matching on a regex over the whole expression conflates the two,
 * and a character class narrow enough to avoid that misses the cast form.
 */
const deletesDomGlobal = (source: string): boolean =>
  [...source.matchAll(/\bdelete\s+([^\n;]+)/g)].some((match) => {
    const target = match[1].trim().replace(/\]\s*$/, '')
    const last = target.split(/\.|\[\s*['"]?/).pop()?.trim() ?? ''
    return /^(?:window|document)['"]?$/.test(last)
  })

const harnessImports = (source: string): string[] =>
  [...source.matchAll(SPECIFIER)]
    .map((match) => match[1])
    .filter((specifier) => HARNESS_SPECIFIER.some((pattern) => pattern.test(specifier)))

const isTestFile = (path: string): boolean =>
  path.includes('.test.') || path.replace(/\\/g, '/').includes('/src/test/')

const walk = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) return walk(full)
    return /\.(ts|tsx)$/.test(entry.name) ? [full] : []
  })

describe('test harness isolation', () => {
  test('no production module imports the harness or a test-only package', () => {
    const offenders = walk(SRC)
      .filter((path) => !isTestFile(path))
      .flatMap((path) =>
        harnessImports(readFileSync(path, 'utf8')).map(
          (specifier) => `${path.slice(SRC.length + 1)} -> ${specifier}`
        )
      )

    // A single such import would pull happy-dom (or a whole React render
    // harness) into a chunk the browser downloads.
    expect(offenders).toEqual([])
  })

  test('the harness is reached through the runner preload, not an import', () => {
    const bunfig = readFileSync(join(SRC, '..', 'bunfig.toml'), 'utf8')

    expect(bunfig).toContain('./src/test/happydom.ts')
    expect(bunfig).toContain('./src/test/setup.ts')
    // Ordering is load-bearing: Testing Library binds to whatever `document`
    // exists when it is first evaluated.
    expect(bunfig.indexOf('happydom.ts')).toBeLessThan(bunfig.indexOf('setup.ts'))
  })

  test('no test file deletes a DOM global outside the domGlobals helper', () => {
    // A bare `delete globalThis.window` removes the preloaded DOM for every
    // file Bun runs after this one (single process), and the resulting
    // failure lands in an unrelated file. src/test/domGlobals.ts is the one
    // place allowed to do it, because it puts the global back.
    const HELPER = join(SRC, 'test', 'domGlobals.ts')

    const offenders = walk(SRC)
      // This file states the pattern it forbids, so it necessarily contains
      // the words; the helper is the one place allowed to do the deletion.
      .filter((path) => path !== HELPER && path !== import.meta.path)
      .filter((path) => deletesDomGlobal(readFileSync(path, 'utf8')))
      .map((path) => path.slice(SRC.length + 1))

    expect(offenders).toEqual([])
  })

  test('the DOM packages are devDependencies', () => {
    const pkg = JSON.parse(readFileSync(join(SRC, '..', 'package.json'), 'utf8'))

    for (const name of [
      '@happy-dom/global-registrator',
      '@testing-library/react',
      '@testing-library/user-event',
      '@testing-library/jest-dom'
    ]) {
      expect(pkg.devDependencies).toHaveProperty(name)
      expect(pkg.dependencies).not.toHaveProperty(name)
    }
  })
})
