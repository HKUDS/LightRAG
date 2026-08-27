import { describe, expect, test } from 'bun:test'
import { readFileSync, existsSync } from 'fs'
import { dirname, join, resolve } from 'path'

/**
 * Static source assertions guarding the migration module's ordering and
 * purity contracts (workspace-entry PRD §7.4):
 *
 * - the split-migration side-effect module is the FIRST static import of
 *   BOTH entry files;
 * - the migration modules' transitive import graph contains no store, no
 *   `App`, no API client and no navigation module (importing a store would
 *   hydrate it BEFORE the migration ran).
 */

const SRC = resolve(import.meta.dir, '..')

const IMPORT_RE = /^import\s+(?:[^'"]*?from\s+)?['"]([^'"]+)['"]/gm

function importsOf(filePath: string): string[] {
  const source = readFileSync(filePath, 'utf8')
  const specs: string[] = []
  for (const match of source.matchAll(IMPORT_RE)) {
    specs.push(match[1])
  }
  return specs
}

function resolveSpec(spec: string, fromFile: string): string | null {
  let base: string
  if (spec.startsWith('@/')) {
    base = join(SRC, spec.slice(2))
  } else if (spec.startsWith('.')) {
    base = resolve(dirname(fromFile), spec)
  } else {
    return null // bare package import — not project source
  }
  for (const candidate of [
    base,
    `${base}.ts`,
    `${base}.tsx`,
    join(base, 'index.ts'),
    join(base, 'index.tsx')
  ]) {
    if (existsSync(candidate) && !candidate.endsWith('/')) {
      try {
        if (readFileSync(candidate, 'utf8') !== null) return candidate
      } catch {
        /* directory — keep looking */
      }
    }
  }
  return null
}

function transitiveGraph(entryFile: string): Set<string> {
  const seen = new Set<string>()
  const stack = [entryFile]
  while (stack.length > 0) {
    const file = stack.pop()!
    if (seen.has(file)) continue
    seen.add(file)
    for (const spec of importsOf(file)) {
      const resolved = resolveSpec(spec, file)
      if (resolved) stack.push(resolved)
    }
  }
  return seen
}

describe('entry import order', () => {
  test.each(['main.tsx', 'workspace-main.tsx'])(
    '%s has the migration side-effect module as its FIRST static import',
    (entry) => {
      const source = readFileSync(join(SRC, entry), 'utf8')
      const firstImport = source.match(IMPORT_RE)?.[0] ?? ''
      expect(firstImport).toContain('@/migrations/runSettingsStorageSplit')
    }
  )
})

describe('migration module purity', () => {
  const FORBIDDEN = [
    /\/stores\//,
    /\/App\.tsx$/,
    /\/api\/lightrag/,
    /\/services\/navigation/
  ]

  test.each(['runSettingsStorageSplit.ts', 'legacySettingsChain.ts', 'splitSettingsStorage.ts'])(
    'migrations/%s transitively imports no store, App, API client or navigation module',
    (moduleFile) => {
      const graph = transitiveGraph(join(SRC, 'migrations', moduleFile))
      const offenders = [...graph].filter((file) =>
        FORBIDDEN.some((pattern) => pattern.test(file))
      )
      expect(offenders).toEqual([])
    }
  )

  test('importing the pure chain module performs no localStorage access', async () => {
    const calls: string[] = []
    const spyStorage = new Proxy(
      {},
      {
        get(_t, prop) {
          calls.push(String(prop))
          return () => null
        }
      }
    )
    const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
    Object.defineProperty(globalThis, 'localStorage', {
      value: spyStorage,
      configurable: true
    })
    try {
      await import('./legacySettingsChain')
    } finally {
      if (original) {
        Object.defineProperty(globalThis, 'localStorage', original)
      } else {
        delete (globalThis as Record<string, unknown>).localStorage
      }
    }
    expect(calls).toEqual([])
  })
})

describe('navigation core stays graph-free', () => {
  test('services/navigation transitively imports no stores/graph (graphology stays out of the workspace entry)', () => {
    const graph = transitiveGraph(join(SRC, 'services', 'navigation.ts'))
    const offenders = [...graph].filter((file) => /stores\/graph/.test(file))
    expect(offenders).toEqual([])
  })
})

describe('no path-rewriting hard navigation in the SPA', () => {
  test('frontend source contains no location.href= / location.replace( / location.assign(', () => {
    const { execSync } = require('child_process') as typeof import('child_process')
    const output = execSync(
      String.raw`grep -rnE "location\.href\s*=|location\.replace\(|location\.assign\(" --include='*.ts' --include='*.tsx' . || true`,
      { cwd: SRC, encoding: 'utf8' }
    )
    const offenders = output
      .split('\n')
      .filter(Boolean)
      // window.location.reload() is allowed (same-URL reload, keeps entry);
      // test files may reference these strings.
      .filter((line) => !line.includes('.test.'))
    expect(offenders).toEqual([])
  })
})
