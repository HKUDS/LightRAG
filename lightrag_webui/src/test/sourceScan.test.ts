/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { join } from 'path'

import { relativePath, sourceFiles } from './sourceScan'

describe('source scanning helpers', () => {
  test('relativePath returns the / tail even under a native-separator root', () => {
    // The regression this module exists for, as a pure function so it runs
    // the same on every platform: a Windows root reaches the audits with
    // backslashes, and what they compare against a literal is the tail.
    expect(relativePath('C:\\repo\\src', 'C:\\repo\\src/components/retrieval/mermaidLoader.ts')).toBe(
      'components/retrieval/mermaidLoader.ts'
    )
  })

  test('scanned paths are / separated and exclusions on them fire', () => {
    const dir = join(import.meta.dir, '..', 'lib')
    const files = sourceFiles(dir, (path) => path.endsWith('.ts'))

    expect(files.length).toBeGreaterThan(0)
    expect(files.filter((file) => file.includes('\\'))).toHaveLength(0)

    // A suffix exclusion is exactly what broke on Windows, so pin that one
    // works here rather than only that the paths look right.
    const identity = files.filter((file) => file.endsWith('/loginIdentity.ts'))
    expect(identity).toHaveLength(1)
  })

  test('accept decides what is kept, and directories are descended', () => {
    const dir = join(import.meta.dir, '..')
    const onlyTsx = sourceFiles(dir, (path) => path.endsWith('.tsx'))

    expect(onlyTsx.filter((file) => !file.endsWith('.tsx'))).toHaveLength(0)
    // Nested, so a walk that failed to recurse would come back short.
    expect(onlyTsx.some((file) => file.includes('/components/'))).toBe(true)
  })
})
