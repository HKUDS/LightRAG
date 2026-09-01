/// <reference types="bun" />
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { readdirSync, readFileSync } from 'fs'
import { join } from 'path'
import type { ComponentType } from 'react'
import { act, cleanup } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import type { MessageWithError } from '@/types/retrieval'

/**
 * Mermaid is loaded on demand (`mermaidLoader.ts`) so it stays out of the
 * workspace entry's first-load closure. That introduces a failure mode a
 * static import could not have: the renderer itself can fail to arrive (a
 * deploy invalidates the hashed chunk mid-session, an offline tab, a flaky
 * network).
 *
 * The load-failure branch must therefore write something VISIBLE into the
 * container, exactly like the two render-failure branches beside it — a bare
 * `return` there is a permanently blank gap, not a transient glitch: the
 * container is still empty at that point, nothing retries once the message
 * settles, and the plain-text fallback in the render body only runs while
 * `renderAsDiagram` is false.
 *
 * The LOADER is stubbed here, never the `mermaid` package. Bun's module-mock
 * registry is process-wide and a module mock cannot be undone, so mocking
 * `mermaid` itself would hand every later file this file's stub, and a
 * rejecting mock of it only registers while nothing in the process has loaded
 * mermaid yet — which this file cannot guarantee about the files before it.
 * Stubbing the one-function loader is order-independent and restored exactly
 * in `afterAll`.
 */

const DIAGRAM = 'graph TD;\n  A-->B;'
const LOAD_ERROR = 'Loading chunk mermaid-a1b2c3 failed'

let ChatMessage: ComponentType<{ message: MessageWithError }>
let realLoaderModule: Record<string, unknown>
let consoleErrors: unknown[][]
let originalConsoleError: typeof console.error

/** Point the loader at `outcome` for the next render. */
const stubLoader = (outcome: () => Promise<unknown>): void => {
  mock.module('./mermaidLoader', () => ({ ...realLoaderModule, loadMermaid: outcome }))
}

const rejectingLoader = () => () => Promise.reject(new Error(LOAD_ERROR))

const workingLoader = (mermaid: Record<string, unknown>) => () => Promise.resolve(mermaid)

beforeAll(async () => {
  realLoaderModule = { ...(await import('./mermaidLoader')) }
  stubLoader(rejectingLoader())

  // Dynamic and AFTER the stub, so ChatMessage binds the stubbed loader.
  ChatMessage = (await import('./ChatMessage')).ChatMessage
})

afterAll(() => {
  mock.module('./mermaidLoader', () => realLoaderModule)
})

beforeEach(() => {
  consoleErrors = []
  originalConsoleError = console.error
  console.error = (...args: unknown[]) => {
    consoleErrors.push(args)
  }
})

afterEach(() => {
  cleanup()
  console.error = originalConsoleError
})

const diagramMessage = (): MessageWithError =>
  ({
    id: 'a1',
    role: 'assistant',
    content: '```mermaid\n' + DIAGRAM + '\n```',
    // The stream has finished, so the block renders as a diagram rather than
    // as plain text. This is the state in which the container is JS-filled.
    mermaidRendered: true
  }) as MessageWithError

const container = (): HTMLElement => {
  const element = document.querySelector<HTMLElement>('.mermaid-diagram-container')
  if (!element) throw new Error('no mermaid diagram container was rendered')
  return element
}

/** The effect debounces by 300 ms before it even starts loading. */
const settleDiagram = async (): Promise<void> => {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 600))
  })
}

const renderDiagram = (): void => {
  renderWithProviders(<ChatMessage message={diagramMessage()} />)
}

describe('mermaid renderer fails to load', () => {
  beforeEach(() => {
    stubLoader(rejectingLoader())
  })

  test('nothing is put in the container before the effect runs', async () => {
    renderDiagram()
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50))
    })

    // The premise of the whole fix, and the reason a bare `return` in the
    // load-failure branch is a permanent blank gap rather than a glitch: with
    // `renderAsDiagram` true the render body returns an EMPTY div, so the only
    // thing that can ever put content here is the effect. (This is still inside
    // the effect's own 300 ms debounce, so nothing has run yet.)
    expect(container().innerHTML).toBe('')
  })

  test('the failure is written into the container, with the diagram source', async () => {
    renderDiagram()
    await settleDiagram()

    const text = container().textContent ?? ''
    expect(text).toContain('Mermaid renderer failed to load')
    expect(text).toContain(LOAD_ERROR)
    // Without the echo the user sees an error with no way to recover the
    // content: the plain-text branch is unreachable while renderAsDiagram holds.
    expect(text).toContain('Content:')
    expect(text).toContain('A-->B;')
    // And it is the ONLY thing in the container.
    expect(container().children).toHaveLength(1)
  })

  test('the failure is logged for the operator too', async () => {
    renderDiagram()
    await settleDiagram()

    expect(
      consoleErrors.some((args) => String(args[0]).includes('Failed to load mermaid'))
    ).toBe(true)
  })
})

describe('mermaid renderer loads but the diagram does not', () => {
  test('a rejected render surfaces the reason and the source', async () => {
    stubLoader(
      workingLoader({
        initialize: () => {},
        render: () => Promise.reject(new Error('Parse error on line 2'))
      })
    )

    renderDiagram()
    await settleDiagram()

    const text = container().textContent ?? ''
    expect(text).toContain('Mermaid diagram error')
    expect(text).toContain('Parse error on line 2')
    expect(text).toContain('A-->B;')
  })

  test('a throwing setup surfaces the reason', async () => {
    stubLoader(
      workingLoader({
        initialize: () => {
          throw new Error('unsupported theme')
        },
        render: async () => ({ svg: '' })
      })
    )

    renderDiagram()
    await settleDiagram()

    // The third failure path. Consistency is the actual invariant here: the
    // load path was once the only one that logged and gave up silently.
    expect(container().textContent ?? '').toContain('unsupported theme')
  })

  test('a successful render replaces the container with the diagram', async () => {
    stubLoader(
      workingLoader({
        initialize: () => {},
        render: async () => ({ svg: '<svg data-testid="diagram"><title>A to B</title></svg>' })
      })
    )

    renderDiagram()
    await settleDiagram()

    // Proves the failure assertions above are not vacuous — this component
    // really does fill the container when the renderer works.
    expect(container().innerHTML).toContain('data-testid="diagram"')
    expect(container().textContent ?? '').not.toContain('failed to load')
  })
})

describe('mermaid failure branches (source-level)', () => {
  const SOURCE = readFileSync(join(import.meta.dir, 'ChatMessage.tsx'), 'utf8')

  test('mermaidLoader is the only module that names the mermaid package', () => {
    // A forbidden import, which is a genuinely source-level property. Two
    // things rest on it: a static `import ... from 'mermaid'` anywhere in the
    // graph would put the renderer in the workspace entry's first-load chunk,
    // and a second dynamic import would be a load path the stub above cannot
    // reach — so a failure there would be untested again.
    const sourceFiles = (dir: string): string[] =>
      readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
        const path = join(dir, entry.name)
        if (entry.isDirectory()) return sourceFiles(path)
        if (!entry.isFile()) return []
        return /\.tsx?$/.test(path) && !/\.test\.tsx?$/.test(path) ? [path] : []
      })

    // Matched by SPECIFIER, not by import form. The form-by-form version of
    // this guard is how `harnessIsolation.test.ts` first went wrong: it looked
    // for `from '…'` and let a bare side-effect `import '…'` — just as capable
    // of pulling the package into the first-load chunk — straight through.
    const importSpecifiers = (source: string): string[] =>
      [
        ...source.matchAll(
          /(?:\bfrom\s*|\bimport\s*|\bimport\s*\(\s*|\brequire\s*\(\s*)(['"`])([^'"`]+)\1/g
        )
      ].map((match) => match[2])

    const namesMermaid = (specifier: string): boolean =>
      specifier === 'mermaid' || specifier.startsWith('mermaid/')

    const srcDir = join(import.meta.dir, '..', '..')
    const importers = sourceFiles(srcDir)
      .filter((file) => importSpecifiers(readFileSync(file, 'utf8')).some(namesMermaid))
      .map((file) => file.slice(srcDir.length + 1))

    expect(importers).toEqual(['components/retrieval/mermaidLoader.ts'])
  })

  /**
   * Whether `marker` sits INSIDE a block guarded by the container-identity
   * check, established by brace matching rather than by counting occurrences:
   * `ChatMessage` has four such guards and only three of them are on failure
   * branches, so any count that a dropped failure guard could still satisfy is
   * not an assertion about that branch at all.
   */
  const isGuarded = (marker: string): boolean => {
    const at = SOURCE.indexOf(marker)
    expect({ marker, found: at > -1 }).toEqual({ marker, found: true })

    const guardAt = SOURCE.lastIndexOf('if (mermaidRef.current === container', at)
    if (guardAt < 0) return false

    const open = SOURCE.indexOf('{', SOURCE.indexOf(')', guardAt))
    let depth = 0
    for (let i = open; i < SOURCE.length; i++) {
      if (SOURCE[i] === '{') depth += 1
      else if (SOURCE[i] === '}') {
        depth -= 1
        // The guard's block closes here; the write is guarded only if it is
        // before that.
        if (depth === 0) return at < i
      }
    }
    return false
  }

  test.each([
    ['the renderer failed to load', '`Mermaid renderer failed to load: ${errorMessage}'],
    ['mermaid.render() rejected', '`Mermaid diagram error: ${errorMessage}'],
    ['synchronous setup threw', '`Mermaid diagram setup error: ${errorMessage}']
  ])('the write for "%s" is inside the container-identity guard', (_case, marker) => {
    // An await elapsed, so the ref may now point at a different node — and a
    // write into a detached container is invisible either way, which is why no
    // rendered assertion can see this one.
    expect({ marker, guarded: isGuarded(marker) }).toEqual({ marker, guarded: true })
  })
})
