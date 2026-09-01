/// <reference types="bun" />
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import type { ComponentType } from 'react'
import { act, cleanup } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import { relativePath, sourceFiles } from '@/test/sourceScan'
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

/**
 * A loader parked mid-flight, so a test can observe the window between the
 * effect calling it and the module arriving. `settle` hands over the mermaid
 * stub the component then uses.
 */
const pendingLoader = (): { loader: () => Promise<unknown>; settle: (mermaid: Record<string, unknown>) => void } => {
  let release: (mermaid: Record<string, unknown>) => void = () => {}
  const pending = new Promise<unknown>((resolve) => {
    release = resolve
  })
  return { loader: () => pending, settle: release }
}

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

  test('the container stays empty until the renderer actually arrives', async () => {
    // The premise of the whole fix, and the reason a bare `return` in the
    // load-failure branch is a permanent blank gap rather than a glitch:
    // NOTHING has been put in the container by the time the load can fail.
    // With `renderAsDiagram` true the render body returns an empty div, and
    // the effect writes its loading indicator only AFTER `await loadMermaid()`
    // resolves — so the failure branch has no spinner to replace and an empty
    // container to leave behind.
    //
    // Waiting a fixed 50 ms would assert none of that: 50 ms is inside the
    // effect's own 300 ms debounce, so it observes a component that has not
    // started, and moving the loading write to before the await would keep it
    // green. Park the LOADER instead and wait well past the debounce, so what
    // is being observed is the effect suspended on the import.
    const { loader, settle } = pendingLoader()
    stubLoader(loader)

    renderDiagram()
    await settleDiagram()

    expect(container().innerHTML).toBe('')

    // And the emptiness above is a real window rather than a component that
    // never ran: releasing the same loader fills the container.
    await act(async () => {
      settle({
        initialize: () => {},
        render: async () => ({ svg: '<svg data-testid="late"></svg>' })
      })
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(container().innerHTML).toContain('data-testid="late"')
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
    /**
     * Comments removed before scanning, string literals respected so a URL's
     * `//` is not mistaken for one. Comments can sit ANYWHERE inside an import
     * — `import(/* webpackChunkName: "mermaid" *\/ 'mermaid')` is the common
     * bundler idiom — and allowing for them position by position inside the
     * pattern would be another form list. Normalising the input instead
     * removes the whole class.
     */
    const withoutComments = (source: string): string => {
      let out = ''
      let quote: string | null = null

      for (let i = 0; i < source.length; i++) {
        const char = source[i]

        if (quote) {
          out += char
          if (char === '\\') {
            out += source[i + 1] ?? ''
            i += 1
          } else if (char === quote) {
            quote = null
          }
          continue
        }

        if (char === '\'' || char === '"' || char === '`') {
          quote = char
          out += char
          continue
        }

        if (char === '/' && source[i + 1] === '/') {
          const end = source.indexOf('\n', i)
          i = end < 0 ? source.length : end - 1
          continue
        }

        if (char === '/' && source[i + 1] === '*') {
          const end = source.indexOf('*/', i + 2)
          i = end < 0 ? source.length : end + 1
          // Keep a separator so `import/*x*/(` does not become `import(`
          // glued to whatever preceded it.
          out += ' '
          continue
        }

        out += char
      }

      return out
    }

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

    // Paths come back `/`-separated on every platform (`@/test/sourceScan`),
    // so the expectation below stays a literal instead of quietly failing on
    // Windows, where `path.join` would report `components\\retrieval\\...`.
    const srcDir = join(import.meta.dir, '..', '..')
    const importers = sourceFiles(
      srcDir,
      (path) => /\.tsx?$/.test(path) && !/\.test\.tsx?$/.test(path)
    )
      .filter((file) => importSpecifiers(withoutComments(readFileSync(file, 'utf8'))).some(namesMermaid))
      .map((file) => relativePath(srcDir, file))

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
