/// <reference types="bun" />
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import type { ComponentType } from 'react'
import { act, cleanup } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import type { MessageWithError } from '@/types/retrieval'

/**
 * Mermaid became a DYNAMIC import so it stays out of the workspace entry's
 * first-load closure. That introduced a failure mode a static import could
 * not have: the renderer itself can fail to arrive (a deploy invalidates the
 * hashed chunk mid-session, an offline tab, a flaky network).
 *
 * The load-failure branch must therefore write something VISIBLE into the
 * container, exactly like the two render-failure branches beside it — a bare
 * `return` there is a permanently blank gap, not a transient glitch: the
 * container is still empty at that point, nothing retries once the message
 * settles, and the plain-text fallback in the render body only runs while
 * `renderAsDiagram` is false.
 *
 * ORDERING NOTE: the load-failure mock must be registered before anything in
 * the process has successfully imported `mermaid`, because Bun caches the
 * failed module record — which is why it is installed in `beforeAll` and its
 * tests run first, with the working stubs swapped in afterwards. If another
 * file ever imports mermaid first, that `mock.module` throws here rather than
 * passing quietly.
 */

const DIAGRAM = 'graph TD;\n  A-->B;'
const LOAD_ERROR = 'Loading chunk mermaid-a1b2c3 failed'

let ChatMessage: ComponentType<{ message: MessageWithError }>
let consoleErrors: unknown[][]
let originalConsoleError: typeof console.error

beforeAll(async () => {
  // A rejected dynamic import: the renderer chunk never arrives.
  mock.module('mermaid', () => {
    throw new Error(LOAD_ERROR)
  })
  ChatMessage = (await import('./ChatMessage')).ChatMessage
})

afterAll(() => {
  mock.module('mermaid', () => ({
    default: { initialize: () => {}, render: async () => ({ svg: '' }) }
  }))
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
  // One render, one dynamic import: Bun keeps a failed module record, so a
  // second import of the rejecting mock does not fail the same way. Everything
  // this branch owes the user is therefore asserted from a single mount.
  test('the container starts empty and ends up carrying the failure', async () => {
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

    await settleDiagram()

    const text = container().textContent ?? ''
    expect(text).toContain('Mermaid renderer failed to load')
    expect(text).toContain(LOAD_ERROR)
    // Without the echo the user sees an error with no way to recover the
    // content: the plain-text branch is unreachable while renderAsDiagram holds.
    expect(text).toContain('Content:')
    expect(text).toContain('A-->B;')
    // And it is the ONLY thing in the container — written after a clear, so a
    // half-rendered diagram cannot survive underneath it.
    expect(container().children).toHaveLength(1)

    // Logged for the operator as well as shown to the user.
    expect(
      consoleErrors.some((args) => String(args[0]).includes('Failed to load mermaid'))
    ).toBe(true)
  })
})

describe('mermaid renderer loads but the diagram does not', () => {
  test('a rejected render surfaces the reason and the source', async () => {
    mock.module('mermaid', () => ({
      default: {
        initialize: () => {},
        render: () => Promise.reject(new Error('Parse error on line 2'))
      }
    }))

    renderDiagram()
    await settleDiagram()

    const text = container().textContent ?? ''
    expect(text).toContain('Mermaid diagram error')
    expect(text).toContain('Parse error on line 2')
    expect(text).toContain('A-->B;')
  })

  test('a throwing setup surfaces the reason', async () => {
    mock.module('mermaid', () => ({
      default: {
        initialize: () => {
          throw new Error('unsupported theme')
        },
        render: async () => ({ svg: '' })
      }
    }))

    renderDiagram()
    await settleDiagram()

    // The third failure path. Consistency is the actual invariant here: the
    // load path was once the only one that logged and gave up silently.
    expect(container().textContent ?? '').toContain('unsupported theme')
  })

  test('a successful render replaces the container with the diagram', async () => {
    mock.module('mermaid', () => ({
      default: {
        initialize: () => {},
        render: async () => ({ svg: '<svg data-testid="diagram"><title>A to B</title></svg>' })
      }
    }))

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

  test('every failure branch re-checks the container identity before writing', () => {
    // An await elapsed, so the ref may now point at a different node — and a
    // render into a detached container is invisible either way, which is
    // exactly why no rendered assertion can see this one.
    const guards = SOURCE.match(/mermaidRef\.current === container/g) ?? []
    expect(guards.length).toBeGreaterThanOrEqual(3)
  })
})
