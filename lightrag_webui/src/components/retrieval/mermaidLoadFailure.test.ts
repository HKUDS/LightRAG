import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

/**
 * Mermaid became a DYNAMIC import so it stays out of the workspace entry's
 * first-load closure. That introduced a failure mode a static import could
 * not have: the renderer itself can fail to arrive (a deploy invalidates the
 * hashed chunk mid-session, an offline tab, a flaky network).
 *
 * The load-failure branch must therefore write something VISIBLE into the
 * container, exactly like the two render-failure branches beside it. Three
 * facts make a bare `return` there a permanently blank gap rather than a
 * transient glitch, and each is pinned below:
 *
 *  1. the container is still EMPTY at that point — the loading indicator is
 *     written only after the await;
 *  2. nothing retries — the effect's dependencies cannot change again once
 *     the message settles;
 *  3. the source is unreachable — while `renderAsDiagram` is true the render
 *     body returns an empty <div>, not the plain-text fallback.
 *
 * The component cannot be imported here (this suite has no DOM and
 * ChatMessage pulls in react-markdown/KaTeX), so these are source-level
 * assertions — the same technique serializeQueryRequest.test.ts uses for its
 * no-entry-branches guarantee.
 */

const SOURCE = readFileSync(join(import.meta.dir, 'ChatMessage.tsx'), 'utf8')

/** The body of `catch (<param>) { … }`, by brace matching. */
function catchBody(source: string, param: string): string {
  const opener = `catch (${param}) {`
  const start = source.indexOf(opener)
  expect(start).toBeGreaterThan(-1)
  let depth = 0
  for (let i = start + opener.length - 1; i < source.length; i++) {
    if (source[i] === '{') depth++
    else if (source[i] === '}') {
      depth--
      if (depth === 0) return source.slice(start + opener.length, i)
    }
  }
  throw new Error(`unbalanced braces after ${opener}`)
}

describe('mermaid dynamic-import failure', () => {
  test('the load-failure branch writes the failure into the container', () => {
    const body = catchBody(SOURCE, 'loadError')
    // Fix-proof: the previous code was `console.error(...); return;` — it
    // touched the container not at all.
    expect(body).toContain('container.appendChild')
    // Cleared first, so a half-written container cannot survive underneath.
    expect(body).toMatch(/container\.innerHTML\s*=\s*''/)
  })

  test('the load-failure branch echoes the diagram source', () => {
    const body = catchBody(SOURCE, 'loadError')
    // Without this the user sees an error with no way to recover the content:
    // the plain-text render branch is unreachable while renderAsDiagram holds.
    expect(body).toContain('Content:')
    expect(body).toContain('String(children)')
  })

  test('it guards the container identity before writing, like its siblings', () => {
    // An await elapsed; the ref may now point at a different node.
    expect(catchBody(SOURCE, 'loadError')).toContain('mermaidRef.current === container')
  })

  test('every mermaid failure path surfaces something, none returns silently', () => {
    // Consistency is the actual invariant — the load path was the only one
    // that logged and gave up. Anchored on each path's own user-visible
    // wording, because `catch (error)` is not unique in this file (the KaTeX
    // plugin loader has one too).
    const paths = [
      'Mermaid renderer failed to load:', // dynamic import failed (this fix)
      'Mermaid diagram error:', // mermaid.render() rejected
      'Mermaid diagram setup error:' // synchronous setup threw
    ]
    for (const marker of paths) {
      const at = SOURCE.indexOf(marker)
      expect(at).toBeGreaterThan(-1)
      expect(SOURCE.slice(at, at + 300)).toContain('container.appendChild')
    }
  })

  test('the loading indicator is written only AFTER the import resolves', () => {
    // This ordering is why a bare return leaves the container empty rather
    // than stuck on a spinner; if it ever moves before the await, the
    // reasoning in the branch comment stops holding.
    const awaitAt = SOURCE.indexOf('await loadMermaid()')
    const spinnerAt = SOURCE.indexOf('animate-spin h-5 w-5 text-primary')
    expect(awaitAt).toBeGreaterThan(-1)
    expect(spinnerAt).toBeGreaterThan(awaitAt)
  })

  test('the diagram container renders empty, so only JS can fill it', () => {
    // The premise of the whole fix: with renderAsDiagram true the component
    // returns a bare <div>, so nothing is shown unless the effect writes.
    expect(SOURCE).toContain(
      '<div className="mermaid-diagram-container my-4 overflow-x-auto" ref={mermaidRef}></div>'
    )
  })
})
