/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

/**
 * The composer holds state and renders two interchangeable input elements, so
 * it cannot be invoked directly the way a hook-free component can, and this
 * suite has no DOM. These are therefore source-level assertions, in the style
 * of `mermaidLoadFailure.test.ts` — but written against STRUCTURE (the element
 * that carries the alert, the body of a named handler) rather than against
 * occurrence counts or the whole file, so that adding an unrelated element or
 * a third input variant cannot make them lie in either direction.
 */

const SOURCE = readFileSync(join(import.meta.dir, 'QueryComposer.tsx'), 'utf8')

/** The text of the JSX element opened at `<tag`, matched by angle depth. */
function openingTag(source: string, startIndex: number): string {
  let depth = 0
  for (let i = startIndex; i < source.length; i++) {
    if (source[i] === '<') depth++
    else if (source[i] === '>') {
      depth--
      if (depth === 0) return source.slice(startIndex, i + 1)
    }
  }
  throw new Error(`unterminated JSX element at ${startIndex}`)
}

/** Every `<Name ...>` opening tag in the source. */
function elements(name: string): string[] {
  const found: string[] = []
  const pattern = new RegExp(`<${name}[\\s>]`, 'g')
  for (const match of SOURCE.matchAll(pattern)) {
    found.push(openingTag(SOURCE, match.index!))
  }
  return found
}

/** The body of `const <name> = useCallback(` … `)`, by brace matching. */
function handlerBody(name: string): string {
  const opener = `const ${name} = useCallback(`
  const start = SOURCE.indexOf(opener)
  expect(start).toBeGreaterThan(-1)
  let depth = 0
  for (let i = start + opener.length - 1; i < SOURCE.length; i++) {
    if (SOURCE[i] === '(') depth++
    else if (SOURCE[i] === ')') {
      depth--
      if (depth === 0) return SOURCE.slice(start, i + 1)
    }
  }
  throw new Error(`unterminated handler ${name}`)
}

const NOTICE_ID = 'query-input-error'

/** The opening tag of the element carrying the notice id. */
function noticeElement(): string {
  const idAt = SOURCE.indexOf(`id="${NOTICE_ID}"`)
  expect(idAt).toBeGreaterThan(-1)
  return openingTag(SOURCE, SOURCE.lastIndexOf('<', idAt))
}

describe('query composer validation notice', () => {
  const notice = noticeElement()

  test('the notice is an alert region', () => {
    expect(notice).toContain('role="alert"')
  })

  test('the notice floats ABOVE the composer', () => {
    // Below the composer it would sit off-screen on a short viewport, under
    // the mobile keyboard, or behind the send button.
    const classes = notice.match(/className="([^"]*)"/)?.[1] ?? ''
    expect(classes.split(/\s+/)).toContain('bottom-full')
    expect(classes.split(/\s+/)).not.toContain('top-full')
    expect(classes.split(/\s+/)).toContain('absolute')
  })

  test('every input variant is wired to the notice for assistive tech', () => {
    const inputs = [...elements('Textarea'), ...elements('Input')]
    // Both variants exist and neither is left unwired — asserted per element
    // rather than by counting occurrences across the file.
    expect(inputs.length).toBeGreaterThanOrEqual(2)
    for (const input of inputs) {
      expect(input).toContain('aria-invalid={inputError ? true : undefined}')
      expect(input).toContain(`aria-describedby={inputError ? '${NOTICE_ID}' : undefined}`)
    }
  })

  test('editing does not clear the notice', () => {
    // The message has to survive editing: it names what to change, and
    // wiping it on the first keystroke leaves the user correcting blind.
    expect(handlerBody('handleChange')).not.toContain('setInputError')
  })

  test('an accepted send clears the notice with the draft', () => {
    const submit = handlerBody('handleSubmit')
    expect(submit).toContain('setInputError(\'\')')
    expect(submit).toContain('setInputValue(\'\')')
    // The clear happens on the accepted path only — the rejected path returns
    // after storing the new message.
    const rejected = submit.slice(submit.indexOf('if (error) {'), submit.indexOf('setInputError(\'\')'))
    expect(rejected).toContain('setInputError(error)')
    expect(rejected).toContain('return')
  })

  test('the notice can be dismissed explicitly', () => {
    expect(handlerBody('dismissInputError')).toContain('setInputError(\'\')')
    expect(SOURCE).toContain('onClick={dismissInputError}')
    expect(SOURCE).toContain('aria-label={t(\'retrievePanel.retrieval.dismissError\')}')
  })
})
