/// <reference types="bun" />
import { afterEach, describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import { act, cleanup } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import RetrievalView from './RetrievalView'
import StatusIndicator from '@/components/status/StatusIndicator'
import { CardContent } from '@/components/ui/Card'

/**
 * The admin retrieval page's bottom clearance is a NUMBER agreed with two
 * other components: it matches the document panel's own bottom gap (so the
 * two tabs line up when you switch between them) and it has to stay above the
 * floating status indicator. Four independent `toContain` assertions could all
 * pass while the numbers disagreed, so the classes are read off the rendered
 * elements and the agreement is checked as arithmetic.
 *
 * happy-dom computes no layout, so this can only ever be about the spacing
 * classes that land on the elements — never about measured geometry.
 */

/** Tailwind's default spacing scale: one unit is 0.25rem. */
const SPACING_PX = 4

/** The single `<prefix>-N` class on an element, as pixels. */
const spacingPx = (element: Element, prefix: string): number => {
  const classes = (element.getAttribute('class') ?? '').split(/\s+/).filter(Boolean)
  const matches = classes.filter((token) => new RegExp(`^${prefix}-\\d+$`).test(token))
  expect({ prefix, matches }).toEqual({ prefix, matches: [matches[0]] })
  return Number(matches[0].slice(prefix.length + 1)) * SPACING_PX
}

afterEach(() => {
  cleanup()
})

describe('admin retrieval layout', () => {
  test('the page bottom gap matches the document panel and clears the status indicator', async () => {
    const { container } = renderWithProviders(<RetrievalView />)
    await act(async () => {})
    const page = container.firstElementChild!

    // Nothing else may set a bottom padding. Both halves matter: a VARIANT
    // PREFIX (`md:pb-12`) and an ARBITRARY VALUE (`md:pb-[3rem]`) each change
    // the desktop clearance while leaving the unprefixed numeric class — and
    // therefore the arithmetic below — looking correct. So the prefix is
    // stripped at the last `:` (which also covers `[&_p]:` forms) and the
    // utility is matched by NAME, with no assumption about the value's shape.
    // The deleted test's `not.toContain('pb-12')` is what used to catch this.
    const setsBottomPadding = (token: string): boolean =>
      /^(?:p|py|pb)-/.test(token.slice(token.lastIndexOf(':') + 1))

    const pageClasses = (page.getAttribute('class') ?? '').split(/\s+/)
    expect(pageClasses.filter(setsBottomPadding)).toEqual(['pb-8'])
    const pageBottomPx = spacingPx(page, 'pb')
    cleanup()

    // The document panel's clearance is its card's own bottom margin plus the
    // card's content padding. The card is rendered here directly: mounting
    // DocumentManager would pull in the whole document pipeline for a number.
    const { container: cardContainer } = renderWithProviders(<CardContent />)
    const contentPaddingPx = spacingPx(cardContainer.firstElementChild!, 'p')
    cleanup()

    // DocumentManager itself stays a source read for the same reason — but of
    // the NUMBER, so a change to it fails the arithmetic rather than a string.
    const documentManager = readFileSync(join(import.meta.dir, 'DocumentManager.tsx'), 'utf8')
    const listCard = documentManager.match(/<Card className="[^"]*\bmin-h-0 mb-(\d+)\b[^"]*">/)
    expect(listCard === null).toBe(false)
    const listCardMarginPx = Number(listCard![1]) * SPACING_PX

    const { container: statusContainer } = renderWithProviders(<StatusIndicator />)
    const statusInsetPx = spacingPx(statusContainer.firstElementChild!, 'bottom')
    // Unmounted straight away: its health-change animation timers would
    // otherwise fire outside act() once this test returns.
    cleanup()

    // Switching between the two tabs must not shift the content up or down.
    expect(pageBottomPx).toBe(contentPaddingPx + listCardMarginPx)

    // And the answer must not run underneath the floating indicator.
    expect(pageBottomPx).toBeGreaterThan(statusInsetPx)
  })
})
