import { describe, expect, test } from 'bun:test'
import { readFileSync, readdirSync } from 'fs'
import { join } from 'path'
import { act, cleanup, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './Tooltip'
import Text from './Text'

const VIEWPORT_CLAMP = 'max-w-[calc(100vw-2rem)]'

const viewportAwareWidths = [
  'max-w-[min(42rem,calc(100vw-2rem))]',
  'max-w-[min(24rem,calc(100vw-2rem))]',
  'max-w-[min(20rem,calc(100vw-2rem))]'
]

/**
 * Radix keeps the content mounted only while the tooltip is open, and hovering
 * in happy-dom would depend on pointer timers. `open` puts it straight into the
 * portal, which is the state the width matters in.
 */
const openTooltip = async (className?: string): Promise<void> => {
  renderWithProviders(
    <TooltipProvider>
      <Tooltip open>
        <TooltipTrigger>trigger</TooltipTrigger>
        <TooltipContent className={className}>Tooltip body</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
  // Radix's popper positions itself in an effect; flushing it here keeps the
  // act() warning out of the run.
  await act(async () => {})
}

const maxWidthClasses = (element: Element): string[] =>
  (element.getAttribute('class') ?? '').split(/\s+/).filter((token) => token.startsWith('max-w-'))

describe('tooltip viewport width constraints', () => {
  test('a tooltip with no custom width is clamped to the viewport', async () => {
    await openTooltip()

    expect(maxWidthClasses(screen.getByRole('tooltip'))).toEqual([VIEWPORT_CLAMP])
  })

  test('a viewport-aware custom width replaces the default clamp and keeps clamping', async () => {
    for (const width of viewportAwareWidths) {
      await openTooltip(width)

      // Exactly one surviving `max-w-`: tailwind-merge drops the default rather
      // than emitting both, so the caller's value is the one that applies — and
      // it carries the clamp inside itself.
      const widths = maxWidthClasses(screen.getByRole('tooltip'))
      expect(widths).toEqual([width])
      expect(widths[0]).toContain('100vw')

      // The preload's `cleanup()` only runs between tests; this loop mounts
      // several tooltips within one.
      cleanup()
    }
  })

  test('a desktop-only custom width silently drops the viewport clamp', async () => {
    // This is the hazard the convention exists for, and the reason a call site
    // may not write a bare `max-w-[42rem]`: tailwind-merge REPLACES the default,
    // so the tooltip stops fitting a narrow screen with no visible warning.
    await openTooltip('max-w-[42rem]')

    const widths = maxWidthClasses(screen.getByRole('tooltip'))
    expect(widths).toEqual(['max-w-[42rem]'])
    expect(widths.some((width) => width.includes('100vw'))).toBe(false)
  })

  test('Text forwards tooltipClassName onto the rendered tooltip', async () => {
    // PropertiesView and PropertyRowComponents set their widths through this
    // prop. If the forwarding broke, their tooltips would fall back to the
    // default clamp at a width nobody chose, and the call-site audit below
    // would still pass — it only reads what the call site WRITES.
    const user = userEvent.setup()
    renderWithProviders(
      <Text text="Entity" tooltip="Entity description" tooltipClassName={viewportAwareWidths[1]} />
    )

    await user.hover(screen.getByText('Entity'))
    await waitFor(() => expect(screen.queryAllByRole('tooltip')).toHaveLength(1))
    await act(async () => {})

    expect(maxWidthClasses(screen.getByRole('tooltip'))).toEqual([viewportAwareWidths[1]])
  })
})

const sourceFiles = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) return sourceFiles(path)
    return entry.isFile() && path.endsWith('.tsx') && !path.endsWith('.test.tsx') ? [path] : []
  })

/**
 * The text of the JSX tag opened at `start`, up to the `>` that closes it.
 * Braces and strings are skipped, so an attribute whose value is an EXPRESSION
 * — `className={cn('max-w-[42rem]', extra)}`, which is valid, lint-clean and
 * loses the clamp through tailwind-merge — is part of the returned text rather
 * than ending the scan early.
 */
const openingTag = (source: string, start: number): string => {
  let depth = 0
  let quote: string | null = null

  for (let i = start; i < source.length; i++) {
    const char = source[i]
    if (quote) {
      if (char === quote) quote = null
      continue
    }
    if (char === '\'' || char === '"' || char === '`') quote = char
    else if (char === '{') depth += 1
    else if (char === '}') depth -= 1
    else if (char === '>' && depth === 0) return source.slice(start, i + 1)
  }

  throw new Error(`unterminated JSX tag at ${start}`)
}

/**
 * The module-scope string constants a JSX attribute refers to by NAME, so
 * `const TOOLTIP_WIDTH = 'max-w-lg'` used as `className={TOOLTIP_WIDTH}` is
 * audited like an inline literal.
 *
 * Deliberately bounded: same file, `const X = '…'` only. A width imported from
 * another module or built at runtime is still out of reach, and no text scan
 * closes that without becoming a type checker — which is why the RENDERED half
 * of this file, not this one, is what proves the clamp actually applies.
 */
const resolvedConstants = (source: string, site: string): string => {
  const constants = new Map<string, string>()
  for (const match of source.matchAll(
    /(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=\n]+)?=\s*(['"`])([^'"`\n]*)\2/g
  )) {
    constants.set(match[1], match[3])
  }

  const referenced = [...site.matchAll(/[A-Za-z_$][\w$]*/g)]
    .map((match) => constants.get(match[0]))
    .filter((value): value is string => value !== undefined)

  return referenced.length > 0 ? ` ${referenced.join(' ')}` : ''
}

/** The attribute value starting at `at`, whether `"quoted"` or `{braced}`. */
const attributeValue = (source: string, at: number): string => {
  const opener = source[at]

  if (opener === '\'' || opener === '"') {
    const close = source.indexOf(opener, at + 1)
    return close < 0 ? '' : source.slice(at + 1, close)
  }

  if (opener !== '{') return ''

  let depth = 0
  for (let i = at; i < source.length; i++) {
    if (source[i] === '{') depth += 1
    else if (source[i] === '}') {
      depth -= 1
      if (depth === 0) return source.slice(at + 1, i)
    }
  }
  return ''
}

const srcDir = join(import.meta.dir, '..', '..')

/**
 * Where a custom width is WRITTEN is a source-level property — no single render
 * can enumerate every call site — so this half stays a source scan. It replaces
 * three hard-coded file/string pairs: a new call site is now covered the moment
 * it is added, instead of being invisible to the test.
 */
const customTooltipWidths = (): { file: string; width: string }[] => {
  const found: { file: string; width: string }[] = []

  for (const file of sourceFiles(srcDir)) {
    const source = readFileSync(file, 'utf8')
    const sites: string[] = []

    for (const match of source.matchAll(/<TooltipContent\b/g)) {
      sites.push(openingTag(source, match.index))
    }
    for (const match of source.matchAll(/\btooltipClassName\s*=\s*/g)) {
      sites.push(attributeValue(source, match.index + match[0].length))
    }

    for (const site of sites) {
      // Both syntaxes: the bracket form FIRST so an arbitrary value is taken
      // whole, then the standard scale. `max-w-lg`, `max-w-96` and
      // `max-w-none` drop the clamp through tailwind-merge exactly as
      // `max-w-[42rem]` does, and a scan that only knows brackets would call
      // such a call site covered while it is not.
      const text = site + resolvedConstants(source, site)

      for (const width of text.match(/max-w-\[[^\]]*\]|max-w-[\w./-]+/g) ?? []) {
        found.push({ file: file.slice(srcDir.length + 1), width })
      }
    }
  }

  return found
}

describe('tooltip width call sites', () => {
  /**
   * The widths the deleted test named, per file. Kept as an exact expectation
   * rather than a count: a site that DROPS its width is the regression that
   * matters most — the tooltip silently falls back to the near-viewport-wide
   * default — and it is invisible to any "at least N, none of them bad" check,
   * because the three DocumentManager widths alone satisfy that.
   */
  const REQUIRED_WIDTHS: Record<string, string[]> = {
    'features/DocumentManager.tsx': [
      'max-w-[min(42rem,calc(100vw-2rem))]',
      'max-w-[min(42rem,calc(100vw-2rem))]',
      'max-w-[min(42rem,calc(100vw-2rem))]'
    ],
    'components/graph/PropertiesView.tsx': ['max-w-[min(24rem,calc(100vw-2rem))]'],
    'components/graph/PropertyRowComponents.tsx': ['max-w-[min(20rem,calc(100vw-2rem))]']
  }

  test('each known call site still declares its own width', () => {
    const widths = customTooltipWidths()

    for (const [file, expected] of Object.entries(REQUIRED_WIDTHS)) {
      const found = widths.filter((site) => site.file === file).map((site) => site.width)
      expect({ file, found }).toEqual({ file, found: expected })
    }
  })

  test('every custom tooltip width is viewport-aware, including new ones', () => {
    // The half the old test could not do: this reaches call sites nobody has
    // written yet, so a new one is covered the moment it is added.
    const widths = customTooltipWidths()

    // Guards the scan itself: a regex that stopped matching would otherwise
    // make this test pass by finding nothing.
    expect(widths.length).toBeGreaterThanOrEqual(5)

    const desktopOnly = widths.filter(({ width }) => !width.includes('100vw'))
    expect(desktopOnly).toEqual([])
  })
})
