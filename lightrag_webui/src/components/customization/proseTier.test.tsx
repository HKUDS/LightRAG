import { afterEach, beforeAll, describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import type { ComponentType } from 'react'
import { act, cleanup, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import CustomizedMarkdown from './CustomizedMarkdown'
import type { MessageWithError } from '@/types/retrieval'

const css = readFileSync(join(import.meta.dir, '..', '..', 'index.css'), 'utf8')

/**
 * The `prose` tier is registered in CSS, not in `tailwind.config.js` —
 * Tailwind v4 never reads that file here (no `@config`), so a plugin missing
 * from `index.css` simply does not exist. The failure is silent and total:
 * every `prose*` class resolves to nothing and the Markdown surfaces render
 * as unstyled text with no heading sizes, list markers or table rules, while
 * the build stays green. Hence a test on the stylesheet itself — no render
 * can see it, because the test DOM has no stylesheet at all.
 */
describe('prose tier registration', () => {
  test('the typography plugin is registered in index.css', () => {
    expect(css).toContain('@plugin \'@tailwindcss/typography\';')
  })

  test('inline code is not wrapped in the plugin default backticks', () => {
    expect(css).toMatch(/\.prose code::before,\s*\.prose code::after\s*{\s*content: '';/)
  })

  test('index.css defines the inherit-color palette', () => {
    expect(css).toContain('.prose-inherit-color {')
    // The runs that were unreadable on a colored bubble: each must resolve to
    // the container's own color rather than to a prose-tier one.
    for (const token of [
      '--tw-prose-body: currentColor',
      '--tw-prose-headings: currentColor',
      '--tw-prose-links: currentColor',
      '--tw-prose-bold: currentColor',
      '--tw-prose-code: currentColor',
      '--tw-prose-quotes: currentColor'
    ]) {
      expect(css).toContain(token)
    }
  })
})

const classesOf = (element: Element | null | undefined): string[] =>
  (element?.getAttribute('class') ?? '').split(/\s+/).filter(Boolean)

afterEach(() => {
  cleanup()
})

describe('surfaces that depend on the prose tier', () => {
  test('the customization Markdown block renders inside a prose container', () => {
    renderWithProviders(<CustomizedMarkdown content={'# Heading\n\ntext'} />)

    const heading = screen.getByRole('heading', { name: 'Heading' })
    const prose = heading.closest('.prose')

    // Asserted from the rendered heading upwards, so a wrapper that carried
    // the classes but no longer contained the content would fail.
    expect(prose === null).toBe(false)
    expect(classesOf(prose)).toContain('dark:prose-invert')
  })
})

/**
 * Registering the plugin gives every `.prose` surface the prose PALETTE too,
 * and that palette was chosen against the page background. A chat bubble sets
 * its own text color by role, so on the user's `bg-primary` bubble the prose
 * colors render headings, bold, links, code and quotes near-black in light
 * mode and near-white in dark — invisible either way. `.prose-inherit-color`
 * points the palette back at `currentColor` for that container.
 */
describe('prose colors inside a colored chat bubble', () => {
  let ChatMessage: ComponentType<{ message: MessageWithError }>

  beforeAll(async () => {
    ChatMessage = (await import('../retrieval/ChatMessage')).ChatMessage
    // The component loads KaTeX through a dynamic import on mount and stores
    // the plugin in state. A cold import currently still settles inside the
    // `act` flush below, but that is a property of how fast this module
    // resolves rather than a guarantee: warming the cache here makes the
    // component's import a cache hit, so the flush cannot start missing it
    // when the module graph grows or the run order changes.
    await import('rehype-katex')
  })

  const bubbleOf = async (role: 'user' | 'assistant', text: string): Promise<Element | null> => {
    // Rendering inside `act` so the KaTeX state update lands under it rather
    // than after the test has returned (the React act(...) warning).
    await act(async () => {
      renderWithProviders(
        <ChatMessage message={{ id: 'm1', role, content: text } as MessageWithError} />
      )
    })
    return screen.getByText(text).closest('.prose')
  }

  test('the user bubble carries the correction on the element that sets the color', async () => {
    const bubble = classesOf(await bubbleOf('user', 'my question'))

    // The pairing IS the fix: the correction has to land on the same element
    // as the role color it corrects, which is the part a source substring
    // cannot check.
    expect(bubble).toContain('text-primary-foreground')
    expect(bubble).toContain('prose-inherit-color')
  })

  test('the assistant bubble keeps the normal prose palette', async () => {
    const bubble = classesOf(await bubbleOf('assistant', 'my answer'))

    // It renders on the page background the palette was chosen against, so
    // applying the correction here would flatten every heading, link and code
    // run to the body color for no reason.
    expect(bubble).not.toContain('prose-inherit-color')
    expect(bubble).not.toContain('text-primary-foreground')
  })
})
