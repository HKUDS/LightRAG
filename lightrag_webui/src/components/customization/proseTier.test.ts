import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const css = readFileSync(join(import.meta.dir, '..', '..', 'index.css'), 'utf8')
const markdown = readFileSync(join(import.meta.dir, 'CustomizedMarkdown.tsx'), 'utf8')

/**
 * The `prose` tier is registered in CSS, not in `tailwind.config.js` —
 * Tailwind v4 never reads that file here (no `@config`), so a plugin missing
 * from `index.css` simply does not exist. The failure is silent and total:
 * every `prose*` class resolves to nothing and the Markdown surfaces render
 * as unstyled text with no heading sizes, list markers or table rules, while
 * the build stays green. Hence a test on the stylesheet itself.
 */
describe('prose tier registration', () => {
  test('the typography plugin is registered in index.css', () => {
    expect(css).toContain('@plugin \'@tailwindcss/typography\';')
  })

  test('inline code is not wrapped in the plugin default backticks', () => {
    expect(css).toMatch(/\.prose code::before,\s*\.prose code::after\s*{\s*content: '';/)
  })

  test('the Markdown component actually depends on that tier', () => {
    expect(markdown).toContain('prose dark:prose-invert')
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
describe('prose colors inside a colored container', () => {
  const chat = readFileSync(
    join(import.meta.dir, '..', 'retrieval', 'ChatMessage.tsx'),
    'utf8'
  )

  test('index.css defines the inherit-color palette', () => {
    expect(css).toContain('.prose-inherit-color {')
    // The runs that were unreadable: each must resolve to the container's own
    // color rather than to a prose-tier one.
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

  test('the user message bubble uses it', () => {
    // Paired with the role color it corrects; separating them re-breaks it.
    expect(chat).toContain('\'text-primary-foreground prose-inherit-color\'')
  })
})
