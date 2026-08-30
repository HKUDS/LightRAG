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
