import { describe, expect, test } from 'bun:test'
import { readFileSync, readdirSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'WorkspaceWelcome.tsx'), 'utf8')
const customizedContentSource = readFileSync(
  join(import.meta.dir, '..', '..', 'components', 'customization', 'useCustomizedContent.ts'),
  'utf8'
)
const localesDir = join(import.meta.dir, '..', '..', 'locales')
const localeFiles = readdirSync(localesDir)
  .filter((file) => file.endsWith('.json'))
  .sort()

describe('workspace welcome branding', () => {
  test('does not render the deployment description', () => {
    expect(source).not.toContain('content.brandDescription')
    expect(customizedContentSource).not.toContain('brandDescription')
  })

  test('closes the welcome card with a copyright line', () => {
    const copyright = source.indexOf('t(\'workspace.welcome.copyright\'')
    expect(copyright).toBeGreaterThan(-1)
    // Last element of the card: nothing renders after it inside the fragment.
    expect(source.indexOf('</Button>')).toBeLessThan(copyright)
    expect(source.slice(copyright)).not.toContain('<Button')
    // The year is interpolated, never hard-coded.
    expect(source).toContain('year: new Date().getFullYear()')
  })

  test('translates the copyright line in every locale', () => {
    expect(localeFiles.length).toBeGreaterThan(0)

    for (const localeFile of localeFiles) {
      const locale = JSON.parse(readFileSync(join(localesDir, localeFile), 'utf8'))
      const copyright = locale.workspace.welcome.copyright
      expect(copyright).toBeTruthy()
      expect(copyright).toContain('{{year}}')
      expect(copyright).toContain('\u00a9')
    }
  })
})
