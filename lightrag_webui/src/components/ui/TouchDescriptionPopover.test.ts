import { describe, expect, test } from 'bun:test'
import { readFileSync, readdirSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'TouchDescriptionPopover.tsx'), 'utf8')
const localesDir = join(import.meta.dir, '..', '..', 'locales')
const localeFiles = readdirSync(localesDir)
  .filter((file) => file.endsWith('.json'))
  .sort()

describe('touch description popover', () => {
  test('provides an explicit control on devices without hover', () => {
    expect(source).toContain('<PopoverTrigger asChild>')
    expect(source).toContain('[@media(any-hover:none)]:inline-flex')
    expect(source).toContain('h-9 w-9')
    expect(source).toContain('after:-inset-1')
    expect(source).toContain('after:content-[\'\']')
    expect(source).toContain('aria-label={t(\'header.deploymentInfo\')}')
    expect(source).not.toContain('aria-label={description}')
    expect(source).toContain('<PopoverContent')
    expect(source).toContain('max-w-[calc(100vw-2rem)]')
  })

  test('provides a short accessible name in every locale', () => {
    expect(localeFiles.length).toBeGreaterThan(0)

    for (const localeFile of localeFiles) {
      const locale = JSON.parse(readFileSync(join(localesDir, localeFile), 'utf8'))
      expect(locale.header.deploymentInfo).toBeTruthy()
    }
  })
})
