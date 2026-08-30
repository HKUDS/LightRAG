import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'TouchDescriptionPopover.tsx'), 'utf8')

describe('touch description popover', () => {
  test('provides an explicit control on devices without hover', () => {
    expect(source).toContain('<PopoverTrigger asChild>')
    expect(source).toContain('[@media(any-hover:none)]:inline-flex')
    expect(source).toContain('aria-label={description}')
    expect(source).toContain('<PopoverContent')
    expect(source).toContain('max-w-[calc(100vw-2rem)]')
  })
})
