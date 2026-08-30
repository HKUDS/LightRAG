import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import { cn } from '@/lib/utils'

const documentManagerSource = readFileSync(
  join(import.meta.dir, '..', '..', 'features', 'DocumentManager.tsx'),
  'utf8'
)
const propertiesViewSource = readFileSync(
  join(import.meta.dir, '..', 'graph', 'PropertiesView.tsx'),
  'utf8'
)
const propertyRowsSource = readFileSync(
  join(import.meta.dir, '..', 'graph', 'PropertyRowComponents.tsx'),
  'utf8'
)

const viewportAwareWidths = [
  'max-w-[min(42rem,calc(100vw-2rem))]',
  'max-w-[min(24rem,calc(100vw-2rem))]',
  'max-w-[min(20rem,calc(100vw-2rem))]'
]

describe('tooltip viewport width constraints', () => {
  test('preserves the viewport clamp when a caller requests a desktop maximum', () => {
    for (const width of viewportAwareWidths) {
      expect(cn('max-w-[calc(100vw-2rem)]', width)).toBe(width)
    }
  })

  test('uses viewport-aware widths at every custom-width call site', () => {
    expect(documentManagerSource.match(/max-w-\[min\(42rem,calc\(100vw-2rem\)\)\]/g)).toHaveLength(3)
    expect(propertiesViewSource).toContain('max-w-[min(24rem,calc(100vw-2rem))]')
    expect(propertyRowsSource).toContain('max-w-[min(20rem,calc(100vw-2rem))]')
  })
})
