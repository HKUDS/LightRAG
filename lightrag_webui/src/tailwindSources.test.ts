import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, test } from 'bun:test'

const css = readFileSync(join(import.meta.dir, 'index.css'), 'utf8')

/**
 * Tailwind's automatic source detection follows git ignore rules, so the
 * emitted stylesheet depended on where the tree sat: under a path an enclosing
 * repository ignores, it scanned more files and the build failed the
 * first-load budget. The bundle must come only from the explicit `@source`
 * list.
 */
describe('tailwind source detection', () => {
  test('automatic detection is off and the explicit sources remain', () => {
    expect(css).toMatch(/^@import 'tailwindcss' source\(none\);$/m)
    expect(css).toContain('@source \'./**/*.{ts,tsx}\';')
  })
})
