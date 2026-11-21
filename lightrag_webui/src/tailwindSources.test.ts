import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, test } from 'bun:test'

const css = readFileSync(join(import.meta.dir, 'index.css'), 'utf8')

const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** A whole-line CSS directive `<name> '<quoted>'<rest>;`. Either quote style
 *  is the same CSS, so both are accepted. */
const directiveLine = (name: string, quoted: string, rest = '') =>
  new RegExp(
    `^${name}\\s+(['"])${escapeRegExp(quoted)}\\1${escapeRegExp(rest).replace(/ /g, '\\s+')}\\s*;\\s*$`,
    'm'
  )

/**
 * Tailwind's automatic source detection follows git ignore rules, so the
 * emitted stylesheet depended on where the tree sat: under a path an enclosing
 * repository ignores, it scanned more files and the build failed the
 * first-load budget. The bundle must come only from the explicit `@source`
 * list.
 */
describe('tailwind source detection', () => {
  test('automatic detection is off and every explicit source remains', () => {
    expect(css).toMatch(directiveLine('@import', 'tailwindcss', ' source(none)'))
    expect(css).toMatch(directiveLine('@source', '../index.html'))
    expect(css).toMatch(directiveLine('@source', './**/*.{ts,tsx}'))
  })
})
