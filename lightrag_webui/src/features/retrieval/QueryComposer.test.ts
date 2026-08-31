import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'QueryComposer.tsx'), 'utf8')

describe('query composer validation notice', () => {
  test('renders errors as a floating alert above the composer', () => {
    expect(source).toContain('role="alert"')
    expect(source).toContain('bottom-full')
    expect(source).not.toContain('top-full')
  })

  test('associates the active error with both input variants', () => {
    expect(source.match(/aria-invalid=/g)).toHaveLength(2)
    expect(source.match(/aria-describedby=/g)).toHaveLength(2)
    expect(source).toContain('id="query-input-error"')
  })

  test('keeps the error while editing and clears it on accepted send', () => {
    const changeStart = source.indexOf('const handleChange')
    const changeEnd = source.indexOf('// Unified height adjustment', changeStart)
    const changeHandler = source.slice(changeStart, changeEnd)

    expect(changeHandler).not.toContain('setInputError')
    expect(source).toMatch(/setInputError\(''\)\s+setInputValue\(''\)/)
  })
})
