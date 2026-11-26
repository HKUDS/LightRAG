import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import { Command, CommandInput } from './Command'

afterEach(cleanup)

describe('CommandInput focus indicator', () => {
  test('keeps a visible focus-visible ring despite outline-none', () => {
    renderWithProviders(
      <Command>
        <CommandInput placeholder="Search options" />
      </Command>
    )
    const input = screen.getByPlaceholderText('Search options')
    expect(input.className).toContain('focus-visible:ring-1')
  })
})
