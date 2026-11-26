/**
 * The compact pagination strip sits in the document panel header with no
 * visible text labels, so its page input and page-size select must carry
 * accessible names; the input also needs a numeric inputmode for soft
 * keyboards. The non-compact variant keeps its visible spans but names the
 * controls the same way.
 */
import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import PaginationControls from './PaginationControls'
import { renderWithProviders } from '@/test/render'

afterEach(() => {
  cleanup()
})

const noop = () => {}

describe('PaginationControls compact', () => {
  test('page input and page-size select are named, and the input submits a page', async () => {
    const user = userEvent.setup()
    const pages: number[] = []
    renderWithProviders(
      <PaginationControls
        currentPage={2}
        totalPages={5}
        pageSize={10}
        totalCount={50}
        onPageChange={(page) => pages.push(page)}
        onPageSizeChange={noop}
        compact
      />
    )

    const pageInput = screen.getByRole('textbox', { name: 'Page' })
    expect(pageInput.getAttribute('inputmode')).toBe('numeric')
    expect(screen.queryAllByRole('combobox', { name: 'Page Size' })).toHaveLength(1)

    await user.clear(pageInput)
    await user.type(pageInput, '3{Enter}')
    expect(pages).toEqual([3])
  })
})

describe('PaginationControls non-compact', () => {
  test('page input and page-size select carry the same accessible names', () => {
    renderWithProviders(
      <PaginationControls
        currentPage={2}
        totalPages={5}
        pageSize={10}
        totalCount={50}
        onPageChange={noop}
        onPageSizeChange={noop}
      />
    )

    const pageInput = screen.getByRole('textbox', { name: 'Page' })
    expect(pageInput.getAttribute('inputmode')).toBe('numeric')
    expect(screen.queryAllByRole('combobox', { name: 'Page Size' })).toHaveLength(1)
  })
})