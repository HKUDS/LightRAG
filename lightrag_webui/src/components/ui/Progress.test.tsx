import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import Progress from './Progress'

afterEach(cleanup)

describe('Progress value semantics', () => {
  test('forwards value to the progressbar as aria-valuenow', () => {
    renderWithProviders(<Progress value={40} />)
    const bar = screen.getByRole('progressbar')
    expect(bar.getAttribute('aria-valuenow')).toBe('40')
  })

  test('no value stays indeterminate', () => {
    renderWithProviders(<Progress />)
    const bar = screen.getByRole('progressbar')
    expect(bar.hasAttribute('aria-valuenow')).toBe(false)
  })
})
