import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import Button from './Button'

afterEach(cleanup)

describe('Button accessible name from tooltip', () => {
  test('icon-only button with a tooltip is named by the tooltip', () => {
    renderWithProviders(
      <Button tooltip="Refresh documents" size="icon">
        <svg aria-hidden="true" />
      </Button>
    )
    expect(screen.getByRole('button', { name: 'Refresh documents' })).toBeTruthy()
  })

  test('a visible-text button keeps its text as the name, not the tooltip', () => {
    renderWithProviders(
      <Button tooltip="Save changes">Save</Button>
    )
    expect(screen.getByRole('button', { name: 'Save' })).toBeTruthy()
    expect(screen.queryAllByRole('button', { name: 'Save changes' })).toHaveLength(0)
  })

  test('an explicit aria-label wins over the tooltip', () => {
    renderWithProviders(
      <Button tooltip="Refresh documents" aria-label="Custom label" size="icon">
        <svg aria-hidden="true" />
      </Button>
    )
    expect(screen.getByRole('button', { name: 'Custom label' })).toBeTruthy()
    expect(screen.queryAllByRole('button', { name: 'Refresh documents' })).toHaveLength(0)
  })
})
