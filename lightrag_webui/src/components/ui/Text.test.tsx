import { afterEach, describe, expect, mock, test } from 'bun:test'
import { cleanup, fireEvent, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import Text from './Text'

afterEach(cleanup)

describe('Text interactive semantics', () => {
  test('a clickable Text is a button that fires onClick', () => {
    const onClick = mock(() => {})
    renderWithProviders(<Text text="Open node" onClick={onClick} />)
    const button = screen.getByRole('button', { name: 'Open node' })
    fireEvent.click(button)
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  test('a tooltip-only Text trigger is keyboard focusable', () => {
    renderWithProviders(<Text text="Value" tooltip="Full value" />)
    const trigger = screen.getByText('Value')
    expect(trigger.tabIndex).toBe(0)
  })

  test('a plain Text without tooltip or onClick is not focusable', () => {
    renderWithProviders(<Text text="Plain" />)
    const trigger = screen.getByText('Plain')
    expect(trigger.tabIndex).toBe(-1)
  })
})
