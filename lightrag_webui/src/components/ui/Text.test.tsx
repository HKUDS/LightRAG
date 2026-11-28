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

  test.each([
    ['a tooltip-only Text is keyboard focusable', 'Full value', 0],
    ['a plain Text without tooltip or onClick is not focusable', undefined, -1]
  ] as const)('%s', (_name, tooltip, tabIndex) => {
    renderWithProviders(<Text text="Value" tooltip={tooltip} />)
    expect(screen.getByText('Value').tabIndex).toBe(tabIndex)
  })
})
