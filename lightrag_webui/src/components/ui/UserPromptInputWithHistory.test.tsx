import { afterEach, describe, expect, mock, test } from 'bun:test'
import { cleanup, fireEvent, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import UserPromptInputWithHistory from './UserPromptInputWithHistory'

afterEach(cleanup)

describe('prompt history input method composition', () => {
  test.each([{ isComposing: true }, { isComposing: false, keyCode: 229 }])(
    'candidate confirmation does not select a saved prompt with %j', (composition) => {
      const selectPrompt = mock(() => {})
      renderWithProviders(<UserPromptInputWithHistory value="中文提示" onChange={() => {}}
        history={['Saved prompt']} onSelectFromHistory={selectPrompt} />)
      const input = screen.getByRole('combobox')
      fireEvent.keyDown(input, { key: 'ArrowDown' })
      fireEvent.keyDown(input, { key: 'Enter', ...composition })
      expect(selectPrompt).toHaveBeenCalledTimes(0)
      expect((input as HTMLInputElement).value).toBe('中文提示')

      fireEvent.keyDown(input, { key: 'Enter' })
      expect(selectPrompt).toHaveBeenCalledTimes(1)
      expect(selectPrompt).toHaveBeenCalledWith('Saved prompt')
    }
  )
})

describe('prompt history combobox semantics', () => {
  test('exposes combobox/listbox/option roles and expanded state', () => {
    renderWithProviders(
      <UserPromptInputWithHistory
        value=""
        onChange={() => {}}
        history={['First prompt', 'Second prompt']}
        onSelectFromHistory={() => {}}
      />
    )
    const input = screen.getByRole('combobox')
    expect(input.getAttribute('aria-expanded')).toBe('false')
    expect(input.hasAttribute('aria-controls')).toBe(true)

    fireEvent.keyDown(input, { key: 'ArrowDown' })
    expect(input.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByRole('listbox')).toBeTruthy()
    expect(screen.getAllByRole('option')).toHaveLength(2)
    expect(input.getAttribute('aria-activedescendant')).toBe(
      screen.getAllByRole('option')[0].id
    )
  })
})
