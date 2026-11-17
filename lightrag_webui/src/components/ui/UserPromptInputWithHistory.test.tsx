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
      const input = screen.getByRole('textbox')
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
