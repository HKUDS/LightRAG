import { afterEach, describe, expect, mock, test } from 'bun:test'
import { act, cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import { AsyncSearch } from './AsyncSearch'
import { AsyncSelect } from './AsyncSelect'

afterEach(cleanup)

describe.each(['search', 'select'] as const)('Async %s results', (kind) => {
  test.each(['success', 'failure'] as const)('ignores an older %s after the latest search finishes', async (outcome) => {
    const pending = new Map<string, { resolve: (options: string[]) => void; reject: (error: Error) => void }>()
    const fetcher = (query?: string) => query
      ? new Promise<string[]>((resolve, reject) => pending.set(query, { resolve, reject }))
      : Promise.resolve<string[]>([])
    const common = {
      fetcher,
      renderOption: (option: string) => <span>{option}</span>,
      getOptionValue: (option: string) => option,
      onChange: () => {},
      placeholder: 'Search entities'
    }

    renderWithProviders(kind === 'search'
      ? <AsyncSearch {...common} value={null} onFocus={() => {}} />
      : <AsyncSelect {...common} value="" getDisplayValue={(option) => option} searchPlaceholder="Search entities" />)
    if (kind === 'select') await userEvent.setup().click(screen.getByRole('combobox'))
    const input = await screen.findByPlaceholderText('Search entities')

    fireEvent.change(input, { target: { value: 'old' } })
    await waitFor(() => expect(pending.has('old')).toBe(true))
    fireEvent.change(input, { target: { value: 'new' } })
    await waitFor(() => expect(pending.has('new')).toBe(true))
    await act(async () => pending.get('new')!.resolve(['New entity']))
    expect(await screen.findByText('New entity')).toBeTruthy()

    await act(async () => {
      if (outcome === 'success') pending.get('old')!.resolve(['Old entity'])
      else pending.get('old')!.reject(new Error('Old request failed'))
    })
    expect(screen.queryAllByText('Old entity').length).toBe(0)
    expect(screen.queryAllByText('Old request failed').length).toBe(0)
    expect(screen.queryAllByText('New entity').length).toBe(1)
  })
})

describe('Async primitive accessibility', () => {
  const searchProps = {
    fetcher: () => Promise.resolve(['Alpha']),
    preload: true,
    renderOption: (o: string) => <span>{o}</span>,
    getOptionValue: (o: string) => o,
    onChange: () => {},
    onFocus: () => {},
    value: null,
    placeholder: 'Search entities'
  }

  test('AsyncSearch disabled actually disables the input', () => {
    renderWithProviders(<AsyncSearch {...searchProps} disabled />)
    const input = screen.getByPlaceholderText('Search entities') as HTMLInputElement
    expect(input.disabled).toBe(true)
  })

  test.each(['click', 'ArrowDown'] as const)('AsyncSearch keeps focus on the input after a selection and reopens on %s', async (reopen) => {
    renderWithProviders(<AsyncSearch {...searchProps} />)
    const input = screen.getByPlaceholderText('Search entities') as HTMLInputElement
    // Real focus() so document.activeElement moves (fireEvent.focus does not
    // set it in happy-dom).
    await act(async () => { input.focus() })
    const option = await screen.findByText('Alpha')
    // Leave the input so selection exercises the focus-return path: focus()
    // inside handleSelect re-fires handleFocus, and the close must still win.
    input.blur()
    fireEvent.click(option)
    expect(document.activeElement === input).toBe(true)
    await waitFor(() => expect(screen.queryAllByRole('option')).toHaveLength(0))
    // Focus is already on the input, so only the click or key handler can reopen.
    if (reopen === 'click') fireEvent.click(input)
    else fireEvent.keyDown(input, { key: 'ArrowDown' })
    expect(await screen.findByText('Alpha')).toBeTruthy()
    expect(screen.queryAllByRole('option')).toHaveLength(1)
  })

  test.each([
    ['open', false, true],
    ['open', true, false],
    ['closed', false, false]
  ] as const)('AsyncSearch Escape while %s (IME composing: %p): consumed only when it closes the dropdown', async (state, composing, consumed) => {
    renderWithProviders(<AsyncSearch {...searchProps} />)
    const input = screen.getByPlaceholderText('Search entities') as HTMLInputElement
    if (state === 'open') {
      await act(async () => { input.focus() })
      await screen.findByText('Alpha')
    } else {
      // Let the mount fetch settle; the dropdown must still be closed.
      await waitFor(() => expect(screen.queryAllByRole('option')).toHaveLength(0))
    }
    // A prevented Escape while closed would block a parent Radix Dialog's dismissal.
    // An Escape that dismisses an IME candidate window belongs to the IME.
    const event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true, isComposing: composing })
    act(() => { input.dispatchEvent(event) })
    expect(event.defaultPrevented).toBe(consumed)
    expect(screen.queryAllByRole('option')).toHaveLength(state === 'open' && composing ? 1 : 0)
  })

  // 'Alpha ' differs from 'Alpha' only by whitespace, which cmdk trims from
  // item values: the keyboard case picks it to prove the two stay distinct.
  test.each([
    ['keyboard', true, 'Alpha '],
    ['pointer', false, 'Alpha']
  ] as const)('AsyncSelect after a %s selection', async (input, viaKeyboard, picked) => {
    const onChange = mock<(value: string, meta: { viaKeyboard: boolean }) => void>(() => {})
    renderWithProviders(
      <AsyncSelect
        fetcher={() => Promise.resolve(['*', 'Alpha', 'Alpha '])}
        preload
        renderOption={(o: string) => <span>{o}</span>}
        getOptionValue={(o: string) => o}
        getDisplayValue={(o: string) => o}
        onChange={onChange}
        value=""
        ariaLabel="Pick entity"
        triggerTooltip="Choose a label"
      />
    )
    const trigger = screen.getByRole('combobox', { name: 'Pick entity' })
    const user = userEvent.setup()
    await user.click(trigger)
    const option = (await screen.findAllByText('Alpha'))[0]
    // Radix wires the trigger to the popover (a dialog holding the search
    // input and the listbox); a caller-set id would be overwritten by cmdk.
    expect(trigger.getAttribute('aria-haspopup')).toBe('dialog')
    const popup = document.getElementById(trigger.getAttribute('aria-controls')!)
    expect(popup?.contains(screen.getByRole('listbox'))).toBe(true)
    if (input === 'keyboard') {
      // Clicking into the search box first: the key presses after it must
      // still count as a keyboard selection. The search stays empty, so the
      // arrow keys must reach the options without a typed term ('*' starts
      // highlighted).
      await user.click(screen.getByPlaceholderText('Search...'))
      await user.keyboard('{ArrowDown}{ArrowDown}{Enter}')
    } else {
      await user.click(option)
    }
    expect(onChange.mock.calls).toEqual([[picked, { viaKeyboard }]])
    // Radix returns focus from a setTimeout(0) queued during the close;
    // one zero-delay macrotask queued after it is enough (no fixed sleep).
    await act(async () => { await new Promise((r) => setTimeout(r, 0)) })
    expect(screen.queryAllByRole('option')).toHaveLength(0)
    // Keyboard: focus is back on the trigger. Pointer: it is not, so the
    // trigger's tooltip does not pop up under the mouse user.
    expect(document.activeElement === trigger).toBe(viaKeyboard)
    expect(screen.queryAllByRole('tooltip').length > 0).toBe(viaKeyboard)
  })
})
