import { afterEach, describe, expect, test } from 'bun:test'
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
  test('AsyncSearch disabled actually disables the input', () => {
    renderWithProviders(
      <AsyncSearch
        fetcher={() => Promise.resolve<string[]>([])}
        renderOption={(o: string) => <span>{o}</span>}
        getOptionValue={(o: string) => o}
        onChange={() => {}}
        onFocus={() => {}}
        value={null}
        disabled
        placeholder="Search entities"
      />
    )
    const input = screen.getByPlaceholderText('Search entities') as HTMLInputElement
    expect(input.disabled).toBe(true)
  })

  test('AsyncSearch keeps focus on the input and stays closed after selecting an option', async () => {
    renderWithProviders(
      <AsyncSearch
        fetcher={() => Promise.resolve(['Alpha'])}
        preload
        renderOption={(o: string) => <span>{o}</span>}
        getOptionValue={(o: string) => o}
        onChange={() => {}}
        onFocus={() => {}}
        value={null}
        placeholder="Search entities"
      />
    )
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
  })

  test('AsyncSearch Escape closes the open dropdown and consumes the event', async () => {
    renderWithProviders(
      <AsyncSearch
        fetcher={() => Promise.resolve(['Alpha'])}
        preload
        renderOption={(o: string) => <span>{o}</span>}
        getOptionValue={(o: string) => o}
        onChange={() => {}}
        onFocus={() => {}}
        value={null}
        placeholder="Search entities"
      />
    )
    const input = screen.getByPlaceholderText('Search entities') as HTMLInputElement
    await act(async () => { input.focus() })
    await screen.findByText('Alpha')
    const event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })
    act(() => { input.dispatchEvent(event) })
    expect(event.defaultPrevented).toBe(true)
    expect(screen.queryAllByRole('option')).toHaveLength(0)
  })

  test('AsyncSearch Escape while closed does not preventDefault', async () => {
    renderWithProviders(
      <AsyncSearch
        fetcher={() => Promise.resolve(['Alpha'])}
        preload
        renderOption={(o: string) => <span>{o}</span>}
        getOptionValue={(o: string) => o}
        onChange={() => {}}
        onFocus={() => {}}
        value={null}
        placeholder="Search entities"
      />
    )
    const input = screen.getByPlaceholderText('Search entities') as HTMLInputElement
    // Let the mount fetch settle; the dropdown must still be closed.
    await waitFor(() => expect(screen.queryAllByRole('option')).toHaveLength(0))
    // A prevented Escape here would block a parent Radix Dialog's dismissal.
    const event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })
    act(() => { input.dispatchEvent(event) })
    expect(event.defaultPrevented).toBe(false)
  })

  test('AsyncSearch reopens on click after a selection', async () => {
    renderWithProviders(
      <AsyncSearch
        fetcher={() => Promise.resolve(['Alpha'])}
        preload
        renderOption={(o: string) => <span>{o}</span>}
        getOptionValue={(o: string) => o}
        onChange={() => {}}
        onFocus={() => {}}
        value={null}
        placeholder="Search entities"
      />
    )
    const input = screen.getByPlaceholderText('Search entities') as HTMLInputElement
    await act(async () => { input.focus() })
    const option = await screen.findByText('Alpha')
    fireEvent.click(option)
    await waitFor(() => expect(screen.queryAllByRole('option')).toHaveLength(0))
    // Focus is already on the input, so only the click handler can reopen.
    fireEvent.click(input)
    expect(await screen.findByText('Alpha')).toBeTruthy()
    expect(screen.queryAllByRole('option')).toHaveLength(1)
  })

  test('AsyncSelect returns focus to the trigger after selecting an option', async () => {
    renderWithProviders(
      <AsyncSelect
        fetcher={() => Promise.resolve(['Alpha'])}
        preload
        renderOption={(o: string) => <span>{o}</span>}
        getOptionValue={(o: string) => o}
        getDisplayValue={(o: string) => o}
        onChange={() => {}}
        value=""
        ariaLabel="Pick entity"
      />
    )
    const trigger = screen.getByRole('combobox', { name: 'Pick entity' })
    await userEvent.setup().click(trigger)
    const option = await screen.findByText('Alpha')
    fireEvent.click(option)
    // Radix restores focus asynchronously on close.
    await waitFor(() => expect(document.activeElement === trigger).toBe(true))
    expect(screen.queryAllByRole('option')).toHaveLength(0)
  })

  test('AsyncSelect combobox advertises its listbox', async () => {
    renderWithProviders(
      <AsyncSelect
        fetcher={() => Promise.resolve<string[]>([])}
        renderOption={(o: string) => <span>{o}</span>}
        getOptionValue={(o: string) => o}
        getDisplayValue={(o: string) => o}
        onChange={() => {}}
        value=""
        ariaLabel="Pick entity"
      />
    )
    const combobox = screen.getByRole('combobox', { name: 'Pick entity' })
    expect(combobox.getAttribute('aria-haspopup')).toBe('listbox')
    expect(combobox.hasAttribute('aria-controls')).toBe(true)
  })
})
