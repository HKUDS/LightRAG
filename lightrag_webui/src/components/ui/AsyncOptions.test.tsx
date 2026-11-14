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
