import { afterAll, afterEach, beforeAll, describe, expect, mock, test } from 'bun:test'
import { act, cleanup, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import { useSettingsStore } from '@/stores/settings'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'

/**
 * GraphLabels remounts AsyncSelect (a new `key`) whenever the label changes,
 * so Radix's own focus return lands on a detached trigger. The stub stands in
 * for AsyncSelect to report how the option was chosen; GraphLabels is
 * imported AFTER the mock so it picks the stub up, and the real module is
 * restored in afterAll.
 */
let realAsyncSelect: Record<string, unknown>
let GraphLabels: typeof import('./GraphLabels').default
let selectViaKeyboard = false

beforeAll(async () => {
  realAsyncSelect = { ...(await import('@/components/ui/AsyncSelect')) }
  mock.module('@/components/ui/AsyncSelect', () => ({
    AsyncSelect: ({ ariaLabel, onChange }: {
      ariaLabel?: string
      onChange: (value: string, meta: { viaKeyboard: boolean }) => void
    }) => (
      <button type="button" role="combobox" aria-label={ariaLabel}
        onClick={() => onChange('Alpha', { viaKeyboard: selectViaKeyboard })}>
        pick
      </button>
    )
  }))
  GraphLabels = (await import('./GraphLabels')).default
})

afterAll(() => {
  mock.module('@/components/ui/AsyncSelect', () => realAsyncSelect)
})

const initialLabel = useSettingsStore.getState().queryLabel

afterEach(() => {
  cleanup()
  act(() => { useSettingsStore.getState().setQueryLabel(initialLabel) })
  SearchHistoryManager.clearHistory()
})

describe('GraphLabels focus after a label change', () => {
  test.each([
    ['keyboard', true],
    ['pointer', false]
  ] as const)('a %s selection', async (_input, viaKeyboard) => {
    // A non-empty history keeps the mount from asking the API for popular labels.
    SearchHistoryManager.addToHistory('Beta')
    selectViaKeyboard = viaKeyboard
    renderWithProviders(<GraphLabels />)
    const before = screen.getByRole('combobox', { name: 'Search node name' })
    await userEvent.setup().click(before)
    await act(async () => { await new Promise((r) => requestAnimationFrame(() => r(null))) })

    expect(useSettingsStore.getState().queryLabel).toBe('Alpha')
    const after = screen.getByRole('combobox', { name: 'Search node name' })
    // The trigger was replaced, so focus can only be on it if GraphLabels moved it there.
    expect(after === before).toBe(false)
    expect(document.activeElement === after).toBe(viaKeyboard)
  })
})
