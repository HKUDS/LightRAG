/// <reference types="bun" />
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { act, cleanup, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HashRouter } from 'react-router-dom'

import { renderWithProviders } from '@/test/render'
import { captureProcessState, restoreProcessState, type ProcessStateSnapshot } from '@/test/processState'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { useBackendState } from '@/stores/state'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'
import GraphLabels from './GraphLabels'

/**
 * The selected graph label is the view's primary filter. Persisting it only to
 * the settings store means a graph view can never be shared or reloaded from a
 * link. These tests pin the two halves of the URL<->store contract: a ?label=
 * param wins over the stored value on load, and every label change is mirrored
 * back into the hash query (encoded) so the address bar is always shareable.
 */
const stores = [useSettingsStore, useGraphStore, useBackendState]
const comboboxName = 'Search node name'

const renderGraphLabels = () =>
  renderWithProviders(
    <HashRouter>
      <GraphLabels />
    </HashRouter>
  )

describe('GraphLabels URL sync', () => {
  let snapshot: ProcessStateSnapshot
  let originalHash: string

  beforeEach(() => {
    snapshot = captureProcessState(stores)
    originalHash = window.location.hash
    // Seed history so mount-time initialization and the dropdown never hit the API.
    SearchHistoryManager.clearHistory()
    SearchHistoryManager.addToHistory('Alpha')
    SearchHistoryManager.addToHistory('Beta')
    useSettingsStore.setState({ queryLabel: '*' })
  })

  afterEach(() => {
    cleanup()
    restoreProcessState(stores, snapshot)
    window.location.hash = originalHash
  })

  test('a shared ?label= link opens that graph even when the store remembers another', async () => {
    useSettingsStore.setState({ queryLabel: 'StoredLabel' })
    window.location.hash = '#/?label=SharedLabel'

    renderGraphLabels()

    await waitFor(() => {
      expect(useSettingsStore.getState().queryLabel).toBe('SharedLabel')
    })
    // The trigger shows the URL's label, and the URL is not clobbered by the store.
    expect(screen.getByRole('combobox', { name: comboboxName }).textContent).toContain('SharedLabel')
    expect(window.location.hash).toContain('label=SharedLabel')
  })

  test('a label change is written into the hash query, encoded for sharing', async () => {
    window.location.hash = '#/'

    renderGraphLabels()

    act(() => {
      useSettingsStore.getState().setQueryLabel('A B&C')
    })

    await waitFor(() => {
      expect(window.location.hash).toBe('#/?label=A+B%26C')
    })
  })

  test('choosing a label from the dropdown updates the store and the shareable URL', async () => {
    const user = userEvent.setup()
    window.location.hash = '#/'

    renderGraphLabels()

    await user.click(screen.getByRole('combobox', { name: comboboxName }))
    await user.click(await screen.findByRole('option', { name: 'Beta' }))

    await waitFor(() => {
      expect(useSettingsStore.getState().queryLabel).toBe('Beta')
      expect(window.location.hash).toContain('label=Beta')
    })
  })

  test('a label change preserves foreign query params in the hash', async () => {
    window.location.hash = '#/?foo=bar&label=X'

    renderGraphLabels()

    act(() => {
      useSettingsStore.getState().setQueryLabel('Beta')
    })

    // Only the functional setSearchParams updater keeps params GraphLabels
    // does not own; a plain {label} object would silently drop them.
    await waitFor(() => {
      expect(window.location.hash).toContain('label=Beta')
      expect(window.location.hash).toContain('foo=bar')
    })
  })
})
