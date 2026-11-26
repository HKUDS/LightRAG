/**
 * The floating status indicator opens the server-status dialog. It must be a
 * real button so keyboard users can reach and activate it; the dot + label
 * layout and the fixed corner position are unchanged.
 */
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { act, cleanup, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import StatusIndicator from './StatusIndicator'
import { renderWithProviders } from '@/test/render'
import { useBackendState } from '@/stores/state'

beforeEach(() => {
  // status:null seeds the dialog's no-data branch; a leftover health snapshot
  // from another test file would lack `configuration` and crash StatusCard.
  useBackendState.setState({ health: true, status: null, pipelineBusy: false, pipelineActive: false })
})

afterEach(() => {
  cleanup()
  useBackendState.getState().clearHealthCheckTimer()
})

describe('StatusIndicator', () => {
  test('the connected indicator is a button that opens the dialog from the keyboard', async () => {
    const user = userEvent.setup()
    renderWithProviders(<StatusIndicator />)

    const opener = screen.getByRole('button', { name: 'Connected' })
    act(() => opener.focus())
    await user.keyboard('{Enter}')

    await waitFor(() => {
      expect(screen.queryAllByRole('dialog')).toHaveLength(1)
    })
    expect(screen.queryAllByText('LightRAG Server Settings')).toHaveLength(1)
  })

  test('a disconnected backend names the button Disconnected', () => {
    useBackendState.setState({ health: false })
    renderWithProviders(<StatusIndicator />)

    expect(screen.queryAllByRole('button', { name: 'Disconnected' })).toHaveLength(1)
  })
})