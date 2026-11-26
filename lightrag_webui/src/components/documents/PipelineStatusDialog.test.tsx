/**
 * The pipeline dialog's three position buttons are icon-only: each needs an
 * accessible name and an aria-pressed state so the active position is not
 * color-only. The polling history container must be a live log region so
 * screen readers hear the messages that arrive every 2s.
 */
import { afterAll, afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { cleanup, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import {
  __resetPaginatedDocumentRequestsForTests,
  __setAxiosAdapterForTests
} from '@/api/lightrag'
import PipelineStatusDialog from './PipelineStatusDialog'
import { renderWithProviders } from '@/test/render'

const backend = {
  statusCalls: 0
}

const respond = (config: any, data: unknown) => ({
  data,
  status: 200,
  statusText: 'OK',
  headers: {},
  config
})

const backendAdapter = async (config: any) => {
  const url: string = config.url ?? ''
  if (url.endsWith('/documents/pipeline_status')) {
    backend.statusCalls += 1
    // The second poll appends a message: the live region must receive it.
    const history = backend.statusCalls > 1
      ? ['first line', 'second line', 'third line']
      : ['first line', 'second line']
    return respond(config, {
      busy: true,
      job_name: 'ingest',
      docs: 10,
      batchs: 4,
      cur_batch: 2,
      latest_message: history[history.length - 1],
      history_messages: history
    })
  }
  return respond(config, {})
}

beforeEach(() => {
  backend.statusCalls = 0
  __setAxiosAdapterForTests(backendAdapter)
})

afterEach(() => {
  cleanup()
  __resetPaginatedDocumentRequestsForTests()
})

afterAll(() => {
  __setAxiosAdapterForTests(undefined)
})

describe('PipelineStatusDialog position buttons', () => {
  test('each position button is named and aria-pressed follows the active position', async () => {
    const user = userEvent.setup()
    renderWithProviders(<PipelineStatusDialog open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(screen.queryAllByText('Pipeline Status')).toHaveLength(1)
    })

    const left = screen.getByRole('button', { name: 'Align left' })
    const center = screen.getByRole('button', { name: 'Align center' })
    const right = screen.getByRole('button', { name: 'Align right' })
    expect(left.getAttribute('aria-pressed')).toBe('false')
    expect(center.getAttribute('aria-pressed')).toBe('true')
    expect(right.getAttribute('aria-pressed')).toBe('false')

    await user.click(left)
    expect(left.getAttribute('aria-pressed')).toBe('true')
    expect(center.getAttribute('aria-pressed')).toBe('false')

    await user.click(right)
    expect(right.getAttribute('aria-pressed')).toBe('true')
    expect(left.getAttribute('aria-pressed')).toBe('false')
  })
})

describe('PipelineStatusDialog history region', () => {
  test('history renders in a polite log region that receives polling updates', async () => {
    renderWithProviders(<PipelineStatusDialog open={true} onOpenChange={() => {}} />)

    const log = await screen.findByRole('log')
    expect(log.getAttribute('aria-live')).toBe('polite')
    expect(log.textContent).toContain('first line')
    expect(log.textContent).toContain('second line')

    // The 2s poll appends a message; it must land inside the same log region.
    await waitFor(
      () => {
        expect(screen.getByRole('log').textContent).toContain('third line')
      },
      { timeout: 6000 }
    )
  }, 10000)
})