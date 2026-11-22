/**
 * The LLM-cache checkbox must ride along on the clear request itself.
 *
 * It used to fire a second call to `POST /documents/clear_cache`, an endpoint
 * with no concurrency control at all — it cleared the cache whether or not the
 * ingestion pipeline was running, wiping the extraction rows in-flight chunks
 * had already paid for. The capability now folds into `DELETE /documents`, so
 * it runs inside the destructive reservation that endpoint already takes, and
 * the dialog must pass the option rather than making its own call.
 *
 * The fake backend is installed at the axios adapter seam, NOT with
 * `mock.module('@/api/lightrag')`: a module-level mock stays installed for
 * every file bun evaluates afterwards, and `src/api/lightrag.test.ts` then
 * asserts against the stub instead of the real module. The adapter also makes
 * this a stronger test — the real `clearDocuments` runs, so the request it
 * actually sends is what gets checked.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'
import { cleanup, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import * as api from '@/api/lightrag'
import { renderWithProviders, testI18n } from '@/test/render'
import ClearDocumentsDialog from './ClearDocumentsDialog'

type ClearStatus = 'success' | 'partial_success' | 'fail' | 'busy'

const clearRequests: Array<{ method?: string; url?: string; clearLlmCache?: boolean }> = []
let nextResult: { status: ClearStatus; message: string } = {
  status: 'success',
  message: 'All documents cleared successfully. Deleted 0 files.'
}

const onDocumentsCleared = mock(async () => {})

beforeEach(() => {
  api.__setAxiosAdapterForTests(async (config: any) => {
    if (config.url === '/documents' && config.method?.toLowerCase() === 'delete') {
      clearRequests.push({
        method: config.method,
        url: config.url,
        // Still a raw boolean at adapter time: axios has not serialised the
        // query string yet.
        clearLlmCache: config.params?.clear_llm_cache
      })
      return {
        data: nextResult,
        status: 200,
        statusText: 'OK',
        headers: { 'content-type': 'application/json' },
        config
      }
    }
    return {
      data: {},
      status: 200,
      statusText: 'OK',
      headers: { 'content-type': 'application/json' },
      config
    }
  })
})

afterEach(() => {
  // Restored per test, not in afterAll: a thrown expectation mid-test must not
  // leak the adapter into the next file.
  api.__setAxiosAdapterForTests(undefined)
  cleanup()
  onDocumentsCleared.mockClear()
  clearRequests.length = 0
  nextResult = {
    status: 'success',
    message: 'All documents cleared successfully. Deleted 0 files.'
  }
})

const openAndConfirm = async (options: { checkCache: boolean }) => {
  const user = userEvent.setup()
  renderWithProviders(<ClearDocumentsDialog onDocumentsCleared={onDocumentsCleared} />)

  await user.click(screen.getByRole('button', { name: /clear/i }))
  await user.type(await screen.findByPlaceholderText(/type yes to confirm/i), 'yes')
  if (options.checkCache) {
    await user.click(screen.getByLabelText(/clear llm cache/i))
  }
  await user.click(screen.getByRole('button', { name: 'YES' }))
}

describe('ClearDocumentsDialog', () => {
  test('clears without the cache by default, and closes on an unqualified success', async () => {
    await openAndConfirm({ checkCache: false })

    await waitFor(() => expect(clearRequests.length).toBe(1))
    expect(clearRequests[0].url).toBe('/documents')
    expect(clearRequests[0].method?.toLowerCase()).toBe('delete')
    expect(clearRequests[0].clearLlmCache).toBe(false)

    await waitFor(() =>
      expect(screen.queryAllByPlaceholderText(/type yes to confirm/i).length).toBe(0)
    )
  })

  test('folds the opted-in cache drop into the single clear request', async () => {
    await openAndConfirm({ checkCache: true })

    await waitFor(() => expect(clearRequests.length).toBe(1))
    expect(clearRequests[0].url).toBe('/documents')
    expect(clearRequests[0].method?.toLowerCase()).toBe('delete')
    expect(clearRequests[0].clearLlmCache).toBe(true)
  })

  test('no longer exposes a standalone cache-clearing call', () => {
    expect('clearCache' in api).toBe(false)
  })

  // At least one storage was dropped, so the list on screen is stale
  // whatever else failed: not refreshing would leave the user looking at
  // documents the server no longer has. But partial_success does NOT prove the
  // documents are gone either: the server returns it whenever one drop failed
  // and another succeeded, so a failed `doc_status` or `full_docs` drop lands
  // here with rows still in place. So it refreshes, keeps the dialog open (a
  // retry is one step), and the toast must not claim the documents were
  // cleared.
  test('partial_success refreshes, stays open and does not claim the clear', async () => {
    nextResult = {
      status: 'partial_success',
      message: 'Cleared documents with some errors. Deleted 0 files.'
    }

    await openAndConfirm({ checkCache: true })

    await waitFor(() => expect(onDocumentsCleared).toHaveBeenCalledTimes(1))
    expect(screen.queryAllByPlaceholderText(/type yes to confirm/i).length).toBe(1)
    const partial = testI18n.t('documentPanel.clearDocuments.partialSuccess', {
      message: ''
    })
    expect(/cleared|deleted|removed/i.test(partial)).toBe(false)
  })

  // `fail` (every storage drop failed) and `busy` (nothing ran) are the cases
  // where the documents really do survive. They share one branch (anything
  // that is neither success nor partial_success), so `fail` stands for both.
  test('keeps the dialog open and does not refresh on fail', async () => {
    nextResult = { status: 'fail', message: 'nothing was cleared' }

    await openAndConfirm({ checkCache: false })

    await waitFor(() => expect(clearRequests.length).toBe(1))
    expect(onDocumentsCleared).toHaveBeenCalledTimes(0)
    expect(screen.queryAllByPlaceholderText(/type yes to confirm/i).length).toBe(1)
  })
})
