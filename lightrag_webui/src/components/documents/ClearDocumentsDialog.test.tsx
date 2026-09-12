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
 * `mock.module` only reaches importers that have not been evaluated yet, so
 * the dialog is imported dynamically AFTER the stub is installed, and the real
 * module is restored in afterAll for later test files.
 */
import { afterAll, afterEach, beforeAll, describe, expect, mock, test } from 'bun:test'
import { cleanup, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'

let realApiModule: Record<string, unknown>
let ClearDocumentsDialog: typeof import('./ClearDocumentsDialog').default

type ClearStatus = 'success' | 'partial_success' | 'fail' | 'busy'

const clearLlmCacheArgs: Array<boolean | undefined> = []
let nextResult: { status: ClearStatus; message: string } = {
  status: 'success',
  message: 'All documents cleared successfully. Deleted 0 files.'
}
const clearDocuments = mock(async (clearLlmCache?: boolean) => {
  clearLlmCacheArgs.push(clearLlmCache)
  return nextResult
})

const onDocumentsCleared = mock(async () => {})

beforeAll(async () => {
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({ ...realApiModule, clearDocuments }))
  ClearDocumentsDialog = (await import('./ClearDocumentsDialog')).default
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

afterEach(() => {
  cleanup()
  clearDocuments.mockClear()
  onDocumentsCleared.mockClear()
  clearLlmCacheArgs.length = 0
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
  test('clears without the cache by default', async () => {
    await openAndConfirm({ checkCache: false })

    await waitFor(() => expect(clearDocuments).toHaveBeenCalledTimes(1))
    expect(clearLlmCacheArgs).toEqual([false])
  })

  test('folds the opted-in cache drop into the single clear request', async () => {
    await openAndConfirm({ checkCache: true })

    await waitFor(() => expect(clearDocuments).toHaveBeenCalledTimes(1))
    expect(clearLlmCacheArgs).toEqual([true])
  })

  test('no longer exposes a standalone cache-clearing call', () => {
    expect('clearCache' in realApiModule).toBe(false)
  })

  // The documents ARE gone on partial_success -- a storage, an input file or
  // the opted-in cache drop reported an error. Reporting that as a failed
  // clear leaves the dialog open over a list that no longer reflects the
  // server, telling the user their documents survived when they did not.
  test('refreshes and closes on partial_success', async () => {
    nextResult = {
      status: 'partial_success',
      message: 'Cleared documents with some errors. Deleted 0 files.'
    }

    await openAndConfirm({ checkCache: true })

    await waitFor(() => expect(onDocumentsCleared).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(screen.queryAllByPlaceholderText(/type yes to confirm/i).length).toBe(0)
    )
  })

  // `busy` (nothing ran) and `fail` (every storage drop failed) are the cases
  // where the documents really do survive.
  test.each(['fail', 'busy'] as const)(
    'keeps the dialog open and does not refresh on %s',
    async (status) => {
      nextResult = { status, message: 'nothing was cleared' }

      await openAndConfirm({ checkCache: false })

      await waitFor(() => expect(clearDocuments).toHaveBeenCalledTimes(1))
      expect(onDocumentsCleared).toHaveBeenCalledTimes(0)
      expect(screen.queryAllByPlaceholderText(/type yes to confirm/i).length).toBe(1)
    }
  )
})
