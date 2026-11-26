/**
 * Rendered regression test for the "deleted rows linger on screen" bug.
 *
 * `/documents/delete_document` answers `deletion_started`: it reserves
 * `busy` + `destructive_busy` before answering, and the background task
 * releases them only after every `doc_status` row is gone. So the rows
 * disappear at some point AFTER the response, and `pipeline_busy === false`
 * observed after the response is proof they are gone.
 *
 * The scenario pinned here is a deletion that commits after the response but
 * before the UI's first health check would have run: the UI then never sees a
 * busy→idle TRANSITION, only `false → false`. What that used to leave:
 *   - the one immediate refresh raced the background deletion and re-rendered
 *     the row it was supposed to drop,
 *   - the `pipelineActive` effect fires on a transition, so it refreshed
 *     nothing,
 *   - and `startPollingInterval(2000)` was destroyed by the polling effect —
 *     every refresh hands `setStatusCounts` a fresh object, the effect re-runs
 *     and reinstates the 30s idle cadence — before its first tick.
 * The row therefore stayed on screen until the 30s idle poll.
 *
 * The probe's own timers and its dense observation window are what bring the
 * row down here. The pure level reading (idle on the very first observation,
 * because the busy window was missed entirely) is pinned in
 * documentDeletionProbe.test.ts.
 */
import { afterAll, afterEach, beforeEach, describe, expect, jest, test } from 'bun:test'
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import {
  __resetPaginatedDocumentRequestsForTests,
  __setAxiosAdapterForTests,
  type DocStatusResponse,
  type PaginatedDocsResponse
} from '@/api/lightrag'
import DocumentManager from '@/features/DocumentManager'
import { renderWithProviders } from '@/test/render'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'

const DOC_ID = 'doc-to-delete'
const SURVIVOR_ID = 'doc-that-stays'
const FAILED_ID = 'doc-that-failed'

/** Default time the simulated background deletion runs past its own response. */
const DELETION_COMMIT_MS = 150

const makeDoc = (id: string): DocStatusResponse => ({
  id,
  content_summary: `summary of ${id}`,
  content_length: 42,
  status: 'processed' as DocStatusResponse['status'],
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  chunks_count: 1,
  file_path: `${id}.txt`
})

/**
 * Minimal document backend honouring the endpoint's ordering contract: while
 * the deletion is in flight the pipeline reports busy, and it reports idle
 * only once the rows are actually gone. A model that answered idle with the
 * rows still present would be testing a backend that cannot exist.
 *
 * Driven through the api module's axios adapter seam rather than
 * `mock.module`: a module mock leaks into whichever test file runs next (it
 * broke `src/api/lightrag.test.ts`), while the adapter is restored by
 * `__setAxiosAdapterForTests(undefined)` and leaves the real module in place.
 */
const backend = {
  docs: [makeDoc(DOC_ID), makeDoc(SURVIVOR_ID)],
  /** Served for the "failed" filter tab, so a filter change is observable. */
  failedDocs: [{ ...makeDoc(FAILED_ID), status: 'failed' as DocStatusResponse['status'] }],
  commitDelayMs: DELETION_COMMIT_MS,
  deletionInFlight: false,
  /** Keeps /health reporting busy, so the activity probe keeps its burst going. */
  scanInFlight: false,
  /** Status filters requested while `recordFilters` is on. */
  requestedFilters: [] as (string | null)[],
  recordFilters: false,
  /** `hang` never answers (until aborted); `reject` answers with an HTTP 500. */
  paginatedMode: 'ok' as 'ok' | 'hang' | 'reject',
  paginatedCalls: 0,
  healthCalls: 0,
  deleteCalls: 0
}

const resetBackend = () => {
  backend.docs = [makeDoc(DOC_ID), makeDoc(SURVIVOR_ID)]
  backend.failedDocs = [
    { ...makeDoc(FAILED_ID), status: 'failed' as DocStatusResponse['status'] }
  ]
  backend.commitDelayMs = DELETION_COMMIT_MS
  backend.deletionInFlight = false
  backend.requestedFilters = []
  backend.recordFilters = false
  backend.scanInFlight = false
  backend.paginatedMode = 'ok'
  backend.paginatedCalls = 0
  backend.healthCalls = 0
  backend.deleteCalls = 0
}

const paginate = (statusFilter: string | null): PaginatedDocsResponse => {
  const documents = statusFilter === 'failed' ? backend.failedDocs : backend.docs

  return {
    documents,
    pagination: {
      page: 1,
      page_size: 10,
      total_count: documents.length,
      total_pages: 1,
      has_next: false,
      has_prev: false
    },
    status_counts: {
      all: backend.docs.length,
      processed: backend.docs.length,
      failed: backend.failedDocs.length
    }
  }
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

  if (url.endsWith('/documents/paginated')) {
    backend.paginatedCalls += 1
    const body = JSON.parse(config.data ?? '{}')
    const statusFilter = body.status_filter ?? null
    if (backend.recordFilters) backend.requestedFilters.push(statusFilter)
    if (backend.paginatedMode === 'hang') {
      // Settles only when the request is aborted (the afterEach reset does).
      return new Promise((_resolve, reject) => {
        config.signal?.addEventListener('abort', () => reject(new Error('aborted')))
      })
    }
    if (backend.paginatedMode === 'reject') {
      const error: any = new Error('Request failed with status code 500')
      error.response = { ...respond(config, { detail: 'boom' }), status: 500 }
      error.isAxiosError = true
      throw error
    }
    return respond(config, paginate(statusFilter))
  }

  if (url.endsWith('/health')) {
    backend.healthCalls += 1
    return respond(config, {
      status: 'healthy',
      pipeline_busy: backend.deletionInFlight || backend.scanInFlight,
      pipeline_active: backend.deletionInFlight || backend.scanInFlight
    })
  }

  if (url.endsWith('/documents/scan')) {
    backend.scanInFlight = true
    return respond(config, { status: 'scanning_started', message: 'ok' })
  }

  if (url.endsWith('/documents/delete_document')) {
    backend.deleteCalls += 1
    const docIds: string[] = JSON.parse(config.data ?? '{}').doc_ids ?? []
    // Reserved before the answer, released by the background task once the
    // rows are gone — the endpoint's contract, in miniature.
    backend.deletionInFlight = true
    setTimeout(() => {
      backend.docs = backend.docs.filter((doc) => !docIds.includes(doc.id))
      backend.deletionInFlight = false
    }, backend.commitDelayMs)
    return respond(config, {
      status: 'deletion_started',
      message: 'started',
      doc_id: docIds.join(', ')
    })
  }

  return respond(config, {})
}

// Let each probe observation finish before advancing to its next timer.
const advanceTime = async (ms: number) => {
  await act(async () => { jest.advanceTimersByTime(ms) })
}

const rowFor = (docId: string) => {
  const cell = screen.queryAllByText(docId)[0]
  return cell?.closest('tr') ?? null
}

beforeEach(() => {
  resetBackend()
  __setAxiosAdapterForTests(backendAdapter)
  useSettingsStore.getState().setCurrentTab('documents')
  useBackendState.setState({
    health: true,
    pipelineBusy: false,
    pipelineActive: false
  })
})

afterEach(() => {
  // Before any store reset: a file-local afterEach runs BEFORE the preload's
  // cleanup(), so resetting first would land on a still-mounted component.
  cleanup()
  useBackendState.getState().clearHealthCheckTimer()
  __resetPaginatedDocumentRequestsForTests()
  if (jest.isFakeTimers()) jest.useRealTimers()
})

afterAll(() => {
  __setAxiosAdapterForTests(undefined)
})

describe('DocumentManager initial list state', () => {
  const loadingCount = () => screen.queryAllByRole('status', { name: 'Loading documents...' }).length
  const emptyCount = () => screen.queryAllByText('No Documents').length

  test('before the first response it shows loading, not an empty corpus', async () => {
    backend.paginatedMode = 'hang'
    renderWithProviders(<DocumentManager />)
    await waitFor(() => {
      expect(backend.paginatedCalls).toBe(1)
    })
    expect({ loading: loadingCount(), empty: emptyCount() }).toEqual({ loading: 1, empty: 0 })
  })

  test('a failed first load shows the error and Retry, not an empty corpus', async () => {
    backend.paginatedMode = 'reject'
    renderWithProviders(<DocumentManager />)
    await waitFor(() => {
      expect(screen.queryAllByText('Could not load documents')).toHaveLength(1)
    })
    expect(screen.queryAllByRole('button', { name: 'Retry' })).toHaveLength(1)
    expect({ loading: loadingCount(), empty: emptyCount() }).toEqual({ loading: 0, empty: 0 })
  })

  test('a confirmed empty response shows the empty corpus and stops loading', async () => {
    backend.docs = []
    backend.failedDocs = []
    renderWithProviders(<DocumentManager />)
    await waitFor(() => {
      expect(emptyCount()).toBe(1)
    })
    expect(loadingCount()).toBe(0)
  })
})

describe('DocumentManager BFCache restore', () => {
  test('a restore after beforeunload refetches the list', async () => {
    renderWithProviders(<DocumentManager />)
    await waitFor(() => {
      expect(rowFor(DOC_ID) === null).toBe(false)
    })
    const callsBeforeRestore = backend.paginatedCalls

    // Leaving through BFCache fires beforeunload, and the same mounted tree is
    // resumed: nothing on beforeunload may stop the restore's refetch.
    window.dispatchEvent(new Event('beforeunload'))
    const pageShow = new Event('pageshow')
    Object.defineProperty(pageShow, 'persisted', { value: true })
    window.dispatchEvent(pageShow)

    await waitFor(() => {
      expect(backend.paginatedCalls > callsBeforeRestore).toBe(true)
    })
  })
})

describe('DocumentManager post-action refreshes', () => {
  test('partially deleted documents have a badge, filter and retry selection', async () => {
    backend.docs = [{ ...makeDoc(DOC_ID), status: 'deleting' }]
    backend.recordFilters = true
    const user = userEvent.setup()
    renderWithProviders(<DocumentManager />)
    await waitFor(() => {
      expect(screen.queryAllByText('Deleting').length).toBe(1)
    })
    const row = rowFor(DOC_ID)
    expect(row === null).toBe(false)
    await user.click(within(row as HTMLElement).getByRole('checkbox'))
    expect(screen.getByRole('button', { name: /^Delete$/ }).hasAttribute('disabled')).toBe(false)
    await user.click(screen.getByRole('button', { name: /^Deleting/ }))
    await waitFor(() => {
      expect(backend.requestedFilters.includes('deleting')).toBe(true)
    })
  })

  test(
    'a deletion that commits before the first health check still drops the row',
    async () => {
      const user = userEvent.setup()
      renderWithProviders(<DocumentManager />)

      await waitFor(() => {
        expect(rowFor(DOC_ID) === null).toBe(false)
      })

      const row = rowFor(DOC_ID)
      expect(row === null).toBe(false)
      await user.click(within(row as HTMLElement).getByRole('checkbox'))

      await user.click(screen.getByRole('button', { name: /^Delete$/ }))
      await user.type(screen.getByPlaceholderText('Type yes to confirm'), 'yes')
      const confirmedAt = Date.now()
      await user.click(screen.getByRole('button', { name: 'YES' }))

      await waitFor(() => {
        expect(backend.deleteCalls).toBe(1)
      })
      const healthCallsAtDelete = backend.healthCalls

      // Nothing but the confirmation probe can bring this row down: the
      // pipeline is never seen flipping, and no cadence faster than 30s
      // survives the refresh that follows the delete.
      await waitFor(
        () => {
          expect(screen.queryAllByText(DOC_ID)).toHaveLength(0)
        },
        { timeout: 8000 }
      )

      // ...and promptly. The probe's confirming refresh carries user intent, so
      // it must not queue behind the automatic refresh that the probe's own
      // health check provokes through the pipelineActive transition effect.
      // While it did, this deletion took ~2.06s to leave the table (the 2s
      // throttle boundary); it now takes ~0.6s, one probe tick plus a round
      // trip. The bound is loose enough for a slow machine and still an order
      // of magnitude below the boundary it pins.
      expect(Date.now() - confirmedAt < 1500).toBe(true)

      // The probe refreshed the list; it did not blank the table.
      expect(screen.queryAllByText(SURVIVOR_ID).length > 0).toBe(true)
      // Necessary, not sufficient: no periodic health timer is installed in
      // this render, so a health request after the delete could only have come
      // from the probe. What it does NOT prove on its own is which observation
      // ended the probe — the module tests cover that.
      expect(backend.healthCalls > healthCallsAtDelete).toBe(true)
    },
    20000
  )

  test(
    'a refresh landing after a filter change queries the new view, not the old',
    async () => {
      // The probe outlives the render that created it, so the callback it holds
      // must be resolved from CURRENT state. A captured one pairs an OLD query
      // snapshot (page/filter/sort) with the CURRENT request version — the
      // combination runRefreshRequest's staleness guard cannot catch, since it
      // only compares versions — and the response overwrites the view the user
      // navigated to.
      //
      // Asserted on the requests rather than the rendered rows: the clobber is
      // self-healing within ~2s (the next refresh re-fetches the current view),
      // so the DOM window is racy while "which query did we ask for" is exact.
      backend.commitDelayMs = 3000
      const user = userEvent.setup()
      renderWithProviders(<DocumentManager />)

      await waitFor(() => {
        expect(rowFor(DOC_ID) === null).toBe(false)
      })

      const row = rowFor(DOC_ID)
      await user.click(within(row as HTMLElement).getByRole('checkbox'))
      await user.click(screen.getByRole('button', { name: /^Delete$/ }))
      await user.type(screen.getByPlaceholderText('Type yes to confirm'), 'yes')
      jest.useFakeTimers()
      await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'YES' })) })
      expect(backend.deleteCalls).toBe(1)
      await advanceTime(1)

      // Navigate away while the deletion is still running. Every paginated
      // request from here on must carry the new filter.
      expect(backend.deletionInFlight).toBe(true)
      backend.recordFilters = true
      await act(async () => { fireEvent.click(screen.getByRole('button', { name: /^Failed/ })) })
      expect(screen.queryAllByText(FAILED_ID).length > 0).toBe(true)

      // The probe chains its health checks at 0, 500, 1000, 2000 and 4000 ms.
      // Allow each response to schedule the next check, then observe the
      // commit at 3000 ms separately from the confirming refresh at 4000 ms.
      await advanceTime(500)
      await advanceTime(500)
      await advanceTime(1000)
      await advanceTime(1000)
      expect(backend.deletionInFlight).toBe(false)
      const callsAtCommit = backend.paginatedCalls
      await advanceTime(1000)
      expect(backend.paginatedCalls).toBeGreaterThan(callsAtCommit)

      // A stale refresh can sit behind the 2s trailing throttle.
      await advanceTime(2000)

      expect(backend.requestedFilters.filter((filter) => filter !== 'failed')).toHaveLength(0)
      expect(screen.queryAllByText(SURVIVOR_ID)).toHaveLength(0)
      expect(screen.queryAllByText(FAILED_ID).length > 0).toBe(true)
    }
  )

  test(
    'the scan activity probe also refreshes the current view, not the captured one',
    async () => {
      // Same defect, wider window: the activity probe's burst runs for up to
      // 16s off the closure captured when scan was clicked. Pre-existing, and
      // fixed by the same latest-render ref.
      renderWithProviders(<DocumentManager />)

      await waitFor(() => {
        expect(rowFor(DOC_ID) === null).toBe(false)
      })

      jest.useFakeTimers()
      await act(async () => { fireEvent.click(screen.getByRole('button', { name: /^Scan\/Retry/ })) })
      expect(backend.scanInFlight).toBe(true)
      await advanceTime(1)

      backend.recordFilters = true
      await act(async () => { fireEvent.click(screen.getByRole('button', { name: /^Failed/ })) })
      expect(screen.queryAllByText(FAILED_ID).length > 0).toBe(true)
      expect(backend.requestedFilters).toEqual(['failed'])

      // Both scheduled refreshes must run and query the new view.
      await advanceTime(1000)
      await advanceTime(1000)
      expect(backend.requestedFilters).toEqual(['failed', 'failed'])
      await advanceTime(2000)
      expect(backend.requestedFilters).toEqual(['failed', 'failed', 'failed'])
    }
  )
})
