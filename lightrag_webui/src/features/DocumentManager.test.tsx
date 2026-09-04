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
import { afterAll, afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { cleanup, screen, waitFor, within } from '@testing-library/react'
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

/** How long the simulated background deletion runs past its own response. */
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
  deletionInFlight: false,
  paginatedCalls: 0,
  healthCalls: 0,
  deleteCalls: 0
}

const resetBackend = () => {
  backend.docs = [makeDoc(DOC_ID), makeDoc(SURVIVOR_ID)]
  backend.deletionInFlight = false
  backend.paginatedCalls = 0
  backend.healthCalls = 0
  backend.deleteCalls = 0
}

const paginate = (): PaginatedDocsResponse => ({
  documents: backend.docs,
  pagination: {
    page: 1,
    page_size: 10,
    total_count: backend.docs.length,
    total_pages: 1,
    has_next: false,
    has_prev: false
  },
  status_counts: { all: backend.docs.length, processed: backend.docs.length }
})

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
    return respond(config, paginate())
  }

  if (url.endsWith('/health')) {
    backend.healthCalls += 1
    return respond(config, {
      status: 'healthy',
      pipeline_busy: backend.deletionInFlight,
      pipeline_active: backend.deletionInFlight
    })
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
    }, DELETION_COMMIT_MS)
    return respond(config, {
      status: 'deletion_started',
      message: 'started',
      doc_id: docIds.join(', ')
    })
  }

  return respond(config, {})
}

const rowFor = (docId: string) => {
  const cell = screen.queryAllByText(docId)[0]
  return cell?.closest('tr') ?? null
}

describe('DocumentManager deletion confirmation', () => {
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
  })

  afterAll(() => {
    __setAxiosAdapterForTests(undefined)
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
})
