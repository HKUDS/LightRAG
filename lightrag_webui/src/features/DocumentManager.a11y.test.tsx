/**
 * Accessibility regressions for the document panel: keyboard-operable sort
 * headers with aria-sort, named row checkboxes, deferred row rendering, and
 * page/filter/sort state synced to the URL (deep-link in, hashchange back).
 *
 * Driven through the api module's axios adapter seam (same pattern as
 * DocumentManager.test.tsx): the fake backend honours page/filter/sort so a
 * request log and the rendered row order both observe the panel's query.
 */
import { afterAll, afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { act, cleanup, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import {
  __resetPaginatedDocumentRequestsForTests,
  __setAxiosAdapterForTests,
  type DocStatusResponse,
  type DocumentsRequest
} from '@/api/lightrag'
import DocumentManager from '@/features/DocumentManager'
import { renderWithProviders } from '@/test/render'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'

const makeDoc = (id: string, status: DocStatusResponse['status'], day: number): DocStatusResponse => ({
  id,
  content_summary: `summary of ${id}`,
  content_length: 42,
  status,
  created_at: `2026-01-${String(day).padStart(2, '0')}T00:00:00Z`,
  updated_at: `2026-02-${String(day).padStart(2, '0')}T00:00:00Z`,
  chunks_count: 1,
  file_path: `${id}.txt`
})

const processedDocs = Array.from({ length: 15 }, (_, i) =>
  makeDoc(`doc-p${String(i + 1).padStart(2, '0')}`, 'processed', i + 1)
)
const failedDocs = Array.from({ length: 12 }, (_, i) => ({
  ...makeDoc(`doc-f${String(i + 1).padStart(2, '0')}`, 'failed', i + 1),
  updated_at: `2026-01-${String(i + 1).padStart(2, '0')}T00:00:00Z`
}))

const backend = {
  requests: [] as DocumentsRequest[]
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
    const body = JSON.parse(config.data ?? '{}')
    backend.requests.push(body)
    const pool = body.status_filter === 'failed' ? failedDocs : [...processedDocs, ...failedDocs]
    const sortField: string = body.sort_field ?? 'updated_at'
    const direction = body.sort_direction === 'asc' ? 1 : -1
    const sorted = [...pool].sort((a, b) => {
      const av = String((a as Record<string, unknown>)[sortField] ?? '')
      const bv = String((b as Record<string, unknown>)[sortField] ?? '')
      return av.localeCompare(bv) * direction
    })
    const page: number = body.page ?? 1
    const pageSize: number = body.page_size ?? 10
    const documents = sorted.slice((page - 1) * pageSize, page * pageSize)
    const totalPages = Math.ceil(sorted.length / pageSize)
    return respond(config, {
      documents,
      pagination: {
        page,
        page_size: pageSize,
        total_count: sorted.length,
        total_pages: totalPages,
        has_next: page < totalPages,
        has_prev: page > 1
      },
      status_counts: {
        all: processedDocs.length + failedDocs.length,
        processed: processedDocs.length,
        failed: failedDocs.length
      }
    })
  }

  if (url.endsWith('/health')) {
    return respond(config, { status: 'healthy', pipeline_busy: false, pipeline_active: false })
  }

  return respond(config, {})
}

const lastRequest = (): DocumentsRequest => backend.requests[backend.requests.length - 1]

const columnheader = (name: string) => screen.getByRole('columnheader', { name })

const firstBodyRowText = (): string => {
  const rows = screen.getAllByRole('row')
  return rows[1]?.textContent ?? ''
}

const waitForRows = async () => {
  await waitFor(() => {
    expect(screen.getAllByRole('row').length).toBeGreaterThan(1)
  })
}

beforeEach(() => {
  backend.requests = []
  // The DOM (and its URL) is process-wide: a hash left by another test would
  // deep-link this render into a different view.
  window.location.hash = ''
  __setAxiosAdapterForTests(backendAdapter)
  useSettingsStore.getState().setCurrentTab('documents')
  useSettingsStore.getState().setShowFileName(false)
  useSettingsStore.getState().setDocumentsPageSize(10)
  useBackendState.setState({ health: true, pipelineBusy: false, pipelineActive: false })
})

afterEach(() => {
  // Before any store/URL reset: a file-local afterEach runs BEFORE the
  // preload's cleanup(), so resetting first lands on a still-mounted panel.
  cleanup()
  window.location.hash = ''
  useBackendState.getState().clearHealthCheckTimer()
  __resetPaginatedDocumentRequestsForTests()
})

afterAll(() => {
  __setAxiosAdapterForTests(undefined)
})

describe('DocumentManager sortable headers', () => {
  test('keyboard activation sorts and aria-sort tracks the active column', async () => {
    const user = userEvent.setup()
    renderWithProviders(<DocumentManager />)
    await waitForRows()

    // Default query: updated_at desc.
    expect(columnheader('Updated').getAttribute('aria-sort')).toBe('descending')
    expect(columnheader('Created').getAttribute('aria-sort')).toBe('none')
    expect(columnheader('ID').getAttribute('aria-sort')).toBe('none')
    expect(firstBodyRowText()).toContain('doc-p15')

    // Enter on the focused header button toggles the active column to asc.
    const updatedButton = screen.getByRole('button', { name: 'Updated' })
    act(() => updatedButton.focus())
    await user.keyboard('{Enter}')
    await waitFor(() => {
      expect(columnheader('Updated').getAttribute('aria-sort')).toBe('ascending')
    })
    expect(lastRequest().sort_field).toBe('updated_at')
    expect(lastRequest().sort_direction).toBe('asc')
    expect(firstBodyRowText()).toContain('doc-f01')

    // Enter on another header moves the sort there (desc on a new column).
    // Space activation is the native button's browser default action, which
    // happy-dom does not synthesize; Enter proves the keyboard path.
    const createdButton = screen.getByRole('button', { name: 'Created' })
    act(() => createdButton.focus())
    await user.keyboard('{Enter}')
    await waitFor(() => {
      expect(columnheader('Created').getAttribute('aria-sort')).toBe('descending')
    })
    expect(columnheader('Updated').getAttribute('aria-sort')).toBe('none')
    expect(lastRequest().sort_field).toBe('created_at')
    expect(lastRequest().sort_direction).toBe('desc')
    expect(firstBodyRowText()).toContain('doc-p15')
  })
})

describe('DocumentManager row selection labels', () => {
  test('each checkbox is named by its document id and still selects', async () => {
    const user = userEvent.setup()
    renderWithProviders(<DocumentManager />)
    await waitForRows()

    const checkbox = screen.getByRole('checkbox', { name: 'Select document doc-p15' })
    expect(screen.queryAllByRole('checkbox', { name: 'Select document doc-p14' })).toHaveLength(1)
    await user.click(checkbox)
    expect(checkbox.getAttribute('aria-checked')).toBe('true')
    expect(screen.queryAllByRole('button', { name: /^Delete$/ })).toHaveLength(1)
  })

  test('with file names shown the checkbox is named by the file path', async () => {
    useSettingsStore.getState().setShowFileName(true)
    renderWithProviders(<DocumentManager />)
    await waitForRows()

    expect(screen.queryAllByRole('checkbox', { name: 'Select document doc-p15.txt' })).toHaveLength(1)
  })
})

describe('DocumentManager deferred row rendering', () => {
  test('every body row carries content-visibility with an intrinsic size', async () => {
    renderWithProviders(<DocumentManager />)
    await waitForRows()

    const bodyRows = screen.getAllByRole('row').slice(1)
    expect(bodyRows).toHaveLength(10)
    for (const row of bodyRows) {
      expect(row.className).toContain('[content-visibility:auto]')
      expect(row.className).toContain('[contain-intrinsic-size:auto_3.25rem]')
    }
  })
})

describe('DocumentManager URL state', () => {
  test('a deep link initializes page, filter and sort (URL wins)', async () => {
    window.location.hash = '#/?page=2&status=failed&sort=created_at&order=asc'
    renderWithProviders(<DocumentManager />)
    await waitForRows()

    expect(backend.requests[0].page).toBe(2)
    expect(backend.requests[0].status_filter).toBe('failed')
    expect(backend.requests[0].sort_field).toBe('created_at')
    expect(backend.requests[0].sort_direction).toBe('asc')
    // failed pool sorted created_at asc, page 2 of size 10.
    expect(firstBodyRowText()).toContain('doc-f11')
    expect(columnheader('Created').getAttribute('aria-sort')).toBe('ascending')
  })

  test('invalid URL values fall back to the defaults', async () => {
    window.location.hash = '#/?page=banana&status=bogus&sort=nope&order=sideways'
    renderWithProviders(<DocumentManager />)
    await waitForRows()

    expect(backend.requests[0].page).toBe(1)
    expect(backend.requests[0].status_filter).toBe(null)
    expect(backend.requests[0].sort_field).toBe('updated_at')
    expect(backend.requests[0].sort_direction).toBe('desc')
  })

  test('filter, sort and page changes are written to the URL', async () => {
    const user = userEvent.setup()
    renderWithProviders(<DocumentManager />)
    await waitForRows()
    expect(window.location.hash).toBe('')

    await user.click(screen.getByRole('button', { name: /^Failed/ }))
    await waitFor(() => {
      expect(window.location.hash).toContain('status=failed')
    })

    await user.click(screen.getByRole('button', { name: 'Created' }))
    await waitFor(() => {
      expect(window.location.hash).toContain('sort=created_at')
    })

    const pageInput = screen.getByRole('textbox', { name: 'Page' })
    await user.clear(pageInput)
    await user.type(pageInput, '2{Enter}')
    await waitFor(() => {
      expect(window.location.hash).toBe('#/?page=2&status=failed&sort=created_at&order=desc')
    })
    expect(lastRequest().page).toBe(2)
  })

  test('a hashchange (back/forward) applies the URL state and settles without a loop', async () => {
    renderWithProviders(<DocumentManager />)
    await waitForRows()

    window.location.hash = '#/?status=failed&sort=created_at&order=asc'
    await waitFor(() => {
      expect(lastRequest().status_filter).toBe('failed')
      expect(lastRequest().sort_field).toBe('created_at')
      expect(lastRequest().sort_direction).toBe('asc')
    })
    expect(firstBodyRowText()).toContain('doc-f01')

    // Stepping back to the bare route resets every piece of panel state.
    window.location.hash = '#/'
    await waitFor(() => {
      expect(lastRequest().status_filter).toBe(null)
      expect(lastRequest().sort_field).toBe('updated_at')
      expect(lastRequest().sort_direction).toBe('desc')
    })
    expect(columnheader('Updated').getAttribute('aria-sort')).toBe('descending')
    expect(firstBodyRowText()).toContain('doc-p15')
    // The panel's own write must not re-add params the URL just dropped.
    expect(window.location.hash).toBe('#/')
  })
})