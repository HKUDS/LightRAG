import { afterEach, beforeAll, describe, expect, test } from 'bun:test'

type DocumentsRequest = {
  status_filter?: 'pending' | 'processing' | 'preprocessed' | 'processed' | 'failed' | null
  page: number
  page_size: number
  sort_field: 'created_at' | 'updated_at' | 'id' | 'file_path'
  sort_direction: 'asc' | 'desc'
}

type LightragApiModule = typeof import('./lightrag')

const storageMock = () => {
  const data = new Map<string, string>()

  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
    clear: () => {
      data.clear()
    }
  }
}

let apiModule: LightragApiModule

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })

  apiModule = await import('./lightrag')
})

afterEach(() => {
  apiModule.__resetPaginatedDocumentRequestsForTests()
})

describe('getDocumentsPaginated', () => {
  test('issues a fresh request after aborting a timed-out in-flight request', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    const resolvers: Array<(value: any) => void> = []

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolvers.push(resolve)
        controller.signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        )
      })
    })

    const firstRequest = apiModule.getDocumentsPaginated(request)
    const secondRequest = apiModule.getDocumentsPaginated(request)

    expect(callCount).toBe(1)

    apiModule.abortDocumentsPaginated(request)
    const [firstResult, secondResult] = await Promise.allSettled([
      firstRequest,
      secondRequest
    ])
    expect(firstResult.status).toBe('rejected')
    expect(secondResult.status).toBe('rejected')

    const thirdRequest = apiModule.getDocumentsPaginated(request)
    expect(callCount).toBe(2)

    resolvers[1]({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(thirdRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  test('times out hanging requests and allows a fresh retry', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    const resolvers: Array<(value: any) => void> = []

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolvers.push(resolve)
        controller.signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        )
      })
    })

    await expect(
      apiModule.getDocumentsPaginatedWithTimeout(request, 1)
    ).rejects.toThrow('Document fetch timeout')

    expect(callCount).toBe(1)

    const retryRequest = apiModule.getDocumentsPaginated(request)
    expect(callCount).toBe(2)

    resolvers[1]({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(retryRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  test('does not abort a shared request when only one timeout subscriber expires', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    let resolveSharedRequest: ((value: any) => void) | undefined
    let abortCount = 0

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolveSharedRequest = resolve
        controller.signal.addEventListener(
          'abort',
          () => {
            abortCount += 1
            reject(new DOMException('Aborted', 'AbortError'))
          },
          { once: true }
        )
      })
    })

    const shortTimeoutRequest = apiModule.getDocumentsPaginatedWithTimeout(request, 1)
    const longTimeoutRequest = apiModule.getDocumentsPaginatedWithTimeout(request, 100)

    await expect(shortTimeoutRequest).rejects.toThrow('Document fetch timeout')

    expect(callCount).toBe(1)
    expect(abortCount).toBe(0)

    resolveSharedRequest?.({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(longTimeoutRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })
})

describe('isUserAbortError', () => {
  // Regression: the Stop button must suppress query cancellation everywhere it
  // surfaces — both the main stream catch and the guest-token retry catch (which
  // otherwise redirects an aborting guest to the login page). Both sites share
  // this predicate, so locking down its behavior guards both fixes.
  test('treats an aborted signal as a user abort regardless of the error', () => {
    const controller = new AbortController()
    controller.abort()
    expect(apiModule.isUserAbortError(controller.signal, new Error('boom'))).toBe(true)
  })

  test('treats an AbortError as a user abort even when the signal is absent', () => {
    const abortError = new DOMException('Aborted', 'AbortError')
    expect(apiModule.isUserAbortError(undefined, abortError)).toBe(true)
  })

  test('does not treat a real failure on a live signal as a user abort', () => {
    const controller = new AbortController()
    expect(apiModule.isUserAbortError(controller.signal, new Error('network down'))).toBe(false)
    expect(apiModule.isUserAbortError(undefined, new Error('network down'))).toBe(false)
  })
})

describe('response interceptor', () => {
  test('an HTTP failure carries its status, not just a formatted message', async () => {
    // The interceptor rewrites every non-401 AxiosError into a plain Error.
    // Without the status as a property, callers that branch on it (the
    // document list treats a 4xx as permanent and must not let it trip the
    // refresh circuit breaker) see only `undefined` and fall back to
    // "unknown, retry".
    apiModule.__setAxiosAdapterForTests(async () => {
      throw {
        response: {
          status: 404,
          statusText: 'Not Found',
          data: { detail: 'no such workspace' }
        },
        config: { url: '/documents/pipeline_status' }
      }
    })

    try {
      const failure = await apiModule
        .getPipelineStatus()
        .then(() => null)
        .catch((error: unknown) => error as Error & { status?: number })

      expect(failure).toBeInstanceOf(Error)
      expect(failure?.status).toBe(404)
      expect(failure?.message).toContain('404 Not Found')
      expect(failure?.message).toContain('no such workspace')
    } finally {
      apiModule.__setAxiosAdapterForTests(undefined)
    }
  })

  test('toHttpRequestError builds the same shape directly', () => {
    const error = apiModule.toHttpRequestError(422, 'Unprocessable Entity', { detail: 'bad' }, '/documents/paginated')

    expect(error.status).toBe(422)
    expect(error.message).toContain('422 Unprocessable Entity')
    expect(error.message).toContain('/documents/paginated')
  })
})

describe('ai content notice flag', () => {
  const makeResponse = (data: unknown) => async (config: any) => ({
    data,
    status: 200,
    statusText: 'OK',
    headers: { 'content-type': 'application/json' },
    config
  })

  afterEach(() => {
    apiModule.__setAxiosAdapterForTests(undefined)
  })

  test('/auth-status carries the deployment flag into the store', async () => {
    const { useAiContentNoticeStore } = await import('@/stores/aiContentNotice')
    useAiContentNoticeStore.setState({ enabled: false })

    apiModule.__setAxiosAdapterForTests(
      makeResponse({
        auth_configured: false,
        access_token: 'guest-token',
        ai_content_notice_enabled: true
      })
    )

    await apiModule.getAuthStatus()

    expect(useAiContentNoticeStore.getState().enabled).toBe(true)
  })

  test('/login carries the deployment flag into the store', async () => {
    const { useAiContentNoticeStore } = await import('@/stores/aiContentNotice')
    useAiContentNoticeStore.setState({ enabled: false })

    apiModule.__setAxiosAdapterForTests(
      makeResponse({
        access_token: 'user-token',
        token_type: 'bearer',
        ai_content_notice_enabled: true
      })
    )

    await apiModule.loginToServer('user', 'password')

    expect(useAiContentNoticeStore.getState().enabled).toBe(true)
  })

  test('a server that omits the field does not turn an enabled notice off', async () => {
    const { useAiContentNoticeStore } = await import('@/stores/aiContentNotice')
    useAiContentNoticeStore.setState({ enabled: true })

    apiModule.__setAxiosAdapterForTests(
      makeResponse({ auth_configured: false, access_token: 'guest-token' })
    )

    await apiModule.getAuthStatus()

    expect(useAiContentNoticeStore.getState().enabled).toBe(true)
  })
})
