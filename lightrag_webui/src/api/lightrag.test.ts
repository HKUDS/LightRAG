import { afterEach, beforeAll, describe, expect, test } from 'bun:test'

type DocumentsRequest = {
  status_filter?: 'pending' | 'processing' | 'preprocessed' | 'processed' | 'failed' | null
  page: number
  page_size: number
  sort_field: 'created_at' | 'updated_at' | 'id' | 'file_path'
  sort_direction: 'asc' | 'desc'
}

type LightragApiModule = typeof import('./lightrag')
type SettingsModule = typeof import('../stores/settings')

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
let settingsModule: SettingsModule

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
  settingsModule = await import('../stores/settings')
})

afterEach(() => {
  apiModule.__resetPaginatedDocumentRequestsForTests()
  ;(apiModule as any).__resetConfigWorkbenchRequestsForTests?.()
  ;(apiModule as any).__resetGraphRequestsForTests?.()
  apiModule.__resetKBIterationRequestsForTests()
  settingsModule.useSettingsStore.setState({ graphViewMode: 'medical' })
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
    const [firstResult, secondResult] = await Promise.allSettled([firstRequest, secondRequest])
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

    await expect(apiModule.getDocumentsPaginatedWithTimeout(request, 1)).rejects.toThrow(
      'Document fetch timeout'
    )

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

describe('graph api', () => {
  test('builds graph query paths with medical browse enabled by default', () => {
    expect(apiModule.buildGraphQueryPath('COVID-19 & fever', 2, 150)).toBe(
      '/graphs?label=COVID-19%20%26%20fever&max_depth=2&max_nodes=150&medical_view=true&medical_browse=true'
    )
  })

  test('builds raw graph query paths without medical browse flags', () => {
    expect(apiModule.buildGraphQueryPath('COVID-19 & fever', 2, 150, 'raw')).toBe(
      '/graphs?label=COVID-19%20%26%20fever&max_depth=2&max_nodes=150'
    )
  })

  test('queries medical browse graph by default', async () => {
    let capturedPath: string | undefined

    ;(apiModule as any).__setGraphGetForTests(async (path: string) => {
      capturedPath = path
      return { nodes: [], edges: [] }
    })

    await expect(apiModule.queryGraphs('COVID-19 & fever', 2, 150)).resolves.toEqual({
      nodes: [],
      edges: []
    })

    expect(capturedPath).toBe(
      '/graphs?label=COVID-19%20%26%20fever&max_depth=2&max_nodes=150&medical_view=true&medical_browse=true'
    )
  })

  test('queries raw graph without medical browse flags when requested', async () => {
    let capturedPath: string | undefined

    ;(apiModule as any).__setGraphGetForTests(async (path: string) => {
      capturedPath = path
      return { nodes: [], edges: [] }
    })

    await expect(apiModule.queryGraphs('COVID-19 & fever', 2, 150, 'raw')).resolves.toEqual({
      nodes: [],
      edges: []
    })

    expect(capturedPath).toBe(
      '/graphs?label=COVID-19%20%26%20fever&max_depth=2&max_nodes=150'
    )
  })
})

describe('settings store graph view mode', () => {
  test('defaults to medical browsing mode and can switch to raw', () => {
    expect(settingsModule.useSettingsStore.getState().graphViewMode).toBe('medical')

    settingsModule.useSettingsStore.getState().setGraphViewMode('raw')

    expect(settingsModule.useSettingsStore.getState().graphViewMode).toBe('raw')
  })
})

describe('config workbench api', () => {
  test('gets the config workbench payload', async () => {
    let capturedEnvProfile: string | undefined
    let capturedPromptProfile: string | undefined
    const payload = {
      workspace: { current: 'project_a', dynamic_switching: false },
      env: { active_profile: '.env', profiles: [], sections: [] },
      prompts: {
        entity_type_active_profile: 'finance.yml',
        entity_type_profiles: [],
        stages: []
      },
      requires_restart: true
    }

    ;(apiModule as any).__setConfigWorkbenchGetForTests(
      async (envProfile?: string, promptProfile?: string) => {
        capturedEnvProfile = envProfile
        capturedPromptProfile = promptProfile
        return payload
      }
    )

    await expect(
      (apiModule as any).getConfigWorkbench('.env.local', 'medical.yml')
    ).resolves.toEqual(payload)
    expect(capturedEnvProfile).toBe('.env.local')
    expect(capturedPromptProfile).toBe('medical.yml')
  })

  test('updates env values through the config workbench endpoint', async () => {
    let capturedValues: Record<string, string | null> | undefined
    const response = { updated: ['LLM_MODEL'], requires_restart: true }

    ;(apiModule as any).__setConfigEnvPutForTests(async (values: Record<string, string | null>) => {
      capturedValues = values
      return response
    })

    await expect(
      (apiModule as any).updateEnvConfig({
        LLM_MODEL: 'gpt-5-mini',
        LLM_BINDING_API_KEY: ''
      })
    ).resolves.toEqual(response)

    expect(capturedValues).toEqual({
      LLM_MODEL: 'gpt-5-mini',
      LLM_BINDING_API_KEY: ''
    })
  })

  test('picks a workspace folder through the config workbench endpoint', async () => {
    let capturedRequest: { initial_dir?: string | null } | undefined
    const response = {
      selected_path: 'D:\\LightRAG\\inputs\\project_a',
      workspace: 'project_a',
      input_dir: 'D:\\LightRAG\\inputs'
    }

    ;(apiModule as any).__setConfigFolderPickPostForTests(
      async (request: { initial_dir?: string | null }) => {
        capturedRequest = request
        return response
      }
    )

    await expect(
      (apiModule as any).pickWorkspaceFolder({
        initial_dir: 'D:\\LightRAG\\inputs'
      })
    ).resolves.toEqual(response)

    expect(capturedRequest).toEqual({
      initial_dir: 'D:\\LightRAG\\inputs'
    })
  })

  test('updates an entity-type prompt profile', async () => {
    let capturedRequest: { profile: string; entity_types_guidance: string } | undefined
    const response = { profile: 'finance.yml', requires_restart: true }

    ;(apiModule as any).__setEntityPromptPutForTests(
      async (request: { profile: string; entity_types_guidance: string }) => {
        capturedRequest = request
        return response
      }
    )

    await expect(
      (apiModule as any).updateEntityTypePrompt({
        profile: 'finance.yml',
        entity_types_guidance: '- Company\n- Person'
      })
    ).resolves.toEqual(response)

    expect(capturedRequest).toEqual({
      profile: 'finance.yml',
      entity_types_guidance: '- Company\n- Person'
    })
  })
})

describe('kb iteration api', () => {
  test('kb llm review wrappers call expected paths', async () => {
    const getCalls: string[] = []
    const postCalls: Array<{ path: string; body: any }> = []
    apiModule.__setKBIterationGetForTests(async (path) => {
      getCalls.push(path)
      return { artifactKey: path, contentType: 'text/markdown', content: 'ok' }
    })
    apiModule.__setKBIterationPostForTests(async (path, body) => {
      postCalls.push({ path, body })
      return { workspace: 'demo', stopReason: 'pending_human_review', proposalIds: [] }
    })

    await apiModule.runKBIterationLLMReview('demo workspace', { max_review_rounds: 1 })
    await apiModule.getKBIterationLLMReviewTrace('demo workspace')
    await apiModule.getKBIterationLLMReviewReport('demo workspace')
    await apiModule.getKBIterationLLMReviewProposals('demo workspace')
    await apiModule.getKBIterationLLMJudgeReport('demo workspace')
    await apiModule.getKBIterationLLMReviewPatch('demo workspace', 'proposal 1')

    expect(postCalls[0]).toEqual({
      path: '/kb-iteration/demo%20workspace/llm-review/runs',
      body: { max_review_rounds: 1 }
    })
    expect(getCalls).toEqual([
      '/kb-iteration/demo%20workspace/llm-review/trace',
      '/kb-iteration/demo%20workspace/llm-review/report',
      '/kb-iteration/demo%20workspace/llm-review/proposals',
      '/kb-iteration/demo%20workspace/llm-review/judge-report',
      '/kb-iteration/demo%20workspace/llm-review/patches/proposal%201'
    ])
  })

  test('gets a workspace summary through an encoded kb-iteration path', async () => {
    let capturedPath: string | undefined
    const payload = {
      workspace: 'influenza medical',
      latestRunId: 'latest',
      phase: 'pending_user_review'
    }

    ;(apiModule as any).__setKBIterationGetForTests(async (path: string) => {
      capturedPath = path
      return payload
    })

    await expect((apiModule as any).getKBIterationSummary('influenza medical')).resolves.toEqual(
      payload
    )
    expect(capturedPath).toBe('/kb-iteration/influenza%20medical/summary')
  })

  test('posts proposal decisions without mutating facts directly', async () => {
    let capturedPath: string | undefined
    let capturedBody: Record<string, string> | undefined
    const response = { proposalId: 'p1', decision: 'reject' }

    ;(apiModule as any).__setKBIterationPostForTests(
      async (path: string, body?: Record<string, string>) => {
        capturedPath = path
        capturedBody = body
        return response
      }
    )

    await expect(
      (apiModule as any).recordKBIterationProposalDecision(
        'influenza_medical_v1',
        'p1',
        'reject',
        { reviewer: 'maintainer', reason: 'Needs source evidence' }
      )
    ).resolves.toEqual(response)

    expect(capturedPath).toBe('/kb-iteration/influenza_medical_v1/proposals/p1/reject')
    expect(capturedBody).toEqual({
      reviewer: 'maintainer',
      reason: 'Needs source evidence'
    })
  })
})
