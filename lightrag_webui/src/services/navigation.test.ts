import { afterAll, afterEach, beforeAll, describe, expect, mock, test } from 'bun:test'

/**
 * The shared navigation singleton must be configured EXPLICITLY per entry:
 * 401/logout land on '#/welcome' under the workspace bootstrap and '#/login'
 * under the admin bootstrap — and an UNCONFIGURED service must fail loudly
 * instead of silently falling back to the admin default (that failure mode
 * would only surface on the workspace entry's 401 path).
 */

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

let navigationModule: typeof import('./navigation')
let realApiModule: Record<string, unknown>

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    checkHealth: mock(async () => ({ status: 'error', message: 'offline' }))
  }))
  navigationModule = await import('./navigation')
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

afterEach(() => {
  navigationModule.navigationService.resetForTests()
})

describe('per-entry unauthenticated targets', () => {
  test('workspace bootstrap: 401/logout navigates to /welcome', () => {
    const targets: string[] = []
    const cleared: string[] = []
    const service = navigationModule.navigationService
    service.configureEntry({
      unauthenticatedRoute: '/welcome',
      clearRetrievalHistory: () => cleared.push('workspace')
    })
    service.setNavigate(((to: string) => targets.push(to)) as any)

    service.navigateToUnauthenticated()
    expect(targets).toEqual(['/welcome'])
    // preserveHistory=true on the unauthenticated path: history NOT cleared.
    expect(cleared).toEqual([])
  })

  test('admin bootstrap: 401/logout navigates to /login and runs reset adapters', () => {
    const targets: string[] = []
    const adapterRuns: string[] = []
    const service = navigationModule.navigationService
    service.configureEntry({
      unauthenticatedRoute: '/login',
      resetAdapters: [() => adapterRuns.push('graph-reset')],
      clearRetrievalHistory: () => {}
    })
    service.setNavigate(((to: string) => targets.push(to)) as any)

    service.navigateToUnauthenticated()
    expect(targets).toEqual(['/login'])
    expect(adapterRuns).toEqual(['graph-reset'])
  })

  test('explicit reset (no preserve) clears the ENTRY-INJECTED history only', () => {
    const cleared: string[] = []
    const service = navigationModule.navigationService
    service.configureEntry({
      unauthenticatedRoute: '/welcome',
      clearRetrievalHistory: () => cleared.push('workspace')
    })
    service.resetAllApplicationState(false)
    expect(cleared).toEqual(['workspace'])
  })

  test('unconfigured service throws instead of silently using the admin default', () => {
    const service = navigationModule.navigationService
    service.setNavigate((() => {}) as any)
    expect(() => service.navigateToUnauthenticated()).toThrow(/not configured/)
    expect(() => service.resetAllApplicationState()).toThrow(/not configured/)
  })

  test('navigation core works with NO reset adapter registered (workspace has none)', () => {
    const service = navigationModule.navigationService
    service.configureEntry({
      unauthenticatedRoute: '/welcome',
      clearRetrievalHistory: () => {}
    })
    expect(() => service.resetAllApplicationState(true)).not.toThrow()
  })
})
