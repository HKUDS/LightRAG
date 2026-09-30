import { afterEach, beforeEach, describe, expect, spyOn, test } from 'bun:test'
import { __setAxiosAdapterForTests, getAuthStatus, queryText, queryTextStream } from './lightrag'
import { useAuthStore } from '@/stores/state'
import { useWebUIAuthStore } from '@/stores/webuiAuth'
import { useAiContentNoticeStore } from '@/stores/aiContentNotice'
import { captureProcessState, restoreProcessState, type ProcessStateSnapshot } from '@/test/processState'
import { navigationService } from '@/services/navigation'

const stores = [useAuthStore, useWebUIAuthStore, useAiContentNoticeStore]
let snapshot: ProcessStateSnapshot
let nav: ReturnType<typeof spyOn>
let log: ReturnType<typeof spyOn>
let fetchSpy: ReturnType<typeof spyOn> | undefined
const installFetch = (implementation: (url: RequestInfo | URL, init?: RequestInit) => Promise<Response>) => {
  fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(implementation as typeof fetch)
}
const guestStatus = { auth_configured: false, api_key_configured: false, access_token: 'new-guest', core_version: '1', api_version: '1' }
const request = { query: 'hello', mode: 'mix' as const }
const ok = (config: any, data: any = {}, headers = {}) => ({ config, data, headers, status: 200, statusText: 'OK' })
const fail = (config: any, status: number, detail = 'Unauthorized'): never => {
  throw { config, response: { status, statusText: 'Error', data: { detail } } }
}

beforeEach(() => {
  snapshot = captureProcessState(stores)
  localStorage.setItem('settings-storage', JSON.stringify({ version: 22, state: { apiKey: 'legacy-secret' } }))
  localStorage.setItem('LIGHTRAG-API-TOKEN', 'old-token')
  useAuthStore.setState({ isAuthenticated: true, isGuestMode: false })
  useWebUIAuthStore.setState({ mode: 'account' })
  nav = spyOn(navigationService, 'navigateToUnauthenticated').mockImplementation(() => {})
  log = spyOn(console, 'error').mockImplementation(() => {})
})
afterEach(() => {
  __setAxiosAdapterForTests(undefined)
  fetchSpy?.mockRestore()
  fetchSpy = undefined
  nav.mockRestore()
  log.mockRestore()
  restoreProcessState(stores, snapshot)
})

describe('WebUI credential transport', () => {
  test('regular requests send only bearer; public discovery sends neither credential', async () => {
    const seen: any[] = []
    __setAxiosAdapterForTests(async (config: any) => {
      seen.push(config.headers.toJSON())
      return ok(config, guestStatus)
    })
    await queryText(request)
    await getAuthStatus()
    expect(seen[0].Authorization).toBe('Bearer old-token')
    expect(seen[0]['X-API-Key']).toBeUndefined()
    expect(seen[1].Authorization).toBeUndefined()
    expect(seen[1]['X-API-Key']).toBeUndefined()
    expect(seen[1]['X-Skip-Interceptor']).toBeUndefined()
  })

  test('guest Axios refresh retries once with only the refreshed bearer', async () => {
    useAuthStore.setState({ isGuestMode: true })
    const queries: any[] = []
    __setAxiosAdapterForTests(async (config: any) => {
      if (config.url === '/auth-status') {
        expect(config.headers.has('Authorization')).toBe(false)
        return ok(config, guestStatus)
      }
      queries.push(config.headers.toJSON())
      if (queries.length === 1) return fail(config, 401)
      return ok(config, { response: 'answer' })
    })
    expect(await queryText(request)).toEqual({ response: 'answer' })
    expect(queries.map(h => h.Authorization)).toEqual(['Bearer old-token', 'Bearer new-guest'])
    expect(queries.every(h => !('X-API-Key' in h))).toBe(true)
  })

  test.each(['axios', 'stream'] as const)('%s never activates or retries a guest in key-only mode', async (transport) => {
    useAuthStore.setState({ isGuestMode: true })
    let requests = 0
    let discoveries = 0
    __setAxiosAdapterForTests(async (config: any) => {
      if (config.url === '/auth-status') {
        discoveries++
        return ok(config, { ...guestStatus, api_key_configured: true })
      }
      requests++
      return fail(config, 401)
    })
    installFetch(async () => {
      requests++
      return new Response('{}', { status: 401 })
    })
    const call = transport === 'axios' ? queryText(request) : queryTextStream(request, () => {})
    await expect(call).rejects.toThrow()
    expect(requests).toBe(1)
    expect(discoveries).toBe(1)
    expect(useWebUIAuthStore.getState().mode).toBe('api-key-only')
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBe('old-token')
  })

  test('account 401 returns to login without guest refresh or key fallback', async () => {
    let calls = 0
    __setAxiosAdapterForTests(async (config: any) => { calls++; return fail(config, 401) })
    await expect(queryText(request)).rejects.toThrow()
    expect(calls).toBe(1)
    expect(nav).toHaveBeenCalledTimes(1)
  })

  test.each(['axios', 'stream'] as const)('%s rediscovers only exact key failures, not arbitrary 403s', async (transport) => {
    let discoveries = 0
    let detail = 'Permission denied'
    __setAxiosAdapterForTests(async (config: any) => {
      if (config.url === '/auth-status') { discoveries++; return ok(config, { ...guestStatus, api_key_configured: true }) }
      return fail(config, 403, detail)
    })
    installFetch(async () => new Response(JSON.stringify({ detail }), { status: 403 }))
    const call = () => transport === 'axios' ? queryText(request).catch(() => {}) : queryTextStream(request, () => {}, () => {})
    await call()
    expect(discoveries).toBe(0)
    expect(useWebUIAuthStore.getState().mode).toBe('account')
    detail = 'API Key required'
    await call()
    expect(discoveries).toBe(1)
    expect(useWebUIAuthStore.getState().mode).toBe('api-key-only')
  })

  test('stream initial and refreshed requests never transmit a persisted legacy key', async () => {
    useAuthStore.setState({ isGuestMode: true })
    __setAxiosAdapterForTests(async (config: any) => ok(config, guestStatus))
    const headers: Headers[] = []
    installFetch(async (_url, init) => {
      headers.push(new Headers(init?.headers))
      return headers.length === 1 ? new Response('{}', { status: 401 }) : new Response('{"response":"answer"}\n')
    })
    await queryTextStream(request, () => {})
    expect(headers.map(h => h.get('Authorization'))).toEqual(['Bearer old-token', 'Bearer new-guest'])
    expect(headers.every(h => !h.has('X-API-Key'))).toBe(true)
  })

  test.each([false, true])('existing token renewal is retained for guest=%s', async (guest) => {
    useAuthStore.setState({ isGuestMode: guest })
    const exp = 4102444800
    const token = `header.${btoa(JSON.stringify({ sub: guest ? 'guest' : 'alice', exp }))}.signature`
    const seen: string[] = []
    __setAxiosAdapterForTests(async (config: any) => {
      seen.push(config.headers.get('Authorization'))
      return ok(config, { response: 'answer' }, { 'x-new-token': token })
    })
    await queryText(request)
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBe(token)
    expect(useAuthStore.getState().tokenExpiresAt).toBe(exp * 1000)
    await queryText(request)
    expect(seen).toEqual(['Bearer old-token', `Bearer ${token}`])
  })

  test.each(['', '   ', undefined])('discovery rejects unusable guest token %j', async (accessToken) => {
    useWebUIAuthStore.setState({ mode: 'unknown' })
    __setAxiosAdapterForTests(async (config: any) => ok(config, { ...guestStatus, access_token: accessToken }))
    await expect(getAuthStatus()).rejects.toThrow('Guest access requires a guest token')
    expect(useWebUIAuthStore.getState().mode).toBe('unknown')
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBe('old-token')
  })

  test('discovery failures do not recurse into refresh or retry loops', async () => {
    useAuthStore.setState({ isGuestMode: true })
    let calls = 0
    __setAxiosAdapterForTests(async (config: any) => { calls++; return fail(config, 403, 'API Key required') })
    await expect(getAuthStatus()).rejects.toBeDefined()
    expect(calls).toBe(1)
  })
})
