import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import { renderWithProviders } from '@/test/render'
import { seedCustomization } from '@/test/customization'
import { captureProcessState, restoreProcessState, type ProcessStateSnapshot } from '@/test/processState'
import { useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useWebUIAuthStore } from '@/stores/webuiAuth'
import { useCustomizationStore } from '@/stores/customization'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import { __setAxiosAdapterForTests } from '@/api/lightrag'
import type { ComponentType } from 'react'
import { toast } from 'sonner'

import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'

// Entry bootstraps register process-lifetime listeners once. Run these cases
// in a child so their module cache and navigation policy cannot poison other
// bootstrap tests, while keeping this rendered test colocated with the routers.
if (process.env.LIGHTRAG_WEBUI_ENTRY_TEST_CHILD !== '1') {
  test('both WebUI entries obey account/guest/key-only discovery', () => {
    const result = spawnSync(process.execPath, ['test', './src/webuiAuthEntries.test.tsx'], {
      cwd: resolve(import.meta.dir, '..'), encoding: 'utf8', timeout: 30000,
      env: { ...process.env, LIGHTRAG_WEBUI_ENTRY_TEST_CHILD: '1' }
    })
    expect({ exit: result.status, output: result.status === 0 ? '' : result.stdout + result.stderr })
      .toEqual({ exit: 0, output: '' })
  }, 35000)
} else {
  // Keep both real routers, the boundary, login and welcome pages. Only the
  // authenticated product views are stubs: no graph rendering or backend polling.
  let AppRouter: ComponentType
  let WorkspaceRouter: ComponentType
  let realApp: typeof import('@/App')
  let realWorkspace: typeof import('@/features/workspace/WorkspaceApp')
  beforeAll(async () => {
    // Sigma reads these constants at module evaluation; product views are not
    // mounted in this test. Do not leave a fake WebGL context in the process.
    const glNames = ['WebGLRenderingContext', 'WebGL2RenderingContext']
    const descriptors = glNames.map(name => Object.getOwnPropertyDescriptor(globalThis, name))
    for (const name of glNames) Object.defineProperty(globalThis, name, { configurable: true, value: {
      BOOL: 35670, BYTE: 5120, UNSIGNED_BYTE: 5121, SHORT: 5122,
      UNSIGNED_SHORT: 5123, INT: 5124, UNSIGNED_INT: 5125, FLOAT: 5126
    } })
    try { realApp = { ...(await import('@/App')) } } finally {
      glNames.forEach((name, i) => {
        if (descriptors[i]) Object.defineProperty(globalThis, name, descriptors[i]!)
        else Reflect.deleteProperty(globalThis, name)
      })
    }
    realWorkspace = { ...(await import('@/features/workspace/WorkspaceApp')) }
    mock.module('@/App', () => ({ ...realApp, default: () => <div>Protected admin</div> }))
    mock.module('@/features/workspace/WorkspaceApp', () => ({ ...realWorkspace, default: () => <div>Protected workspace</div> }))
    AppRouter = (await import('@/AppRouter')).default
    WorkspaceRouter = (await import('@/WorkspaceAppRouter')).default
  }, 30000)
  afterAll(() => {
    mock.module('@/App', () => realApp)
    mock.module('@/features/workspace/WorkspaceApp', () => realWorkspace)
  })

  const stores = [useAuthStore, useSettingsStore, useWebUIAuthStore, useCustomizationStore,
    useWebuiRetrievalHistoryStore, useWorkspaceRetrievalHistoryStore]
  let snapshot: ProcessStateSnapshot
  let originalHash: string
  let calls: string[]
  let status: Record<string, unknown>
  const guestToken = `header.${btoa(JSON.stringify({ sub: 'guest', role: 'guest', exp: 4102444800 }))}.signature`

  beforeEach(() => {
    snapshot = captureProcessState(stores)
    originalHash = window.location.hash
    window.location.hash = '#/'
    localStorage.removeItem('LIGHTRAG-API-TOKEN')
    useAuthStore.setState({ isAuthenticated: false, isGuestMode: false, username: null })
    useWebUIAuthStore.setState({ mode: 'unknown' })
    useSettingsStore.setState({ theme: 'light', language: 'en', languageUserSelected: true })
    seedCustomization()
    calls = []
    status = { auth_configured: false, api_key_configured: true, access_token: guestToken,
      auth_mode: 'disabled', core_version: '1', api_version: '1' }
    __setAxiosAdapterForTests(async (config: any) => {
      calls.push(config.url)
      return { data: { ...status }, status: 200, statusText: 'OK', headers: {}, config }
    })
  })
  afterEach(() => {
    act(() => toast.dismiss())
    cleanup()
    __setAxiosAdapterForTests(undefined)
    restoreProcessState(stores, snapshot)
    window.location.hash = originalHash
  })

  for (const entry of ['admin', 'workspace'] as const) {
    const renderEntry = () => renderWithProviders(entry === 'admin' ? <AppRouter /> : <WorkspaceRouter />)
    describe(`${entry} WebUI auth entry`, () => {
      test.each([false, true])('key-only blocks protected views, even with a persisted session=%s', async (authenticated) => {
        if (authenticated) {
          localStorage.setItem('LIGHTRAG-API-TOKEN', 'old-token')
          useAuthStore.setState({ isAuthenticated: true })
        }
        renderEntry()
        await screen.findByRole('alert')
        expect(screen.getByRole('alert').textContent).toContain('AUTH_ACCOUNTS')
        expect(screen.getByRole('alert').textContent).toContain('TOKEN_SECRET')
        expect(screen.queryAllByText(/Protected (admin|workspace)/)).toHaveLength(0)
        expect(screen.queryAllByRole('textbox')).toHaveLength(0)
        expect(screen.queryAllByRole('dialog')).toHaveLength(0)
        await new Promise(resolve => setTimeout(resolve, 40))
        expect(calls).toEqual(['/auth-status'])
        expect(useAuthStore.getState().isGuestMode).toBe(false)
        expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBe(authenticated ? 'old-token' : null)
      })

      test('missing capability metadata is not treated as an open server', async () => {
        delete status.api_key_configured
        renderEntry()
        const alert = await screen.findByRole('alert')
        expect(alert.textContent).toContain('matching')
        expect(calls).toEqual(['/auth-status'])
        expect(useAuthStore.getState().isAuthenticated).toBe(false)
      })

      test('manual recheck allows an administrator to enable account login', async () => {
        renderEntry()
        await screen.findByRole('alert')
        status = { ...status, auth_configured: true }
        fireEvent.click(screen.getByRole('button', { name: 'Check again' }))
        if (entry === 'workspace') {
          const signIn = await screen.findByRole('button', { name: 'Sign in' })
          await waitFor(() => expect(signIn).toBeEnabled())
          fireEvent.click(signIn)
        }
        await waitFor(() => expect(screen.queryAllByLabelText(/username/i).length).toBeGreaterThan(0))
        expect(screen.queryAllByRole('alert')).toHaveLength(0)
        expect(screen.queryAllByLabelText(/api key/i)).toHaveLength(0)
      })

      test('account form login uses only the returned user session', async () => {
        status.auth_configured = true
        status.auth_mode = 'enabled'
        const token = `header.${btoa(JSON.stringify({ sub: 'alice', role: 'user', exp: 4102444800 }))}.signature`
        let loginHeaders: Record<string, unknown> = {}
        __setAxiosAdapterForTests(async (config: any) => {
          if (config.url === '/login') loginHeaders = config.headers.toJSON()
          return { data: { ...status, access_token: token }, status: 200, statusText: 'OK', headers: {}, config }
        })
        window.location.hash = '#/login'
        renderEntry()
        fireEvent.change(await screen.findByLabelText('Username'), { target: { value: 'alice' } })
        fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password' } })
        fireEvent.click(screen.getByRole('button', { name: 'Login' }))
        await screen.findByText(`Protected ${entry}`)
        expect(useAuthStore.getState().isGuestMode).toBe(false)
        expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBe(token)
        expect(loginHeaders['X-API-Key']).toBeUndefined()
        expect(loginHeaders.Authorization).toBeUndefined()
      })

      test('a deployment becoming key-only during login cannot activate its guest response', async () => {
        status.auth_configured = true
        status.auth_mode = 'enabled'
        window.location.hash = '#/login'
        renderEntry()
        fireEvent.change(await screen.findByLabelText('Username'), { target: { value: 'alice' } })
        fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password' } })
        status.auth_configured = false
        status.auth_mode = 'disabled'
        fireEvent.click(screen.getByRole('button', { name: 'Login' }))
        await screen.findByRole('alert')
        expect(useAuthStore.getState().isAuthenticated).toBe(false)
        expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBeNull()
      })

      test('a stale guest session cannot skip account login', async () => {
        status.auth_configured = true
        localStorage.setItem('LIGHTRAG-API-TOKEN', guestToken)
        useAuthStore.setState({ isAuthenticated: true, isGuestMode: true })
        renderEntry()
        await waitFor(() => expect(useAuthStore.getState().isAuthenticated).toBe(false))
        expect(screen.queryAllByText(/Protected (admin|workspace)/)).toHaveLength(0)
      })
    })
  }

  test('open admin automatically activates a guest token', async () => {
    status.api_key_configured = false
    renderWithProviders(<AppRouter />)
    await screen.findByText('Protected admin')
    expect(useAuthStore.getState().isGuestMode).toBe(true)
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBe(guestToken)
  })

  test('open workspace requires the welcome-page action before activating a guest', async () => {
    status.api_key_configured = false
    renderWithProviders(<WorkspaceRouter />)
    const enter = await screen.findByRole('button', { name: 'Enter workspace' })
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBeNull()
    expect(useAuthStore.getState().isAuthenticated).toBe(false)
    fireEvent.click(enter)
    await screen.findByText('Protected workspace')
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBe(guestToken)
    expect(useAuthStore.getState().isGuestMode).toBe(true)
    act(() => useWebUIAuthStore.setState({ mode: 'api-key-only' }))
    await screen.findByRole('alert')
    expect(screen.queryAllByText('Protected workspace')).toHaveLength(0)
  })
}
