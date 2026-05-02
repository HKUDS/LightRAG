/**
 * Workspace API Tests
 *
 * Tests for sanitizeHeader, getWorkspaces, and header injection logic.
 *
 * IMPORTANT: All imports of modules that depend on localStorage must be dynamic
 * (inside beforeAll) because import statements are hoisted before runtime code.
 */

import { afterEach, beforeAll, describe, expect, test, mock } from 'bun:test'
import type { Workspace } from './lightrag'

type LightragApiModule = typeof import('./lightrag')
type SettingsModule = typeof import('@/stores/settings')

const storageMock = () => {
  const data = new Map<string, string>()
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
    removeItem: (key: string) => data.delete(key),
    clear: () => data.clear()
  }
}

let api: LightragApiModule
let settings: SettingsModule

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })

  // Dynamic import ensures localStorage is set before module evaluation
  api = await import('./lightrag')
  settings = await import('@/stores/settings')
})

describe('sanitizeHeader', () => {
  test('returns null when input is null', () => {
    expect(api.sanitizeHeader(null)).toBeNull()
  })

  test('returns the string unchanged when no CRLF characters', () => {
    expect(api.sanitizeHeader('my-workspace')).toBe('my-workspace')
    expect(api.sanitizeHeader('workspace-123')).toBe('workspace-123')
    expect(api.sanitizeHeader('my workspace with spaces')).toBe('my workspace with spaces')
  })

  test('strips \\r from the value', () => {
    expect(api.sanitizeHeader('my\rworkspace')).toBe('myworkspace')
    expect(api.sanitizeHeader('\rworkspace')).toBe('workspace')
    expect(api.sanitizeHeader('workspace\r')).toBe('workspace')
  })

  test('strips \\n from the value', () => {
    expect(api.sanitizeHeader('my\nworkspace')).toBe('myworkspace')
    expect(api.sanitizeHeader('\nworkspace')).toBe('workspace')
    expect(api.sanitizeHeader('workspace\n')).toBe('workspace')
  })

  test('strips both \\r\\n from the value', () => {
    expect(api.sanitizeHeader('my\r\nworkspace')).toBe('myworkspace')
    expect(api.sanitizeHeader('\r\nworkspace')).toBe('workspace')
    expect(api.sanitizeHeader('workspace\r\n')).toBe('workspace')
  })

  test('handles string with multiple CRLF sequences', () => {
    expect(api.sanitizeHeader('my\r\nwork\r\nspace')).toBe('myworkspace')
    expect(api.sanitizeHeader('\r\n\r\n')).toBe('')
    expect(api.sanitizeHeader('a\rb\rc')).toBe('abc')
    expect(api.sanitizeHeader('a\nb\nc')).toBe('abc')
    expect(api.sanitizeHeader('a\r\nb\r\nc')).toBe('abc')
  })

  test('returns empty string when input is empty string (not null)', () => {
    expect(api.sanitizeHeader('')).toBe('')
  })
})

describe('getWorkspaces', () => {
  const mockWorkspaces: Workspace[] = [
    { name: 'workspace-1', first_seen: '2024-01-01', last_seen: '2024-01-15' },
    { name: 'workspace-2', first_seen: '2024-01-02', last_seen: '2024-01-16' }
  ]

  let originalGet: typeof api.axiosInstance.get

  beforeAll(() => {
    originalGet = api.axiosInstance.get
  })

  afterEach(() => {
    api.axiosInstance.get = originalGet
  })

  test('returns workspace array on valid response { workspaces: [...] }', async () => {
    api.axiosInstance.get = mock(async () => ({
      data: { workspaces: mockWorkspaces }
    })) as typeof api.axiosInstance.get

    const result = await api.getWorkspaces()
    expect(result).toEqual(mockWorkspaces)
  })

  test('throws Error when response has no workspaces field', async () => {
    api.axiosInstance.get = mock(async () => ({
      data: {}
    })) as typeof api.axiosInstance.get

    await expect(api.getWorkspaces()).rejects.toThrow('Invalid workspaces response')
  })

  test('throws Error when workspaces is not an array', async () => {
    api.axiosInstance.get = mock(async () => ({
      data: { workspaces: 'not-an-array' }
    })) as typeof api.axiosInstance.get

    await expect(api.getWorkspaces()).rejects.toThrow('Invalid workspaces response')
  })

  test('throws Error when response is null/undefined', async () => {
    api.axiosInstance.get = mock(async () => ({
      data: null
    })) as typeof api.axiosInstance.get

    await expect(api.getWorkspaces()).rejects.toThrow('Invalid workspaces response')
  })
})

describe('Header injection (axios interceptor)', () => {
  afterEach(() => {
    settings.useSettingsStore.getState().setCurrentWorkspace(null)
  })

  test('when currentWorkspace is null, LIGHTRAG-WORKSPACE header is NOT set', () => {
    settings.useSettingsStore.getState().setCurrentWorkspace(null)

    const config = {
      headers: {} as Record<string, string>
    }

    const workspace = api.sanitizeHeader(settings.useSettingsStore.getState().currentWorkspace)
    expect(workspace).toBeNull()

    if (workspace) {
      config.headers['LIGHTRAG-WORKSPACE'] = workspace
    }

    expect(config.headers['LIGHTRAG-WORKSPACE']).toBeUndefined()
  })

  test('when currentWorkspace is a string, LIGHTRAG-WORKSPACE header IS set with that value', () => {
    settings.useSettingsStore.getState().setCurrentWorkspace('my-test-workspace')

    const config = {
      headers: {} as Record<string, string>
    }

    const workspace = api.sanitizeHeader(settings.useSettingsStore.getState().currentWorkspace)
    expect(workspace).toBe('my-test-workspace')

    if (workspace) {
      config.headers['LIGHTRAG-WORKSPACE'] = workspace
    }

    expect(config.headers['LIGHTRAG-WORKSPACE']).toBe('my-test-workspace')
  })

  test('when currentWorkspace contains \\r\\n, header value has them stripped', () => {
    settings.useSettingsStore.getState().setCurrentWorkspace('my-workspace\r\ninjected')

    expect(settings.useSettingsStore.getState().currentWorkspace).toBe('my-workspace\r\ninjected')

    const workspace = api.sanitizeHeader(settings.useSettingsStore.getState().currentWorkspace)
    if (workspace) {
      expect(workspace).toBe('my-workspaceinjected')
      expect(workspace).not.toContain('\r')
      expect(workspace).not.toContain('\n')
    }
  })

  test('header is sanitized via sanitizeHeader before being set', () => {
    const testCases = [
      { input: 'workspace\rwith-cr' as string | null, expected: 'workspacewith-cr' },
      { input: 'workspace\nwith-lf' as string | null, expected: 'workspacewith-lf' },
      { input: 'workspace\r\nwith-crlf' as string | null, expected: 'workspacewith-crlf' },
      { input: 'normal-workspace' as string | null, expected: 'normal-workspace' },
      { input: null, expected: null }
    ]

    for (const { input, expected } of testCases) {
      if (input === null) {
        settings.useSettingsStore.getState().setCurrentWorkspace(null)
      } else {
        settings.useSettingsStore.getState().setCurrentWorkspace(input)
      }

      const rawValue = settings.useSettingsStore.getState().currentWorkspace
      const sanitizedValue = api.sanitizeHeader(rawValue)
      expect(sanitizedValue).toBe(expected)
    }
  })
})
