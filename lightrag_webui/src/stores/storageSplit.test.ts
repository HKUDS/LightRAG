import { beforeAll, describe, expect, test } from 'bun:test'
import {
  QUERY_SETTINGS_STORAGE_KEY,
  WEBUI_RETRIEVAL_HISTORY_KEY,
  WORKSPACE_RETRIEVAL_HISTORY_KEY
} from '@/lib/storageKeys'

/**
 * Storage-split isolation: the three split keys have disjoint writers, so a
 * write through one store must never touch the other keys — the whole-
 * envelope write-back race of the legacy single key is gone by construction.
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
    },
    _data: data
  }
}

let storage: ReturnType<typeof storageMock>
let webuiHistory: typeof import('./webuiRetrievalHistory').useWebuiRetrievalHistoryStore
let workspaceHistory: typeof import('./workspaceRetrievalHistory').useWorkspaceRetrievalHistoryStore
let querySettings: typeof import('./querySettings').useQuerySettingsStore

beforeAll(async () => {
  storage = storageMock()
  Object.defineProperty(globalThis, 'localStorage', {
    value: storage,
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })
  webuiHistory = (await import('./webuiRetrievalHistory')).useWebuiRetrievalHistoryStore
  workspaceHistory = (await import('./workspaceRetrievalHistory'))
    .useWorkspaceRetrievalHistoryStore
  querySettings = (await import('./querySettings')).useQuerySettingsStore
})

describe('disjoint writers', () => {
  test('workspace history writes touch ONLY the workspace history key', () => {
    const settingsBefore = storage.getItem(QUERY_SETTINGS_STORAGE_KEY)
    const webuiBefore = storage.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)

    workspaceHistory
      .getState()
      .setHistory([{ id: 'w1', role: 'user', content: 'workspace question' }])

    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBe(settingsBefore)
    expect(storage.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).toBe(webuiBefore)
    expect(
      JSON.parse(storage.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)!).state.history
    ).toHaveLength(1)
  })

  test('the two histories are invisible to each other', () => {
    webuiHistory
      .getState()
      .setHistory([{ id: 'a1', role: 'user', content: 'admin debug' }])

    expect(workspaceHistory.getState().history.map((m) => m.id)).toEqual(['w1'])
    expect(webuiHistory.getState().history.map((m) => m.id)).toEqual(['a1'])

    // Clearing one side leaves the other side's history untouched.
    workspaceHistory.getState().clearHistory()
    expect(workspaceHistory.getState().history).toEqual([])
    expect(webuiHistory.getState().history.map((m) => m.id)).toEqual(['a1'])
  })

  test('query settings writes touch ONLY the query-settings key', () => {
    const webuiBefore = storage.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)
    const workspaceBefore = storage.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)

    querySettings.getState().updateQuerySettings({ top_k: 99 })

    expect(
      JSON.parse(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)!).state.querySettings.top_k
    ).toBe(99)
    expect(storage.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).toBe(webuiBefore)
    expect(storage.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)).toBe(workspaceBefore)
  })

  test('history_turns stays pinned to 0 through updates', () => {
    querySettings.getState().updateQuerySettings({ history_turns: 7 } as any)
    expect(querySettings.getState().querySettings.history_turns).toBe(0)
  })
})
