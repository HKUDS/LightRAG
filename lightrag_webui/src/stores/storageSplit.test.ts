import { beforeAll, describe, expect, test } from 'bun:test'
import {
  QUERY_SETTINGS_STORAGE_KEY,
  WEBUI_RETRIEVAL_HISTORY_KEY,
  WORKSPACE_RETRIEVAL_HISTORY_KEY
} from '@/lib/storageKeys'
import type { RetrievalHistoryStore } from './retrievalHistory'

/**
 * Storage-split isolation: the three split keys have disjoint writers, so a
 * write through one store must never touch the other keys — the whole-
 * envelope write-back race of the legacy single key is gone by construction.
 *
 * Isolation rule this file follows: zustand v5's createJSONStorage captures
 * the Storage object ONCE at store creation, and module-cached store
 * singletons may have been created by an EARLIER test file under a different
 * globalThis.localStorage stub. Swapping the global here would not rebind
 * them (reads would hit the wrong object and return null). So:
 * - the two history stores are FRESH instances built via the exported
 *   factory AFTER this file's stub is installed — they are provably bound
 *   to `stub`;
 * - the (possibly cached) query-settings singleton is only ever accessed
 *   through its OWN `persist.getOptions().storage`, wherever that lives.
 */

const makeStub = () => {
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

let stub: ReturnType<typeof makeStub>
let webuiHistory: RetrievalHistoryStore
let workspaceHistory: RetrievalHistoryStore
let querySettings: typeof import('./querySettings').useQuerySettingsStore

/** The query-settings singleton's own persisted envelope (parsed), or null. */
const readQuerySettingsEnvelope = async () => {
  const storage = querySettings.persist.getOptions().storage!
  return await storage.getItem(QUERY_SETTINGS_STORAGE_KEY)
}

beforeAll(async () => {
  stub = makeStub()
  Object.defineProperty(globalThis, 'localStorage', {
    value: stub,
    configurable: true
  })
  if (typeof sessionStorage === 'undefined') {
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: makeStub(),
      configurable: true
    })
  }

  // Fresh instances, created NOW: createRetrievalHistoryStore calls
  // createJSONStorage at store-creation time, so these bind to `stub`
  // regardless of what any module-cached singleton captured earlier.
  const { createRetrievalHistoryStore } = await import('./retrievalHistory')
  webuiHistory = createRetrievalHistoryStore(WEBUI_RETRIEVAL_HISTORY_KEY)
  workspaceHistory = createRetrievalHistoryStore(WORKSPACE_RETRIEVAL_HISTORY_KEY)

  querySettings = (await import('./querySettings')).useQuerySettingsStore
})

describe('disjoint writers', () => {
  test('workspace history writes touch ONLY the workspace history key', async () => {
    const settingsBefore = JSON.stringify(await readQuerySettingsEnvelope())
    const webuiBefore = stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)

    workspaceHistory
      .getState()
      .setHistory([{ id: 'w1', role: 'user', content: 'workspace question' }])

    // Its own key (in this file's stub) got the write…
    expect(
      JSON.parse(stub.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)!).state.history
    ).toHaveLength(1)
    // …and neither sibling key moved: not in this stub, and not wherever the
    // query-settings singleton actually persists.
    expect(stub.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    expect(stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).toBe(webuiBefore)
    expect(JSON.stringify(await readQuerySettingsEnvelope())).toBe(settingsBefore)
  })

  test('the two histories are invisible to each other', () => {
    webuiHistory
      .getState()
      .setHistory([{ id: 'a1', role: 'user', content: 'admin debug' }])

    expect(workspaceHistory.getState().history.map((m) => m.id)).toEqual(['w1'])
    expect(webuiHistory.getState().history.map((m) => m.id)).toEqual(['a1'])
    expect(
      JSON.parse(stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)!).state.history[0].id
    ).toBe('a1')
    expect(
      JSON.parse(stub.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)!).state.history[0].id
    ).toBe('w1')

    // Clearing one side leaves the other side's history untouched.
    workspaceHistory.getState().clearHistory()
    expect(workspaceHistory.getState().history).toEqual([])
    expect(webuiHistory.getState().history.map((m) => m.id)).toEqual(['a1'])
    expect(
      JSON.parse(stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)!).state.history[0].id
    ).toBe('a1')
  })

  test('query settings writes touch ONLY the query-settings key', async () => {
    const webuiBefore = stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)
    const workspaceBefore = stub.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)

    querySettings.getState().updateQuerySettings({ top_k: 99 })

    // Read back through the singleton's OWN storage binding.
    const envelope = (await readQuerySettingsEnvelope()) as {
      state: { querySettings: { top_k: number } }
    }
    expect(envelope.state.querySettings.top_k).toBe(99)
    // The history keys in this file's stub did not move.
    expect(stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).toBe(webuiBefore)
    expect(stub.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)).toBe(workspaceBefore)
  })

  test('history_turns stays pinned to 0 through updates', () => {
    querySettings.getState().updateQuerySettings({ history_turns: 7 } as never)
    expect(querySettings.getState().querySettings.history_turns).toBe(0)
  })
})
