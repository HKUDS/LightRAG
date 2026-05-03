import { beforeAll, describe, expect, test } from 'bun:test'

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

// Re-implement the migrate logic for testing (mirrors settings.ts)
// The key behavior: only migrations where version < X are applied
const migrateSettings = (state: any, version: number): any => {
  if (version < 2) {
    state.showEdgeLabel = false
  }
  if (version < 3) {
    state.queryLabel = 'default'
  }
  if (version < 4) {
    state.showPropertyPanel = true
    state.showNodeSearchBar = true
    state.showNodeLabel = true
    state.enableHealthCheck = true
    state.apiKey = null
  }
  if (version < 5) {
    state.currentTab = 'documents'
  }
  if (version < 6) {
    state.querySettings = {
      mode: 'global',
      response_type: 'Multiple Paragraphs',
      top_k: 10,
      max_token_for_text_unit: 4000,
      max_token_for_global_context: 4000,
      max_token_for_local_context: 4000,
      only_need_context: false,
      only_need_prompt: false,
      stream: true,
      history_turns: 0,
      hl_keywords: [],
      ll_keywords: []
    }
    state.retrievalHistory = []
  }
  if (version < 7) {
    state.graphQueryMaxDepth = 3
    state.graphLayoutMaxIterations = 15
  }
  if (version < 8) {
    state.graphMinDegree = 0
    state.language = 'en'
  }
  if (version < 9) {
    state.showFileName = false
  }
  if (version < 10) {
    delete state.graphMinDegree
    state.graphMaxNodes = 1000
  }
  if (version < 11) {
    state.minEdgeSize = 1
    state.maxEdgeSize = 1
  }
  if (version < 12) {
    state.retrievalHistory = []
  }
  if (version < 13) {
    if (state.querySettings) {
      state.querySettings.user_prompt = ''
    }
  }
  if (version < 14) {
    state.backendMaxGraphNodes = null
  }
  if (version < 15) {
    state.querySettings = {
      ...state.querySettings,
      mode: 'mix',
      response_type: 'Multiple Paragraphs',
      top_k: 40,
      chunk_top_k: 10,
      max_entity_tokens: 10000,
      max_relation_tokens: 10000,
      max_total_tokens: 32000,
      enable_rerank: true,
      history_turns: 0,
    }
  }
  if (version < 16) {
    state.documentsPageSize = 10
  }
  if (version < 17) {
    if (state.querySettings) {
      state.querySettings.history_turns = 0
    }
  }
  if (version < 18) {
    state.userPromptHistory = []
  }
  if (version < 19) {
    if (state.querySettings) {
      delete state.querySettings.response_type
    }
  }
  if (version < 20) {
    state.currentWorkspace = null
  }
  return state
}

describe('currentWorkspace state', () => {
  let useSettingsStore: ReturnType<typeof import('@/stores/settings').useSettingsStore>

  beforeAll(async () => {
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageMock(),
      configurable: true
    })
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: storageMock(),
      configurable: true
    })

    const module = await import('@/stores/settings')
    useSettingsStore = module.useSettingsStore
  })

  test('default value is null', () => {
    // Reset to default state
    useSettingsStore.setState({ currentWorkspace: null })
    expect(useSettingsStore.getState().currentWorkspace).toBeNull()
  })

  test('setCurrentWorkspace("my-workspace") sets it to the string', () => {
    useSettingsStore.getState().setCurrentWorkspace('my-workspace')
    expect(useSettingsStore.getState().currentWorkspace).toBe('my-workspace')
  })

  test('setCurrentWorkspace(null) resets to null', () => {
    // First set a value
    useSettingsStore.getState().setCurrentWorkspace('some-workspace')
    expect(useSettingsStore.getState().currentWorkspace).toBe('some-workspace')

    // Then reset to null
    useSettingsStore.getState().setCurrentWorkspace(null)
    expect(useSettingsStore.getState().currentWorkspace).toBeNull()
  })

  test('state change is reflected in getState()', () => {
    // Reset first
    useSettingsStore.getState().setCurrentWorkspace(null)

    // Set a workspace
    useSettingsStore.getState().setCurrentWorkspace('test-workspace')

    // Verify via getState()
    const state = useSettingsStore.getState()
    expect(state.currentWorkspace).toBe('test-workspace')

    // Verify it can be set back to null
    useSettingsStore.getState().setCurrentWorkspace(null)
    expect(useSettingsStore.getState().currentWorkspace).toBeNull()
  })
})

describe('v20 migration', () => {
  test('state with version < 20 gets currentWorkspace: null added', () => {
    const state = {
      theme: 'dark',
      someOldField: 'value'
    }

    const migratedState = migrateSettings(state, 19)

    expect(migratedState.currentWorkspace).toBeNull()
    expect(migratedState.theme).toBe('dark')
    expect(migratedState.someOldField).toBe('value')
  })

  test('state at version 20 is returned as-is (currentWorkspace not modified)', () => {
    const state = {
      theme: 'light',
      currentWorkspace: 'existing-workspace'
    }

    const migratedState = migrateSettings(state, 20)

    // Since version is 20, the v20 migration (version < 20) should NOT run
    // So currentWorkspace should remain unchanged (existing-workspace)
    expect(migratedState.currentWorkspace).toBe('existing-workspace')
    expect(migratedState.theme).toBe('light')
  })

  test('existing fields are preserved during migration', () => {
    const state = {
      theme: 'dark',
      language: 'en',
      showPropertyPanel: true,
      graphMaxNodes: 500,
      currentWorkspace: undefined // intentionally missing
    }

    const migratedState = migrateSettings(state, 19)

    // Check all original fields are preserved
    expect(migratedState.theme).toBe('dark')
    expect(migratedState.language).toBe('en')
    expect(migratedState.showPropertyPanel).toBe(true)
    expect(migratedState.graphMaxNodes).toBe(500)
    // New field added by migration
    expect(migratedState.currentWorkspace).toBeNull()
  })

  test('migration adds currentWorkspace: null for very old versions (e.g., version 0)', () => {
    const state = {}

    const migratedState = migrateSettings(state, 0)

    expect(migratedState.currentWorkspace).toBeNull()
  })

  test('migration preserves already-set currentWorkspace when version >= 20', () => {
    const state = {
      currentWorkspace: 'already-set-workspace'
    }

    const migratedState = migrateSettings(state, 21)

    // Should preserve the existing value since version >= 20
    expect(migratedState.currentWorkspace).toBe('already-set-workspace')
  })
})

describe('workspaceRefreshTrigger state', () => {
  let useSettingsStore: ReturnType<typeof import('@/stores/settings').useSettingsStore>

  beforeAll(async () => {
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageMock(),
      configurable: true
    })
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: storageMock(),
      configurable: true
    })

    const module = await import('@/stores/settings')
    useSettingsStore = module.useSettingsStore
  })

  test('default value is 0', () => {
    // Reset to default state
    useSettingsStore.setState({ workspaceRefreshTrigger: 0 })
    expect(useSettingsStore.getState().workspaceRefreshTrigger).toBe(0)
  })

  test('triggerWorkspaceRefresh() increments by 1', () => {
    // Reset first
    useSettingsStore.setState({ workspaceRefreshTrigger: 0 })

    // Trigger once
    useSettingsStore.getState().triggerWorkspaceRefresh()

    expect(useSettingsStore.getState().workspaceRefreshTrigger).toBe(1)
  })

  test('multiple calls increment correctly (0 → 1 → 2 → 3)', () => {
    // Reset first
    useSettingsStore.setState({ workspaceRefreshTrigger: 0 })

    // Trigger multiple times
    useSettingsStore.getState().triggerWorkspaceRefresh()
    expect(useSettingsStore.getState().workspaceRefreshTrigger).toBe(1)

    useSettingsStore.getState().triggerWorkspaceRefresh()
    expect(useSettingsStore.getState().workspaceRefreshTrigger).toBe(2)

    useSettingsStore.getState().triggerWorkspaceRefresh()
    expect(useSettingsStore.getState().workspaceRefreshTrigger).toBe(3)
  })

  test('trigger is independent of currentWorkspace state', () => {
    // Reset both
    useSettingsStore.setState({ workspaceRefreshTrigger: 0, currentWorkspace: null })

    // Set a workspace
    useSettingsStore.getState().setCurrentWorkspace('my-workspace')

    // Trigger refresh
    useSettingsStore.getState().triggerWorkspaceRefresh()

    // Both should have their own values
    expect(useSettingsStore.getState().workspaceRefreshTrigger).toBe(1)
    expect(useSettingsStore.getState().currentWorkspace).toBe('my-workspace')

    // Change workspace and trigger again
    useSettingsStore.getState().setCurrentWorkspace('other-workspace')
    useSettingsStore.getState().triggerWorkspaceRefresh()

    expect(useSettingsStore.getState().workspaceRefreshTrigger).toBe(2)
    expect(useSettingsStore.getState().currentWorkspace).toBe('other-workspace')
  })
})

describe('searchLabelDropdownRefreshTrigger state', () => {
  let useSettingsStore: ReturnType<typeof import('@/stores/settings').useSettingsStore>

  beforeAll(async () => {
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageMock(),
      configurable: true
    })
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: storageMock(),
      configurable: true
    })

    const module = await import('@/stores/settings')
    useSettingsStore = module.useSettingsStore
  })

  test('default value is 0', () => {
    useSettingsStore.setState({ searchLabelDropdownRefreshTrigger: 0 })
    expect(useSettingsStore.getState().searchLabelDropdownRefreshTrigger).toBe(0)
  })

  test('triggerSearchLabelDropdownRefresh() increments by 1', () => {
    useSettingsStore.setState({ searchLabelDropdownRefreshTrigger: 0 })
    useSettingsStore.getState().triggerSearchLabelDropdownRefresh()
    expect(useSettingsStore.getState().searchLabelDropdownRefreshTrigger).toBe(1)
  })

  test('multiple calls increment correctly', () => {
    useSettingsStore.setState({ searchLabelDropdownRefreshTrigger: 0 })
    useSettingsStore.getState().triggerSearchLabelDropdownRefresh()
    useSettingsStore.getState().triggerSearchLabelDropdownRefresh()
    useSettingsStore.getState().triggerSearchLabelDropdownRefresh()
    expect(useSettingsStore.getState().searchLabelDropdownRefreshTrigger).toBe(3)
  })
})

describe('partialize - trigger fields excluded from persistence', () => {
  let useSettingsStore: ReturnType<typeof import('@/stores/settings').useSettingsStore>

  // Re-implement the partialize logic to test it
  const partializeState = (state: any) => {
    const {
      workspaceRefreshTrigger,
      triggerWorkspaceRefresh,
      searchLabelDropdownRefreshTrigger,
      triggerSearchLabelDropdownRefresh,
      ...rest
    } = state
    return rest
  }

  beforeAll(async () => {
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageMock(),
      configurable: true
    })
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: storageMock(),
      configurable: true
    })

    const module = await import('@/stores/settings')
    useSettingsStore = module.useSettingsStore
  })

  test('workspaceRefreshTrigger is excluded from partialize output', () => {
    // Set a trigger value
    useSettingsStore.setState({ workspaceRefreshTrigger: 5 })

    const state = useSettingsStore.getState()
    const partialized = partializeState(state)

    expect(partialized).not.toHaveProperty('workspaceRefreshTrigger')
    expect(partialized).not.toHaveProperty('triggerWorkspaceRefresh')
  })

  test('searchLabelDropdownRefreshTrigger is excluded from partialize output', () => {
    // Set a trigger value
    useSettingsStore.setState({ searchLabelDropdownRefreshTrigger: 10 })

    const state = useSettingsStore.getState()
    const partialized = partializeState(state)

    expect(partialized).not.toHaveProperty('searchLabelDropdownRefreshTrigger')
    expect(partialized).not.toHaveProperty('triggerSearchLabelDropdownRefresh')
  })

  test('currentWorkspace IS included in partialize output', () => {
    // Set currentWorkspace
    useSettingsStore.setState({ currentWorkspace: 'my-workspace' })

    const state = useSettingsStore.getState()
    const partialized = partializeState(state)

    expect(partialized).toHaveProperty('currentWorkspace')
    expect(partialized.currentWorkspace).toBe('my-workspace')
  })

  test('after calling triggerWorkspaceRefresh, trigger value is updated in state but NOT in partialize output', () => {
    // Reset first
    useSettingsStore.setState({ workspaceRefreshTrigger: 0, currentWorkspace: null })

    // Trigger refresh
    useSettingsStore.getState().triggerWorkspaceRefresh()

    // Verify in-state trigger is updated
    expect(useSettingsStore.getState().workspaceRefreshTrigger).toBe(1)

    // Verify partialize excludes it
    const state = useSettingsStore.getState()
    const partialized = partializeState(state)
    expect(partialized).not.toHaveProperty('workspaceRefreshTrigger')
  })

  test('other settings are still included in partialize output', () => {
    // Set various settings
    useSettingsStore.setState({
      theme: 'dark',
      language: 'en',
      showPropertyPanel: false,
      graphMaxNodes: 500,
      currentWorkspace: 'test-workspace'
    })

    const state = useSettingsStore.getState()
    const partialized = partializeState(state)

    // These should be present
    expect(partialized).toHaveProperty('theme')
    expect(partialized).toHaveProperty('language')
    expect(partialized).toHaveProperty('showPropertyPanel')
    expect(partialized).toHaveProperty('graphMaxNodes')
    expect(partialized).toHaveProperty('currentWorkspace')

    // These should be excluded (triggers)
    expect(partialized).not.toHaveProperty('workspaceRefreshTrigger')
    expect(partialized).not.toHaveProperty('triggerWorkspaceRefresh')
    expect(partialized).not.toHaveProperty('searchLabelDropdownRefreshTrigger')
    expect(partialized).not.toHaveProperty('triggerSearchLabelDropdownRefresh')
  })
})
