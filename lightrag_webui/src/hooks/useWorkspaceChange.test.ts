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

// ============================================
// Pure Logic Functions (extracted from hook)
// ============================================

/**
 * Determines if a workspace refresh should be triggered.
 * Matches the condition in useWorkspaceChange: previous !== current
 */
const shouldTriggerRefresh = (
  previous: string | null,
  current: string | null
): boolean => {
  return previous !== current
}

/**
 * Executes the actions that should happen on workspace change.
 * These are the side effects extracted from useWorkspaceChange.
 */
const executeWorkspaceChangeActions = (
  settingsStore: { setRetrievalHistory: (h: unknown[]) => void },
  graphStore: { reset: () => void }
) => {
  graphStore.reset()
  settingsStore.setRetrievalHistory([])
}

// ============================================
// Tests: shouldTriggerRefresh (Pure Logic)
// ============================================

describe('shouldTriggerRefresh (pure logic)', () => {
  test('Same workspace (null → null): returns false', () => {
    expect(shouldTriggerRefresh(null, null)).toBe(false)
  })

  test('Same workspace ("ws" → "ws"): returns false', () => {
    expect(shouldTriggerRefresh('ws', 'ws')).toBe(false)
  })

  test('Changed workspace (null → "ws"): returns true', () => {
    expect(shouldTriggerRefresh(null, 'ws')).toBe(true)
  })

  test('Changed workspace ("ws" → null): returns true', () => {
    expect(shouldTriggerRefresh('ws', null)).toBe(true)
  })

  test('Changed workspace ("ws-a" → "ws-b"): returns true', () => {
    expect(shouldTriggerRefresh('ws-a', 'ws-b')).toBe(true)
  })

  test('Changed workspace ("ws-a" → "ws-a"): returns false', () => {
    expect(shouldTriggerRefresh('ws-a', 'ws-a')).toBe(false)
  })
})

// ============================================
// Integration Tests with Real Stores
// ============================================

describe('Workspace change integration with real stores', () => {
  let useSettingsStore: ReturnType<typeof import('@/stores/settings').useSettingsStore>
  let useGraphStore: ReturnType<typeof import('@/stores/graph').useGraphStore>

  beforeAll(async () => {
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageMock(),
      configurable: true
    })
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: storageMock(),
      configurable: true
    })

    const [settingsModule, graphModule] = await Promise.all([
      import('@/stores/settings'),
      import('@/stores/graph')
    ])

    useSettingsStore = settingsModule.useSettingsStore
    useGraphStore = graphModule.useGraphStore
  })

  describe('executeWorkspaceChangeActions', () => {
    test('Calls graphStore.reset() — verify graph state is reset', () => {
      // Set up graph state with some data
      useGraphStore.setState({
        selectedNode: 'test-node',
        focusedNode: 'test-focused',
        selectedEdge: 'test-edge',
        isFetching: true,
        graphDataFetchAttempted: true,
        labelsFetchAttempted: true
      })

      // Verify state is set
      expect(useGraphStore.getState().selectedNode).toBe('test-node')
      expect(useGraphStore.getState().isFetching).toBe(true)
      expect(useGraphStore.getState().graphDataFetchAttempted).toBe(true)

      // Execute the workspace change action
      executeWorkspaceChangeActions(
        useSettingsStore.getState(),
        useGraphStore.getState()
      )

      // Verify graph state is reset
      const graphState = useGraphStore.getState()
      expect(graphState.selectedNode).toBeNull()
      expect(graphState.focusedNode).toBeNull()
      expect(graphState.selectedEdge).toBeNull()
      expect(graphState.isFetching).toBe(false)
      expect(graphState.graphDataFetchAttempted).toBe(false)
      expect(graphState.labelsFetchAttempted).toBe(false)
    })

    test('Calls settingsStore.setRetrievalHistory([]) — verify retrieval history is empty', () => {
      // Set up retrieval history with messages
      const testMessages = [
        { role: 'user' as const, content: 'Hello' },
        { role: 'assistant' as const, content: 'Hi there' }
      ]
      useSettingsStore.setState({ retrievalHistory: testMessages })

      // Verify state is set
      expect(useSettingsStore.getState().retrievalHistory).toEqual(testMessages)

      // Execute the workspace change action
      executeWorkspaceChangeActions(
        useSettingsStore.getState(),
        useGraphStore.getState()
      )

      // Verify retrieval history is empty
      expect(useSettingsStore.getState().retrievalHistory).toEqual([])
    })

    test('Both actions called together on workspace change', () => {
      // Set up both stores with data
      useGraphStore.setState({
        selectedNode: 'some-node',
        isFetching: true
      })
      const testMessages = [
        { role: 'user' as const, content: 'Test message' }
      ]
      useSettingsStore.setState({ retrievalHistory: testMessages })

      // Execute workspace change actions
      executeWorkspaceChangeActions(
        useSettingsStore.getState(),
        useGraphStore.getState()
      )

      // Verify both were cleared
      const graphState = useGraphStore.getState()
      expect(graphState.selectedNode).toBeNull()
      expect(graphState.isFetching).toBe(false)

      expect(useSettingsStore.getState().retrievalHistory).toEqual([])
    })
  })

  describe('Integration: full workspace change scenario', () => {
    test('Populate graph state, execute workspace change, verify reset', () => {
      // Reset to clean state first
      useGraphStore.setState({
        selectedNode: null,
        focusedNode: null,
        selectedEdge: null,
        isFetching: false,
        graphDataFetchAttempted: false,
        labelsFetchAttempted: false
      })
      useSettingsStore.setState({ retrievalHistory: [] })

      // Populate graph state
      useGraphStore.setState({
        selectedNode: 'entity-123',
        focusedNode: 'entity-456',
        selectedEdge: 'edge-789',
        isFetching: true,
        graphDataFetchAttempted: true,
        labelsFetchAttempted: true
      })

      // Set retrieval history with messages
      const messages = [
        { role: 'user' as const, content: 'Query 1' },
        { role: 'assistant' as const, content: 'Response 1' },
        { role: 'user' as const, content: 'Query 2' }
      ]
      useSettingsStore.setState({ retrievalHistory: messages })

      // Execute workspace change (actions from hook)
      executeWorkspaceChangeActions(
        useSettingsStore.getState(),
        useGraphStore.getState()
      )

      // Note: triggerWorkspaceRefresh() is called separately in the hook
      // We verify the clear actions here

      // Verify graph state is reset
      const graphState = useGraphStore.getState()
      expect(graphState.selectedNode).toBeNull()
      expect(graphState.focusedNode).toBeNull()
      expect(graphState.selectedEdge).toBeNull()
      expect(graphState.isFetching).toBe(false)
      expect(graphState.graphDataFetchAttempted).toBe(false)
      expect(graphState.labelsFetchAttempted).toBe(false)

      // Verify retrieval history is empty
      expect(useSettingsStore.getState().retrievalHistory).toEqual([])
    })
  })

  describe('Rapid switching simulation', () => {
    test('Simulate: null → ws-a → ws-b → null in sequence', () => {
      // Reset to clean state
      useGraphStore.setState({
        selectedNode: null,
        focusedNode: null,
        selectedEdge: null,
        isFetching: false,
        graphDataFetchAttempted: false,
        labelsFetchAttempted: false
      })
      useSettingsStore.setState({ retrievalHistory: [] })

      const workspaceSequence: (string | null)[] = [null, 'ws-a', 'ws-b', null]

      for (let i = 1; i < workspaceSequence.length; i++) {
        const previous = workspaceSequence[i - 1]
        const current = workspaceSequence[i]

        // Only trigger if workspace changed
        if (shouldTriggerRefresh(previous, current)) {
          // Set up data before change
          useGraphStore.setState({
            selectedNode: `node-${previous ?? 'null'}`,
            isFetching: true
          })
          useSettingsStore.setState({
            retrievalHistory: [{ role: 'user' as const, content: `msg-${previous ?? 'null'}` }]
          })

          // Execute workspace change actions
          executeWorkspaceChangeActions(
            useSettingsStore.getState(),
            useGraphStore.getState()
          )

          // Verify actions were triggered
          expect(useGraphStore.getState().selectedNode).toBeNull()
          expect(useGraphStore.getState().isFetching).toBe(false)
          expect(useSettingsStore.getState().retrievalHistory).toEqual([])
        }
      }

      // Final state should be consistent
      const finalState = useGraphStore.getState()
      expect(finalState.selectedNode).toBeNull()
      expect(finalState.focusedNode).toBeNull()
      expect(finalState.selectedEdge).toBeNull()
      expect(finalState.isFetching).toBe(false)
      expect(finalState.graphDataFetchAttempted).toBe(false)
      expect(finalState.labelsFetchAttempted).toBe(false)

      expect(useSettingsStore.getState().retrievalHistory).toEqual([])
    })

    test('After each change, verify actions were triggered for distinct workspaces', () => {
      // Reset to clean state
      useGraphStore.setState({
        selectedNode: null,
        isFetching: false,
        graphDataFetchAttempted: false,
        labelsFetchAttempted: false
      })
      useSettingsStore.setState({ retrievalHistory: [] })

      // Test each distinct workspace change
      const changes = [
        { from: null, to: 'workspace-1' },
        { from: 'workspace-1', to: 'workspace-2' },
        { from: 'workspace-2', to: 'workspace-3' },
        { from: 'workspace-3', to: null }
      ]

      changes.forEach(({ from, to }) => {
        // Verify shouldTriggerRefresh returns true
        expect(shouldTriggerRefresh(from, to)).toBe(true)

        // Set up data
        useGraphStore.setState({
          selectedNode: `node-${from ?? 'null'}`,
          isFetching: true,
          graphDataFetchAttempted: true,
          labelsFetchAttempted: true
        })
        useSettingsStore.setState({
          retrievalHistory: [
            { role: 'user' as const, content: `message for ${from ?? 'null'}` }
          ]
        })

        // Execute workspace change
        executeWorkspaceChangeActions(
          useSettingsStore.getState(),
          useGraphStore.getState()
        )

        // Verify actions were triggered
        expect(useGraphStore.getState().selectedNode).toBeNull()
        expect(useGraphStore.getState().isFetching).toBe(false)
        expect(useGraphStore.getState().graphDataFetchAttempted).toBe(false)
        expect(useSettingsStore.getState().retrievalHistory).toEqual([])
      })
    })
  })
})
