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

let useGraphStore: ReturnType<typeof import('@/stores/graph').useGraphStore>

beforeAll(async () => {
  // Try direct import first (graph store doesn't use persist middleware)
  try {
    const module = await import('@/stores/graph')
    useGraphStore = module.useGraphStore
  } catch {
    // Fallback: mock localStorage if needed
    Object.defineProperty(globalThis, 'localStorage', {
      value: storageMock(),
      configurable: true
    })
    Object.defineProperty(globalThis, 'sessionStorage', {
      value: storageMock(),
      configurable: true
    })
    const module = await import('@/stores/graph')
    useGraphStore = module.useGraphStore
  }
})

describe('graphDataVersion state', () => {
  test('default value is 0', () => {
    // Reset first
    useGraphStore.setState({ graphDataVersion: 0 })
    expect(useGraphStore.getState().graphDataVersion).toBe(0)
  })

  test('incrementGraphDataVersion() increments by 1', () => {
    // Reset first
    useGraphStore.setState({ graphDataVersion: 0 })

    useGraphStore.getState().incrementGraphDataVersion()

    expect(useGraphStore.getState().graphDataVersion).toBe(1)
  })

  test('multiple calls increment correctly (0 → 1 → 2 → 3)', () => {
    // Reset first
    useGraphStore.setState({ graphDataVersion: 0 })

    useGraphStore.getState().incrementGraphDataVersion()
    expect(useGraphStore.getState().graphDataVersion).toBe(1)

    useGraphStore.getState().incrementGraphDataVersion()
    expect(useGraphStore.getState().graphDataVersion).toBe(2)

    useGraphStore.getState().incrementGraphDataVersion()
    expect(useGraphStore.getState().graphDataVersion).toBe(3)
  })
})

describe('graph.reset() - completeness', () => {
  test('resets all graph state fields to defaults', () => {
    // Setup: populate state with various values
    useGraphStore.setState({
      selectedNode: 'node-1',
      focusedNode: 'node-2',
      selectedEdge: 'edge-1',
      focusedEdge: 'edge-2',
      rawGraph: { nodes: [], edges: [] } as any,
      sigmaGraph: {} as any,
      searchEngine: {} as any,
      moveToSelectedNode: true,
      graphIsEmpty: true,
      isFetching: true,
      graphDataFetchAttempted: true,
      labelsFetchAttempted: true
    })

    // Verify state is populated
    const populatedState = useGraphStore.getState()
    expect(populatedState.selectedNode).toBe('node-1')
    expect(populatedState.focusedNode).toBe('node-2')
    expect(populatedState.selectedEdge).toBe('edge-1')
    expect(populatedState.focusedEdge).toBe('edge-2')
    expect(populatedState.rawGraph).not.toBeNull()
    expect(populatedState.sigmaGraph).not.toBeNull()
    expect(populatedState.searchEngine).not.toBeNull()
    expect(populatedState.moveToSelectedNode).toBe(true)
    expect(populatedState.graphIsEmpty).toBe(true)
    expect(populatedState.isFetching).toBe(true)
    expect(populatedState.graphDataFetchAttempted).toBe(true)
    expect(populatedState.labelsFetchAttempted).toBe(true)

    // Call reset
    useGraphStore.getState().reset()

    // Verify all fields reset to defaults
    const resetState = useGraphStore.getState()
    expect(resetState.selectedNode).toBeNull()
    expect(resetState.focusedNode).toBeNull()
    expect(resetState.selectedEdge).toBeNull()
    expect(resetState.focusedEdge).toBeNull()
    expect(resetState.rawGraph).toBeNull()
    expect(resetState.sigmaGraph).toBeNull()
    expect(resetState.searchEngine).toBeNull()
    expect(resetState.moveToSelectedNode).toBe(false)
    expect(resetState.graphIsEmpty).toBe(false)
    expect(resetState.isFetching).toBe(false)
    expect(resetState.graphDataFetchAttempted).toBe(false)
    expect(resetState.labelsFetchAttempted).toBe(false)
  })
})

describe('fetch flags', () => {
  test('setGraphDataFetchAttempted(true) sets the flag', () => {
    useGraphStore.setState({ graphDataFetchAttempted: false })
    expect(useGraphStore.getState().graphDataFetchAttempted).toBe(false)

    useGraphStore.getState().setGraphDataFetchAttempted(true)

    expect(useGraphStore.getState().graphDataFetchAttempted).toBe(true)
  })

  test('setLabelsFetchAttempted(true) sets the flag', () => {
    useGraphStore.setState({ labelsFetchAttempted: false })
    expect(useGraphStore.getState().labelsFetchAttempted).toBe(false)

    useGraphStore.getState().setLabelsFetchAttempted(true)

    expect(useGraphStore.getState().labelsFetchAttempted).toBe(true)
  })

  test('after reset(), both fetch flags are false', () => {
    // Set flags to true
    useGraphStore.setState({
      graphDataFetchAttempted: true,
      labelsFetchAttempted: true
    })

    expect(useGraphStore.getState().graphDataFetchAttempted).toBe(true)
    expect(useGraphStore.getState().labelsFetchAttempted).toBe(true)

    // Reset
    useGraphStore.getState().reset()

    // Verify flags are false
    expect(useGraphStore.getState().graphDataFetchAttempted).toBe(false)
    expect(useGraphStore.getState().labelsFetchAttempted).toBe(false)
  })
})

describe('selectedNode/Edge', () => {
  test('setSelectedNode("node-1") works', () => {
    useGraphStore.setState({ selectedNode: null })
    expect(useGraphStore.getState().selectedNode).toBeNull()

    useGraphStore.getState().setSelectedNode('node-1')

    expect(useGraphStore.getState().selectedNode).toBe('node-1')
  })

  test('setSelectedNode with moveToSelectedNode option', () => {
    useGraphStore.setState({ selectedNode: null, moveToSelectedNode: false })

    useGraphStore.getState().setSelectedNode('node-1', true)

    expect(useGraphStore.getState().selectedNode).toBe('node-1')
    expect(useGraphStore.getState().moveToSelectedNode).toBe(true)
  })

  test('clearSelection() clears all selections', () => {
    // Set multiple selections
    useGraphStore.setState({
      selectedNode: 'node-1',
      focusedNode: 'node-2',
      selectedEdge: 'edge-1',
      focusedEdge: 'edge-2'
    })

    // Verify all are set
    expect(useGraphStore.getState().selectedNode).toBe('node-1')
    expect(useGraphStore.getState().focusedNode).toBe('node-2')
    expect(useGraphStore.getState().selectedEdge).toBe('edge-1')
    expect(useGraphStore.getState().focusedEdge).toBe('edge-2')

    // Clear
    useGraphStore.getState().clearSelection()

    // Verify all are null
    expect(useGraphStore.getState().selectedNode).toBeNull()
    expect(useGraphStore.getState().focusedNode).toBeNull()
    expect(useGraphStore.getState().selectedEdge).toBeNull()
    expect(useGraphStore.getState().focusedEdge).toBeNull()
  })

  test('after reset(), all selections are null', () => {
    // Set selections
    useGraphStore.setState({
      selectedNode: 'node-1',
      focusedNode: 'node-2',
      selectedEdge: 'edge-1',
      focusedEdge: 'edge-2'
    })

    // Reset
    useGraphStore.getState().reset()

    // Verify all are null
    expect(useGraphStore.getState().selectedNode).toBeNull()
    expect(useGraphStore.getState().focusedNode).toBeNull()
    expect(useGraphStore.getState().selectedEdge).toBeNull()
    expect(useGraphStore.getState().focusedEdge).toBeNull()
  })
})

describe('other graph state setters', () => {
  test('setRawGraph() updates rawGraph', () => {
    const mockGraph = { nodes: [{ id: 'n1' }], edges: [] } as any
    useGraphStore.getState().setRawGraph(mockGraph)
    expect(useGraphStore.getState().rawGraph).toBe(mockGraph)
  })

  test('setSigmaGraph() updates sigmaGraph', () => {
    const mockGraph = {} as any
    useGraphStore.getState().setSigmaGraph(mockGraph)
    expect(useGraphStore.getState().sigmaGraph).toBe(mockGraph)
  })

  test('setIsFetching() updates isFetching', () => {
    useGraphStore.setState({ isFetching: false })
    expect(useGraphStore.getState().isFetching).toBe(false)

    useGraphStore.getState().setIsFetching(true)
    expect(useGraphStore.getState().isFetching).toBe(true)
  })

  test('setGraphIsEmpty() updates graphIsEmpty', () => {
    useGraphStore.setState({ graphIsEmpty: false })
    expect(useGraphStore.getState().graphIsEmpty).toBe(false)

    useGraphStore.getState().setGraphIsEmpty(true)
    expect(useGraphStore.getState().graphIsEmpty).toBe(true)
  })

  test('setMoveToSelectedNode() updates moveToSelectedNode', () => {
    useGraphStore.setState({ moveToSelectedNode: false })
    expect(useGraphStore.getState().moveToSelectedNode).toBe(false)

    useGraphStore.getState().setMoveToSelectedNode(true)
    expect(useGraphStore.getState().moveToSelectedNode).toBe(true)
  })

  test('setSearchEngine() updates searchEngine', () => {
    const mockEngine = {} as any
    useGraphStore.getState().setSearchEngine(mockEngine)
    expect(useGraphStore.getState().searchEngine).toBe(mockEngine)
  })

  test('resetSearchEngine() sets searchEngine to null', () => {
    const mockEngine = {} as any
    useGraphStore.getState().setSearchEngine(mockEngine)
    expect(useGraphStore.getState().searchEngine).not.toBeNull()

    useGraphStore.getState().resetSearchEngine()
    expect(useGraphStore.getState().searchEngine).toBeNull()
  })
})
