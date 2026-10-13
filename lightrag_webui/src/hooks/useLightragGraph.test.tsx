import { afterEach, beforeEach, describe, expect, spyOn, test } from 'bun:test'
import { act, cleanup, screen } from '@testing-library/react'
import * as api from '@/api/lightrag'
import type { LightragGraphType } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import { renderWithProviders } from '@/test/render'
import useLightragGraph from './useLightragGraph'

type Request = {
  label: string
  depth: number
  maxNodes: number
  resolve: (data: LightragGraphType) => void
  reject: (error: Error) => void
}
let requests: Request[] = []
let queryGraphs: ReturnType<typeof spyOn<typeof api, 'queryGraphs'>>
const graphSnapshot = useGraphStore.getState()
const settingsSnapshot = useSettingsStore.getState()
const backendSnapshot = useBackendState.getState()
const graph = (id: string): LightragGraphType => ({
  nodes: [{ id, labels: [id], properties: { entity_type: id } }],
  edges: []
})
const emptyGraph: LightragGraphType = { nodes: [], edges: [] }

const View = () => {
  useLightragGraph()
  const label = useSettingsStore.use.queryLabel()
  const data = useGraphStore.use.rawGraph()
  const isFetching = useGraphStore.use.isFetching()
  return <output>{JSON.stringify({ label, nodes: data?.nodes.map((node) => node.id) ?? [], isFetching })}</output>
}
const view = () => JSON.parse(screen.getByRole('status').textContent!)
const flush = async (body: () => void) => {
  await act(async () => {
    body()
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}
const select = async (label: string) => flush(() => {
  useGraphStore.getState().setGraphDataFetchAttempted(false)
  useSettingsStore.getState().setQueryLabel(label)
  useGraphStore.getState().incrementGraphDataVersion()
})

beforeEach(() => {
  requests = []
  useGraphStore.setState({ ...graphSnapshot, typeColorMap: new Map() })
  useSettingsStore.setState({ ...settingsSnapshot, queryLabel: 'A', graphQueryMaxDepth: 3, graphMaxNodes: 100 })
  useBackendState.setState(backendSnapshot)
  queryGraphs = spyOn(api, 'queryGraphs').mockImplementation((label, depth, maxNodes) => new Promise((resolve, reject) => {
    requests.push({
      label,
      depth,
      maxNodes,
      resolve,
      reject
    })
  }))
})

afterEach(() => {
  cleanup()
  queryGraphs.mockRestore()
  useGraphStore.setState(graphSnapshot)
  useSettingsStore.setState(settingsSnapshot)
  useBackendState.setState(backendSnapshot)
})

describe('graph request ownership', () => {
  test('a current empty result clears the label and settles without repeating requests', async () => {
    renderWithProviders(<View />)
    await flush(() => requests[0].resolve(emptyGraph))
    for (let turn = 0; turn < 4; turn++) await flush(() => {})
    expect(view()).toEqual({ label: '', nodes: [], isFetching: false })
    expect(useGraphStore.getState().graphIsEmpty).toBe(true)
    expect(useGraphStore.getState().graphNodeCount).toBe(0)
    expect(useGraphStore.getState().graphEdgeCount).toBe(0)
    expect(requests.map((request) => request.label)).toEqual(['A'])
    await select('B')
    await flush(() => requests[1].resolve(graph('B')))
    expect(view()).toEqual({ label: 'B', nodes: ['B'], isFetching: false })
    expect(useGraphStore.getState().graphIsEmpty).toBe(false)
  })

  test('clearing the label during a fetch discards its response and settles without a new network request', async () => {
    renderWithProviders(<View />)
    await select('')
    await flush(() => requests[0].resolve(graph('A')))
    for (let turn = 0; turn < 4; turn++) await flush(() => {})
    expect(view()).toEqual({ label: '', nodes: [], isFetching: false })
    expect(useGraphStore.getState().graphIsEmpty).toBe(true)
    expect(useGraphStore.getState().graphNodeCount).toBe(0)
    expect(useGraphStore.getState().graphEdgeCount).toBe(0)
    expect(requests.map((request) => request.label)).toEqual(['A'])
  })

  for (const [name, result] of [['empty', emptyGraph], ['nonempty', graph('A')]] as const) {
    test(`ignores a stale ${name} response and loads the latest selection`, async () => {
      renderWithProviders(<View />)
      await select('B')
      await flush(() => requests[0].resolve(result))
      expect(view().label).toBe('B')
      expect(view().nodes).toEqual([])
      expect(requests.map((request) => request.label)).toEqual(['A', 'B'])
      expect(view().isFetching).toBe(true)
      await flush(() => requests[1].resolve(graph('B')))
      expect(view()).toEqual({ label: 'B', nodes: ['B'], isFetching: false })
    })
  }

  test('ignores a stale error without overwriting the current backend message', async () => {
    renderWithProviders(<View />)
    await select('B')
    await flush(() => requests[0].reject(new Error('stale failure')))
    expect(useBackendState.getState().message).toBe(backendSnapshot.message)
    expect(requests.map((request) => request.label)).toEqual(['A', 'B'])
    await flush(() => requests[1].resolve(graph('B')))
    expect(view().nodes).toEqual(['B'])
  })

  test('uses the latest depth, node limit and refresh version after the active request settles', async () => {
    renderWithProviders(<View />)
    await flush(() => {
      useSettingsStore.getState().setGraphQueryMaxDepth(5)
      useSettingsStore.getState().setGraphMaxNodes(200)
      useGraphStore.getState().incrementGraphDataVersion()
    })
    await flush(() => requests[0].resolve(graph('old')))
    expect(requests.map(({ label, depth, maxNodes }) => ({ label, depth, maxNodes }))).toEqual([
      { label: 'A', depth: 3, maxNodes: 100 },
      { label: 'A', depth: 5, maxNodes: 200 }
    ])
    await flush(() => requests[1].resolve(graph('fresh')))
    expect(view().nodes).toEqual(['fresh'])
  })

  test('does not apply a response after unmount or release a newer mount request', async () => {
    const first = renderWithProviders(<View />)
    await flush(() => {})
    first.unmount()
    useSettingsStore.getState().setQueryLabel('B')
    useGraphStore.getState().setGraphDataFetchAttempted(false)
    renderWithProviders(<View />)
    await flush(() => {})
    expect(requests.map((request) => request.label)).toEqual(['A', 'B'])
    await flush(() => requests[0].resolve(emptyGraph))
    expect(view().label).toBe('B')
    expect(view().isFetching).toBe(true)
    await flush(() => requests[1].resolve(graph('B')))
    expect(view().nodes).toEqual(['B'])
  })

  test('discards a build superseded while yielding, including its legend colors', async () => {
    let finishYield!: () => void
    const originalScheduler = Object.getOwnPropertyDescriptor(globalThis, 'scheduler')
    Object.defineProperty(globalThis, 'scheduler', {
      configurable: true,
      value: { yield: () => new Promise<void>((resolve) => { finishYield = resolve }) }
    })
    let clock = 0
    const now = spyOn(performance, 'now').mockImplementation(() => clock += 20)
    try {
      renderWithProviders(<View />)
      await flush(() => {})
      await flush(() => requests[0].resolve(graph('A')))
      expect(typeof finishYield).toBe('function')
      await select('B')
      now.mockRestore()
      await flush(() => finishYield())
      expect(view().nodes).toEqual([])
      expect(useGraphStore.getState().typeColorMap.size).toBe(0)
      expect(requests.map((request) => request.label)).toEqual(['A', 'B'])
      await flush(() => requests[1].resolve(graph('B')))
      expect(view().nodes).toEqual(['B'])
    } finally {
      now.mockRestore()
      if (originalScheduler) Object.defineProperty(globalThis, 'scheduler', originalScheduler)
      else Reflect.deleteProperty(globalThis, 'scheduler')
    }
  })

  test('keeps bounded backoff and permits recovery through refresh', async () => {
    const retries: { callback: () => void; delay: number }[] = []
    const realTimeout = globalThis.setTimeout
    const timeout = spyOn(globalThis, 'setTimeout').mockImplementation(((callback: () => void, delay: number) => {
      if (delay >= 1000) {
        retries.push({ callback, delay })
        return 0
      }
      return realTimeout(callback, delay)
    }) as typeof setTimeout)
    try {
      renderWithProviders(<View />)
      await flush(() => {})
      for (let attempt = 0; attempt < 4; attempt++) {
        await flush(() => requests[attempt].reject(new Error('temporary graph failure')))
        if (attempt < 3) await flush(() => retries[attempt].callback())
      }
      expect(retries.map(({ delay }) => delay)).toEqual([1000, 2000, 4000])
      expect(requests).toHaveLength(4)
      expect(view().isFetching).toBe(false)
      await flush(() => useGraphStore.getState().incrementGraphDataVersion())
      expect(requests).toHaveLength(5)
      await flush(() => requests[4].resolve(graph('A')))
      expect(view().nodes).toEqual(['A'])
    } finally {
      timeout.mockRestore()
    }
  })

  test('a superseded backoff callback cannot reopen the newer request gate', async () => {
    let retry!: () => void
    const realTimeout = globalThis.setTimeout
    const timeout = spyOn(globalThis, 'setTimeout').mockImplementation(((callback: () => void, delay: number) => {
      if (delay >= 1000) {
        retry = callback
        return 0
      }
      return realTimeout(callback, delay)
    }) as typeof setTimeout)
    try {
      renderWithProviders(<View />)
      await flush(() => {})
      await flush(() => requests[0].reject(new Error('temporary graph failure')))
      expect(typeof retry).toBe('function')
      await select('B')
      await flush(() => retry())
      expect(useGraphStore.getState().graphDataFetchAttempted).toBe(true)
      expect(requests.map((request) => request.label)).toEqual(['A', 'B'])
      await flush(() => requests[1].resolve(graph('B')))
      expect(view().nodes).toEqual(['B'])
      expect(requests).toHaveLength(2)
    } finally {
      timeout.mockRestore()
    }
  })
})
