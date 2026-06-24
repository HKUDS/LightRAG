import Graph, { UndirectedGraph } from 'graphology'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '@/lib/utils'
import * as Constants from '@/lib/constants'
import { useGraphStore, RawGraph, RawNodeType, RawEdgeType } from '@/stores/graph'
import { toast } from 'sonner'
import { queryGraphs } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'

import { resolveNodeColor, DEFAULT_NODE_COLOR } from '@/utils/graphColor'

// Every node gets this border (the node-border program reads `borderColor`).
const NODE_BORDER_COLOR = '#FFFFFF'

// Bounded auto-retry for transient graph fetch failures. We must NOT retry
// without a cap (that was the original unbounded zero-backoff loop that
// hammered the backend), but we also must not wedge the query permanently on
// a single transient blip. After MAX_FETCH_RETRIES the query is suppressed
// until the user changes a parameter or hits refresh (graphDataVersion bump).
const MAX_FETCH_RETRIES = 3
const RETRY_BASE_DELAY_MS = 1000 // 1s -> 2s -> 4s exponential backoff

// Marks a TERMINAL failure that happened AFTER the network fetch succeeded:
// building the sigma graph from the payload, or a store/sigma subscriber
// throwing while applying it. These are deterministic — retrying re-fetches
// identical data and fails identically — so they must NOT enter the bounded
// backoff retry path (which exists only for transient network failures).
class GraphBuildError extends Error {
  constructor(cause: unknown) {
    super('Graph build/apply failed')
    this.name = 'GraphBuildError'
    this.cause = cause
  }
}

// --- Performance helpers ----------------------------------------------------

// Deterministic, cheap replacement for `seedrandom(node.id)`.
// seedrandom's ARC4 setup does key mixing + a 256-iteration state init PER
// INSTANCE, which is pure overhead when we only need two stable floats per
// node. This hash gives the same property (same id -> same position) at a
// tiny fraction of the cost.
const hashNodeIdToPosition = (id: string): { x: number; y: number } => {
  let h1 = 0xdeadbeef ^ id.length
  let h2 = 0x41c6ce57 ^ id.length
  for (let i = 0; i < id.length; i++) {
    const ch = id.charCodeAt(i)
    h1 = Math.imul(h1 ^ ch, 2654435761)
    h2 = Math.imul(h2 ^ ch, 1597334677)
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  return { x: (h1 >>> 0) / 4294967296, y: (h2 >>> 0) / 4294967296 }
}

// Yield to the browser WITHOUT setTimeout's nested-timer clamp.
// `await setTimeout(0)` in a loop is clamped to >= 4ms per call by the HTML
// spec once timers nest more than 5 deep, so batched yielding via setTimeout
// adds (items / batchSize) * 4ms of pure idle time on top of the real work.
// scheduler.yield() (where available) or a MessageChannel message resolves in
// microseconds instead.
const yieldToBrowser = (): Promise<void> => {
  const scheduler = (globalThis as { scheduler?: { yield?: () => Promise<void> } }).scheduler
  if (scheduler?.yield) return scheduler.yield()
  return new Promise<void>((resolve) => {
    const { port1, port2 } = new MessageChannel()
    port1.onmessage = () => {
      port1.close()
      resolve()
    }
    port2.postMessage(null)
  })
}

// Cooperative time-slicing: run synchronously until the frame budget is spent,
// then yield ONCE. Keeps the page responsive while wasting almost no time.
const FRAME_BUDGET_MS = 12
const CHECK_EVERY = 256 // power of two; performance.now() is sampled sparsely

// Per-type node colors, performance-safe. The original implementation
// resolved the palette (and potentially wrote to the zustand store) ONCE PER
// NODE; this resolver calls resolveNodeColor once per DISTINCT entity type
// (graphs have dozens of types, not thousands of them) and commits the
// type->color map to the store in a single write, which also keeps the
// Legend functional.
const createTypeColorResolver = () => {
  let typeColorMap = useGraphStore.getState().typeColorMap
  const cache = new Map<string, string>()
  let mapUpdated = false

  return {
    colorFor(entityType: string | undefined): string {
      const key = entityType ?? ''
      let color = cache.get(key)
      if (color === undefined) {
        const resolved = resolveNodeColor(entityType, typeColorMap)
        if (resolved.updated) {
          typeColorMap = resolved.map
          mapUpdated = true
        }
        color = resolved.color || DEFAULT_NODE_COLOR
        cache.set(key, color)
      }
      return color
    },
    commit() {
      if (mapUpdated) {
        useGraphStore.setState({ typeColorMap })
        mapUpdated = false
      }
    }
  }
}

// Parse an edge's `weight` property into a finite number, preserving a
// legitimate 0. `Number(x) || 1` would coerce a real weight of 0 (the thinnest
// edge) to 1; use a finite check so 0 survives and only undefined/NaN fall back.
const parseEdgeWeight = (properties: Record<string, unknown> | undefined): number => {
  const w = Number(properties?.weight)
  return Number.isFinite(w) ? w : 1
}

// Build a node label defensively: a malformed payload node missing its
// `labels` array must not throw (labels.join) and abort the whole graph build
// — fall back to the node id so the rest of the graph still renders.
const safeNodeLabel = (labels: unknown, fallbackId: string): string =>
  Array.isArray(labels) ? labels.join(', ') : fallbackId

// Add an undirected edge, working around a graphology 0.26.0 bug: the non-multi
// duplicate check does a bare `adjacency[target]` object lookup, so a TARGET
// node named like an Object.prototype property ('constructor', 'toString',
// '__proto__', 'hasOwnProperty', ...) makes addEdge throw "already exists" even
// though hasEdge correctly returns false. The broken check only runs on the
// source side, and on an undirected graph the flipped edge is the same edge —
// so adding it reversed recovers it. Returns the dynamic edge id, or null if
// both orientations fail (e.g. both endpoints are prototype-named). Callers are
// expected to have pre-checked hasEdge, so a real duplicate is never masked.
const addUndirectedEdgeSafe = (
  graph: UndirectedGraph,
  source: string,
  target: string,
  attributes: Record<string, unknown>
): string | null => {
  try {
    return graph.addEdge(source, target, attributes)
  } catch {
    try {
      return graph.addEdge(target, source, attributes)
    } catch {
      return null
    }
  }
}

export type NodeType = {
  x: number
  y: number
  label: string
  size: number
  color: string
  highlighted?: boolean
}
export type EdgeType = {
  label: string
  originalWeight?: number
  size?: number
  color?: string
  hidden?: boolean
}

const fetchGraph = async (label: string, maxDepth: number, maxNodes: number) => {
  let rawData: any

  // Trigger GraphLabels component to check if the label is valid
  // console.log('Setting labelsFetchAttempted to true');
  useGraphStore.getState().setLabelsFetchAttempted(true)

  // If label is empty, use default label '*'
  const queryLabel = label || '*'

  try {
    console.log(`Fetching graph label: ${queryLabel}, depth: ${maxDepth}, nodes: ${maxNodes}`)
    rawData = await queryGraphs(queryLabel, maxDepth, maxNodes)
  } catch (e) {
    useBackendState.getState().setErrorMessage(errorMessage(e), 'Query Graphs Error!')
    return null
  }

  let rawGraph = null

  if (rawData) {
    // Null-prototype objects: node ids come from extracted entity names, so
    // they can collide with Object.prototype properties. On a plain object,
    // `map['__proto__'] = i` silently does NOT create an own property, and
    // `map['constructor']` for a missing key returns an inherited function
    // instead of undefined.
    const nodeIdMap: Record<string, number> = Object.create(null)
    const edgeIdMap: Record<string, number> = Object.create(null)

    for (let i = 0; i < rawData.nodes.length; i++) {
      const node = rawData.nodes[i]
      nodeIdMap[node.id] = i

      node.degree = 0
      node.size = 10
    }

    for (let i = 0; i < rawData.edges.length; i++) {
      const edge = rawData.edges[i]
      edgeIdMap[edge.id] = i

      const source = nodeIdMap[edge.source]
      const target = nodeIdMap[edge.target]
      if (source !== undefined && target !== undefined) {
        const sourceNode = rawData.nodes[source]
        if (!sourceNode) {
          console.error(`Source node ${edge.source} is undefined`)
          continue
        }

        const targetNode = rawData.nodes[target]
        if (!targetNode) {
          console.error(`Target node ${edge.target} is undefined`)
          continue
        }
        sourceNode.degree += 1
        targetNode.degree += 1
      }
    }

    // generate node size
    let minDegree = Number.MAX_SAFE_INTEGER
    let maxDegree = 0

    for (const node of rawData.nodes) {
      minDegree = Math.min(minDegree, node.degree)
      maxDegree = Math.max(maxDegree, node.degree)
    }
    const range = maxDegree - minDegree
    if (range > 0) {
      const scale = Constants.maxNodeSize - Constants.minNodeSize
      for (const node of rawData.nodes) {
        node.size = Math.round(
          Constants.minNodeSize + scale * Math.pow((node.degree - minDegree) / range, 0.5)
        )
      }
    }

    rawGraph = new RawGraph()
    rawGraph.nodes = rawData.nodes
    rawGraph.edges = rawData.edges
    rawGraph.nodeIdMap = nodeIdMap
    rawGraph.edgeIdMap = edgeIdMap

    console.log('Graph data loaded')
  }

  // console.debug({ data: JSON.parse(JSON.stringify(rawData)) })
  return { rawGraph, is_truncated: rawData?.is_truncated }
}

// Create a new graph instance with the raw graph data
const createSigmaGraph = async (rawGraph: RawGraph | null): Promise<UndirectedGraph | null> => {
  if (!rawGraph || !rawGraph.nodes.length) return null

  const graph = new UndirectedGraph()
  const typeColors = createTypeColorResolver()
  let sliceStart = performance.now()

  const nodes = rawGraph.nodes
  for (let i = 0; i < nodes.length; i++) {
    // Yield only when ~12ms of real work has accumulated, not every N items.
    if ((i & (CHECK_EVERY - 1)) === 0 && performance.now() - sliceStart > FRAME_BUDGET_MS) {
      await yieldToBrowser()
      sliceStart = performance.now()
    }

    const rawNode = nodes[i]
    // Defensive: a duplicate id in the payload would make addNode throw
    if (graph.hasNode(rawNode.id)) continue

    const { x, y } = hashNodeIdToPosition(rawNode.id)
    rawNode.color = typeColors.colorFor(rawNode.properties?.entity_type as string | undefined)

    graph.addNode(rawNode.id, {
      label: safeNodeLabel(rawNode.labels, rawNode.id),
      color: rawNode.color,
      x,
      y,
      size: rawNode.size,
      borderColor: NODE_BORDER_COLOR
    })
  }

  // Single store write for the whole build (keeps the Legend in sync)
  typeColors.commit()

  rawGraph.edgeDynamicIdMap = Object.create(null) as Record<string, number>
  let skippedEdges = 0

  const edges = rawGraph.edges
  for (let i = 0; i < edges.length; i++) {
    if ((i & (CHECK_EVERY - 1)) === 0 && performance.now() - sliceStart > FRAME_BUDGET_MS) {
      await yieldToBrowser()
      sliceStart = performance.now()
    }

    const rawEdge = edges[i]
    // Truncated BFS responses contain many dangling/duplicate edges. Three
    // cheap hash lookups beat a thrown-and-caught exception per bad edge
    // (exception throw/stack capture is orders of magnitude more expensive).
    if (
      !graph.hasNode(rawEdge.source) ||
      !graph.hasNode(rawEdge.target) ||
      graph.hasEdge(rawEdge.source, rawEdge.target)
    ) {
      continue
    }

    // `rawEdge.type` is the storage-level relationship type ("DIRECTED" for
    // every edge); the human-readable relation name is properties.keywords.
    // originalWeight feeds GraphControl's thickness scaling.
    const attributes = {
      label: (rawEdge.properties?.keywords as string | undefined) || undefined,
      originalWeight: parseEdgeWeight(rawEdge.properties)
    }
    const dynamicId = addUndirectedEdgeSafe(graph, rawEdge.source, rawEdge.target, attributes)
    if (dynamicId === null) {
      skippedEdges++
      continue
    }
    rawEdge.dynamicId = dynamicId
    rawGraph.edgeDynamicIdMap[rawEdge.dynamicId] = i
  }

  if (skippedEdges > 0) {
    console.warn(`[useLightragGraph] ${skippedEdges} edges could not be added to the graph`)
  }

  return graph
}

const useLightrangeGraph = () => {
  const { t } = useTranslation()
  const queryLabel = useSettingsStore.use.queryLabel()
  const rawGraph = useGraphStore.use.rawGraph()
  const sigmaGraph = useGraphStore.use.sigmaGraph()
  const maxQueryDepth = useSettingsStore.use.graphQueryMaxDepth()
  const maxNodes = useSettingsStore.use.graphMaxNodes()
  const isFetching = useGraphStore.use.isFetching()
  const nodeToExpand = useGraphStore.use.nodeToExpand()
  const nodeToPrune = useGraphStore.use.nodeToPrune()
  const graphDataVersion = useGraphStore.use.graphDataVersion()

  // Use ref to track if data has been loaded and initial load
  const dataLoadedRef = useRef(false)
  const initialLoadRef = useRef(false)
  // Use ref to track if empty data has been handled
  const emptyDataHandledRef = useRef(false)

  const getNode = useCallback(
    (nodeId: string) => {
      return rawGraph?.getNode(nodeId) || null
    },
    [rawGraph]
  )

  const getEdge = useCallback(
    (edgeId: string, dynamicId: boolean = true) => {
      return rawGraph?.getEdge(edgeId, dynamicId) || null
    },
    [rawGraph]
  )

  // Track if a fetch is in progress to prevent multiple simultaneous fetches
  const fetchInProgressRef = useRef(false)

  // Signature (label|depth|maxNodes|version) of the last fetch that COMPLETED
  // SUCCESSFULLY. Guards the completed-fetch loop: if some other code path
  // resets graphDataFetchAttempted after a successful fetch, an identical
  // signature here suppresses the redundant re-issue. Failures do NOT stamp
  // this (they are handled by the bounded-retry state below) until retries
  // are exhausted, at which point the failed signature is stamped to suppress.
  const lastFetchSignatureRef = useRef<string | null>(null)

  // Bounded-retry bookkeeping for transient fetch failures. `attempts` is
  // keyed by signature so a parameter change or manual refresh (both change
  // the signature) starts a fresh retry budget. The nonce drives the backoff
  // re-fire without changing the signature, so it does not reset the counter.
  const retryStateRef = useRef<{ signature: string; attempts: number }>({
    signature: '',
    attempts: 0
  })
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [retryNonce, setRetryNonce] = useState(0)

  // Reset graph when query label is cleared
  useEffect(() => {
    if (!queryLabel && (rawGraph !== null || sigmaGraph !== null)) {
      const state = useGraphStore.getState()
      state.reset()
      state.setGraphDataFetchAttempted(false)
      state.setLabelsFetchAttempted(false)
      dataLoadedRef.current = false
      initialLoadRef.current = false
    }
  }, [queryLabel, rawGraph, sigmaGraph])

  // Graph data fetching logic
  useEffect(() => {
    // Skip if fetch is already in progress
    if (fetchInProgressRef.current) {
      return
    }

    // Empty queryLabel should be only handle once(avoid infinite loop)
    if (!queryLabel && emptyDataHandledRef.current) {
      return
    }

    // Only fetch data when graphDataFetchAttempted is false (avoids re-fetching on vite dev mode)
    // GraphDataFetchAttempted must set to false when queryLabel is changed
    if (!isFetching && !useGraphStore.getState().graphDataFetchAttempted) {
      // Hard re-entrancy guard. The only things allowed to trigger a refetch
      // are an actual change of the query (label / depth / max nodes) or an
      // explicit refresh (graphDataVersion bump). If we arrive here again
      // with an identical signature, some code path reset
      // graphDataFetchAttempted after a completed fetch — refetching would
      // start an infinite request loop against the backend.
      const fetchSignature = `${queryLabel}|${maxQueryDepth}|${maxNodes}|v${graphDataVersion}`
      if (lastFetchSignatureRef.current === fetchSignature) {
        console.warn(
          '[useLightragGraph] Suppressed duplicate graph fetch:',
          fetchSignature,
          '— graphDataFetchAttempted was reset after this query already ran'
        )
        useGraphStore.getState().setGraphDataFetchAttempted(true)
        return
      }

      // A fresh fetch is starting for this signature: cancel any pending
      // backoff retry timer (e.g. a parameter change arrived mid-backoff).
      if (retryTimerRef.current !== null) {
        clearTimeout(retryTimerRef.current)
        retryTimerRef.current = null
      }

      // Set flags
      fetchInProgressRef.current = true
      useGraphStore.getState().setGraphDataFetchAttempted(true)

      const state = useGraphStore.getState()
      state.setIsFetching(true)

      // Clear selection and highlighted nodes before fetching new graph
      state.clearSelection()
      if (state.sigmaGraph) {
        // Only touch nodes that are actually highlighted: setNodeAttribute
        // emits a graph event per call, and blanket-writing it on every node
        // of a large live graph causes a visible stall before each fetch.
        state.sigmaGraph.forEachNode((node, attributes) => {
          if (attributes.highlighted) {
            state.sigmaGraph?.setNodeAttribute(node, 'highlighted', false)
          }
        })
      }

      console.log('Preparing graph data...')

      // Use a local copy of the parameters
      const currentQueryLabel = queryLabel
      const currentMaxQueryDepth = maxQueryDepth
      const currentMaxNodes = maxNodes

      // Declare a variable to store data promise
      let dataPromise: Promise<{
        rawGraph: RawGraph | null
        is_truncated: boolean | undefined
      } | null>

      // 1. If query label is not empty, use fetchGraph
      if (currentQueryLabel) {
        dataPromise = fetchGraph(currentQueryLabel, currentMaxQueryDepth, currentMaxNodes)
      } else {
        // 2. If query label is empty, set data to null
        console.log('Query label is empty, show empty graph')
        dataPromise = Promise.resolve({ rawGraph: null, is_truncated: false })
      }

      // 3. Process data
      dataPromise
        .then(async (result) => {
          const state = useGraphStore.getState()
          const data = result?.rawGraph

          if (result?.is_truncated) {
            toast.info(t('graphPanel.dataIsTruncated', 'Graph data is truncated to Max Nodes'))
          }

          // Reset state
          state.reset()

          // Check if data is empty or invalid
          if (!data || !data.nodes || data.nodes.length === 0) {
            // Create a graph with a single "Graph Is Empty" node
            const emptyGraph = new UndirectedGraph()

            // Add a single node with "Graph Is Empty" label
            emptyGraph.addNode('empty-graph-node', {
              label: t('graphPanel.emptyGraph'),
              color: '#5D6D7E', // gray color
              x: 0.5,
              y: 0.5,
              size: 15,
              borderColor: Constants.nodeBorderColor,
              borderSize: 0.2
            })

            // Set graph to store
            state.setSigmaGraph(emptyGraph)
            state.setRawGraph(null)
            // The placeholder carries one synthetic node; the graph is logically
            // empty, so override the reactive counts back to 0/0.
            state.setGraphCounts(0, 0)

            // Still mark graph as empty for other logic
            state.setGraphIsEmpty(true)

            // Check if the empty graph is due to 401 authentication error
            const errorMessage = useBackendState.getState().message
            const isAuthError = errorMessage && errorMessage.includes('Authentication required')

            // Only clear queryLabel if it's not an auth error and current label is not empty
            if (!isAuthError && currentQueryLabel) {
              useSettingsStore.getState().setQueryLabel('')
            }

            // Only clear last successful query label if it's not an auth error
            if (!isAuthError) {
              state.setLastSuccessfulQueryLabel('')
            } else {
              console.log('Keep queryLabel for post-login reload')
            }

            console.log(
              `Graph data is empty, created graph with empty graph node. Auth error: ${isAuthError}`
            )
          } else {
            // Create and set new graph. Wrapped separately so the console
            // tells us WHERE a failure happens — a build or subscriber error
            // here would otherwise be indistinguishable from a fetch error.
            let newSigmaGraph: UndirectedGraph | null
            try {
              newSigmaGraph = await createSigmaGraph(data)
            } catch (buildError) {
              console.error('[useLightragGraph] createSigmaGraph failed (graph build):', buildError)
              throw new GraphBuildError(buildError)
            }

            try {
              // Set new graph data
              state.setSigmaGraph(newSigmaGraph)
              state.setRawGraph(data)
              state.setGraphIsEmpty(false)
            } catch (subscriberError) {
              console.error(
                '[useLightragGraph] a store/sigma subscriber threw while applying the new graph:',
                subscriberError
              )
              throw new GraphBuildError(subscriberError)
            }

            // Update last successful query label
            state.setLastSuccessfulQueryLabel(currentQueryLabel)

            console.log(
              `[useLightragGraph] sigma graph ready: ${newSigmaGraph?.order} nodes, ${newSigmaGraph?.size} edges`
            )

            // Reset camera view
            state.setMoveToSelectedNode(true)
          }

          // Fetch succeeded: stamp the signature so a later reset of
          // graphDataFetchAttempted cannot re-issue this completed query in a
          // loop, and clear the retry budget.
          lastFetchSignatureRef.current = fetchSignature
          retryStateRef.current = { signature: '', attempts: 0 }

          // Update flags
          dataLoadedRef.current = true
          initialLoadRef.current = true
          fetchInProgressRef.current = false
          state.setIsFetching(false)

          // Mark empty data as handled if data is empty and query label is empty
          if ((!data || !data.nodes || data.nodes.length === 0) && !currentQueryLabel) {
            emptyDataHandledRef.current = true
          }
        })
        .catch((error) => {
          console.error(
            '[useLightragGraph] graph load failed (see preceding log for stage):',
            error
          )

          // Reset state on error
          const state = useGraphStore.getState()
          state.setIsFetching(false)
          dataLoadedRef.current = false
          fetchInProgressRef.current = false
          state.setLastSuccessfulQueryLabel('') // Clear last successful query label on error

          // Terminal failure: the network fetch succeeded but building/applying
          // the graph failed. This is deterministic, so do NOT retry (that would
          // re-fetch identical data and fail identically). Stamp the signature
          // to suppress re-issue and surface the error once.
          if (error instanceof GraphBuildError) {
            console.error(
              '[useLightragGraph] graph build failed (not retrying — deterministic):',
              fetchSignature
            )
            lastFetchSignatureRef.current = fetchSignature
            retryStateRef.current = { signature: '', attempts: 0 }
            toast.error(
              t('graphPanel.graphBuildFailed', 'Failed to render graph data. Use refresh to retry.')
            )
            return
          }

          // Bounded retry. graphDataFetchAttempted is KEPT true here so the
          // immediate re-fire (isFetching flips false -> effect deps change)
          // is blocked at the gate above. Recovery is driven only by the
          // backoff timer below, never by an instant retry — that instant
          // path was the original unbounded zero-backoff loop.
          const retry = retryStateRef.current
          if (retry.signature !== fetchSignature) {
            retry.signature = fetchSignature
            retry.attempts = 0
          }

          if (retry.attempts < MAX_FETCH_RETRIES) {
            retry.attempts += 1
            const delay = RETRY_BASE_DELAY_MS * 2 ** (retry.attempts - 1)
            console.warn(
              `[useLightragGraph] graph fetch failed, retry ${retry.attempts}/${MAX_FETCH_RETRIES} in ${delay}ms:`,
              fetchSignature
            )
            if (retryTimerRef.current !== null) {
              clearTimeout(retryTimerRef.current)
            }
            retryTimerRef.current = setTimeout(() => {
              retryTimerRef.current = null
              // Re-open the gate and bump the nonce (NOT graphDataVersion, so
              // the signature — and therefore the retry counter — is stable)
              // to re-run the fetch effect for the same query.
              useGraphStore.getState().setGraphDataFetchAttempted(false)
              setRetryNonce((n) => n + 1)
            }, delay)
          } else {
            // Retries exhausted: stamp the signature to suppress further
            // re-issues. The query can still recover via a parameter change or
            // an explicit refresh (graphDataVersion bump), both of which yield
            // a new signature and reset the retry budget.
            console.error(
              `[useLightragGraph] graph fetch failed after ${MAX_FETCH_RETRIES} retries, giving up:`,
              fetchSignature
            )
            lastFetchSignatureRef.current = fetchSignature
            toast.error(
              t(
                'graphPanel.fetchRetriesExhausted',
                'Failed to load graph data. Use refresh to try again.'
              )
            )
          }
        })
    }
  }, [queryLabel, maxQueryDepth, maxNodes, isFetching, t, graphDataVersion, retryNonce])

  // Clean up any pending backoff retry timer on unmount.
  useEffect(() => {
    return () => {
      if (retryTimerRef.current !== null) {
        clearTimeout(retryTimerRef.current)
        retryTimerRef.current = null
      }
    }
  }, [])

  // Handle node expansion
  useEffect(() => {
    const nodeId = useGraphStore.getState().nodeToExpand
    if (!nodeId) return

    const handleNodeExpand = async () => {
      const { sigmaGraph, rawGraph } = useGraphStore.getState()
      if (!sigmaGraph || !rawGraph) return

      try {
        // Get the node to expand
        const nodeToExpand = rawGraph.getNode(nodeId)
        if (!nodeToExpand) {
          console.error('Node not found:', nodeId)
          return
        }

        // Get the label of the node to expand
        const label = nodeToExpand.labels[0]
        if (!label) {
          console.error('Node has no label:', nodeId)
          return
        }

        // Fetch the extended subgraph with depth 2
        const extendedGraph = await queryGraphs(label, 2, 1000)

        if (!extendedGraph || !extendedGraph.nodes || !extendedGraph.edges) {
          console.error('Failed to fetch extended graph')
          return
        }

        // Process nodes to add required properties for RawNodeType
        const typeColors = createTypeColorResolver()
        const processedNodes: RawNodeType[] = []
        for (const node of extendedGraph.nodes) {
          const { x, y } = hashNodeIdToPosition(node.id)
          const color = typeColors.colorFor(node.properties?.entity_type as string | undefined)

          // Create a properly typed RawNodeType
          processedNodes.push({
            id: node.id,
            labels: node.labels,
            properties: node.properties,
            size: 10, // Default size, will be calculated later
            x, // Deterministic position, will be adjusted later
            y, // Deterministic position, will be adjusted later
            color: color,
            degree: 0 // Initial degree, will be calculated later
          })
        }
        typeColors.commit()

        // Process edges to add required properties for RawEdgeType
        const processedEdges: RawEdgeType[] = []
        for (const edge of extendedGraph.edges) {
          // Create a properly typed RawEdgeType
          processedEdges.push({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            type: edge.type,
            properties: edge.properties,
            dynamicId: '' // Will be set when adding to sigma graph
          })
        }

        // O(1) lookup structures. The code below used to re-scan
        // processedNodes/processedEdges with .find()/.some() inside loops,
        // making node expansion quadratic in the fetched subgraph size.
        const processedNodeById = new Map(processedNodes.map((n) => [n.id, n]))
        const processedEdgeById = new Map(processedEdges.map((e) => [e.id, e]))
        const neighborsOfExpandedNode = new Set<string>()
        for (const edge of processedEdges) {
          if (edge.source === nodeId) neighborsOfExpandedNode.add(edge.target)
          else if (edge.target === nodeId) neighborsOfExpandedNode.add(edge.source)
        }

        // Store current node positions (null-prototype: ids may be names
        // like '__proto__' that break plain-object assignment)
        const nodePositions: Record<string, { x: number; y: number }> = Object.create(null)
        sigmaGraph.forEachNode((node) => {
          nodePositions[node] = {
            x: sigmaGraph.getNodeAttribute(node, 'x'),
            y: sigmaGraph.getNodeAttribute(node, 'y')
          }
        })

        // Get existing node IDs
        const existingNodeIds = new Set(sigmaGraph.nodes())

        // Identify nodes and edges to keep
        const nodesToAdd = new Set<string>()
        const edgesToAdd = new Set<string>()

        // Get degree maxDegree from existing graph for size calculations
        const minDegree = 1
        let maxDegree = 0

        // Initialize edge weight min and max values
        let minWeight = Number.MAX_SAFE_INTEGER
        let maxWeight = 0

        // Calculate node degrees and edge weights from existing graph
        sigmaGraph.forEachNode((node) => {
          const degree = sigmaGraph.degree(node)
          maxDegree = Math.max(maxDegree, degree)
        })

        // Calculate edge weights from existing graph
        sigmaGraph.forEachEdge((edge) => {
          const weight = sigmaGraph.getEdgeAttribute(edge, 'originalWeight') || 1
          minWeight = Math.min(minWeight, weight)
          maxWeight = Math.max(maxWeight, weight)
        })

        // First identify connectable nodes (nodes connected to the expanded node)
        for (const node of processedNodes) {
          // Skip if node already exists
          if (existingNodeIds.has(node.id)) {
            continue
          }

          // Check if this node is connected to the selected node
          if (neighborsOfExpandedNode.has(node.id)) {
            nodesToAdd.add(node.id)
          }
        }

        // Calculate node degrees and track discarded edges in one pass
        const nodeDegrees = new Map<string, number>()
        const existingNodeDegreeIncrements = new Map<string, number>() // Track degree increments for existing nodes
        const nodesWithDiscardedEdges = new Set<string>()

        for (const edge of processedEdges) {
          const sourceExists = existingNodeIds.has(edge.source) || nodesToAdd.has(edge.source)
          const targetExists = existingNodeIds.has(edge.target) || nodesToAdd.has(edge.target)

          if (sourceExists && targetExists) {
            edgesToAdd.add(edge.id)
            // Add degrees for both new and existing nodes
            if (nodesToAdd.has(edge.source)) {
              nodeDegrees.set(edge.source, (nodeDegrees.get(edge.source) || 0) + 1)
            } else if (existingNodeIds.has(edge.source)) {
              // Track degree increments for existing nodes
              existingNodeDegreeIncrements.set(
                edge.source,
                (existingNodeDegreeIncrements.get(edge.source) || 0) + 1
              )
            }

            if (nodesToAdd.has(edge.target)) {
              nodeDegrees.set(edge.target, (nodeDegrees.get(edge.target) || 0) + 1)
            } else if (existingNodeIds.has(edge.target)) {
              // Track degree increments for existing nodes
              existingNodeDegreeIncrements.set(
                edge.target,
                (existingNodeDegreeIncrements.get(edge.target) || 0) + 1
              )
            }
          } else {
            // Track discarded edges for both new and existing nodes
            if (sigmaGraph.hasNode(edge.source)) {
              nodesWithDiscardedEdges.add(edge.source)
            } else if (nodesToAdd.has(edge.source)) {
              nodesWithDiscardedEdges.add(edge.source)
              nodeDegrees.set(edge.source, (nodeDegrees.get(edge.source) || 0) + 1) // +1 for discarded edge
            }
            if (sigmaGraph.hasNode(edge.target)) {
              nodesWithDiscardedEdges.add(edge.target)
            } else if (nodesToAdd.has(edge.target)) {
              nodesWithDiscardedEdges.add(edge.target)
              nodeDegrees.set(edge.target, (nodeDegrees.get(edge.target) || 0) + 1) // +1 for discarded edge
            }
          }
        }

        // Helper function to update node sizes
        const updateNodeSizes = (
          sigmaGraph: UndirectedGraph,
          nodesWithDiscardedEdges: Set<string>,
          minDegree: number,
          maxDegree: number
        ) => {
          // Calculate derived values inside the function
          const range = maxDegree - minDegree || 1 // Avoid division by zero
          const scale = Constants.maxNodeSize - Constants.minNodeSize

          // Update node sizes
          for (const nodeId of nodesWithDiscardedEdges) {
            if (sigmaGraph.hasNode(nodeId)) {
              let newDegree = sigmaGraph.degree(nodeId)
              newDegree += 1 // Add +1 for discarded edges
              // Limit newDegree to maxDegree + 1 to prevent nodes from being too large
              const limitedDegree = Math.min(newDegree, maxDegree + 1)

              const newSize = Math.round(
                Constants.minNodeSize + scale * Math.pow((limitedDegree - minDegree) / range, 0.5)
              )

              sigmaGraph.setNodeAttribute(nodeId, 'size', newSize)
            }
          }
        }

        // Helper function to update edge sizes
        const updateEdgeSizes = (
          sigmaGraph: UndirectedGraph,
          minWeight: number,
          maxWeight: number
        ) => {
          // Update edge sizes
          const minEdgeSize = useSettingsStore.getState().minEdgeSize
          const maxEdgeSize = useSettingsStore.getState().maxEdgeSize
          const weightRange = maxWeight - minWeight || 1 // Avoid division by zero
          const sizeScale = maxEdgeSize - minEdgeSize

          sigmaGraph.forEachEdge((edge) => {
            const weight = sigmaGraph.getEdgeAttribute(edge, 'originalWeight') || 1
            const scaledSize =
              minEdgeSize + sizeScale * Math.pow((weight - minWeight) / weightRange, 0.5)
            sigmaGraph.setEdgeAttribute(edge, 'size', scaledSize)
          })
        }

        // If no new connectable nodes found, show toast and return
        if (nodesToAdd.size === 0) {
          updateNodeSizes(sigmaGraph, nodesWithDiscardedEdges, minDegree, maxDegree)
          toast.info(t('graphPanel.propertiesView.node.noNewNodes'))
          return
        }

        // Update maxDegree considering all nodes (both new and existing)
        // 1. Consider degrees of new nodes
        for (const [, degree] of nodeDegrees.entries()) {
          maxDegree = Math.max(maxDegree, degree)
        }

        // 2. Consider degree increments for existing nodes
        for (const [nodeId, increment] of existingNodeDegreeIncrements.entries()) {
          const currentDegree = sigmaGraph.degree(nodeId)
          const projectedDegree = currentDegree + increment
          maxDegree = Math.max(maxDegree, projectedDegree)
        }

        const range = maxDegree - minDegree || 1 // Avoid division by zero
        const scale = Constants.maxNodeSize - Constants.minNodeSize

        // SAdd nodes and edges to the graph
        // Calculate camera ratio and spread factor once before the loop
        const cameraRatio = useGraphStore.getState().sigmaInstance?.getCamera().ratio || 1
        const spreadFactor =
          Math.max(
            Math.sqrt(nodeToExpand.size) * 4, // Base on node size
            Math.sqrt(nodesToAdd.size) * 3 // Scale with number of nodes
          ) / cameraRatio // Adjust for zoom level
        // Seeding a PRNG with the current time is non-deterministic anyway,
        // so Math.random() does the same job without the ARC4 setup cost.
        const randomAngle = Math.random() * 2 * Math.PI

        console.log('nodeSize:', nodeToExpand.size, 'nodesToAdd:', nodesToAdd.size)
        console.log(
          'cameraRatio:',
          Math.round(cameraRatio * 100) / 100,
          'spreadFactor:',
          Math.round(spreadFactor * 100) / 100
        )

        // Add new nodes
        const nodesToAddList = Array.from(nodesToAdd)
        for (let newNodeIdx = 0; newNodeIdx < nodesToAddList.length; newNodeIdx++) {
          const nodeId = nodesToAddList[newNodeIdx]
          const newNode = processedNodeById.get(nodeId)!
          const nodeDegree = nodeDegrees.get(nodeId) || 0

          // Calculate node size
          // Limit nodeDegree to maxDegree + 1 to prevent new nodes from being too large
          const limitedDegree = Math.min(nodeDegree, maxDegree + 1)
          const nodeSize = Math.round(
            Constants.minNodeSize + scale * Math.pow((limitedDegree - minDegree) / range, 0.5)
          )

          // Calculate angle for polar coordinates
          const angle = 2 * Math.PI * (newNodeIdx / nodesToAddList.length)

          // Calculate final position
          const x =
            nodePositions[nodeId]?.x ||
            nodePositions[nodeToExpand.id].x + Math.cos(randomAngle + angle) * spreadFactor
          const y =
            nodePositions[nodeId]?.y ||
            nodePositions[nodeToExpand.id].y + Math.sin(randomAngle + angle) * spreadFactor

          // Add the new node to the sigma graph with calculated position
          sigmaGraph.addNode(nodeId, {
            label: safeNodeLabel(newNode.labels, nodeId),
            color: newNode.color,
            x: x,
            y: y,
            size: nodeSize,
            borderColor: NODE_BORDER_COLOR
          })

          // Add the node to the raw graph
          if (!rawGraph.getNode(nodeId)) {
            // Update node properties
            newNode.size = nodeSize
            newNode.x = x
            newNode.y = y
            newNode.degree = nodeDegree

            // Add to nodes array
            rawGraph.nodes.push(newNode)
            // Update nodeIdMap
            rawGraph.nodeIdMap[nodeId] = rawGraph.nodes.length - 1
          }
        }

        // Add new edges
        for (const edgeId of edgesToAdd) {
          const newEdge = processedEdgeById.get(edgeId)!

          // Skip if edge already exists
          if (sigmaGraph.hasEdge(newEdge.source, newEdge.target)) {
            continue
          }

          // Get weight from edge properties or default to 1 (preserves 0)
          const weight = parseEdgeWeight(newEdge.properties)

          // Update min and max weight values
          minWeight = Math.min(minWeight, weight)
          maxWeight = Math.max(maxWeight, weight)

          // Add the edge to the sigma graph
          const edgeAttributes = {
            label: newEdge.properties?.keywords || undefined,
            size: weight, // Set initial size based on weight
            originalWeight: weight // Store original weight for recalculation
            // no `type`: use the default (cheap straight-line) edge program
          }
          const dynamicId = addUndirectedEdgeSafe(
            sigmaGraph,
            newEdge.source,
            newEdge.target,
            edgeAttributes
          )
          if (dynamicId === null) {
            console.warn('[useLightragGraph] could not add expansion edge:', newEdge.id)
            continue
          }
          newEdge.dynamicId = dynamicId

          // Add the edge to the raw graph
          if (!rawGraph.getEdge(newEdge.id, false)) {
            // Add to edges array
            rawGraph.edges.push(newEdge)
            // Update edgeIdMap
            rawGraph.edgeIdMap[newEdge.id] = rawGraph.edges.length - 1
            // Update dynamic edge map
            rawGraph.edgeDynamicIdMap[newEdge.dynamicId] = rawGraph.edges.length - 1
          } else {
            console.error('Edge already exists in rawGraph:', newEdge.id)
          }
        }

        // Update the dynamic edge map and invalidate search cache
        rawGraph.buildDynamicMap()

        // Reset search engine to force rebuild
        useGraphStore.getState().resetSearchEngine()

        // The graph was mutated in place (addNode/addEdge), which does NOT
        // re-notify the store. Refresh the reactive counts so the status bar and
        // the edge-count adaptive behavior react to the new size.
        useGraphStore.getState().setGraphCounts(sigmaGraph.order, sigmaGraph.size)

        // Update sizes for all nodes and edges
        updateNodeSizes(sigmaGraph, nodesWithDiscardedEdges, minDegree, maxDegree)
        updateEdgeSizes(sigmaGraph, minWeight, maxWeight)

        // Final update for the expanded node
        if (sigmaGraph.hasNode(nodeId)) {
          const finalDegree = sigmaGraph.degree(nodeId)
          const limitedDegree = Math.min(finalDegree, maxDegree + 1)
          const newSize = Math.round(
            Constants.minNodeSize + scale * Math.pow((limitedDegree - minDegree) / range, 0.5)
          )
          sigmaGraph.setNodeAttribute(nodeId, 'size', newSize)
          nodeToExpand.size = newSize
          nodeToExpand.degree = finalDegree
        }
      } catch (error) {
        console.error('Error expanding node:', error)
      }
    }

    handleNodeExpand()
    // Reset the nodeToExpand state after handling
    window.setTimeout(() => {
      useGraphStore.getState().triggerNodeExpand(null)
    }, 0)
  }, [nodeToExpand, sigmaGraph, rawGraph, t])

  // Helper function to get all nodes that will be deleted
  const getNodesThatWillBeDeleted = useCallback((nodeId: string, graph: UndirectedGraph) => {
    const nodesToDelete = new Set<string>([nodeId])

    // Only direct neighbors of the deleted node can become isolated by its
    // removal, so there is no need to scan (and allocate a neighbors array
    // for) every node in the graph.
    graph.forEachNeighbor(nodeId, (neighbor) => {
      // degree === 1 means this neighbor's only edge is the one to nodeId,
      // so it will become isolated and should be deleted too
      if (graph.degree(neighbor) === 1) {
        nodesToDelete.add(neighbor)
      }
    })

    return nodesToDelete
  }, [])

  // Handle node pruning
  useEffect(() => {
    const nodeId = useGraphStore.getState().nodeToPrune
    if (!nodeId) return

    const handleNodePrune = () => {
      const state = useGraphStore.getState()
      const { sigmaGraph, rawGraph } = state
      if (!sigmaGraph || !rawGraph) return

      try {
        // 1. Check if node exists
        if (!sigmaGraph.hasNode(nodeId)) {
          console.error('Node not found:', nodeId)
          return
        }

        // 2. Get nodes to delete
        const nodesToDelete = getNodesThatWillBeDeleted(nodeId, sigmaGraph)

        // 3. Check if this would delete all nodes
        if (nodesToDelete.size === sigmaGraph.nodes().length) {
          toast.error(t('graphPanel.propertiesView.node.deleteAllNodesError'))
          return
        }

        // 4. Clear selection - this will cause PropertiesView to close immediately
        state.clearSelection()

        // 5. Delete nodes and related edges
        // Drop from the sigma graph first (this also removes connected edges)
        for (const nodeToDelete of nodesToDelete) {
          sigmaGraph.dropNode(nodeToDelete)
        }

        // Rebuild the raw graph arrays and index maps in ONE pass.
        // The previous implementation spliced the arrays and re-walked
        // Object.entries(...) of the id maps for every single removed edge
        // and node — O(k * (V + E)) with an array allocation per walk — which
        // could freeze the tab when pruning a well-connected node on a large
        // graph.
        rawGraph.nodes = rawGraph.nodes.filter((node) => !nodesToDelete.has(node.id))
        rawGraph.edges = rawGraph.edges.filter(
          (edge) => !nodesToDelete.has(edge.source) && !nodesToDelete.has(edge.target)
        )

        rawGraph.nodeIdMap = Object.create(null) as Record<string, number>
        for (let i = 0; i < rawGraph.nodes.length; i++) {
          rawGraph.nodeIdMap[rawGraph.nodes[i].id] = i
        }

        rawGraph.edgeIdMap = Object.create(null) as Record<string, number>
        for (let i = 0; i < rawGraph.edges.length; i++) {
          rawGraph.edgeIdMap[rawGraph.edges[i].id] = i
        }

        // Rebuild the dynamic edge map and invalidate search cache
        rawGraph.buildDynamicMap()

        // Reset search engine to force rebuild
        useGraphStore.getState().resetSearchEngine()

        // In-place dropNode/dropEdge does NOT re-notify the store; refresh the
        // reactive counts so the status bar and edge-count adaptive behavior
        // react to the smaller graph (and auto-recover curves/edge events when
        // the edge count drops back to <= EDGE_PERF_LIMIT).
        useGraphStore.getState().setGraphCounts(sigmaGraph.order, sigmaGraph.size)

        // Show notification if we deleted more than just the selected node
        if (nodesToDelete.size > 1) {
          toast.info(
            t('graphPanel.propertiesView.node.nodesRemoved', { count: nodesToDelete.size })
          )
        }
      } catch (error) {
        console.error('Error pruning node:', error)
      }
    }

    handleNodePrune()
    // Reset the nodeToPrune state after handling
    window.setTimeout(() => {
      useGraphStore.getState().triggerNodePrune(null)
    }, 0)
  }, [nodeToPrune, sigmaGraph, rawGraph, getNodesThatWillBeDeleted, t])

  const lightrageGraph = useCallback(() => {
    // If we already have a graph instance, return it
    if (sigmaGraph) {
      return sigmaGraph as Graph<NodeType, EdgeType>
    }

    // If no graph exists yet, create a new one and store it
    console.log('Creating new Sigma graph instance')
    const graph = new UndirectedGraph()
    useGraphStore.getState().setSigmaGraph(graph)
    return graph as Graph<NodeType, EdgeType>
  }, [sigmaGraph])

  return { lightrageGraph, getNode, getEdge }
}

export default useLightrangeGraph
