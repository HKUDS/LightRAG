import Graph, { DirectedGraph } from 'graphology'
import { useCallback, useEffect, useRef } from 'react'
import { randomColor, errorMessage } from '@/lib/utils'
import * as Constants from '@/lib/constants'
import { useGraphStore, RawGraph } from '@/stores/graph'
import { queryGraphs } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useTabVisibility } from '@/contexts/useTabVisibility'

import seedrandom from 'seedrandom'

const validateGraph = (graph: RawGraph) => {
  if (!graph) {
    return false
  }
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    return false
  }

  for (const node of graph.nodes) {
    if (!node.id || !node.labels || !node.properties) {
      return false
    }
  }

  for (const edge of graph.edges) {
    if (!edge.id || !edge.source || !edge.target) {
      return false
    }
  }

  for (const edge of graph.edges) {
    const source = graph.getNode(edge.source)
    const target = graph.getNode(edge.target)
    if (source == undefined || target == undefined) {
      return false
    }
  }

  return true
}

export type NodeType = {
  x: number
  y: number
  label: string
  size: number
  color: string
  highlighted?: boolean
}
export type EdgeType = { label: string }

const fetchGraph = async (label: string, maxDepth: number, minDegree: number) => {
  let rawData: any = null

  try {
    rawData = await queryGraphs(label, maxDepth, minDegree)
  } catch (e) {
    useBackendState.getState().setErrorMessage(errorMessage(e), 'Query Graphs Error!')
    return null
  }

  let rawGraph = null

  if (rawData) {
    const nodeIdMap: Record<string, number> = {}
    const edgeIdMap: Record<string, number> = {}

    for (let i = 0; i < rawData.nodes.length; i++) {
      const node = rawData.nodes[i]
      nodeIdMap[node.id] = i

      // const seed = node.labels.length > 0 ? node.labels[0] : node.id
      seedrandom(node.id, { global: true })
      node.color = randomColor()
      node.x = Math.random()
      node.y = Math.random()
      node.degree = 0
      node.size = 10
    }

    for (let i = 0; i < rawData.edges.length; i++) {
      const edge = rawData.edges[i]
      edgeIdMap[edge.id] = i

      const source = nodeIdMap[edge.source]
      const target = nodeIdMap[edge.target]
      if (source !== undefined && source !== undefined) {
        const sourceNode = rawData.nodes[source]
        const targetNode = rawData.nodes[target]
        if (!sourceNode) {
          console.error(`Source node ${edge.source} is undefined`)
          continue
        }
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

    if (!validateGraph(rawGraph)) {
      rawGraph = null
      console.error('Invalid graph data')
    }
    console.log('Graph data loaded')
  }

  // console.debug({ data: JSON.parse(JSON.stringify(rawData)) })
  return rawGraph
}

const createSigmaGraph = (rawGraph: RawGraph | null) => {
  const graph = new DirectedGraph()

  for (const rawNode of rawGraph?.nodes ?? []) {
    graph.addNode(rawNode.id, {
      label: rawNode.labels.join(', '),
      color: rawNode.color,
      x: rawNode.x,
      y: rawNode.y,
      size: rawNode.size,
      // for node-border
      borderColor: Constants.nodeBorderColor,
      borderSize: 0.2
    })
  }

  for (const rawEdge of rawGraph?.edges ?? []) {
    rawEdge.dynamicId = graph.addDirectedEdge(rawEdge.source, rawEdge.target, {
      label: rawEdge.type || undefined
    })
  }

  return graph
}

const useLightrangeGraph = () => {
  const queryLabel = useSettingsStore.use.queryLabel()
  const rawGraph = useGraphStore.use.rawGraph()
  const sigmaGraph = useGraphStore.use.sigmaGraph()
  const maxQueryDepth = useSettingsStore.use.graphQueryMaxDepth()
  const minDegree = useSettingsStore.use.graphMinDegree()
  const isFetching = useGraphStore.use.isFetching()
  
  // Get tab visibility
  const { isTabVisible } = useTabVisibility()
  const isGraphTabVisible = isTabVisible('knowledge-graph')
  
  // Track previous parameters to detect actual changes
  const prevParamsRef = useRef({ queryLabel, maxQueryDepth, minDegree })
  
  // Use ref to track if data has been loaded and initial load
  const dataLoadedRef = useRef(false)
  const initialLoadRef = useRef(false)
  
  // Check if parameters have changed
  const paramsChanged = 
    prevParamsRef.current.queryLabel !== queryLabel ||
    prevParamsRef.current.maxQueryDepth !== maxQueryDepth ||
    prevParamsRef.current.minDegree !== minDegree

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
  
  // Data fetching logic - use a separate effect with minimal dependencies to prevent multiple triggers
  useEffect(() => {
    // Skip if fetch is already in progress
    if (fetchInProgressRef.current) {
      return
    }
    
    // If there's no query label, reset the graph only if it hasn't been reset already
    if (!queryLabel) {
      if (rawGraph !== null || sigmaGraph !== null) {
        const state = useGraphStore.getState()
        state.reset()
        state.setSigmaGraph(new DirectedGraph())
        state.setGraphLabels(['*'])
        // Reset fetch attempt flags when resetting graph
        state.setGraphDataFetchAttempted(false)
        state.setLabelsFetchAttempted(false)
      }
      dataLoadedRef.current = false
      initialLoadRef.current = false
      return
    }
    
    // Check if we've already attempted to fetch this data in this session
    const graphDataFetchAttempted = useGraphStore.getState().graphDataFetchAttempted
    
    // Fetch data if:
    // 1. We're not already fetching
    // 2. We haven't attempted to fetch in this session OR parameters have changed
    if (!isFetching && !fetchInProgressRef.current && (!graphDataFetchAttempted || paramsChanged)) {
      // Set flag to prevent multiple fetches
      fetchInProgressRef.current = true
      // Set global flag to indicate we've attempted to fetch in this session
      useGraphStore.getState().setGraphDataFetchAttempted(true)
      
      const state = useGraphStore.getState()
      
      // Set rendering control state
      state.setIsFetching(true)
      state.setShouldRender(false) // Disable rendering during data loading
      
      // Clear selection and highlighted nodes before fetching new graph
      state.clearSelection()
      if (state.sigmaGraph) {
        state.sigmaGraph.forEachNode((node) => {
          state.sigmaGraph?.setNodeAttribute(node, 'highlighted', false)
        })
      }
      
      // Update parameter reference
      prevParamsRef.current = { queryLabel, maxQueryDepth, minDegree }
      
      console.log('Fetching graph data (once per session unless params change)...')
      
      // Use a local copy of the parameters to avoid closure issues
      const currentQueryLabel = queryLabel
      const currentMaxQueryDepth = maxQueryDepth
      const currentMinDegree = minDegree
      
      fetchGraph(currentQueryLabel, currentMaxQueryDepth, currentMinDegree).then((data) => {
        const state = useGraphStore.getState()
        const newSigmaGraph = createSigmaGraph(data)
        data?.buildDynamicMap()
        
        // Update all graph data at once to minimize UI flicker
        state.clearSelection()
        state.setMoveToSelectedNode(false)
        state.setSigmaGraph(newSigmaGraph)
        state.setRawGraph(data)
        
        // Extract labels from current graph data for local use
        if (data) {
          const labelSet = new Set<string>()
          for (const node of data.nodes) {
            if (node.labels && Array.isArray(node.labels)) {
              for (const label of node.labels) {
                if (label !== '*') {  // filter out label "*"
                  labelSet.add(label)
                }
              }
            }
          }
          // Put * on top of other labels
          const sortedLabels = Array.from(labelSet).sort()
          state.setGraphLabels(['*', ...sortedLabels])
        } else {
          // Ensure * is there eventhough there is no graph data
          state.setGraphLabels(['*'])
        }
        
        // Mark data as loaded and initial load completed
        dataLoadedRef.current = true
        initialLoadRef.current = true
        fetchInProgressRef.current = false
        
        // Reset camera view by triggering FocusOnNode component
        state.setMoveToSelectedNode(true)
        
        // Enable rendering if the tab is visible
        state.setShouldRender(isGraphTabVisible)
        state.setIsFetching(false)
      }).catch((error) => {
        console.error('Error fetching graph data:', error)
        // Reset fetching state and remove flag in case of error
        const state = useGraphStore.getState()
        state.setIsFetching(false)
        state.setShouldRender(isGraphTabVisible) // Restore rendering state
        dataLoadedRef.current = false // Allow retry
        fetchInProgressRef.current = false
        // Reset global flag to allow retry
        state.setGraphDataFetchAttempted(false)
      })
    }
  }, [queryLabel, maxQueryDepth, minDegree, isFetching, paramsChanged, isGraphTabVisible, rawGraph, sigmaGraph]) // Added missing dependencies
  
  // Update rendering state when tab visibility changes
  useEffect(() => {
    // Only update rendering state if data is loaded and not fetching
    if (rawGraph) {
      useGraphStore.getState().setShouldRender(isGraphTabVisible)
    }
  }, [isGraphTabVisible, rawGraph])

  const lightrageGraph = useCallback(() => {
    if (sigmaGraph) {
      return sigmaGraph as Graph<NodeType, EdgeType>
    }
    const graph = new DirectedGraph()
    useGraphStore.getState().setSigmaGraph(graph)
    return graph as Graph<NodeType, EdgeType>
  }, [sigmaGraph])

  return { lightrageGraph, getNode, getEdge }
}

export default useLightrangeGraph
