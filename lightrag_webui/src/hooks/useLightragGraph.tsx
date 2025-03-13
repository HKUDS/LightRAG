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

// Create a new graph instance with the raw graph data
const createSigmaGraph = (rawGraph: RawGraph | null) => {
  // Always create a new graph instance
  const graph = new DirectedGraph()

  // Add nodes from raw graph data
  for (const rawNode of rawGraph?.nodes ?? []) {
    // Ensure we have fresh random positions for nodes
    seedrandom(rawNode.id + Date.now().toString(), { global: true })
    const x = Math.random()
    const y = Math.random()
    
    graph.addNode(rawNode.id, {
      label: rawNode.labels.join(', '),
      color: rawNode.color,
      x: x,
      y: y,
      size: rawNode.size,
      // for node-border
      borderColor: Constants.nodeBorderColor,
      borderSize: 0.2
    })
  }

  // Add edges from raw graph data
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
  
  // Data fetching logic - simplified but preserving TAB visibility check
  useEffect(() => {
    // Skip if fetch is already in progress
    if (fetchInProgressRef.current) {
      return
    }
    
    // If there's no query label, reset the graph
    if (!queryLabel) {
      if (rawGraph !== null || sigmaGraph !== null) {
        const state = useGraphStore.getState()
        state.reset()
        state.setGraphLabels(['*'])
        state.setGraphDataFetchAttempted(false)
        state.setLabelsFetchAttempted(false)
      }
      dataLoadedRef.current = false
      initialLoadRef.current = false
      return
    }
    
    // Check if parameters have changed
    if (!isFetching && !fetchInProgressRef.current && 
        (paramsChanged || !useGraphStore.getState().graphDataFetchAttempted)) {
      
      // Only fetch data if the Graph tab is visible
      if (!isGraphTabVisible) {
        console.log('Graph tab not visible, skipping data fetch');
        return;
      }
      
      // Set flags
      fetchInProgressRef.current = true
      useGraphStore.getState().setGraphDataFetchAttempted(true)
      
      const state = useGraphStore.getState()
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
      
      console.log('Fetching graph data...')
      
      // Use a local copy of the parameters
      const currentQueryLabel = queryLabel
      const currentMaxQueryDepth = maxQueryDepth
      const currentMinDegree = minDegree
      
      // Fetch graph data
      fetchGraph(currentQueryLabel, currentMaxQueryDepth, currentMinDegree).then((data) => {
        const state = useGraphStore.getState()
        
        // Reset state
        state.reset()
        
        // Create and set new graph directly
        const newSigmaGraph = createSigmaGraph(data)
        data?.buildDynamicMap()
        
        // Set new graph data
        state.setSigmaGraph(newSigmaGraph)
        state.setRawGraph(data)
        
        // Extract labels from current graph data
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
          state.setGraphLabels(['*'])
        }
        
        // Update flags
        dataLoadedRef.current = true
        initialLoadRef.current = true
        fetchInProgressRef.current = false
        
        // Reset camera view
        state.setMoveToSelectedNode(true)
        
        // Enable rendering if the tab is visible
        state.setShouldRender(isGraphTabVisible)
        state.setIsFetching(false)
      }).catch((error) => {
        console.error('Error fetching graph data:', error)
        
        // Reset state on error
        const state = useGraphStore.getState()
        state.setIsFetching(false)
        state.setShouldRender(isGraphTabVisible)
        dataLoadedRef.current = false
        fetchInProgressRef.current = false
        state.setGraphDataFetchAttempted(false)
      })
    }
  }, [queryLabel, maxQueryDepth, minDegree, isFetching, paramsChanged, isGraphTabVisible, rawGraph, sigmaGraph])
  
  // Update rendering state and handle tab visibility changes
  useEffect(() => {
    // When tab becomes visible
    if (isGraphTabVisible) {
      // If we have data, enable rendering
      if (rawGraph) {
        useGraphStore.getState().setShouldRender(true)
      }
      
      // We no longer reset the fetch attempted flag here to prevent continuous API calls
    } else {
      // When tab becomes invisible, disable rendering
      useGraphStore.getState().setShouldRender(false)
    }
  }, [isGraphTabVisible, rawGraph])

  const lightrageGraph = useCallback(() => {
    // If we already have a graph instance, return it
    if (sigmaGraph) {
      return sigmaGraph as Graph<NodeType, EdgeType>
    }
    
    // If no graph exists yet, create a new one and store it
    console.log('Creating new Sigma graph instance')
    const graph = new DirectedGraph()
    useGraphStore.getState().setSigmaGraph(graph)
    return graph as Graph<NodeType, EdgeType>
  }, [sigmaGraph])

  return { lightrageGraph, getNode, getEdge }
}

export default useLightrangeGraph
