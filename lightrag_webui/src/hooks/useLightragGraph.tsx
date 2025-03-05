import Graph, { DirectedGraph } from 'graphology'
import { useCallback, useEffect } from 'react'
import { randomColor, errorMessage } from '@/lib/utils'
import * as Constants from '@/lib/constants'
import { useGraphStore, RawGraph } from '@/stores/graph'
import { queryGraphs } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'

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

const lastQueryLabel = { label: '', maxQueryDepth: 0, minDegree: 0 }

const useLightrangeGraph = () => {
  const queryLabel = useSettingsStore.use.queryLabel()
  const rawGraph = useGraphStore.use.rawGraph()
  const sigmaGraph = useGraphStore.use.sigmaGraph()
  const maxQueryDepth = useSettingsStore.use.graphQueryMaxDepth()
  const minDegree = useSettingsStore.use.graphMinDegree()

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

  useEffect(() => {
    if (queryLabel) {
      if (lastQueryLabel.label !== queryLabel ||
          lastQueryLabel.maxQueryDepth !== maxQueryDepth ||
          lastQueryLabel.minDegree !== minDegree) {
        lastQueryLabel.label = queryLabel
        lastQueryLabel.maxQueryDepth = maxQueryDepth
        lastQueryLabel.minDegree = minDegree

        const state = useGraphStore.getState()
        state.reset()
        fetchGraph(queryLabel, maxQueryDepth, minDegree).then((data) => {
          // console.debug('Query label: ' + queryLabel)
          state.setSigmaGraph(createSigmaGraph(data))
          data?.buildDynamicMap()
          state.setRawGraph(data)
        })
      }
    } else {
      const state = useGraphStore.getState()
      state.reset()
      state.setSigmaGraph(new DirectedGraph())
    }
  }, [queryLabel, maxQueryDepth, minDegree])

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
