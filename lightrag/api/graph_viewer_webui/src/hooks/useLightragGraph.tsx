import Graph, { DirectedGraph } from 'graphology'
import { useCallback, useEffect, useState } from 'react'
import { randomColor } from '@/lib/utils'
import * as Constants from '@/lib/constants'
import { useGraphStore, RawGraph } from '@/stores/graph'
import { queryGraphs } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'

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
    if (!edge.id || !edge.source || !edge.target || !edge.type || !edge.properties) {
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

const fetchGraph = async (label: string) => {
  let rawData: any = null

  try {
    rawData = await queryGraphs(label)
  } catch (e) {
    useBackendState
      .getState()
      .setErrorMessage(e instanceof Error ? e.message : `${e}`, 'Query Graphs Error!')
    return null
  }

  let rawGraph = null

  if (rawData) {
    const nodeIdMap: Record<string, number> = {}
    const edgeIdMap: Record<string, number> = {}

    for (let i = 0; i < rawData.nodes.length; i++) {
      const node = rawData.nodes[i]
      nodeIdMap[node.id] = i

      node.x = Math.random()
      node.y = Math.random()
      node.color = randomColor()
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
      label: rawEdge.type
    })
  }

  return graph
}

const useLightrangeGraph = () => {
  const [fetchLabel, setFetchLabel] = useState<string>('*')
  const rawGraph = useGraphStore.use.rawGraph()
  const sigmaGraph = useGraphStore.use.sigmaGraph()

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
    if (fetchLabel) {
      const state = useGraphStore.getState()
      if (state.queryLabel !== fetchLabel) {
        state.reset()
        fetchGraph(fetchLabel).then((data) => {
          state.setQueryLabel(fetchLabel)
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
  }, [fetchLabel])

  const lightrageGraph = useCallback(() => {
    if (sigmaGraph) {
      return sigmaGraph as Graph<NodeType, EdgeType>
    }
    const graph = new DirectedGraph()
    useGraphStore.getState().setSigmaGraph(graph)
    return graph as Graph<NodeType, EdgeType>
  }, [sigmaGraph])

  return { lightrageGraph, fetchLabel, setFetchLabel, getNode, getEdge }
}

export default useLightrangeGraph
