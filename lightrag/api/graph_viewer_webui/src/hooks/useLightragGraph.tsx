import Graph, { DirectedGraph } from 'graphology'
import { useCallback, useEffect, useState } from 'react'
import { randomColor } from '@/lib/utils'
import * as Constants from '@/lib/constants'

type RawNodeType = {
  id: string
  labels: string[]
  properties: Record<string, any>

  size: number
  x: number
  y: number
  color: string

  degree: number
}

type RawEdgeType = {
  id: string
  source: string
  target: string
  type: string
  properties: Record<string, any>
}

class RawGraph {
  nodes: RawNodeType[] = []
  edges: RawEdgeType[] = []
  nodeIdMap: Record<string, number> = {}
  edgeIdMap: Record<string, number> = {}

  getNode = (nodeId: string) => {
    const nodeIndex = this.nodeIdMap[nodeId]
    if (nodeIndex !== undefined) {
      return this.nodes[nodeIndex]
    }
    return undefined
  }

  getEdge = (edgeId: string) => {
    const edgeIndex = this.edgeIdMap[edgeId]
    if (edgeIndex !== undefined) {
      return this.edges[edgeIndex]
    }
    return undefined
  }
}

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
  const response = await fetch(`http://localhost:9621/graphs?label=${label}`)
  const rawData = await response.json()

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

const graphCache: {
  label: string | null
  rawGraph: RawGraph | null
  convertedGraph: DirectedGraph | null
} = {
  label: null,
  rawGraph: null,
  convertedGraph: null
}

const useLightrangeGraph = () => {
  const [fetchLabel, setFetchLabel] = useState<string>('*')
  const [rawGraph, setRawGraph] = useState<RawGraph | null>(graphCache.rawGraph)

  useEffect(() => {
    if (fetchLabel) {
      if (graphCache.label !== fetchLabel) {
        fetchGraph(fetchLabel).then((data) => {
          graphCache.convertedGraph = null
          graphCache.rawGraph = data
          graphCache.label = fetchLabel
          setRawGraph(data)
        })
      }
    } else {
      setRawGraph(null)
    }
  }, [fetchLabel, setRawGraph])

  const lightrageGraph = useCallback(() => {
    if (graphCache.convertedGraph) {
      return graphCache.convertedGraph as Graph<NodeType, EdgeType>
    }

    // Create the graph
    const graph = new DirectedGraph()

    for (const rawNode of rawGraph?.nodes ?? []) {
      graph.addNode(rawNode.id, {
        label: rawNode.labels.join(' '),
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
      graph.addDirectedEdge(rawEdge.source, rawEdge.target, {
        label: rawEdge.type
      })
    }

    graphCache.convertedGraph = graph
    return graph as Graph<NodeType, EdgeType>
  }, [rawGraph])

  return { lightrageGraph, fetchLabel, setFetchLabel }
}

export default useLightrangeGraph
