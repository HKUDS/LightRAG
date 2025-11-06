import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { DirectedGraph } from 'graphology'
import MiniSearch from 'minisearch'
import { resolveNodeColor, DEFAULT_NODE_COLOR } from '@/utils/graphColor'

export type RawNodeType = {
  // for NetworkX: id is identical to properties['entity_id']
  // for Neo4j: id is unique identifier for each node
  id: string
  labels: string[]
  properties: Record<string, any>

  size: number
  x: number
  y: number
  color: string

  degree: number
}

export type RawEdgeType = {
  // for NetworkX: id is "source-target"
  // for Neo4j: id is unique identifier for each edge
  id: string
  source: string
  target: string
  type?: string
  properties: Record<string, any>
  // dynamicId: key for sigmaGraph
  dynamicId: string
}

/**
 * Interface for tracking edges that need updating when a node ID changes
 */
interface EdgeToUpdate {
  originalDynamicId: string
  newEdgeId: string
  edgeIndex: number
}

export class RawGraph {
  nodes: RawNodeType[] = []
  edges: RawEdgeType[] = []
  // nodeIDMap: map node id to index in nodes array (SigmaGraph has nodeId as key)
  nodeIdMap: Record<string, number> = {}
  // edgeIDMap: map edge id to index in edges array (SigmaGraph not use id as key)
  edgeIdMap: Record<string, number> = {}
  // edgeDynamicIdMap: map edge dynamic id to index in edges array (SigmaGraph has DynamicId as key)
  edgeDynamicIdMap: Record<string, number> = {}

  getNode = (nodeId: string) => {
    const nodeIndex = this.nodeIdMap[nodeId]
    if (nodeIndex !== undefined) {
      return this.nodes[nodeIndex]
    }
    return undefined
  }

  getEdge = (edgeId: string, dynamicId: boolean = true) => {
    const edgeIndex = dynamicId ? this.edgeDynamicIdMap[edgeId] : this.edgeIdMap[edgeId]
    if (edgeIndex !== undefined) {
      return this.edges[edgeIndex]
    }
    return undefined
  }

  buildDynamicMap = () => {
    this.edgeDynamicIdMap = {}
    for (let i = 0; i < this.edges.length; i++) {
      const edge = this.edges[i]
      this.edgeDynamicIdMap[edge.dynamicId] = i
    }
  }
}

interface GraphState {
  selectedNode: string | null
  focusedNode: string | null
  selectedEdge: string | null
  focusedEdge: string | null

  rawGraph: RawGraph | null
  sigmaGraph: DirectedGraph | null
  sigmaInstance: any | null

  searchEngine: MiniSearch | null

  moveToSelectedNode: boolean
  isFetching: boolean
  graphIsEmpty: boolean
  lastSuccessfulQueryLabel: string

  typeColorMap: Map<string, string>

  // Global flags to track data fetching attempts
  graphDataFetchAttempted: boolean
  labelsFetchAttempted: boolean

  setSigmaInstance: (instance: any) => void
  setSelectedNode: (nodeId: string | null, moveToSelectedNode?: boolean) => void
  setFocusedNode: (nodeId: string | null) => void
  setSelectedEdge: (edgeId: string | null) => void
  setFocusedEdge: (edgeId: string | null) => void
  clearSelection: () => void
  reset: () => void

  setMoveToSelectedNode: (moveToSelectedNode: boolean) => void
  setGraphIsEmpty: (isEmpty: boolean) => void
  setLastSuccessfulQueryLabel: (label: string) => void

  setRawGraph: (rawGraph: RawGraph | null) => void
  setSigmaGraph: (sigmaGraph: DirectedGraph | null) => void
  setIsFetching: (isFetching: boolean) => void

  // Legend color mapping methods
  setTypeColorMap: (typeColorMap: Map<string, string>) => void

  // Search engine methods
  setSearchEngine: (engine: MiniSearch | null) => void
  resetSearchEngine: () => void

  // Methods to set global flags
  setGraphDataFetchAttempted: (attempted: boolean) => void
  setLabelsFetchAttempted: (attempted: boolean) => void

  // Event trigger methods for node operations
  triggerNodeExpand: (nodeId: string | null) => void
  triggerNodePrune: (nodeId: string | null) => void

  // Node operation state
  nodeToExpand: string | null
  nodeToPrune: string | null

  // Version counter to trigger data refresh
  graphDataVersion: number
  incrementGraphDataVersion: () => void

  // Methods for updating graph elements and UI state together
  updateNodeAndSelect: (nodeId: string, entityId: string, propertyName: string, newValue: string) => Promise<void>
  updateEdgeAndSelect: (edgeId: string, dynamicId: string, sourceId: string, targetId: string, propertyName: string, newValue: string) => Promise<void>
}

const useGraphStoreBase = create<GraphState>()((set, get) => ({
  selectedNode: null,
  focusedNode: null,
  selectedEdge: null,
  focusedEdge: null,

  moveToSelectedNode: false,
  isFetching: false,
  graphIsEmpty: false,
  lastSuccessfulQueryLabel: '', // Initialize as empty to ensure fetchAllDatabaseLabels runs on first query

  // Initialize global flags
  graphDataFetchAttempted: false,
  labelsFetchAttempted: false,

  rawGraph: null,
  sigmaGraph: null,
  sigmaInstance: null,

  typeColorMap: new Map<string, string>(),

  searchEngine: null,

  setGraphIsEmpty: (isEmpty: boolean) => set({ graphIsEmpty: isEmpty }),
  setLastSuccessfulQueryLabel: (label: string) => set({ lastSuccessfulQueryLabel: label }),


  setIsFetching: (isFetching: boolean) => set({ isFetching }),
  setSelectedNode: (nodeId: string | null, moveToSelectedNode?: boolean) =>
    set({ selectedNode: nodeId, moveToSelectedNode }),
  setFocusedNode: (nodeId: string | null) => set({ focusedNode: nodeId }),
  setSelectedEdge: (edgeId: string | null) => set({ selectedEdge: edgeId }),
  setFocusedEdge: (edgeId: string | null) => set({ focusedEdge: edgeId }),
  clearSelection: () =>
    set({
      selectedNode: null,
      focusedNode: null,
      selectedEdge: null,
      focusedEdge: null
    }),
  reset: () => {
    set({
      selectedNode: null,
      focusedNode: null,
      selectedEdge: null,
      focusedEdge: null,
      rawGraph: null,
      sigmaGraph: null,  // to avoid other components from acccessing graph objects
      searchEngine: null,
      moveToSelectedNode: false,
      graphIsEmpty: false
    });
  },

  setRawGraph: (rawGraph: RawGraph | null) =>
    set({
      rawGraph
    }),

  setSigmaGraph: (sigmaGraph: DirectedGraph | null) => {
    // Replace graph instance, no need to keep WebGL context
    set({ sigmaGraph });
  },

  setMoveToSelectedNode: (moveToSelectedNode?: boolean) => set({ moveToSelectedNode }),

  setSigmaInstance: (instance: any) => set({ sigmaInstance: instance }),

  setTypeColorMap: (typeColorMap: Map<string, string>) => set({ typeColorMap }),

  setSearchEngine: (engine: MiniSearch | null) => set({ searchEngine: engine }),
  resetSearchEngine: () => set({ searchEngine: null }),

  // Methods to set global flags
  setGraphDataFetchAttempted: (attempted: boolean) => set({ graphDataFetchAttempted: attempted }),
  setLabelsFetchAttempted: (attempted: boolean) => set({ labelsFetchAttempted: attempted }),

  // Node operation state
  nodeToExpand: null,
  nodeToPrune: null,

  // Event trigger methods for node operations
  triggerNodeExpand: (nodeId: string | null) => set({ nodeToExpand: nodeId }),
  triggerNodePrune: (nodeId: string | null) => set({ nodeToPrune: nodeId }),

  // Version counter implementation
  graphDataVersion: 0,
  incrementGraphDataVersion: () => set((state) => ({ graphDataVersion: state.graphDataVersion + 1 })),

  // Methods for updating graph elements and UI state together
  updateNodeAndSelect: async (nodeId: string, entityId: string, propertyName: string, newValue: string) => {
    // Get current state
    const state = get()
    const { sigmaGraph, rawGraph } = state

    // Validate graph state
    if (!sigmaGraph || !rawGraph || !sigmaGraph.hasNode(nodeId)) {
      return
    }

    try {
      const nodeAttributes = sigmaGraph.getNodeAttributes(nodeId)

      console.log('updateNodeAndSelect', nodeId, entityId, propertyName, newValue)

      // For entity_id changes (node renaming) with raw graph storage
      if ((nodeId === entityId) && (propertyName === 'entity_id')) {
        // Create new node with updated ID but same attributes
        sigmaGraph.addNode(newValue, { ...nodeAttributes, label: newValue })

        const edgesToUpdate: EdgeToUpdate[] = []

        // Process all edges connected to this node
        sigmaGraph.forEachEdge(nodeId, (edge, attributes, source, target) => {
          const otherNode = source === nodeId ? target : source
          const isOutgoing = source === nodeId

          // Get original edge dynamic ID for later reference
          const originalEdgeDynamicId = edge
          const edgeIndexInRawGraph = rawGraph.edgeDynamicIdMap[originalEdgeDynamicId]

          // Create new edge with updated node reference
          const newEdgeId = sigmaGraph.addEdge(
            isOutgoing ? newValue : otherNode,
            isOutgoing ? otherNode : newValue,
            attributes
          )

          // Track edges that need updating in the raw graph
          if (edgeIndexInRawGraph !== undefined) {
            edgesToUpdate.push({
              originalDynamicId: originalEdgeDynamicId,
              newEdgeId: newEdgeId,
              edgeIndex: edgeIndexInRawGraph
            })
          }

          // Remove the old edge
          sigmaGraph.dropEdge(edge)
        })

        // Remove the old node after all edges are processed
        sigmaGraph.dropNode(nodeId)

        // Update node reference in raw graph data
        const nodeIndex = rawGraph.nodeIdMap[nodeId]
        if (nodeIndex !== undefined) {
          rawGraph.nodes[nodeIndex].id = newValue
          rawGraph.nodes[nodeIndex].labels = [newValue]
          rawGraph.nodes[nodeIndex].properties.entity_id = newValue
          delete rawGraph.nodeIdMap[nodeId]
          rawGraph.nodeIdMap[newValue] = nodeIndex
        }

        // Update all edge references in raw graph data
        edgesToUpdate.forEach(({ originalDynamicId, newEdgeId, edgeIndex }) => {
          if (rawGraph.edges[edgeIndex]) {
            // Update source/target references
            if (rawGraph.edges[edgeIndex].source === nodeId) {
              rawGraph.edges[edgeIndex].source = newValue
            }
            if (rawGraph.edges[edgeIndex].target === nodeId) {
              rawGraph.edges[edgeIndex].target = newValue
            }

            // Update dynamic ID mappings
            rawGraph.edges[edgeIndex].dynamicId = newEdgeId
            delete rawGraph.edgeDynamicIdMap[originalDynamicId]
            rawGraph.edgeDynamicIdMap[newEdgeId] = edgeIndex
          }
        })

        // Update selected node in store
        set({ selectedNode: newValue, moveToSelectedNode: true })
      } else {
        // For non-NetworkX nodes or non-entity_id changes
        const nodeIndex = rawGraph.nodeIdMap[String(nodeId)]
        if (nodeIndex !== undefined) {
          const nodeRef = rawGraph.nodes[nodeIndex]
          nodeRef.properties[propertyName] = newValue
          if (propertyName === 'entity_id') {
            nodeRef.labels = [newValue]
            sigmaGraph.setNodeAttribute(String(nodeId), 'label', newValue)
          }
          if (propertyName === 'entity_type') {
            const { color, map, updated } = resolveNodeColor(newValue, state.typeColorMap)
            const resolvedColor = color || DEFAULT_NODE_COLOR
            nodeRef.color = resolvedColor
            sigmaGraph.setNodeAttribute(String(nodeId), 'color', resolvedColor)
            if (updated) {
              set({ typeColorMap: map })
            }
          }
        }

        // Trigger a re-render by incrementing the version counter
        set((state) => ({ graphDataVersion: state.graphDataVersion + 1 }))
      }
    } catch (error) {
      console.error('Error updating node in graph:', error)
      throw new Error('Failed to update node in graph')
    }
  },

  updateEdgeAndSelect: async (edgeId: string, dynamicId: string, sourceId: string, targetId: string, propertyName: string, newValue: string) => {
    // Get current state
    const state = get()
    const { sigmaGraph, rawGraph } = state

    // Validate graph state
    if (!sigmaGraph || !rawGraph) {
      return
    }

    try {
      const edgeIndex = rawGraph.edgeIdMap[String(edgeId)]
      if (edgeIndex !== undefined && rawGraph.edges[edgeIndex]) {
        rawGraph.edges[edgeIndex].properties[propertyName] = newValue
        if(dynamicId !== undefined && propertyName === 'keywords') {
          sigmaGraph.setEdgeAttribute(dynamicId, 'label', newValue)
        }
      }

      // Trigger a re-render by incrementing the version counter
      set((state) => ({ graphDataVersion: state.graphDataVersion + 1 }))

      // Update selected edge in store to ensure UI reflects changes
      set({ selectedEdge: dynamicId })
    } catch (error) {
      console.error(`Error updating edge ${sourceId}->${targetId} in graph:`, error)
      throw new Error('Failed to update edge in graph')
    }
  }
}))

const useGraphStore = createSelectors(useGraphStoreBase)

export { useGraphStore }
