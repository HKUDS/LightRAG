import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { DirectedGraph } from 'graphology'
import { getGraphLabels } from '@/api/lightrag'

export type RawNodeType = {
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
  id: string
  source: string
  target: string
  type?: string
  properties: Record<string, any>

  dynamicId: string
}

export class RawGraph {
  nodes: RawNodeType[] = []
  edges: RawEdgeType[] = []
  nodeIdMap: Record<string, number> = {}
  edgeIdMap: Record<string, number> = {}
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
  allDatabaseLabels: string[]

  moveToSelectedNode: boolean
  isFetching: boolean
  shouldRender: boolean

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

  setRawGraph: (rawGraph: RawGraph | null) => void
  setSigmaGraph: (sigmaGraph: DirectedGraph | null) => void
  setAllDatabaseLabels: (labels: string[]) => void
  fetchAllDatabaseLabels: () => Promise<void>
  setIsFetching: (isFetching: boolean) => void
  setShouldRender: (shouldRender: boolean) => void

  // Methods to set global flags
  setGraphDataFetchAttempted: (attempted: boolean) => void
  setLabelsFetchAttempted: (attempted: boolean) => void

  // Event trigger methods for node operations
  triggerNodeExpand: (nodeId: string | null) => void
  triggerNodePrune: (nodeId: string | null) => void

  // Node operation state
  nodeToExpand: string | null
  nodeToPrune: string | null
}

const useGraphStoreBase = create<GraphState>()((set) => ({
  selectedNode: null,
  focusedNode: null,
  selectedEdge: null,
  focusedEdge: null,

  moveToSelectedNode: false,
  isFetching: false,
  shouldRender: false,

  // Initialize global flags
  graphDataFetchAttempted: false,
  labelsFetchAttempted: false,

  rawGraph: null,
  sigmaGraph: null,
  sigmaInstance: null,
  allDatabaseLabels: ['*'],


  setIsFetching: (isFetching: boolean) => set({ isFetching }),
  setShouldRender: (shouldRender: boolean) => set({ shouldRender }),
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
      moveToSelectedNode: false,
      shouldRender: false
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

  setAllDatabaseLabels: (labels: string[]) => set({ allDatabaseLabels: labels }),

  fetchAllDatabaseLabels: async () => {
    try {
      console.log('Fetching all database labels...');
      const labels = await getGraphLabels();
      set({ allDatabaseLabels: ['*', ...labels] });
      return;
    } catch (error) {
      console.error('Failed to fetch all database labels:', error);
      set({ allDatabaseLabels: ['*'] });
      throw error;
    }
  },

  setMoveToSelectedNode: (moveToSelectedNode?: boolean) => set({ moveToSelectedNode }),

  setSigmaInstance: (instance: any) => set({ sigmaInstance: instance }),

  // Methods to set global flags
  setGraphDataFetchAttempted: (attempted: boolean) => set({ graphDataFetchAttempted: attempted }),
  setLabelsFetchAttempted: (attempted: boolean) => set({ labelsFetchAttempted: attempted }),

  // Node operation state
  nodeToExpand: null,
  nodeToPrune: null,

  // Event trigger methods for node operations
  triggerNodeExpand: (nodeId: string | null) => set({ nodeToExpand: nodeId }),
  triggerNodePrune: (nodeId: string | null) => set({ nodeToPrune: nodeId }),

}))

const useGraphStore = createSelectors(useGraphStoreBase)

export { useGraphStore }
