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
  allDatabaseLabels: string[]

  moveToSelectedNode: boolean
  isFetching: boolean
  shouldRender: boolean

  // Global flags to track data fetching attempts
  graphDataFetchAttempted: boolean
  labelsFetchAttempted: boolean

  refreshLayout: () => void
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
}

const useGraphStoreBase = create<GraphState>()((set, get) => ({
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
  allDatabaseLabels: ['*'],

  refreshLayout: () => {
    const currentGraph = get().sigmaGraph;
    if (currentGraph) {
      get().clearSelection();
      get().setSigmaGraph(null);
      setTimeout(() => {
        get().setSigmaGraph(currentGraph);
      }, 10);
    }
  },

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
    // Get the existing graph
    const existingGraph = get().sigmaGraph;

    // If we have an existing graph, clear it by removing all nodes
    if (existingGraph) {
      const nodes = Array.from(existingGraph.nodes());
      nodes.forEach(node => existingGraph.dropNode(node));
    }

    set({
      selectedNode: null,
      focusedNode: null,
      selectedEdge: null,
      focusedEdge: null,
      rawGraph: null,
      // Keep the existing graph instance but with cleared data
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

  // Methods to set global flags
  setGraphDataFetchAttempted: (attempted: boolean) => set({ graphDataFetchAttempted: attempted }),
  setLabelsFetchAttempted: (attempted: boolean) => set({ labelsFetchAttempted: attempted })
}))

const useGraphStore = createSelectors(useGraphStoreBase)

export { useGraphStore }
