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

  // Legacy node expansion and pruning methods - will be removed after refactoring
  expandNode: async (nodeId: string) => {
    const state = get();
    if (!state.sigmaGraph || !state.rawGraph || !nodeId) {
      console.error('Cannot expand node: graph or node not available');
      return;
    }

    try {
      // Set fetching state
      state.setIsFetching(true);

      // Import queryGraphs dynamically to avoid circular dependency
      const { queryGraphs } = await import('@/api/lightrag');

      // Get the node to expand
      const nodeToExpand = state.rawGraph.getNode(nodeId);
      if (!nodeToExpand) {
        console.error('Node not found:', nodeId);
        state.setIsFetching(false);
        return;
      }

      // Get the label of the node to expand
      const label = nodeToExpand.labels[0];
      if (!label) {
        console.error('Node has no label:', nodeId);
        state.setIsFetching(false);
        return;
      }

      // Fetch the extended subgraph with depth 2
      const extendedGraph = await queryGraphs(label, 2, 0);

      if (!extendedGraph || !extendedGraph.nodes || !extendedGraph.edges) {
        console.error('Failed to fetch extended graph');
        state.setIsFetching(false);
        return;
      }

      // Process nodes to add required properties for RawNodeType
      const processedNodes: RawNodeType[] = [];
      for (const node of extendedGraph.nodes) {
        // Generate random color
        const randomColorValue = () => Math.floor(Math.random() * 256);
        const color = `rgb(${randomColorValue()}, ${randomColorValue()}, ${randomColorValue()})`;

        // Create a properly typed RawNodeType
        processedNodes.push({
          id: node.id,
          labels: node.labels,
          properties: node.properties,
          size: 10, // Default size
          x: Math.random(), // Random position
          y: Math.random(), // Random position
          color: color, // Random color
          degree: 0 // Initial degree
        });
      }

      // Process edges to add required properties for RawEdgeType
      const processedEdges: RawEdgeType[] = [];
      for (const edge of extendedGraph.edges) {
        // Create a properly typed RawEdgeType
        processedEdges.push({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          type: edge.type,
          properties: edge.properties,
          dynamicId: '' // Will be set when adding to sigma graph
        });
      }

      // Store current node positions
      const nodePositions: Record<string, {x: number, y: number}> = {};
      state.sigmaGraph.forEachNode((node) => {
        nodePositions[node] = {
          x: state.sigmaGraph!.getNodeAttribute(node, 'x'),
          y: state.sigmaGraph!.getNodeAttribute(node, 'y')
        };
      });

      // Get existing node IDs
      const existingNodeIds = new Set(state.sigmaGraph.nodes());

      // Create a map from id to processed node for quick lookup
      const processedNodeMap = new Map<string, RawNodeType>();
      for (const node of processedNodes) {
        processedNodeMap.set(node.id, node);
      }

      // Create a map from id to processed edge for quick lookup
      const processedEdgeMap = new Map<string, RawEdgeType>();
      for (const edge of processedEdges) {
        processedEdgeMap.set(edge.id, edge);
      }

      // Add new nodes from the processed nodes
      for (const newNode of processedNodes) {
        // Skip if node already exists
        if (existingNodeIds.has(newNode.id)) {
          continue;
        }

        // Check if this node is connected to the selected node
        const isConnected = processedEdges.some(
          edge => (edge.source === nodeId && edge.target === newNode.id) ||
                 (edge.target === nodeId && edge.source === newNode.id)
        );

        if (isConnected) {
          // Add the new node to the graph
          state.sigmaGraph.addNode(newNode.id, {
            label: newNode.labels.join(', '),
            color: newNode.color,
            x: nodePositions[nodeId].x + (Math.random() - 0.5) * 0.5,
            y: nodePositions[nodeId].y + (Math.random() - 0.5) * 0.5,
            size: newNode.size,
            borderColor: '#000',
            borderSize: 0.2
          });

          // Add the node to the raw graph
          if (!state.rawGraph.getNode(newNode.id)) {
            // Add to nodes array
            state.rawGraph.nodes.push(newNode);
            // Update nodeIdMap
            state.rawGraph.nodeIdMap[newNode.id] = state.rawGraph.nodes.length - 1;
          }
        }
      }

      // Add new edges
      for (const newEdge of processedEdges) {
        // Only add edges where both source and target exist in the graph
        if (state.sigmaGraph.hasNode(newEdge.source) && state.sigmaGraph.hasNode(newEdge.target)) {
          // Skip if edge already exists
          if (state.sigmaGraph.hasEdge(newEdge.source, newEdge.target)) {
            continue;
          }

          // Add the edge to the sigma graph
          newEdge.dynamicId = state.sigmaGraph.addDirectedEdge(newEdge.source, newEdge.target, {
            label: newEdge.type || undefined
          });

          // Add the edge to the raw graph
          if (!state.rawGraph.getEdge(newEdge.id, false)) {
            // Add to edges array
            state.rawGraph.edges.push(newEdge);
            // Update edgeIdMap
            state.rawGraph.edgeIdMap[newEdge.id] = state.rawGraph.edges.length - 1;
            // Update dynamic edge map
            state.rawGraph.edgeDynamicIdMap[newEdge.dynamicId] = state.rawGraph.edges.length - 1;
          }
        }
      }

      // Update the dynamic edge map
      state.rawGraph.buildDynamicMap();

      // Restore positions for existing nodes
      Object.entries(nodePositions).forEach(([nodeId, position]) => {
        if (state.sigmaGraph!.hasNode(nodeId)) {
          state.sigmaGraph!.setNodeAttribute(nodeId, 'x', position.x);
          state.sigmaGraph!.setNodeAttribute(nodeId, 'y', position.y);
        }
      });

    } catch (error) {
      console.error('Error expanding node:', error);
    } finally {
      // Reset fetching state
      state.setIsFetching(false);
    }
  },

  pruneNode: (nodeId: string) => {
    const state = get();
    if (!state.sigmaGraph || !state.rawGraph || !nodeId) {
      console.error('Cannot prune node: graph or node not available');
      return;
    }

    try {
      // Check if the node exists
      if (!state.sigmaGraph.hasNode(nodeId)) {
        console.error('Node not found:', nodeId);
        return;
      }

      // If the node is selected or focused, clear selection
      if (state.selectedNode === nodeId || state.focusedNode === nodeId) {
        state.clearSelection();
      }

      // Remove the node from the sigma graph (this will also remove connected edges)
      state.sigmaGraph.dropNode(nodeId);

      // Remove the node from the raw graph
      const nodeIndex = state.rawGraph.nodeIdMap[nodeId];
      if (nodeIndex !== undefined) {
        // Find all edges connected to this node
        const edgesToRemove = state.rawGraph.edges.filter(
          edge => edge.source === nodeId || edge.target === nodeId
        );

        // Remove edges from raw graph
        for (const edge of edgesToRemove) {
          const edgeIndex = state.rawGraph.edgeIdMap[edge.id];
          if (edgeIndex !== undefined) {
            // Remove from edges array
            state.rawGraph.edges.splice(edgeIndex, 1);
            // Update edgeIdMap for all edges after this one
            for (const [id, idx] of Object.entries(state.rawGraph.edgeIdMap)) {
              if (idx > edgeIndex) {
                state.rawGraph.edgeIdMap[id] = idx - 1;
              }
            }
            // Remove from edgeIdMap
            delete state.rawGraph.edgeIdMap[edge.id];
            // Remove from edgeDynamicIdMap
            delete state.rawGraph.edgeDynamicIdMap[edge.dynamicId];
          }
        }

        // Remove node from nodes array
        state.rawGraph.nodes.splice(nodeIndex, 1);

        // Update nodeIdMap for all nodes after this one
        for (const [id, idx] of Object.entries(state.rawGraph.nodeIdMap)) {
          if (idx > nodeIndex) {
            state.rawGraph.nodeIdMap[id] = idx - 1;
          }
        }

        // Remove from nodeIdMap
        delete state.rawGraph.nodeIdMap[nodeId];

        // Rebuild the dynamic edge map
        state.rawGraph.buildDynamicMap();
      }
      
      // 图形更新后会自动触发重新布局

    } catch (error) {
      console.error('Error pruning node:', error);
    }
  }
}))

const useGraphStore = createSelectors(useGraphStoreBase)

export { useGraphStore }
