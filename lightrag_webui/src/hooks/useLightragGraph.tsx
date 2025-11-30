import Graph, { UndirectedGraph } from 'graphology'
import { useCallback, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '@/lib/utils'
import * as Constants from '@/lib/constants'
import { useGraphStore, RawGraph, RawNodeType, RawEdgeType } from '@/stores/graph'
import { toast } from 'sonner'
import { queryGraphs } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'

import seedrandom from 'seedrandom'
import { resolveNodeColor, DEFAULT_NODE_COLOR } from '@/utils/graphColor'

// Select color based on node type
const getNodeColorByType = (nodeType: string | undefined): string => {
  const state = useGraphStore.getState()
  const { color, map, updated } = resolveNodeColor(nodeType, state.typeColorMap)

  if (updated) {
    useGraphStore.setState({ typeColorMap: map })
  }

  return color || DEFAULT_NODE_COLOR
};


const validateGraph = (graph: RawGraph) => {
  // Check if graph exists
  if (!graph) {
    console.log('Graph validation failed: graph is null');
    return false;
  }

  // Check if nodes and edges are arrays
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    console.log('Graph validation failed: nodes or edges is not an array');
    return false;
  }

  // Check if nodes array is empty
  if (graph.nodes.length === 0) {
    console.log('Graph validation failed: nodes array is empty');
    return false;
  }

  // Validate each node
  for (const node of graph.nodes) {
    if (!node.id || !node.labels || !node.properties) {
      console.log('Graph validation failed: invalid node structure');
      return false;
    }
  }

  // Validate each edge
  for (const edge of graph.edges) {
    if (!edge.id || !edge.source || !edge.target) {
      console.log('Graph validation failed: invalid edge structure');
      return false;
    }
  }

  // Validate edge connections
  for (const edge of graph.edges) {
    const source = graph.getNode(edge.source);
    const target = graph.getNode(edge.target);
    if (source == undefined || target == undefined) {
      console.log('Graph validation failed: edge references non-existent node');
      return false;
    }
  }

  console.log('Graph validation passed');
  return true;
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

const fetchGraph = async (label: string, maxDepth: number, maxNodes: number, minDegree: number = 0, includeOrphans: boolean = false) => {
  let rawData: any = null;

  // Trigger GraphLabels component to check if the label is valid
  // console.log('Setting labelsFetchAttempted to true');
  useGraphStore.getState().setLabelsFetchAttempted(true)

  // If label is empty, use default label '*'
  const queryLabel = label || '*';

  try {
    console.log(`Fetching graph label: ${queryLabel}, depth: ${maxDepth}, nodes: ${maxNodes}, minDegree: ${minDegree}, includeOrphans: ${includeOrphans}`);
    rawData = await queryGraphs(queryLabel, maxDepth, maxNodes, minDegree, includeOrphans);
  } catch (e) {
    useBackendState.getState().setErrorMessage(errorMessage(e), 'Query Graphs Error!');
    return null;
  }

  let rawGraph = null;

  if (rawData) {
    const nodeIdMap: Record<string, number> = {}
    const edgeIdMap: Record<string, number> = {}

    for (let i = 0; i < rawData.nodes.length; i++) {
      const node = rawData.nodes[i]
      nodeIdMap[node.id] = i

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

    if (!validateGraph(rawGraph)) {
      rawGraph = null
      console.warn('Invalid graph data')
    }
    console.log('Graph data loaded')
  }

  // console.debug({ data: JSON.parse(JSON.stringify(rawData)) })
  return { rawGraph, is_truncated: rawData.is_truncated }
}

// Create a new graph instance with the raw graph data
const createSigmaGraph = (rawGraph: RawGraph | null) => {
  // Get edge size settings from store
  const minEdgeSize = useSettingsStore.getState().minEdgeSize
  const maxEdgeSize = useSettingsStore.getState().maxEdgeSize
  // Skip graph creation if no data or empty nodes
  if (!rawGraph || !rawGraph.nodes.length) {
    console.log('No graph data available, skipping sigma graph creation');
    return null;
  }

  // Create new graph instance
  const graph = new UndirectedGraph()

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
      borderSize: 0.2,
      // Store db_degree for hidden connections indicator
      db_degree: rawNode.properties?.db_degree ?? 0
    })
  }

  // Add edges from raw graph data
  for (const rawEdge of rawGraph?.edges ?? []) {
    // Get weight from edge properties or default to 1
    const weight = rawEdge.properties?.weight !== undefined ? Number(rawEdge.properties.weight) : 1

    rawEdge.dynamicId = graph.addEdge(rawEdge.source, rawEdge.target, {
      label: rawEdge.properties?.keywords || undefined,
      size: weight, // Set initial size based on weight
      originalWeight: weight, // Store original weight for recalculation
      type: 'curvedNoArrow' // Explicitly set edge type to no arrow
    })
  }

  // Calculate edge size based on weight range, similar to node size calculation
  let minWeight = Number.MAX_SAFE_INTEGER
  let maxWeight = 0

  // Find min and max weight values
  graph.forEachEdge(edge => {
    const weight = graph.getEdgeAttribute(edge, 'originalWeight') || 1
    minWeight = Math.min(minWeight, weight)
    maxWeight = Math.max(maxWeight, weight)
  })

  // Scale edge sizes based on weight range
  const weightRange = maxWeight - minWeight
  if (weightRange > 0) {
    const sizeScale = maxEdgeSize - minEdgeSize
    graph.forEachEdge(edge => {
      const weight = graph.getEdgeAttribute(edge, 'originalWeight') || 1
      const scaledSize = minEdgeSize + sizeScale * Math.pow((weight - minWeight) / weightRange, 0.5)
      graph.setEdgeAttribute(edge, 'size', scaledSize)
    })
  } else {
    // If all weights are the same, use default size
    graph.forEachEdge(edge => {
      graph.setEdgeAttribute(edge, 'size', minEdgeSize)
    })
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
  const minDegree = useSettingsStore.use.graphMinDegree()
  const includeOrphans = useSettingsStore.use.graphIncludeOrphans()
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
      return;
    }

    // Only fetch data when graphDataFetchAttempted is false (avoids re-fetching on vite dev mode)
    // GraphDataFetchAttempted must set to false when queryLabel is changed
    if (!isFetching && !useGraphStore.getState().graphDataFetchAttempted) {
      // Set flags
      fetchInProgressRef.current = true
      useGraphStore.getState().setGraphDataFetchAttempted(true)

      const state = useGraphStore.getState()
      state.setIsFetching(true)

      // Clear selection and highlighted nodes before fetching new graph
      state.clearSelection()
      if (state.sigmaGraph) {
        state.sigmaGraph.forEachNode((node) => {
          state.sigmaGraph?.setNodeAttribute(node, 'highlighted', false)
        })
      }

      console.log('Preparing graph data...')

      // Use a local copy of the parameters
      const currentQueryLabel = queryLabel
      const currentMaxQueryDepth = maxQueryDepth
      const currentMaxNodes = maxNodes
      const currentMinDegree = minDegree
      const currentIncludeOrphans = includeOrphans

      // Declare a variable to store data promise
      let dataPromise: Promise<{ rawGraph: RawGraph | null; is_truncated: boolean | undefined } | null>;

      // 1. If query label is not empty, use fetchGraph
      if (currentQueryLabel) {
        dataPromise = fetchGraph(currentQueryLabel, currentMaxQueryDepth, currentMaxNodes, currentMinDegree, currentIncludeOrphans);
      } else {
        // 2. If query label is empty, set data to null
        console.log('Query label is empty, show empty graph')
        dataPromise = Promise.resolve({ rawGraph: null, is_truncated: false });
      }

      // 3. Process data
      dataPromise.then((result) => {
        const state = useGraphStore.getState()
        const data = result?.rawGraph;

        // Assign colors based on entity_type *after* fetching
        if (data && data.nodes) {
          data.nodes.forEach(node => {
            // Use entity_type instead of type
            const nodeEntityType = node.properties?.entity_type as string | undefined;
            node.color = getNodeColorByType(nodeEntityType);
          });
        }

        if (result?.is_truncated) {
          toast.info(t('graphPanel.dataIsTruncated', 'Graph data is truncated to Max Nodes'));
        }

        // Reset state
        state.reset()

        // Check if data is empty or invalid
        if (!data || !data.nodes || data.nodes.length === 0) {
          // Create a graph with a single "Graph Is Empty" node
          const emptyGraph = new UndirectedGraph();

          // Add a single node with "Graph Is Empty" label
          emptyGraph.addNode('empty-graph-node', {
            label: t('graphPanel.emptyGraph'),
            color: '#5D6D7E', // gray color
            x: 0.5,
            y: 0.5,
            size: 15,
            borderColor: Constants.nodeBorderColor,
            borderSize: 0.2
          });

          // Set graph to store
          state.setSigmaGraph(emptyGraph);
          state.setRawGraph(null);

          // Still mark graph as empty for other logic
          state.setGraphIsEmpty(true);

          // Check if the empty graph is due to 401 authentication error
          const errorMessage = useBackendState.getState().message;
          const isAuthError = errorMessage && errorMessage.includes('Authentication required');

          // Only clear queryLabel if it's not an auth error and current label is not empty
          if (!isAuthError && currentQueryLabel) {
            useSettingsStore.getState().setQueryLabel('');
          }

          // Only clear last successful query label if it's not an auth error
          if (!isAuthError) {
            state.setLastSuccessfulQueryLabel('');
          } else {
            console.log('Keep queryLabel for post-login reload');
          }

          console.log(`Graph data is empty, created graph with empty graph node. Auth error: ${isAuthError}`);
        } else {
          // Create and set new graph
          const newSigmaGraph = createSigmaGraph(data);
          data.buildDynamicMap();

          // Set new graph data
          state.setSigmaGraph(newSigmaGraph);
          state.setRawGraph(data);
          state.setGraphIsEmpty(false);

          // Update last successful query label
          state.setLastSuccessfulQueryLabel(currentQueryLabel);

          // Reset camera view
          state.setMoveToSelectedNode(true);
        }

        // Update flags
        dataLoadedRef.current = true
        initialLoadRef.current = true
        fetchInProgressRef.current = false
        state.setIsFetching(false)

        // Mark empty data as handled if data is empty and query label is empty
        if ((!data || !data.nodes || data.nodes.length === 0) && !currentQueryLabel) {
          emptyDataHandledRef.current = true;
        }
      }).catch((error) => {
        console.error('Error fetching graph data:', error)

        // Reset state on error
        const state = useGraphStore.getState()
        state.setIsFetching(false)
        dataLoadedRef.current = false;
        fetchInProgressRef.current = false
        state.setGraphDataFetchAttempted(false)
        state.setLastSuccessfulQueryLabel('') // Clear last successful query label on error
      })
    }
  }, [queryLabel, maxQueryDepth, maxNodes, minDegree, includeOrphans, isFetching, t, graphDataVersion])

  // Handle node expansion
  useEffect(() => {
    const handleNodeExpand = async (nodeId: string | null) => {
      if (!nodeId || !sigmaGraph || !rawGraph) return;

      try {
        // Get the node to expand
        const nodeToExpand = rawGraph.getNode(nodeId);
        if (!nodeToExpand) {
          console.error('Node not found:', nodeId);
          return;
        }

        // Get the label of the node to expand
        const label = nodeToExpand.labels[0];
        if (!label) {
          console.error('Node has no label:', nodeId);
          return;
        }

        // Fetch neighbors using the configured expand depth and max nodes (minDegree=0, includeOrphans=true)
        // This ensures we get all hidden connections, even orphan neighbors
        const expandDepth = useSettingsStore.getState().graphExpandDepth
        const maxNodes = useSettingsStore.getState().graphMaxNodes
        const extendedGraph = await queryGraphs(label, expandDepth, maxNodes, 0, true);

        if (!extendedGraph || !extendedGraph.nodes || !extendedGraph.edges) {
          console.error('Failed to fetch extended graph');
          return;
        }

        // Process nodes to add required properties for RawNodeType
        const processedNodes: RawNodeType[] = [];
        for (const node of extendedGraph.nodes) {
          // Generate random color values
          seedrandom(node.id, { global: true });
          const nodeEntityType = node.properties?.entity_type as string | undefined;
          const color = getNodeColorByType(nodeEntityType);

          // Create a properly typed RawNodeType
          processedNodes.push({
            id: node.id,
            labels: node.labels,
            properties: node.properties,
            size: 10, // Default size, will be calculated later
            x: Math.random(), // Random position, will be adjusted later
            y: Math.random(), // Random position, will be adjusted later
            color: color, // Random color
            degree: 0 // Initial degree, will be calculated later
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
        sigmaGraph.forEachNode((node) => {
          nodePositions[node] = {
            x: sigmaGraph.getNodeAttribute(node, 'x'),
            y: sigmaGraph.getNodeAttribute(node, 'y')
          };
        });

        // Get existing node IDs
        const existingNodeIds = new Set(sigmaGraph.nodes());

        // Identify nodes and edges to keep
        const nodesToAdd = new Set<string>();
        const edgesToAdd = new Set<string>();

        // Get degree maxDegree from existing graph for size calculations
        const minDegree = 1;
        let maxDegree = 0;

        // Initialize edge weight min and max values
        let minWeight = Number.MAX_SAFE_INTEGER;
        let maxWeight = 0;

        // Calculate node degrees and edge weights from existing graph
        sigmaGraph.forEachNode(node => {
          const degree = sigmaGraph.degree(node);
          maxDegree = Math.max(maxDegree, degree);
        });

        // Calculate edge weights from existing graph
        sigmaGraph.forEachEdge(edge => {
          const weight = sigmaGraph.getEdgeAttribute(edge, 'originalWeight') || 1;
          minWeight = Math.min(minWeight, weight);
          maxWeight = Math.max(maxWeight, weight);
        });

        // First identify connectable nodes (nodes connected to the expanded node)
        for (const node of processedNodes) {
          // Skip if node already exists
          if (existingNodeIds.has(node.id)) {
            continue;
          }

          // Check if this node is connected to the selected node
          const isConnected = processedEdges.some(
            edge => (edge.source === nodeId && edge.target === node.id) ||
                   (edge.target === nodeId && edge.source === node.id)
          );

          if (isConnected) {
            nodesToAdd.add(node.id);
          }
        }

        // Calculate node degrees and track discarded edges in one pass
        const nodeDegrees = new Map<string, number>();
        const existingNodeDegreeIncrements = new Map<string, number>(); // Track degree increments for existing nodes
        const nodesWithDiscardedEdges = new Set<string>();

        for (const edge of processedEdges) {
          const sourceExists = existingNodeIds.has(edge.source) || nodesToAdd.has(edge.source);
          const targetExists = existingNodeIds.has(edge.target) || nodesToAdd.has(edge.target);

          if (sourceExists && targetExists) {
            edgesToAdd.add(edge.id);
            // Add degrees for both new and existing nodes
            if (nodesToAdd.has(edge.source)) {
              nodeDegrees.set(edge.source, (nodeDegrees.get(edge.source) || 0) + 1);
            } else if (existingNodeIds.has(edge.source)) {
              // Track degree increments for existing nodes
              existingNodeDegreeIncrements.set(edge.source, (existingNodeDegreeIncrements.get(edge.source) || 0) + 1);
            }

            if (nodesToAdd.has(edge.target)) {
              nodeDegrees.set(edge.target, (nodeDegrees.get(edge.target) || 0) + 1);
            } else if (existingNodeIds.has(edge.target)) {
              // Track degree increments for existing nodes
              existingNodeDegreeIncrements.set(edge.target, (existingNodeDegreeIncrements.get(edge.target) || 0) + 1);
            }
          } else {
            // Track discarded edges for both new and existing nodes
            if (sigmaGraph.hasNode(edge.source)) {
              nodesWithDiscardedEdges.add(edge.source);
            } else if (nodesToAdd.has(edge.source)) {
              nodesWithDiscardedEdges.add(edge.source);
              nodeDegrees.set(edge.source, (nodeDegrees.get(edge.source) || 0) + 1); // +1 for discarded edge
            }
            if (sigmaGraph.hasNode(edge.target)) {
              nodesWithDiscardedEdges.add(edge.target);
            } else if (nodesToAdd.has(edge.target)) {
              nodesWithDiscardedEdges.add(edge.target);
              nodeDegrees.set(edge.target, (nodeDegrees.get(edge.target) || 0) + 1); // +1 for discarded edge
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
          const range = maxDegree - minDegree || 1; // Avoid division by zero
          const scale = Constants.maxNodeSize - Constants.minNodeSize;

          // Update node sizes
          for (const nodeId of nodesWithDiscardedEdges) {
            if (sigmaGraph.hasNode(nodeId)) {
              let newDegree = sigmaGraph.degree(nodeId);
              newDegree += 1; // Add +1 for discarded edges
              // Limit newDegree to maxDegree + 1 to prevent nodes from being too large
              const limitedDegree = Math.min(newDegree, maxDegree + 1);

              const newSize = Math.round(
                Constants.minNodeSize + scale * Math.pow((limitedDegree - minDegree) / range, 0.5)
              );

              sigmaGraph.setNodeAttribute(nodeId, 'size', newSize);
            }
          }
        };

        // Helper function to update edge sizes
        const updateEdgeSizes = (
          sigmaGraph: UndirectedGraph,
          minWeight: number,
          maxWeight: number
        ) => {
          // Update edge sizes
          const minEdgeSize = useSettingsStore.getState().minEdgeSize;
          const maxEdgeSize = useSettingsStore.getState().maxEdgeSize;
          const weightRange = maxWeight - minWeight || 1; // Avoid division by zero
          const sizeScale = maxEdgeSize - minEdgeSize;

          sigmaGraph.forEachEdge(edge => {
            const weight = sigmaGraph.getEdgeAttribute(edge, 'originalWeight') || 1;
            const scaledSize = minEdgeSize + sizeScale * Math.pow((weight - minWeight) / weightRange, 0.5);
            sigmaGraph.setEdgeAttribute(edge, 'size', scaledSize);
          });
        };

        // If no new connectable nodes found, show toast and return
        if (nodesToAdd.size === 0) {
          updateNodeSizes(sigmaGraph, nodesWithDiscardedEdges, minDegree, maxDegree);
          toast.info(t('graphPanel.propertiesView.node.noNewNodes'));
          return;
        }

        // Update maxDegree considering all nodes (both new and existing)
        // 1. Consider degrees of new nodes
        for (const [, degree] of nodeDegrees.entries()) {
          maxDegree = Math.max(maxDegree, degree);
        }

        // 2. Consider degree increments for existing nodes
        for (const [nodeId, increment] of existingNodeDegreeIncrements.entries()) {
          const currentDegree = sigmaGraph.degree(nodeId);
          const projectedDegree = currentDegree + increment;
          maxDegree = Math.max(maxDegree, projectedDegree);
        }

        const range = maxDegree - minDegree || 1; // Avoid division by zero
        const scale = Constants.maxNodeSize - Constants.minNodeSize;

        // SAdd nodes and edges to the graph
        // Calculate camera ratio and spread factor once before the loop
        const cameraRatio = useGraphStore.getState().sigmaInstance?.getCamera().ratio || 1;
        const spreadFactor = Math.max(
          Math.sqrt(nodeToExpand.size) * 4, // Base on node size
          Math.sqrt(nodesToAdd.size) * 3 // Scale with number of nodes
        ) / cameraRatio; // Adjust for zoom level
        seedrandom(Date.now().toString(), { global: true });
        const randomAngle = Math.random() * 2 * Math.PI

        console.log('nodeSize:', nodeToExpand.size, 'nodesToAdd:', nodesToAdd.size);
        console.log('cameraRatio:', Math.round(cameraRatio*100)/100, 'spreadFactor:', Math.round(spreadFactor*100)/100);

        // Add new nodes
        for (const nodeId of nodesToAdd) {
          const newNode = processedNodes.find(n => n.id === nodeId)!;
          const nodeDegree = nodeDegrees.get(nodeId) || 0;

          // Calculate node size
          // Limit nodeDegree to maxDegree + 1 to prevent new nodes from being too large
          const limitedDegree = Math.min(nodeDegree, maxDegree + 1);
          const nodeSize = Math.round(
            Constants.minNodeSize + scale * Math.pow((limitedDegree - minDegree) / range, 0.5)
          );

          // Calculate angle for polar coordinates
          const angle = 2 * Math.PI * (Array.from(nodesToAdd).indexOf(nodeId) / nodesToAdd.size);

          // Calculate final position
          const x = nodePositions[nodeId]?.x ||
                    (nodePositions[nodeToExpand.id].x + Math.cos(randomAngle + angle) * spreadFactor);
          const y = nodePositions[nodeId]?.y ||
                    (nodePositions[nodeToExpand.id].y + Math.sin(randomAngle + angle) * spreadFactor);

          // Add the new node to the sigma graph with calculated position
          sigmaGraph.addNode(nodeId, {
            label: newNode.labels.join(', '),
            color: newNode.color,
            x: x,
            y: y,
            size: nodeSize,
            borderColor: Constants.nodeBorderColor,
            borderSize: 0.2
          });

          // Add the node to the raw graph
          if (!rawGraph.getNode(nodeId)) {
            // Update node properties
            newNode.size = nodeSize;
            newNode.x = x;
            newNode.y = y;
            newNode.degree = nodeDegree;

            // Add to nodes array
            rawGraph.nodes.push(newNode);
            // Update nodeIdMap
            rawGraph.nodeIdMap[nodeId] = rawGraph.nodes.length - 1;
          }
        }

        // Add new edges
        for (const edgeId of edgesToAdd) {
          const newEdge = processedEdges.find(e => e.id === edgeId)!;

          // Skip if edge already exists
          if (sigmaGraph.hasEdge(newEdge.source, newEdge.target)) {
            continue;
          }

          // Get weight from edge properties or default to 1
          const weight = newEdge.properties?.weight !== undefined ? Number(newEdge.properties.weight) : 1;

          // Update min and max weight values
          minWeight = Math.min(minWeight, weight);
          maxWeight = Math.max(maxWeight, weight);

          // Add the edge to the sigma graph
          newEdge.dynamicId = sigmaGraph.addEdge(newEdge.source, newEdge.target, {
            label: newEdge.properties?.keywords || undefined,
            size: weight, // Set initial size based on weight
            originalWeight: weight, // Store original weight for recalculation
            type: 'curvedNoArrow' // Explicitly set edge type to no arrow
          });

          // Add the edge to the raw graph
          if (!rawGraph.getEdge(newEdge.id, false)) {
            // Add to edges array
            rawGraph.edges.push(newEdge);
            // Update edgeIdMap
            rawGraph.edgeIdMap[newEdge.id] = rawGraph.edges.length - 1;
            // Update dynamic edge map
            rawGraph.edgeDynamicIdMap[newEdge.dynamicId] = rawGraph.edges.length - 1;
          } else {
            console.error('Edge already exists in rawGraph:', newEdge.id);
          }
        }

        // Update the dynamic edge map and invalidate search cache
        rawGraph.buildDynamicMap();

        // Reset search engine to force rebuild
        useGraphStore.getState().resetSearchEngine();

        // Update sizes for all nodes and edges
        updateNodeSizes(sigmaGraph, nodesWithDiscardedEdges, minDegree, maxDegree);
        updateEdgeSizes(sigmaGraph, minWeight, maxWeight);

        // Final update for the expanded node
        if (sigmaGraph.hasNode(nodeId)) {
          const finalDegree = sigmaGraph.degree(nodeId);
          const limitedDegree = Math.min(finalDegree, maxDegree + 1);
          const newSize = Math.round(
            Constants.minNodeSize + scale * Math.pow((limitedDegree - minDegree) / range, 0.5)
          );
          sigmaGraph.setNodeAttribute(nodeId, 'size', newSize);
          nodeToExpand.size = newSize;
          nodeToExpand.degree = finalDegree;
        }

      } catch (error) {
        console.error('Error expanding node:', error);
      }
    };

    // If there's a node to expand, handle it
    if (nodeToExpand) {
      handleNodeExpand(nodeToExpand);
      // Reset the nodeToExpand state after handling
      window.setTimeout(() => {
        useGraphStore.getState().triggerNodeExpand(null);
      }, 0);
    }
  }, [nodeToExpand, sigmaGraph, rawGraph, t]);

  // Helper function to get all nodes that will be deleted
  const getNodesThatWillBeDeleted = useCallback((nodeId: string, graph: UndirectedGraph) => {
    const nodesToDelete = new Set<string>([nodeId]);

    // Find all nodes that would become isolated after deletion
    graph.forEachNode((node) => {
      if (node === nodeId) return; // Skip the node being deleted

      // Get all neighbors of this node
      const neighbors = graph.neighbors(node);

      // If this node has only one neighbor and that neighbor is the node being deleted,
      // this node will become isolated, so we should delete it too
      if (neighbors.length === 1 && neighbors[0] === nodeId) {
        nodesToDelete.add(node);
      }
    });

    return nodesToDelete;
  }, []);

  // Handle node pruning
  useEffect(() => {
    const handleNodePrune = (nodeId: string | null) => {
      if (!nodeId || !sigmaGraph || !rawGraph) return;

      try {
        const state = useGraphStore.getState();

        // 1. Check if node exists
        if (!sigmaGraph.hasNode(nodeId)) {
          console.error('Node not found:', nodeId);
          return;
        }

        // 2. Get nodes to delete
        const nodesToDelete = getNodesThatWillBeDeleted(nodeId, sigmaGraph);

        // 3. Check if this would delete all nodes
        if (nodesToDelete.size === sigmaGraph.nodes().length) {
          toast.error(t('graphPanel.propertiesView.node.deleteAllNodesError'));
          return;
        }

        // 4. Clear selection - this will cause PropertiesView to close immediately
        state.clearSelection();

        // 5. Delete nodes and related edges
        for (const nodeToDelete of nodesToDelete) {
          // Remove the node from the sigma graph (this will also remove connected edges)
          sigmaGraph.dropNode(nodeToDelete);

          // Remove the node from the raw graph
          const nodeIndex = rawGraph.nodeIdMap[nodeToDelete];
          if (nodeIndex !== undefined) {
            // Find all edges connected to this node
            const edgesToRemove = rawGraph.edges.filter(
              edge => edge.source === nodeToDelete || edge.target === nodeToDelete
            );

            // Remove edges from raw graph
            for (const edge of edgesToRemove) {
              const edgeIndex = rawGraph.edgeIdMap[edge.id];
              if (edgeIndex !== undefined) {
                // Remove from edges array
                rawGraph.edges.splice(edgeIndex, 1);
                // Update edgeIdMap for all edges after this one
                for (const [id, idx] of Object.entries(rawGraph.edgeIdMap)) {
                  if (idx > edgeIndex) {
                    rawGraph.edgeIdMap[id] = idx - 1;
                  }
                }
                // Remove from edgeIdMap
                delete rawGraph.edgeIdMap[edge.id];
                // Remove from edgeDynamicIdMap
                delete rawGraph.edgeDynamicIdMap[edge.dynamicId];
              }
            }

            // Remove node from nodes array
            rawGraph.nodes.splice(nodeIndex, 1);

            // Update nodeIdMap for all nodes after this one
            for (const [id, idx] of Object.entries(rawGraph.nodeIdMap)) {
              if (idx > nodeIndex) {
                rawGraph.nodeIdMap[id] = idx - 1;
              }
            }

            // Remove from nodeIdMap
            delete rawGraph.nodeIdMap[nodeToDelete];
          }
        }

        // Rebuild the dynamic edge map and invalidate search cache
        rawGraph.buildDynamicMap();

        // Reset search engine to force rebuild
        useGraphStore.getState().resetSearchEngine();

        // Show notification if we deleted more than just the selected node
        if (nodesToDelete.size > 1) {
          toast.info(t('graphPanel.propertiesView.node.nodesRemoved', { count: nodesToDelete.size }));
        }


      } catch (error) {
        console.error('Error pruning node:', error);
      }
    };

    // If there's a node to prune, handle it
    if (nodeToPrune) {
      handleNodePrune(nodeToPrune);
      // Reset the nodeToPrune state after handling
      window.setTimeout(() => {
        useGraphStore.getState().triggerNodePrune(null);
      }, 0);
    }
  }, [nodeToPrune, sigmaGraph, rawGraph, getNodesThatWillBeDeleted, t]);

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
